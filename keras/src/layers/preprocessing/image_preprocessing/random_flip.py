from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.image_preprocessing.base_image_preprocessing_layer import (  # noqa: E501
    BaseImagePreprocessingLayer,
)
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.validation import (  # noqa: E501
    densify_bounding_boxes,
)
from keras.src.random.seed_generator import SeedGenerator

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


@keras_export("keras.layers.RandomFlip")
class RandomFlip(BaseImagePreprocessingLayer):
    """A preprocessing layer which randomly flips images during training.

    This layer will flip the images horizontally and or vertically based on the
    `mode` attribute. During inference time, the output will be identical to
    input. Call the layer with `training=True` to flip the input.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        mode: String indicating which flip mode to use. Can be `"horizontal"`,
            `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a
            left-right flip and `"vertical"` is a top-bottom flip. Defaults to
            `"horizontal_and_vertical"`
        seed: Integer. Used to create a random seed.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    _USE_BASE_FACTOR = False

    def __init__(
        self,
        mode=HORIZONTAL_AND_VERTICAL,
        seed=None,
        data_format=None,
        **kwargs
    ):
        super().__init__(data_format=data_format, **kwargs)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.mode = mode
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def get_random_transformation(self, data, training=True, seed=None):
        if not training:
            return None

        if isinstance(data, dict):
            images = data["images"]
        else:
            images = data
        shape = self.backend.core.shape(images)
        if len(shape) == 3:
            flips_shape = (1, 1, 1)
        else:
            flips_shape = (shape[0], 1, 1, 1)

        if seed is None:
            seed = self._get_seed_generator(self.backend._backend)

        flips = self.backend.numpy.less_equal(
            self.backend.random.uniform(shape=flips_shape, seed=seed), 0.5
        )
        return {"flips": flips}

    def transform_images(self, images, transformation, training=True):
        images = self.backend.cast(images, self.compute_dtype)
        if training:
            return self._flip_inputs(images, transformation)
        return images

    def transform_labels(self, labels, transformation, training=True):
        return labels

    def transform_bounding_boxes(self, bounding_boxes, transformation, training=True):
        """
        bounding_boxes = {
            "boxes": (batch, num_boxes, 4),  # left, top, right, bottom
            "labels": (batch, num_boxes, num_classes),
        }
        or
        bounding_boxes = {
            "boxes": (num_boxes, 4),
            "labels": (num_boxes, num_classes),
        }
        """
        if transformation is None:
            return bounding_boxes
        
        flips = transformation["flips"]
        
        if not self.backend.is_tensor(bounding_boxes["boxes"]):
            bounding_boxes = densify_bounding_boxes(bounding_boxes, backend=self.backend)
        
        boxes = bounding_boxes["boxes"]
        
        image_shape = self.backend.shape(transformation["image_shape"])
        image_height = image_shape[-3]
        image_width = image_shape[-2]
        
        if self.mode == HORIZONTAL or self.mode == HORIZONTAL_AND_VERTICAL:
            if len(self.backend.shape(boxes)) == 3:
                boxes = self.backend.numpy.stack(
                    [
                        image_width - boxes[:, :, 2],
                        boxes[:, :, 1],
                        image_width - boxes[:, :, 0],
                        boxes[:, :, 3],
                    ],
                    axis=-1,
                )
            else:
                boxes = self.backend.numpy.stack(
                    [
                        image_width - boxes[:, 2],
                        boxes[:, 1],
                        image_width - boxes[:, 0],
                        boxes[:, 3],
                    ],
                    axis=-1,
                )
        
        if self.mode == VERTICAL or self.mode == HORIZONTAL_AND_VERTICAL:
            if len(self.backend.shape(boxes)) == 3:
                boxes = self.backend.numpy.stack(
                    [
                        boxes[:, :, 0],
                        image_height - boxes[:, :, 3],
                        boxes[:, :, 2],
                        image_height - boxes[:, :, 1],
                    ],
                    axis=-1,
                )
            else:
                boxes = self.backend.numpy.stack(
                    [
                        boxes[:, 0],
                        image_height - boxes[:, 3],
                        boxes[:, 2],
                        image_height - boxes[:, 1],
                    ],
                    axis=-1,
                )
        
        return {
            "boxes": boxes,
            "labels": bounding_boxes["labels"],
        }


    def transform_segmentation_masks(
        self, segmentation_masks, transformation, training=True
    ):
        return self.transform_images(
            segmentation_masks, transformation, training=training
        )

    def _flip_inputs(self, inputs, transformation):
        if transformation is None:
            return inputs

        flips = transformation["flips"]
        inputs_shape = self.backend.shape(inputs)
        unbatched = len(inputs_shape) == 3
        if unbatched:
            inputs = self.backend.numpy.expand_dims(inputs, axis=0)

        flipped_outputs = inputs
        if self.data_format == "channels_last":
            horizontal_axis = -2
            vertical_axis = -3
        else:
            horizontal_axis = -1
            vertical_axis = -2

        if self.mode == HORIZONTAL or self.mode == HORIZONTAL_AND_VERTICAL:
            flipped_outputs = self.backend.numpy.where(
                flips,
                self.backend.numpy.flip(flipped_outputs, axis=horizontal_axis),
                flipped_outputs,
            )
        if self.mode == VERTICAL or self.mode == HORIZONTAL_AND_VERTICAL:
            flipped_outputs = self.backend.numpy.where(
                flips,
                self.backend.numpy.flip(flipped_outputs, axis=vertical_axis),
                flipped_outputs,
            )
        if unbatched:
            flipped_outputs = self.backend.numpy.squeeze(
                flipped_outputs, axis=0
            )
        return flipped_outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seed": self.seed,
                "mode": self.mode,
                "data_format": self.data_format,
            }
        )
        return config
