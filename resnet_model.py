"""Constructing ResNet models for training
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model_base
import model_base1 #To be merged with each other

"""
__author__ = "Mohammad Mahdi Kamani"
__copyright__ = "Copyright 2019, Mohammad Mahdi Kamani"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Madhi Kamani"
__status__ = "Prototype"
"""

class ResNetCifar(model_base.ResNet):
  """Cifar10/100 model with ResNetV1 and basic residual block."""

  def __init__(self,
               num_layers,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               dataset_name='cifar10',
               data_format='channels_first'):
    super(ResNetCifar, self).__init__(
        is_training,
        data_format,
        batch_norm_decay,
        batch_norm_epsilon
    )
    self.n = (num_layers - 2) // 6
    # Add one in case label starts with 1. No impact if label starts with 0.
    self.num_classes = 10+1 if dataset_name == 'cifar10' else 100+1
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]

  def forward_pass(self, x, input_data_format='channels_last'):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # Image standardization.
    x = x / 128 - 1

    x = self._conv(x, 3, 16, 1)
    x = self._batch_norm(x)
    x = self._relu(x)

    # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
    res_func = self._residual_v1

    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(self.n):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x = res_func(x, 3, self.filters[i], self.filters[i + 1],
                         self.strides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x = res_func(x, 3, self.filters[i + 1], self.filters[i + 1], 1)

    x = self._global_avg_pool(x)
    x = self._fully_connected(x, self.num_classes)

    return x


class ImagenetModel(model_base1.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, num_classes,
               resnet_version=model_base1.DEFAULT_VERSION,
               dtype=model_base1.DEFAULT_DTYPE,
               data_format='channels_first'):
    """These are the parameters that work for Imagenet data.
    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.
  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  Args:
    resnet_size: The number of convolutional layers needed in the model.
  Returns:
    A list of block sizes to use in building the model.
  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)



