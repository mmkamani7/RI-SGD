"""CIFAR-10, CIFAR-100, or ImageNet data set loader.
"""
# Name should change to dataloader
import os
import numpy as np
import glob

import tensorflow as tf

"""
__author__ = "Mohammad Mahdi Kamani"
__copyright__ = "Copyright 2019, Mohammad Mahdi Kamani"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Madhi Kamani"
__status__ = "Prototype"
"""

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class CifarDataSet(object):
  """Cifar10 or CIFAR100 data set.

  Described by http://www.cs.toronto.edu/~kriz/cifar.html.
  """

  def __init__(self,
              data_dir,
              num_shards,
              subset='train',
              use_distortion=True,
              redundancy=0.0,
              dataset='cifar10'):

    self.data_dir = data_dir
    self.num_shards = num_shards
    self.subset = subset
    self.use_distortion = use_distortion
    self.dataset = dataset
    self.redundancy = redundancy
    if self.redundancy > 0:
      self.redun_vector = np.random.normal(self.redundancy, 0.01, self.num_shards)
      self.redun_vector += self.redundancy - np.mean(self.redun_vector)

  def get_filenames(self):
    if self.subset in ['train', 'validation', 'eval']:
      if self.dataset in ['cifar10','cifar100']:
        return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
      elif self.dataset == 'imagenet':
        if self.subset == 'eval':
          subset = 'validation'
        else:
          subset = self.subset
        filenames = glob.glob(os.path.join(self.data_dir, subset + '*'))
        return filenames
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10/100 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    if self.dataset in ['cifar10','cifar100']:
      features = tf.parse_single_example(
          serialized_example,
          features={
              'image': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64),
          })
      image = tf.decode_raw(features['image'], tf.uint8)
      image.set_shape([DEPTH * HEIGHT * WIDTH])

      # Reshape from [depth * height * width] to [depth, height, width].
      image = tf.cast(
          tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
          tf.float32)
      label = tf.cast(features['label'], tf.int32)

      # Custom preprocessing.
      image = self.preprocess(image)
    elif self.dataset == 'imagenet':
      features = tf.parse_single_example(
          serialized_example,
          features={
              'image/encoded': tf.FixedLenFeature([], tf.string),
              'image/class/label': tf.FixedLenFeature([], tf.int64),
              'image/class/text': tf.FixedLenFeature([], tf.string)
          })
      image = tf.image.decode_png(features['image/encoded'], channels=3)
      image = tf.image.resize_images(image,
                                    [224, 224],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      image = tf.cast(image, tf.float32)
      assert len(image.shape) == 3
      assert image.shape[2] == 3
      label = tf.cast(features['image/class/label'], tf.int32)

    return image, label

  def make_batch(self, batch_size):
    feature_shards = [[] for i in range(self.num_shards)]
    label_shards = [[] for i in range(self.num_shards)]
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    dataset = tf.data.TFRecordDataset(filenames)

    for device_id in range(self.num_shards):
      if self.subset == 'train':
        d0 = dataset.shard(self.num_shards, device_id)
        d0 = d0.repeat()

        # Parse records.
        d0 = d0.map(
          self.parser, num_parallel_calls=int(batch_size / self.num_shards))

        # Potentially shuffle records.
        min_queue_examples = int(
          CifarDataSet.num_examples_per_epoch(self.subset,self.dataset) * 0.4 / self.num_shards)
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        d0 = d0.shuffle(buffer_size= 10000)#min_queue_examples + int(3 * batch_size / self.num_shards))

        # Batch it up.
        d0 = d0.batch(int(batch_size / self.num_shards))
        iterator0 = d0.make_one_shot_iterator()
        image_batch, label_batch = iterator0.get_next()

        if self.redundancy > 0:
          remained_redundancy = self.redun_vector[device_id]
          num_devices = int(np.ceil(self.redundancy))
          for redun_device in range(num_devices):
            r = remained_redundancy if ((remained_redundancy > 0) & (remained_redundancy < 1.0)) else 1.0
            remained_redundancy -= r
            # d1 = dataset.shard(self.num_shards,
            #                    device_id).shard(int(np.ceil(1 / r)), 0)

            d1 = dataset.shard(self.num_shards,
                              (device_id + redun_device + 1) % self.num_shards).shard(
              int(np.ceil(1 / r)), 0)
              # d1 = d1.shard(int(2/self.redundancy) , 0)
            d1 = d1.repeat()

            # Parse records.
            d1 = d1.map(
              self.parser, num_parallel_calls=int(batch_size / self.num_shards * r))

            min_queue_examples = int(
              CifarDataSet.num_examples_per_epoch(self.subset,self.dataset) * 0.4 / self.num_shards * r)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            d1 = d1.shuffle(buffer_size=10000)#min_queue_examples + int(3 * batch_size / self.num_shards * r))

            # Batch it up.
            d1 = d1.batch(int(batch_size / self.num_shards * r))
            iterator1 = d1.make_one_shot_iterator()
            image_batch1, label_batch1 = iterator1.get_next()

            image_batch = tf.concat((image_batch, image_batch1), 0)
            label_batch = tf.concat((label_batch, label_batch1), 0)

      elif self.subset == 'eval':
        d = dataset.repeat()
        d = d.map(
          self.parser, num_parallel_calls=batch_size)
        d = d.batch(batch_size)
        iterator = d.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()


      feature_shards[device_id] = image_batch
      label_shards[device_id] = label_batch

    return feature_shards, label_shards

  def preprocess(self, image):
    """Preprocess a single image in [height, wi dth, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
      image = tf.image.random_flip_left_right(image)
    return image

  @staticmethod
  def num_examples_per_epoch(subset='train', dataset='cifar'):
    if dataset in ['cifar10','cifar100'] :
      if subset == 'train':
        return 50000
      elif subset == 'validation':
        return 5000
      elif subset == 'eval':
        return 10000
      else:
        raise ValueError('Invalid data subset "%s"' % subset)
    if dataset == 'imagenet':
      if subset == 'train':
        return 1281167
      elif subset == 'validation':
        return 50000
      elif subset == 'eval':
        return 50000
      else:
        raise ValueError('Invalid data subset "%s"' % subset)