"""Implementation of distributed training for training deep learning models from
Haddadpour, F.,  Kamani, M.M., Mahdavi, M., & Cadambe, V.
"Trading Redundancy for Communication: Speeding up Distributed SGD for Non-convex Optimization."
 International Conference on Machine Learning. 2019.

Support single-host training with one or multiple devices.
"""
from __future__ import division
from __future__ import print_function

import cifar
import utils
import resnet_model

import argparse
import functools
import itertools
import os
import json
from collections import namedtuple
import numpy as np
import six
from six.moves import xrange 
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

"""
__author__ = "Mohammad Mahdi Kamani"
__copyright__ = "Copyright 2019, Mohammad Mahdi Kamani"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Madhi Kamani"
__status__ = "Prototype"
"""


def get_model_fn(num_gpus, variable_strategy, num_workers, run_type='local'):
  """Returns a function that will build the resnet model."""

  def _resnet_model_fn_sync(features, labels, mode, params):
    """Resnet model body.

    Support single host, one or more GPU training. Parameter distribution can
    be either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params.weight_decay
    momentum = params.momentum

    features = features[0:num_gpus]
    labels = labels[0:num_gpus]
    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_gradvars = []
    tower_preds = []

    # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
    # on CPU. The exception is Intel MKL on CPU which is optimal with
    # channels_last.
    data_format = params.data_format
    if not data_format:
      if num_gpus == 0:
        data_format = 'channels_last'
      else:
        data_format = 'channels_first'

    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'

    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        device_setter = utils.local_device_setter(
            worker_device=worker_device)
      elif variable_strategy == 'GPU':
        device_setter = utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
      with tf.variable_scope('resnet', reuse=bool(i != 0)) as var_scope:
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            loss, gradvars, preds = _tower_fn(
                is_training, weight_decay, tower_features[i], tower_labels[i],
                data_format, params.num_layers, params.batch_norm_decay,
                params.batch_norm_epsilon, var_scope.name, params.dataset)
            tower_losses.append(loss)
            tower_gradvars.append(gradvars)
            tower_preds.append(preds)
            if i == 0:
              # Only trigger batch_norm moving mean and variance update from
              # the 1st tower. Ideally, we should grab the updates from all
              # towers but these stats accumulate extremely fast so we can
              # ignore the other stats from the other towers without
              # significant detriment.
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             name_scope)

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
      all_grads = {}
      for grad, var in itertools.chain(*tower_gradvars):
        if grad is not None:
          all_grads.setdefault(var, []).append(grad)
      for var, grads in six.iteritems(all_grads):
        # Average gradients on the same device as the variables
        # to which they apply.
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
        gradvars.append((avg_grad, var))

    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):
      # Suggested learning rate scheduling from
      # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar100-resnet.py#L155
      num_images = cifar.CifarDataSet.num_examples_per_epoch('train', params.dataset) 
      if params.dataset in ['cifar10','cifar100']:
        learning_rate_fn = utils.learning_rate_with_decay(
                          batch_size=params.train_batch_size, batch_denom=params.train_batch_size,
                          num_images=num_images, boundary_epochs=[91, 136, 182],
                          decay_rates=[1, 0.1, 0.01, 0.001])
      elif params.dataset == 'imagenet':
        learning_rate_fn = utils.learning_rate_with_decay(
                            batch_size=params.train_batch_size, batch_denom=params.train_batch_size,
                            num_images=num_images, boundary_epochs=[30, 60, 80, 90],
                            decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], 
                            warmup=params.warmup, base_lr=params.learning_rate)

      learning_rate = learning_rate_fn(tf.train.get_global_step())

      loss = tf.reduce_mean(tower_losses, name='loss')

      examples_sec_hook = utils.ExamplesPerSecondHook(
          params.train_batch_size, every_n_steps=10)

      tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      train_hooks = [logging_hook, examples_sec_hook]

      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=momentum)

      if params.run_type == 'sync':
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer, replicas_to_aggregate=num_workers)
        sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
        train_hooks.append(sync_replicas_hook)

      # Create single grouped train op
      train_op = [
          optimizer.apply_gradients(
              gradvars, global_step=tf.train.get_global_step())
      ]
      train_op.extend(update_ops)
      train_op = tf.group(*train_op)

      predictions = {
          'classes':
              tf.concat([p['classes'] for p in tower_preds], axis=0),
          'probabilities':
              tf.concat([p['probabilities'] for p in tower_preds], axis=0)
      }
      stacked_labels = tf.concat(labels, axis=0)
      accuracy = tf.metrics.accuracy(stacked_labels, predictions['classes'])
      metrics = {'accuracy': accuracy}
      tf.summary.scalar('accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)

  def _resnet_model_fn_local(features, labels, mode, params):
    """Resnet model body for asynchoronous mode.

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    init_op = []
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params.weight_decay
    momentum = params.momentum

    features = features[0:num_gpus]
    labels = labels[0:num_gpus]
    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_ops= []
    tower_preds = []
    var_scopes=[]


    # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
    # on CPU. The exception is Intel MKL on CPU which is optimal with
    # channels_last.
    data_format = params.data_format
    if not data_format:
      if num_gpus == 0:
        data_format = 'channels_last'
      else:
        data_format = 'channels_first'

    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'

    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        device_setter = utils.local_device_setter(
            worker_device=worker_device)
        # device_setter = tf.train.replica_device_setter(
        #     worker_device=worker_device)
      elif variable_strategy == 'GPU':
        device_setter = utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
        # device_setter = tf.train.replica_device_setter(
        #     ps_device=worker_device,
        #     worker_device=worker_device
        # )
      with tf.variable_scope('resnet_{}'.format(i)) as var_scope:
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            loss, gradvars, preds = _tower_fn(
                is_training, weight_decay, tower_features[i], tower_labels[i],
                data_format, params.num_layers, params.batch_norm_decay,
                params.batch_norm_epsilon, var_scope.name, params.dataset)
            var_scopes.append(var_scope.name)
            # if ckpt_dir:
            #     init_op.append(tf.train.init_from_checkpoint(ckpt_dir,{'resnet_{}/'.format(i): 'resnet_{}/'.format(i)}))
            tower_losses.append(loss)
            # tower_gradvars.append(gradvars)
            tower_preds.append(preds)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

            # Updating parameters
            # Suggested learning rate scheduling from
            # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar100-resnet.py#L155
            num_images = cifar.CifarDataSet.num_examples_per_epoch('train', params.dataset) 
            # boundaries = [
            #     num_batches_per_epoch * x
            #     for x in np.array([30, 60, 80, 90], dtype=np.int64)
            # ]
            # staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.001, 1e-4]]

            # learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
            #                                           boundaries, staged_lr, name='lr_tower_{}'.format(i))
            if params.dataset in ['cifar10','cifar100']:
              learning_rate_fn = utils.learning_rate_with_decay(
                                batch_size=params.train_batch_size, batch_denom=params.train_batch_size,
                                num_images=num_images, boundary_epochs=[91, 136, 182],
                                decay_rates=[1, 0.1, 0.01, 0.001])
            elif params.dataset == 'imagenet':
              learning_rate_fn = utils.learning_rate_with_decay(
                                  batch_size=params.train_batch_size, batch_denom=params.train_batch_size,
                                  num_images=num_images, boundary_epochs=[30, 60, 80, 90],
                                  decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
                                  warmup=params.warmup, base_lr=params.learning_rate)

            learning_rate = learning_rate_fn(tf.train.get_global_step())
            optimizer = tf.train.MomentumOptimizer(
                      learning_rate=learning_rate, momentum=momentum)

            # Create single grouped train op
            train_op = [
                optimizer.apply_gradients(
                    gradvars, global_step=tf.train.get_global_step(), name='apply_gradient_tower_{}'.format(i))
            ]
            train_op.extend(update_ops)
            tower_ops.append(train_op)


    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):

      examples_sec_hook = utils.ExamplesPerSecondHook(
        params.train_batch_size * (1 + params.redundancy), every_n_steps=10)
      loss = tf.reduce_mean(tower_losses, name='loss')
      tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}
      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)
      train_hooks = [ logging_hook, examples_sec_hook]
      if params.run_type == 'multi':
        sync_hook = utils.SyncHook(scopes=var_scopes, every_n_steps=params.sync_step)
        train_hooks.append(sync_hook)


      train_ops = tf.group(*tower_ops)

      predictions = {
          'classes':
              tf.concat([p['classes'] for p in tower_preds], axis=0),
          'probabilities':
              tf.concat([p['probabilities'] for p in tower_preds], axis=0)
      }
      stacked_labels = tf.concat(labels, axis=0)
      accuracy = tf.metrics.accuracy(stacked_labels, predictions['classes'])
      metrics = {'accuracy': accuracy}
      tf.summary.scalar('accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_ops,
        training_hooks=train_hooks,
        eval_metric_ops=metrics
        )


  if run_type in ['sync', 'async']:
    return _resnet_model_fn_sync
  else:
    return _resnet_model_fn_local



def _tower_fn(is_training, weight_decay, feature, label, data_format,
              num_layers, batch_norm_decay, batch_norm_epsilon, scope, dataset_name='cifar10'):
  """Build computation tower (Resnet).

  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.
    scope: is the scope name that this tower is building its graph on
    dataset_name: choices between cifar10 and cifar100

  Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.

  """
  if dataset_name in ['cifar10','cifar100']:
    model = resnet_model.ResNetCifar(
        num_layers,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        is_training=is_training,
        data_format=data_format,
        dataset_name=dataset_name)
    logits = model.forward_pass(feature, input_data_format='channels_last')
  elif dataset_name == 'imagenet':
    model = resnet_model.ImagenetModel(num_layers,1001, data_format=data_format)
    logits =  model(feature,is_training)
    logits = tf.cast(logits, tf.float32)
  
  tower_pred = {
      'classes': tf.argmax(input=logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
  }

  tower_loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=label)
  tower_loss = tf.reduce_mean(tower_loss)

  model_params = tf.trainable_variables(scope=scope)
  tower_loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])

  tower_grad = tf.gradients(tower_loss, model_params)

  return tower_loss, zip(tower_grad, model_params), tower_pred


def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             use_distortion_for_training=True,
             redundancy=0.0,
             dataset_name='cifar10'):
  """Create input graph for model.

  Args:
    data_dir: Directory where TFRecords representing the dataset are located.
    subset: one of 'train', 'validate' and 'eval'.
    num_shards: num of towers participating in data-parallel training.
    batch_size: total batch size for training to be divided by the number of
    shards.
    use_distortion_for_training: True to use distortions.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  with tf.device('/cpu:0'):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = cifar.CifarDataSet(data_dir, num_shards, subset, use_distortion, redundancy, dataset=dataset_name)
    feature_shards, label_shards = dataset.make_batch(batch_size)
  
    return feature_shards, label_shards


def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  if hparams['config_path']:
      TF_CONFIG = json.load(open(hparams['config_path'], "r"))
      TF_CONFIG['model_dir'] = job_dir
      os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))

  config = utils.RunConfig(
      session_config=sess_config, model_dir=job_dir)
  if hparams['eval']:
    config = config.replace(save_checkpoints_steps=5000)

  train_input_fn = functools.partial(
      input_fn,
      data_dir,
      subset='train',
      num_shards=num_gpus,
      batch_size=hparams['train_batch_size'],
      use_distortion_for_training=use_distortion_for_training,
      redundancy=hparams['redundancy'],
      dataset_name=hparams['dataset'])

  eval_input_fn = functools.partial(
      input_fn,
      data_dir,
      subset='eval',
      batch_size=hparams['eval_batch_size'],
      num_shards=num_gpus,
      dataset_name=hparams['dataset'])

  num_eval_examples = cifar.CifarDataSet.num_examples_per_epoch('eval', hparams['dataset'])
  if num_eval_examples % hparams['eval_batch_size'] != 0:
      raise ValueError(
          'validation set size must be multiple of eval_batch_size')

  train_steps = hparams['train_steps']
  eval_steps = num_eval_examples // hparams['eval_batch_size']

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps, start_delay_secs=0, throttle_secs=1200)

  classifier = tf.estimator.Estimator(
      model_fn=get_model_fn(num_gpus, variable_strategy,
                            config.num_worker_replicas or 1,
                            run_type=hparams['run_type']),
      config=config,
      params=tf.contrib.training.HParams(
                is_chief=config.is_chief,
                **hparams)
  )

  # Create experiment.
  tf.estimator.train_and_evaluate(
      estimator=classifier,
      train_spec=train_spec,
      eval_spec=eval_spec)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the CIFAR-10 input data is stored.')
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--variable-strategy',
      choices=['CPU', 'GPU'],
      type=str,
      default='CPU',
      help='Where to locate variable operations')
  parser.add_argument(
      '--num-gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
      '--num-layers',
      type=int,
      default=50,
      help='The number of layers of the model.')
  parser.add_argument(
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.')
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=128,
      help='Batch size for training.')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=100,
      help='Batch size for validation.')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='Momentum for MomentumOptimizer.')
  parser.add_argument(
      '--weight-decay',
      type=float,
      default=2e-4,
      help='Weight decay for convolutions.')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.1,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
  parser.add_argument(
      '--use-distortion-for-training',
      type=bool,
      default=True,
      help='If doing image distortion for training.')
  parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
  parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
  parser.add_argument(
      '--data-format',
      type=str,
      default=None,
      help="""\
      If not set, the data format best for the training device is used. 
      Allowed values: channels_first (NCHW) channels_last (NHWC).\
      """)
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.')
  parser.add_argument(
      '--batch-norm-decay',
      type=float,
      default=0.997,
      help='Decay for batch norm.')
  parser.add_argument(
      '--batch-norm-epsilon',
      type=float,
      default=1e-5,
      help='Epsilon for batch norm.')
  parser.add_argument(
      '--redundancy',
      type=float,
      default=0.0,
      help='Redundancy for input data.')
  parser.add_argument(
      '--sync-step',
      type=int,
      default=100,
      help='Sync step for local version')
  parser.add_argument(
      '--run-type',
      type=str,
      default='local',
      choices=['sync','async','local','multi'],
      help='The type for running the experiment')
  parser.add_argument(
      '--config-path',
      type=str,
      default=None,
      help='The path to json file of config')
  parser.add_argument(
      '--eval',
      action='store_true',
      default=False,
      help="""If present when running in a distributed environment will run on eval mode.""")
  parser.add_argument(
      '--dataset',
      type=str,
      default='cifar10',
      choices=['cifar10','cifar100','imagenet'],
      help='The dataset for training')
  parser.add_argument(
      '--warmup',
      action='store_true',
      default=True,
      help="""If present when running warmup learning rate will be used""")



  args = parser.parse_args()

  if args.num_gpus > 0:
    assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.variable_strategy == 'GPU':
    raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                     '--variable-strategy=CPU.')
  if (args.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid --num-layers parameter.')
  if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
    raise ValueError('--train-batch-size must be multiple of --num-gpus.')
  if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
    raise ValueError('--eval-batch-size must be multiple of --num-gpus.')
  args.redundancy *= args.num_gpus
  main(**vars(args))
