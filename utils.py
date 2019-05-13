"""Utilities for training
"""
import collections
import six

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config


"""
__author__ = "Mohammad Mahdi Kamani"
__copyright__ = "Copyright 2019, Mohammad Mahdi Kamani"

__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Madhi Kamani"
__status__ = "Prototype"
"""

class RunConfig(tf.estimator.RunConfig):
  def uid(self, whitelist=None):
    """Generates a 'Unique Identifier' based on all internal fields.
    Caller should use the uid string to check `RunConfig` instance integrity
    in one session use, but should not rely on the implementation details, which
    is subject to change.
    Args:
      whitelist: A list of the string names of the properties uid should not
        include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
        includes most properties user allowes to change.
    Returns:
      A uid string.
    """
    if whitelist is None:
      whitelist = run_config._DEFAULT_UID_WHITE_LIST

    state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    # Pop out the keys in whitelist.
    for k in whitelist:
      state.pop('_' + k, None)

    ordered_state = collections.OrderedDict(
        sorted(state.items(), key=lambda t: t[0]))
    # For class instance without __repr__, some special cares are required.
    # Otherwise, the object address will be used.
    if '_cluster_spec' in ordered_state:
      ordered_state['_cluster_spec'] = collections.OrderedDict(
         sorted(ordered_state['_cluster_spec'].as_dict().items(),
                key=lambda t: t[0])
      )
    return ', '.join(
        '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state)) 


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
  """Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
  """

  def __init__(
      self,
      batch_size,
      every_n_steps=100,
      every_n_secs=None,):
    """Initializer for ExamplesPerSecondHook.

      Args:
      batch_size: Total batch size used to calculate examples/second from
      global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds.
    """
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        current_examples_per_sec = steps_per_sec * self._batch_size
        # Average examples/sec followed by current examples/sec
        logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                     average_examples_per_sec, current_examples_per_sec,
                     self._total_steps)

def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
  if ps_ops == None:
    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

  if ps_strategy is None:
    ps_strategy = device_setter._RoundRobinStrategy(num_devices)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")

  def _local_device_chooser(op):
    current_device = pydev.DeviceSpec.from_string(op.device or "")

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if node_def.op in ps_ops:
      ps_device_spec = pydev.DeviceSpec.from_string(
          '/{}:{}'.format(ps_device_type, ps_strategy(op)))

      ps_device_spec.merge_from(current_device)
      return ps_device_spec.to_string()
    else:
      worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
      worker_device_spec.merge_from(current_device)
      return worker_device_spec.to_string()
  return _local_device_chooser


class SyncHook(tf.train.SessionRunHook):

  def __init__(self, scopes, every_n_steps=1000, every_n_secs=None):

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)
    self.scopes = scopes
    self.parameters = {}
    for scope in self.scopes:
      self.parameters[scope] = tf.trainable_variables(scope=scope)
    self.par_len = len(self.parameters[self.scopes[0]])
    assert self.par_len == np.mean([len(self.parameters[s]) for s in self.scopes])

    self.update_ops=[]
    for i in range(self.par_len):
      avg = tf.reduce_mean([self.parameters[s][i] for s in self.scopes],axis=0)
      self.update_ops.extend([tf.assign(self.parameters[s][i],avg) for s in self.scopes])


  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use SyncHook.')

  def before_run(self, run_context):
    global_step = run_context.session.run(self._global_step_tensor)


    if self._timer.should_trigger_for_step(global_step):

      logging.info('Parameters sync in progress for step: {}'.format(global_step))

      run_context.session.run(self.update_ops)
      self._timer.update_last_triggered_step(global_step)
      # return tf.train.SessionRunArgs(op)



def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
    base_lr=0.1, warmup=False, enable_lars=False):
  """Get a learning rate that decays step-wise as training progresses.
  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup: Run a 5 epoch warmup to the initial lr.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 5 epoch warm up."""
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    if warmup:
      warmup_steps = int(batches_per_epoch * 5)
      warmup_lr = (
          initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
              warmup_steps, tf.float32))
      return tf.cond(pred=global_step < warmup_steps,
                     true_fn=lambda: warmup_lr,
                     false_fn=lambda: lr)
    return lr

  def poly_rate_fn(global_step,batch_size):
    """Handles linear scaling rule, gradual warmup, and LR decay.
    The learning rate starts at 0, then it increases linearly per step.  After
    FLAGS.poly_warmup_epochs, we reach the base learning rate (scaled to account
    for batch size). The learning rate is then decayed using a polynomial rate
    decay schedule with power 2.0.
    Args:
      global_step: the current global_step
    Returns:
      returns the current learning rate
    """

    # Learning rate schedule for LARS polynomial schedule
    if batch_size < 8192:
      plr = 5.0
      w_epochs = 5
    elif batch_size < 16384:
      plr = 10.0
      w_epochs = 5
    elif batch_size < 32768:
      plr = 25.0
      w_epochs = 5
    else:
      plr = 32.0
      w_epochs = 14

    w_steps = int(w_epochs * batches_per_epoch)
    wrate = (plr * tf.cast(global_step, tf.float32) / tf.cast(
        w_steps, tf.float32))

    num_epochs = 90
    train_steps = batches_per_epoch * num_epochs

    min_step = tf.constant(1, dtype=tf.int64)
    decay_steps = tf.maximum(min_step, tf.subtract(global_step, w_steps))
    poly_rate = tf.train.polynomial_decay(
        plr,
        decay_steps,
        train_steps - w_steps + 1,
        power=2.0)
    return tf.where(global_step <= w_steps, wrate, poly_rate)

  # For LARS we have a new learning rate schedule
  if enable_lars:
    return poly_rate_fn

  return learning_rate_fn