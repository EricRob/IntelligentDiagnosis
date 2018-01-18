# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import reader
import util
import pdb

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("recur_data_path", None,
                    "Where the recurrence binary file is stored.")
flags.DEFINE_string("nonrecur_data_path", None,
                    "Where the nonrecurrence binary file is stored.")
flags.DEFINE_string("eval_dir", "/tmp/recurrence_lstm/", "Directory where to write event logs." )
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class SeqInput(object):
  """The input data."""

  def __init__(self, config, mode, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.mode = mode
    epoch_max = ((807000 // batch_size) - 1) // num_steps
    if(mode == "train"):
      epoch_mode = int(epoch_max*0.7)
    elif(mode == "valid"):
      epoch_mode = int(epoch_max*0.2)
    else:
      epoch_mode = int(epoch_max*0.1)
    self.epoch_size = epoch_mode
    #pdb.set_trace()
    self.input_data, self.targets = reader.read_data([os.path.join(FLAGS.recur_data_path, str("recurrence_" + mode + ".bin"))],
                                                    [os.path.join(FLAGS.nonrecur_data_path, str("nonrecurrence_" + mode + ".bin"))],
                                                    config)


class SeqModel(object):
  """Based on the PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    self.mode = input_.mode
    size = config.hidden_size

    class_size = config.num_classes
    #inputs = input_.input_data
    #pdb.set_trace()
    inputs = tf.reshape(input_.input_data,[self.batch_size, self.num_steps, -1])
    
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph(inputs, config, is_training)

    softmax_w = tf.get_variable(
        "softmax_w", [size, class_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [class_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, class_size])
    logits = logits[:,self.num_steps-1,:]
    logits_scaled = tf.nn.softmax(logits)


    # Use the contrib sequence loss and average over the batches
    # loss = tf.contrib.seq2seq.sequence_loss(
    #     logits,
    #     input_.targets,
    #     tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
    #     average_across_timesteps=False,
    #     average_across_batch=True)

    #pdb.set_trace()
    # Loss:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels = input_.targets,
      logits = logits)

    value, indice = tf.nn.top_k(logits_scaled, 1)

    labels = tf.Variable(tf.to_int32(tf.ones([self.batch_size])))
    self._labels = tf.assign(labels, input_.targets)
    self._output = indice

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    #pdb.set_trace()

  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(tf.cast(inputs[:, time_step, :], tf.float32), state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}

    tf.add_to_collection(util.with_prefix(self._name, "output"),self._output)
    tf.add_to_collection(util.with_prefix(self._name, "labels"),self._labels)

    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    self._output = tf.get_collection_ref(util.with_prefix(self._name, "output"))
    self._labels = tf.get_collection_ref(util.with_prefix(self._name, "labels"))

    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def output(self):
    return self._output

  @property
  def labels(self):
    return self._labels

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.002 # 10e-5 to 5e-3
  max_grad_norm = 5
  num_layers = 1
  num_steps = 50 #20-500
  hidden_size = 100 
  max_epoch = 3 
  max_max_epoch = 30
  keep_prob = 1.0 # 0.2-0.8
  lr_decay = 1 / 1.15
  batch_size = 50 #10-100
  num_classes = 2
  rnn_mode = BLOCK
  image_size = 100
  image_depth = 1

class ColorConfig(object):
  """Color config."""
  init_scale = 0.1
  learning_rate = 0.001 # 10e-5 to 5e-3
  max_grad_norm = 5
  num_layers = 1
  num_steps = 20 #50-500
  hidden_size = 500 
  max_epoch = 5 
  max_max_epoch = 50
  keep_prob = 1.0 # 0.2-0.8
  lr_decay = 1 / 1.15
  batch_size = 50 #10-100
  num_classes = 2
  rnn_mode = BLOCK
  image_size = 100
  image_depth = 3

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = .001
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  num_classes = 2
  rnn_mode = BLOCK
  image_size = 100
  image_depth = 1

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  num_classes = 2
  rnn_mode = BLOCK
  image_size = 100
  image_depth = 1


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  num_classes = 2
  rnn_mode = BLOCK
  image_size = 100
  image_depth = 3


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  total_loss = 0.0

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
      "output": model.output,
      "labels": model.labels
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  result = np.zeros(4) 
  # pdb.set_trace()
  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    #pdb.set_trace()  
    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
    output = vals["output"]
    labels = vals["labels"]

    costs += cost
    iters += model.input.num_steps

    output = np.squeeze(output)
    labels = np.squeeze(labels)

    if np.size(labels) == 1:
      if labels == 1:
        if labels == output: result[0] += 1  # TP
        else: result[1] += 1  # FN
      else:
        if labels == output: result[2] += 1  # TN
        else: result[3] += 1  # FP
    else:
      for i in range(np.size(labels)):
        if labels[i] == 1:
          if labels[i] == output[i]:
            result[0] += 1  # TP
          else:
            result[1] += 1  # FN
        else:
          if labels[i] == output[i]:
            result[2] += 1  # TN
          else:
            result[3] += 1  # FP

    if verbose and step % (model.input.epoch_size // 10) == 10 and model.mode == "train":
      print("%.3f loss: %.3f speed: %.0f images per second" %
            (step * 1.0 / model.input.epoch_size,
              np.exp(costs / iters),
              iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))
    total_loss += np.exp(costs / iters) 
    start_time = time.time()
  #pdb.set_trace()
  #print("Accuracy : %.3f  Epoch Size : %d " % (correct_num * 1.0 / model.input.epoch_size / model.input.batch_size, model.input.epoch_size)
  print(model.mode + " -- Sens : %.3f  Spec: %.3f  FDR : %.3f FOR: %.3f ACC: %.3f Pos : %d  Neg : %d" 
    % (result[0] * 1.0 / (result[0] + result[1]), # Sensitivity
      result[2] * 1.0 / (result[2] + result[3]), # Specificity
      result[3] * 1.0 / (result[0] + result[3]), # False Discovery
      result[1] * 1.0 / (result[1] + result[2]), # False Omission
      (result[0] + result[2]) * 1.0 / np.sum(result), #Accurary
      result[0] + result[1], #Pos cases
      result[2] + result[3])) #Neg cases
  print("Total Loss: %.2f" % total_loss)
  return costs / model.input.epoch_size


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  elif FLAGS.model == "color":
    config = ColorConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config


def main(_):
  if not FLAGS.recur_data_path:
    raise ValueError("Must set --recur_data_path to recurrence data directory")

  if not FLAGS.nonrecur_data_path:
    raise ValueError("Must set --nonrecur_data_path to recurrence data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is fewer than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))
  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 10
  eval_config.num_steps = 50

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = SeqInput(
        config=config,
        mode="train",
        name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = SeqModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = SeqInput(
        config=config,
        mode="valid",
        name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = SeqModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = SeqInput(
        config=eval_config,
        mode="test",
        name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = SeqModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    #models = {"Train": m}

    for name, model in models.items():
      model.export_ops(name)
  

    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
      training_loss=[]
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.4f" % (i + 1, session.run(m.lr)))
        
        avg_train_cost = run_epoch(session, m, eval_op=m.train_op, verbose=True)
        # print("Train Epoch: %d Avg Train Cost: %.3f" % (i + 1, avg_train_cost))
        #pdb.set_trace()
        avg_valid_cost = run_epoch(session, mvalid, verbose=True)
        # print("Valid Epoch: %d Avg Valid Cost: %.3f" % (i + 1, avg_valid_cost))

        training_loss.append(avg_train_cost)

      avg_test_cost = run_epoch(session, mtest, verbose=True)
      # print("Avg Test Cost: %.3f" % avg_test_cost)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == "__main__":
  tf.app.run()
