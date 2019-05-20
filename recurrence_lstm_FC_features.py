"""
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
from datetime import datetime

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import os
import sys

import reader_features
import util
# from scipy.misc import imsave
import csv
from termcolor import cprint
# import pdb
# from math import sqrt

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "config", "color",
    "Configuration for the current run. Can be small, medium, large, test, or (default) color.")
flags.DEFINE_string("recur_data_path", None,
                    "Where the recurrence binary file is stored.")
flags.DEFINE_string("nonrecur_data_path", None,
                    "Where the nonrecurrence binary file is stored.")
flags.DEFINE_float("learning_rate", 0.005,
                    "hyperparameters of model's learning rate")
flags.DEFINE_float("keep_prob", 0.7,
                    "hyperparameter of model's dropout rate")
flags.DEFINE_string("eval_dir", "/tmp/recurrence_lstm/", "Directory where to write event logs." )
flags.DEFINE_string("results_prepend", None, "Identifier for saving results to prevent overwrite with similar tests" )
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
flags.DEFINE_integer("epochs", None, "Number of epochs to run")
flags.DEFINE_string("model_path", None, "Location of model to load from last checkpoint")
flags.DEFINE_bool("save_model", False, "Save model and checkpoints for future testing")
flags.DEFINE_integer("num_steps", 20, "Steps in LSTM sequence")
flags.DEFINE_bool("save_samples", False, "Save every sequence as a TIFF in a /samples folder")
flags.DEFINE_string('base_path', '/data/recurrence_seq_lstm/', 'Results folder for holding ')

FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

def arrange_kernels_on_grid(kernel, layer, config, pad = 0):
 #kernel: 3,3,32,32
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  # print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad x_dim and y_dim
  x_pad = tf.pad(kernel, tf.constant( [[pad,pad],[pad,pad],[0,0],[0,0]] ), mode = 'CONSTANT')
  
  # x_dim and y_dim dimensions, w.r.t. padding
  y_dim = kernel.get_shape()[0] + 2 * pad
  x_dim = kernel.get_shape()[1]+ 2* pad
  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x_pad = tf.transpose(x_pad, (3, 0, 1, 2)) #8, 12, 3, 3
  # organize grid on y_dim axis
  x_pad = tf.reshape(x_pad, tf.stack([grid_X, y_dim * grid_Y, x_dim, channels])) #8, 12, 3, 3

  # switch x_dim and y_dim axes
  x_pad = tf.transpose(x_pad, (0, 2, 1, 3)) #8, 3, 12, 3
  # organize grid on x_dim axis
  x_pad = tf.reshape(x_pad, tf.stack([1, x_dim * grid_X, y_dim * grid_Y, channels])) #1, 24, 12, 3

  # back to normal order (not combining with the next step for clarity)
  x_pad = tf.transpose(x_pad, (2, 1, 3, 0)) # 12, 24, 3, 1

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x_pad = tf.transpose(x_pad, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x_pad

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def epoch_size(mode, batch_size, num_steps, image_size):
  meta_data_bytes = 345 # (subject ID bytes + image name bytes + features bytes)
  total_recur_sequences = os.path.getsize(os.path.join(FLAGS.recur_data_path, 'recurrence_' + mode + '.bin')) // (image_size * image_size * 3 * num_steps + meta_data_bytes)
  total_nonrecur_sequences = os.path.getsize(os.path.join(FLAGS.nonrecur_data_path, 'nonrecurrence_' + mode + '.bin')) // (image_size * image_size * 3 * num_steps + meta_data_bytes)
  max_sequences_per_label = max(total_recur_sequences, total_nonrecur_sequences)
  
  # Need to be certain all sequences are run in the test condition, so we go over them 1.5 times.
  # The repeate sequence votes are deduplicated in majority_vote.py.
  if mode == 'test':
    epoch_size = int(1.5*((max_sequences_per_label * 2) //  batch_size))
  else:
    epoch_size = ((max_sequences_per_label * 2) //  batch_size)
  
  return epoch_size #50 

class SeqInput(object):
  """The input data."""

  def __init__(self, config, mode, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.mode = mode
    self.epoch_size = epoch_size(mode, batch_size, num_steps, config.image_size)
    self.input_data, self.targets, self.subjects, self.names, self.coords, self.features = reader_features.read_data([os.path.join(FLAGS.recur_data_path, str("recurrence_" + mode + ".bin"))],
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
    self._num_steps = input_.num_steps
    self.mode = input_.mode
    self._image_size = config.image_size
    self._image_depth = config.image_depth
    self._image_bytes = config.image_size*config.image_size*config.image_depth
    self.features = tf.cast(input_.features, tf.float32)
    size = config.hidden_size

    
    class_size = config.num_classes
    # cnn_input = tf.reshape(input_.input_data,[self.batch_size * self._num_steps, -1])

    # cnn_output = self._add_conv_layers(cnn_input, config, is_training)  
    # cnn_output = tf.reshape(cnn_output, [self.batch_size*self._num_steps, -1]) 

    # # Fully Connected Layer after cnn
    # fc_size = [512] #[4096, 4096]
    # fc_in = cnn_output
    # fc1 = tf.layers.dense(fc_in, fc_size[0], activation=tf.nn.relu)
    # #fc2 = tf.layers.dense(fc1, fc_size[1], activation=tf.nn.relu)

    # inputs = tf.reshape(fc1, [self.batch_size, self._num_steps, fc_size[len(fc_size)-1]])
    # if is_training and config.keep_prob < 1:
    #   inputs = tf.layers.dropout(tf.cast(inputs, tf.float32), 1-config.keep_prob, training=is_training)

    # rnn_output, state = self._build_rnn_graph(inputs, config, is_training)

    # # Fully connected layer after rnn
    # rnn_output = tf.reshape(rnn_output, [self.batch_size, self.num_steps, size])
    # rnn_output = tf.reshape(rnn_output, [self.batch_size, -1])
    # fc1 = tf.layers.dense(rnn_output, 25, tf.nn.elu)


    # pasted = tf.concat([fc1, self.features], 1)


    fc2 = tf.layers.dense(self.features, 25, tf.nn.elu)


    fc_o = tf.layers.dense(fc2, class_size)

    self.subject_ids = input_.subjects
    self.names = input_.names
    self.coords = input_.coords
    
    logits = fc_o	
    # logits = tf.cast(logits, tf.float32)
    unsc_logits = tf.Variable(tf.zeros([self.batch_size, config.num_classes]), name="unscaled_logits",dtype=tf.float32)
    self.unscaled_logits = tf.assign(unsc_logits, logits)

    logits_scaled = tf.nn.softmax(logits)
    sc_logits = tf.Variable(tf.zeros([self.batch_size, config.num_classes]), name="scaled_logits",dtype=tf.float32)
    self.scaled_logits = tf.assign(sc_logits, logits_scaled)

    # Loss:
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels = input_.targets,
      logits = logits)

    value, indice = tf.nn.top_k(logits_scaled, 1)

    labels = tf.Variable(tf.to_int32(tf.ones([self.batch_size])), trainable=False)
    batch = tf.Variable(tf.zeros([self.batch_size, self._num_steps, config.image_size*config.image_size*config.image_depth]),tf.float32)
    self._labels = tf.assign(labels, input_.targets)
    self._input_data = tf.assign(batch, input_.input_data)
    self._output = indice
    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = None

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    #optimizer = tf.train.RMSPropOptimizer(self._lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self._train_op = optimizer.apply_gradients(
          zip(grads, tvars),
          global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training, scope="rnn"):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training, scope)

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

  def _build_rnn_graph_lstm(self, inputs, config, is_training, scope):
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
    inputs = tf.unstack(inputs, num=self._num_steps, axis=1)
    outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
                               initial_state=self._initial_state,
                               scope=scope)
    # outputs = []
    # with tf.variable_scope("RNN"):
    #   for time_step in range(self._num_steps):
    #     if time_step > 0: tf.get_variable_scope().reuse_variables()
    #     (cell_output, state) = cell(tf.cast(inputs[:, time_step, :], tf.float32), state)
    #     outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def _add_conv_layers(self, inputs, config, is_training):
    filters = []
    for _ in np.arange(config.num_cnn_layers):
      filters = filters + [32]

    # filters = [32, 32, 32, 32, 32, 32] #, 512, 512, 512, 512] #[64, 128, 256, 256, 512, 512, 512, 512]
    pools = [1, 2, 6, 8]
    
    cnn_batch_size = config.num_steps * config.batch_size
    conv_in = tf.reshape(inputs, [cnn_batch_size, config.image_size, config.image_size, config.image_depth])

    convolved = conv_in
    # all_layers = tf.Variable()
    for i in range(config.num_cnn_layers):
      
      with tf.variable_scope("conv%s" % str(i+1)):
        convolved_input = convolved
        # Add dropout layer if enabled and not first convolution layer.
        if i > 0 and config.keep_prob < 1:
          convolved_input= tf.layers.dropout(convolved_input,
            rate=1-config.keep_prob,
            training=is_training)

        norm = tf.layers.batch_normalization(convolved_input, axis=-1, momentum=0.1, training=True, epsilon=1e-5, name="norm%s" % str(i+1))

        conv = tf.layers.conv2d(norm,
                filters=filters[i],
                kernel_size=3,
                activation=tf.nn.relu,
                strides=1,
                padding="same",
                name = "conv%s" % str(i+1))

        if i == 0 and is_training:
          kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model/conv1/conv1/kernel')[0]
          grid = arrange_kernels_on_grid(kernel, i, config)
          tf.summary.image('Model/conv1/conv1/kernel', grid, max_outputs=1)
        if i+1 in pools:
          pool = tf.layers.max_pooling2d(conv, pool_size=2, strides=2, padding="same", name="pool1")
        else:
          pool = conv 
        convolved = pool

    return convolved


  def _activation_summary(x):

    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                         tf.nn.zero_fraction(x))

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}

    tf.add_to_collection(util.with_prefix(self._name, "output"),self._output)
    tf.add_to_collection(util.with_prefix(self._name, "labels"),self._labels)
    tf.add_to_collection(util.with_prefix(self._name, "input_data"),self._input_data)
    tf.add_to_collection(util.with_prefix(self._name, "unscaled_logits"), self.unscaled_logits)
    tf.add_to_collection(util.with_prefix(self._name, "scaled_logits"), self.scaled_logits)
    tf.add_to_collection(util.with_prefix(self._name, "subject_ids"), self.subject_ids)
    tf.add_to_collection(util.with_prefix(self._name, "names"), self.names)
    tf.add_to_collection(util.with_prefix(self._name, "coords"), self.coords)
    tf.add_to_collection(util.with_prefix(self._name, "features"), self.features)
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    # util.export_state_tuples(self._initial_state, self._initial_state_name)
    # util.export_state_tuples(self._final_state, self._final_state_name)

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
    self._input_data = tf.get_collection_ref(util.with_prefix(self._name, "input_data"))
    self.unscaled_logits = tf.get_collection_ref(util.with_prefix(self._name, "unscaled_logits"))
    self.scaled_logits = tf.get_collection_ref(util.with_prefix(self._name, "scaled_logits"))
    self.subject_ids = tf.get_collection_ref(util.with_prefix(self._name, "subject_ids"))
    self.names = tf.get_collection_ref(util.with_prefix(self._name, "names"))
    self.coords = tf.get_collection_ref(util.with_prefix(self._name, "coords"))
    self.features = tf.get_collection_ref(util.with_prefix(self._name, "features"))
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    # self._initial_state = util.import_state_tuples(
    #     self._initial_state, self._initial_state_name, num_replicas)
    # self._final_state = util.import_state_tuples(
    #     self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  # @property
  # def features(self):
  #   return self._features
  
  # @property
  # def initial_state(self):
  #   return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def image_bytes(self):
    return self._image_bytes

  @property
  def image_size(self):
    return self._image_size

  @property
  def num_steps(self):
    return self._num_steps

  @property
  def image_depth(self):
    return self._image_depth

  # @property
  # def final_state(self):
  #   return self._final_state

  @property
  def output(self):
    return self._output

  @property
  def labels(self):
    return self._labels

  @property
  def input_data(self):
    return self._input_data

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
  num_features = 7

class ColorConfig(object):
  """Color config."""
  init_scale = 0.1
  learning_rate = 0.001 # 10e-5 to 5e-3 #parameter
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20 #50-500
  hidden_size = 500 #100
  max_epoch = 5 
  max_max_epoch = 50 #100 #50
  keep_prob = 0.50 # 0.2-0.8 #parameter
  lr_decay = 1 #/ 1.15
  batch_size = 30 #10-100
  num_classes = 2
  rnn_mode = BLOCK
  image_size = 100
  image_depth = 3
  num_cnn_layers = 6
  cnn_filters = 20 # not used
  test_mode = 0
  num_features = 7

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
  num_features = 7

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
  num_features = 7

class TestConfig(object):
  """Color config."""
  init_scale = 0.1
  learning_rate = 0.005 # 10e-5 to 5e-3 #parameter
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20 #50-500
  hidden_size = 500 #100
  max_epoch = 5 
  max_max_epoch = 50 #100 #50
  keep_prob = 0.50 # 0.2-0.8 #parameter
  lr_decay = 1 #/ 1.15
  batch_size = 30 #30 #10-100
  num_classes = 2
  rnn_mode = BLOCK
  image_size = 100
  image_depth = 3
  num_cnn_layers = 6
  cnn_filters = 20 # not used
  test_mode = 1
  num_features = 7

def create_voting_file(subject_ids, names, labels, unscaled_logits, scaled_logits, output, coords, features, csv_file):
  # output[0][x][0]
  # labels[0][x]
  # unscaled_logits[0][x][0,1]
  # scaled_logits[0][x][0,1]
  x = 0
  writer = csv.writer(csv_file)
  new_row = []
  for subject in subject_ids[0]:
    new_row = [subject_ids[0][x].tobytes().decode("utf-8").rstrip() ,
                names[0][x].tobytes().decode("utf-8").rstrip() ,
              output[0][x][0],
              labels[0][x],
              unscaled_logits[0][x][0],
              unscaled_logits[0][x][1],
              scaled_logits[0][x][0],
              scaled_logits[0][x][1],
              coords[0][x].tobytes().decode("utf-8"),
              features[0][x] 
              ]
    writer.writerow(new_row)
    x += 1



def save_sample_image(input_data, label, model, step, epoch_count, vals):
  batch = np.array(input_data)
  arr = batch[0,:,:,:]
  seq_pixels = model.num_steps * model.image_size
  arr = np.reshape(arr, (model.batch_size, seq_pixels, model.image_size, model.image_depth))
  samples_folder = os.path.join(FLAGS.base_path, 'samples', FLAGS.results_prepend)
  
  os.makedirs(samples_folder, exist_ok=True)
  for x in range(model.batch_size):
    subject_folder = os.path.join(samples_folder, vals['subject_ids'][0][x].tobytes().decode("utf-8"))
    os.makedirs(subject_folder, exist_ok=True)
    image_folder = os.path.join(subject_folder, vals['names'][0][x].tobytes().decode("utf-8").replace(" ", ""))
    os.makedirs(image_folder, exist_ok=True)
    coords_list = vals['coords'][0][x].tobytes().decode("utf-8").split(" ")
    if not coords_list[1]:
      coords_list[1] = coords_list[2]
    sample_name = coords_list[0] + "_" + coords_list[1] + "_epoch" + str(epoch_count) + "_" + model.mode + "_batch"+ str(step+1) + "_seq" + str(x+1) + "_label" + str(np.array(label)[0,x])+ "_net" + str(vals['output'][0][x][0]) + ".tif"
    imsave(os.path.join(image_folder, sample_name), arr[x,:,:,:])


def run_epoch(session, model, results_file, epoch_count, csv_file=None, eval_op=None, verbose=False, test_mode=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  # state = session.run(model.initial_state)
  fetches = {
      "cost": model.cost,
      # "final_state": model.final_state,
      "output": model.output,
      "labels": model.labels,
      "input_data": model.input_data,
      "unscaled_logits": model.unscaled_logits,
      "scaled_logits": model.scaled_logits,
      "subject_ids": model.subject_ids,
      "names": model.names,
      "coords": model.coords,
      'features': model.features
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  result = np.zeros(4) 
  for step in range(model.input.epoch_size):
    # vals = session.run(fetches, feed_dict)
    vals = session.run(fetches)
    cost = vals["cost"]
    # state = vals["final_state"]
    output = vals["output"]
    labels = vals["labels"]
    input_data = vals["input_data"]
    unscaled_logits = vals["unscaled_logits"]
    scaled_logits = vals["scaled_logits"]
    subject_ids = vals["subject_ids"]
    names = vals["names"]
    coords = vals["coords"]
    features = vals['features']

    if test_mode:
      create_voting_file(subject_ids, names, labels, unscaled_logits, scaled_logits, output, coords, features, csv_file)
    if FLAGS.save_samples:
      save_sample_image(input_data, labels, model, step, epoch_count, vals)
    
    # if np.isnan(cost):
    #   pdb.set_trace()

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
         
    if model.mode == "train":
      if verbose and step % (model.input.epoch_size // 10) == 5:
        print("%.3f loss: %.3f" %
              (step * 1.0 / model.input.epoch_size,
                 costs / (step + 1.0)
                ))
        # loss_file.write("%.3f, %.3f\n" %
        #       (step * 1.0 / model.input.epoch_size,
        #         np.exp(costs / iters)
        #         ))
    
    start_time = time.time()
  #print("Accuracy : %.3f  Epoch Size : %d " % (correct_num * 1.0 / model.input.epoch_size / model.input.batch_size, model.input.epoch_size)
  print(model.mode + " -- Sens : %.3f  Spec: %.3f  FDR : %.3f FOR: %.3f ACC: %.3f Pos : %d  Neg : %d" 
    % (result[0] * 1.0 / (result[0] + result[1]), # Sensitivity
      result[2] * 1.0 / (result[2] + result[3]), # Specificity
      result[3] * 1.0 / (result[0] + result[3]), # False Discovery
      result[1] * 1.0 / (result[1] + result[2]), # False Omission
      (result[0] + result[2]) * 1.0 / np.sum(result), #Accurary
      result[0] + result[1], #Pos cases
      result[2] + result[3])) #Neg cases

  #print(model.mode + " loss: %.2f" % cost)

  results_file.write("%.3f, %.3f, %.3f, %.3f, %.3f, %d, %d, %.2f\n" 
    % (result[0] * 1.0 / (result[0] + result[1]), # Sensitivity
      result[2] * 1.0 / (result[2] + result[3]), # Specificity
      result[3] * 1.0 / (result[0] + result[3]), # False Discovery
      result[1] * 1.0 / (result[1] + result[2]), # False Omission
      (result[0] + result[2]) * 1.0 / np.sum(result), #Accurary
      result[0] + result[1], #Pos cases
      result[2] + result[3], #Neg cases
      costs)) 

  return costs / model.input.epoch_size


def get_config():
  """Get model config."""
  config = None
  if FLAGS.config == "small":
    config = SmallConfig()
  elif FLAGS.config == "medium":
    config = MediumConfig()
  elif FLAGS.config == "large":
    config = LargeConfig()
  elif FLAGS.config == "test":
    config = TestConfig()
  elif FLAGS.config == "color":
    config = ColorConfig()
  else:
    raise ValueError("Invalid config: %s", FLAGS.config)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config

def create_log_directory(eval_dir):
  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)

def data_exists():
  conditions = ['train', 'valid', 'test']
  for c in conditions:
    r_path = os.path.join(FLAGS.recur_data_path, 'recurrence_' + c + '.bin')
    nr_path = os.path.join(FLAGS.recur_data_path, 'nonrecurrence_' + c + '.bin')
    if not os.path.exists(r_path) or not os.path.exists(nr_path):
      return False
  return True
def main(_):
  waiting = False
  while (not data_exists()):
    waiting = True
    cprint(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' -- Waiting another minute for data...', 'yellow')
    time.sleep(60)
  cprint('Data exists!', 'green')
  if waiting:
    cprint('waited for data, now waiting five more minutes to make sure it is all there ... ', 'yellow')
    time.sleep(300)
  stdout_backup = sys.stdout

  config = get_config()

  if not FLAGS.recur_data_path:
    raise ValueError("Must set --recur_data_path to recurrence data directory")

  if not FLAGS.nonrecur_data_path:
    raise ValueError("Must set --nonrecur_data_path to recurrence data directory")
  
  #If training (not testing) need to set these training parameters (but they are set by default in flags)
  if config.test_mode == 0:
    if not FLAGS.learning_rate:
      raise ValueError("Must set --learning_rate hyperparameter")

    if not FLAGS.keep_prob:
      raise ValueError("Must set --keep_prob hyperparameter (dropout rate)")
    config.learning_rate = FLAGS.learning_rate
    config.keep_prob = FLAGS.keep_prob
  
  # gpus = [
  #     x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  # ]
  # if FLAGS.num_gpus > len(gpus):
  #   raise ValueError(
  #       "Your machine has only %d gpus "
  #       "which is fewer than the requested --num_gpus=%d."
  #       % (len(gpus), FLAGS.num_gpus))

  if FLAGS.epochs:
    config.max_max_epoch = FLAGS.epochs

  # if FLAGS.omen_run:
  #   base_directory = os.path.join('/hdd', 'ID_net')
  # elif FLAGS.park:
  #   base_directory = os.path.join('/home', 'param', 'IntelligentDiagnosis')
  # else:
  #   base_directory = os.path.join('/data', 'recurrence_seq_lstm')
  os.makedirs(os.path.join(FLAGS.base_path, 'results'), exist_ok=True)
  results_path = os.path.join(FLAGS.base_path, "results", FLAGS.results_prepend)
  
  cprint("Data Sources:", 'white', 'on_magenta')
  cprint(FLAGS.recur_data_path, 'magenta', 'on_white')
  cprint(FLAGS.nonrecur_data_path, 'magenta', 'on_white')
  if FLAGS.config == 'test':
    cprint('Testing with model %s' % (FLAGS.model_path), 'white', 'on_magenta' )
  cprint("Results saved to %s" % (results_path), 'white', 'on_magenta')
  
  os.makedirs(results_path, exist_ok=True)
  if config.test_mode == 0:  
    # os.makedirs(os.path.join(base_directory,"samples"), exist_ok=True)
    train_file = open(os.path.join(results_path,"train_results.txt"), 'at+')
    valid_file = open(os.path.join(results_path,"valid_results.txt"), 'at+')
    test_file = open(os.path.join(results_path,"test_results.txt"), 'at+')
  else:
    test_file = open(os.path.join(results_path,"secondary_test_results.txt"), 'at+')
    # csv_file = open(os.path.join(results_path,"secondary_voting_file.csv"), 'wt+')
    csv_file = open(os.path.join(results_path,"voting_file.csv"), 'wt+')
    csv_file.write("ID,names,output,label,unscaled_nr,unscaled_rec,scaled_nr,scaled_rec,coords\n")

  if FLAGS.save_samples:
    os.makedirs(os.path.join(FLAGS.base_path,"samples"), exist_ok=True)


  eval_config = get_config()
  #eval_config.batch_size = 10
  #eval_config.num_steps = 50

  create_log_directory(FLAGS.eval_dir)

  with tf.Graph().as_default() as g:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    #init_op = tf.global_variables_initializer()
    with tf.name_scope("Train"):
      train_input = SeqInput(config=config, mode="train", name="TrainInput")

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
      with tf.variable_scope("Model", reuse=True, initializer=None):
        mtest = SeqModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    
    # Create new saver separate from managed_session in order to set max_to_keep=None
    # and save every epoch, rather than default of max_to_keep=5 checkpoints.
    saver = tf.train.Saver(max_to_keep=None)

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
    if FLAGS.save_model:
      FLAGS.save_path = os.path.join(results_path,"model")
    
    sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_model_secs=1500)
    # summary_op = tf.summary.merge_all()
    if config.test_mode == 0:
      summary_writer = tf.summary.FileWriter(FLAGS.save_path)
    
    gpu_options = tf.GPUOptions(allow_growth=True) #per_process_gpu_memory_fraction=1.0)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement, gpu_options=gpu_options)


    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        #print(variable_parameters)
        total_parameters += variable_parameters
    cprint(total_parameters, 'green')  

    with sv.managed_session(config=config_proto) as session:
      if config.test_mode == 0:
        training_loss=[]
        training_start = time.time()
        epoch_start = time.time()
        for i in range(config.max_max_epoch):
          if i:
            last_epoch_time = epoch_start
            epoch_start = time.time()
            epoch_min_elapsed = int((epoch_start - last_epoch_time) // 60)
            epoch_sec_elapsed = int((epoch_start - last_epoch_time) % 60)
            cprint('Epoch length: ' + "{:02d}".format(epoch_min_elapsed) + ':' + "{:02d}".format(epoch_sec_elapsed), 'grey', 'on_white')

            total_hour_elapsed = int((epoch_start - training_start) // 3600)
            total_min_elapsed = int(((epoch_start - training_start) % 3600) // 60)
            total_sec_elapsed = int((epoch_start - training_start) % 60)
            
            avg_time = (epoch_start - training_start) / i
            avg_min = int(avg_time // 60)
            avg_sec = int(avg_time % 60)

            cprint('Total training time: ' + "{:02d}".format(total_hour_elapsed) + ':'+ "{:02d}".format(total_min_elapsed) + ':' + "{:02d}".format(total_sec_elapsed) + ', Avg Epoch: ' + "{:02d}".format(avg_min) + ':' + "{:02d}".format(avg_sec) , 'grey', 'on_white')
            remaining_time = avg_time * (config.max_max_epoch - i)
            rem_hour = int(remaining_time // 3600)
            rem_min = int((remaining_time % 3600) // 60)
            rem_sec = int(remaining_time % 60)
            cprint('Estimated remaining time: '  + "{:02d}".format(rem_hour) + ':' + "{:02d}".format(rem_min) + ':' + "{:02d}".format(rem_sec), 'green', 'on_white')
          
          lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          m.assign_lr(session, config.learning_rate * lr_decay)
          print("Epoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
          
          avg_train_cost = run_epoch(session, m, train_file, i + 1, eval_op=m.train_op, verbose=True)
          print("Train Epoch: %d Avg Train Cost: %.3f" % (i + 1, avg_train_cost))
          avg_valid_cost = run_epoch(session, mvalid, valid_file, i + 1, verbose=True)
          print("Valid Epoch: %d Avg Valid Cost: %.3f" % (i + 1, avg_valid_cost))

          training_loss.append(avg_train_cost)


          if config.max_max_epoch == (i+1):
            csv_file = open(os.path.join(results_path,"voting_file.csv"), 'wt+')
            csv_file.write("ID,name,output,label,unscaled_nr,unscaled_rec,scaled_nr,scaled_rec,coords\n")
            avg_test_cost = run_epoch(session, mtest, test_file, i + 1, csv_file=csv_file, verbose=True, test_mode=True)
            csv_file.close()
          else:
            avg_test_cost = run_epoch(session, mtest, test_file, i + 1, verbose=True)
          print("Avg Test Cost: %.3f" % avg_test_cost)

          if FLAGS.save_model and (i > config.max_max_epoch / 2):
            saver.save(session, FLAGS.save_path, global_step=sv.global_step)
     
        train_file.close()
        valid_file.close()
        test_file.close()
      
      else:
        if not FLAGS.model_path:
          raise ValueError("If running in test mode, must set --model_path to checkpoint directory")
        saver.restore(session, tf.train.latest_checkpoint(FLAGS.model_path))
        test_loss = run_epoch(session, mtest, test_file, 1, csv_file=csv_file, verbose=False, test_mode=True)
        print("Test Loss: %.3f" % test_loss)
  sys.stdout = stdout_backup
if __name__ == "__main__":
  tf.app.run()
