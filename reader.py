import tensorflow as tf
import sys
import pdb
from IPython import embed

Py3 = sys.version_info[0] == 3


def _generate_half_batch(sequence, label, min_queue_examples, batch_size):
	""" Construct half of a queued batch, all of the same label
	Args:
		sequence: 2-D tensor of [sequence_length, sequence_length*image_bytes]
		label: 1-D tensor of size [1]
		min_queue_examples: minimum samples to retain in the queue
		config: set of hyperparameters for the current run
	Returns:
		sequences: Half batch of images. 3-D tensor of
		[batch_size // 2, sequence_length, image_bytes] size.
		label_batch: Half batch of labels. 1-D tensor of [batch_size // 2] size.
	"""

	num_preprocess_threads = 16

	# Create a batch of this data type's sequences, half the size of the
	# batch that will be used in the RNN 
	# sequences, label_batch = tf.train.batch(
	# 	[sequence, label],
	# 	batch_size = (batch_size // 2),
	# 	num_threads = num_preprocess_threads,
	# 	capacity = (min_queue_examples + 3 * batch_size) // 2)

	sequences, label_batch = tf.train.shuffle_batch(
		[sequence, label],
		batch_size = (batch_size // 2),
		num_threads = num_preprocess_threads,
		capacity = (min_queue_examples + 3 * batch_size) // 2,
		min_after_dequeue = min_queue_examples // 2)

	# Remove one dimension from label_batch
	#pdb.set_trace()
	label_batch = tf.reshape(label_batch, [batch_size // 2])


	return sequences, label_batch


def _read_from_file(queue, config, class_label):
	""" Reads data from a binary file of cell image data.
	Create an object with information about sequence and
	batch that will be filled with data obtained from the
	queue by the FixedLengthRecordReader
	
	Args:
		queue: FIFOQueue from which records will be read.
		config: set of hyperparameters for the current run
		class_label: All images in a binary file must have same class. This is
		their label.
	Returns:
		An object representing a single sequence with features
			height: Patch image height in pixels
			width: Patch image width in pixels
			depth: Patch image depth in pixels
			image_bytes: Size of a single patch (one element of sequence)
			key: a scalar string Tensor describing the binary filename and
			record number for this sequence
			label: a Tensor with the label classification of 0 or 1
			sequence: a [sequence_length, image_size] size Tensor
	"""
	
	class SequenceRecord(object):
		pass
	result = SequenceRecord()
	
	# Dimensions of the images and the bytes they each take
	# up in the binary file
	result.height = config.image_size
	result.width = config.image_size
	result.depth = config.image_depth
	result.sequence_length = config.num_steps
	#
	# ***** This is where the 30140 would be specified ******
	#
	# ***** The reshaping across the script becomes difficult, particularly with the normalization. ******
	#
	result.image_bytes = (result.height * result.width * result.depth) #+ 140
	record_bytes = result.image_bytes * result.sequence_length
	
	# Create reader with the fixed record length and
	# read off one record
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(queue)

	# Convert from a string to a vector of uint8 that is record_bytes long.
	sequence_data = tf.decode_raw(value, tf.uint8)

	# Treat sequence as an image of dimensions [(steps * patch height), width, depth] and normalize per image
	# Then reshape back to a single sequence

	with tf.device("/cpu:0"):
		normalized_sequence = tf.reshape(sequence_data,
			[result.sequence_length*result.height,result.width, result.depth])
		normalized_sequence = tf.image.per_image_standardization(normalized_sequence)

		result.sequence = tf.reshape(normalized_sequence,
								[result.sequence_length, result.height * result.width * result.depth]) #result.image_bytes])

	#pdb.set_trace()								
	# result.sequence = tf.cast(result.sequence,tf.float32)
	result.label = tf.constant(class_label, shape=[1])

	return result



def read_data(r_filename, nr_filename, config):
	"""Construct input for Sequence RNN using reader ops.
  Args:
    r_filename: filepath to binary file containing recurrence data
    nr_filename: filepath to binary file containing nonrecurrence data
    config: set of hyperparameters for the current run
  Returns:
    sequence_batches_joined: sequences. 3D tensor of [batch_size, sequence_length, image_bytes] size.
    label_batches_joined: Labels. 1D tensor of [batch_size] size.
    """
	
	# Create a queue for each file
	r_queue = tf.train.string_input_producer(r_filename)
	nr_queue = tf.train.string_input_producer(nr_filename)


	r_result = _read_from_file(r_queue, config, class_label = 1)
	nr_result = _read_from_file(nr_queue, config, class_label = 0)

	min_queue_examples = 100 #000 # Currently an arbitrary number

	r_sequences, r_label_batch = _generate_half_batch(
												r_result.sequence,
												r_result.label,
												min_queue_examples,
												batch_size = config.batch_size)
	nr_sequences, nr_label_batch = _generate_half_batch(
												nr_result.sequence,
												nr_result.label,
												min_queue_examples,
												batch_size = config.batch_size)

	sequence_batches_joined = tf.concat([r_sequences, nr_sequences], 0)
	label_batches_joined = tf.concat([r_label_batch, nr_label_batch], 0)

	return sequence_batches_joined, label_batches_joined
