import tensorflow as tf
import sys

Py3 = sys.version_info[0] == 3


def _generate_half_batch(sequence, label, min_queue_examples, batch_size):
	""" Construct half of a queued batch, all of the same label
	Args:
		sequence: 2-D tensor of [sequence_length, sequence_length*image_bytes]
		label: 1-D tensor of size [1]
		min_queue_examples: minimum samples to retain in the queue
		batch_size: number of image sequences per batch
	Returns:
		sequences: Half batch of images. 3-D tensor of
		[batch_size // 2, sequence_length, image_bytes] size.
		label_batch: Half batch of labels. 1-D tensor of [batch_size // 2] size.
	"""

	num_preprocess_threads = 16

	# Create a batch of this data type's sequences, half the size of the
	# batch that will be used in the RNN 
	sequences, label_batch = tf.train.batch(
		[sequence, label],
		batch_size = (batch_size // 2),
		num_threads = num_preprocess_threads,
		capacity = (min_queue_examples + 3 * batch_size) // 2)

	# Remove one dimension from label_batch
	label_batch = tf.reshape(label_batch, [batch_size // 2])


	return sequences, label_batch


def _read_from_file(queue, class_label):
	""" Reads data from a binary file of cell image data.
	Create an object with information about sequence and
	batch that will be filled with data obtained from the
	queue by the FixedLengthRecordReader
	
	Args:
		queue: FIFOQueue from which records will be read.
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
	result.height = 100
	result.width = 100
	result.depth = 1
	result.sequence_length = 250
	result.image_bytes = result.height * result.width * result.depth
	record_bytes = result.image_bytes * result.sequence_length
	
	# Create reader with the fixed record length and
	# read off one record
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(queue)

	# Convert from a string to a vector of uint8 that is record_bytes long.
	sequence_data = tf.decode_raw(value, tf.uint8)
	result.sequence = tf.reshape(sequence_data,
								[result.sequence_length, result.image_bytes])

	result.label = tf.constant(class_label, shape=[1])

	return result



def read_data(r_filename, nr_filename, batch_size):
	"""Construct input for Sequence RNN using reader ops.
  Args:
    r_filename: filepath to binary file containing recurrence data
    nr_filename: filepath to binary file containing nonrecurrence data
    batch_size: Number of sequences per batch
  Returns:
    sequence_batches_joined: sequences. 3D tensor of [batch_size, sequence_length, image_bytes] size.
    label_batches_joined: Labels. 1D tensor of [batch_size] size.
    """
	
	# Create a queue for each file
	r_queue = tf.train.string_input_producer(r_filename)
	nr_queue = tf.train.string_input_producer(nr_filename)


	r_result = _read_from_file(r_queue, class_label = 1)
	nr_result = _read_from_file(nr_queue, class_label = 0)

	min_queue_examples = 10000 # Currently an arbitrary number

	r_sequences, r_label_batch = _generate_half_batch(
												r_result.sequence,
												r_result.label,
												min_queue_examples,
												batch_size)
	nr_sequences, nr_label_batch = _generate_half_batch(
												nr_result.sequence,
												nr_result.label,
												min_queue_examples,
												batch_size)

	sequence_batches_joined = tf.concat([r_sequences, nr_sequences], 0)
	label_batches_joined = tf.concat([r_label_batch, nr_label_batch], 0)

	return sequence_batches_joined, label_batches_joined
