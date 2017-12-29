
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
