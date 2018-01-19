#!/usr/bin/env python3

import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
import sys
import math
from skimage import io
from random import shuffle
import pdb

import tiff_patching

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# flags = tf.flags

# FLAGS = flags.FLAGS


class PatchConfig(object):
	original_path = "image_data" # Location of image to be split into patches
	patch_size = 500 # Pixel length and width of each patch square
	overlap = 30 # Amount of overlap between patches within a sample
	sample_size = 100 # Final size of patch (usually 100)
	keep_percentage = 75 # Percentage of patch that must be data (i.e. non-background)
	train_portion = 0.7
	valid_portion = 0.2
	test_portion = 0.1
	image_height = sample_size
	image_width = sample_size
	image_depth = 3
	recurrence_overlap_percentage = 98
	nonrecurrence_overlap_percentage = 70 

def create_patch_folder(filename, label, num_steps, config):
	
	folder_path = os.path.join(os.path.abspath(config.original_path), label, os.path.splitext(filename)[0] +'_patches')

	if os.path.exists(folder_path):
		print("Skipping " + filename)
		return

	patch_size = config.patch_size
	overlap = config.overlap
	sample_size = config.sample_size
	keep_percentage = config.keep_percentage

	file_path = os.path.join(config.original_path, filename)

	mask_path = "mask_" + filename
	mask_path = os.path.join(config.original_path,"masks", mask_path)

	sample_patches = tiff_patching.extract_patches(file_path, patch_size, overlap)
	mask_patches = tiff_patching.extract_patches(mask_path, patch_size, overlap)
	save_path = os.path.join(label, filename)

	num_patches = tiff_patching.save_patches(filename,
		config.original_path,
		label,
		sample_patches,
		mask_patches,
		keep_percentage,
		patch_size,
		overlap,
		sample_size)

	print('Created ' + str(num_patches) + ' patches from ' + filename)

# Image bytes are appended to the binary file, so if a binary file exists at the start then it must be removed
# After (potentially) deleting file, open a new appendable binary file with read/write capabilities
def remove_exisiting_binary_file_then_create_new(label, filename):
	binary_file_path = os.path.join(label, label + filename)
	try:
		os.remove(binary_file_path)
	except OSError:
		pass
	
	return open(binary_file_path, "ab+")

# Write binary files of specifified proportions for network data.

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def create_binary_file(label, config):

	TRAIN_PORTION = config.train_portion
	VALID_PORTION = config.valid_portion
	TEST_PORTION = config.test_portion

	bin_train_file = remove_exisiting_binary_file_then_create_new(label, "_train.bin")
	bin_valid_file = remove_exisiting_binary_file_then_create_new(label,  "_valid.bin")
	bin_test_file = remove_exisiting_binary_file_then_create_new(label, "_test.bin")

	image_bytes = config.image_height * config.image_width * config.image_depth

	label_path = os.path.join(config.original_path, label + '_patches')

	if label == "recurrence":
		sequence_overlap_percentage = config.recurrence_overlap_percentage
	else:
		sequence_overlap_percentage = config.nonrecurrence_overlap_percentage
	
	write_array = np.zeros([num_steps, image_bytes], dtype=np.uint8)
	write_stride = int(math.ceil(num_steps * (1-sequence_overlap_percentage)))

	total_sequences = np.zeros([num_steps, image_bytes, 1]) # Initialize array that will be shuffled. Third dimension is 1 because it will be appended to

	total_files = sum([len(files) for r, d, files in os.walk(label_path)])
	counter = 0
	
	dir_list = get_immediate_subdirectories(label_path)
	dir_list = shuffle(dir_list)

	# walk over all subdirectories of image patches
	for dirpath in dir_list:
		lst = os.listdir(dirpath)
		lst.sort() # need files in order by index (see fix_patch_alpha) so time series is accurately fed to RNN
		
		for filename in [f for f in lst if f.endswith(".tif")]:
			
			counter +=1 
			

			# open tiff as grayscale, convert to unsiged 8-bit integer, and append to file
			# img = io.imread(os.path.join(dirpath,filename), as_grey=True)
			# arr = np.floor(img*255)
			# arr = np.uint8(arr)
			
			# open tiff in color
			img = io.imread(os.path.join(dirpath,filename))
			
			arr = img.flatten()

			# At start of iterations, first fill write_array with data until it contains one complete sequence
			if(counter <= num_steps):
				write_array[counter-1,:] = arr
			else: # Shift all images in the write_array to the left one step, then replace the last value with the new image data
				np.append(total_sequences, write_array, axis=2)
				write_array = np.roll(write_array, -1, axis=0)
				write_array[num_steps - 1,:] = arr

		# After the a new portion of the sequence has been added, add the write_array to the binary file
		if(counter % write_stride == 0):
			print(filename + " -- writing " + str(counter))
			
			# Write train data
			if(counter / total_files < TRAIN_PORTION):
				bin_train_file.write(write_array.tobytes())
			
			# Write validation data
			elif(counter / total_files < (TRAIN_PORTION + VALID_PORTION)):
				bin_valid_file.write(write_array.tobytes())
				bin_valid_file.write(arr.tobytes())

			# Write test data
			else:
				bin_test_file.write(write_array.tobytes())
				bin_test_file.write(arr.tobytes())

	bin_train_file.close()
	bin_valid_file.close()
	bin_test_file.close()

def get_config():
  # """Get model config."""
  # config = None
  # if FLAGS.mode == "patch":
  #   config = PatchConfig()
  # else:
  #   raise ValueError("Invalid mode: %s", FLAGS.model)

  return PatchConfig()

if __name__ == '__main__':

	# if not FLAGS.data_dir:
	# 	raise ValueError("Must set --data_dir to directory of original and mask images")

	# if not FLAGS.num_steps:
	# 	raise ValueError("must set --num_steps to the length of RNN sequence")

	#directory = os.fsencode(FLAGS.data_dir)
	# num_steps = FLAGS.num_steps
	config = get_config()
	directory = config.original_path
	num_steps = 1

	r_file = open("recurrence.txt")
	r_list = r_file.read().splitlines()
	# r_path = os.path.join(directory, "recurrence")
	# if not os.path.exists(r_path):
	# 	os.makedirs(r_path)

	nr_file = open("nonrecurrence.txt")
	nr_list = nr_file.read().splitlines()
	nr_path = os.path.join(directory, "nonrecurrence")	
	# if not os.path.exists(nr_path):
	# 	os.makedirs(nr_path)		

	for file in os.listdir(directory):
		if file.endswith(".tif"):
			filename = os.fsdecode(file)
			print(filename)
			if filename in r_list:
				create_patch_folder(filename, "recurrence", num_steps, config)
			elif filename in nr_list:
				create_patch_folder(filename, "nonrecurrence", num_steps, config)
			else:
				print("Unable to find label for " + filename)

	create_binary_file("recurrence", config)
	create_binary_file("nonrecurrence", config)

sys.exit(0)
