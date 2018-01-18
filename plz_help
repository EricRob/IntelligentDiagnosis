#!/usr/bin/env python3

import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import math
from skimage import io

import tiff_patching
import binary_files

# flags = tf.flags

# FLAGS = flags.FLAGS


class PatchConfig(object):
	original_path = "." # Location of image to be split into patches
	patch_size = 500 # Pixel length and width of each patch square
	overlap = 30 # Amount of overlap between each patch image
	sample_size = 100 # Final size of patch (usually 100)
	keep_percentage = 75 # Percentage of patch that must be data (i.e. non-background)
	train_portion = 0.7
	valid_portion = 0.2
	test_portion = 0.1
	image_height = sample_size
	image_width = sample_size
	image_depth = 3
	recurrence_overlap_percentage = 20
	nonrecurrence_overlap_percentage = 10 

def create_patch_folder(label, num_steps, config):

	original_path = os.path.join(config.original_path, label)
	patch_size = config.patch_size
	overlap = config.overlap
	sample_size = config.sample_size
	keep_percentage = config.keep_percentage

	mask_path = os.path.join(original_path,"masks")

	sample_patches = tiff_patching.extract_patches(original_path, patch_size, overlap)
	mask_patches = tiff_patching.extract_patches(mask_path, patch_size, overlap)

	num_patches = tiff_patching.save_patches(original_path,
		sample_patches,
		mask_patches,
		keep_percentage,
		patch_size,
		overlap,
		sample_size)

	print('Created ' + str(num_patches) + ' patches from ' + original_path)

def create_binary_file(label, config):
	bin_train_file = open(label + "_train.bin", "ab+")
	bin_valid_file = open(label + "_valid.bin", "ab+")
	bin_test_file = open(label + "_test.bin", "ab+")

	image_bytes = config.image_height * config.image_width * config.image_depth

	if label == "recurrence":
		sequence_overlap_percentage = config.recurrence_overlap_percentage
	else:
		sequence_overlap_percentage = config.nonrecurrence_overlap_percentage
	
	write_array = np.zeros([num_steps, image_bytes], dtype=np.uint8)
	write_stride = int(math.ceil(num_steps * (1-sequence_overlap_percentage)))

	total_files = sum([len(files) for r, d, files in os.walk(".")])
	counter = 0
	
	# walk over all files in starting directory and sub-directories
	for dirpath, dirnames, filenames in os.walk(config.original_path):
		lst = os.listdir(dirpath)
		lst.sort() # need files in order by index (see fix_patch_alpha) so time series is accurately fed to RNN]
		
		for filename in [f for f in lst if f.endswith(".tif")]:
			
			counter +=1 
			

			# open tiff as grayscale, convert to unsiged 8-bit integer, and append to file
			# img = io.imread(os.path.join(dirpath,filename), as_grey=True)
			# arr = np.floor(img*255)
			# arr = np.uint8(arr)
			
			# open tiff in color
			img = io.imread(os.path.join(dirpath,filename))
			
			arr = img.flatten()

			if((counter / total_files > TRAIN_PORTION) & (counter / total_files  < (1 - TEST_PORTION))):
				bin_valid_file.write(arr.tobytes())
				continue
			if(counter / total_files > (TRAIN_PORTION + VALID_PORTION)):
				#pdb.set_trace()
				bin_test_file.write(arr.tobytes())
				continue


			# At start of iterations, first fill write_array with data until it contains one complete sequence
			if(counter <= num_steps):
				write_array[counter-1,:] = arr
			else: # Shift all images in the write_array to the left one step, then replace the last value with the new image data
				write_array = np.roll(write_array, -1, axis=0)
				write_array[num_steps - 1,:] = arr

			# After the a new portion of the sequence has been added, add the write_array to the binary file
			if(counter % write_stride == 0):
				print(filename + " -- writing " + str(counter))
				if(counter / total_files < TRAIN_PORTION):
					bin_train_file.write(write_array.tobytes())
				#elif(counter / total_files < (TRAIN_PORTION + VALID_PORTION)):
					#bin_valid_file.write(write_array.tobytes())
					#bin_valid_file.write(arr.tobytes())
				#else:
					#bin_test_file.write(write_array.tobytes())
					#bin_test_file.write(arr.tobytes())

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
	print("enter")
	config = get_config()
	directory = os.fsencode(config.original_path)
	print(directory)
	num_steps = 50

	recurrence_file = open("recurrence.txt")
	recurrence_list = recurrence_file.readlines()

	nonrecurrence_file = open("nonrecurrence.txt")
	nonrecurrence_list = nonrecurrence_file.readlines()

	for filename in [f for f in lst if f.endswith(".tif")]:
		filename = os.fsdecode(file)
		if filename in recurrence_list:
			create_patch_folder("recurrence_patches", num_steps, config)
		elif filename in nonrecurrence_list:
			create_patch_folder("nonrecurrence_patches", num_steps, config)
		else:
			print("Unable to find label for " + filename)

	create_binary_file("recurrence", config)
	create_binary_file("nonrecurrence", config)


sys.exit(0)
