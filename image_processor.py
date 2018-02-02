#!/usr/bin/env python3

import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
import sys
import math
from PIL import Image
from skimage import io
from random import shuffle
import pdb

import tiff_patching

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# flags = tf.flags

# FLAGS = flags.FLAGS


class PatchConfig(object):
	original_path = "/home/wanglab/Desktop/recurrence_seq_lstm/image_data" # Location of image to be split into patches
	patch_size = 500 # Pixel length and width of each patch square
	overlap = 30 # Amount of overlap between patches within a sample
	sample_size = 100 # Final size of patch (usually 100)
	keep_percentage = 75 # Percentage of patch that must be data (i.e. non-background)
	image_height = sample_size
	image_width = sample_size
	image_depth = 3
	recurrence_overlap_percentage = 0
	nonrecurrence_overlap_percentage = 0
	num_steps = 20

def create_patch_folder(filename, label, config):
	
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
def remove_exisiting_binary_file_then_create_new(binary_file_path):
	try:
		os.remove(binary_file_path)
		print("--> existing binary file removed <--")
	except OSError:
		pass
	f = open(binary_file_path, "ab+")
	return f

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def create_binary_file(label, mode, config):

	bin_name = label + "_" + mode + ".bin"
	bin_path = os.path.join(os.path.abspath(config.original_path), label, bin_name)

	if label == "recurrence" :
		sequence_overlap_percentage = config.recurrence_overlap_percentage / 100
	else:
		sequence_overlap_percentage = config.nonrecurrence_overlap_percentage / 100

	num_steps = config.num_steps
	print("*********" + label + " " + mode + "*************")
	bin_file = remove_exisiting_binary_file_then_create_new(bin_path)
	# bin_file = open(label + "_" + mode + ".bin", "ab+")

	IMAGE_HEIGHT = config.image_height
	IMAGE_WIDTH = config.image_width
	IMAGE_DEPTH = config.image_depth
	
	image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH

	write_array = np.zeros([num_steps, image_bytes], dtype=np.uint8)
	write_stride = int(math.floor(num_steps * (1-sequence_overlap_percentage)))

	total_files = sum([len(files) for r, d, files in os.walk(".")])
	total_dirs = sum([len(d) for r, d, f in os.walk(".")])
	counter = 0
	dir_counter = 0
	subjects_file = label + "_" + mode + "_subjects.txt"
	
	# walk over all files in starting directory and sub-directories
	subjects_file = open(os.path.join(config.original_path, "per_mode_subjects", subjects_file), "r")
	subjects_list = subjects_file.read().splitlines()
	
	for subject in subjects_list:
		dir_counter +=1
		print("Writing " + subject + " -- %i/%i" % (dir_counter, len(subjects_list)))
		patch_folder_path = os.path.join(os.path.abspath(config.original_path), label, subject)
		image_list = os.listdir(patch_folder_path)
				
		img_dict = {}	
		for image_name in image_list:
			if image_name.endswith(".tif"):
				image_name_noex = os.path.splitext(image_name)[0]
				image_name_chop = image_name_noex.split("_")
				x_coord = int(image_name_chop[len(image_name_chop)-2])
				y_coord = int(image_name_chop[len(image_name_chop)-1])
				img_dict[x_coord] = img_dict.get(x_coord, {})
				img_dict[x_coord][y_coord] = image_name		

		for x_key in sorted(img_dict.keys()):
			for y_key in sorted(img_dict[x_key].keys()):
				filename = img_dict[x_key][y_key]
				counter +=1
				img = io.imread(os.path.join(patch_folder_path,filename))
				arr = img.flatten()

				if(counter <= num_steps):
					write_array[counter-1,:] = arr
				else: # Shift all images in the write_array to the left one step, then replace the last value with the new image data
					write_array = np.roll(write_array, -1, axis=0)
					write_array[num_steps - 1,:] = arr

				write_count = counter - num_steps

				# After the a new portion of the sequence has been added, add the write_array to the binary file
				if(write_count >= 0) & (write_count % write_stride == 0):
					writing = np.reshape(write_array, (num_steps, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
					bin_file.write(writing)					
	bin_file.close()

def create_binary_mode_files(label, config):
	create_binary_file(label, "train", config)
	create_binary_file(label, "valid", config)
	create_binary_file(label, "test", config)

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
	og_images_directory = os.path.join(config.original_path, "original_images")

	r_file = open(os.path.join(og_images_directory,"recurrence_complete_image_list.txt"))
	r_list = r_file.read().splitlines()

	nr_file = open(os.path.join(og_images_directory,"nonrecurrence_complete_image_list.txt"))
	nr_list = nr_file.read().splitlines()
	

	for file in os.listdir(og_images_directory):
		if file.endswith(".tif"):
			filename = os.fsdecode(file)
			if filename in r_list:
				create_patch_folder(filename, "recurrence", config)
			elif filename in nr_list:
				create_patch_folder(filename, "nonrecurrence", config)
			else:
				print("Unable to find label for " + filename)

	create_binary_mode_files("recurrence", config)
	create_binary_mode_files("nonrecurrence", config)

sys.exit(0)