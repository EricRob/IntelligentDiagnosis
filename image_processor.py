#!/usr/bin/python3 python3

import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
import sys
import math
from PIL import Image
from tensorflow import flags
from skimage import io
from random import shuffle
import pdb
from termcolor import cprint
import csv
from shutil import copyfile
import centroid_sampler as gauss

import tiff_patching

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

flags.DEFINE_integer("r_overlap", 75, "Percentage of overlap of recurrence samples")
flags.DEFINE_integer("nr_overlap", 50, "Percentage of overlap of nonrecurrence samples")
flags.DEFINE_integer("num_steps", 20, "Number of steps in RNN sequence")
flags.DEFINE_bool("recurrence_only", False, "Generate only recurrence binary file")
flags.DEFINE_bool("nonrecurrence_only", False, "Generate nonrecurrence binary file only")
flags.DEFINE_string("seg_path", None, "Location of segmentations folder")
flags.DEFINE_string("mode", None, "Create binary file for a specific mode between train, validation, or test")
flags.DEFINE_bool("randomized_conditions", False, "Generate multiple binary files for randomized conditions with subject variability")
flags.DEFINE_string("condition_path", None, "Path of condition folder with condition subject lists")
flags.DEFINE_string("config", None, "Configuration for generating patches.")
flags.DEFINE_string("sampling_method", "gauss", "Sampling pattern for generating sequences")
flags.DEFINE_string("patches_only",False,"Generate only patches for new images, and no binary files.")
flags.DEFINE_integer("gauss_seq", 6, "Number of sequences to generate per tile with gaussian sampling")
flags.DEFINE_integer("gauss_stdev", 1500, "Standard deviation of pixel distance from center for gaussian sampling")

FLAGS = flags.FLAGS

class OriginalPatchConfig(object):
	image_data_folder_path = "/home/wanglab/Desktop/recurrence_seq_lstm/image_data" # Location of image to be split into patches
	patch_size = 500 # Pixel length and width of each patch square
	overlap = 30 # Amount of overlap between patches within a sample
	sample_size = 100 # Final size of patch (usually 100)
	keep_percentage = 75 # Percentage of patch that must be data (i.e. non-background)
	image_height = sample_size
	image_width = sample_size
	image_depth = 3
	recurrence_overlap_percentage = FLAGS.r_overlap
	nonrecurrence_overlap_percentage = FLAGS.nr_overlap
	num_steps = FLAGS.num_steps

def create_segmentation(seg_path, seg_folder_path, img_dict, patch_folder_path, config):
	seg_img = io.imread(seg_path)[:,:,0]
	os.makedirs(seg_folder_path, exist_ok=True)
	for x_key in sorted(img_dict.keys()):
			for y_key in sorted(img_dict[x_key].keys()):
				filename = img_dict[x_key][y_key]
				single_patch_path = os.path.join(patch_folder_path, filename)
				
				x_max = x_key + config.patch_size
				y_max = y_key + config.patch_size
				if (tiff_patching.seg_above_threshold(seg_img[y_key:y_max, x_key:x_max], config.keep_percentage)):
					copyfile(single_patch_path, os.path.join(seg_folder_path, filename))
					print("Copied %s to %s" % (single_patch_path, os.path.join(seg_folder_path, filename)))
				else:
					print("%s not in segmentation" % (single_patch_path))
	
def create_patch_folder(filename, label, config):
	
	patch_folder_path = os.path.join(os.path.abspath(config.image_data_folder_path), label, str(config.patch_size) + "_pixel_patches", os.path.splitext(filename)[0] +'_patches')
	
	patch_size = config.patch_size
	overlap = config.overlap
	sample_size = config.sample_size
	keep_percentage = config.keep_percentage

	if os.path.exists(patch_folder_path):
		if not FLAGS.seg_path:
			return
		seg_patch_folder_path = os.path.join(FLAGS.seg_path, label, os.path.splitext(filename)[0] + '_patches')
		if os.path.exists(seg_patch_folder_path):
			print("Skipping " + filename)
			return
		if not os.path.exists(os.path.join(FLAGS.seg_path,"seg_masks","seg_" + filename)):
			print("Skipping %s " % filename)
			return
	else:
		file_path = os.path.join(config.image_data_folder_path, "original_images", filename)

		mask_path = "mask_" + filename
		mask_path = os.path.join(config.image_data_folder_path,"masks", mask_path)

		sample_patches = tiff_patching.extract_patches(file_path, patch_size, overlap)
		mask_patches = tiff_patching.extract_patches(mask_path, patch_size, overlap)
		save_path = os.path.join(label, filename)

		num_patches = tiff_patching.save_patches(filename,
			config.image_data_folder_path,
			label,
			sample_patches,
			mask_patches,
			keep_percentage,
			patch_size,
			overlap,
			sample_size)
		print('Created ' + str(num_patches) + ' patches from ' + filename)

	if (FLAGS.seg_path) and (os.path.exists(os.path.join(FLAGS.seg_path,"seg_masks","seg_" + filename))):
		seg_path = os.path.join(FLAGS.seg_path, 'seg_masks', 'seg_' + filename)
		image_list = os.listdir(patch_folder_path)
		img_dict = {}
		img_dict = get_patch_coords(img_dict, image_list)
		os.makedirs(os.path.join(FLAGS.seg_path, label), exist_ok = True)
		create_segmentation(seg_path, seg_patch_folder_path, img_dict, patch_folder_path, config)

def write_patch_binary_file(filename, label, config):
	patch_folder_path = os.path.join(os.path.abspath(config.image_data_folder_path), label, str(config.patch_size) + "_pixel_patches", os.path.splitext(filename)[0] +'_patches')

	bin_name = os.path.splitext(filename)[0] + ".bin"
	bin_path = os.path.join(patch_folder_path, bin_name)
	
	if os.path.exists(bin_path):
		return
	cprint()
	
	if label == "recurrence" :
		sequence_overlap_percentage = FLAGS.r_overlap / 100
	else:
		sequence_overlap_percentage = FLAGS.nr_overlap / 100

	num_steps = config.num_steps
	bin_file = open(bin_path, "ab+")

	IMAGE_HEIGHT = config.image_height
	IMAGE_WIDTH = config.image_width
	IMAGE_DEPTH = config.image_depth
	
	image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH

	write_array = np.zeros([num_steps, image_bytes], dtype=np.uint8)
	patch_coord_array = np.zeros([num_steps, 2], dtype=np.uint32)
	write_stride = int(math.floor(num_steps * (1-sequence_overlap_percentage)))
	
	image_to_ID_csv_file = open(os.path.join(config.image_data_folder_path,"image_to_subject_ID.csv"),"r")
	reader = csv.reader(image_to_ID_csv_file, delimiter=",")
	_ = next(reader) # discard header line
	image_to_ID_dict = dict()
	for line in reader:
		image_to_ID_dict[line[0].split(".")[0]+"_patches"] = line[1]

	patient_ID_byte_string = str.encode(image_to_ID_dict[filename])
	padded_subject_file_name = "{:<100}".format(image[:-8])
	image_base_name = str.encode(padded_subject_file_name)
	img_dict = {}
	img_dict = get_patch_coords(img_dict, patch_image_list)

	for x_key in sorted(img_dict.keys()):
		for y_key in sorted(img_dict[x_key].keys()):
			filename = img_dict[x_key][y_key]
			counter +=1
			img = io.imread(os.path.join(patch_folder_path,filename))
			arr = img.flatten()

			if(counter <= num_steps):
				write_array[counter-1,:] = arr
				patch_coord_array[counter-1, 0] = x_key
				patch_coord_array[counter-1, 1] = y_key
			else: # Shift all images in the write_array to the left one step, then replace the last value with the new image data
				write_array = np.roll(write_array, -1, axis=0)
				patch_coord_array = np.roll(patch_coord_array, -1, axis=0)

				write_array[num_steps - 1,:] = arr
				patch_coord_array[num_steps-1, 0] = x_key
				patch_coord_array[num_steps-1, 1] = y_key


			write_count = counter - num_steps

			# After the a new portion of the sequence has been added, add the write_array to the binary file
			if(write_count >= 0) & (write_count % write_stride == 0):
				writing = np.reshape(write_array, (num_steps, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
				coord_str = create_string_from_coord_array(patch_coord_array, config.num_steps)
				bin_file.write(patient_ID_byte_string)
				bin_file.write(image_base_name)
				bin_file.write(str.encode(coord_str))
				bin_file.write(writing)
	bin_file.close()

def remove_exisiting_binary_file_then_create_new(binary_file_path):
	# Image bytes are appended to the binary file, so if a binary file exists at the start then it must be removed
	# After (potentially) deleting file, open a new appendable binary file with read/write capabilities
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

def get_patch_coords(img_dict, image_list):
	for image_name in image_list:
			if image_name.endswith(".tif"):
				image_name_noex = os.path.splitext(image_name)[0]
				image_name_chop = image_name_noex.split("_")
				x_coord = int(image_name_chop[len(image_name_chop)-2])
				y_coord = int(image_name_chop[len(image_name_chop)-1])
				img_dict[x_coord] = img_dict.get(x_coord, {})
				img_dict[x_coord][y_coord] = image_name	
	return img_dict

def gauss_sampling(image_to_ID_dict, images_list, bin_file, config):
	gauss_config = gauss.OriginalPatchConfig()
	gauss_config.maximum_seq_per_tile = FLAGS.gauss_seq
	gauss_config.maximum_std_dev = FLAGS.gauss_stdev
	gauss_category_folder = "std_dev" + str(gauss_config.maximum_std_dev) + "_seq" + str(gauss_config.maximum_seq_per_tile)
	gauss_folder = os.path.join(config.image_data_folder_path,'gaussian_patches', gauss_category_folder)
	os.makedirs(gauss_folder, exist_ok=True)
	# remove_characters = -8
	for image in images_list:
		if not image:
			continue
		image_bin_name = image[:-8] + ".bin"
		image_bin_path = os.path.join(gauss_folder, image_bin_name)
		if not os.path.exists(image_bin_path):
			cprint('Creating image binary file for ' + image[:-8], 'white', 'on_green')
			mask_name = 'mask_' + image[:-8] + '.tif'
			mask_path = os.path.join(config.image_data_folder_path,'masks', mask_name)
			sequences, _ = gauss.generate_sequences(mask_path, gauss_config)
			image_bin = open(image_bin_path, 'wb+')
			image_tiff_name = image[:-8] + '.tif'
			image_patch_name = image[:-8] + '_patches'
			cprint("Writing binary file...", 'green', 'on_white')
			gauss.write_image_bin(image_bin, image_tiff_name, image_to_ID_dict[image_patch_name], sequences, gauss_config)
			image_bin.close()

		cprint("Appending " + image[:-8], 'green')
		image_bin = open(image_bin_path, 'rb+')
		image_bytes = image_bin.read(os.path.getsize(image_bin_path))
		bin_file.write(image_bytes)
		image_bin.close()

def create_binary_file(label, mode, config, cond_path=None):
	bin_name = label + "_" + mode + ".bin"
	
	if FLAGS.randomized_conditions:
		bin_path = os.path.join(FLAGS.condition_path, cond_path, label, bin_name)
	else:
		bin_path = os.path.join(os.path.abspath(config.image_data_folder_path), label, bin_name)

	if label == "recurrence" :
		sequence_overlap_percentage = FLAGS.r_overlap / 100
	else:
		sequence_overlap_percentage = FLAGS.nr_overlap / 100

	num_steps = config.num_steps
	cprint("*********" + label + " " + mode + "*************", 'magenta', 'on_white')
	bin_file = remove_exisiting_binary_file_then_create_new(bin_path)
	# bin_file = open(label + "_" + mode + ".bin", "ab+")

	IMAGE_HEIGHT = config.image_height
	IMAGE_WIDTH = config.image_width
	IMAGE_DEPTH = config.image_depth
	
	image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH

	write_array = np.zeros([num_steps, image_bytes], dtype=np.uint8)
	patch_coord_array = np.zeros([num_steps, 2], dtype=np.uint32)
	write_stride = int(math.floor(num_steps * (1-sequence_overlap_percentage)))

	dir_counter = 0
	images_filename = label + "_" + mode + "_subjects.txt"
	
	# walk over all files in starting directory and sub-directories
	if FLAGS.randomized_conditions:
		images_file = open(os.path.join(FLAGS.condition_path, cond_path, images_filename))
	else:
		images_file = open(os.path.join(config.image_data_folder_path, "per_mode_subjects", images_filename), "r")
	images_list = images_file.read().splitlines()
	image_to_ID_csv_file = open(os.path.join(config.image_data_folder_path,"image_to_subject_ID.csv"),"r")
	reader = csv.reader(image_to_ID_csv_file, delimiter=",")
	_ = next(reader) # discard header line
	image_to_ID_dict = dict()
	for line in reader:
		image_to_ID_dict[line[0].split(".")[0]+"_patches"] = line[1]

	if FLAGS.sampling_method == 'gauss':
		gauss_sampling(image_to_ID_dict, images_list, bin_file, config)
	else:
		for image in images_list:
			counter = 0
			if image == "":
				break
			dir_counter +=1
			
			patch_folder_path = os.path.join(os.path.abspath(config.image_data_folder_path), label, str(config.patch_size) + "_pixel_patches", image)

			if FLAGS.seg_path and os.path.exists(os.path.join(FLAGS.seg_path,label,image)):
				patch_folder_path = os.path.join(FLAGS.seg_path,label, config.patch_size + "_pixel_patches", image)
				print("Writing " + image + " -- %i/%i -- FROM SEGMENTATION" % (dir_counter, len(images_list)))
			else:
				print("Writing " + image + " -- %i/%i" % (dir_counter, len(images_list)))
			
			patch_image_list = os.listdir(patch_folder_path)

			patient_ID_byte_string = str.encode(image_to_ID_dict[image])
			padded_subject_file_name = "{:<100}".format(image[:-8])
			image_base_name = str.encode(padded_subject_file_name)
			img_dict = {}
			img_dict = get_patch_coords(img_dict, patch_image_list)

			if FLAGS.sampling_method == 'square':
				square_sampling(img_dict, patch_image_list, label, config)
			elif FLAGS.sampling_method == 'row':
				img_dict = swap_x_and_y_coords(img_dict)

				for y_key in sorted(img_dict.keys()):
					for x_key in sorted(img_dict[y_key].keys()):
						filename = img_dict[y_key][x_key]
						counter +=1
						img = io.imread(os.path.join(patch_folder_path,filename))
						arr = img.flatten()

						if(counter <= num_steps):
							write_array[counter-1,:] = arr
							patch_coord_array[counter-1, 0] = x_key
							patch_coord_array[counter-1, 1] = y_key
						else: # Shift all images in the write_array to the left one step, then replace the last value with the new image data
							write_array = np.roll(write_array, -1, axis=0)
							patch_coord_array = np.roll(patch_coord_array, -1, axis=0)

							write_array[num_steps - 1,:] = arr
							patch_coord_array[num_steps-1, 0] = x_key
							patch_coord_array[num_steps-1, 1] = y_key

						write_count = counter - num_steps

						# After the a new portion of the sequence has been added, add the write_array to the binary file
						if(write_count >= 0) & (write_count % write_stride == 0):
							writing = np.reshape(write_array, (num_steps, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
							coord_str = create_string_from_coord_array(patch_coord_array, config.num_steps)
							bin_file.write(patient_ID_byte_string)
							bin_file.write(image_base_name)
							bin_file.write(str.encode(coord_str))
							bin_file.write(writing)	
			elif FLAGS.sampling_method == 'column':
				for x_key in sorted(img_dict.keys()):
					for y_key in sorted(img_dict[x_key].keys()):
						filename = img_dict[x_key][y_key]
						counter +=1
						img = io.imread(os.path.join(patch_folder_path,filename))
						arr = img.flatten()

						if(counter <= num_steps):
							write_array[counter-1,:] = arr
							patch_coord_array[counter-1, 0] = x_key
							patch_coord_array[counter-1, 1] = y_key
						else: # Shift all images in the write_array to the left one step, then replace the last value with the new image data
							write_array = np.roll(write_array, -1, axis=0)
							patch_coord_array = np.roll(patch_coord_array, -1, axis=0)

							write_array[num_steps - 1,:] = arr
							patch_coord_array[num_steps-1, 0] = x_key
							patch_coord_array[num_steps-1, 1] = y_key


						write_count = counter - num_steps

						# After the a new portion of the sequence has been added, add the write_array to the binary file
						if(write_count >= 0) & (write_count % write_stride == 0):
							writing = np.reshape(write_array, (num_steps, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
							coord_str = create_string_from_coord_array(patch_coord_array, config.num_steps)
							bin_file.write(patient_ID_byte_string)
							bin_file.write(image_base_name)
							bin_file.write(str.encode(coord_str))
							bin_file.write(writing)
	bin_file.close()

def create_binary_mode_files(label, config, cond_path=None):
	if FLAGS.randomized_conditions:
		os.makedirs(os.path.join(FLAGS.condition_path, cond_path, label), exist_ok=True)
		create_binary_file(label, "train", config, cond_path=condition_folder)
		create_binary_file(label, "valid", config, cond_path=condition_folder)
		create_binary_file(label, "test", config, cond_path=condition_folder)
	else:
		if not FLAGS.mode:
			create_binary_file(label, "train", config)
			create_binary_file(label, "valid", config)
			create_binary_file(label, "test", config)
		else:
			create_binary_file(label, FLAGS.mode, config)

def swap_x_and_y_coords(img_dict):
	swapped_coords = dict()
	for x_key in sorted(img_dict.keys()):
		for y_key in sorted(img_dict[x_key].keys()):
			if y_key not in swapped_coords:
				swapped_coords[y_key] = dict()
			swapped_coords[y_key][x_key] = img_dict[x_key][y_key]
	return swapped_coords

def create_string_from_coord_array(coord_array, num_steps):
	coord_string = ""
	for y in np.arange(num_steps):
		for x in np.arange(2):
			coord = "{:<6}".format(coord_array[y][x])
			coord_string = coord_string + coord
	return coord_string

def get_config():
	if FLAGS.config == "300":
		return SmallConfig()
	else:
		return OriginalPatchConfig()

if __name__ == '__main__':

	if not FLAGS.num_steps:
		raise ValueError("Must set --num_steps to integer for number of steps in RNN sequence")

	# if not FLAGS.num_steps:
	# 	raise ValueError("must set --num_steps to the length of RNN sequence")

	#directory = os.fsencode(FLAGS.data_dir)
	# num_steps = FLAGS.num_steps

	if FLAGS.condition_path:
		FLAGS.randomized_conditions = True
	config = get_config()
	og_images_directory = os.path.join(config.image_data_folder_path, "original_images")

	image_to_ID_csv_file = open(os.path.join(config.image_data_folder_path,"image_to_subject_ID.csv"),"r")
	reader = csv.DictReader(image_to_ID_csv_file, delimiter=",")
	if not FLAGS.sampling_method == 'gauss':
		for row in reader:
			if int(row["label"]) == 1:
				create_patch_folder(row["image_name"], "recurrence", config)
				# write_patch_binary_file(row["image_name"], "recurrence", config)
			elif int(row["label"]) == 0:
				create_patch_folder(row["image_name"], "nonrecurrence", config)
				# write_patch_binary_file(row["image_name"], "nonrecurrence", config)

	if not FLAGS.patches_only:
		if FLAGS.randomized_conditions:
			if not FLAGS.condition_path:
				raise ValueError("If creating binary files for with randomized subjects, must specify the condition's enclosing folder with --condition_path.")
			for condition_folder in sorted(os.listdir(FLAGS.condition_path)):
				if "condition" not in condition_folder:
					break
				else:
					cprint("===============" + condition_folder + "===============", 'white', 'on_magenta')
					create_binary_mode_files("recurrence", config, cond_path=condition_folder)
					create_binary_mode_files("nonrecurrence", config, cond_path=condition_folder)
		else:
			if not FLAGS.nonrecurrence_only:
				create_binary_mode_files("recurrence", config)
			if not FLAGS.recurrence_only:
				create_binary_mode_files("nonrecurrence", config)

sys.exit(0)