#!/usr/bin/python3 python3

import numpy as np
import argparse
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
import subprocess
from shutil import copyfile
import centroid_lstm as gauss
import subject_list_generator

import tiff_patching

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

parser = argparse.ArgumentParser(description='Create binary files for feeding into recurrence_seq_lstm')

## Overall File Creation Flags
parser.add_argument("--recurrence_only", default=False, action='store_true',
	help="Generate only recurrence binary file from lists in the config.image_data_folder_path/per_mode_subjects directory")
parser.add_argument("--nonrecurrence_only", default=False, action='store_true',
	help="Generate only nonrecurrence binary file from lists in the config.image_data_folder_path/per_mode_subjects directory")
# Example: Create only nonrecurrence data
# python3 preprocess_lstm.py --nonrecurrence_only
# This does not work with the condition_path parameter. If condition_path is specified, this flag is ignored.

parser.add_argument("--seg_path", default=None, type=str,
	help="Segmentations are custom-made binary masks for sampling from specific areas within the image. \
	This parameter specifies the directory containing binary segmentations")

parser.add_argument("--mode", default=None, type=str,
	help="Create the condition binary file for a specific mode, either train, validation, or test")

parser.add_argument("--condition_path", default=None, type=str,
	help="Path of directory with condition folders containing lists of images. This should be a \
	complete file path.")

parser.add_argument("--generate_conditions", default=False, action='store_true',
	help="Generate randomized lists of subjects for cross-validation testing. Succesfully using this \
	parameter also requires specifying verified_list and new_condition.")

parser.add_argument("--verified_list",  default="YALE_CUMC.csv", type=str,
	help="List of subjects used for creating new cross-validation conditions.")

parser.add_argument("--new_condition", default='YALE_CUMC', type=str,
	help="Name for the new cross-validation condition to be created.")

parser.add_argument("--cross_val_folds", default=6, type=int,
	help="Number of folds for the new cross-validation condition.")

parser.add_argument("--data_conditions",default="/data/recurrence_seq_lstm/data_conditions", type=str,
	help="Location of data_conditions directory, aka parent directory for the new cross-validation \
	condition folder." )

parser.add_argument("--config", default='original', type=str,
	help="Configuration for generating patches. Configurations can be added as needed, but by default \
	there are only two: the -original- config previously used for all testing and training and -300- \
	used for smaller-patch testing.")

parser.add_argument("--sampling_method", default="gauss",  type=str,
	help="Sampling pattern for generating sequences. Three methods \
	currently exist: gauss, column, and row")

parser.add_argument("--check_new_masks", default=False, action='store_true',
	help="Check for new masks in the new_image_dir directory that need to be created. This allows \
	a method to run that triggers the batch-mask gimp script.")

parser.add_argument("--new_image_dir", default='/home/wanglab/Desktop/new_masks/', type=str,
	help="location of new images to be masked and moved to image_data")

parser.add_argument("--original_images_dir", default='/data/recurrence_seq_lstm/image_data/original_images', type=str,
	help='location of original_images directory within image_data')

parser.add_argument("--masks_dir", default='/data/recurrence_seq_lstm/image_data/masks', type=str,
	help='location of masks directory within image_data' )

parser.add_argument("--preverify", default=False, action='store_true',
	help="Check image folders for verified_list images and masks")

parser.add_argument('--overwrite', default=False, action='store_true',
	help='Overwrite existing binary files')

parser.add_argument('--no_write', default=False, action='store_true',
	help='Skip writing binary files (used for patch data creation)')

# Gauss Configuration Flags
parser.add_argument("--gauss_seq", default=None, type=int,
	help="Number of sequences to generate per tile with gaussian sampling")

parser.add_argument("--gauss_stdev", default=None, type=int,
	help="Standard deviation of pixel distance from center for gaussian sampling")

parser.add_argument("--gauss_tile_size", default=None, type=int,
	help="Tile dimensions for splitting sample image for gauss distribution")

parser.add_argument('--min_patch', default=None,
	help='Minimum number of detections in patch area (if not specified, uses config default)')

parser.add_argument('--patch_thresh', default=None ,
	help='Threshold for \"Other\" detection percentage in a patch area (if not specified, uses config default)')

parser.add_argument('--delaunay_radius', default=40, type=int,
	help='The desired pixel radius of delaunay triangulation. This parameter is only used for naming \
	the directory that will hold binary files for qupath output of this radius.')

parser.add_argument('--detections_path', default='/data/recurrence_seq_lstm/qupath_output/', type=str,
	help='Directory of qupath cell detection information used for creating binary files.')

parser.add_argument('--remove_features', default=False, action='store_true',
	help='Remove features from writing image binaries (features still used for restricting sampling).\
	This parameter was created for testing the accuracy of (features vs. no features), and is mostly \
	ignored since using features was significantly more successful.')



### Column Sampling Flags (Deprecated)
parser.add_argument("--r_overlap", default=75, type=int,
	help="Percentage of overlap of recurrence samples")
parser.add_argument("--nr_overlap", default=50, type=int,
	help="Percentage of overlap of nonrecurrence samples")
parser.add_argument("--num_steps", default=20, type=int,
	help="Number of steps in RNN sequence")
parser.add_argument("--patches_only",default=False, action='store_true',
	help="Generate only binary files for new images, and not for conditions or network testing.")

# FLAGS = flags.FLAGS
FLAGS = parser.parse_args()
FILLER = '                                            '
SKIP_LIST = []


class OriginalPatchConfig(object):
	image_data_folder_path = '/data/recurrence_seq_lstm/image_data/' # Location of images folder
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
	add_features = not FLAGS.remove_features

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

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
		image_to_ID_dict[line[0].split(".")[0]] = line[1]

	patient_ID_byte_string = str.encode(image_to_ID_dict[filename])
	padded_subject_file_name = "{:<100}".format(image)
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

def gauss_sampling(image_to_ID_dict, images_list, bin_file, mode, config):
	if FLAGS.config == 'yale':
		if mode == 'test':
			gauss_config = gauss.YaleConfig_test()
		else:
			gauss_config = gauss.YaleConfig_train()
		cprint('Using Yale configuration', 'blue', 'on_white')
	elif FLAGS.config == 'yale_small':
		if mode == 'test':
			gauss_config = gauss.YaleConfig_test_small()
		else:
			gauss_config = gauss.YaleConfig_train_small()
		cprint('Using Yale SMALL configuration', 'blue', 'on_red')
	elif FLAGS.config == 'sinai':
		if mode == 'test':
			gauss_config = gauss.SinaiConfig_test()
		else:
			gauss_config = gauss.SinaiConfig_train()
		cprint('Using Sinai configuration', 'magenta', 'on_white')
	elif FLAGS.config == 'small':
		if mode == 'test':
			gauss_config = gauss.OriginalPatchConfig_test_small()
		else:
			gauss_config = gauss.OriginalPatchConfig_train_small()	
		cprint('Using Original SMALL configuration', 'green', 'on_red')
	else:
		if mode == 'test':
			gauss_config = gauss.OriginalPatchConfig_test()
		else:
			gauss_config = gauss.OriginalPatchConfig_train()
		cprint('Using Original configuration', 'green', 'on_grey')
	
	if FLAGS.gauss_seq:
		gauss_config.maximum_seq_per_tile = FLAGS.gauss_seq
	if FLAGS.gauss_tile_size:
		gauss_config.tile_size = FLAGS.gauss_tile_size
	if FLAGS.gauss_stdev:
		gauss_config.maximum_std_dev = FLAGS.gauss_stdev
	if FLAGS.min_patch:
		gauss_config.MINIMUM_PATCH_CELLS = FLAGS.min_patch
	if FLAGS.patch_thresh:
		gauss_config.OTHER_PATCH_THRESHOLD = FLAGS.patch_thresh / 100

	print('OTHER_PATCH_THRESHOLD: ' + str(gauss_config.OTHER_PATCH_THRESHOLD))
	print('MINIMUM_PATCH_CELLS: ' + str(gauss_config.MINIMUM_PATCH_CELLS))
	
	if FLAGS.remove_features:
		prepend = "NoFEATURES_"
	else:
		prepend = "FEATURES_"

	gauss_config.add_features = not FLAGS.remove_features
	gauss_category_folder = prepend + "tile" + str(gauss_config.tile_size) + "_std_dev" + str(gauss_config.maximum_std_dev) + "_seq" + str(gauss_config.maximum_seq_per_tile) + '_patch' + str(gauss_config.patch_size)
	gauss_folder = os.path.join(config.image_data_folder_path,'feature_patches','OTHER_gaussian_patches_' + str(gauss_config.pixel_radius) + '_min' + str(gauss_config.MINIMUM_PATCH_CELLS) + '_' + str(gauss_config.large_cluster) + '_' + str(int(gauss_config.OTHER_PATCH_THRESHOLD*100)) + 'p', gauss_category_folder)
	os.makedirs(gauss_folder, exist_ok=True)
	# remove_characters = -8
	for image in images_list:
		image = os.path.splitext(image)[0]
		csv_path = os.path.join(gauss_config.features_path, str(gauss_config.patch_size) + 'patch_histogram_values.csv')
		header = ['subject','image', 'patch_size', 'x', 'y','tile','tumor_count','imm_count','other_count', 'total_count']
		if not os.path.exists(csv_path):
			with open(csv_path, 'w') as csvfile:
				writer = csv.writer(csvfile)
				writer.writerow(header)
		if not image:
			continue
		if image in SKIP_LIST:
			cprint('Skipping ' + image + ', already skipped in current run', 'red')
			continue
		image_bin_name = image + ".bin"
		image_bin_path = os.path.join(gauss_folder, image_bin_name)
		#pdb.set_trace()
		if not os.path.exists(image_bin_path) or FLAGS.overwrite or FLAGS.no_write:
			detections_filename = os.path.join(FLAGS.detections_path, image + '_Detectionstxt.txt')
			if not os.path.exists(detections_filename):
				cprint('No detections exist for ' + image + ', skipping due to lack of detections file', 'red')
				SKIP_LIST.append(image)
				continue
			if image not in image_to_ID_dict:
				cprint(image + ' not in image dictionary, skipping', 'red')
				pdb.set_trace()	
				continue
			cprint('Creating image binary file for ' + image, 'white', 'on_green')
			mask_name = 'mask_' + image + '.tif'
			mask_path = os.path.join(config.image_data_folder_path,'masks', mask_name)
			img_info = (image_to_ID_dict[image], image)
			if FLAGS.no_write:
				seq_features = gauss.generate_sequences(mask_path, gauss_config, image, image_to_ID_dict[image], detections=FLAGS.detections_path, image_info=img_info)
			else:
				seq_features = gauss.generate_sequences(mask_path, gauss_config, image, image_to_ID_dict[image], detections=FLAGS.detections_path)
			if FLAGS.no_write:
				continue
			if not seq_features:
				cprint('No sequences processed, skipping due to lack of features', 'red')
				SKIP_LIST.append(image)
				continue
			gauss.regional_verification(seq_features, gauss_config, image, image_to_ID_dict[image])
			image_bin = open(image_bin_path, 'wb+')
			image_tiff_name = image + '.tif'
			# cprint("Writing binary file...", 'green', 'on_white')
			gauss.write_image_bin(image_bin, image_tiff_name, image_to_ID_dict[image], seq_features, gauss_config)
			image_bin.close()

		cprint("Appending " + image + FILLER, 'green', end="\r")
		image_bin = open(image_bin_path, 'rb+')
		image_bytes = image_bin.read(os.path.getsize(image_bin_path))
		bin_file.write(image_bytes)
		image_bin.close()
	with open('/data/recurrence_seq_lstm/need_qupath.csv', 'a') as csvfile:
				writer = csv.writer(csvfile)
				for image in SKIP_LIST:
					writer.writerow([os.path.join(config.image_data_folder_path,'original_images',image +'.tif')])

def create_binary_file(label, mode, config, cond_path=None):
	bin_name = label + "_" + mode + ".bin"
	
	if cond_path:
		bin_path = os.path.join(FLAGS.condition_path, cond_path, label, bin_name)
	else:
		bin_path = os.path.join(os.path.abspath(config.image_data_folder_path), label, bin_name)

	cprint("*********" + label + " " + mode + "*************", 'magenta', 'on_white')
	bin_file = remove_exisiting_binary_file_then_create_new(bin_path)
	
	images_filename = label + "_" + mode + "_subjects.txt"
	
	# walk over all files in starting directory and sub-directories
	if cond_path:
		images_file = open(os.path.join(FLAGS.condition_path, cond_path, images_filename))
	else:
		images_file = open(os.path.join(config.image_data_folder_path, "per_mode_subjects", images_filename), "r")
	images_list = images_file.read().splitlines()
	images_file.close()
	with open(os.path.join(config.image_data_folder_path,"image_to_subject_ID.csv"),"r") as image_to_ID_csv_file:
		reader = csv.reader(image_to_ID_csv_file, delimiter=",")
		_ = next(reader) # discard header line
		image_to_ID_dict = dict()
		for line in reader:
			image_to_ID_dict[line[0].split(".")[0]] = line[1]

	#                                                          #
	# ********** Default behavior is gauss sampling ********** #
	#                                                          #
	if FLAGS.sampling_method == 'gauss':
		gauss_sampling(image_to_ID_dict, images_list, bin_file, mode, config)


	#                                                                                           #
	# ********** Set sampling method to 'column' or 'row' for the following sampling ********** #
	#                                                                                           #
	else:

		if label == "recurrence" :
			sequence_overlap_percentage = FLAGS.r_overlap / 100
		else:
			sequence_overlap_percentage = FLAGS.nr_overlap / 100
		dir_counter = 0
		num_steps = config.num_steps

		IMAGE_HEIGHT = config.image_height
		IMAGE_WIDTH = config.image_width
		IMAGE_DEPTH = config.image_depth
		image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH

		write_array = np.zeros([num_steps, image_bytes], dtype=np.uint8)
		patch_coord_array = np.zeros([num_steps, 2], dtype=np.uint32)
		write_stride = int(math.floor(num_steps * (1-sequence_overlap_percentage)))
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
			padded_ID_string = "{:<5}".format(image_to_ID_dict[image])
			patient_ID_byte_string = str.encode(padded_ID_string)
			padded_subject_file_name = "{:<100}".format(image)
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
	# Wrap it up!
	bin_file.close()

def create_binary_mode_files(label, config, cond_path=None):
	if cond_path:
		os.makedirs(os.path.join(FLAGS.condition_path, cond_path, label), exist_ok=True)
		create_binary_file(label, "train", config, cond_path=cond_path)
		create_binary_file(label, "valid", config, cond_path=cond_path)
		create_binary_file(label, "test", config, cond_path=cond_path)
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

def generate_new_masks():
	#
	# This is a hacky method that sort of works. There must be a new_image_dir containing images to be masked.
	# Binary masks are created, then moved to the mask directory, and the original images are moved to their
	# proper place. In general, it's safer to create masks separately using batch-mask before running preprocess_lstm
	#

	wd = os.getcwd()
	os.chdir(FLAGS.new_image_dir)
	subprocess.check_call("gimp -i -b '(batch-mask \"*.tif\" 217)' -b '(gimp-quit 0)'", shell=True)
	subprocess.check_call("mv mask_* " + FLAGS.masks_dir, shell=True)
	subprocess.check_call("mv *.tif " + FLAGS.original_images_dir, shell=True)
	os.chdir(wd)

def generate_new_conditions():
	#
	# You really do not want to overwrite an existing set of conditions, so this check is in place.
	#

	if not query_yes_no("Do you want to generate new testing conditions?", default="no"):
		return False
	new_condition_name = FLAGS.new_condition
	verified_csv = FLAGS.verified_list
	if not new_condition_name:
		new_condition_name = os.path.splitext(verified_csv)[0]
		FLAGS.new_condition = new_condition_name
	c_val_folds = FLAGS.cross_val_folds
	base_path = FLAGS.data_conditions

	subject_list_generator.write_lists(new_condition_name, verified_csv, c_val_folds, base_path)
	return True

def pre_verify_subjects():
	
	#
	# This is a time saving method that needs to be updated for usage with FLAGS.condition_path
	#
	
	verified_path = os.path.join(FLAGS.data_conditions, FLAGS.verified_list)
	verified_file = open(verified_path, "r")
	missing_mask_list = []
	missing_image_list = []
	exit_bool = False
	reader = csv.DictReader(verified_file, delimiter = ",")
	for row in reader:
		image_base = row['image']
		row_mask = "mask_" + image_base + ".tif"
		row_image = image_base + ".tif"
		if not os.path.exists(os.path.join(FLAGS.original_images_dir, row_image)):
			missing_image_list = missing_image_list + [row_image]
		if not os.path.exists(os.path.join(FLAGS.masks_dir, row_mask)):
			missing_mask_list = missing_mask_list + [row_mask]
	if missing_image_list:
		exit_bool = True
		cprint("\nUnable to create binary files due to missing images:", 'white', 'on_red')
		for image in sorted(missing_image_list):
			cprint(image)
	print("")
	if missing_mask_list:
		if not missing_image_list and not FLAGS.check_new_masks:
			exit_bool = True
		cprint("Missing masks:", 'white', 'on_red')
		for mask in sorted(missing_mask_list):
			cprint(mask)
	return exit_bool

def get_config():
	if FLAGS.config == "300":
		return SmallConfig()
	else:
		return OriginalPatchConfig()

if __name__ == '__main__':

	if FLAGS.check_new_masks:
		print("Generating new image masks ... ")
		generate_new_masks()

	if FLAGS.preverify:
		exit = pre_verify_subjects()
		if exit:
			sys.exit(1)
		else:
			cprint("Verification Successful!", 'white', 'on_green')

	if FLAGS.generate_conditions:
		print("Generating new cross-validation conditions ... ")
		exit = generate_new_conditions()
		if not exit:
			cprint('Check default flag settings before proceeding', 'red')
			sys.exit(1)
		FLAGS.condition_path = os.path.join(FLAGS.data_conditions, FLAGS.new_condition)

	config = get_config()


	#
	# Individual image binary preparation for deprecated sampling method, only used if reverting
	# back to row or column sampling.
	#
	if FLAGS.sampling_method == 'row' or FLAGS.sampling_method == 'column':
		image_to_ID_csv_file = open(os.path.join(config.image_data_folder_path,"image_to_subject_ID.csv"),"r")
		reader = csv.DictReader(image_to_ID_csv_file, delimiter=",")
		for row in reader:
			if int(row["label"]) == 1:
				create_patch_folder(row["image_name"], "recurrence", config)
			elif int(row["label"]) == 0:
				create_patch_folder(row["image_name"], "nonrecurrence", config)
	#
	#

	if not FLAGS.patches_only:
		if FLAGS.condition_path:
			cprint("--> Generating condition binary files", 'white', 'on_green')
			for condition_folder in sorted(os.listdir(FLAGS.condition_path)):
				if "condition" not in condition_folder:
					continue
				else:
					cprint("===============" + condition_folder + "===============", 'white', 'on_magenta')
					create_binary_mode_files("recurrence", config, cond_path=condition_folder)
					create_binary_mode_files("nonrecurrence", config, cond_path=condition_folder)
		else:
			if FLAGS.recurrence_only:
				create_binary_mode_files("recurrence", config)
			elif FLAGS.nonrecurrence_only:
				create_binary_mode_files("nonrecurrence", config)
			else:
				create_binary_mode_files("recurrence", config)
				create_binary_mode_files("nonrecurrence", config)

sys.exit(0)