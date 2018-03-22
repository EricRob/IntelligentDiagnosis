#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import os
import csv
import pdb
import numpy as np
from termcolor import cprint
from skimage import io
from tensorflow import flags

flags.DEFINE_string('image_list', 'large_images.csv', 'File containing list of large images to be split' )
flags.DEFINE_string('base_path', "/home/wanglab/Desktop/recurrence_seq_lstm/image_data", "")
flags.DEFINE_string('original_images_folder', 'original_images', 'Folder within base_path containing the original images')

# Global variables
FLAGS = flags.FLAGS

# Class declarations

# Function declarations
def split_image_in_half(img, large_image_name):
	y_center = img.shape[0] // 2

	top = img[0:y_center,:,:]
	bottom = img[y_center:,:,:]

	new_filename = os.path.join(FLAGS.base_path, FLAGS.original_images_folder,large_image_name.split('.')[0])
	cprint("Saving top...", 'green')
	io.imsave(new_filename + "_B1.tif", top)
	del top
	cprint("Saving bottom...", 'green')
	io.imsave(new_filename + "_B2.tif", bottom)
	del bottom

def split_image_into_fourths(img, large_image_name):
	x_shape = img.shape[1]
	y_shape = img.shape[0]
	x_center = x_shape // 2
	y_center = y_shape // 2
	nw = img[0:y_center, 0:x_center, :]
	sw = img[y_center:, 0:x_center, :]
	ne = img[0:y_center, x_center:, :]
	se = img[y_center:, x_center:, :]
	del img
	new_filename = os.path.join(FLAGS.base_path, FLAGS.original_images_folder,large_image_name.split('.')[0])
	cprint("Saving NW...", 'green')
	io.imsave(new_filename + "_Q1.tif", nw )
	del nw
	cprint("Saving SW...", 'green')
	io.imsave(new_filename + "_Q3.tif", sw )
	del sw
	cprint("Saving NE...", 'green')
	io.imsave(new_filename + "_Q3.tif", ne )
	del ne
	cprint("Saving SE...", 'green')
	io.imsave(new_filename + "_Q4.tif", se )

def split_image_into_ninths(img, large_image_name):
	new_filename = os.path.join(FLAGS.base_path, FLAGS.original_images_folder,large_image_name.split('.')[0])
	x_shape = img.shape[1]
	y_shape = img.shape[0]
	x_left = x_shape // 3
	x_right = x_left * 2
	y_top = y_shape // 3
	y_bottom = y_top * 2

	one = img[0:y_top, 0:x_left, :]
	two = img[0:y_top, x_left:x_right, :]
	three = img[0:y_top, x_right:, :]
	cprint('Saving region 1...', 'green')
	io.imsave(new_filename + "_N1.tif", one)
	del one

	cprint('Saving region 2...', 'green')
	io.imsave(new_filename + "_N2.tif", two)
	del two

	cprint('Saving region 3...', 'green')
	io.imsave(new_filename + "_N3.tif", three)
	del three

	four = img[y_top:y_bottom, 0:x_left, :]
	five = img[y_top:y_bottom, x_left:x_right, :]
	six = img[y_top:y_bottom, x_right:, :]
	cprint('Saving region 4...', 'green')
	io.imsave(new_filename + "_N4.tif", four)
	del four

	cprint('Saving region 5...', 'green')
	io.imsave(new_filename + "_N5.tif", five)
	del five

	cprint('Saving region 6...', 'green')
	io.imsave(new_filename + "_N6.tif", six)
	del six

	seven = img[y_bottom:, 0:x_left, :]
	eight = img[y_bottom:, x_left:x_right, :]
	nine = img[y_bottom:, x_right:, :]
	cprint('Saving region 7...', 'green')
	io.imsave(new_filename + "_N7.tif", seven)
	del seven

	cprint('Saving region 8...', 'green')
	io.imsave(new_filename + "_N8.tif", eight)
	del eight

	cprint('Saving region 9...', 'green')
	io.imsave(new_filename + "_N9.tif", nine)
	del nine

def main():
	with open(os.path.join(FLAGS.base_path, FLAGS.image_list), newline="") as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			large_image_name = line[0]
			img_path = os.path.join(FLAGS.base_path, FLAGS.original_images_folder,large_image_name)
			if not os.path.exists(img_path):
				continue
			cprint("Loading " + large_image_name + " into memory", 'yellow')
			img = io.imread(img_path)
			size = os.path.getsize(img_path)
			if size < 8000000000:
				cprint("Splitting " + large_image_name + 'into half', 'yellow')
				split_image_in_half(img, large_image_name)
			elif size < 16000000000:
				cprint("Splitting " + large_image_name + 'into fourths', 'yellow')
				split_image_into_fourths(img, large_image_name)
			elif size < 36000000000:
				cprint("Splitting " + large_image_name + ' into ninths', 'yellow')
				split_image_into_ninths(img, large_image_name)
			else:
				cprint(large_image_name + 'is extremely large, and was skipped. I assumed this would never happen', 'red')
			del img



# Main body
if __name__ == '__main__':
	main()