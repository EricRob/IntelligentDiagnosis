#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import argparse
import os
import csv
from termcolor import cprint
import skimage.external.tifffile as tiff
import subprocess

parser = argparse.ArgumentParser(description='Split images larger than 4GB into smaller tiles for QuPath processing and GIMP masking.')

parser.add_argument("--base_path", default="/data/recurrence_seq_lstm/image_data/original_images", type=str,
	help="[FULL FILE PATH] Path of images to be split. Also the path for the --image_list.")
parser.add_argument("--image_list", default='large_images.csv', type=str,
	help="File containing list of large images to be split")
parser.add_argument("--dst", default=None, type=str,
	help="[FULL FILE PATH] Destination folder for split images")
parser.add_argument('--destructive', default=False, action='store_true',
	help='Delete images after splitting to free disk space')
parser.add_argument('--force_split', default=False, action='store_true',
	help="Force images to be split (only into halves)")

# Global variables
FLAGS = parser.parse_args()
if not FLAGS.dst:
	print("No destination specified, using --base_path directory")
	FLAGS.dst = FLAGS.base_path

# Class declarations

# Function declarations
def split_image_in_half(img, large_image_name):
	cprint("Loading " + large_image_name + " into memory", 'yellow')
	new_filename = os.path.join(FLAGS.dst,large_image_name.split('.tif')[0])

	if os.path.exists(new_filename + "_B1.tif"):
		cprint("Region 1 already exists, skipping " + large_image_name + " entirely")
		return
	
	y_center = img.shape[0] // 2

	top = img[0:y_center,:,:]
	bottom = img[y_center:,:,:]

	cprint("Saving top...", 'green')
	tiff.imsave(new_filename + "_B1.tif", top)

	del top
	cprint("Saving bottom...", 'green')
	tiff.imsave(new_filename + "_B2.tif", bottom)
	del bottom

	del img

	return

def split_image_into_fourths(img, large_image_name):
	cprint("Loading " + large_image_name + " into memory", 'yellow')
	
	new_filename = os.path.join(FLAGS.dst,large_image_name.split('.tif')[0])
	x_shape = img.shape[1]
	y_shape = img.shape[0]
	x_center = x_shape // 2
	y_center = y_shape // 2

	if os.path.exists(new_filename + "_Q1.tif"):
		cprint("Region 1 already exists, skipping Q1")
	else:
		nw = img[0:y_center, 0:x_center, :]
		cprint("Saving NW...", 'green')
		tiff.imsave(new_filename + "_Q1.tif", nw )
		del nw
	
	if os.path.exists(new_filename + "_Q2.tif"):
		cprint("Region 2 already exists, skipping Q2")
	else:	
		sw = img[y_center:, 0:x_center, :]
		cprint("Saving SW...", 'green')
		tiff.imsave(new_filename + "_Q2.tif", sw )
		del sw
	
	if os.path.exists(new_filename + "_Q3.tif"):
		cprint("Region 3 already exists, skipping Q3")
	else:
		ne = img[0:y_center, x_center:, :]
		cprint("Saving NE...", 'green')
		tiff.imsave(new_filename + "_Q3.tif", ne )
		del ne
	
	if os.path.exists(new_filename + "_Q4.tif"):
		cprint("Region 3 already exists, skipping Q4")
	else:
		se = img[y_center:, x_center:, :]
		cprint("Saving SE...", 'green')
		tiff.imsave(new_filename + "_Q4.tif", se )
		del se

	del img

	return

def split_image_into_ninths(img, large_image_name):
	cprint("Loading " + large_image_name + " into memory", 'yellow')
	new_filename = os.path.join(FLAGS.dst,large_image_name.split('.tif')[0])
	if os.path.exists(new_filename + "_N1.tif"):
		cprint("Region 1 already exists, skipping " + large_image_name)
		return
	x_shape = img.shape[1]
	y_shape = img.shape[0]
	x_left = x_shape // 3
	x_right = x_left * 2
	y_top = y_shape // 3
	y_bottom = y_top * 2

	one = img[0:y_top, 0:x_left, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_N1.tif", one)
	cprint(one.size, 'green')

	del one

	two = img[0:y_top, x_left:x_right, :]
	cprint('Saving region 2...', 'green')
	tiff.imsave(new_filename + "_N2.tif", two)
	del two

	three = img[0:y_top, x_right:, :]
	cprint('Saving region 3...', 'green')
	tiff.imsave(new_filename + "_N3.tif", three)
	del three

	four = img[y_top:y_bottom, 0:x_left, :]
	cprint('Saving region 4...', 'green')
	tiff.imsave(new_filename + "_N4.tif", four)
	del four

	five = img[y_top:y_bottom, x_left:x_right, :]
	cprint('Saving region 5...', 'green')
	tiff.imsave(new_filename + "_N5.tif", five)
	del five

	six = img[y_top:y_bottom, x_right:, :]
	cprint('Saving region 6...', 'green')
	tiff.imsave(new_filename + "_N6.tif", six)
	del six

	seven = img[y_bottom:, 0:x_left, :]
	cprint('Saving region 7...', 'green')
	tiff.imsave(new_filename + "_N7.tif", seven)
	del seven

	eight = img[y_bottom:, x_left:x_right, :]
	cprint('Saving region 8...', 'green')
	tiff.imsave(new_filename + "_N8.tif", eight)
	del eight

	nine = img[y_bottom:, x_right:, :]
	cprint('Saving region 9...', 'green')
	tiff.imsave(new_filename + "_N9.tif", nine)
	del nine

	del img

	return

def split_image_into_sixteenths(img, large_image_name):
	cprint("Loading " + large_image_name + " into memory", 'yellow')
	new_filename = os.path.join(FLAGS.dst,large_image_name.split('.tif')[0])
	if os.path.exists(new_filename + "_P1.tif"):
		cprint("Region 1 already exists, skipping " + large_image_name)
		return
	x_shape = img.shape[1]
	y_shape = img.shape[0]
	x_left = x_shape // 4
	x_mid = x_left * 2
	x_right = x_left * 3
	y_top = y_shape // 4
	y_mid = y_top * 2
	y_bottom = y_top * 3

	one = img[0:y_top, 0:x_left, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del one

	two = img[0:y_top, x_left:x_mid, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P2.tif", one)
	cprint(one.size, 'green')

	del two

	three = img[0:y_top, x_mid:x_right, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P3.tif", one)
	cprint(one.size, 'green')

	del three

	four = img[0:y_top, x_right:, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P4.tif", one)
	cprint(one.size, 'green')

	del four

	five = img[y_top:y_mid, 0:x_left, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del five

	six = img[y_top:y_mid, x_left:x_mid, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del six

	seven = img[y_top:y_mid, x_mid:x_right, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del seven

	eight = img[y_top:y_mid, x_right:, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del eight

	nine = img[y_mid:y_bottom, 0:x_left, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del nine

	ten = img[y_mid:y_bottom, x_left:x_mid, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del ten

	eleven = img[y_mid:y_bottom, x_mid:x_right, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del eleven

	twelve = img[y_mid:y_bottom, x_right:, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del twelve

	thirteen = img[y_bottom:, 0:x_left, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del thirteen

	fourteen = img[y_bottom:, x_left:x_mid, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del fourteen

	fifteen = img[y_bottom:, x_mid:x_right, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del fifteen

	sixteen = img[y_bottom:, x_right:, :]
	cprint('Saving region 1...', 'green')
	tiff.imsave(new_filename + "_P1.tif", one)
	cprint(one.size, 'green')

	del sixteen


	del img

	return

def main():
	with open(os.path.join(FLAGS.base_path, FLAGS.image_list), newline="") as csvfile:
		reader = csv.reader(csvfile)
		cprint("MADE IT HERE", 'red')
		for line in reader:
			large_image_name = line[0]
			img_path = os.path.join(FLAGS.base_path,large_image_name)
			cprint(img_path, 'red')
			if not os.path.exists(img_path):
				continue

			img = tiff.imread(img_path)
			size = img.size
			if size < 4000000000 and not FLAGS.force_split:
				cprint(large_image_name + ' is under 4GB, will not split', 'yellow')
			elif size < 4000000000 and FLAGS.force_split:
				cprint("Force splitting " + large_image_name + ' into half', 'yellow')
				split_image_in_half(img, large_image_name)
			elif size < 8000000000:
				cprint("Splitting " + large_image_name + ' into half', 'yellow')
				split_image_in_half(img, large_image_name)
			elif size < 16000000000:
				cprint("Splitting " + large_image_name + ' into fourths', 'yellow')
				split_image_into_fourths(img, large_image_name)
			elif size < 36000000000:
				cprint("Splitting " + large_image_name + ' into ninths', 'yellow')
				split_image_into_ninths(img, large_image_name)
			elif size < 64000000000
				cprint("Splitting " + large_image_name + ' into sixteenths', 'yellow')
				split_image_into_sixteenths(img, large_image_name)
			else:
				cprint(large_image_name + ' is extremely large, and was skipped.', 'red')

			if FLAGS.destructive:
				cprint("DESTROYING " + img_path, 'white', 'on_red')
				subprocess.check_call("rm " + img_path, shell=True)


# Main body
if __name__ == '__main__':
	main()