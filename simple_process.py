#!/usr/bin/python3 python3

import csv
import json
import os
import pdb
import subprocess
import sys
import warnings
from termcolor import cprint

import centroid_lstm as gauss


warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

class OriginalPatchConfig(object):
	images_dir = './out'
	json_file = './output.json'
	qupath_script = './qupath_script.groovy'
	image_bin_dir = './'
	bin_path = './'
	qupath_is_working = False
	detections_path = './out'

def bin_file_requirements_met(image, config):
	detections_filename = os.path.join(config.detections_path, '%s Detectionstxt' % image)
	mask_path = os.path.join(config.images_dir, 'mask_%s.tif' % image)
	val = 0
	if os.path.exists(detections_filename):
		val += 1
	if os.path.exists(mask_path):
		val += 2
	return val

def gauss_sampling(data, bin_file, config):

	gauss_config = gauss.OriginalPatchConfig_test()

	gauss_config.image_data_folder_path = "./data/image"

	for subject in data:
		for image in data[subject]:

			image = os.path.splitext(os.path.basename(image))[0]
			image_bin_name = "%s.bin" % image
			image_bin_path = os.path.join(config.image_bin_dir, image_bin_name)

			if not os.path.exists(image_bin_path):
				cprint('Creating image binary file for %s: %s' % (subject, image), 'white', 'on_green')
				val = bin_file_requirements_met(image, config)
				if val == 3:
					seq_features = gauss.generate_sequences(os.path.join(config.images_dir, 'mask_%s.tif' % image), \
						gauss_config, image_name=image, subject_id=subject, detections=config.detections_path)

					if not seq_features:
						print('No sequences processed, skipping due to lack of features')
						continue

					gauss.regional_verification(seq_features, gauss_config, image, data['subject'])

					with open(image_bin_path, 'wb+') as image_bin:
						image_tiff_name = '%s.tif' % image
						gauss.write_image_bin(image_bin, image_tiff_name, data['subject'], seq_features, gauss_config)
				else:
					if val == 0:
						cprint('Detections and mask file not found for %s: %s' % (subject, image), 'red')
					elif val == 1:
						cprint('Detections file not found for %s: %s' % (subject, image), 'red')
					elif val == 2:
						cprint('Mask tiff not found for %s: %s' % (subject, image), 'red')						
					continue

			# cprint("Appending " + image + FILLER, 'green', end="\r")
			with open(image_bin_path, 'rb+') as image_bin:
				image_bytes = image_bin.read(os.path.getsize(image_bin_path))

			bin_file.write(image_bytes)

def main():
	config = OriginalPatchConfig()
	data = json.load(open(config.json_file))

	with open(os.path.join(config.bin_path, 'test.bin'), "ab+") as bin_file:
		gauss_sampling(data, bin_file, config)
	return 0

if __name__ == '__main__':
	main()
	sys.exit(0)