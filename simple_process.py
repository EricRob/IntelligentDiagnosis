#!/usr/bin/python3 python3

import csv
import json
import os
import pdb
import subprocess
import sys
import warnings
from termcolor import cprint

import simple_centroid as gauss


warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

class OriginalPatchConfig(object):
	images_dir = './'
	json_file = './data.json'
	image_csv = './image_list.csv'
	image_bin_dir = './'
	output_bin_dir = './'
	detections_path = './'
	err_list = './error_list.csv'
	os.makedirs(image_bin_dir, exist_ok=True)
	os.makedirs(detections_path, exist_ok=True)
	os.makedirs(output_bin_dir, exist_ok=True)

class HE_Image:
	def __init__(self, meta_data, config):
		self.subject = meta_data['subject']
		self.label = int(meta_data['label'])
		self.str_label = self.assign_str_label(self.label)
		self.image = meta_data['image']
		self.img_base = os.path.splitext(os.path.basename(self.image))[0]
		self.bin = os.path.join(config.image_bin_dir, self.img_base + '.bin')
		self.source = meta_data['source']
		self.mode = meta_data['mode']
		self.mask = os.path.join(config.images_dir, 'mask_%s.tif' % self.img_base)
		self.detections = os.path.join(config.detections_path, '%s Detectionstxt' % self.img_base)

	def assign_str_label(self, label):
		if label:
			return 'recurrence'
		else:
			return 'nonrecurrence'


def process_csv_input(config):
	data = {'recurrence' : {'train': [], 'valid': [], 'test': []}, \
	'nonrecurrence' : {'train': [], 'valid': [], 'test': []}}
	with open(config.image_csv) as csvfile:
		csvreader = csv.DictReader(csvfile)
		for row in csvreader:
			he_img = HE_Image(row, config)
			data[he_img.str_label][he_img.mode].append(he_img)
	return data

def bin_file_requirements_met(image):
	val = 0
	if os.path.exists(image.detections):
		val += 1
	if os.path.exists(image.mask):
		val += 2
	return val

def gauss_sampling(data, bin_file, config):
	err_list = []
	gauss_config = gauss.OriginalPatchConfig_test()

	gauss_config.image_data_folder_path = config.images_dir

	for image in data:
		if not os.path.exists(image.bin):
			cprint('Creating image binary file for %s: %s' % (image.subject, image.img_base), 'white', 'on_green')
			val = bin_file_requirements_met(image)
			if val == 3:
				seq_features = gauss.generate_sequences(image.mask, \
					gauss_config, image_name=image.img_base, subject_id=image.subject, detections=config.detections_path)

				if not seq_features:
					print('No sequences processed for %s: %s' % (image.subject, image.img_base), 'red')
					err_list.append([image.mode, image.subject, image.image, str(image.label), image.source, 'ok', 'ok', 'no features'])
					continue

				gauss.regional_verification(seq_features, gauss_config, image.img_base, image.subject)

				with open(image.bin, 'wb+') as image_bin:
					gauss.write_image_bin(image_bin, image.image, image.subject, seq_features, gauss_config)
			else:
				if val == 0:
					cprint('Detections and mask file not found for %s: %s' % (image.subject, image.img_base), 'red')
					err_list.append([image.mode, image.subject, image.image, str(image.label), image.source, 'not found', 'not found', 'n/a'])
				elif val == 1:
					cprint('Mask tiff not found for %s: %s' % (image.subject, image.img_base), 'red')
					err_list.append([image.mode, image.subject, image.image, str(image.label), image.source, 'not found', 'ok', 'n/a'])
				elif val == 2:
					cprint('Detections file not found for %s: %s' % (image.subject, image.img_base), 'red')
					err_list.append([image.mode, image.subject, image.image, str(image.label), image.source, 'ok', 'not found', 'n/a'])
				continue
			# Add error log for resolution with missing files

		# cprint("Appending " + image + FILLER, 'green', end="\r")
		with open(image.bin, 'rb+') as image_bin:
			image_bytes = image_bin.read(os.path.getsize(image.bin))

		bin_file.write(image_bytes)
	return err_list

def main():
	config = OriginalPatchConfig()
	data = process_csv_input(config)
	err_list = []
	for label in data:
		for mode in data[label]:
			with open(os.path.join(config.output_bin_dir, '%s_%s.bin' % (label, mode)), "wb+") as bin_file:
				err_append = gauss_sampling(data[label][mode], bin_file, config)
			err_list = err_list + err_append
	for error in err_list:
		with open(config.err_list, 'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['mode', 'subject', 'image', 'label', 'source', 'mask', 'detections', 'other'])
			for error in err_list:
				writer.writerow(error)
	return 0

if __name__ == '__main__':
	main()
	sys.exit(0)