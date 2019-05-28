#!/usr/bin/python3 python3

'''
##################

Written by Eric J Robinson (https://github.com/EricRob)

##################

Processing of an H&E image, qupath detections, and a binary mask
to generate a binary file readable by the recurrence_lstm network.

Requires a valid image, tsv of qupath output, binary mask from gimp,
and image_list.csv.

##################

Sample image_list.csv:

|  mode  | subject |    image     | label | source |
|--------|---------|--------------|-------|--------|
| train  |  00-00  | 00_00_00.tif |   0   |  CUMC  |
| train  |  11-11  | 11_11_11.tif |   1   |  CUMC  |
| valid  |  00-00  | 00_00_00.tif |   0   |  CUMC  |
| valid  |  11-11  | 11_11_11.tif |   1   |  CUMC  |
| test   |  00-00  | 00_00_00.tif |   0   |  CUMC  |
| test   |  11-11  | 11_11_11.tif |   1   |  CUMC  |

####################

'''

import csv
import os
import pdb
import sys
import warnings
from termcolor import cprint
from art import tprint
import argparse
import pickle
from config import Config

import simple_centroid as gauss


warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)


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
		self.mask = os.path.join(config.mask_dir, 'mask_%s.tif' % self.img_base)
		self.detections = os.path.join(config.detections_dir, '%s Detectionstxt' % self.img_base)
		self.error_code = self.bin_requirements_met(config)
		self.gauss_config = self.assign_gauss_config(config)

	def assign_str_label(self, label):
		if label:
			return 'recurrence'
		else:
			return 'nonrecurrence'
	def assign_gauss_config(self, config):
		if self.source.lower() == 'yale':
			gauss_config = gauss.YaleConfig()
		else:
			gauss_config = gauss.OriginalConfig()
		gauss_config.image_data_folder_path = config.images_dir
		return gauss_config
	def bin_requirements_met(self, config):
		err = 7
		if os.path.exists(os.path.join(config.images_dir, self.image)):
			err -= 4
		if os.path.exists(self.detections):
			err -= 2
		if os.path.exists(self.mask):
			err -= 1
		return err
	def create_bin(self):
		return not os.path.exists(self.bin)
	def raise_error(self, feature_err=False):
			if self.error_code == 7:
				cprint('H&E image, detections, and mask files not found for %s: %s' % (self.subject, self.img_base), 'red')
				return [self.mode, self.subject, self.image, str(self.label), self.source, 'not found','not found', 'not found', 'n/a']
			elif self.error_code == 1:
				cprint('Mask tiff not found for %s: %s' % (self.subject, self.img_base), 'red')
				return [self.mode, self.subject, self.image, str(self.label), self.source, 'ok', 'not found', 'ok', 'n/a']
			elif self.error_code == 2:
				cprint('Detections file not found for %s: %s' % (self.subject, self.img_base), 'red')
				return [self.mode, self.subject, self.image, str(self.label), self.source, 'ok','ok', 'not found', 'n/a']
			elif self.error_code == 3:
				cprint('Detections and mask files not found for %s: %s' % (self.subject, self.img_base), 'red')
				return [self.mode, self.subject, self.image, str(self.label), self.source, 'ok','not found', 'not found', 'n/a']
			elif self.error_code == 4:
				cprint('H&E image not found for %s: %s' % (self.subject, self.img_base), 'red')
				return [self.mode, self.subject, self.image, str(self.label), self.source, 'not found','ok', 'ok', 'n/a']
			elif self.error_code == 5:
				cprint('H&E image and mask files not found for %s: %s' % (self.subject, self.img_base), 'red')
				return [self.mode, self.subject, self.image, str(self.label), self.source, 'not found','not found', 'ok', 'n/a']
			elif self.error_code == 6:
				cprint('H&E image and detections files not found for %s: %s' % (self.subject, self.img_base), 'red')
				return [self.mode, self.subject, self.image, str(self.label), self.source, 'not found','ok', 'not found', 'n/a']
			elif feature_err:
				cprint('No sequences processed for %s: %s' % (self.subject, self.img_base), 'red')
				return [self.mode, self.subject, self.image, str(self.label), self.source, 'ok','ok', 'ok', 'no features']


def process_input_csv(config):
	data = {'recurrence' : {'train': [], 'valid': [], 'test': []}, \
	'nonrecurrence' : {'train': [], 'valid': [], 'test': []}}
	with open(config.image_csv) as csvfile:
		csvreader = csv.DictReader(csvfile)
		for row in csvreader:
			he_img = HE_Image(row, config)
			data[he_img.str_label][he_img.mode].append(he_img)
	return data

def write_error_csv(err_list, config):
	for error in err_list:
		with open(config.err_csv, 'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['mode', 'subject', 'image', 'label', 'source', 'image', 'mask', 'detections', 'other'])
			for error in err_list:
				writer.writerow(error)
	return

def generate_and_append_bin(image_list, bin_file, config):
	err_list = []

	for image in image_list:
		if image.create_bin():
			if image.error_code:
				err_list.append(image.raise_error())
				continue
			else:
				cprint('Creating image binary file for %s: %s' % (image.subject, image.img_base), 'white', 'on_green')
				features = gauss.generate_sequences(image.mask, \
					image.gauss_config, \
					image_name=image.img_base, \
					subject_id=image.subject, \
					detections=config.detections_dir)
				
				if not features:
					err_list.append(image.raise_error(feature_err=True))
					continue
				with open(image.bin, 'wb+') as image_bin:
					gauss.write_image_bin(image_bin, \
						image.image, \
						image.subject, \
						features, \
						image.gauss_config)

		with open(image.bin, 'rb+') as image_bin:
			image_bytes = image_bin.read(os.path.getsize(image.bin))

		bin_file.write(image_bytes)
	return err_list

def get_config(config_name):
	try:
		if config_name == 'default':
			with open('./default_config.file', 'rb') as f:
				config = pickle.load(f)
		else:
			config = pickle.load(os.path.join(config_name + '.file'))
	except:
		print('[ERROR] No valid config file: %s' % config_name)

def main(ars):
	tprint('simple_process', font='speed')
	config = get_config(ars.conf)

	input_data = process_input_csv(config)
	
	err_list = []
	for label in input_data:
		for mode in input_data[label]:
			with open(os.path.join(config.image_bin_dir, '%s_%s.bin' % (label, mode)), "wb+") as bin_file:
				cprint('Working on %s_%s.bin' % (label, mode), 'blue', 'on_white')
				err_append = generate_and_append_bin(input_data[label][mode], bin_file, config)
			err_list = err_list + err_append
	tprint('binaries\ngenerated', font='sub-zero')
	write_error_csv(err_list, config)

	return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create binary files for feeding into recurrence_seq_lstm')
	parser.add_argument('--conf', default='default', type=str, help='Name of configuration file for processing and voting')
	ars = parser.parse_args()
	main(ars)
	sys.exit(0)
