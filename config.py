#!/user/bin/env python3 -tt
"""
##################

Written by Eric J Robinson (https://github.com/EricRob)

##################

Create configuration file for use with simple_* scripts.

Also initializes directories for data storage, if desired.

##################
"""

# Imports
import sys
import os
import pickle
import pdb
from art import tprint
from termcolor import cprint

# Global variables

# Class declarations
class Config:
	def __init__(self, filename='./default_config.file'):
		self.filename = filename
		self.images_dir = './data/images'
		self.mask_dir = './data/masks'
		self.image_csv = './data/image_list.csv'
		self.image_bin_dir = './data/image_binaires'
		self.detections_dir = './data/detections'
		self.err_csv = './error_list.csv'
		self.voting_csv = './voting_file.csv'
		self.output_csv = './voting_results.csv'
		self.vote_cutoff = 0.5

	def initialize_dirs(self):
		os.makedirs(self.image_bin_dir, exist_ok=True)
		os.makedirs(self.detections_dir, exist_ok=True)
		os.makedirs(self.images_dir, exist_ok=True)
		os.makedirs(self.mask_dir, exist_ok=True)
	def save(self):
		with open(self.filename, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
	def return_type(self, val):
		if 'dir' in val:
			return 'dir path, default=%s' % getattr(self, val)
		elif 'csv' in val:
			return 'csv, default=%s' % getattr(self, val)
	def valid_suffix(self, val, inp):
		if 'csv' in val:
			if '.csv' not in inp:
				inp = inp.split('.')[0] + '.csv'
		elif val == 'filename':
			inp = inp + '.file'
		elif 'cutoff' in val:
			inp = str(0.5)
		return inp

# Function declarations
def query_yes_no(question, default="yes"):
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

def create_default(config):
	if query_yes_no('Use default values for configuration?'):
		if query_yes_no('Initialize project directories?'):
			config.initialize_dirs()
		return True
	else:
		return False

def set_custom_values(config):
	config_values = config.__dict__
	for val in config_values:
		if val == 'filename':
			continue
		inp = input('Setting for %s (%s): ' % (val, config.return_type(val)))
		if inp:
			inp = config.valid_suffix(val, inp)
			setattr(config, val, inp)

def main():

	tprint('recurrence_lstm', font='speed')
	config = Config()
	if os.path.exists(config.filename):
		if query_yes_no('Overwrite default configuration: %s?' % config.filename, default='no'):
			print('**Overwriting %s' % config.filename)
		elif query_yes_no('Create new configuration?'):
				config.filename = config.valid_suffix('filename', input('New configuration name: '))
		else:
			print('Exiting: No changes made to configurations')
			return
	elif not query_yes_no('No default configuration found, create default configuration?' % config.filename):
			config.filename = config.valid_suffix('filename', input('New configuration name: '))
	else:
		print('Exiting: No changes made to configurations')
		return

	cprint('\n--> Creating configuration...', 'grey', 'on_white')
	if create_default(config):
		config.save()
		cprint('Default configuration %s saved! Ready to create binaries' % config.filename, 'green')
	else:
		cprint('Leave empty (hit return) to use default value', 'grey', 'on_white')
		set_custom_values(config)
		config.save()
		cprint('Custom configuration %s saved! Ready to create binaries' % config.filename, 'green')

# Main body
if __name__ == '__main__':
	main()
		