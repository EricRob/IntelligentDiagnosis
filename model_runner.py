#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import argparse
import os
import pdb
from termcolor import cprint
from datetime import datetime
import subprocess

# Global variables
RESULTS_DIR = '/data/recurrence_seq_lstm/results/'
DATA_DEFAULT = '/data/recurrence_seq_lstm/image_data/'
DATA_CONDITIONS = '/data/recurrence_seq_lstm/data_conditions/'
SCRIPT_DIR = '/home/wanglab/Desktop/recurrence_seq_lstm/IntelligentDiagnosis/'

parser = argparse.ArgumentParser(description='Run a set of recurrence models on a given data set')

parser.add_argument('--model', default=None, type=str, help='Base name of models to run')
parser.add_argument('--data', default=DATA_DEFAULT, type=str, help='[NOT WHOLE PATH] Folder of network binary files')
parser.add_argument('--name', default=None, type=str, required=True, help='Name of results file to save')
parser.add_argument('--cross_valid', default='', type=str, help='Data set for cross validation testing')
parser.add_argument('--folds', default=6, type=int, help='Number of cross validation folds (should almost always be 6)')
parser.add_argument('--config', default='test', type=str, help='Run network in test or train mode')

parser.add_argument('--epochs', default='50', type=str, help='Number of epochs for training network')

parser.add_argument('--preprocess', default=False, action='store_true', help='Create data before testing or running network')
parser.add_argument('--detections', default='/data/recurrence_seq_lstm/qupath_output/', type=str, help='Location of qupath output for preprocessing')
parser.add_argument('--pp_config', default='', type=str, help='Configuration for preprocessing')
parser.add_argument('--pp_min', default=10, type=int, help='min_patch parameter for preprocess_lstm')
parser.add_argument('--pp_thresh', default=30, type=int, help='patch_thresh parameter for preprocess_lstm')

parser.add_argument('--summarize', default=False, action='store_true', help='Summarize results using majority_vote.py')
parser.add_argument('--no_script', default=False, action='store_true', help='Skip creating .sh script with network list')
parser.add_argument('--no_execute', default=False, action='store_true', help='Skip running networks (i.e. only create scripts)')
parser.add_argument('--omen', default=False, action='store_true', help='Model runner is running on OMEN rather than PrecisionTower')


ARGS = parser.parse_args()

if ARGS.config == 'test':
	CONFIG = ' --config=test'
else:
	CONFIG = None



# Class declarations

# Function declarations

def retest(retest_list):
	if retest_list:
		cprint('Retest list:', 'white', 'on_red')
	for failure in retest_list:
		cprint(failure, 'red')

def get_models_list():
	models_list = []
	for root, dirs, _ in os.walk(os.path.join(RESULTS_DIR)):
		for folder in sorted(dirs):
			if ARGS.model in folder:
				num_loc = folder.find('_00') + 1
				if not num_loc:
					models_list.append((folder, ''))
				else:
					model_num = folder[num_loc:num_loc+3]
					models_list.append((folder[:num_loc], model_num))
	return models_list

def get_data_list():
	data_list = []
	for root, dirs, _ in os.walk(os.path.join(DATA_CONDITIONS)):
		for folder in sorted(dirs):
			if ARGS.data in folder:
				n = 0
				while(n < ARGS.folds):
					condition_folder = "00%i_condition" % (n + 1)
					data_list.append((folder, condition_folder, "00%i" % (n + 1)))
					n += 1
	return data_list

def get_script_file(new=True):
	os.makedirs(os.path.join(SCRIPT_DIR, 'scripts'), exist_ok=True)
	script_name = os.path.join(SCRIPT_DIR, 'scripts', ARGS.name + '.sh')
	if new:
		script = open(script_name, 'wt+')
	else:
		script = open(script_name, 'at+')
	if new:
		script.write("##\n## Script created automatically on " + datetime.now().strftime('%Y-%m-%d, at %H:%M:%S') + '\n##\n\n')
	return script, script_name

def save_script(script_file, script_name):
	script_file.close()
	subprocess.check_call('chmod +x ' + script_name, shell=True)
	print("script saved!")

def test_default_data():
	retest_list = []
	models_list = []
	models_list = get_models_list()
	if not models_list:
		print('No models matching ' + ARGS.model + ' found, check parameters and retry.')
		return
	if not ARGS.no_script:
		script, script_name = get_script_file(new=False)
	python_base = 'python3 ' + os.path.join(SCRIPT_DIR,'recurrence_lstm_features.py')
	if ARGS.omen:
		python_base = python_base + ' --omen_run=True'
	model_path_base = ' --model_path=' + RESULTS_DIR
	recur = ' --recur_data_path=' + os.path.join(ARGS.data, 'recurrence')
	nonrecur = ' --nonrecur_data_path=' + os.path.join(ARGS.data, 'nonrecurrence')
	for model in sorted(models_list):
		run_line = python_base + recur + nonrecur + CONFIG + model_path_base + model[0] + model[1] + ' --results_prepend=' + ARGS.name + '_' + model[1]
		if not model[1]:
			run_line = run_line[:-1]
		if ARGS.summarize:
			name = ARGS.name + '_' + model[1]
			base_path = ' --base_path=' + os.path.join(RESULTS_DIR, name)
			majority_vote = 'python3 ' + os.path.join(SCRIPT_DIR,'majority_vote.py') + base_path
		try:
			if not ARGS.no_execute:
				subprocess.check_call(run_line, shell=True)
				if ARGS.summarize:
					subprocess.check_call(majority_vote, shell=True)
			if not ARGS.no_script:
				script.write(run_line + '\n')
				if ARGS.summarize:
					script.write(majority_vote + '\n')

		except:
			cprint('Error while testing ' + model[0] + model[1] + ' with default data (' + ARGS.data + '), must retest!', 'red')
			retest_list.append(model[0]+model[1])
		if ARGS.summarize:
			python_base = 'python3 ' + os.path.join(SCRIPT_DIR,'multitest_summary.py')
			model = ' --model=' + model[0][:-1]
			condition = ' --condition=' + ARGS.name
			run_line = python_base + model + condition
			if not ARGS.no_script:
				script.write(run_line + '\n')
			if not ARGS.no_execute:
				subprocess.check_call(run_line, shell=True)

	if not ARGS.no_script:
		save_script(script, script_name)
	retest(retest_list)

def test_cross_valid_data():
	retest_list = []
	python_base = 'python3 ' + SCRIPT_DIR + 'recurrence_lstm_features.py'
	if ARGS.omen:
		python_base = python_base + ' --omen_run=True'
	models_list = []
	models_list = get_models_list()
	if not models_list:
		print('No models matching ' + ARGS.model + ' found, check parameters and retry.')
		return
	if not ARGS.no_script:
		script, script_name = get_script_file()

	for model in sorted(models_list):
		recur = ' --recur_data_path=' + os.path.join(DATA_CONDITIONS, ARGS.cross_valid, model[1] + "_condition", 'recurrence')
		nonrecur = ' --nonrecur_data_path=' + os.path.join(DATA_CONDITIONS, ARGS.cross_valid, model[1] + "_condition", 'nonrecurrence')
		model_name = ' --model=' + os.path.join(RESULTS_DIR, model[0] + model[1])
		results = ' --results_prepend=' + ARGS.name + '_' + model[1]
		run_line = python_base + recur + nonrecur + CONFIG + model_name + results
		if not model[1]:
			run_line = run_line[:-1]
		if ARGS.summarize:
			name = ARGS.name + '_' + model[1]
			base_path = ' --base_path=' + os.path.join(RESULTS_DIR, name)
			majority_vote = 'python3 ' + SCRIPT_DIR + 'majority_vote.py' + base_path[:-1]
		try:
			if not ARGS.no_execute:
				subprocess.check_call(run_line, shell=True)
				if ARGS.summarize:
						subprocess.check_call(majority_vote, shell=True)
			if not ARGS.no_script:
				script.write(run_line + '\n')
				if ARGS.summarize:
					script.write(majority_vote + '\n')
		except:
			cprint('Error testing ' + model[0] + model[1] + ' with cross validation data ' + ARGS.cross_valid + '(condition ' + model[1] + '), must retest!', 'red')
			retest_list.append(model[0] + model[1])
	if not ARGS.no_script:
		save_script(script, script_name)
	retest(retest_list)


def test_outside_data():
	retest_list = []
	python_base = 'python3 ' + SCRIPT_DIR + 'recurrence_lstm_features.py'
	if ARGS.omen:
		python_base = python_base + ' --omen_run=True'
	models_list = []
	models_list = get_models_list()
	if not models_list:
		print('No models matching ' + ARGS.model + ' found, check parameters and retry.')
		return
	data_list = get_data_list()
	for model in sorted(models_list):
		for data in data_list:
			recur = ' --recur_data_path=' + os.path.join(DATA_CONDITIONS, data[0], data[1], 'recurrence')
			nonrecur = ' --nonrecur_data_path=' + os.path.join(DATA_CONDITIONS, data[0], data[1], 'nonrecurrence')
			model_name = ' --model=' + os.path.join(RESULTS_DIR, model[0] + model[1])
			results = ' --results_prepend=' + ARGS.name + '_' + model[1]
			run_line = python_base + recur + nonrecur + CONFIG + model_name + results
			try:
				subprocess.check_call(run_line, shell=True)
			except:
				cprint('Error testing ' + model[0] + model[1] + ' with outside data ' + ARGS.data + '(condition ' + data[0] + ', ' + data[1] + '), must retest!', 'red')
				retest_list.append(model[0] + model[1] + ', ' + data[0] + '/' + data[2])
	retest(retest_list)

def train_cross_valid_data():
	retest_list = []
	python_base = 'python3 ' + SCRIPT_DIR + 'recurrence_lstm_features.py'
	if ARGS.omen:
		python_base = python_base + ' --omen_run=True'
	train_middle = ' --save_model=True'
	epochs = ' --epochs=' + ARGS.epochs
	if ARGS.data == DATA_DEFAULT:
		data_list = [DATA_DEFAULT]
	else:
		data_list = get_data_list()
	if not ARGS.no_script:
		script, script_name = get_script_file(new=False)
	
	for data in data_list:
		if data == DATA_DEFAULT:
			recur = ' --recur_data_path=' + os.path.join(data, 'recurrence')
			nonrecur = ' --nonrecur_data_path=' + os.path.join(data, 'nonrecurrence')
			results = ' --results_prepend=' + ARGS.name
		else:
			recur = ' --recur_data_path=' + os.path.join(DATA_CONDITIONS, data[0], data[1], 'recurrence')
			nonrecur = ' --nonrecur_data_path=' + os.path.join(DATA_CONDITIONS, data[0], data[1], 'nonrecurrence')
			results = ' --results_prepend=' + ARGS.name + '_' + data[2]
		run_line = python_base + recur + nonrecur + train_middle + epochs + results
		try:
			if not ARGS.no_execute:
				subprocess.check_call(run_line, shell=True)
			if not ARGS.no_script:
				script.write(run_line + '\n')
		except:
			if data == DATA_DEFAULT:
				cprint('Error training with default data, must retest!', 'red')
				retest_list.append(data)
			else:
				cprint('Error training with ' + ARGS.data + '(condition ' + data[0] + ', ' + data[1] + '), must retest!', 'red')
				retest_list.append(data[0] + '/' + data[2])
	if ARGS.summarize:
		concatenate = 'python3 ' + SCRIPT_DIR + 'concatenate_voting_csv.py --condition_name=' + ARGS.name
		majority_vote = 'python3 ' + SCRIPT_DIR + 'majority_vote.py --base_path=' + os.path.join(RESULTS_DIR, ARGS.name)
		if not ARGS.no_script:
			script.write(concatenate + '\n')
			script.write(majority_vote + '\n')
		if not ARGS.no_execute:
			subprocess.check_call(concatenate, shell=True)
			subprocess.check_call(majority_vote, shell=True)

	if not ARGS.no_script:
		save_script(script, script_name)

	retest(retest_list)

def preprocess_train_data():
	python_base = 'python3 ' + SCRIPT_DIR + 'preprocess_lstm.py'
	cond_path = ' --condition_path=' + os.path.join(DATA_CONDITIONS,ARGS.data)
	pp_config = ' --config=' + ARGS.pp_config
	min_patch = ' --min_patch=' + str(ARGS.pp_min)
	patch_thresh = ' --patch_thresh=' + str(ARGS.pp_thresh)
	detections = ' --detections_path=' + ARGS.detections
	run_line = python_base + cond_path + detections + pp_config + min_patch + patch_thresh
	print(run_line)
	if not ARGS.no_script:
		script, script_name = get_script_file()
	try:
		if not ARGS.no_script:
			script.write(run_line + '\n')
		if not ARGS.no_execute:
			subprocess.check_call(run_line, shell=True)
	except:
		cprint('Unable to generate preprocessing data! Must exit!', 'red')
		sys.exit(1)
	if not ARGS.no_script:
		save_script(script, script_name)

def preprocess_test_default_data():
	python_base = 'python3 ' + SCRIPT_DIR + 'preprocess_lstm.py'
	if ARGS.pp_config:
		pp_config = ' --config=' + ARGS.pp_config
	else:
		pp_config = ''
	detections = ' --detections_path=' + ARGS.detections
	min_patch = ' --min_patch=' + str(ARGS.pp_min)
	patch_thresh = ' --patch_thresh=' + str(ARGS.pp_thresh)
	run_line = python_base + detections + pp_config + min_patch + patch_thresh
	if not ARGS.no_script:
		script, script_name = get_script_file()
	try:
		if not ARGS.no_script:
			script.write(run_line + '\n')
		if not ARGS.no_execute:
			subprocess.check_call(run_line, shell=True)
	except:
		cprint('Unable to generate preprocessing data! Must exit!', 'red')
		sys.exit(1)
	if not ARGS.no_script:
		save_script(script, script_name)

def swap_machine():
	if ARGS.omen:
		global RESULTS_DIR
		global DATA_CONDITIONS
		global SCRIPT_DIR

		if ARGS.preprocess:
			print('*** Unable to preprocess data from OMEN, please check command line arguments. ***')
			return 1
		RESULTS_DIR = '/hdd/ID_net/results'
		DATA_CONDITIONS = '/hdd/ID_net/data_conditions/'
		SCRIPT_DIR = '/hdd/ID_net/IntelligentDiagnosis/'
	return 0

def main():


	exit = swap_machine()
	if exit:
		sys.exit(1)
	if ARGS.config == 'test':
		if not ARGS.model:
			cprint('TEST CONFIG: NO MODEL SPECIFIED FOR TESTING', 'red')
			sys.exit(1)
		if ARGS.data == DATA_DEFAULT:
			if ARGS.cross_valid:
				test_cross_valid_data()
			else:
				if ARGS.preprocess:
					print('CAUTION: ONLY VANILLA PREPROCESSING')
					preprocess_test_default_data()
				test_default_data()
		else:
			test_outside_data()
	else:
		if ARGS.preprocess:
			print('CAUTION: ONLY VANILLA PREPROCESSING')
			preprocess_train_data()
		train_cross_valid_data()

	# subprocess.check_call('PLACEHOLDER FOR NETWORK LINE', shell=True)

# Main body
if __name__ == '__main__':
	main()