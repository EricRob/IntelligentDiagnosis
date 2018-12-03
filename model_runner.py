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
import subprocess

# Global variables
RESULTS_DIR = '/data/recurrence_seq_lstm/results/'
DATA_DEFAULT = '/data/recurrence_seq_lstm/image_data/'
DATA_CONDITIONS = '/data/recurrence_seq_lstm/data_conditions/'
SCRIPT_DIR = '/home/wanglab/Desktop/recurrence_seq_lstm/IntelligentDiagnosis'

parser = argparse.ArgumentParser(description='Run a set of recurrence models on a given data set')

parser.add_argument('--model', default=None, type=str, help='Base name of models to run')
parser.add_argument('--data', default=DATA_DEFAULT, type=str, help='Base path of network binary files')
parser.add_argument('--name', default=None, type=str, required=True, help='Name of results file to save')
parser.add_argument('--cross_valid', default=False, type=str, help='Data set for cross validation testing')
parser.add_argument('--folds', default=6, type=int, help='Number of cross validation folds (should almost always be 6)')
parser.add_argument('--config', default='test', type=str, help='Run network in test or train mode')

parser.add_argument('--preprocess', default=False, type=bool, help='Create data before testing or running network')
parser.add_argument('--detections', default='/data/yale_qupath/measurements/', type=str, help='Location of qupath output for preprocessing')
parser.add_argument('--pp_config', default=None, type=str, help='Configuration for preprocessing')

parser.add_argument('--summarize', default=True, type=bool, help='Summarize results using majority_vote.py')
parser.add_argument('--create_script', default=False, type=bool, help='Create .sh script with network list')
parser.add_argument('--run_nets', default=True, type=bool, help='Run networks in network list')


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

def test_default_data():
	retest_list = []
	models_list = get_models_list()
	python_base = 'python3 recurrence_lstm_features.py'
	model_path_base = ' --model_path=' + RESULTS_DIR
	recur = ' --recur_data_path=' + os.path.join(ARGS.data, 'recurrence')
	nonrecur = ' --nonrecur_data_path=' + os.path.join(ARGS.data, 'nonrecurrence')
	for model in sorted(models_list):
		run_line = python_base + recur + nonrecur + CONFIG + model_path_base + model[0] + model[1] + ' --results_prepend=' + ARGS.name + '_' + model[1]
		if ARGS.summarize:
			name = ARGS.name + '_' + model[1]
			base_path = ' --base_path=' + os.path.join(RESULTS_DIR, name)
			majority_vote = 'python3 majority_vote.py' + base_path
		try:
			subprocess.check_call(run_line, shell=True)
			subprocess.check_call(majority_vote, shell=True)

		except:
			cprint('Error testing ' + model[0] + model[1] + ' with default data (' + ARGS.data + '), must retest!', 'red')
			retest_list.append(model[0]+model[1])
	retest(retest_list)

def test_cross_valid_data():
	retest_list = []
	python_base = 'python3 recurrence_lstm_features.py'
	models_list = get_models_list()
	for model in sorted(models_list):
		recur = ' --recur_data_path=' + os.path.join(DATA_CONDITIONS, ARGS.cross_valid, model[1] + "_condition", 'recurrence')
		nonrecur = ' --nonrecur_data_path=' + os.path.join(DATA_CONDITIONS, ARGS.cross_valid, model[1] + "_condition", 'nonrecurrence')
		model_name = ' --model=' + os.path.join(RESULTS_DIR, model[0] + model[1])
		results = ' --results_prepend=' + ARGS.name + '_' + model[1]
		run_line = python_base + recur + nonrecur + CONFIG + model_name + results
		try:
			subprocess.check_call(run_line, shell=True)
		except:
			cprint('Error testing ' + model[0] + model[1] + ' with cross validation data ' + ARGS.cross_valid + '(condition ' + model[1] + '), must retest!', 'red')
			retest_list.append(model[0] + model[1])
	retest(retest_list)

def test_outside_data():
	retest_list = []
	python_base = 'python3 recurrence_lstm_features.py'
	models_list = get_models_list()
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
	python_base = 'python3 recurrence_lstm_features.py'
	train_middle = ' --epochs=50 --save_model=True'
	data_list = get_data_list()
	if ARGS.create_script:
		script_name = os.path.join(SCRIPT_DIR, ARGS.name + '.sh')
		script = open(script_name, 'wt+')
	
	for data in data_list:
		recur = ' --recur_data_path=' + os.path.join(DATA_CONDITIONS, data[0], data[1], 'recurrence')
		nonrecur = ' --nonrecur_data_path=' + os.path.join(DATA_CONDITIONS, data[0], data[1], 'nonrecurrence')
		results = ' --results_prepend=' + ARGS.name + '_' + data[2]
		run_line = python_base + recur + nonrecur + train_middle + results
		try:
			# if ARGS.run_nets:
				# subprocess.check_call(run_line, shell=True)
			if ARGS.create_script:
				script.write(run_line + '\n')
		except:
			cprint('Error training with ' + ARGS.data + '(condition ' + data[0] + ', ' + data[1] + '), must retest!', 'red')
			retest_list.append(data[0] + '/' + data[2])
	if ARGS.create_script:
		script.close()
		subprocess.check_call('chmod +x ' + script_name, shell=True)
		print("script saved!")

	retest(retest_list)

def preprocess_train_data():
	python_base = 'python3 preprocess_lstm.py'
	cond_path = ' --condition_path=' + os.path.join(DATA_CONDITIONS,ARGS.data)
	pp_config = ' --config=' + ARGS.pp_config
	detections = ' --detections_path=' + ARGS.detections
	run_line = python_base + cond_path + detections + pp_config
	try:
		subprocess.check_call(run_line, shell=True)
	except:
		cprint('Unable to generate preprocessing data! Must exit!', 'red')
		sys.exit(1)

def main():

	if ARGS.config == 'test':
		if not ARGS.model:
			cprint('TEST CONFIG: NO MODEL SPECIFIED FOR TESTING', 'red')
			sys.exit(1)
		if ARGS.data == DATA_DEFAULT:
			if ARGS.cross_valid:
				test_cross_valid_data()
			else:
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