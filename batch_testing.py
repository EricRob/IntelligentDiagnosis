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

parser = argparse.ArgumentParser(description='Run sets of models for a given test set')

parser.add_argument('--cumc', default=False, action='store_true', help='Run CU models for testing')
parser.add_argument('--yale', default=False, action='store_true', help='Run Yale models for testing')
parser.add_argument('--sinai_test', default=False, action='store_true', help='Test models on Sinai data')
parser.add_argument('--yale_test', default=False, action='store_true', help='Test models on Yale data')
parser.add_argument('--geis_test', default=False, action='store_true', help='Test models on Geisinger data')
parser.add_argument('--cu_test', default=False, action='store_true', help='Test models on CUMC data')
parser.add_argument('--omen', default=False, action='store_true', help='Run on OMEN tower')
parser.add_argument('--fullyc', default=False, action='store_true', help='Run fully connected models only')


CU_MODELS = ['CU_Apr_05', 'FC_CU_Apr_05','CU_Feb_11_200', 'CU_Sinai_Feb_23_200']
CU_MODELS = ['CU_Feb_11_200', 'CU_Sinai_Feb_23_200']
YALE_MODELS = ['Yale_Feb_15_200']


ARGS = parser.parse_args()


class PrecisionConfig(object):
	RESULTS_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'results')
	REPLACE_DIR = os.path.join('/hdd', 'ID_net', 'results')
	DATA_CONDITIONS = '/data/recurrence_seq_lstm/data_conditions/'
	SCRIPT_DIR = '/home/wanglab/Desktop/recurrence_seq_lstm/IntelligentDiagnosis/'
	YALE_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'yale_testing_data')
	GEIS_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'geis_testing_data')
	CU_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'cu_testing_data')
	SINAI_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'sinai_testing_data')
	RESULTS_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'results')
	OMEN = ''

class OmenConfig(object):
	RESULTS_DIR = os.path.join('/hdd', 'ID_net', 'results')
	REPLACE_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'results')
	DATA_CONDITIONS = os.path.join('/hdd', 'ID_net', 'data_conditions')
	SCRIPT_DIR = '/hdd/ID_net/IntelligentDiagnosis/'
	YALE_TEST_DIR = os.path.join('/hdd', 'ID_net', 'data_conditions', 'yale_testing_data')
	GEIS_TEST_DIR = os.path.join('/hdd', 'ID_net', 'data_conditions', 'geis_testing_data')
	CU_TEST_DIR = os.path.join('/hdd', 'ID_net', 'data_conditions', 'cu_testing_data')
	SINAI_TEST_DIR = os.path.join('/hdd', 'ID_net', 'data_conditions', 'sinai_testing_data')
	RESULTS_DIR = os.path.join('/hdd', 'ID_net', 'results')
	OMEN = '--omen_run=True '

def test_model(base, model, test_dir, name, config):
	if name in model:
		return

	data_t = '--recur_data_path=' + test_dir + ' --nonrecur_data_path=' + test_dir
	model_t = ' --model_path=' + os.path.join(config.RESULTS_DIR, model)
	results_base = model + '___' + name + '_' + datetime.now().strftime('%m_%d')
	results_name = results_base
	results_t = ' --results_prepend=' + results_name
	results_path = os.path.join(config.RESULTS_DIR, results_name)
	i = 1
	while(os.path.exists(results_path)):
		i += 1
		results_name = results_base + "_%i" % i
		results_path = os.path.join(config.RESULTS_DIR, results_name)
		results_t = ' --results_prepend=' + results_name

	run_call = base + data_t + model_t + results_t
	subprocess.check_call(run_call, shell=True)

	maj_call = 'python3 ' + os.path.join(config.SCRIPT_DIR + 'majority_vote.py') + ' --base_path=' + os.path.join(config.RESULTS_DIR, results_name)

	subprocess.check_call(maj_call, shell=True)

	return

def fix_checkpoints(config):
	# TBD - Loop over all available models and change checkpoint path to config's path.
	for root, dirs, files in os.walk(config.RESULTS_DIR):
		for d in dirs:
			replacing = False
			checkpoint_file = os.path.join(root, d, 'checkpoint')
			writing = []
			if os.path.exists(checkpoint_file):
				with open(checkpoint_file, 'r') as f:
					for line in f:
						if config.RESULTS_DIR in line:
							continue
						else:
							replacing = True
							writing.append(line.replace(config.REPLACE_DIR, config.RESULTS_DIR))
				if replacing:
					cprint('Replacing checkpoints in ' + d)
					with open(checkpoint_file, 'w+') as f:
						for line in writing:
							f.write(line)

							
		return
	return

def fix_checkpoint(model, config):
	checkpoint_file = os.path.join(config.RESULTS_DIR, model, 'checkpoint')
	writing = []
	if os.path.exists(checkpoint_file):
		print('Verifying checkpoints for ' + model + '...')
		with open(checkpoint_file, 'r') as f:
			for line in f:
				if config.REPLACE_DIR in line:
					writing.append(line.replace(config.REPLACE_DIR, config.RESULTS_DIR))
			if writing:
				cprint('Replacing checkpoints in ' + d)
				with open(checkpoint_file, 'w+') as f:
					for line in writing:
						f.write(line)
		return 0
	else:
		return 1

def main():

	tower_name = os.uname().nodename
	if 'OMEN' in tower_name:
		ARGS.omen = True

	if ARGS.omen:
		config = OmenConfig()
	else:
		config = PrecisionConfig()
	
	# fix_checkpoints(config)

	fc_base = 'python3 ' + os.path.join(config.SCRIPT_DIR, 'recurrence_lstm_FC_features.py') + ' --config=test ' + config.OMEN
	dnn_base = 'python3 ' + os.path.join(config.SCRIPT_DIR, 'recurrence_lstm_features.py') + ' --config=test ' + config.OMEN
	models = []
	if ARGS.cumc:
		models = models + CU_MODELS

	if ARGS.fullyc:
		fc_models = []
		for model in models:
			if 'FC' in model:
				fc_models.append(model)
		models = fc_models

	for model in models:
		exit = fix_checkpoint(model, config)
		if exit:
			cprint('No valid checkpoint file for ' + model)
			continue

		if 'FC' in model:
			base = fc_base
		else:
			base = dnn_base

		if 	ARGS.yale_test:
			test_model(base, model, config.YALE_TEST_DIR, 'Yale', config)
		if ARGS.sinai_test:
			test_model(base, model, config.SINAI_TEST_DIR, 'Sinai', config)
		if ARGS.geis_test:
			test_model(base, model, config.GEIS_TEST_DIR, 'Geis', config)
		if ARGS.cu_test:
			test_model(base, model, config.CU_TEST_DIR, 'CU', config)


# Main body
if __name__ == '__main__':
	main()