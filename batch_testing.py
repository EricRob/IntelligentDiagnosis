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


CU_MODELS = ['CU_Sinai_Feb_23_200']
YALE_MODELS = ['Yale_Feb_15_200']

YALE_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'yale_testing_data')
GEIS_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'geis_testing_data')
CU_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'cu_testing_data')
SINAI_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'sinai_testing_data')
RESULTS_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'results')


ARGS = parser.parse_args()


class PrecisionConfig(object):
	RESULTS_DIR = '/data/recurrence_seq_lstm/results/'
	DATA_CONDITIONS = '/data/recurrence_seq_lstm/data_conditions/'
	SCRIPT_DIR = '/home/wanglab/Desktop/recurrence_seq_lstm/IntelligentDiagnosis/'
	YALE_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'yale_testing_data')
	GEIS_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'geis_testing_data')
	CU_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'cu_testing_data')
	SINAI_TEST_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions', 'sinai_testing_data')
	RESULTS_DIR = os.path.join('/data', 'recurrence_seq_lstm', 'results')

class OmenConfig(object):
	RESULTS_DIR = os.path.join('/hdd', 'ID_net', 'results')
	DATA_CONDITIONS = os.path.join('/hdd', 'ID_net', 'data_conditions')
	SCRIPT_DIR = '/hdd/ID_net/IntelligentDiagnosis/'
	YALE_TEST_DIR = os.path.join('/hdd', 'ID_net', 'data_conditions', 'yale_testing_data')
	GEIS_TEST_DIR = os.path.join('/hdd', 'ID_net', 'data_conditions', 'geis_testing_data')
	CU_TEST_DIR = os.path.join('/hdd', 'ID_net', 'data_conditions', 'cu_testing_data')
	SINAI_TEST_DIR = os.path.join('/hdd', 'ID_net', 'data_conditions', 'sinai_testing_data')
	RESULTS_DIR = os.path.join('/hdd', 'ID_net', 'results')

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
	pdb.set_trace()
	subprocess.check_call(run_call, shell=True)

	maj_call = 'python3 ' + os.path.join(config.SCRIPT_DIR + 'majority_vote.py') + ' --base_path=' + results

	subprocess.check_call(maj_call, shell=True)

	return


def main():
	if ARGS.omen:
		config = OmenConfig()
	else:
		config = PrecisionConfig()

	base = 'python3 ' + os.path.join(config.SCRIPT_DIR, 'recurrence_lstm_features.py') + ' --config=test ' 
	models = []
	if ARGS.cumc:
		models = models + CU_MODELS

	if 	ARGS.yale_test:
		for model in models:
			test_model(base, model, config.YALE_TEST_DIR, 'Yale', config)
	if ARGS.sinai_test:
		for model in models:
			test_model(base, model, config.SINAI_TEST_DIR, 'Sinai', config)
	if ARGS.geis_test:
		for model in models:
			test_model(base, model, config.GEIS_TEST_DIR, 'Geis', config)
	if ARGS.cu_test:
		for model in models:
			test_model(base, model, config.CU_TEST_DIR, 'CU', config)


# Main body
if __name__ == '__main__':
	main()