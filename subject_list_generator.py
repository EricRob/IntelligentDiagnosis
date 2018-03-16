#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import csv
import os
import pdb
import random
import numpy as np
from termcolor import cprint
from tensorflow import flags


# Global variables
flags.DEFINE_string("subject_list", "verified_images.csv", "Master list of subjects from which to create per-mode subject lists")
flags.DEFINE_string("base_path", None, "Location of subject list and where the current testing condition files will be created")
flags.DEFINE_integer("conditions", 20, "Number of patient sets to create")
flags.DEFINE_string("condition_name", "leave_one_out_conditions", "Name of current testing condition")
flags.DEFINE_integer("test_subjects", 1, "Number of subjects to include in test conditions")
FLAGS = flags.FLAGS
# Class declarations

# Function declarations
def give_string_label(label):
    if label == '1':
        return "RECURRENT"
    else:
        return "NONRECURRENT"

def initialize_subject(subject, row):
	subject["id"] = row[0]
	subject["images"] = [row[1]]
	subject["label"] = give_string_label(row[2])

def add_to_existing_subject(subject, row):
	label = give_string_label(row[2])
	if not label == subject["label"]:
		cprint("LABEL MISMATCH FOR %s, images %s " % (subject["id"], row[1]), 'white', 'on_red', attrs=['bold'])
		return
	else:
		subject["images"].append(row[1])
	return

def generate_label_dicts(subject_dict):
	recurrence_dict = {}
	nonrecurrence_dict = {}
	
	for subject in subject_dict:
		if subject_dict[subject]["label"] == 'RECURRENT':
			recurrence_dict[subject] = subject_dict[subject]["images"]
		elif subject_dict[subject]["label"] == 'NONRECURRENT':
			nonrecurrence_dict[subject] = subject_dict[subject]["images"]
	return recurrence_dict, nonrecurrence_dict

def sort_recurrence_from_nonrecurrence(train_list, test_list, subject_dict):
	recur_train = []
	nonrecur_train = []
	recur_test = []
	nonrecur_test = []

	for subj in train_list:
		if subject_dict[subj]['label'] == 'NONRECURRENT':
			nonrecur_train.append(subj)
		else:
			recur_train.append(subj)

	for subj in test_list:
		if subject_dict[subj]['label'] == 'NONRECURRENT':
			nonrecur_test.append(subj)
		else:
			recur_test.append(subj)
	return recur_train, recur_test, nonrecur_train, nonrecur_test

def one_third_subject_lists(folder_path, train_list, test_list, subject_dict):
	os.makedirs(folder_path, exist_ok=True)
	recur_train, recur_test, nonrecur_train, nonrecur_test = sort_recurrence_from_nonrecurrence(train_list, test_list, subject_dict)
	create_mode_lists_per_label(subject_dict, recur_test, recur_train, folder_path, 'recurrence')
	create_mode_lists_per_label(subject_dict, nonrecur_test, nonrecur_train, folder_path, 'nonrecurrence')

def generate_subject_lists(subject_dict):
	recurrence_dict, nonrecurrence_dict = generate_label_dicts(subject_dict)
	
	# subject_list = random.shuffle(list(subject_dict))
	
	subject_list = list(subject_dict)
	random.shuffle(subject_list)

	conditions_folder_path = os.path.join(FLAGS.base_path, FLAGS.condition_name)
	os.makedirs(conditions_folder_path, exist_ok=True)

	test_subject_count = len(subject_list) // 3
	test_subjects = list(subject_list)

	condition_one_train = list(subject_list)
	condition_two_train = list(subject_list)
	condition_three_train = list(subject_list)

	condition_one_test = []
	condition_two_test = []
	condition_three_test = []


	for _ in np.arange(test_subject_count):
		subject_one = test_subjects.pop()
		condition_one_test.append(subject_one)
		condition_one_train.remove(subject_one)

		subject_two = test_subjects.pop()
		condition_two_test.append(subject_two)
		condition_two_train.remove(subject_two)

	for subject_three in test_subjects:
		condition_three_train.remove(subject_three)


	folder_path_one = os.path.join(conditions_folder_path, "001_condition")
	one_third_subject_lists(folder_path_one, condition_one_train, condition_one_test, subject_dict)
	
	folder_path_two = os.path.join(conditions_folder_path, "002_condition")
	one_third_subject_lists(folder_path_two, condition_two_train, condition_two_test, subject_dict)

	folder_path_three = os.path.join(conditions_folder_path, "003_condition")
	one_third_subject_lists(folder_path_three, condition_three_train, test_subjects, subject_dict)


	# recur_subject_list = list(recurrence_dict.keys())
	# nonrecur_subject_list = list(nonrecurrence_dict.keys())
	# for i in range(FLAGS.conditions):
	# 	folder_path = os.path.join(conditions_folder_path, '{0:03d}'.format(i+1) + "_condition")
	# 	os.makedirs(folder_path, exist_ok=True)
	# 	recur_test_subjects = []
	# 	nonrecur_test_subjects = []

	# 	while(len(recur_test_subjects) < (FLAGS.test_subjects / 2)):
	# 		rand_recur_subject = random.choice(recur_subject_list)
	# 		if rand_recur_subject not in recur_test_subjects:
	# 			recur_test_subjects.append(rand_recur_subject)

	# 	while(len(nonrecur_test_subjects) < (FLAGS.test_subjects / 2)):
	# 		rand_nonrecur_subject = random.choice(nonrecur_subject_list)
	# 		if rand_nonrecur_subject not in nonrecur_test_subjects:
	# 			nonrecur_test_subjects.append(rand_nonrecur_subject)
		
	# 	recur_test_removed = list(recur_subject_list)
	# 	for test_subject in recur_test_subjects:
	# 		recur_test_removed.remove(test_subject)

	# 	nonrecur_test_removed = list(nonrecur_subject_list)
	# 	for test_subject in nonrecur_test_subjects:
	# 		nonrecur_test_removed.remove(test_subject)

	# 	create_mode_lists_per_label(subject_dict, recur_test_subjects, recur_test_removed, folder_path, 'recurrence')
	# 	create_mode_lists_per_label(subject_dict, nonrecur_test_subjects, nonrecur_test_removed, folder_path,'nonrecurrence')

def create_mode_lists_per_label(subject_dict, test_subjects, subject_list, folder_path, label):
	test_file = open(os.path.join(folder_path, label + "_test_subjects.txt"), "wt")
	valid_file = open(os.path.join(folder_path, label + "_valid_subjects.txt"), "wt")
	train_file = open(os.path.join(folder_path, label + "_train_subjects.txt"), "wt")
	write_test_file(subject_dict, test_subjects, test_file)
	write_train_and_valid_files(subject_dict, subject_list, valid_file, train_file)

def write_test_file(subject_dict, test_subjects, test_file):
	for subject in test_subjects:
		for image in subject_dict[subject]["images"]:
			test_file.write(image + '\n')
	test_file.close()

def write_train_and_valid_files(subject_dict, subject_list, valid_file, train_file):
	for subject in subject_list:
		image_list = list(subject_dict[subject]["images"])
		random.shuffle(image_list)
		if len(image_list) == 1:
			train_file.write(image_list[0] + '\n')
		elif len(image_list) == 2:
			train_file.write(image_list[0] + '\n')
			valid_file.write(image_list[1] + '\n')
		else:
			num_valid_subjects = len(image_list) // 3
			for x in range(num_valid_subjects):
				valid_file.write(image_list.pop(0) + '\n')
			for image in image_list:
				train_file.write(image + '\n')
	train_file.close()
	valid_file.close()

def main():
    base_path = FLAGS.base_path
    filename = FLAGS.subject_list
    csv_file = os.path.join(base_path, filename)

    subject_dict = {}
    with open(csv_file, newline="") as csvfile:
    	csvreader = csv.reader(csvfile, delimiter=",")
    	header = next(csvreader)
    	for row in csvreader:
    		if row[0] not in subject_dict:
    			subject_dict[row[0]] = {}
    			initialize_subject(subject_dict[row[0]], row)
    		else:
    			add_to_existing_subject(subject_dict[row[0]], row)
    # recurrent_count = 0
    # nonrecurrent_count = 0
    # for key in subject_dict.keys():
    # 	if subject_dict[key]["label"] == 'RECURRENT':
    # 		recurrent_count +=1
    # 	elif subject_dict[key]["label"] == 'NONRECURRENT':
    # 		nonrecurrent_count +=1
    generate_subject_lists(subject_dict)

# Main body
if __name__ == '__main__':
	main()