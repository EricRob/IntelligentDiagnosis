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
flags.DEFINE_string("base_path", "/data/recurrence_seq_lstm/data_conditions/", "Location of subject list and where the current testing condition files will be created")
flags.DEFINE_integer("conditions", 6, "Number of patient sets to create")
flags.DEFINE_string("condition_name", None, "Name of current testing condition")
flags.DEFINE_bool("image_processor", False, "List generator is being called rom image_processor.py")

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

def write_subject_lists(folder_path, train_list, test_list, subject_dict, condition_dict, i):
	os.makedirs(folder_path, exist_ok=True)
	recur_train, recur_test, nonrecur_train, nonrecur_test = sort_recurrence_from_nonrecurrence(train_list, test_list, subject_dict)
	condition_dict[i+1]["recur_test"] = recur_test
	condition_dict[i+1]["recur_test_count"] = len(recur_test)
	condition_dict[i+1]["nonrecur_test"] = nonrecur_test
	condition_dict[i+1]['nonrecur_test_count'] = len(nonrecur_test)

	create_mode_lists_per_label(subject_dict, recur_test, recur_train, folder_path, 'recurrence')
	create_mode_lists_per_label(subject_dict, nonrecur_test, nonrecur_train, folder_path, 'nonrecurrence')

def make_single_condition_list(subject_list, all_test_subjects, test_count, i, cond_path, subject_dict, condition_dict):
	train_list = list(subject_list)
	test_list = []

	for _ in np.arange(test_count):
		subject = all_test_subjects.pop()
		test_list.append(subject)
		train_list.remove(subject)
	folder_path = os.path.join(cond_path, '{0:03d}'.format(i+1) + "_condition")
	write_subject_lists(folder_path, train_list, test_list, subject_dict, condition_dict, i)
	return all_test_subjects

def write_conditions_csv(condition_dict, conditions_folder_path):
	with open(os.path.join(conditions_folder_path,FLAGS.condition_name + '_tests.csv'), 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(['subject', 'label', 'condition'])
		for test_cond in condition_dict:
			for recur in condition_dict[test_cond]["recur_test"]:
				csvwriter.writerow([recur, '1', str(test_cond)])
			for nonrecur in condition_dict[test_cond]["nonrecur_test"]:
				csvwriter.writerow([nonrecur, '0', str(test_cond)])

def generate_subject_lists(subject_dict):
	# recurrence_dict, nonrecurrence_dict = generate_label_dicts(subject_dict)
	
	# subject_list = random.shuffle(list(subject_dict))
	
	subject_list = list(subject_dict)
	if not FLAGS.condition_name:
		FLAGS.condition_name = os.path.splitext(FLAGS.subject_list)[0]
	conditions_folder_path = os.path.join(FLAGS.base_path, FLAGS.condition_name)
	os.makedirs(conditions_folder_path, exist_ok=True)
	test_subject_count = len(subject_list) // FLAGS.conditions
	condition_dict = dict()
	for i in np.arange(FLAGS.conditions):
		condition_dict[i+1] = dict()
	adequate_distribution = False
	count = 0
	while(not adequate_distribution):
		count += 1
		adequate_distribution = True
		random.shuffle(subject_list)
		test_subjects = list(subject_list)

		for i in np.arange(FLAGS.conditions-1):
			test_subjects = make_single_condition_list(subject_list, test_subjects, test_subject_count, i, conditions_folder_path, subject_dict, condition_dict)
		final_condition_train = list(subject_list)

		for subject in test_subjects:
			final_condition_train.remove(subject)
		final_folder_path = os.path.join(conditions_folder_path, '{0:03d}'.format(FLAGS.conditions) + "_condition")
		write_subject_lists(final_folder_path, final_condition_train, test_subjects, subject_dict, condition_dict, FLAGS.conditions-1)
		
		adequate_distribution = subject_condition_distributions(condition_dict)
		
		# Make sure all tests contain at least one subject. If not, reshuffle subject list and start over.
		for condition in condition_dict:
			if not condition_dict[condition]["recur_test"]:
				adequate_distribution = False
				break
			if not condition_dict[condition]["nonrecur_test"]:
				adequate_distribution = False
				break
		if adequate_distribution:
			cprint("Attempt " + str(count) + " -- Success", 'white', 'on_green')
		else:
			cprint("Attempt " + str(count) + " -- Unbalanced", 'white', 'on_red')

		write_conditions_csv(condition_dict, conditions_folder_path)

def create_mode_lists_per_label(subject_dict, test_subjects, subject_list, folder_path, label):
	test_file = open(os.path.join(folder_path, label + "_test_subjects.txt"), "wt")
	valid_file = open(os.path.join(folder_path, label + "_valid_subjects.txt"), "wt")
	train_file = open(os.path.join(folder_path, label + "_train_subjects.txt"), "wt")
	write_test_file(subject_dict, test_subjects, test_file)
	write_train_and_valid_files(subject_dict, subject_list, valid_file, train_file)

def subject_condition_distributions(condition_dict):
	recur_count = 0
	nonrecur_count = 0
	one_off = False

	for cond in condition_dict:
		recur_count += len(condition_dict[cond]['recur_test'])
		nonrecur_count += len(condition_dict[cond]['nonrecur_test'])
	recur_portion = round(recur_count / (recur_count + nonrecur_count), 1)
	for cond in condition_dict:
		cond_portion = round(condition_dict[cond]['recur_test_count'] / (condition_dict[cond]['nonrecur_test_count'] + condition_dict[cond]['recur_test_count']), 1)
		if (abs(round(cond_portion - recur_portion, 1)) > 0.1):
			if one_off:
				return False
			else:
				one_off = True
	return True

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
		elif len(image_list) == 3:
			train_file.write(image_list[0] + '\n')
			valid_file.write(image_list[1] + '\n')
			train_file.write(image_list[2] + '\n')
		else:
			num_valid_subjects = len(image_list) // 4
			for x in range(num_valid_subjects):
				valid_file.write(image_list.pop(0) + '\n')
			for image in image_list:
				train_file.write(image + '\n')
	train_file.close()
	valid_file.close()

def write_lists(new_condition_name, verified_csv, c_val_folds, base_path):
	FLAGS.image_processor = True
	FLAGS.condition_name = new_condition_name
	FLAGS.subject_list = verified_csv
	FLAGS.conditions = c_val_folds
	FLAGS.base_path = base_path
	main()
	return True

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