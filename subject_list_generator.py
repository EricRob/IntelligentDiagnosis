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
from termcolor import cprint
from tensorflow import flags


# Global variables
flags.DEFINE_string("subject_list", "verified_images.csv", "Master list of subjects from which to create per-mode subject lists")
flags.DEFINE_string("base_path", None, "Location of current testing condition files")
flags.DEFINE_integer("conditions", 20, "Number of patient sets to create")
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

def generate_subject_lists(subject_dict):
	recurrence_dict, nonrecurrence_dict = generate_label_dicts(subject_dict)
	
	recur_subject_list = list(recurrence_dict.keys())
	nonrecur_subject_list = list(nonrecurrence_dict.keys())

	# if len(recur_subject_list) > nonrecur_subject_list:
	# 	large_label_list = recur_subject_list
	# 	small_label_list = nonrecur_subject_list
	# else:
	# 	large_label_list = nonrecur_subject_list
	# 	small_label_list = recur_subject_list
	conditions_folder_path = os.path.join(FLAGS.base_path, "conditions")
	os.makedirs(conditions_folder_path, exist_ok=True)

	for i in range(FLAGS.conditions):
		folder_path = os.path.join(conditions_folder_path, str(i+1) + "_condition")
		os.makedirs(folder_path, exist_ok=True)
		recur_test_subject = random.choice(recur_subject_list)
		nonrecur_test_subject = random.choice(nonrecur_subject_list)
		
		recur_test_removed = list(recur_subject_list)
		recur_test_removed.remove(recur_test_subject)
		nonrecur_test_removed = list(nonrecur_subject_list)
		nonrecur_test_removed.remove(nonrecur_test_subject)

		create_mode_lists_per_label(subject_dict, recur_test_subject, recur_test_removed, folder_path, 'recurrence')
		create_mode_lists_per_label(subject_dict, nonrecur_test_subject, nonrecur_test_removed, folder_path,'nonrecurrence')

def create_mode_lists_per_label(subject_dict, test_subject, subject_list, folder_path, label):
	test_file = open(os.path.join(folder_path, label + "_test_subjects.txt"), "wt")
	valid_file = open(os.path.join(folder_path, label + "_valid_subjects.txt"), "wt")
	train_file = open(os.path.join(folder_path, label + "_train_subjects.txt"), "wt")
	write_test_file(subject_dict, test_subject, test_file)
	write_train_and_valid_files(subject_dict, subject_list, valid_file, train_file)

def write_test_file(subject_dict, test_subject, test_file):
	for image in subject_dict[test_subject]["images"]:
		test_file.write(image + '\n')
	test_file.close()

def write_train_and_valid_files(subject_dict, subject_list, valid_file, train_file):
	for subject in subject_list:
		image_list = list(subject_dict[subject]["images"])
		if len(image_list) == 1:
			train_file.write(image_list[0] + '\n')
		elif len(image_list) == 2:
			train_file.write(image_list[0] + '\n')
			valid_file.write(image_list[1] + '\n')
		else:
			num_valid_subjects = len(image_list) // 3
			random.shuffle(image_list)
			for x in range(num_valid_subjects):
				valid_file.write(image_list.pop(0) + '\n')
			for image in image_list:
				train_file.write(image + '\n')
	train_file.close()
	valid_file.close()

def main():
    base_path = FLAGS.base_path
    filename = FLAGS.subject_list
    os.makedirs(os.path.join(base_path, "per_mode_subjects"), exist_ok=True)
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
    recurrent_count = 0
    nonrecurrent_count = 0
    for key in subject_dict.keys():
    	if subject_dict[key]["label"] == 'RECURRENT':
    		recurrent_count +=1
    	elif subject_dict[key]["label"] == 'NONRECURRENT':
    		nonrecurrent_count +=1
    generate_subject_lists(subject_dict)

# Main body
if __name__ == '__main__':
	main()