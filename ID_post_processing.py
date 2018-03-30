#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import os
import csv
import matplotlib.pyplot as plt
from pylab import subplot, plot, subplots_adjust
import numpy as np
import pdb
from termcolor import cprint
from tensorflow import flags
from shutil import move

import majority_vote as voter

# Global variables
flags.DEFINE_string("results_folder", "../results/","Results folder containing all results and voting files")
flags.DEFINE_string('vote_file', 'voting_file.csv', "voting file name")
flags.DEFINE_bool("overwrite",False,"Overwrite existing summary plots")
FLAGS = flags.FLAGS

# Class declarations

# Function declarations

def has_results_files(files):
	return ('train_results.txt' in files) and ('valid_results.txt' in files) and ('test_results.txt' in files)

def create_results_graphs(parent):
	# results_files = [os.path.join(parent,'train_results.txt'), os.path.join(parent,'valid_results.txt'), os.path.join(parent,'test_results.txt')]

	train_dict = fill_results_dict(os.path.join(parent,'train_results.txt'))
	valid_dict = fill_results_dict(os.path.join(parent,'valid_results.txt'))
	test_dict = fill_results_dict(os.path.join(parent,'test_results.txt'))

	epochs = np.arange(1,len(train_dict['loss'])+1)

	figure = plt.figure()

	create_subplot(figure, train_dict, valid_dict, test_dict, epochs, "sensitivity", 231)
	create_subplot(figure, train_dict, valid_dict, test_dict, epochs, "specificity", 232)
	create_subplot(figure, train_dict, valid_dict, test_dict, epochs, "accuracy", 233)
	create_subplot(figure, train_dict, valid_dict, test_dict, epochs, "loss", 234)

	return figure

def fill_results_dict(filename):
	results = dict()
	
	results["sensitivity"] = []
	results["specificity"] = []
	results["FDR"] = []
	results['FOR'] = []
	results['accuracy'] = []
	results['loss'] = []
	results['positive_cases'] = 0
	results['negative_cases'] = 0

	with open(filename, newline="") as csvfile:
		csvreader = csv.reader(csvfile, delimiter=",")
		for row in csvreader:
			results["sensitivity"].append(row[0])
			results["specificity"].append(row[1])
			results["FDR"].append(row[2])
			results['FOR'].append(row[3])
			results['accuracy'].append(row[4])
			results['positive_cases'] = row[5]
			results['negative_cases'] = row[6]
			results['loss'].append(row[7])

	return results

def create_subplot(figure, train_dict, valid_dict, test_dict, epochs, value, plot_loc):
	subplot = figure.add_subplot(plot_loc)

	train = np.array(train_dict[value], dtype=np.float32)
	valid = np.array(valid_dict[value], dtype=np.float32)
	test = np.array(test_dict[value], dtype=np.float32)
	
	subplot.plot(epochs, train, label="train", color='b')
	subplot.plot(epochs, valid, label="valid", color='r')
	subplot.plot(epochs, test, label="test", color='g')

	if not value == 'loss':
		subplot.set_ylim([0, 1])
		subplot.plot(epochs, np.repeat(0.5,train.shape), color='black', ls='--')
	if value == 'accuracy':
		subplot.legend(['train', 'valid', 'test'], loc='lower right')

	subplot.set_title(value.title())

def create_majority_votes(parent, figure):

	image_dict, subject_dict = voter.majority_vote(parent, voting_filename=FLAGS.vote_file)
	fpr, tpr, thresholds, roc_auc = voter.create_overall_roc_curve(image_dict)
	roc = figure.add_subplot(235)
	roc.plot(1-fpr, tpr, color='darkorange', lw=2)
	roc.plot([0, 1], [1,0], color='black', lw=1, linestyle='--')
	roc.set_title('ROC')
	legend = 'ROC curve (area = %0.2f)' % roc_auc
	roc.legend([legend])
	
	subject_data = voter.analysis_per_subject(subject_dict, image_dict)

	bar_graph = figure.add_subplot(236)
	index = 0
	half_index = len(subject_data.keys()) // 2
	# if len(subject_data.keys()) % 2 == 1:
	# 	half_index += 1

	names =[]
	votes = []
	recur_votes = []
	nonrecur_names = []
	nonrecur_votes = []

	recur_dict = dict()
	nonrecur_dict = dict()

	for subject in subject_data:
		if subject_data[subject]["truth_label"] == 'RECURRENT':
			recur_dict[subject] = subject_data[subject]
		else:
			nonrecur_dict[subject] = subject_data[subject]
	for subject in sorted(recur_dict):
		if not recur_dict[subject]["truth_label"] == recur_dict[subject]["net_label"]:
			recur_dict[subject]["vote"] = 1 - recur_dict[subject]["vote"]
		rec_legend = bar_graph.bar(index, recur_dict[subject]["vote"], color='orange', label="Recurrent")
		index += 1
		names.append(subject)
		votes.append(recur_dict[subject]["vote"])

	for subject in sorted(nonrecur_dict):
		if not nonrecur_dict[subject]["truth_label"] == nonrecur_dict[subject]["net_label"]:
			nonrecur_dict[subject]["vote"] = 1 - nonrecur_dict[subject]["vote"]
		nonrec_legend = bar_graph.bar(index, nonrecur_dict[subject]["vote"], color='blue', label="Nonrecurrent")
		index += 1
		names.append(subject)
		votes.append(nonrecur_dict[subject]["vote"])


	# for subject in sorted(subject_data):
	# 	if not subject_data[subject]["truth_label"] == subject_data[subject]["net_label"]:
	# 		subject_data[subject]["vote"] = 1 - subject_data[subject]["vote"]
		
	# 	if subject_data[subject]["truth_label"] == 'RECURRENT':
	# 		bar_graph.bar(index, subject_data[subject]["vote"], color='orange')
	# 		index += 1
	# 		recur_names.append(subject)
	# 		recur_votes.append(subject_data[subject]["vote"])

	# 	elif subject_data[subject]["truth_label"] == 'NONRECURRENT':
	# 		bar_graph.bar(half_index, subject_data[subject]["vote"], color='blue')
	# 		nonrecur_names.append(subject)
	# 		nonrecur_votes.append(subject_data[subject]["vote"])
	# 		half_index += 1
	# names = recur_names + nonrecur_names
	# votes = recur_votes + nonrecur_votes
	bar_graph.set_xticks(np.arange(len(subject_data.keys())))
	bar_graph.set_xticklabels(names, rotation=90)
	bar_graph.set_ylim([0, 1])
	bar_graph.axhline(0.5, color='gray')
	bar_graph.set_title("Majority Vote")
	percent_vote = ["{:.4f}".format(vote) for vote in votes]
	bar_graph.legend([rec_legend, nonrec_legend], ["Recurrent", "Nonrecurrent"])
	return

def main():
	count = 0
	for root, dirs, _ in os.walk(FLAGS.results_folder):
		for folder in dirs:
			files = os.listdir(os.path.join(root,folder))
			
			if not (has_results_files(files) and (FLAGS.vote_file in files)):
				pass
			else:
				results_name = os.path.join(root, folder, os.path.splitext(FLAGS.vote_file)[0] + "_results_summary.jpg")
				if os.path.exists(results_name) and not FLAGS.overwrite:
					# print("skipping " + folder)
					pass
				else:
					if has_results_files(files):
						figure = create_results_graphs(os.path.join(root, folder))

					if (FLAGS.vote_file in files) and (os.stat(os.path.join(root,folder,FLAGS.vote_file)).st_size > 0):
						create_majority_votes(os.path.join(root, folder), figure)
					
					figure.suptitle(folder, fontsize=32)
					figure.tight_layout(rect=[0, 0.03, 1, 0.95])
					figure.set_size_inches(15, 12)
					cprint("Summarizing and saving " + folder, 'magenta', 'on_white')
					figure.savefig(results_name)
					plt.close()
					count +=1
	    		# results_files = ['train_results.txt', 'valid_results.txt', 'test_results.txt', 'voting_file.csv']
	    		# move_path = os.path.join(root, folder,"raw_data")
	    		# os.makedirs(move_path, exist_ok=True)
	    		# for f in results_files:
	    		# 	if os.path.exists(os.path.join(root, folder,f)):
	    		# 		move(os.path.join(root, folder,f), os.path.join(move_path,f))
	if count == 0:
		cprint("¸¸♬·¯·♩¸¸♪·¯·♫¸¸No new results to summarize¸¸♬·¯·♩¸¸♪·¯·♫¸¸", 'yellow', attrs=['bold'])

# Main body
if __name__ == '__main__':
	main()