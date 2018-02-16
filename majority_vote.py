#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import csv
import pdb
import collections
import numpy as np
#import os

# Global variables

# Function declarations

def average_value(data, value):
	return sum(data[value]) / len(data[value])

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

def initialize_subject(data, row):
	data["id"]=row[0]
	data["output"]=[row[1]]
	data["labels"]=[row[2]]
	data["unscaled_nr"]=[float(row[3])]
	data["unscaled_rec"]=[float(row[4])]
	data["scaled_nr"]=[float(row[5])]
	data["scaled_rec"]=[float(row[6])]

def add_data_to_existing_subect(data, row):
	data["output"].append(row[1])
	data["labels"].append(row[2])
	data["unscaled_nr"].append(float(row[3]))
	data["unscaled_rec"].append(float(row[4]))
	data["scaled_nr"].append(float(row[5]))
	data["scaled_rec"].append(float(row[6]))

def main():
    args = sys.argv[1:]
    filename = args[0]
    subject_dict={}

    with open(filename, newline="") as csvfile:
    	csvreader = csv.reader(csvfile, delimiter=",")
    	header = next(csvreader) # Discard header line
    	for row in csvreader:
    		if row[0] not in subject_dict:
    			subject_dict[row[0]]={}
    			initialize_subject(subject_dict[row[0]], row)
    		else:
    			add_data_to_existing_subect(subject_dict[row[0]], row)
    for id in subject_dict:
    	print("\n******  " + id + "  ******")
    	print("Label: " + subject_dict[id]["labels"][0])
    	c1 = collections.Counter(subject_dict[id]["output"]).most_common(1)
    	print("Network vote: %i, %i%% " % (int(c1[0][0]), int((c1[0][1])/len(subject_dict[id]["output"])*100)))
    	unscaled_nr = average_value(subject_dict[id], "unscaled_nr")
    	unscaled_rec = average_value(subject_dict[id], "unscaled_rec")
    	scaled_nr = average_value(subject_dict[id], "scaled_nr")
    	scaled_rec = average_value(subject_dict[id], "scaled_rec")
    	smax1 = softmax([unscaled_nr, unscaled_rec])
    	smax2 = softmax([scaled_nr, scaled_rec])

    	print("Unscaled average: %.3f, %.3f" % (unscaled_nr, unscaled_rec) )
    	print("Softmax of unscaled average: " + str(smax1))
    	print("Scaled Average: %.3f, %.3f" % (scaled_nr, scaled_rec))
    	    	
    if not args:
        print('usage: enter location of csv file as first and only parameter ')
        sys.exit(1)


# Main body
if __name__ == '__main__':
	main()