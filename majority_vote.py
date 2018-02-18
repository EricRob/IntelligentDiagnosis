#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import os
import csv
import pdb
import collections
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint
from tensorflow import flags
from IPython import embed

# Global variables
flags.DEFINE_string("info","subject","Display per subject or per image info")
FLAGS = flags.FLAGS

# Function declarations

def average_value(data, value):
	return sum(data[value]) / len(data[value])

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

def initialize_image(data, row):
    data["id"]=row[0]
    data["name"]=row[1]
    data["output"]=[row[2]]
    data["labels"]=[row[3]]
    data["unscaled_nr"]=[float(row[4])]
    data["unscaled_rec"]=[float(row[5])]
    data["scaled_nr"]=[float(row[6])]
    data["scaled_rec"]=[float(row[7])]
    data["coords"] = {}
    coord_array = create_coord_list(row[8])
    for coord in coord_array:
        data["coords"][coord]={}
        data["coords"][coord]["nr"]=[float(row[4])]
        data["coords"][coord]["rec"]=[float(row[5])]

def initialize_subject(subj_dict, row):
    subj_dict[row[0]]=[row[1]]

def add_data_to_existing_image(data, row):
    data["output"].append(row[2])
    data["labels"].append(row[3])
    data["unscaled_nr"].append(float(row[4]))
    data["unscaled_rec"].append(float(row[5]))
    data["scaled_nr"].append(float(row[6]))
    data["scaled_rec"].append(float(row[7]))
    coord_array = create_coord_list(row[8])
    for coord in coord_array:
        if coord not in data["coords"]:
            data["coords"][coord]={}
            data["coords"][coord]["nr"] = [float(row[4])]
            data["coords"][coord]["rec"] = [float(row[5])]
        else:
            data["coords"][coord]["nr"].append(float(row[4]))
            data["coords"][coord]["rec"].append(float(row[5]))

def add_data_to_existing_subject(image_list, row):
    if row[1] not in image_list:
        image_list.append(row[1])

def give_string_label(label):
    if label == '1':
        return "RECURRENT"
    else:
        return "NONRECURRENT"

def create_histogram(subject_image_list, image_dict, net_label):
    scaled_hist_list = []
    unscaled_hist_list = []
    if net_label == "RECURRENT":
        scaled = "scaled_rec"
        unscaled = "unscaled_rec"
    else:
        scaled = "scaled_nr"
        unscaled = "unscaled_nr"
    for image in subject_image_list:
        scaled_hist_list = scaled_hist_list + image_dict[image][scaled]
        unscaled_hist_list = unscaled_hist_list + image_dict[image][unscaled]
    scaled_arr = np.array(scaled_hist_list)
    unscaled_arr = np.array(unscaled_hist_list)
    plt.hist(scaled_arr, bins='auto')
    plt.title("Scaled histogram of sequences from " + image)
    os.makedirs("histogram", exist_ok=True)
    plt.savefig(os.path.join("histogram", image_dict[image]["id"] + ".jpg"))
    plt.clf()

def create_heat_map(image_name):
    pass
def ops_within_patches(image):
    for coord in image:
        nr_sum = sum(image[coord]["nr"])
        rec_sum = sum(image[coord]["rec"])
        nr_len = len(image[coord]["nr"])
        rec_len = len(image[coord]["rec"])
        image[coord]["softmax"] = softmax([nr_sum, rec_sum])
        image[coord]["avg_softmax"] = softmax([nr_sum / nr_len, rec_sum / rec_len])
def create_coord_list(row):
    raw = row.split()
    coord_array = []
    for x, y in zip(raw[0::2], raw[1::2]):
        coord_array.append((int(x), int(y)))
    return coord_array

def across_image_avg(sums, lens):
    return sums / lens

def print_info_per_image(image_dict):
    for image_name in image_dict:
        c1 = collections.Counter(image_dict[image_name]["output"]).most_common(1)
        truth_label = give_string_label(image_dict[image_name]["labels"][0])
        network_label = give_string_label(c1[0][0])
        if truth_label == network_label:
            color = 'on_green'
        else:
            color = 'on_red'
        cprint("\n" + image_name, 'white', color, attrs=['bold'])
        print("Label: " + truth_label)
        print("Network majority vote: %s, %i%% (%i/%i) " % (network_label, int((c1[0][1])/len(image_dict[image_name]["output"])*100), int(c1[0][1]), len(image_dict[image_name]["output"])))
        unscaled_nr = average_value(image_dict[image_name], "unscaled_nr")
        unscaled_rec = average_value(image_dict[image_name], "unscaled_rec")
        scaled_nr = average_value(image_dict[image_name], "scaled_nr")
        scaled_rec = average_value(image_dict[image_name], "scaled_rec")
        smax1 = softmax([unscaled_nr, unscaled_rec])

        print("Unscaled average: %.3f, %.3f" % (unscaled_nr, unscaled_rec) )
        print("Softmax of unscaled average: " + str(smax1))
        print("Scaled Average: %.3f, %.3f" % (scaled_nr, scaled_rec))

def print_info_per_subject(subject_dict, image_dict):
    for subject in subject_dict:
            unscaled_nr_sum = 0
            unscaled_rec_sum = 0
            scaled_nr_sum = 0
            scaled_rec_sum = 0
            unscaled_nr_len = 0
            unscaled_rec_len = 0
            scaled_nr_len = 0
            scaled_rec_len = 0
            network_outputs = []

            for image in subject_dict[subject]:
                network_outputs = network_outputs + image_dict[image]["output"]
                unscaled_nr_sum += sum(image_dict[image]["unscaled_nr"])
                unscaled_nr_len += len(image_dict[image]["unscaled_nr"])
                unscaled_rec_sum += sum(image_dict[image]["unscaled_rec"])
                unscaled_rec_len += len(image_dict[image]["unscaled_rec"])
                scaled_rec_sum += sum(image_dict[image]["scaled_rec"])
                scaled_rec_len += len(image_dict[image]["scaled_rec"])
                scaled_nr_sum += sum(image_dict[image]["scaled_nr"])
                scaled_nr_len += len(image_dict[image]["scaled_nr"])
            c2 = collections.Counter(network_outputs).most_common(1)
            avg_unscaled_nr = across_image_avg(unscaled_nr_sum, unscaled_nr_len)
            avg_unscaled_rec = across_image_avg(unscaled_rec_sum, unscaled_rec_len)
            avg_scaled_nr = across_image_avg(scaled_nr_sum, scaled_nr_len)
            avg_scaled_rec = across_image_avg(scaled_rec_sum, scaled_rec_len)
            
            subject_truth_label = give_string_label(image_dict[image]["labels"][0])
            subject_network_label = give_string_label(c2[0][0])
            
            create_histogram(subject_dict[subject], image_dict, subject_network_label)
            smax_subject = softmax([avg_unscaled_nr, avg_unscaled_rec])
            if subject_truth_label == subject_network_label:
                color = 'on_green'
            else:
                color = 'on_red'
            header_line = "\n"+subject + " -- " + str(len(subject_dict[subject])) + " images"
            cprint(header_line, 'white', color, attrs=['bold'])
            print("Label: "+ subject_truth_label)
            print("Network majority vote: %s, %i%% (%i/%i) " % (subject_network_label, int((c2[0][1])/len(network_outputs)*100), int(c2[0][1]), len(network_outputs)))
            print("Unscaled average: %.3f, %.3f" % (avg_unscaled_nr, avg_unscaled_rec) )
            print("Softmax of unscaled average: " + str(smax_subject))
            print("Scaled Average: %.3f, %.3f" % (avg_scaled_nr, avg_scaled_rec))

def main():
    args = sys.argv[1:]
    filename = args[0]
    image_dict={}
    subject_dict={}

    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        header = next(csvreader) # Discard header line
        for row in csvreader:
            if row[1] not in image_dict:
                image_dict[row[1]]={}
                initialize_image(image_dict[row[1]], row)
            else:
                add_data_to_existing_image(image_dict[row[1]], row)
            if row[0] not in subject_dict:
                initialize_subject(subject_dict, row)
            else:
                add_data_to_existing_subject(subject_dict[row[0]], row)
    for image_name in image_dict:
        image_patch_data = ops_within_patches(image_dict[image_name]["coords"])
        create_heat_map(image_name)
    if FLAGS.info == "subject":
        print_info_per_subject(subject_dict, image_dict)
    elif FLAGS.info == "image":
        print_info_per_image(image_dict)
    else:
        print_info_per_subject(subject_dict, image_dict)
        print_info_per_image(image_dict)
    if not args:
        print('usage: enter location of csv file as first parameter, enter --info="image" for only images, "subject" for only subjects, or any other value for all info')
        sys.exit(1)


# Main body
if __name__ == '__main__':
	main()