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
from skimage import io
from PIL import Image
from termcolor import cprint
from tensorflow import flags

# Global variables
flags.DEFINE_string("info","subject","Display per subject or per image info")
flags.DEFINE_string("histogram_path", None, "Path for saving folder of histogram images")
flags.DEFINE_string("og_image_path","/home/wanglab/Desktop/recurrence_seq_lstm/image_data/original_images/","Location of original images for generating heat map")
flags.DEFINE_integer("patch_size", 500, "Dimensions of square patches taken from original image to generate the sequences given to the network")
flags.DEFINE_float("patch_overlap", 0.3, "Overlap portion of patches taken from the original images")
flags.DEFINE_string("map_path", None, "Path for saving heat maps")
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

def create_histogram_data_per_subject(subject_image_list, image_dict, truth_label):
    scaled_rec_hist_list = []
    unscaled_rec_hist_list = []
    
    scaled_nr_hist_list = []
    unscaled_nr_hist_list = []

    for image in subject_image_list:
        scaled_rec_hist_list = scaled_rec_hist_list + image_dict[image]["scaled_rec"]
        unscaled_rec_hist_list = unscaled_rec_hist_list + image_dict[image]["unscaled_rec"]

        scaled_nr_hist_list = scaled_nr_hist_list + image_dict[image]["scaled_nr"]
        unscaled_nr_hist_list = unscaled_nr_hist_list + image_dict[image]["unscaled_nr"]

    scaled_rec_arr = np.array(scaled_rec_hist_list)
    unscaled_rec_arr = np.array(unscaled_rec_hist_list)
    scaled_nr_arr = np.array(scaled_nr_hist_list)
    unscaled_nr_arr = np.array(unscaled_nr_hist_list)

    bins = np.linspace(0,1,100)

    if truth_label == "RECURRENT":
        plt.hist(scaled_rec_arr, bins, alpha=0.8, label=truth_label)
    else:
        plt.hist(scaled_nr_arr, bins, alpha=0.8, label=truth_label)

    plt.legend(loc="upper right")
    plt.title("Scaled histogram of sequences from " + image_dict[image]["id"])
    plt.savefig(os.path.join(FLAGS.histogram_path, image_dict[image]["id"] + ".jpg"))
    plt.clf()

def generate_heat_map_single_image(image_info):
    raw_patch_info = sum_neighboring_patches(image_info)
    # raw_patch_info = image_info["coords"]
    original_img_shape = io.imread(os.path.join(FLAGS.og_image_path,image_info["name"] + ".tif")).shape
    x_length = original_img_shape[1]
    y_length = original_img_shape[0]
    label = int(image_info["labels"][0])
    blank_image = np.zeros((y_length, x_length), dtype=np.uint8)
    stride = 500
    for patch in raw_patch_info:
        rec_avg = average_value(raw_patch_info[patch], "rec")
        nr_avg = average_value(raw_patch_info[patch], "nr")
        soft_max = softmax([nr_avg, rec_avg])
        x_patch_edge = patch[0] + stride
        y_patch_edge = patch[1] + stride
        blank_image[patch[1]:y_patch_edge, patch[0]:x_patch_edge] = soft_max[label]*200 + 55
        # blank_image[patch[1]:y_patch_edge, patch[0]:x_patch_edge] = 255
    # img = Image.fromarray(blank_image)
    # img.save("heat_map_"+image_info["name"]+".jpg")
    print("Heat mapping %s" % (image_info["name"]))
    io.imsave(os.path.join(FLAGS.map_path,image_info["name"]+".tif"), blank_image)

def include_neighbor_patches(coords, coords_dict, stride):
    neighbors_added  = {}
    neighbors_added["rec"] = coords_dict[coords]["rec"]
    neighbors_added["nr"] = coords_dict[coords]["nr"]
    x_list = [coords[0] - stride, coords[0], coords[0] + stride]
    y_list = [coords[1] - stride, coords[1], coords[1] + stride]
    for x in x_list:
        for y in y_list:
            if (x,y) in coords_dict:
                neighbors_added["rec"] = neighbors_added["rec"] + coords_dict[(x,y)]["rec"]
                neighbors_added["nr"] = neighbors_added["nr"] + coords_dict[(x,y)]["nr"]
    return neighbors_added

def sum_neighboring_patches(image_info):
    patch_size = FLAGS.patch_size
    patch_overlap = FLAGS.patch_overlap
    stride = int(patch_size * (1 - patch_overlap))
    neighbors = {}    
    for coord in image_info["coords"]:
        neighbors[coord] = include_neighbor_patches(coord, image_info["coords"], stride)
    return neighbors

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
            
            create_histogram_data_per_subject(subject_dict[subject], image_dict, subject_truth_label)
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
    base_path = args[0]
    filename = os.path.join(base_path, "voting_file.csv")
    if not FLAGS.histogram_path:
        FLAGS.histogram_path = os.path.join(base_path, "histograms")
    if not FLAGS.map_path:
        FLAGS.map_path = os.path.join(base_path, "maps")
    os.makedirs(FLAGS.histogram_path, exist_ok=True)
    os.makedirs(FLAGS.map_path, exist_ok=True)

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
        generate_heat_map_single_image(image_dict[image_name])
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