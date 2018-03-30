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
from pylab import subplot, plot, subplots_adjust
from skimage import io
from PIL import Image
from termcolor import cprint
from tensorflow import flags
from IPython import embed
from sklearn.metrics import roc_curve, auc

# Global variables
flags.DEFINE_string("info","subject","Display per subject or per image info")
flags.DEFINE_string("base_path", None, "Path of folder containing voting file")
flags.DEFINE_string("histogram_path", None, "Path for saving folder of histogram images")
flags.DEFINE_string("og_image_path","/home/wanglab/Desktop/recurrence_seq_lstm/image_data/original_images/","Location of original images for generating heat map")
flags.DEFINE_integer("patch_size", 500, "Dimensions of square patches taken from original image to generate the sequences given to the network")
flags.DEFINE_float("patch_overlap", 0.3, "Overlap portion of patches taken from the original images")
flags.DEFINE_string("map_path", None, "Path for saving heat maps")
flags.DEFINE_bool("create_maps", False, "Choose to create heat maps of images")
flags.DEFINE_bool("from_outside", False, "Operating from another python script")
flags.DEFINE_bool('per_subject', False, 'Provide ROC and majority votes per subject')
flags.DEFINE_string('voting_file', 'voting_file.csv', 'Name of voting file for gathering data')
flags.DEFINE_bool('print', False, 'Print subject or image majority voting results to command line terminal')
flags.DEFINE_string('subject_conditions', None, 'name of csv file with cross validation conditions')
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
    # raw_patch_info = sum_neighboring_patches(image_info)
    raw_patch_info = image_info["coords"]
    original_img_shape = io.imread(os.path.join(FLAGS.og_image_path,image_info["name"] + ".tif")).shape
    x_length = original_img_shape[1]
    y_length = original_img_shape[0]
    label = int(image_info["labels"][0])
    tiff_arr = np.zeros((y_length, x_length), dtype=np.uint8)
    #heat_arr = np.zeros((y_length, x_length), dtype=np.float32)
    stride = 500
    for patch in raw_patch_info:
        rec_avg = average_value(raw_patch_info[patch], "rec")
        nr_avg = average_value(raw_patch_info[patch], "nr")
        soft_max = softmax([nr_avg, rec_avg])
        x_patch_edge = patch[0] + stride
        y_patch_edge = patch[1] + stride
        tiff_arr[patch[1]:y_patch_edge, patch[0]:x_patch_edge] = soft_max[label]*200 + 55
        #heat_arr[patch[1]:y_patch_edge, patch[0]:x_patch_edge] = soft_max[label] + 0.01

    print("Heat mapping %s" % (image_info["name"]))
    #plt.imshow(heat_arr, cmap='hot', interpolation='nearest')
    #plt.savefig(os.path.join(FLAGS.map_path,"pretty_" + image_info["name"]+".jpg"))
    io.imsave(os.path.join(FLAGS.map_path,image_info["name"]+".tif"), tiff_arr)

    del tiff_arr

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
    for image_name in sorted(image_dict):
        c1 = collections.Counter(image_dict[image_name]["output"]).most_common(1)
        truth_label = give_string_label(image_dict[image_name]["labels"][0])
        network_label = give_string_label(c1[0][0])
        if truth_label == network_label:
            color = 'on_green'
        else:
            color = 'on_red'
        cprint("\n" + image_name + " -- " + image_dict[image_name]["id"], 'white', color, attrs=['bold'])
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

def analysis_per_subject(subject_dict, image_dict):
    subject_data = dict()
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
            subject_data[subject] = dict()

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
            smax_subject = softmax([avg_unscaled_nr, avg_unscaled_rec])
            
            subject_data[subject]["unr"] = avg_unscaled_nr
            subject_data[subject]["urec"] = avg_unscaled_rec
            subject_data[subject]["snr"] = avg_scaled_nr
            subject_data[subject]["srec"] = avg_scaled_rec
            subject_data[subject]["truth_label"] = subject_truth_label
            subject_data[subject]["net_label"] = subject_network_label
            subject_data[subject]["smax"] = smax_subject
            subject_data[subject]["vote"] = c2[0][1]/len(network_outputs)

    return subject_data

def create_overall_roc_curve(image_dict):
    all_seq = dict()

    all_seq["scaled_rec"] = []
    all_seq["scaled_nr"] = []
    all_seq["labels"] = []

    for image in image_dict:
        all_seq["scaled_rec"] = all_seq["scaled_rec"] + image_dict[image]["scaled_rec"]
        all_seq["scaled_nr"] = all_seq["scaled_nr"] + image_dict[image]["scaled_nr"]
        all_seq["labels"] = all_seq["labels"] + image_dict[image]["labels"]

    lw = 2

    labels = np.array(all_seq["labels"], dtype=np.uint8)

    nr_scores = np.array(all_seq["scaled_nr"])
    rec_scores = np.array(all_seq["scaled_rec"])

    fpr, tpr, thresholds = roc_curve(labels, rec_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc

    # plt.figure()
    # plt.plot(1-fpr, tpr, color='darkorange',
    #      lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [1,0], color='black', lw=1, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Specificity')
    # plt.ylabel('Sensitivity')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig(os.path.join(FLAGS.base_path, "ROC.jpg"))
    # plt.clf()

def per_subject_roc_curves(subject_dict, image_dict):
    roc_dict = dict()
    for subject in subject_dict:
        roc_dict[subject] = dict()
        roc_dict[subject]["scaled_rec"] = []
        roc_dict[subject]["labels"] = []
        for image in subject_dict[subject]:
            roc_dict[subject]["scaled_rec"] = roc_dict[subject]["scaled_rec"] + image_dict[image]["scaled_rec"]
            roc_dict[subject]["labels"] = roc_dict[subject]["labels"] + image_dict[image]["labels"]
        labels = np.array(roc_dict[subject]["labels"], dtype=np.uint8)
        rec_scores = np.array(roc_dict[subject]["scaled_rec"])
        fpr, tpr, thresholds = roc_curve(labels, rec_scores, pos_label=1)
        # tpr[np.isnan(tpr)] = 0
        # fpr[np.isnan(fpr)] = 0
        roc_auc = auc(fpr, tpr)
        roc_dict[subject]['fpr'] = fpr
        roc_dict[subject]['tpr'] = tpr
        roc_dict[subject]['thresholds'] = thresholds
        roc_dict[subject]['auc'] = roc_auc
    return roc_dict

def plot_roc_curves_and_votes(roc_dict, subject_dict, image_dict):
    subject_data = analysis_per_subject(subject_dict, image_dict)
    number_of_subplot_rows = len(roc_dict)
    figure = plt.figure()
    counter = 1
    for subject in roc_dict:
        roc_loc = str(number_of_subplot_rows) + '2' + str(counter)
        counter += 1
        bar_loc = str(number_of_subplot_rows) + '2' + str(counter)
        counter += 1

        fpr = roc_dict[subject]['fpr']
        tpr = roc_dict[subject]['tpr']

        roc = figure.add_subplot(roc_loc)
        roc.plot(1-fpr, tpr, color='darkorange', lw=2)
        roc.plot([0, 1], [1,0], color='black', lw=1, linestyle='--')
        roc.set_title('ROC')
        legend = 'ROC curve (area = %0.2f)' % roc_dict[subject]['auc']
        roc.legend([legend])

        bar_graph = figure.add_subplot(bar_loc)

        if not subject_data[subject]["truth_label"] == subject_data[subject]["net_label"]:
            subject_data[subject]["vote"] = 1 - subject_data[subject]["vote"]
        if subject_data[subject]["truth_label"] == 'RECURRENT':
            bar_graph.bar(1, subject_data[subject]["vote"], color='orange')
        elif subject_data[subject]["truth_label"] == 'NONRECURRENT':
            bar_graph.bar(1, subject_data[subject]["vote"], color='blue')
        
        bar_graph.set_xticks([subject])
        # bar_graph.set_xticklabels([subject])
        bar_graph.set_ylim([0, 1])
        bar_graph.legend([subject_data[subject]["truth_label"]])

def save_roc_curve(fpr, tpr, thresholds, roc_auc):
    plt.plot(1-fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [1,0], color='black', lw=1, linestyle='--')
    path_list = FLAGS.base_path.split("/")
    plt.title(path_list[len(path_list)-2])
    legend = 'ROC curve (area = %0.2f)' % roc_auc
    plt.legend([legend])
    plt.savefig(os.path.join(FLAGS.base_path,"ROC.jpg"))
    plt.clf()

def add_test_condition(subject_data):
    if not FLAGS.subject_conditions:
        for subject in subject_data:
            subject_data[subject]['condition'] = 1
    else:
        with open(os.path.join(FLAGS.base_path,FLAGS.subject_conditions), 'r') as csvfile:
            reader = csv.reader(csvfile)
            _ = next(reader) # discard header
            for line in reader:
                if line[0] in subject_data:
                    subject_data[line[0]]["condition"] = line[2]

def save_subjects_vote_bar_graph(image_dict, subject_dict):
    subject_data = analysis_per_subject(subject_dict, image_dict)
    add_test_condition(subject_data)
    index = 0
    names =[]
    votes = []
    recur_dict = dict()
    nonrecur_dict = dict()
    for subject in subject_data:
        if subject_data[subject]["truth_label"] == 'RECURRENT':
            recur_dict[subject] = subject_data[subject]
        else:
            nonrecur_dict[subject] = subject_data[subject]
    
    total_conditions = 1
    if FLAGS.subject_conditions:
        for subject in subject_data:
            if int(subject_data[subject]["condition"]) > total_conditions:
                total_conditions = int(subject_data[subject]["condition"])

    for cond in np.arange(total_conditions):
        cond +=1        
        for subject in sorted(recur_dict):
            if int(recur_dict[subject]['condition']) == cond:
                if not recur_dict[subject]["truth_label"] == recur_dict[subject]["net_label"]:
                    recur_dict[subject]["vote"] = 1 - recur_dict[subject]["vote"]
                rec_legend = plt.bar(index, recur_dict[subject]["vote"], color='orange', label="Recurrent")
                index += 1
                names.append(subject)
                votes.append(recur_dict[subject]["vote"])
                subject_data[subject]["vote"] = recur_dict[subject]["vote"]
        
        for subject in sorted(nonrecur_dict):
            if int(nonrecur_dict[subject]['condition']) == cond:
                if not nonrecur_dict[subject]["truth_label"] == nonrecur_dict[subject]["net_label"]:
                    nonrecur_dict[subject]["vote"] = 1 - nonrecur_dict[subject]["vote"]
                nonrec_legend = plt.bar(index, nonrecur_dict[subject]["vote"], color='blue', label="Nonrecurrent")
                index += 1
                names.append(subject)
                votes.append(nonrecur_dict[subject]["vote"])
                subject_data[subject]["vote"] = nonrecur_dict[subject]["vote"]
        plt.bar(index, 0)
        index += 1
        names.append('^^ %i ^^' % (cond))

    plt.xticks(np.arange(len(subject_data.keys()) + total_conditions), names, rotation=90)
    plt.axhline(0.5, color='gray')
    vote_sum = 0
    for subject in subject_data:
        vote_sum += subject_data[subject]["vote"]
    vote_avg = vote_sum / len(subject_data)

    plt.ylim([0,1])
    results_path = FLAGS.base_path.split("/")
    x = len(results_path)
    plt.title(results_path[x-1] + " (avg vote = %.4f)" % (vote_avg))
    plt.legend([rec_legend, nonrec_legend], ["Recurrent", "Nonrecurrent"])
    plt.savefig(os.path.join(FLAGS.base_path,"majority_voting.jpg"))
    plt.clf()
    auc_dict = per_condition_roc(subject_data, total_conditions)
    write_results_csv(subject_data, recur_dict, nonrecur_dict, auc_dict, total_conditions)

def per_condition_roc(subject_data, total_conditions):
    auc_dict = dict()
    for cond in np.arange(total_conditions):
        cond += 1
        cond_labels = []
        cond_votes = []
        for subject in subject_data:
            if int(subject_data[subject]["condition"]) == cond:
                if subject_data[subject]['truth_label'] == 'RECURRENT':
                    cond_labels = cond_labels + [1]
                    cond_votes = cond_votes + [subject_data[subject]['vote']]
                else:
                    cond_labels = cond_labels + [0]
                    cond_votes = cond_votes + [1 - subject_data[subject]['vote']]
        label_arr = np.array(cond_labels)
        votes_arr = np.array(cond_votes)
        fpr, tpr, thresholds = roc_curve(label_arr, votes_arr, pos_label=1)
        roc_auc = auc(fpr, tpr)
        auc_dict[cond] = roc_auc
        plt.plot(1-fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [1,0], color='black', lw=1, linestyle='--')
        path_list = FLAGS.base_path.split("/")
        plt.title(path_list[len(path_list)-2] + " -- Condition " + str(cond))
        legend = 'ROC curve (area = %0.2f)' % roc_auc
        plt.legend([legend])
        plt.savefig(os.path.join(FLAGS.base_path,str(cond) + "_ROC.jpg"))
        plt.clf()
    return auc_dict

def write_results_csv(subject_data, recur_dict, nonrecur_dict, auc_dict, total_conditions):
    auc_arr = np.zeros(len(auc_dict))
    for cond in auc_dict:
        auc_arr[cond-1] = auc_dict[cond]

    total_votes = np.zeros(len(subject_data))
    recur_votes = np.zeros(len(recur_dict))
    nonrecur_votes = np.zeros(len(nonrecur_dict))
    count = 0
    for subject in subject_data:
        total_votes[count] = subject_data[subject]["vote"]
        count += 1
    
    recur_pass = 0
    recur_count = 0
    for subject in recur_dict:
        recur_votes[recur_count] = subject_data[subject]["vote"]
        recur_count += 1
        if recur_dict[subject]['vote'] > 0.5:
            recur_pass += 1

    nonrecur_pass = 0
    nonrecur_count = 0
    for subject in nonrecur_dict:
        nonrecur_votes[nonrecur_count] = subject_data[subject]["vote"]
        nonrecur_count += 1
        if nonrecur_dict[subject]['vote'] > 0.5:
            nonrecur_pass += 1
    # pdb.set_trace()
    with open(os.path.join(FLAGS.base_path, 'test_summary.csv'), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Recurrent subjects passed:', recur_pass,'','Nonrecurrence subjects passed:', nonrecur_pass])
        csvwriter.writerow(['Recurrence subjects:', len(recur_dict),'','Nonrecurrence subjects:', len(nonrecur_dict)])
        csvwriter.writerow(['Recurrence accuracy:', '{0:.4f}'.format(recur_pass / len(recur_dict)),'','Nonrecurrence accuracy:', '{0:.4f}'.format(nonrecur_pass / len(nonrecur_dict))])
        csvwriter.writerow(['Recurrence average vote:', '{0:.4f}'.format(np.mean(recur_votes)),'','Nonrecurrence average vote:', '{0:.4f}'.format(np.mean(nonrecur_votes))])
        csvwriter.writerow(['Recurrence vote std-dev:', '{0:.4f}'.format(np.std(recur_votes)),'','Nonrecurrence vote std-dev:', '{0:.4f}'.format(np.std(nonrecur_votes))])
        csvwriter.writerow([''])
        csvwriter.writerow(['Average vote:', '{0:.4f}'.format(np.mean(total_votes)),'','Average AUC:', '{0:.4f}'.format(np.mean(auc_arr))])
        csvwriter.writerow(['Vote std-dev:', '{0:.4f}'.format(np.std(total_votes)),'','AUC std-dev:', '{0:.4f}'.format(np.std(auc_arr))])
        csvwriter.writerow([''])
        csvwriter.writerow(['subject', 'label', 'vote', 'condition'])
        for cond in np.arange(total_conditions):
            cond += 1
            for subject in sorted(subject_data):
                if int(subject_data[subject]['condition']) == cond:
                    if subject_data[subject]['truth_label'] == 'RECURRENT':
                        csvwriter.writerow([subject, '1', '{0:.4f}'.format(subject_data[subject]['vote']), subject_data[subject]["condition"]])
                    else:
                        csvwriter.writerow([subject, '0', '{0:.4f}'.format(subject_data[subject]['vote']), subject_data[subject]["condition"]])

def majority_vote(base_path, voting_filename=None, hist_path=None, map_path=None, create_maps=False, info=None, patch_size=None, patch_overlap=None, og_image_path=None):
    if not hist_path:
        hist_path = os.path.join(base_path, "histograms")
    if not map_path:
        map_path = os.path.join(base_path, "maps")
    
    FLAGS.base_path = base_path
    FLAGS.hist_path = hist_path
    FLAGS.map_path = map_path

    if voting_filename:
        FLAGS.voting_file=voting_filename
    if create_maps:
        FLAGS.create_maps = create_maps
    if info:
        FLAGS.info = info
    if patch_size:
        FLAGS.patch_size = patch_size
    if patch_overlap:
        FLAGS.patch_overlap = patch_overlap
    if og_image_path:
        FLAGS.og_image_path = og_image_path

    FLAGS.from_outside = True

    image_dict, subject_dict = main()
    return image_dict, subject_dict

def main():
    if not FLAGS.base_path:
        raise ValueError("Must set --base_path to the directory containing the voting_file.csv")
    base_path = FLAGS.base_path
    filename = os.path.join(base_path, FLAGS.voting_file)
    if not FLAGS.histogram_path:
        FLAGS.histogram_path = os.path.join(base_path, "histograms")
    if not FLAGS.map_path:
        FLAGS.map_path = os.path.join(base_path, "maps")
    if not FLAGS.from_outside:
        os.makedirs(FLAGS.histogram_path, exist_ok=True)
        os.makedirs(FLAGS.map_path, exist_ok=True)

    image_dict={}
    subject_dict={}

    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        header = next(csvreader) # Discard header line
        for row in csvreader:
            if not row:
                continue
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
        if FLAGS.create_maps:
            generate_heat_map_single_image(image_dict[image_name])
    if FLAGS.from_outside:
        return image_dict, subject_dict
    
    if FLAGS.per_subject:
        roc_dict = per_subject_roc_curves(subject_dict, image_dict)
        plot_roc_curves_and_votes(roc_dict, subject_dict, image_dict)
    else:
        fpr, tpr, thresholds, roc_auc = create_overall_roc_curve(image_dict)
        save_roc_curve(fpr, tpr, thresholds, roc_auc)
    
    if FLAGS.info == "subject":
        if FLAGS.print == True:
            print_info_per_subject(subject_dict, image_dict)
        save_subjects_vote_bar_graph(image_dict, subject_dict)
    elif FLAGS.info == "image" and FLAGS.print == True:
        print_info_per_image(image_dict)     
    elif FLAGS.print == True:
        print_info_per_subject(subject_dict, image_dict)
        print_info_per_image(image_dict)


# Main body
if __name__ == '__main__':
	main()