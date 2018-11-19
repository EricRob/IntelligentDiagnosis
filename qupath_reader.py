


#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import pickle
import numpy as np
from PIL import Image
import warnings
import csv
import os
import sys
from IPython import embed
import pdb
from termcolor import cprint
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow import flags
import argparse
from skimage import io
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tempfile import TemporaryFile
from math import isnan
import matplotlib.image as mpimg


warnings.simplefilter('ignore', Image.DecompressionBombWarning)

parser = argparse.ArgumentParser(description='Process QuPath detections and results for feature creation')

# flags.DEFINE_integer("test", 0, "Test mode")
# flags.DEFINE_integer('clusters', 8, 'Number of clusters for KMeans Clustering')
# flags.DEFINE_bool('all_cells', False, 'Create histogram showing features of all cells')
# flags.DEFINE_bool('all_features', False, 'Use all QuPath features')
# flags.DEFINE_string('closest_centers', 5, 'Number of closest cluster centers to find when computing regional immune density')
# flags.DEFINE_float('dense_thresh', 3, 'percentage of clusters used to calculate the cluster density feature')
# flags.DEFINE_string('classifier', 'v2', 'Classifier version to use')
# flags.DEFINE_bool('load', True, 'load >data< array from saved pickle')
# flags.DEFINE_bool('display_clusters', False, 'Display immune clusters in a scatter plot')
# flags.DEFINE_string('show_delaunay', False, 'Show histograms of delaunay triangulation')
# flags.DEFINE_string('detections_dir', '/data/QuPath/CellCounter/detections_v2', "Directory of QuPath detections files that should be used.")
# flags.DEFINE_bool('overwrite_saved', False, 'overwrite existing data (not implemented everywhere!)')
# flags.DEFINE_integer('small_d_cluster', 1, 'size of small cluster for class comparison')
# flags.DEFINE_integer('delaunay_radius', 50, 'pixel radius of delaunay triangulation')

parser.add_argument('--test', default=0, type=int, help='test mode')
parser.add_argument('--clusters', default=8, type=int, help='Number of clusters for KMeans Clustering')
parser.add_argument('--all_cells', default=False, type=bool, help='Create histogram showing features of all cells')
parser.add_argument('--all_features', default=False, type=bool, help='Use all QuPath features')
parser.add_argument('--closest_centers', default=5, type=str, help='Number of closest cluster centers to find when computing regional immune density')
parser.add_argument('--dense_thresh', default=3, type=float, help='percentage of clusters used to calculate the cluster density feature')
parser.add_argument('--classifier', default='v2', type=str, help='Classifier version to use')
parser.add_argument('--load_pickles', default=False, type=bool, help='load >data< array from saved pickle')
parser.add_argument('--display_clusters', default=False, type=bool, help='Display immune clusters in a scatter plot')
parser.add_argument('--show_delaunay', default=False, type=str, help='Show histograms of delaunay triangulation')
parser.add_argument('--detections_dir', default='/data/QuPath/CellCounter/delaunay_px40/CUMC/', type=str, help="Directory of QuPath detections files that should be used.")
parser.add_argument('--overwrite_saved', default=False, type=bool, help='overwrite existing data (not implemented everywhere!)')
parser.add_argument('--small_d_cluster', default=3, type=int, help='size of small cluster for class comparison')
parser.add_argument('--delaunay_radius', default=40, type=int, help='pixel radius of delaunay triangulation')
parser.add_argument('--yale', default=False, type=bool, help='Processing yale data (as opposed to CUMC or Sinai)')
parser.add_argument('--sinai', default=False, type=bool, help='Processing sinai data (as opposed to CUMC or Yale')
parser.add_argument('--subject_list', default=None, type=str, help='List of subjects for feature analysis')
parser.add_argument('--loader', default=False, type=bool, help='load and load_pickles don\'t seem to work, so maybe this will.')

QFLAGS = parser.parse_args()

# Global variables
# DETECTIONS = '/data/QuPath/CellCounter/delaunay_px' + str(QFLAGS.delaunay_radius) + '/CUMC/'
# if QFLAGS.yale:
#     DETECTIONS = '/data/yale_qupath/measurements/'
# else:
#     DETECTIONS = '/data/QuPath/CellCounter/delaunay_px' + str(QFLAGS.delaunay_radius)
#     if QFLAGS.sinai:
#         DETECTIONS = os.path.join(DETECTIONS, 'Sinai')
#     else:
#         DETECTIONS = os.path.join(DETECTIONS, 'CUMC')
DETECTIONS = QFLAGS.detections_dir
# DETECTIONS = '/data/yale_qupath/measurements'

STATUS_CSV = '/data/recurrence_seq_lstm/IntelligentDiagnosis/recurrence_status.csv'
OMIT_KEY = ['Name', 'ROI', 'Centroid X px', 'Centroid Y px']
OMIT_CLASS = [ 'red_cell', 'ulceration']
KEEP_CLASS = ['Other', 'Tumor', 'Immune cells']
MASK_LOCATION = '/data/recurrence_seq_lstm/image_data/masks'
FEATURES = ['Cell: Area', 'Nucleus: Area', 'Nucleus/Cell area ratio', 'Cell: Perimeter', 'Class', 'Centroid X', 'Centroid Y']
THRESHOLD = 0.15
COLORS = ['red', 'green']
FILLER = '                                            '
FEATURE_DIR = '/data/recurrence_seq_lstm/feature_testing'
ORIGINAL_IMAGE_DIR = '/data/recurrence_seq_lstm/image_data/original_images'
DATA_STORAGE = '/data/recurrence_seq_lstm/feature_testing/delaunay_grid_search.csv'

ADD_NAME = ""
if QFLAGS.sinai:
    ADD_NAME = ADD_NAME + "_sinai"
if QFLAGS.yale:
    ADD_NAME = ADD_NAME + "_yale"
# Class declarations

# Function declarations
def convert_nan_to_zero(arr):
    nan_locs = np.isnan(arr)
    arr[nan_locs] = 0
    return arr

def remove_nan_values(arr):
    nan_locs = np.isnan(arr)
    no_nans = arr[np.invert(nan_locs)]
    return no_nans

def get_bins(rec, non):
    maximum = np.amax(rec)
    if np.amax(non) > maximum:
        maximum = np.amax(non)
    minimum = np.amin(rec)
    if np.amin(non) < minimum:
        minimum = np.amin(non)
    return np.linspace(minimum, maximum, num=100)

def show_base_features_histogram(RE, NR):

    for cell_class in RE:
        n = 0
        if cell_class not in NR or cell_class in OMIT_CLASS:
            continue
        cprint('Plotting ' + cell_class, 'green', end="\r")
        for value in RE[cell_class]:
            if value not in NR[cell_class] or value in OMIT_KEY or 'Centroid' in value:
                continue
            n += 1
            plt.subplot(4,9,n)
            rec = convert_nan_to_zero(np.array(RE[cell_class][value]))
            non = convert_nan_to_zero(np.array(NR[cell_class][value]))
            bins = get_bins(rec / len(rec), non / len(non))
            # plt.hist(non, stacked=True, bins=100, alpha=0.5, label='Nonrecurrent', density=True)
            # plt.hist(rec, stacked=True, bins=100, alpha=0.5, label="Recurrent", density=True)
            plt.hist(non,  bins=100, alpha=0.5, label='Nonrecurrent')
            plt.hist(rec,  bins=100, alpha=0.5, label="Recurrent")
            plt.xlabel(value)
        plt.legend(loc='upper right')
        plt.title(cell_class)
        plt.show()

def add_to_cell_dict(cell_dict, subject):
    for image in subject:
        if image == 'status':
            continue
        for cell in subject[image]:
            if cell == 'status':
                continue
            if QFLAGS.all_features:
                cell_class = subject[image][cell]['Class']
            else:
                cell_class = subject[image][cell]['class']
            

            if cell_class in OMIT_CLASS:
                continue

            if QFLAGS.all_features:
                cell_class = "immune_and_tumor"

            if cell_class not in cell_dict:
                cell_dict[cell_class] = {}


            for key in subject[image][cell].keys():
                if key in OMIT_KEY or key == 'Class':
                    continue
                if key not in cell_dict[cell_class]:
                    cell_dict[cell_class][key] = []
                cell_dict[cell_class][key].append(float(subject[image][cell][key]))
    return

def get_cell_counts(image):
    tumor_count = 1
    immune_count = 1
    tumor_area = 1
    immune_area = 1
    for cell in image:
        cell_class = image[cell]['class']
        if cell_class == 'Tumor':
            tumor_count += 1
            tumor_area += image[cell]['total_area']
        elif cell_class == 'Immune cells':
            immune_count += 1
            immune_area += image[cell]['total_area']
    return immune_count, tumor_count, immune_area, tumor_area

def create_roc_curve(fpr, tpr, thresholds, roc_auc):
    plt.plot(1-fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [1,0], color='black', lw=1, linestyle='--')
    legend = 'ROC curve (area = %0.2f)' % roc_auc
    plt.legend([legend])
    plt.show()
    # plt.savefig(os.path.join(QFLAGS.base_path,"ROC.jpg"))
    # plt.clf()

def run_thresholds(start, stop, samples, labels, values):
    holds = []
    accuracies = []
    sensitivities = []
    specificities = []
    for thresh in np.linspace(start,stop,num=samples):
        # threshed = values > thresh
        passed = labels[values >= thresh]
        under = labels[values < thresh]

        rec_under = np.sum(under) # TP
        rec_over = np.sum(passed) # FN

        nr_under = len(under) - rec_under # FP
        nr_over = len(passed) - rec_over # TN

        sens = np.nan_to_num(rec_under / (rec_under + rec_over)) # TP / (FN + TP)
        spec = np.nan_to_num(1 - nr_over / (nr_over + nr_under)) # TN / (FP + TN)

        # nr_acc = nr_under / (nr_under + nr_over)
        # rec_acc =  rec_over / (rec_under + rec_over)

        acc = (nr_over + rec_under) / (len(labels))
        holds.append(thresh)
        accuracies.append(acc)
        sensitivities.append(sens)
        specificities.append(spec)
        # print('Threshold: ' + "{0:02f}".format(thresh) + ' NR accuracy:' + str("{0:03f}".format(nr_acc)) + ' RE accuracy: ' + str("{0:03f}".format(rec_acc)))
        # print('Threshold: ' + "{0:02f}".format(thresh) + ' Acc: ' + "{0:03f}".format(acc) + '  Sens: ' + "{0:03f}".format(sens) + '  Spec:'+ "{0:03f}".format(1-spec) )

    max_acc = max(accuracies)
    
    area = 0 
    for i in np.arange(len(sensitivities)):
        if i == 0:
            area = area + sensitivities[i]*specificities[i]
        else:
            area = area + ((sensitivities[i-1] + sensitivities[i])*(specificities[i] - specificities[i-1])/2)
    print('Max Acc: ' + "{0:03f}".format(max_acc) + '  AUC: ' + "{0:03f}".format(area) )
    return holds, accuracies, sensitivities, specificities

def load_image_masks(data):
    re_immune = []
    re_tumor = []
    nr_immune = []
    nr_tumor = []
    re_portion = []
    nr_portion = []
    for subject in data:
        if subject == 'status':
            continue
        if 'status' not in data[subject]:
            continue
        for image in data[subject]:
            mask_filename = os.path.join(MASK_LOCATION, 'mask_' + image + '.tif')
            if image == 'status':
                continue
            elif not os.path.exists(mask_filename):
                continue
            else:
                cprint('reading ' + image + FILLER, 'cyan', end='\r')
                immune_count, tumor_count, immune_area, tumor_area = get_cell_counts(data[subject][image])
                if (immune_count + tumor_count):
                    imm_portion = immune_count / (immune_count + tumor_count)
                else:
                    continue
                # mask = io.imread(mask_filename)
                # mask = mask[:,:,0]
                # pos = mask > 0
                # mask[pos] = 1
                # mask_total = np.sum(mask)
                # mask_size = mask.shape[0] * mask.shape[1]
                # immune_ratio = immune_area / mask_total
                # tumor_ratio = tumor_area / mask_total

                total_area = immune_area + tumor_area
                immune_ratio = immune_area / total_area
                tumor_ratio = tumor_area / total_area
                if data[subject]['status']:
                    re_immune.append(immune_ratio)
                    re_tumor.append(tumor_ratio)
                    re_portion.append(imm_portion)
                else:
                    nr_immune.append(immune_ratio)
                    nr_tumor.append(tumor_ratio)
                    nr_portion.append(imm_portion)
                # del mask
                # del pos
    re_portion_labels = np.ones(len(re_portion))
    nr_portion_labels = np.zeros(len(nr_portion))
    re_portion_np = np.array(re_portion)
    nr_portion_np = np.array(nr_portion)

    labels = np.concatenate((re_portion_labels, nr_portion_labels))
    values = np.concatenate((re_portion_np, nr_portion_np))

    # thresholds = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    thresh, acc, sens, spec = run_thresholds(0, .5, 501, labels, values)
    plt.plot(thresh, acc, lw=2)
    plt.ylim(0,1)
    plt.title('Immune cell proportions')
    plt.axhline(max(acc), c='red')
    plt.xlabel('Immune cell count / Total cell count')
    plt.ylabel('Accuracy')
    plt.text(.1, .8, 'Threshold: 0.204 Accuracy: ' "{0:03f}".format(max(acc)))
    plt.show()
    


    # fpr, tpr, thresholds = roc_curve(labels, values, pos_label=1)
    # roc_auc = auc(fpr, tpr)
    # create_roc_curve(fpr, tpr, thresholds, roc_auc)

    # pdb.set_trace()
    # plt.subplot(3,1,1)
    # plt.hist(re_immune, bins=25, alpha=0.5, label='Rec immune')
    # plt.hist(nr_immune, bins=25, alpha=0.5, label="nr immune")
    # plt.title('immune')
    # plt.subplot(3,1,2)
    # plt.hist(re_tumor, bins=25, alpha=0.5,label='Rec tumor')
    # plt.hist(nr_tumor, bins=25, alpha=0.5, label="nr tumor")
    # plt.legend(loc='upper right')
    # plt.title('tumor')

    # plt.subplot(3,1,3)
    # plt.hist(re_portion, stacked=True, bins=50, alpha=0.5, label='Recurrent', density=True)
    # plt.hist(nr_portion, stacked=True, bins=50, alpha=0.5, label="Nonrecurrent", density=True)
    # plt.title('Portion')




    # plt.subplot(4,1,4)

    return

def detections(subject_id, image_name):
    return main(subject_id, image_name, image_processor=True)

def add_all_cell_features(detection, row):
    for feature in OMIT_KEY:
        _ = row.pop(feature)
    for feature in row:
        if feature == 'Class':
            detection['Class'] = row[feature]
        else:
            detection[feature] = float(row[feature])
    return 1

def add_cell_features(detection, row):

    detection['cell_area'] = int(row['Cell: Area'])
    detection['nuc_area'] = int(row['Nucleus: Area'])
    detection['area_ratio'] = float(row['Nucleus/Cell area ratio'])
    detection['total_area'] = detection['cell_area'] + detection['nuc_area']
    detection['cell_perimeter'] = float(row['Cell: Perimeter'])
    detection['class'] = row['Class']
    detection['x'] = int(float(row['Centroid X']))
    detection['y'] = int(float(row['Centroid Y']))
    
    return 1

def k_means_clustering_and_features(data, delaunay):
    print()
    plt.clf()
    rec_center_ratio = []
    rec_center_mean = []
    nr_center_ratio = []
    nr_center_mean = []
    rec_density = []
    nr_density = []
    classy = QFLAGS.classifier


    cell_locs_filename = os.path.join(FEATURE_DIR, classy + '_' + str(QFLAGS.clusters) + ADD_NAME + '_cluster_cell_locations.pickle')
    dense_delaunay_filename = os.path.join(FEATURE_DIR, classy + '_' + str(QFLAGS.clusters) + ADD_NAME + '_cluster_dense_delaunay.pickle')

    if os.path.exists(dense_delaunay_filename) and QFLAGS.loader:
        create_dense_delaunay = False
        cprint('Loading dense delaunay pickle', 'yellow')
        with open(dense_delaunay_filename, 'rb') as handle:
            dense_delaunay = pickle.load(handle)
    else:
        create_dense_delaunay = True
        dense_delaunay = {}
    
    if os.path.exists(cell_locs_filename) and QFLAGS.loader:
        cprint('Loading cell locations pickle', 'yellow')
        with open(cell_locs_filename, 'rb') as handle:
            cell_locs = pickle.load(handle)
            create_cell_locs = False
    else:
        cell_locs = {}
        create_cell_locs = True

    if not create_dense_delaunay and not create_cell_locs:
        return cell_locs, dense_delaunay
    for subject in data:
        if subject == 'status':
            continue
        if 'status' not in data[subject]:
            continue
        if subject not in cell_locs:
            cell_locs[subject] = {}
            cell_locs[subject]['status'] = data[subject]['status']
        if create_dense_delaunay:
            dense_delaunay[subject] = {}
            dense_delaunay[subject]['status'] = data[subject]['status']
        for image in data[subject]:
            if 'status' in image:
                continue
            if create_dense_delaunay:
                dense_delaunay[subject][image] = {}
            coord_space = np.zeros([len(data[subject][image]),2])
            tumor_space = np.zeros([len(data[subject][image]),2])
            tumors = []
            tumor_x = []
            tumor_y = []
            immune_x = []
            immune_y = []
            immunes = []
            cell_classes = []
            n = 0
            if image == 'status':
                continue
            if not data[subject][image]:
                cprint(image + ' -- NO CELL DATA', 'white', 'on_red')
                continue
            for cell in data[subject][image]:
                c_dict = data[subject][image][cell]
                if c_dict['class'] in OMIT_CLASS:
                    continue
                else:
                    if c_dict['class'] == 'Tumor':
                        tumor_x.append(int(c_dict['x']))
                        tumor_y.append(int(c_dict['y']))
                        tumor_space[n,0] = int(c_dict['x'])
                        tumor_space[n,1] = int(c_dict['y']) 
                    elif c_dict['class'] == 'Immune cells':
                        immune_x.append(int(c_dict['x']))
                        immune_y.append(int(c_dict['y']))
                        coord_space[n,0] = int(c_dict['x'])
                        coord_space[n,1] = int(c_dict['y']) 
                    n += 1
            if not len(coord_space[:,0] > 0):
                cpring('No immune cells in detections for ' + image + '!, Check qupath output.', 'red', 'on_white')
                pdb.set_trace()
            coord_space = coord_space[coord_space[:,0] > 0, :]
            tumor_space = tumor_space[tumor_space[:,0] > 0, :]
            if coord_space.shape[0] < QFLAGS.clusters:
                cprint('Too few immune cells for clustering, skipping ' + image, 'red', 'on_white')
                continue
            # plt.scatter(tumor_x, max(coord_space[:,1]) - tumor_y, s=1, label='tumor')
            # plt.scatter(immune_x, max(coord_space[:,1]) - immune_y, s=1, label='immune')
            os.makedirs(os.path.join(FEATURE_DIR, image), exist_ok=True)
            centers_filename = os.path.join(FEATURE_DIR, image,classy + image + '_KNNcenters_' + str(QFLAGS.clusters) + '_clusters.npy')
            labels_filename = os.path.join(FEATURE_DIR, image,classy + image + '_KNNcenters_' + str(QFLAGS.clusters) + '_labels.npy')
            # if os.path.exists(centers_filename):
            if 0:
                cprint(image)
                cprint('Loading clusters...', 'grey', 'on_white')
                centers = np.load(centers_filename)
                labels = np.load(labels_filename)
            else:
                cprint('Clustering ' + image + '...', 'grey', 'on_white')
                kmeans = KMeans(n_clusters=QFLAGS.clusters, random_state=0).fit(coord_space)
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                np.save(centers_filename, centers)
                np.save(labels_filename, labels)


            num_densities = int(QFLAGS.dense_thresh*QFLAGS.clusters / 10)
            densities_filename = os.path.join(FEATURE_DIR, image, classy + image + '_' + str(num_densities) + '_of_' + str(QFLAGS.clusters) + '_densities.npy')
            locations_filename = os.path.join(FEATURE_DIR, image, classy + image + '_cells_in_' + str(QFLAGS.clusters) + '_clusters.npy')
            regions_filename = os.path.join(FEATURE_DIR, image, classy + image + '_region_counts_' + str(QFLAGS.clusters) + '_clusters.npy')
            vertices_filename = os.path.join(FEATURE_DIR, image, classy + image + '_cluster_vertices_' + str(QFLAGS.clusters) + '_clusters.npy')
            regions_filename = os.path.join(FEATURE_DIR, image, classy + image + '_cluster_regions_' + str(QFLAGS.clusters) + '_clusters.pickle')
            
            # regions was the last file added, so if it exists then all must exist,
            if os.path.exists(regions_filename) and False:
                cprint('Loading cell locations and KMeans regions...', 'grey', 'on_white')
                cell_locations = np.load(locations_filename)
                region_cell_counts = np.load(regions_filename)
                vertices = np.load(vertices_filename)
                with open(regions_filename, 'rb') as handle:
                    regions = pickle.load(handle)
            else:
                cprint('Calculating voronoi regions ...', 'yellow')
                vor = Voronoi(centers)
                regions, vertices = voronoi_finite_polygons_2d(vor)
                for region in regions:
                    polygon = vertices[region]
                    plt.fill(*zip(*polygon), alpha=0.1)

                cprint('Placing cells in regions...', 'yellow')
                cell_locations = place_cells_in_regions(regions, vertices, coord_space, tumor_space)
                region_cell_counts = fill_region_cell_counts(centers, cell_locations)
                cprint('Saving KMeans cluster data...', 'green')
                np.save(locations_filename, cell_locations)
                np.save(regions_filename, region_cell_counts)
                np.save(vertices_filename, vertices)
                with open(regions_filename, 'wb') as handle:
                    pickle.dump(regions, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Region Cell Counts:
            # [center_x, center_y, immune count, tumor count, imm + tumor]

            if os.path.exists(densities_filename) and not QFLAGS.overwrite_saved:
                largest_densities = np.load(densities_filename)
            else:
                cprint('Calculating region densities...', 'yellow')
                largest_densities = greatest_n_cluster_densities(region_cell_counts, QFLAGS.dense_thresh / 10, QFLAGS.clusters, data[subject]['status'])
                cprint('Saving cluster information', 'green')
                np.save(densities_filename, largest_densities)
            densest_regions = find_densest_regions(largest_densities, region_cell_counts)

            if create_dense_delaunay:
                if subject not in delaunay:
                    continue
                if image not in delaunay[subject]:
                    continue
                dense_delaunay[subject][image] = find_delaunay_clusters_in_densest_regions(regions, vertices, densest_regions, delaunay[subject][image])

            if data[subject]['status']:
                # rec_center_ratio.append(min_center / max_center)
                # rec_center_mean.append(mean_center)
                rec_density.append(np.mean(largest_densities))
            else:
                # nr_center_ratio.append(min_center / max_center)
                # nr_center_mean.append(mean_center)
                nr_density.append(np.mean(largest_densities))

            if QFLAGS.display_clusters:
                # Plot all points
                plt.subplot(1,2,1)
                plt.scatter(tumor_space[:,0], max(coord_space[:,1]) - tumor_space[:,1], s=1, c='lightsteelblue', alpha = 0.5)
                plt.scatter(coord_space[:,0], max(coord_space[:,1]) - coord_space[:,1], s=4, c=labels, cmap='plasma')
                plt.scatter(centers[:, 0], max(coord_space[:,1]) - centers[:, 1], c='black', s=200, alpha=0.8)
                dist = (int(vor.max_bound[0] - vor.min_bound[0]), int(vor.max_bound[1] - vor.min_bound[1]))
                plt.xlim(vor.min_bound[0] - 0.25*dist[0], vor.max_bound[0] + 0.25*dist[0])
                plt.ylim(vor.min_bound[1] - 0.25*dist[1], vor.max_bound[1] + 0.25*dist[1])
                plt.title(image + ', status = ' + str(data[subject]['status']))
                plt.legend(loc='upper right')
                plt.subplot(1,2,2)
                img = mpimg.imread(os.path.join(ORIGINAL_IMAGE_DIR, image + '.tif'))
                plt.imshow(img)
                # plt.scatter(coord_space[:,0],coord_space[:,1], s=2, c=labels, cmap='plasma', alpha = 0.5)
                plt.scatter(centers[:, 0],centers[:, 1], c='black', s=100, alpha=0.8)
                plt.show()
            if not os.path.exists(cell_locs_filename):
                cell_locs[subject][image] = {}
                cell_locs[subject][image]['cell_locations'] = cell_locations
                cell_locs[subject][image]['region_counts'] = region_cell_counts
    # plt.subplot(3,1,1)
    # plt.hist(rec_center_ratio, stacked=True, bins=25, alpha=0.5, label='Rec immune', density=True)
    # plt.hist(nr_center_ratio, stacked=True, bins=25, alpha=0.5, label="nr immune", density=True)
    # plt.title('min/max ratio')
    # plt.legend(loc='upper right')

    # plt.subplot(3,1,2)
    # plt.hist(rec_center_mean, stacked=True, bins=25, alpha=0.5,label='Rec tumor', density=True)
    # plt.hist(nr_center_mean, stacked=True, bins=25, alpha=0.5, label="nr tumor", density=True)
    # plt.legend(loc='upper right')
    # plt.title('center_distances')

    # plt.subplot(3,1,3)
    # plt.hist(rec_density, stacked=True, bins=25, alpha=0.5,label='Rec tumor', density=True)
    # plt.hist(nr_density, stacked=True, bins=25, alpha=0.5, label="nr tumor", density=True)
    # plt.legend(loc='upper right')
    # plt.title('densities')


    threshold_accuracies(rec_density, nr_density, str(QFLAGS.clusters) + ' Cluster Densities, ' + str(QFLAGS.dense_thresh / 10) + ' threshold')
    # plt.show()

    if not os.path.exists(cell_locs_filename) and not QFLAGS.test:
        with open(cell_locs_filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if create_dense_delaunay:
        with open(dense_delaunay_filename, 'wb') as handle:
            pickle.dump(dense_delaunay, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return cell_locs, dense_delaunay

def find_delaunay_clusters_in_densest_regions(regions, vertices, densest_regions, delaunay):
    delaunay_in_regions = {}
    for dense_region in densest_regions:
        region_name = 'region_' + str(dense_region)
        delaunay_in_regions[region_name] = {}
        for cell_class in delaunay:
            delaunay_in_regions[region_name][cell_class] = {}
            polygon = Polygon(vertices[regions[dense_region]])

            for d_cluster in delaunay[cell_class]:
                if isinstance(d_cluster, tuple):
                    point = Point(d_cluster[0], d_cluster[1])
                    if polygon.contains(point):
                        delaunay_in_regions[region_name][cell_class][d_cluster] = {}
                        delaunay_in_regions[region_name][cell_class][d_cluster] = delaunay[cell_class][d_cluster]
    return delaunay_in_regions

def find_densest_regions(largest_densities, region_cell_counts):
    densest_regions = []
    for i in np.arange(len(region_cell_counts)):
        single_region_density = region_cell_counts[i,2] / region_cell_counts[i,4]
        if single_region_density in largest_densities:
            densest_regions.append(i)
    return densest_regions

def threshold_accuracies(rec_list, nr_list, title="[No title]"):
    plt.clf()
    rec_labels = np.ones(len(rec_list))
    nr_labels = np.zeros(len(nr_list))
    rec_values = np.array(rec_list)
    nr_values = np.array(nr_list)


    labels = np.concatenate((rec_labels, nr_labels))
    values = np.concatenate((rec_values, nr_values))

    # thresholds = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    thresh, acc, sens, spec = run_thresholds(0, 0.5, 501, labels, values)
    plt.plot(thresh, acc, lw=2)
    plt.ylim(0,1)
    plt.title(title)
    plt.axhline(max(acc), c='red')
    plt.xlabel('Immune cell count / Total cell count')
    plt.ylabel('Accuracy: ' "{0:03f}".format(max(acc)))
    # plt.text(.1, .8, 'Accuracy: ' "{0:03f}".format(max(acc)))
    figure_dir_name = os.path.join(FEATURE_DIR, 'figures')
    figure_name = os.path.join(figure_dir_name, "{0:03f}".format(max(acc)) + '_accuracy_' + str(QFLAGS.clusters) + 'clusters_' + str(QFLAGS.dense_thresh / 10) + 'threshold.png')
    os.makedirs(figure_dir_name, exist_ok=True)
    plt.savefig(figure_name)

def greatest_n_cluster_densities(cells, percent, k, status):
    densities = np.zeros(len(cells))
    num = 1
    for row in cells:
        # print("Cluster " + str(num) + ' immune density: ' + "{0:.2f}".format(row[2] / row[4]))
        densities[num - 1] = row[2] / row[4] 
        num += 1
    i = int(-(percent * k))
    largest_densities_ind = np.argpartition(densities, i)[i:]
    largest_densities = densities[largest_densities_ind]
    print('Mean density: ' + "{0:.3f}".format(np.mean(largest_densities)) + ', threshold: ' + str(percent) + ', status: ' + str(status))
    return largest_densities

def find_nearest_cluster_centers(centers, k):
    closest_centers = np.zeros([len(centers), k, 2])
    row_num = 0
    for row in centers:
        distances = np.sqrt(np.sum((centers - row)**2, axis=1))
        idx = np.argpartition(distances, k + 1)
        smallest = distances[idx[:k+1]]
        smallest = smallest[smallest > 0]
        for i in np.arange(len(smallest)):
            where = int(np.where(distances == smallest[i])[0][0])
            closest_centers[row_num, i, 1] = where + 1
        # pdb.set_trace()
        closest_centers[row_num,:, 0] = smallest
        row_num += 1
    return closest_centers

def fill_region_cell_counts(centers, cells):
    cells_in_regions = np.zeros([QFLAGS.clusters, 5])
    cells_in_regions[:,:2] = centers.astype(int)
    n = 0
    for row in cells_in_regions:
        n += 1
        in_this_region = cells[cells[:,3]==n]
        cells_in_regions[n-1, 2] = np.sum(in_this_region[:,2])
        cells_in_regions[n-1, 3] = len(in_this_region) - cells_in_regions[n-1, 2]
        cells_in_regions[n-1, 4] = len(in_this_region)
    return cells_in_regions.astype(int)

def place_cells_in_regions(regions, vertices, coord_space, tumor_space):
    cell_locations = np.zeros([(coord_space.shape[0] + tumor_space.shape[0]), 4]).astype(int)
    cell_locations[:coord_space.shape[0],:2] = coord_space[:,:].astype(int)
    cell_locations[:coord_space.shape[0],2] = 1
    cell_locations[coord_space.shape[0]:,:2] = tumor_space[:,:].astype(int)
    cell_locations = cell_locations.astype(int)
    region_num = 0
    for region in regions:
        region_num += 1
        polygon = Polygon(vertices[region])
        row_num = 0
        for row in cell_locations:
            point = Point(row[0], row[1])
            if polygon.contains(point):
                cell_locations[row_num, 3] = region_num
            row_num += 1
    return cell_locations

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def verify_row_contents(row):
    skip_row = False
    for feature in FEATURES:
        if feature not in row:
            skip_row = True
            break
    return skip_row

def add_delaunay_histogram_features(cluster_dict, subject):
    for image in subject:
        if image == 'status':
            continue
        for cell_class in subject[image]:
            class_info = {}
            if cell_class not in cluster_dict:
                cluster_dict[cell_class] = {}

                cluster_dict[cell_class]['small_cluster_percent'] = []
                cluster_dict[cell_class]['small_cluster_cell_percent'] = []
                cluster_dict[cell_class]['delaunay_max_area_zeroed_mean'] = []
                cluster_dict[cell_class]['delaunay_max_area_removed_mean'] = []
                cluster_dict[cell_class]['small_cluster_cells'] = []
                cluster_dict[cell_class]['large_cluster_cells'] = []
                cluster_dict[cell_class]['total_cluster_cells'] = []

            small_cluster_percent = subject[image][cell_class]['small_cluster_count'] / (subject[image][cell_class]['small_cluster_count'] + subject[image][cell_class]['large_cluster_count'])

            class_cluster_cells = np.sum(subject[image][cell_class]['small_cluster_size']) + np.sum(subject[image][cell_class]['large_cluster_size'])
            class_small_cluster_cells = np.sum(subject[image][cell_class]['small_cluster_size'])
            class_large_cluster_cells = np.sum(subject[image][cell_class]['large_cluster_size'])

            cluster_dict[cell_class]['small_cluster_cells'].append(class_small_cluster_cells)
            cluster_dict[cell_class]['large_cluster_cells'].append(class_large_cluster_cells)
            cluster_dict[cell_class]['total_cluster_cells'].append(class_cluster_cells)



            small_cluster_cell_percent = np.sum(subject[image][cell_class]['small_cluster_size']) / (np.sum(subject[image][cell_class]['small_cluster_size']) + np.sum(subject[image][cell_class]['large_cluster_size']))
            
            cluster_dict[cell_class]['small_cluster_percent'].append(small_cluster_percent)
            cluster_dict[cell_class]['small_cluster_cell_percent'].append(small_cluster_cell_percent)
            for cluster in subject[image][cell_class]:
                if isinstance(cluster, tuple):
                    one_cluster = subject[image][cell_class][cluster]
                    for feature in one_cluster:
                        if feature not in cluster_dict[cell_class]:
                            cluster_dict[cell_class][feature] = []
                        if feature not in class_info:
                            class_info[feature] = []
                        class_info[feature].append(subject[image][cell_class][cluster][feature])
            for feature in class_info:
                cluster_dict[cell_class][feature].append(np.mean(class_info[feature]))
            cluster_dict[cell_class]['delaunay_max_area_zeroed_mean'].append(np.mean(convert_nan_to_zero(np.array(class_info['delaunay_max_area']))))
            cluster_dict[cell_class]['delaunay_max_area_removed_mean'].append(np.mean(remove_nan_values(np.array(class_info['delaunay_max_area']))))
    return 1

def add_dense_histogram_features(cluster_dict, subject):
    extra_features = ['small_cluster_percent', 'small_cluster_cell_percent', 'small_cell_count', 'total_cell_count'] #, 'delaunay_max_area_zeroed_mean', 'delaunay_max_area_removed_mean']
    for image in subject:
        if image == 'status':
            continue
        
        within_image_features = {}

        for dense_region in subject[image]:
            for cell_class in subject[image][dense_region]:
                #Organizing
                if cell_class not in within_image_features:
                    within_image_features[cell_class] = {}
                    for feature in extra_features:
                        within_image_features[cell_class][feature] = []
                if cell_class not in cluster_dict:
                    cluster_dict[cell_class] = {}
                    for feature in extra_features:
                        cluster_dict[cell_class][feature] = []

                #Adding data
                total_region_class_clusters = len(subject[image][dense_region][cell_class])
                total_region_class_cell_count = 0
                region_class_small_clusters = 0
                region_class_small_cell_count = 0
                region_class_large_cell_count = 0
                for cluster in subject[image][dense_region][cell_class]:
                    size = subject[image][dense_region][cell_class][cluster]['size']

                    total_region_class_cell_count += size
                    if size <= QFLAGS.small_d_cluster:
                        region_class_small_clusters += 1
                        region_class_small_cell_count += size
                        total_region_class_cell_count += size
                    else:
                        region_class_large_cell_count += size
                        total_region_class_cell_count += size

                    for feature in subject[image][dense_region][cell_class][cluster]:
                        if feature not in cluster_dict[cell_class]:
                            cluster_dict[cell_class][feature] = []
                        if feature not in within_image_features[cell_class]:
                            within_image_features[cell_class][feature] = []
                            
                        within_image_features[cell_class][feature].append(subject[image][dense_region][cell_class][cluster][feature])
                within_image_features[cell_class]['small_cluster_percent'].append(region_class_small_clusters / total_region_class_clusters)
                within_image_features[cell_class]['small_cluster_cell_percent'].append(region_class_small_cell_count / total_region_class_cell_count)
                within_image_features[cell_class]['small_cell_count'].append(region_class_small_cell_count)
                within_image_features[cell_class]['total_cell_count'].append(total_region_class_cell_count)

        for cell_class in within_image_features:
            for feature in within_image_features[cell_class]:
                if feature == 'small_cell_count' or feature == 'total_cell_count':
                    cluster_dict[cell_class][feature].append(np.sum(within_image_features[cell_class][feature]))
                else:
                    cluster_dict[cell_class][feature].append(np.mean(within_image_features[cell_class][feature]))
    return 1

def create_delaunay_features(delaunay, subject, image, row):
 
####################################################
##                                                ##
##  Part of Delaunay data structure, pickle must  ##
##  be overwritten if this section changes        ##
##                                                ##
####################################################

    # Organizing
    row_class = row['Class']
    if row_class in OMIT_CLASS:
        return 1
    if 'Cluster  mean: Centroid X' not in row:
        return 1
    center_x = float(row['Cluster  mean: Centroid X'])
    center_y = float(row['Cluster  mean: Centroid Y'])
    if isnan(center_x) or isnan(center_y):
        return 1
    else:
        cluster_location = (int(center_x), int(center_y))
    
    if subject not in delaunay:
        delaunay[subject] = {}
    if image not in delaunay[subject]:
        delaunay[subject][image] = {}
    if row_class not in delaunay[subject][image]:
        delaunay[subject][image][row_class] = {}
        delaunay[subject][image][row_class]['small_cluster_count'] = 0
        delaunay[subject][image][row_class]['small_cluster_size'] = []
        delaunay[subject][image][row_class]['large_cluster_count'] = 0
        delaunay[subject][image][row_class]['large_cluster_size'] = []
    
    # Adding cluster features
    if cluster_location not in delaunay[subject][image][row_class]:
        if int(row['Cluster  size']) <= QFLAGS.small_d_cluster:
            delaunay[subject][image][row_class]['small_cluster_count'] += 1
            delaunay[subject][image][row_class]['small_cluster_size'].append(int(row['Cluster  size']))
        else:
            delaunay[subject][image][row_class]['large_cluster_count'] += 1
            delaunay[subject][image][row_class]['large_cluster_size'].append(int(row['Cluster  size']))
        delaunay[subject][image][row_class][cluster_location] = {}
        delaunay_feature_list(delaunay[subject][image][row_class][cluster_location], row)

    return 1


def delaunay_feature_list(cluster, row):
    cluster['mean_cell_area'] = float(row['Cluster  mean: Cell: Area'])
    cluster['size'] = int(row['Cluster  size'])
    cluster['hematoxylin_OD_mean'] = float(row['Cluster  mean: Nucleus: Hematoxylin OD mean'])
    cluster['eosin_OD_mean'] = float(row['Cluster  mean: Nucleus: Eosin OD mean'])

    # I know these two are switched. I'm pretty sure QuPath put these results in the wrong category.
    cluster['delaunay_max_area'] = float(row['Cluster  mean: Delaunay: Mean triangle area'])
    cluster['delaunay_mean_area'] = float(row['Cluster  mean: Delaunay: Max triangle area'])


    return 1


def value_for_csv(rec, non, labels, lo, hi, num):
    values = np.concatenate((rec, non))
    thresh, acc, sens, spec = run_thresholds(lo, hi, num, labels, values)
    max_acc = max(acc)
    area = 0 
    for i in np.arange(len(sens)):
        if i == 0:
            area = area + sens[i]*spec[i]
        else:
            area = area + ((sens[i-1] + sens[i])*(spec[i] - spec[i-1])/2)

    return [max_acc, area]


def show_delaunay_histogram(delaunay):
    RE = {}
    NR = {}
    n = 0
    plt.clf()
    for subject in delaunay:
        if 'status' not in delaunay[subject]:
            continue
        if delaunay[subject]['status']:
            add_delaunay_histogram_features(RE, delaunay[subject])
        else:
            add_delaunay_histogram_features(NR, delaunay[subject])
    # for cell_class in RE:
    #     cprint('Plotting ' + cell_class, 'green')
    #     for feature in RE[cell_class]:
    #         if 'Centroid' in feature:
    #             continue
    #         else:
    #             n += 1
    #             plt.subplot(4,5,n)
    #             rec = convert_nan_to_zero(np.array(RE[cell_class][feature]))
    #             non = convert_nan_to_zero(np.array(NR[cell_class][feature]))
    #             bins = get_bins(rec / len(rec), non / len(non))
    #             # plt.hist(non, stacked=True, alpha=0.5, label='Nonrecurrent', density=True)
    #             # plt.hist(rec, stacked=True, alpha=0.5, label="Recurrent", density=True)
    #             plt.hist(non, alpha=0.5, label='Nonrecurrent')
    #             plt.hist(rec, alpha=0.5, label="Recurrent")
    #             plt.xlabel(feature + ' -- ' + cell_class)

    re_imm_large_count = np.array(RE['Immune cells']['large_cluster_cells'])
    nr_imm_large_count = np.array(NR['Immune cells']['large_cluster_cells'])

    re_imm_small_count = np.array(RE['Immune cells']['small_cluster_cells'])
    nr_imm_small_count = np.array(NR['Immune cells']['small_cluster_cells'])

    re_cross_class_total = np.array(RE['Immune cells']['total_cluster_cells']) + np.array(RE['Tumor']['total_cluster_cells'])
    nr_cross_class_total = np.array(NR['Immune cells']['total_cluster_cells']) + np.array(NR['Tumor']['total_cluster_cells'])

    re_tum_small_count = np.array(RE['Tumor']['small_cluster_cells'])
    nr_tum_small_count = np.array(NR['Tumor']['small_cluster_cells'])

    re_tum_large_count = np.array(RE['Tumor']['large_cluster_cells'])
    nr_tum_large_count = np.array(NR['Tumor']['large_cluster_cells'])

    re_labels = np.ones(len(re_imm_large_count))
    nr_labels = np.zeros(len(nr_imm_large_count))
    labels = np.concatenate((re_labels, nr_labels))

    stats = []

    re_portion_one = re_imm_large_count / (re_imm_large_count + re_imm_small_count)
    nr_portion_one = nr_imm_large_count / (nr_imm_large_count + nr_imm_small_count)
    stats.append([re_portion_one, nr_portion_one, 0, 1, 1001])

    re_portion_two = (re_imm_large_count)  / re_cross_class_total
    nr_portion_two = (nr_imm_large_count)  / nr_cross_class_total
    stats.append([re_portion_two, nr_portion_two, 0, 1, 1001])

    re_portion_three = re_imm_small_count  / re_cross_class_total
    nr_portion_three = nr_imm_small_count  / nr_cross_class_total
    stats.append([re_portion_three, nr_portion_three, 0, 1, 1001])

    re_portion_four = re_tum_large_count / (re_tum_large_count + re_tum_small_count)
    nr_portion_four = nr_tum_large_count / (nr_tum_large_count + nr_tum_small_count)
    stats.append([re_portion_four, nr_portion_four, 0, 1, 1001])

    re_portion_five = re_imm_large_count / (re_tum_large_count)
    nr_portion_five = nr_imm_large_count / (nr_tum_large_count)
    stats.append([re_portion_five, nr_portion_five, 0, 1, 1001])

    re_portion_six = re_imm_large_count / re_imm_small_count
    nr_portion_six = nr_imm_large_count / nr_imm_small_count
    stats.append([re_portion_six, nr_portion_six, 0, 10, 10001])

    plot_row = [QFLAGS.classifier]
    if QFLAGS.yale:
        plot_row.append('yale')
        plot_row.append(QFLAGS.small_d_cluster)
    else:
        plot_row.append(QFLAGS.delaunay_radius)
        plot_row.append(QFLAGS.small_d_cluster)

    for stat in stats:
        csv_vals = value_for_csv(stat[0], stat[1], labels, stat[2], stat[3], stat[4])
        plot_row.append(csv_vals[0])
        plot_row.append(csv_vals[1])

    with open(DATA_STORAGE, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(plot_row)



    # plt.subplot(2,1,1)
    # plt.hist(nr_portion_six, alpha=0.5, label='Nonrecurrent')
    # plt.hist(re_portion_six, alpha=0.5, label="Recurrent")

    # re_labels = np.ones(len(re_portion_six))
    # nr_labels = np.zeros(len(nr_portion_six))

    # labels = np.concatenate((re_labels, nr_labels))
    # values = np.concatenate((re_portion_six, nr_portion_six))
    # plt.legend(loc='upper right')

    # plt.subplot(2,1,2)
    # thresh, acc = run_thresholds(0, 10, 10001, labels, values)
    # max_acc = max(acc)
    # plt.plot(thresh, acc, lw=2)
    # plt.ylim(0,1)
    # plt.title('small cluster <= ' + str(QFLAGS.small_d_cluster) + ' accuracy: ' + "{:02f}".format(max_acc))
    # plt.axhline(max_acc, c='red')
    # plt.xlabel('Imm cells in large cluster / total immune')
    # plt.ylabel('Accuracy')
    # plt.show()
    # pdb.set_trace()
    # plt.subplot(2,3,2)
    # values = np.concatenate((re_portion_two, nr_portion_two))

    
    # plt.legend(loc='upper right')

    # plt.subplot(2,3,5)
    # thresh, acc = run_thresholds(0, 1, 1001, labels, values)
    # max_acc = max(acc)
    # plt.plot(thresh, acc, lw=2)
    # plt.ylim(0,1)
    # plt.title('small cluster <= ' + str(QFLAGS.small_d_cluster) + ' accuracy: ' + "{:02f}".format(max_acc))
    # plt.axhline(max_acc, c='red')
    # plt.xlabel('(large imm) / (all imm + all tumor)')
    # plt.ylabel('Accuracy')

    # plt.subplot(2,3,3)
    # plt.hist(nr_portion_three, alpha=0.5, label='Nonrecurrent')
    # plt.hist(re_portion_three, alpha=0.5, label="Recurrent")

    # values = np.concatenate((re_portion_three, nr_portion_three))

    
    # plt.legend(loc='upper right')

    # plt.subplot(2,3,6)
    # thresh, acc = run_thresholds(0, 1, 1001, labels, values)
    # max_acc = max(acc)
    # plt.plot(thresh, acc, lw=2)
    # plt.ylim(0,1)
    # plt.title('small cluster <= ' + str(QFLAGS.small_d_cluster) + ' accuracy: ' + "{:02f}".format(max_acc))
    # plt.axhline(max_acc, c='red')
    # plt.xlabel('(small imm) / (all imm + all tumor)')
    # plt.ylabel('Accuracy')

    return 1

def dense_delaunay_histogram(d_delaunay):
    RE = {}
    NR = {}
    n = 0
    plt.clf()
    for subject in d_delaunay:
        if d_delaunay[subject]['status']:
            add_dense_histogram_features(RE, d_delaunay[subject])
        else:
            add_dense_histogram_features(NR, d_delaunay[subject])
    for cell_class in RE:
        cprint('Plotting ' + cell_class, 'green')
        for feature in RE[cell_class]:
            if 'Centroid' in feature:
                continue
            else:
                n += 1
                plt.subplot(7,8,n)
                rec = convert_nan_to_zero(np.array(RE[cell_class][feature]))
                non = convert_nan_to_zero(np.array(NR[cell_class][feature]))
                bins = get_bins(rec / len(rec), non / len(non))
                # plt.hist(non, stacked=True, alpha=0.5, label='Nonrecurrent', density=True)
                # plt.hist(rec, stacked=True, alpha=0.5, label="Recurrent", density=True)
                plt.hist(non, alpha=0.5, label='Nonrecurrent')
                plt.hist(rec, alpha=0.5, label="Recurrent")
                plt.xlabel(feature + ' -- ' + cell_class)
    plt.legend(loc='upper right')
    # plt.show()
    # pdb.set_trace()
    # plt.clf()
    re_small_cluster_tumor_percent = np.array(RE['Tumor']['small_cluster_percent'])
    nr_small_cluster_tumor_percent = np.array(NR['Tumor']['small_cluster_percent'])

    re_small_cluster_imm_percent = np.array(RE['Immune cells']['small_cluster_percent'])
    nr_small_cluster_imm_percent = np.array(NR['Immune cells']['small_cluster_percent'])

    rec = re_small_cluster_imm_percent 
    non = nr_small_cluster_imm_percent 

    # plt.hist(non, alpha=0.5, label='Nonrecurrent')
    # plt.hist(rec, alpha=0.5, label="Recurrent")
    # plt.legend(loc='upper right')
    # pdb.set_trace()
    # plt.show()

    # plt.clf()
    re_large_cluster_tumor_count = np.array(RE['Tumor']['total_cell_count']) - np.array(RE['Tumor']['small_cell_count'])
    nr_large_cluster_tumor_count = np.array(NR['Tumor']['total_cell_count']) - np.array(NR['Tumor']['small_cell_count'])

    re_large_cluster_imm_count = np.array(RE['Immune cells']['total_cell_count']) - np.array(RE['Immune cells']['small_cell_count'])
    nr_large_cluster_imm_count = np.array(NR['Immune cells']['total_cell_count']) - np.array(NR['Immune cells']['small_cell_count'])

    re_small_cluster_tumor_count = np.array(RE['Tumor']['small_cell_count'])
    nr_small_cluster_tumor_count = np.array(NR['Tumor']['small_cell_count'])

    re_small_cluster_imm_count = np.array(RE['Immune cells']['small_cell_count'])
    nr_small_cluster_imm_count = np.array(NR['Immune cells']['small_cell_count'])

    rec = re_large_cluster_imm_count / (re_large_cluster_tumor_count + re_large_cluster_imm_count)
    non = nr_large_cluster_imm_count / (nr_large_cluster_tumor_count + nr_large_cluster_imm_count)

    rec

    # plt.hist(non, alpha=0.5, label='Nonrecurrent')
    # plt.hist(rec, alpha=0.5, label="Recurrent")
    # plt.legend(loc='upper right')
    
    # plt.show()

    plt.clf()
    # rec = re_small_cluster_imm_count / (re_large_cluster_imm_count + re_small_cluster_tumor_count)
    # non = nr_small_cluster_imm_count / (nr_large_cluster_imm_count + nr_small_cluster_tumor_count)
    

    # rec = np.array(RE['Immune cells']['hematoxylin_OD_mean']) / np.array(RE['Tumor']['eosin_OD_mean'])
    # non = np.array(NR['Immune cells']['hematoxylin_OD_mean']) / np.array(NR['Tumor']['eosin_OD_mean'])


    re_labels = np.ones(len(rec))
    nr_labels = np.zeros(len(non))
    labels = np.concatenate((re_labels, nr_labels))
    values = np.concatenate((rec, non))
    plt.hist(non, alpha=0.5, label='Nonrecurrent')
    plt.hist(rec, alpha=0.5, label="Recurrent")
    thresh, acc, sens, spec = run_thresholds(0, 1, 1001, labels, values)
    cprint('MAX ACCURACY: ' + str(max(acc)), 'white', 'on_green')
    plt.show()

    return 1

def row_has_nan(row):
    skip_set = ['Centroid X', 'Centroid Y', 'Cell: Area']
    for feature in skip_set:
        if feature not in row:
            return True
        else:
            if row[feature] == 'NaN':
                return True
    return False

def main(subject_id = None, image_name=None, image_processor=False):
    # cprint('radius: ' + str(QFLAGS.delaunay_radius) + 'px, small cluster: ' + str(QFLAGS.small_d_cluster), 'white', 'on_green')
    #pdb.set_trace()
    classy = QFLAGS.classifier

    if QFLAGS.all_features:
        data_filename = os.path.join(FEATURE_DIR, classy + ADD_NAME + 'all_features_all_data.pickle')
    else:
        data_filename = os.path.join(FEATURE_DIR, classy + ADD_NAME + 'all_data.pickle')

    if image_processor:
        detections_file = image_name + '_Detectionstxt.txt'
        if not os.path.exists(os.path.join(DETECTIONS, detections_file)):
            return None

    create_delaunay = True
    delaunay_filename = os.path.join(FEATURE_DIR, classy + ADD_NAME + '_' + str(QFLAGS.small_d_cluster) + 'smallCluster_' + str(QFLAGS.delaunay_radius) + 'radius_delaunay.pickle')
    if QFLAGS.yale:
        delaunay_filename = os.path.join(FEATURE_DIR, 'yale_' + str(QFLAGS.small_d_cluster) + 'smallCluster_radius_delaunay.pickle')
    if QFLAGS.load_pickles and os.path.exists(delaunay_filename) and not image_processor:
        print('Loading delaunay features')
        with open(delaunay_filename, 'rb') as handle:
            delaunay = pickle.load(handle)
            create_delaunay = False
    else:
        delaunay = {}

    if not create_delaunay:
        if QFLAGS.load_pickles and os.path.exists(data_filename) and not image_processor:
            print('Loading saved cell data')
            with open(data_filename, 'rb') as handle:
                data = pickle.load(handle)
    else:
        data = {}
        if image_processor:
            # Current run called from image_processor.py
            detections_file = image_name + '_Detectionstxt.txt'
            
            with open(os.path.join(DETECTIONS, detections_file), 'r') as f:

                subject = 'subj'
                data[subject] = {}
                data[subject][image_name] = {}
                reader = csv.DictReader(f, delimiter='\t')
                n = 0
                for row in reader:
                    if row['Class'] not in OMIT_CLASS:

                        if row_has_nan(row):
                            continue
                        n += 1
                        cell_name = '{0:06}'.format(n)
                        data[subject][image_name][cell_name] = {}
                        add_cell_features(data[subject][image_name][cell_name], row)
                        create_delaunay_features(delaunay, subject, image_name, row)
                if n == 0:
                    pdb.set_trace()
        else:
            # Current run called directly
            subject_load_list = []
            image_load_list = []
            if QFLAGS.subject_list:
                with open(QFLAGS.subject_list, 'r') as load_file:
                # load_file = open(load_list, 'r')
                    reader = csv.reader(load_file, delimiter=',')
                    _ = next(reader) #discard header line
                    for line in reader:
                        if line[0] not in subject_load_list:
                            subject_load_list.append(line[0])
                        image_load_list.append(line[1][:-4])
            # Read image_to_subject_ID csv file to get subject dictionary
            image_to_ID_csv_file = open(os.path.join('/data/recurrence_seq_lstm/image_data',"image_to_subject_ID.csv"),"r")
            reader = csv.reader(image_to_ID_csv_file, delimiter=",")
            _ = next(reader) # discard header line
            image_to_ID_dict = {}
            for line in reader:
                image_to_ID_dict[line[0].split(".")[0]] = line[1]

            for (dirpath, dirnames, filenames) in os.walk(DETECTIONS):
                count = 0
                shuffle(filenames)
                for file in filenames:
                    image_name = file[:-18]
                    if not file.endswith('.txt'):
                        # cprint('skipping ' + file + ', not a text file (' + str(count) + '/' + str(len(filenames)) + ')' + FILLER, 'yellow', end="\r")
                        continue
                    
                    if QFLAGS.test:
                        if count > QFLAGS.test:
                            break
                    if 'bleached' in file:
                        # cprint('skipping ' + file + ' (' + str(count) + '/' + str(len(filenames)) + ')' + FILLER, 'yellow', end="\r")
                        continue
                    n = 0
                    
                    if image_name not in image_to_ID_dict:
                        cprint(image_name + ' not in image_to_ID_dict!!!!', 'white', 'on_red')
                        pdb.set_trace()
                    subject = image_to_ID_dict[image_name]
                    if QFLAGS.subject_list:
                        if subject not in subject_load_list:
                            continue
                        elif image_name not in image_load_list:
                            continue
                        else:
                            count +=1
                            cprint('scanning ' + image_name + ' (' + str(count) + '/' + str(len(image_load_list)) + ')' + FILLER, 'yellow', 'on_white', end="\r")

                    if subject not in data:
                        data[subject] = {}
                    if image_name not in data[subject]:
                        data[subject][image_name] = {}
                    if file.endswith('.txt') and 'bleached' not in file:
                        # count += 1
                        # cprint('scanning ' + file + ' (' + str(count) + '/' + str(len(filenames)) + ')' + FILLER, 'yellow', end="\r")
                        with open(os.path.join(dirpath, file), 'r') as f:
                            reader = csv.DictReader(f, delimiter='\t')
                            # print(reader.fieldnames)
                            for row in reader:
                                if row_has_nan(row):
                                    continue
                                n += 1
                                cell_name = '{0:06}'.format(n)
                                skip_row = verify_row_contents(row)
                                if skip_row:
                                    cprint('skipping ' + image_name + FILLER, 'red')
                                    break
                                data[subject][image_name][cell_name] = {}
                                if QFLAGS.all_features:
                                    add_all_cell_features(data[subject][image_name][cell_name], row)
                                else:
                                    add_cell_features(data[subject][image_name][cell_name], row)

                                if create_delaunay:
                                    create_delaunay_features(delaunay, subject, image_name, row)

                            
        status = {}
        if not image_processor:
            with open(STATUS_CSV, 'r') as csvfile:
                cprint('\nAdding recurrence status...', 'green', 'on_white')
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] in data:
                        data[row[0]]['status'] = int(row[1])
                        for image in data[row[0]]:
                            if not 'status':
                                data[row[0]][image]['status'] = int(row[1])
                    if row[0] in delaunay:
                        delaunay[row[0]]['status'] = int(row[1])

        if not QFLAGS.test and not image_processor:
            cprint('\nSaving [data] array', 'yellow')
            with open(data_filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if create_delaunay:
                cprint('Saving [delaunay] array', 'yellow')
                with open(delaunay_filename, 'wb') as handle:
                    pickle.dump(delaunay, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

            cprint('Done!', 'green')

    if image_processor:
        image_dict = {}
        img_delaunay = {}
        for cell in data['subj'][image_name]:
            image_dict[cell] = data['subj'][image_name][cell]
            img_delaunay = delaunay['subj'][image_name]
        return image_dict, img_delaunay

    # Visualization
    if QFLAGS.all_cells:
        RE = {}
        NR = {}
        for subject in data:
            if 'status' in data[subject]:
                if data[subject]['status']:
                    add_to_cell_dict(RE, data[subject])
                else:
                    add_to_cell_dict(NR, data[subject])
        show_base_features_histogram(RE, NR)


    # load_image_masks(data)


    # delaunay:
    #   --[subject]
    #       --[image]
    #           --[cell classification]
    #               --[delaunay centroid]
    #                   --[cluster info]
    # e.g. delaunay['00-05']['00_05_D_3_1']['Tumor'][(682, 5699)]['size']

    # d_delaunay:
    #   --[subject]
    #       --[image]
    #           --[dense region]
    #               --[cell classification]
    #                   --[delaunay centroid]
    #                       --[cluster info]

    # cell_locations:
    #   --[subject]
    #       --[image]
    #           --[region counts] -> [cluster number, distances to 5 closest regions]
    #           --[cell_locations] -> [x, y, label, cluster number]
    cell_locations, d_delaunay = k_means_clustering_and_features(data, delaunay)
    if QFLAGS.show_delaunay:
        show_delaunay_histogram(delaunay)
        dense_delaunay_histogram(d_delaunay)


# Main body
if __name__ == '__main__':
    main()