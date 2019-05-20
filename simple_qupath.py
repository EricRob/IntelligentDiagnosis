#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import csv
import os
from math import isnan

# Function declarations

def detections(subject_id, image_name, config):
    #
    # This is the method that can be called externally
    # from the image_processor (which is now called preprocess_lstm)
    #

    return main(subject_id, image_name, image_processor=True, config=config)

def add_cell_features(detection, row):

    #
    # Extract the useful information from the large QuPath detection file.
    #

    detection['cell_area'] = int(row['Cell: Area'])
    detection['nuc_area'] = int(row['Nucleus: Area'])
    detection['area_ratio'] = float(row['Nucleus/Cell area ratio'])
    detection['total_area'] = detection['cell_area'] + detection['nuc_area']
    detection['cell_perimeter'] = float(row['Cell: Perimeter'])
    detection['class'] = row['Class']
    detection['x'] = int(float(row['Centroid X']))
    detection['y'] = int(float(row['Centroid Y']))

    return 1


def create_delaunay_features(delaunay, subject, image, row, config):

    # Organizing
    row_class = row['Class']
    if row_class in config.omit_class:
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
        if int(row['Cluster  size']) <= config.small_d_cluster:
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

    # I know these two are switched. I'm pretty sure QuPath put these results in the wrong category because in all instances 'mean' >= 'max'.
    cluster['delaunay_max_area'] = float(row['Cluster  mean: Delaunay: Mean triangle area'])
    cluster['delaunay_mean_area'] = float(row['Cluster  mean: Delaunay: Max triangle area'])


    return 1

def row_has_nan(row):
    if 'Centroid X' not in row:
        return True
    if 'Centroid Y' not in row:
        return True
    if 'Cell: Area' not in row:
        return True
    if row['Centroid X'] == 'NaN':
        return True
    elif row['Centroid Y'] == 'NaN':
        return True
    elif row['Cell: Area'] == 'NaN':
        return True
    return False

def main(subject_id = None, image_name=None, image_processor=False, config=None):

    # ::::::::   DICTIONARY STRUCTURES   ::::::::
    #
    #
    # delaunay:
    #   --[subject]
    #       --[image]
    #           --[cell classification]
    #               --[delaunay centroid]
    #                   --[cluster info]
    # e.g. delaunay['00-05']['00_05_D_3_1']['Tumor'][(682, 5699)]['size']
    #
    # d_delaunay:
    #   --[subject]
    #       --[image]
    #           --[dense region]
    #               --[cell classification]
    #                   --[delaunay centroid]
    #                       --[cluster info]
    #
    # cell_locations:
    #   --[subject]
    #       --[image]
    #           --[region counts] -> [cluster number, distances to 5 closest regions]
    #           --[cell_locations] -> [x, y, label, cluster number]


    QFLAGS = config # (In case there are any unexpected variables, because this is a sloppy fix)

    classy = config.classifier
    FEATURE_DIR = config.feature_directory
    DETECTIONS = config.detections
    OMIT_CLASS = config.omit_class

    detections_file = os.path.join(DETECTIONS, image_name + ' Detectionstxt')
    if not os.path.exists(detections_file):
        print('No detections file in qupath_lstm')
        return None, None

    delaunay = {}
    data = {}

    detections_file = image_name + ' Detectionstxt'

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
                create_delaunay_features(delaunay, subject, image_name, row, config)

    if not data or not delaunay:
        return None, None
    image_dict = {}
    img_delaunay = {}
    for cell in data['subj'][image_name]:
        image_dict[cell] = data['subj'][image_name][cell]
        img_delaunay = delaunay['subj'][image_name]
    return image_dict, img_delaunay


# Main body
if __name__ == '__main__':
    main()