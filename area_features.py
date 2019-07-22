#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import csv
import argparse
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import pdb
import pickle
import matplotlib.pyplot as plt
from termcolor import cprint
from scipy import stats
# from sklearn.preprocessing import normalize

# Global variables
DET_PATH = '/Users/ericrobinson/Desktop/test_det'
IMG_TO_SUBJ = '/Users/ericrobinson/Desktop/image_to_subject_ID.csv'
PICKLE_PATH = '/Users/ericrobinson/Desktop/pickles'

# Class declarations
class DetFile(object):
    """docstring for DetFile"""
    def __init__(self, detection, name):
        self.name = name
        self.imm_areas = []
        self.imm_coords = []
        self.tumor_areas = []
        self.tumor_coords = []
        self.base = detection[:-18]
        self.cluster_list = []
        self.imm_large = {}
        self.imm_small = {}
        self.tum_large = {}
        self.tum_small = {}
        self.cluster_limit = 2
        self.region_counts = np.zeros([8,2])
        self.delaunay_region_vals = np.zeros([8,4]) # tum_small count, tum_large count, imm_small count, imm_large count

    def add_row(self, row):
        if len(row) < 65:
            pass
        try:
            if row[0] == 'Immune cells':
                if not np.isnan(float(row[65])) and not np.isnan(float(row[38])) and not np.isnan(float(row[39])):
                    self.imm_coords.append((float(row[38]), float(row[39])))
                    self.imm_areas.append(float(row[65]))
                    cluster_centroid = (float(row[80]), float(row[81]))
                    cluster_size = int(row[89])
                    if cluster_size  <= self.cluster_limit:
                        self.imm_small[cluster_centroid] = cluster_size
                    else:
                        self.imm_large[cluster_centroid] = cluster_size

            elif row[0] == 'Tumor':
                if not np.isnan(float(row[65])) and not np.isnan(float(row[38])) and not np.isnan(float(row[39])):
                    self.tumor_coords.append((float(row[38]), float(row[39])))
                    self.tumor_areas.append(float(row[65]))
                    cluster_centroid = (float(row[80]), float(row[81]))
                    cluster_size = int(row[89])
                    if cluster_size  <= self.cluster_limit:
                        self.tum_small[cluster_centroid] = cluster_size
                    else:
                        self.tum_large[cluster_centroid] = cluster_size
        except Exception as e:
            pass


    def build_coord_array(self):
        self.tum_coord_array = np.zeros([len(self.tumor_coords), 2])
        i = 0
        for coord in self.tumor_coords:
            self.tum_coord_array[i][0] = coord[0]
            self.tum_coord_array[i][1] = coord[1]
            i += 1
        self.imm_coord_array = np.zeros([len(self.imm_coords), 2])
        i = 0
        for coord in self.imm_coords:
            self.imm_coord_array[i][0] = coord[0]
            self.imm_coord_array[i][1] = coord[1]
            i += 1
    def add_region_areas(self, cell_locs):
        self.region_areas = np.zeros([8,2])
        imm = cell_locs[cell_locs[:,2]==1]
        tum = cell_locs[cell_locs[:,2]==0]
        for i in np.arange(8):
            # THIS SHOULD NOT HAVE BEEN i + 1!!!!!
            self.region_areas[i,1] = np.sum(imm[imm[:,4]==(i)][:,3])
            self.region_areas[i,0] = np.sum(tum[tum[:,4]==(i)][:,3])
    
    def add_features(self):
        self.add_feature_array()
        self.features = {}
        self.features['Unclustered Immune Count / Total Count'] = np.mean([self.feature_array[0,2] / sum(self.feature_array[0,:]), self.feature_array[1,2] / sum(self.feature_array[1,:])])
        self.features['Clustered Tumor Count / Tumor Count'] = np.mean([self.feature_array[0,1] / sum(self.feature_array[0,:2]), self.feature_array[1,1] / sum(self.feature_array[1,:2])])
        self.features['Clustered Immune Count / Total Count'] = np.mean([self.feature_array[0,3] / sum(self.feature_array[0,:]), self.feature_array[1,3] / sum(self.feature_array[1,:])])
        self.features['Clustered Immune Count / Immune Count'] = np.mean([self.feature_array[0,3] / sum(self.feature_array[0,2:]), self.feature_array[1,3] / sum(self.feature_array[1,2:])])
        self.features['Clustered Immune Count / Unclustered Immune Count'] = np.mean([self.feature_array[0,3] / self.feature_array[0,2], self.feature_array[1,3] / self.feature_array[1,2]])
        self.features['Clustered Immune Count / Clustered Tumor Count'] = np.mean([self.feature_array[0,3] / self.feature_array[0,1], self.feature_array[1,3] / self.feature_array[1,1]])
        self.features['Immune Area / Tumor Area'] = np.mean([self.region_areas[self.top_dense_ind,1] / self.region_areas[self.top_dense_ind,0], self.region_areas[self.second_dense_ind,1] / self.region_areas[self.second_dense_ind,0]])
        self.features['Immune Area / Total Area'] = np.mean([self.region_areas[self.top_dense_ind,1] / (self.region_areas[self.top_dense_ind,0] + self.region_areas[self.top_dense_ind,1]), self.region_areas[self.second_dense_ind,1] / (self.region_areas[self.second_dense_ind,0] + self.region_areas[self.second_dense_ind,1])])

    def add_feature_array(self):
        self.feature_array = np.zeros([2,4])
        self.feature_array[0,:] = self.delaunay_region_vals[self.top_dense_ind]
        self.feature_array[1,:] = self.delaunay_region_vals[self.second_dense_ind]

    def densest_regions_calculate(self):
        self.top_dense_ind = -1
        self.top_dense_val = -1
        self.second_dense_ind = -1
        self.second_dense_val = -1
        row_num = 0
        for row in self.region_counts:
            density = row[1] / sum(row)
            if density > self.top_dense_val:
                self.second_dense_ind = self.top_dense_ind
                self.top_dense_ind = row_num
                self.second_dense_val = self.top_dense_val
                self.top_dense_val = density
            elif density > self.second_dense_val:
                self.second_dense_val = density
                self.second_dense_ind = row_num
            row_num += 1
# Function declarations

def voronoi_finite_polygons_2d(vor, radius=None):

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def place_cells_in_regions(regions, vertices, det_f):
    cell_locations = np.zeros([(det_f.imm_coord_array.shape[0] + det_f.tum_coord_array.shape[0]), 5]).astype(int)
    cell_locations[:det_f.imm_coord_array.shape[0],:2] = det_f.imm_coord_array[:,:].astype(int)
    cell_locations[:det_f.imm_coord_array.shape[0],2] = 1
    cell_locations[:det_f.imm_coord_array.shape[0],3] = det_f.imm_areas
    cell_locations[det_f.imm_coord_array.shape[0]:,:2] = det_f.tum_coord_array[:,:].astype(int)
    cell_locations[det_f.imm_coord_array.shape[0]:,3] = det_f.tumor_areas
    cell_locations = cell_locations.astype(int)
    region_num = 0
    skip_rows = []

    # It's not recognizing region 7? Hack: place all of the other regions and the leftovers are in 7
    cell_locations[:,4] = 7

    for region in regions:
        polygon = Polygon(vertices[region])
        row_num = 0
        for row in cell_locations:
            point = Point(row[0], row[1])
            if polygon.contains(point):
                cell_locations[row_num, 4] = region_num
                det_f.region_counts[region_num, cell_locations[row_num, 2]] += 1
                skip_rows.append(row)
            if region_num == 7 and cell_locations[row_num, 4] == 7:
                det_f.region_counts[region_num, cell_locations[row_num, 2]] += 1
            row_num += 1
        region_num += 1
    return cell_locations

def img_list():
    img_to_subj = {}
    subj_to_img = {}
    labels = {}
    cumc = []
    yale = []
    geis = []
    with open(IMG_TO_SUBJ, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        _ = next(csvreader)
        for row in csvreader:
            img = row[0][:-4]
            subj = row[1]
            label = int(row[2])
            img_to_subj[img] = subj
            labels[img] = label
            if subj not in subj_to_img:
                subj_to_img[subj] = [img]
                if row[3] == 'cumc':
                    cumc.append(subj)
                elif row[3] == 'yale':
                    yale.append(subj)
                elif row[3] == 'geis':
                    geis.append(subj)
            else:
                if img not in subj_to_img[subj]:
                    subj_to_img[subj].append(img)
    return img_to_subj, subj_to_img, labels, cumc, yale, geis

def remove_nan_values(arr):
    nan_locs = np.isnan(arr)
    no_nans = arr[np.invert(nan_locs)]
    return no_nans  

def quick_hist(arr1, arr2):
    plt.clf()
    # ar1, bin1 = np.histogram(arr1, bins='auto', density=True)
    # ar2, bin2 = np.histogram(arr2, bins='auto', density=True)
    plt.hist(arr1, bin='auto',alpha = 0.5)
    plt.hist(arr2, bin='auto', alpha = 0.5)
    plt.show()

def sort_and_trim(arr, percentile):
    arr = np.sort(arr)
    trim = int(percentile*arr.size/100.0)
    return arr[trim:-trim]

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
    return holds, accuracies, area, sensitivities, specificities

def place_clusters(regions, vertices, det_f):
    region_num = 0
    for region in regions:
        polygon = Polygon(vertices[region])
        run_clusters(det_f, det_f.tum_small, region_num, 0, polygon)
        run_clusters(det_f, det_f.tum_large, region_num, 1, polygon)
        run_clusters(det_f, det_f.imm_small, region_num, 2, polygon)
        run_clusters(det_f, det_f.imm_large, region_num, 3, polygon)

        # for cluster in det_f.tum_small:
        #     point = Point(int(cluster[0]), int(cluster[1]))
        #     if polygon.contains(point):
        #         det_f.delaunay_region_vals[region_num,0] += det_f.tum_small[cluster]
        # for cluster in det_f.tum_large:
        #     point = Point(int(cluster[0]), int(cluster[1]))
        #     if polygon.contains(point):
        #         det_f.delaunay_region_vals[region_num,1] += det_f.tum_large[cluster]
        # for cluster in det_f.imm_small:
        #     point = Point(int(cluster[0]), int(cluster[1]))
        #     if polygon.contains(point):
        #         det_f.delaunay_region_vals[region_num,2] += det_f.imm_small[cluster]
        # for cluster in det_f.imm_large:
        #     point = Point(int(cluster[0]), int(cluster[1]))
        #     if polygon.contains(point):
        #         det_f.delaunay_region_vals[region_num,3] += det_f.imm_large[cluster]
        region_num += 1

def run_clusters(det_f, cluster_dict, region_num, val_ind, polygon):
    for cluster in cluster_dict:
        try:
            # pdb.set_trace()
            point = Point(int(cluster[0]), int(cluster[1]))
            if polygon.contains(point):
                det_f.delaunay_region_vals[region_num, val_ind] += cluster_dict[cluster]
        except Exception as e:
            if e.__class__.__name__ == 'ValueError':
                continue
            else:
                pdb.set_trace()

def calculate_feature_histograms(re_list, nr_list, source='all_sources'):
    print('Calculating features for:')
    for val in re_list:
        print(val.name)
    for val in nr_list:
        print(val.name)

    if re_list:
        features = list(re_list[0].features.keys())
    elif nr_list:
        features = list(nr_list[0].features.keys())
    else:
        print('[ERROR] No images for feature calculation, exiting')
        return
        
    if not re_list:
        re_sum = 0
    else:
        re_sum = len(re_list)

    if not nr_list:
        nr_sum = 0
    else:
        nr_sum = len(nr_list)
    labels = np.zeros([re_sum + nr_sum])
    labels[:re_sum] = 1
    with open(os.path.join(PICKLE_PATH, '%s_results.csv' % source), 'w') as csvfile:
        print('Starting csv')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Source', 'Feature', 'Accuracy', 'AUC'])
        for feature in sorted(features):
            all_scores = []
            for val in re_list:
                all_scores.append(val.features[feature])
            for val in nr_list:
                all_scores.append(val.features[feature])
            arr = np.array(all_scores)
            # arr = arr[~np.isnan(arr)]
            
            arr[np.isnan(arr)] = 0
            sorted_arr = np.copy(arr)
            sorted_arr.sort()
            arr[np.isinf(arr)] = sorted_arr[max(-8, -len(sorted_arr))]
            norm_arr = (arr - np.min(arr)) / np.ptp(arr)
            thresh, acc, area, sens, spec = run_thresholds(0, 1, 10001, labels, norm_arr)
            csvwriter.writerow([source, feature, str(max(acc)), str(area)])
        csvwriter.writerow([''])
        csvwriter.writerow(['Image name', 'Feature', 'Value'])
        both_lists = re_list + nr_list
        for val in both_lists:
            for feature in sorted(features):
                csvwriter.writerow([val.name, feature, val.features[feature]])
        
    pass

def norm_arrays(re_list, nr_list):
    all_scores = []
    for val in re_list:
        all_scores.append(val.features['Immune Area / Tumor Area'])
    for val in nr_list:
        all_scores.append(val.features['Immune Area / Tumor Area'])
    arr = np.array(all_scores)
    arr[np.isnan(arr)] = 0
    sorted_arr = np.copy(arr)
    sorted_arr.sort()
    arr[np.isinf(arr)] = sorted_arr[-8]
    norm_arr = (arr - np.min(arr)) / np.ptp(arr)
    norm_re = norm_arr[:len(re_list)]
    norm_nr = norm_arr[len(re_list):]

    #pdb.set_trace()
    return

def print_features(re_list, nr_list, img_to_subj):
    with open(os.path.join(PICKLE_PATH, 'area_features.csv'), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Subject', 'Image', 'Label', 'Feature', 'Value'])
        for val in re_list:
            csvwriter.writerow([img_to_subj[val.name], val.name, '1', 'Immune Area / Tumor Area', str(val.features['Immune Area / Tumor Area'])])
            csvwriter.writerow([img_to_subj[val.name], val.name, '1', 'Immune Area / Total Area', str(val.features['Immune Area / Total Area'])])
        for val in nr_list:
            csvwriter.writerow([img_to_subj[val.name], val.name, '0', 'Immune Area / Tumor Area', str(val.features['Immune Area / Tumor Area'])])
            csvwriter.writerow([img_to_subj[val.name], val.name, '0', 'Immune Area / Total Area', str(val.features['Immune Area / Total Area'])])

def main(ars):
    # pdb.set_trace()
    img_to_subj, subj_to_img, labels, cumc, yale, geis = img_list()

    re_list = []
    nr_list = []

    for root, dirs, files in os.walk(DET_PATH, topdown=True):
        print()
        print('Entering file walk')
        if ars.reverse:
            files.sort(reverse=True)
        else:
            files.sort()
        if ars.quartile:
            del files[0:int((len(files) // 4)*2.5)]
        for detection in files:
            print('Reading Detection: %s' % detection)

            if '.txt' not in detection:
                continue

            img_name = detection[:-18]
            print('img_name: %s' % img_name)
            print('source: %s' % ars.source)
            
            try:
                if ars.source == 'geis' and img_to_subj[img_name] not in geis:
                    print('skipping %s' % img_name)
                    continue
                elif ars.source == 'yale' and img_to_subj[img_name] not in yale:
                    print('skipping %s' % img_name)
                    continue
                elif ars.source == 'cumc' and img_to_subj[img_name] not in cumc:
                    print('skipping %s' % img_name)
                    continue
                #else:
                 #   if img_to_subj[img_name] in yale or img_to_subj[img_name] in geis:
                  #      print('final else: skipping %s' % img_name)
                   #     continue
                    #else:
                     #   print('using sinai')
            except Exception as e:
                cprint('[ERROR] Entered exception', 'red')
                pass


            pickle_name = os.path.join(PICKLE_PATH, img_name + '.file')
            if 'bleached'in detection:
                print('skipping %s' % detection)
                continue

            if detection[:2] == '01':
                cprint('skipping stage 1 : %s' % detection, 'yellow')
                continue
            
            ff_array = os.path.join(PICKLE_PATH, img_name + '.npy')
            if os.path.exists(pickle_name):
                print('Loading %s' % pickle_name)
                with open(pickle_name, 'rb') as handle:
                    add = pickle.load(handle)
                    if not add or img_name not in labels:
                        continue
                    if labels[img_name] == 1:
                        re_list.append(add)
                    elif labels[img_name] == 0:
                        nr_list.append(add)
            else:
                cprint(img_name, 'grey', 'on_white')
                with open(pickle_name, 'wb') as lock_file:
                    pickle_hold = {}
                    pickle.dump(pickle_hold, lock_file)

                det_f = DetFile(detection, img_name)
                fd = os.path.join(root, detection)
                
                # pdb.set_trace()
                print('reading detections file')
                with open(fd, 'r') as f:
                    header = f.readline()
                    # header = header.split('\t')
                    # line = f.readline()
                    # i = 0
                    # for val in header:
                    #   print('%s: %d' % (val, i))
                    #   print(str(line.split('\t')[i]))
                    #   i += 1
                    # pdb.set_trace()
                    for line in f:
                        if line == '\n':
                            continue
                        det_f.add_row(line.split('\t'))
                det_f.build_coord_array()
                print('clustering...')
                if det_f.imm_coord_array.shape[0] == 0 or len(det_f.imm_coord_array) < 8:
                    cprint('No immune cells in %s' % img_name, 'yellow')
                    continue
                kmeans = KMeans(n_clusters=8, random_state=0).fit(det_f.imm_coord_array)
                cluster_centers = kmeans.cluster_centers_
                cluster_labels = kmeans.labels_
                try:
                    vor = Voronoi(cluster_centers)
                except Exception as e:
                    cprint('Voronoi error - Requires new qupath detections', 'yellow')
                    continue
                regions, vertices = voronoi_finite_polygons_2d(vor)
                print('placing in regions...')
                cell_locs = place_cells_in_regions(regions, vertices, det_f)
                # Determine densest clusters
                det_f.densest_regions_calculate()

                # Place delaunay centroids in densest clusters
                place_clusters(regions, vertices, det_f)
                det_f.add_region_areas(cell_locs)
                det_f.add_features()
                with open(pickle_name, 'wb') as handle:
                    cprint('Saving %s' % img_name, 'green')
                    pickle.dump(det_f, handle)
                if labels[img_name] == 1:
                    re_list.append(det_f)
                elif labels[img_name] == 0:
                    nr_list.append(det_f)
                # Count delaunay clusters
                
                # np.save(ff_array, det_f.feat_arr)
            if img_name not in img_to_subj:
                continue
            subj = img_to_subj[img_name]
            try:
                label = labels[img_name]
            except Exception as e:
                pdb.set_trace()
            
            # if subj in cumc:
            #     top_k = 2
            #     ind_one = np.argpartition(final_feature[:,0], -top_k)[-top_k:]
            #     ind_two = np.argpartition(final_feature[:,1], -top_k)[-top_k:]
            #     mean_one = sum(final_feature[:,0][ind_one]) / top_k
            #     mean_two = sum(final_feature[:,1][ind_two]) / top_k
            #     if mean_one == float('inf') or mean_two == float('inf'):
            #         pass
            #     elif label == 1:
            #         re_scores_one.append(mean_one)
            #         re_scores_two.append(mean_two)
            #     elif label == 0:
            #         nr_scores_one.append(mean_one)
            #         nr_scores_two.append(mean_two)


        # re_labels = np.ones(len(re_scores_one))
        # nr_labels = np.zeros(len(nr_scores_one))

        # re_portion_np = np.array(re_scores_one)
        # nr_portion_np = np.array(nr_scores_one)
        # np_labels = np.concatenate((re_labels, nr_labels))
        # np_values = np.concatenate((re_portion_np, nr_portion_np))
    calculate_feature_histograms(re_list, nr_list, source=ars.source)
    print_features(re_list, nr_list, img_to_subj)
    # norm_arrays(re_list, nr_list)


# Main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process QuPath detections for final feature information')
    parser.add_argument('--reverse', default=False, action='store_true', help='Walk through detections directory in reverse')
    parser.add_argument('--quartile', default=False, action='store_true', help='Start 25 percent of the way through the list')
    parser.add_argument('--source', default=None, type=str, help='Institution for creating feature diagrams')
    ars = parser.parse_args()
    main(ars)
