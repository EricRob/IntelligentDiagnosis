


#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import numpy as np
import csv
import os
import sys
from IPython import embed
import pdb
from termcolor import cprint
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow import flags
from skimage import io
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


flags.DEFINE_integer("test", 0, "Test mode")
flags.DEFINE_integer('clusters', 25, 'Number of clusters for KMeans Clustering')
flags.DEFINE_bool('all_cells', False, 'Create histogram showing features of all cells')
flags.DEFINE_string('closest_centers', 5, 'Number of closest cluster centers to find when computing regional immune density')

FLAGS = flags.FLAGS

# Global variables
DETECTIONS = '/data/QuPath/CellCounter/detections/CUMC'
STATUS_CSV = '/data/recurrence_seq_lstm/IntelligentDiagnosis/recurrence_status.csv'
OMIT_KEY = ['Class', 'Name', 'ROI', 'Centroid X', 'Centroid Y']
OMIT_CLASS = ['Other', 'red_cell']
KEEP_CLASS = ['Tumor', 'Immune cells']
MASK_LOCATION = '/data/recurrence_seq_lstm/image_data/masks'
FEATURES = ['Cell: Area', 'Class', 'Centroid X', 'Centroid Y']
THRESHOLD = 0.15
COLORS = ['red', 'green']
FILLER = '                      '
# Class declarations

# Function declarations
def convert_nan_to_zero(arr):
	nan_locs = np.isnan(arr)
	arr[nan_locs] = 0
	return arr

def get_bins(rec, non):
	maximum = np.amax(rec)
	if np.amax(non) > maximum:
		maximum = np.amax(non)
	minimum = np.amin(rec)
	if np.amin(non) < minimum:
		minimum = np.amin(non)
	return np.linspace(minimum, maximum, num=100)


def show_histogram(RE, NR):

	for cell_class in RE:
		n = 0
		if cell_class not in NR or cell_class in OMIT_CLASS:
			continue
		cprint('Plotting ' + cell_class, 'green', end="\r")
		for value in RE[cell_class]:
			if value not in NR[cell_class] or value in OMIT_KEY:
				continue
			n += 1
			plt.subplot(4,9,n)
			rec = convert_nan_to_zero(np.array(RE[cell_class][value]))
			non = convert_nan_to_zero(np.array(NR[cell_class][value]))
			bins = get_bins(rec / len(rec), non / len(non))
			plt.hist(non, stacked=True, bins=100, alpha=0.5, label='Nonrecurrent', density=True)
			plt.hist(rec, stacked=True, bins=100, alpha=0.5, label="Recurrent", density=True)
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
			# Recurrent Cell
			cell_class = subject[image][cell]['Class']
			if cell_class not in cell_dict:
				cell_dict[cell_class] = {}
			for key in subject[image][cell].keys():
				if key in OMIT_KEY:
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
    # plt.savefig(os.path.join(FLAGS.base_path,"ROC.jpg"))
    # plt.clf()
def run_thresholds(start, stop, samples, labels, values):
	holds = []
	accuracies = []
	for thresh in np.linspace(start,stop,num=samples):
		# threshed = values > thresh
		passed = labels[values >= thresh]
		under = labels[values < thresh]

		rec_under = np.sum(under)
		rec_over = np.sum(passed)

		nr_under = len(under) - rec_under
		nr_over = len(passed) - rec_over
		# nr_acc = nr_under / (nr_under + nr_over)
		# rec_acc =  rec_over / (rec_under + rec_over)

		acc = (nr_over + rec_under) / (len(labels))
		holds.append(thresh)
		accuracies.append(acc)
		# print('Threshold: ' + "{0:02f}".format(thresh) + ' NR accuracy:' + str("{0:03f}".format(nr_acc)) + ' RE accuracy: ' + str("{0:03f}".format(rec_acc)))
		print('Threshold: ' + "{0:02f}".format(thresh) + ' Accuracy: ' + "{0:03f}".format(acc))
	return holds, accuracies
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
	# thresh, acc = run_thresholds(0, .5, 501, labels, values)
	plt.plot(thresh, acc, lw=2)
	plt.ylim(0,1)
	plt.title('Immune cell proportions')
	plt.axhline(max(acc), c='red')
	plt.xlabel('Immune cell count / Total cell count')
	plt.ylabel('Accuracy')
	plt.text(.1, .8, 'Threshold: 0.192 Accuracy: ' "{0:03f}".format(max(acc)))
	# plt.show()
	


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

def detections(image_name):
	return main(image_name, True)

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

def k_means_clustering_and_features(data):
	plt.clf()
	rec_center_ratio = []
	rec_center_mean = []
	nr_center_ratio = []
	nr_center_mean = []
	rec_density = []
	nr_density = []
	for subject in data:
		if subject == 'status':
			continue
		if 'status' not in data[subject]:
			continue
		for image in data[subject]:
			if 'status' in image:
				continue
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
				cprint('NO CELL DATA: ' + image, 'white', 'on_red')
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
				pdb.set_trace()
			coord_space = coord_space[coord_space[:,0] > 0, :]
			tumor_space = tumor_space[tumor_space[:,0] > 0, :]
			plt.scatter(tumor_x, max(coord_space[:,1]) - tumor_y, s=1, label='tumor')
			# plt.scatter(immune_x, max(coord_space[:,1]) - immune_y, s=1, label='immune')
			
			cprint('Clustering ' + image + '(This may take a while)...', 'grey', 'on_white')
			kmeans = KMeans(n_clusters=FLAGS.clusters, random_state=0).fit(coord_space)
			centers = kmeans.cluster_centers_
			labels = kmeans.labels_
			# centers[:,1] = max(coord_space[:,1]) - centers[:, 1]
			# bins = np.bincount(labels)
			center_locs = np.zeros([FLAGS.clusters, 3]).astype(int)
			center_locs[:,:2] = centers.astype(int)
			center_locs[:,2] = np.bincount(labels).astype(int)

			cprint('Calculating voronoi regions ...', 'yellow')
			vor = Voronoi(centers)
			regions, vertices = voronoi_finite_polygons_2d(vor)
			for region in regions:
			    polygon = vertices[region]
			    plt.fill(*zip(*polygon), alpha=0.1)

			cprint('Placing cells in regions...', 'yellow')
			cell_locations = place_cells_in_regions(regions, vertices, coord_space, tumor_space)
			region_cell_counts = fill_region_cell_counts(centers, cell_locations)
			
			cprint('Calculating region densities...', 'yellow')
			largest_densities = greatest_n_cluster_densities(region_cell_counts, 0.5, FLAGS.clusters, data[subject]['status'])
			_ = greatest_n_cluster_densities(region_cell_counts, 0.4, FLAGS.clusters, data[subject]['status'])
			_ = greatest_n_cluster_densities(region_cell_counts, 0.3, FLAGS.clusters, data[subject]['status'])


			# densities = np.zeros(len(region_cell_counts))
			# num = 1
			# for row in region_cell_counts:
			# 	# print("Cluster " + str(num) + ' immune density: ' + "{0:.2f}".format(row[2] / row[4]))
			# 	densities[num - 1] = row[2] / row[4] 
			# 	num += 1
			

			# largest_densities_ind = np.argpartition(densities, int(-(FLAGS.clusters*0.2)))[int(-(FLAGS.clusters*0.2)):]
			# largest_densities = densities[largest_densities_ind]
			# print('Mean density: ' + "{0:.3f}".format(np.mean(largest_densities)) + ', status: ' + str(data[subject]['status']))


			closest_centers = find_nearest_cluster_centers(centers, FLAGS.closest_centers)
			x = closest_centers[:,:,0]
			normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
			closest_centers[:,:,0] = normalized
			min_center = np.partition(normalized.flatten(), 2)[2]
			max_center = np.partition(normalized.flatten(), -2)[-2]
			mean_center = np.mean(normalized.flatten())

			
			if data[subject]['status']:
				rec_center_ratio.append(min_center / max_center)
				rec_center_mean.append(mean_center)
				rec_density.append(np.mean(largest_densities))
			else:
				nr_center_ratio.append(min_center / max_center)
				nr_center_mean.append(mean_center)
				nr_density.append(np.mean(largest_densities))

			# # Plot all points
			# plt.scatter(coord_space[:,0], max(coord_space[:,1]) - coord_space[:,1], s=4, c=labels, cmap='plasma')
			# plt.scatter(centers[:, 0], max(coord_space[:,1]) - centers[:, 1], c='black', s=200, alpha=0.8)
			# dist = (int(vor.max_bound[0] - vor.min_bound[0]), int(vor.max_bound[1] - vor.min_bound[1]))
			# plt.xlim(vor.min_bound[0] - 0.15*dist[0], vor.max_bound[0] + 0.15*dist[0])
			# plt.ylim(vor.min_bound[1] - 0.15*dist[1], vor.max_bound[1] + 0.15*dist[1])
			# plt.title(image + ', status = ' + str(data[subject]['status']))
			# plt.legend(loc='upper right')
			# pdb.set_trace()
			# plt.show()

	plt.subplot(3,1,1)
	plt.hist(rec_center_ratio, stacked=True, bins=25, alpha=0.5, label='Rec immune', density=True)
	plt.hist(nr_center_ratio, stacked=True, bins=25, alpha=0.5, label="nr immune", density=True)
	plt.title('min/max ratio')
	plt.legend(loc='upper right')

	plt.subplot(3,1,2)
	plt.hist(rec_center_mean, stacked=True, bins=25, alpha=0.5,label='Rec tumor', density=True)
	plt.hist(nr_center_mean, stacked=True, bins=25, alpha=0.5, label="nr tumor", density=True)
	plt.legend(loc='upper right')
	plt.title('center_distances')

	plt.subplot(3,1,3)
	plt.hist(rec_density, stacked=True, bins=25, alpha=0.5,label='Rec tumor', density=True)
	plt.hist(nr_density, stacked=True, bins=25, alpha=0.5, label="nr tumor", density=True)
	plt.legend(loc='upper right')
	plt.title('center_distances')
	pdb.set_trace()
	plt.show()
	return cell_locations

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
	print('Mean density: ' + "{0:.3f}".format(np.mean(largest_densities)) + ', threshold: ' + str(k) + ', status: ' + str(status))
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
	cells_in_regions = np.zeros([FLAGS.clusters, 5])
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

def main(image_name=None, image_processor=False):
	
	data = {}
	if image_name:
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
					n += 1
					cell_name = '{0:06}'.format(n)
					data[subject][image_name][cell_name] = {}
					add_cell_features(data[subject][image_name][cell_name], row)
	else:
		# Current run called directly
		for (dirpath, dirnames, filenames) in os.walk(DETECTIONS):
			count = 0
			shuffle(filenames)
			for file in filenames:
				count +=1
				if FLAGS.test:
					if count > FLAGS.test:
						break
				if 'bleached' in file:
					cprint('skipping ' + file + ' (' + str(count) + '/' + str(len(filenames)) + ')' + FILLER, 'yellow', end="\r")
					continue
				n = 0
				subject = file[0:2] + "-" + file[3:5]
				if subject not in data:
					data[subject] = {}
				image_name = file[:-18]
				if image_name not in data[subject]:
					data[subject][image_name] = {}
				if file.endswith('.txt') and 'bleached' not in file:
					# count += 1
					cprint('scanning ' + file + ' (' + str(count) + '/' + str(len(filenames)) + ')' + FILLER, 'yellow', end="\r")
					with open(os.path.join(dirpath, file), 'r') as f:
						reader = csv.DictReader(f, delimiter='\t')
						# print(reader.fieldnames)
						for row in reader:
							n += 1
							cell_name = '{0:06}'.format(n)
							data[subject][image_name][cell_name] = {}
							add_cell_features(data[subject][image_name][cell_name], row)
		print()
		print('Done!')			
						
	status = {}
	with open(STATUS_CSV, 'r') as csvfile:
		cprint('Adding recurrence status...', 'green', 'on_white')
		reader = csv.reader(csvfile)
		for row in reader:
			if row[0] in data:
				data[row[0]]['status'] = int(row[1])
				for image in data[row[0]]:
					if not 'status':
						data[row[0]][image]['status'] = int(row[1])
	if image_processor:
		image_dict = {}
		for cell in data['subj'][image_name]:
			image_dict[cell] = data['subj'][image_name][cell]
		return image_dict

	# Visualization
	if FLAGS.all_cells:
		RE = {}
		NR = {}
		for subject in data:
			if 'status' in data[subject]:
				if data[subject]['status']:
					add_to_cell_dict(RE, data[subject])
				else:
					add_to_cell_dict(NR, data[subject])
	# return RE, NR
	# RE_tumor_cell = count_types[]
	# show_histogram(RE, NR)
	# load_image_masks(data)
	cell_locations = k_means_clustering_and_features(data)

# Main body
if __name__ == '__main__':
	main()