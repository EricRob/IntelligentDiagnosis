#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import os
import csv
import pdb
import numpy as np
from skimage import io
from scipy.ndimage import measurements
from skimage import transform
from math import floor
from numpy.lib.stride_tricks import as_strided
from termcolor import cprint
import numbers
# from tensorflow import flags
import qupath_reader as qupath
import time
import struct


# flags.DEFINE_bool("overwrite", False, "Overwrite image binary files if they exist.")

# FLAGS = flags.FLAGS

# Global variables
FEATURE_LIST = ['total_area', 'cell_perimeter', 'nuc_area', 'cell_area']
DETECTION_SAMPLING = True
OTHER_TILE_THRESHOLD = 0.8
OTHER_PATCH_THRESHOLD = 0.5
MINIMUM_PATCH_CELLS = 20

# Class declarations
class OriginalPatchConfig(object):
	image_data_folder_path = "/data/recurrence_seq_lstm/image_data/" # Location of image to be split into patches
	features_path = '/data/recurrence_seq_lstm/feature_testing'
	patch_size = 500 # Pixel length and width of each patch square
	tile_size = patch_size * 5
	edge_overlap = 75 # Amount of overlap between patches within a sample
	sample_size = 100 # Final size of patch (usually 100)
	scaling_factor = patch_size / sample_size
	patch_keep_percentage = 75 # Percentage of patch that must be data (i.e. non-background)
	tile_keep_percentage = 35 # Percentage of tile that must contain cell data (i.e. non-background)
	maximum_std_dev = 3 * patch_size # Std dev size for a tile with 100% density
	maximum_seq_per_tile = 3 # Round number of sequences to the nearest integer
	maximum_sample_count = 10000
	image_height = sample_size
	image_width = sample_size
	image_depth = 3
	num_steps = 20
	large_cluster = 1
	pixel_radius = 40

class YaleConfig(object):
	image_data_folder_path = "/data/recurrence_seq_lstm/image_data/" # Location of image to be split into patches
	features_path = '/data/recurrence_seq_lstm/feature_testing'
	patch_size = 250 # Pixel length and width of each patch square
	tile_size = patch_size * 5
	edge_overlap = 75 # Amount of overlap between patches within a sample
	sample_size = 100 # Final size of patch (usually 100)
	scaling_factor = patch_size / sample_size
	patch_keep_percentage = 75 # Percentage of patch that must be data (i.e. non-background)
	tile_keep_percentage = 35 # Percentage of tile that must contain cell data (i.e. non-background)
	maximum_std_dev = 3 * patch_size # Std dev size for a tile with 100% density
	maximum_seq_per_tile = 6 # Round number of sequences to the nearest integer
	maximum_sample_count = 10000
	image_height = sample_size
	image_width = sample_size
	image_depth = 3
	num_steps = 20
	large_cluster = 1
	pixel_radius = 40

# Function declarations
def extract_tiles(arr, tile_size, edge_overlap):
	arr_ndim = arr.ndim

	tile_shape = (tile_size,tile_size)
	extraction_step = tile_size

	# Assure all objects are tuples of the same shape
	if isinstance(tile_shape, numbers.Number):
	    tile_shape = tuple([tile_shape] * arr_ndim)
	if isinstance(extraction_step, numbers.Number):
	    extraction_step = tuple([extraction_step] * arr_ndim)

	# consistent stride values necessary for arrays of indeterminate size and shape
	tile_strides = arr.strides

	slices = [slice(None, None, st) for st in extraction_step]
	indexing_strides = arr[slices].strides

	tile_indices_shape = ((np.array(arr.shape) - np.array(tile_shape)) //
	                       np.array(extraction_step)) + 1

	shape = tuple(list(tile_indices_shape) + list(tile_shape))
	strides = tuple(list(indexing_strides) + list(tile_strides))

	# Create views into the array with given shape and stride
	tiles = as_strided(arr, shape=shape, strides=strides)


	bottom_edge = bottom_edge_tiles(arr, tile_size, edge_overlap)
	right_edge = right_edge_tiles(arr, tile_size, edge_overlap)

	tiles = adjust_tile_grid_edges(tiles, arr.shape, tile_size, edge_overlap)

	return tiles, bottom_edge, right_edge

def adjust_tile_grid_edges(tiles, shape, tile_size, edge_overlap):
	if shape[0] / tile_size % 1 < (edge_overlap / 100):
		tiles = tiles[:(max(tiles.shape[0]-1, 0)), :, :, :]
	if shape[1] / tile_size % 1 < (edge_overlap / 100):
		tiles = tiles[:,:(max(tiles.shape[1]-1, 0)), :, :]
	return tiles

def bottom_edge_tiles(image, tile_size, edge_overlap):
	bottom_overlap = image.shape[0] / tile_size % 1 < (edge_overlap / 100)

	if bottom_overlap:
		bottom_edge_y_value = (image.shape[0] // tile_size - 1) * tile_size

		bottom_mask_edge = image[bottom_edge_y_value:,:]
		bottom_edge_columns = max(bottom_mask_edge.shape[1] // tile_size - 1, 0)
		bottom_edge = np.zeros((1, bottom_edge_columns, bottom_mask_edge.shape[0], tile_size))
		
		for column in np.arange(bottom_edge.shape[1]):
			column_pixel = column*tile_size
			bottom_edge[0,column,:,:] = image[bottom_edge_y_value:,column_pixel:(column_pixel + tile_size)]
	else:
		bottom_edge_y_value = (image.shape[0] // tile_size) * tile_size
		bottom_mask_edge = image[bottom_edge_y_value:,:]
		bottom_edge_columns = bottom_mask_edge.shape[1] // tile_size
		bottom_edge = np.zeros((1, bottom_edge_columns, image.shape[0] - bottom_edge_y_value, tile_size))
		for column in np.arange(bottom_edge.shape[1]):
			column_pixel = column * tile_size
			bottom_edge[0,column,:,:] = image[bottom_edge_y_value:, column_pixel:(column_pixel + tile_size)]

	return bottom_edge

def right_edge_tiles(image, tile_size, edge_overlap):
	right_overlap = image.shape[1] / tile_size % 1 < (edge_overlap / 100)

	if right_overlap:
		right_edge_x_value = (image.shape[1] // tile_size - 1) * tile_size
		right_mask_edge = image[:,right_edge_x_value:]
		right_edge_rows = max(right_mask_edge.shape[0] // tile_size - 1, 0)
		right_edge = np.zeros((right_edge_rows, 1, tile_size, right_mask_edge.shape[1]))
		
		for row in np.arange(right_edge.shape[0]):
			row_pixel  = row * tile_size
			right_edge[row,0,:,:] = image[row_pixel:(row_pixel + tile_size),right_edge_x_value:]
	else:
		right_edge_x_value = (image.shape[1] // tile_size) * tile_size
		right_mask_edge = image[ :, right_edge_x_value:]
		right_edge_rows = right_mask_edge.shape[0] // tile_size
		right_edge = np.zeros((right_edge_rows, 1, tile_size, image.shape[1] - right_edge_x_value))

		for row in np.arange(right_edge.shape[0]):
			row_pixel = row * tile_size
			right_edge[row,0,:,:] = image[row_pixel:(row_pixel + tile_size), right_edge_x_value:]

	return right_edge

def threshold_tiles(tile_grid, keep_percentage):
	keep_list = []
	keep_threshold = tile_grid.shape[2] * tile_grid.shape[3] * (1 - keep_percentage / 100)
	
	for row in np.arange(tile_grid.shape[0]):
		for col in np.arange(tile_grid.shape[1]):
			tile = tile_grid[row, col, :, :]
			tile_sum = np.sum(tile)
			if tile_sum <= keep_threshold:
				keep_list = keep_list + [(row, col)]
	
	return keep_list

def adjust_bottom_list(bottom_list, add):
	adjusted_list = []
	for tile in bottom_list:
		adjusted_list = adjusted_list + [(tile[0] + add, tile[1])]
	return adjusted_list

def adjust_right_list(right_list, add):
	adjusted_list = []
	for tile in right_list:
		adjusted_list = adjusted_list + [(tile[0], tile[1] + add)]
	return adjusted_list

def tile_density(tile):
	tile_pixels = tile.shape[0]*tile.shape[1]
	tile_sum = np.sum(tile)
	return 1 - tile_sum / tile_pixels

def calculate_main_tile_masses(grid, keep_list, config):
	centroids = dict()
	for row in np.arange(grid.shape[0]):
		for col in np.arange(grid.shape[1]):
			if (row,col) in keep_list:
				centroids[(row,col)] = dict()
				centroids[(row,col)]["centroid"] = (config.tile_size // 2, config.tile_size // 2)
				centroids[(row,col)]["density"] = tile_density(grid[row,col,:,:])

				# centroids[(row,col)]["centroid"] = measurements.center_of_mass(grid[row,col,:,:])
	return centroids

def calculate_right_tile_masses(grid, keep_list, config):
	centroids = dict()
	if len(keep_list) == 0:
		return
	col = keep_list[0][1]
	for row in np.arange(grid.shape[0]):
		if (row,col) in keep_list:
			centroids[(row,col)] = dict()
			centroids[(row,col)]["centroid"] = (config.tile_size // 2, config.tile_size // 2)
			centroids[(row,col)]["density"] = tile_density(grid[row,0,:,:])

			# centroids[(row,col)]["centroid"] = measurements.center_of_mass(grid[row,0,:,:])
	return centroids

def calculate_bottom_tile_masses(grid, keep_list, config):
	centroids = dict()
	if len(keep_list) == 0:
		return
	row = keep_list[0][0]
	for col in np.arange(grid.shape[1]):
		if (row,col) in keep_list:
			centroids[(row,col)] = dict()
			centroids[(row,col)]["centroid"] = (config.tile_size // 2, config.tile_size // 2)
			centroids[(row,col)]["density"] = tile_density(grid[0,col,:,:])

			# centroids[(row,col)]["centroid"] = measurements.center_of_mass(grid[0,col,:,:])
	return centroids

def sample_from_distribution(mask, tile_info, config):
	keep_threshold = config.patch_size**2 * (1 - config.patch_keep_percentage/100)
	remove_tiles = []
	skip_count = 0

	if not tile_info:
		return 0
	for tile in tile_info:
		std_dev = config.maximum_std_dev
		
		sequence_count = int(round(tile_info[tile]["density"] * config.maximum_seq_per_tile))
		samples = sequence_count * config.num_steps
		tile_info[tile]["coords"] = []
		counter = 0
		while (len(tile_info[tile]["coords"]) < samples) and (counter < 10000):
			counter += 1
			x = int(round(np.random.normal(tile_info[tile]["centroid"][1], std_dev)))
			x = x - config.patch_size // 2 # centroid should be in the center of the patch, x should be left edge. Shift over from center to left edge.
			y = int(round(np.random.normal(tile_info[tile]["centroid"][0], std_dev)))
			y = y - config.patch_size // 2 # centroid should be in center of the patch, y should be top edge. Shift up from center to top.
			if x < 0 or y < 0 or x+config.patch_size > mask.shape[1] or y+config.patch_size > mask.shape[0]:
				continue
			patch = mask[y:(y+config.patch_size), x:(x+config.patch_size)]
			if np.sum(patch) <= keep_threshold:	
				tile_info[tile]["coords"] = tile_info[tile]["coords"] + [(y,x)]
		if counter >= config.maximum_sample_count:
			remove_tiles = remove_tiles + [tile]
			# cprint("Skipping " + str(tile), 'red')
			skip_count += 1

	for tile in remove_tiles:
		del tile_info[tile]
	return skip_count

def corner_detection_sample_from_dist(mask, corner, config, keep_corner, all_cells):
	if not keep_corner:
		return 1, None
	keep_threshold = config.patch_size**2 * (1 - config.patch_keep_percentage/100)
	std_dev = config.maximum_std_dev
	sequence_count = int(round(corner["density"] * config.maximum_seq_per_tile))
	samples = sequence_count * config.num_steps
	
	counter = 0
	skip_count = 0
	corner = place_cells_in_tiles(corner, all_cells, config, corner=True)
	corner = remove_garbage_tiles(corner, config, corner=True)
	
	if not corner:
		cprint("Skipping corner", 'red')
		return 1, None
	corner['corner']['coords'] = []
	while(len(corner['corner']['coords']) < samples) and (counter < 10000):
		counter += 1
		x = int(round(np.random.normal(corner['corner']["centroid"][1], std_dev)))
		x = x - config.patch_size // 2 # centroid should be in the center of the patch, x should be left edge. Shift over from center to left edge.
		y = int(round(np.random.normal(corner['corner']["centroid"][0], std_dev)))
		y = y - config.patch_size // 2 # centroid should be in center of the patch, y should be top edge. Shift up from center to top.
		coord_tuple = (y, y+config.patch_size, x, x+config.patch_size)
		patch = mask[y:(y+config.patch_size), x:(x+config.patch_size)]
		if x < 0 or y < 0 or x+config.patch_size > mask.shape[1] or y+config.patch_size > mask.shape[0]:
			continue
		if np.sum(patch) > keep_threshold:
			continue
		cells_in_patch = find_cells_in_patch(corner['corner']['cells'], coord_tuple)
		if cells_in_patch[2] >= OTHER_PATCH_THRESHOLD or cells_in_patch[3] < MINIMUM_PATCH_CELLS:
			continue
		corner['corner']['coords'] = corner['corner']['coords'] + [(y,x)]
	if counter >= 10000:
		cprint("Skipping corner", 'red')
		corner['corner']['centroid'] = []
		skip_count = 1
	else:
		cprint("Keep corner!", 'green', 'on_white')

	return skip_count, corner

def detection_sample_from_dist(mask, tile_info, config, all_cells):
	if not tile_info:
		return 0
	keep_threshold = config.patch_size**2 * (1 - config.patch_keep_percentage/100)
	remove_tiles = []
	skip_count = 0
	place_cells_in_tiles(tile_info, all_cells, config)
	remove_garbage_tiles(tile_info, config)

	for tile in tile_info:
		std_dev = config.maximum_std_dev
		
		sequence_count = int(round(tile_info[tile]["density"] * config.maximum_seq_per_tile))
		samples = sequence_count * config.num_steps
		tile_info[tile]["coords"] = []
		counter = 0
		while (len(tile_info[tile]["coords"]) < samples) and (counter < 20000):
			counter += 1
			x = int(round(np.random.normal(tile_info[tile]["centroid"][1], std_dev)))
			x = x - config.patch_size // 2 # centroid should be in the center of the patch, x should be left edge. Shift over from center to left edge.
			y = int(round(np.random.normal(tile_info[tile]["centroid"][0], std_dev)))
			y = y - config.patch_size // 2 # centroid should be in center of the patch, y should be top edge. Shift up from center to top.
			coord_tuple = (y, y+config.patch_size, x, x+config.patch_size)
			if x < 0 or y < 0 or x+config.patch_size > mask.shape[1] or y+config.patch_size > mask.shape[0]:
				continue
			patch = mask[y:(y+config.patch_size), x:(x+config.patch_size)]
			if np.sum(patch) > keep_threshold:
				continue
			cells_in_patch = find_cells_in_patch(tile_info[tile]['cells'], coord_tuple)
			if cells_in_patch[2] >= OTHER_PATCH_THRESHOLD or cells_in_patch[3] < MINIMUM_PATCH_CELLS:
				continue
			tile_info[tile]["coords"] = tile_info[tile]["coords"] + [(y,x)]
		if counter >= config.maximum_sample_count:
			remove_tiles = remove_tiles + [tile]
			# cprint("Skipping " + str(tile), 'red')
			skip_count += 1

	for tile in remove_tiles:
		del tile_info[tile]
	return skip_count

def find_cells_in_patch(cells, coord_tuple):
	y_bot, y_top, x_left, x_right = coord_tuple
	other_count = tumor_count = imm_count = total = 0
	for cell_class in cells:
		for cell in cells[cell_class]:
			y, x = cells[cell_class][cell]['loc']
			if (y_bot <= y <= y_top) and (x_left <= x <= x_right):
				total += 1
				cell_class = cells[cell_class][cell]['class']
				if cell_class == 'Tumor':
					tumor_count += 1
				elif cell_class == 'Immune cells':
					imm_count += 1
				elif cell_class == 'Other':
					other_count += 1
	if not total:
		return (0,0,0,0)
	else:
		return (tumor_count / total, imm_count / total, other_count / total, total)

def place_cells_in_tiles(tiles, cells, config, corner=False):
	half_tile = config.tile_size // 2

	if corner:
		tile = tiles
		tiles = {}
		tiles['corner'] = tile

	for tile in tiles:
		other_count = 0
		tumor_count = 0
		imm_count = 0
		x_left = tiles[tile]['centroid'][1] - half_tile
		x_right = x_left + config.tile_size
		y_bottom = tiles[tile]['centroid'][0] - half_tile
		y_top = y_bottom + config.tile_size
		tiles[tile]['cells'] = {}
		for cell in cells:
			if (x_left <= cells[cell]['x'] <= x_right) and (y_bottom <= cells[cell]['y'] <= y_top):
				cell_class = cells[cell]['class']
				if cell_class not in tiles[tile]['cells']:
					tiles[tile]['cells'][cell_class] = {}
				tiles[tile]['cells'][cell_class][cell] = cells[cell]
				tiles[tile]['cells'][cell_class][cell]['loc'] = (cells[cell]['y'], cells[cell]['x'])
				if cell_class == 'Tumor':
					tumor_count += 1
				elif cell_class == 'Immune cells':
					imm_count += 1
				elif cell_class == 'Other':
					other_count += 1
		tiles[tile]['counts'] = (tumor_count, imm_count, other_count)
	if corner:
		return tiles['corner']
	return 

def remove_garbage_tiles(tiles, config, corner=False):
	remove_list = []
	if corner:
		tile = tiles
		tiles = {}
		tiles['corner'] = tile
	for tile in tiles:
		tumor_count, imm_count, other_count = tiles[tile]['counts']
		total = tumor_count + imm_count + other_count
		if total == 0:
			remove_list.append(tile)
		else:
			other_portion = other_count / total
			if other_portion > OTHER_TILE_THRESHOLD:
				remove_list.append(tile)
	for tile in remove_list:
		# cprint('REMOVING GARBAGE TILE ' + str(tile) + ', verify original image before testing', 'red')
		del tiles[tile]
	if corner:
		return tiles
	return


def corner_sample_from_distribution(mask, corner, config, keep_corner):
	if not keep_corner:
		return 1
	keep_threshold = config.patch_size**2 * (1 - config.patch_keep_percentage/100)
	std_dev = config.maximum_std_dev
	sequence_count = int(round(corner["density"] * config.maximum_seq_per_tile))
	samples = sequence_count * config.num_steps
	corner['coords'] = []
	counter = 0
	skip_count = 0
	while(len(corner['coords']) < samples) and (counter < 10000):
		counter += 1
		x = int(round(np.random.normal(corner["centroid"][1], std_dev)))
		x = x - config.patch_size // 2 # centroid should be in the center of the patch, x should be left edge. Shift over from center to left edge.
		y = int(round(np.random.normal(corner["centroid"][0], std_dev)))
		y = y - config.patch_size // 2 # centroid should be in center of the patch, y should be top edge. Shift up from center to top.
		patch = mask[y:(y+config.patch_size), x:(x+config.patch_size)]
		if x < 0 or y < 0 or x+config.patch_size > mask.shape[1] or y+config.patch_size > mask.shape[0]:
			continue
		if np.sum(patch) <= keep_threshold:
			corner['coords'] = corner['coords'] + [(y,x)]
	if counter >= 10000:
		# cprint("Skipping corner", 'red')
		corner['centroid'] = []
		skip_count = 1
	else:
		cprint("Keep corner!", 'green', 'on_white')

	return skip_count

def split_and_combine_patch_lists(tile_dict, bottom_dict, right_dict, corner_dict, keep_corner, corner_tile_num, num_steps):
	sequences = []
	sequences = append_patch_lists(sequences, tile_dict, num_steps)
	sequences = append_patch_lists(sequences, bottom_dict, num_steps)
	sequences = append_patch_lists(sequences, right_dict, num_steps)

	if DETECTION_SAMPLING:
		if keep_corner and corner_dict['corner']['coords']:
			sequences = append_corner_patch_lists(sequences, corner_dict['corner']['coords'], num_steps, corner_tile_num)
	else:
		if keep_corner and corner_dict['coords']:
			sequences = append_corner_patch_lists(sequences, corner_dict['coords'], num_steps, corner_tile_num)
	
	return sequences

def append_corner_patch_lists(patch_list, coords_list, num_steps, corner_tile_num):
	tile = corner_tile_num
	for n in np.arange(len(coords_list) // num_steps):
		add_list = [tile]
		add_list.append(coords_list[(n*num_steps):((n+1)*num_steps)])
		patch_list = patch_list + [add_list]
	return patch_list

def append_patch_lists(patch_list, region_dict, num_steps):
	if not region_dict:
		return patch_list

	for tile in region_dict:
		coords_list = region_dict[tile]['coords']
		for n in np.arange(len(coords_list) // num_steps):
			add_list = [tile]
			add_list.append(coords_list[(n*num_steps):((n+1)*num_steps)])
			patch_list = patch_list + [add_list]
	return patch_list

def adjust_centroids(centroids, tile_size):
	if not centroids:
		return
	for center in centroids:
		centroids[center]["centroid"] = (centroids[center]["centroid"][0] + center[0]*tile_size, centroids[center]["centroid"][1] + center[1]*tile_size)

def corner_threshold(corner, keep_percentage):
	keep_threshold = corner.shape[0]*corner.shape[1]*(1 - keep_percentage/100)
	return np.sum(corner) <= keep_threshold

def calculate_corner_mass(corner_tile, keep_corner, corner_x, corner_y):
	corner = dict()
	if not keep_corner:
		return corner
	corner["density"] = tile_density(corner_tile)
	centroid = measurements.center_of_mass(corner_tile)
	corner["centroid"] = (centroid[0]+corner_x, centroid[1]+corner_y)
	return corner

# Called by image_processor.py directly
def generate_sequences(mask_filename, config, image_name=None, subject_id=None):
	tile_size = config.tile_size
	mask = io.imread(mask_filename)
	mask = mask[:,:,0]
	mask[mask > 0] = 1
	#pdb.set_trace()
	all_cells, all_delaunay = qupath.detections(subject_id, image_name)
	# for cell_class in all_delaunay:
	# 	for centroid in all_delaunay[cell_class]:
	# 		if not isinstance(centroid, tuple):
	# 			continue
	# 		for feature in all_delaunay[cell_class][centroid]:
	# 			if not float(all_delaunay[cell_class][centroid][feature]):
	# 				pdb.set_trace()
	# cprint("Extracting tiles...", 'green', 'on_white')
	tile_grid, bottom_edge, right_edge = extract_tiles(mask, tile_size, config.edge_overlap)

	# cprint("Thresholding tiles...", 'green', 'on_white')
	keep_tile_grid_list = threshold_tiles(tile_grid, config.tile_keep_percentage)

	keep_bottom_edge_list = threshold_tiles(bottom_edge, config.tile_keep_percentage)

	bottom_overlap = mask.shape[0] / tile_size % 1 < (config.edge_overlap / 100)
	right_overlap = mask.shape[1] / tile_size % 1 < (config.edge_overlap / 100)

	# Bottom edge
	if bottom_overlap:
		keep_bottom_edge_list = adjust_bottom_list(keep_bottom_edge_list, tile_grid.shape[0] + 1)
	else:
		keep_bottom_edge_list = adjust_bottom_list(keep_bottom_edge_list, tile_grid.shape[0])

	# Right edge
	keep_right_edge_list = threshold_tiles(right_edge, config.tile_keep_percentage)
	if right_overlap:
		keep_right_edge_list = adjust_right_list(keep_right_edge_list, tile_grid.shape[1] + 1)
	else:
		keep_right_edge_list = adjust_right_list(keep_right_edge_list, tile_grid.shape[1])

	# Bottom-right corner
	corner_tile = mask[tile_grid.shape[0]*tile_size:, tile_grid.shape[1]*tile_size:]
	corner_y = tile_grid.shape[0]*tile_size
	corner_x = tile_grid.shape[1]*tile_size
	keep_corner = corner_threshold(corner_tile, config.tile_keep_percentage)
	# if keep_corner:
	# 	cprint("corner passes threshold", 'white', 'on_red')

	# cprint("Calculating centroids...", 'green', 'on_white')
	tile_centroids = calculate_main_tile_masses(tile_grid, keep_tile_grid_list, config)
	bottom_centroids = calculate_bottom_tile_masses(bottom_edge, keep_bottom_edge_list, config)
	right_centroids = calculate_right_tile_masses(right_edge, keep_right_edge_list, config)
	corner_centroid = calculate_corner_mass(corner_tile, keep_corner, corner_x, corner_y)

	adjust_centroids(tile_centroids, tile_size)
	adjust_centroids(bottom_centroids, tile_size)
	adjust_centroids(right_centroids, tile_size)

	# cprint("Sampling around centroids...", 'green', 'on_white')
	tile_count = get_tile_count(tile_grid, bottom_edge, right_edge)
	centroid_count = get_centroid_count(tile_centroids, bottom_centroids, right_centroids)

	skip_count = 0
	if DETECTION_SAMPLING:
		skip_count += detection_sample_from_dist(mask, tile_centroids, config, all_cells)
		skip_count += detection_sample_from_dist(mask, bottom_centroids, config, all_cells)
		skip_count += detection_sample_from_dist(mask, right_centroids, config, all_cells)
		skip_corner, corner_centroid = corner_detection_sample_from_dist(mask, corner_centroid, config, keep_corner, all_cells)
		skip_count += skip_corner
		if not corner_centroid:
			keep_corner = False
	else:
		skip_count += sample_from_distribution(mask, tile_centroids, config)
		skip_count += sample_from_distribution(mask, bottom_centroids, config)
		skip_count += sample_from_distribution(mask, right_centroids, config)
		skip_count += corner_sample_from_distribution(mask, corner_centroid, config, keep_corner)

	cprint("Keeping " + str(centroid_count - skip_count) + "/" + str(tile_count) + " tiles")

	# cprint("Listing sequences...", 'green', 'on_white')
	
	# sequences is a list of [tile tuple, [lists of tuples]], each tuple in the list of tuples
	# containing coordinates for a top-left patch corner
	corner_tile_num = (tile_grid.shape[0], tile_grid.shape[1])
	sequences = split_and_combine_patch_lists(tile_centroids, bottom_centroids, right_centroids, corner_centroid, keep_corner, corner_tile_num, config.num_steps)

	# all_cells, all_delaunay = qupath.detections(subject_id, image_name)
	if not all_cells or not all_delaunay:
		return None

	all_tiles = list(tile_centroids)
	if bottom_centroids:
		all_tiles = all_tiles + list(bottom_centroids)
	if right_centroids:
		all_tiles = all_tiles + list(right_centroids)
	if keep_corner:
		all_tiles.append((tile_grid.shape[0], tile_grid.shape[1]))


	##                                                                             ##
	##  Place each delaunay cluster and cell into a tile for regional information  ##
	##                                                                             ##
	delaunay_in_tiles = sort_delaunay_into_tiles(all_cells, all_delaunay, all_tiles, config)
	remove_empty_delaunay_tiles(delaunay_in_tiles, all_tiles)


	##                                                                             ##
	##     Add together the delaunay clusters in all passed neighboring tiles      ##
	##                                                                             ##
	delaunay_in_neighbors = add_neighboring_tile_info(delaunay_in_tiles, all_tiles, config)

	# Adds features by sequence
	seq_and_features = process_detections(sequences, delaunay_in_neighbors, all_tiles, mask, config)
	return seq_and_features

def remove_empty_delaunay_tiles(dels, all_tiles):
	remove_list = []
	for tile in dels:
		if not dels[tile]['Tumor']:
			remove_list.append(tile)
	for tile in remove_list:
		dels.pop(tile)
		all_tiles.remove(tile)
	return

def sort_delaunay_into_tiles(cells, dels, tiles, config):
	#
	# ATTN:
	# Tiles are (y, x), delaunay is (x, y), whoops!!
	#

	tile_infos = {}
	for tile in tiles:
		tile_infos[tile] = {}
		x_range = (tile[1]*config.tile_size, tile[1]*config.tile_size + config.tile_size)
		y_range = (tile[0]*config.tile_size, tile[0]*config.tile_size + config.tile_size)
		for cell_class in list(dels):
			tile_infos[tile][cell_class] = {}
			for cluster in list(dels[cell_class]):
				if not isinstance(cluster, tuple):
					continue
				x = cluster[0]
				y = cluster[1]

				if (x_range[0] <= x < x_range[1]) and (y_range[0] <= y < y_range[1]):
					tile_infos[tile][cell_class][cluster] = dels[cell_class].pop(cluster)
	return tile_infos

def add_neighboring_tile_info(dels, tiles, config):
	del_and_neighbor = {}
	for tile in dels:
		del_and_neighbor[tile] = {}
		neighbors = neighboring_tile_numbers(tile, tiles)
		for cell_class in dels[tile]:
			if cell_class not in del_and_neighbor:
				del_and_neighbor[tile][cell_class] = {}

			for neighbor in neighbors:
				if neighbor not in dels:
					continue
				del_and_neighbor[tile][cell_class] = {**del_and_neighbor[tile][cell_class], **dels[neighbor][cell_class]}
	return del_and_neighbor

def tile_coords_x_y(tile):

	return (x, y)

def neighboring_tile_numbers(tile, all_tiles):
	neighbors = []
	keep_neighbors = []
	rows = [tile[0] - 1, tile[0], tile[0] + 1]
	cols = [tile[1] - 1, tile[1], tile[1] + 1]
	for col in cols:
		for row in rows:
			neighbors.append((row, col))
	for k in neighbors:
		if k in all_tiles:
			keep_neighbors.append(k)
	return keep_neighbors

def timing_estimate(start, now, count, length):
	avg_time = (now - start) / count
	remaining_time = (length - count) * avg_time
	minutes_remaining = int(remaining_time // 60)
	seconds_remaining = int(remaining_time % 60)
	minutes_elapsed = int((now - start) // 60)
	seconds_elapsed = int((now - start) % 60)
	print('Sequence: ' + str(count) + ' / ' + str(length), ', ' + "{:02d}".format(minutes_elapsed) + ':' + "{:02d}".format(seconds_elapsed) + ' elapsed, '
		' average: ' + "{0:.2f}".format(round(avg_time,2)) + 's per sequence, '+ "{:02d}".format(minutes_remaining) + ':' + "{:02d}".format(seconds_remaining) + ' remaining', end="\r")
	return 1

def process_detections(sequences, delaunay_in_neighbors, all_tiles, mask, config):
	cprint("Processing detections of " + str(len(sequences)) + ' sequences', 'green', 'on_white')
	seq_detections = {}
	tile_skip_list = []
	for item in sequences:
		# timing_estimate(start, time.time(), count, len(sequences))
		tile = item[0]
		seq = item[1]
		if tile not in all_tiles or tile in tile_skip_list:
			continue
		for seq_unique in seq:
			if seq_unique not in seq_detections:
				seq_key = seq_unique
				break
			cprint('DUPLICATE SEQUENCE KEY', 'red', 'on_white')
		if tile not in seq_detections:
			seq_detections[tile] = {}
			seq_detections[tile]['features'] = sequence_features(seq, tile, delaunay_in_neighbors, mask, config)
			seq_detections[tile]['seq'] = []
		if not seq_detections[tile]['features']:
			tile_skip_list.append(tile)
			seq_detections.pop(tile)
		else:
			seq_detections[tile]['seq'].append(seq)
	return seq_detections

# ********* In Patch coordinates, Y is FIRST coordinate, X is SECOND ***************
def sequence_features(seq, tile, delys, mask, config):
	features = {}
	class_features = {}
	if tile not in delys:
		return 0

	# Summarize info for clusters associated with a given tile
	for cell_class in delys[tile]:
		class_features[cell_class] = {}
		class_features[cell_class]['large_cluster_count'] = 0
		class_features[cell_class]['small_cluster_count'] = 0
		class_features[cell_class]['large_cell_count'] = 0
		class_features[cell_class]['small_cell_count'] = 0
		class_features[cell_class]['total_cluster_count'] = 0
		class_features[cell_class]['total_cell_count'] = 0
		class_features[cell_class]['cell_area'] = 0
		for cluster in delys[tile][cell_class]:
			size = delys[tile][cell_class][cluster]['size']
			if size > config.large_cluster:
				class_features[cell_class]['large_cluster_count'] += 1
				class_features[cell_class]['large_cell_count'] += size
			else:
				class_features[cell_class]['small_cluster_count'] += 1
				class_features[cell_class]['small_cell_count'] += size
			class_features[cell_class]['total_cell_count'] += size
			class_features[cell_class]['total_cluster_count'] += 1
			class_features[cell_class]['cell_area'] += size * delys[tile][cell_class][cluster]['mean_cell_area']

	# Process summarized info into features
	features['imm large / imm total'] = safe_division(class_features['Immune cells']['large_cell_count'],  class_features['Immune cells']['total_cell_count'])
	features['imm large / all cells'] = safe_division(class_features['Immune cells']['large_cell_count'], (class_features['Immune cells']['total_cell_count'] + class_features['Tumor']['total_cell_count']))
	features['imm large / tumor large'] = safe_division(class_features['Immune cells']['large_cell_count'], class_features['Tumor']['large_cell_count'])
	features['imm large / imm small'] = safe_division(class_features['Immune cells']['large_cell_count'], class_features['Immune cells']['small_cell_count'])
	features['all imm / all cells'] = safe_division(class_features['Immune cells']['total_cell_count'], (class_features['Immune cells']['total_cell_count'] + class_features['Tumor']['total_cell_count']))
	features['imm area / tumor area'] = safe_division(class_features['Immune cells']['cell_area'], class_features['Tumor']['cell_area'])
	features['imm area / total area'] = safe_division(class_features['Immune cells']['cell_area'], (class_features['Tumor']['cell_area'] + class_features['Immune cells']['cell_area']))

	# for feature in features:
	# 	if not features[feature]:
	# 		pdb.set_trace()

	return features

def safe_division(numerator, denominator):
	if denominator == 0:
		return 0
	else:
		return numerator / denominator

def add_one_feature(feature, cell, seq):
	if feature not in seq:
		seq[feature] = 0
	seq[feature] += cell[feature]

def regional_verification(seq_features, config, image, subject):
	one_tile = list(seq_features.keys())[0]
	features_list = sorted(seq_features[one_tile]['features'])
	header = ['subject', 'image', 'tile', 'sequence_number'] + features_list
	csv_path = os.path.join(config.features_path, 'regional_features.csv')
	if not os.path.exists(csv_path):
		with open(csv_path, 'w') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(header)
	with open(csv_path, 'a') as csvfile:
		writer = csv.writer(csvfile)
		for tile in seq_features:
			n = 0
			for seq in seq_features[tile]['seq']:
				n += 1
				row = [subject, image, str(tile), str(n)]
				for feature in features_list:
					row.append(seq_features[tile]['features'][feature])
				writer.writerow(row)
	return 1

def write_image_bin(image_bin, image_name, subject_ID, seq_features, config):
	image_path = os.path.join(config.image_data_folder_path, 'original_images', image_name)
	image = io.imread(image_path)
	# Data saved to binary files as [subject ID][image name][coordinates][sequence data]
	id_str_length = 5
	name_str_length = 92
	coord_str_length = config.num_steps*2*6

	num_features = len(next(iter(seq_features.items()))[1]['features'])*8
	sum_len = id_str_length + name_str_length + coord_str_length + num_features
	name_str_length += 8 - (sum_len % 8)
	
	ID_byte_string = str.encode(subject_ID.rjust(id_str_length))
	image_name_byte_string = str.encode(image_name[:-4].rjust(name_str_length))
	count = 0
	length = 0
	for tile in seq_features:
		length += len(seq_features[tile]['seq'])
	start = time.time()
	for tile in seq_features:
		feature_set = seq_features[tile]['features']
		for sequence in seq_features[tile]['seq']:
			if not len(sequence) == 20:
				pdb.set_trace()
			count += 1
			timing_estimate(start, time.time(), count, length)
			coord_byte_string = byte_string_from_coord_array(sequence, config.num_steps)
			if not len(ID_byte_string) == 5:
				pdb.set_trace()
			if not len(image_name_byte_string) == 99:
				pdb.set_trace()
			if not len(coord_byte_string) == 20*2*6:
				pdb.set_trace()
			image_bin.write(ID_byte_string)
			image_bin.write(image_name_byte_string)
			image_bin.write(coord_byte_string)
			# image_bin.write(struct.pack('i', len(feature_set))) # Number of feautures to follow, used for fixed-length reading of features
			feat_count = 0
			for feature in feature_set:
				# if float(feature_set[feature]) == 0:
				# 	pdb.set_trace()
				# if not feature_set[feature]:
				# 	pdb.set_trace()
				feat_count += 1
				image_bin.write(struct.pack('d', feature_set[feature]))
			for y,x in sequence:
				patch = image[y:(y+config.patch_size),x:(x+config.patch_size),:]
				# Need to verify this downscaling method				
				pyr = transform.pyramid_gaussian(patch, downscale=config.scaling_factor, multichannel=True)
				for p in pyr: 	
					if p.shape[0] == 100:
						patch_scaled = (p*255).astype('uint8')
						#patch_scaled = patch_scaled.astype('uint8')
				# patch_scaled = transform.downscale_local_mean(patch, (config.scaling_factor,config.scaling_factor,1)).astype('uint8')
				if not patch_scaled.size == 30000:
					pdb.set_trace()
				image_bin.write(patch_scaled)
	print()

#Checks if we're actually writing the correct image data as patches of sequences
def verify_images(all_centroids, image_name, subject_ID, config):
	image_path = os.path.join(config.image_data_folder_path, 'original_images', image_name)
	image = io.imread(image_path)
	for tile in all_centroids:
		folder_path = os.path.join(config.image_data_folder_path, image_name, str(tile))
		os.makedirs(folder_path, exist_ok=True)
		cprint(all_centroids[tile]['coords'],'red')
		for y, x in all_centroids[tile]['coords']:
			patch = image[y:(y+config.patch_size),x:(x+config.patch_size),:]
			patch_scaled = transform.downscale_local_mean(patch, (config.scaling_factor,config.scaling_factor,1)).astype('uint8')
			patch_name = os.path.join(folder_path,'' + str(x) + "_" + str(y) + ".tif")
			cprint(str((y,x)),'cyan')
			io.imsave(patch_name, patch_scaled)

def get_tile_count(main, bottom, right):
	tile_count = 1 # Always include corner
	tile_count += main.shape[0]*main.shape[1]
	tile_count += bottom.shape[0]*bottom.shape[1]
	tile_count += right.shape[0]*right.shape[1]
	return tile_count

def get_centroid_count(main, bottom, right):
	centroid_count = 1 # Always include corner
	if main:
		centroid_count += len(main)
	if bottom:
		centroid_count += len(bottom)
	if right:
		centroid_count += len(right)
	return centroid_count

def byte_string_from_coord_array(coord_array, num_steps):
	coord_string = ""
	for y in np.arange(num_steps):
		for x in np.arange(2):
			coord = "{:<6}".format(coord_array[y][x])
			coord_string = coord_string + coord
	return str.encode(coord_string)

def get_config():
	return OriginalPatchConfig()

def main():
	config = get_config()
	image_to_ID_csv_file = open(os.path.join(config.image_data_folder_path,"image_to_subject_ID.csv"),"r")
	reader = csv.reader(image_to_ID_csv_file, delimiter=",")
	image_to_ID_dict = dict()
	for line in reader:
		image_to_ID_dict[line[0]] = line[1]
	gauss_folder = os.makedirs(os.path.join(config.image_data_folder_path,'gaussian_patches_other_sampling'), exist_ok=True)
	mask_path = os.path.join(config.image_data_folder_path, "masks")
	mask_list = [os.path.join(config.image_data_folder_path, mask_path, f) for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]
	for mask in mask_list:
		image_name = mask.split('/')[-1][5:]
		image_path = os.path.join(config.image_data_folder_path, 'original_images', image_name)
		gauss_bin_path = os.path.join(gauss_folder, image_name.split('.')[0] + '.bin')
		
		if image_name not in image_to_ID_dict:
			cprint('Skipping ' + image_name, 'white', 'on_grey')
			continue

		if os.path.exists(gauss_bin_path):# and not FLAGS.overwrite:
			cprint('Skipping ' + image_name, 'white', 'on_grey')
			continue
		cprint("*-._.-*-._.*" + image_name + "*-._.-*-._.-*", 'white', 'on_green')
		sequences = generate_sequences(mask, config)
		image_bin = open(gauss_bin_path, 'wb+')
		# verify_images(all_centroids, image_name, image_to_ID_dict[image_name], config)
		# cprint("Writing binary file...", 'green', 'on_white')
		write_image_bin(image_bin, image_name, image_to_ID_dict[image_name], sequences, config)
		image_bin.close()
		print('\n')

# Main body
if __name__ == '__main__':
	main()