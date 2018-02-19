#!/usr/bin/env python3

import numpy as np
import numbers
import os
import sys
from scipy import misc
from matplotlib.image import imread
from numpy.lib.stride_tricks import as_strided
from math import floor
from PIL import Image
import pdb

# Create patches and load into 6-dimension array with properties:
# (patch column, patches row, 0, patch_size, patch_size, depth of original image)
def extract_patches (original_path, patch_size, overlap):
    print("Patching " + original_path)
    # Load image and convert to numpy array
    im = imread(original_path)
    arr = np.array(im)
    arr_ndim = arr.ndim

    # Create patches in same shape as original image, and determing step size for given overlap
    patch_shape = (patch_size,patch_size,arr.shape[2])
    extraction_step = floor(patch_size*(1-overlap*.01))

    # Assure all objects are tuples of the same shape
    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    # consistent stride values necessary for arrays of indeterminate size and shape
    patchStrides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patchStrides))

    # Create views into the array with given shape and stride
    print('Extracting...')
    patches = as_strided(arr, shape=shape, strides=strides)
    
    return patches

# Create a folder to hold patches if one doesn't exist, then
# use it to save tiffs from the patches array
def save_patches (original_path, data_parent, label, sample_patches, mask_patches, keep_percentage, patch_size, overlap, sample_size):

    # Create enclosing folder to hold patch dump
    # folder_path = os.path.abspath(os.path.join(original_path, os.pardir)) + '/' + label + '/' + os.path.splitext(original_path)[0] + '_patches'
    folder_path = os.path.join(os.path.abspath(os.path.join(original_path, os.pardir)), data_parent, label, (os.path.splitext(original_path)[0] + '_patches'))
    os.makedirs(folder_path, exist_ok=True)
    x_index = 0
    y_index = 0
    patch_count = 0

    # Iterate through sample_patches array, downsample the patch to patch_size,
    # and create individual tiffs of each downsampled patch
    print('Saving...')
    for i in range(sample_patches.shape[0]):
        for j in range(sample_patches.shape[1]):
            single_mask_patch = mask_patches[i,j,0,:,:,:]
            
            if (patch_threshold(single_mask_patch, keep_percentage)):
                # name is [orignal title]_[top left x pixel]_[top left y pixel]
                name = folder_path + '/' +  os.path.splitext(original_path)[0] + '_' + str(x_index) + '_' + str(y_index) + '.tif'
                img = Image.fromarray(sample_patches[i,j,0,:,:,:], 'RGB')
                img = img.resize((sample_size,sample_size), Image.ANTIALIAS)
                img.save(name)
                patch_count += 1
            
            x_index += floor(patch_size*(1-overlap*.01))
        x_index = 0
        y_index += floor(patch_size*(1-overlap*.01))

    return patch_count

# Determine if patch should be saved depending on the amount of background space
def patch_threshold(single_mask_patch, keep_percentage):
    arr = single_mask_patch[:,:,0]
    counter = 0
    for x in np.nditer(arr):
        if x >= 217:
            counter += 1

    if (counter / arr.size) < ((100 - keep_percentage)/100):
        return True
    
    return False

def seg_threshold_list(seg_patches, config):
    meet_threshold_list = []
    x_index = 0
    y_index = 0
    for i in range(seg_patches.shape[0]):
        for j in range(seg_patches[1]):
            if (seg_above_threshold(seg_patches[i,j,0,:,:,0], config.keep_percentage)):
                meet_threshold_list = meet_threshold_list + [(x_index, y_index)]
            x_index += floor(config.patch_size*(1-config.overlap*.01))
        x_index = 0
        y_index += floor(config.patch_size*(1-config.overlap*.01))

    return meet_threshold_list

def seg_above_threshold(single_seg_patch, keep_percentage):
    total_pixels = single_seg_patch.shape[0]*single_seg_patch.shape[1]
    patch_sum = np.sum(single_seg_patch)
    patch_threshold = total_pixels * (1-keep_percentage/100) * 255
    if patch_sum <= patch_threshold:
        return True
    else:
        return False