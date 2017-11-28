#!/usr/bin/env python3

import numpy as np
import numbers
import os
import sys
from scipy import misc
from matplotlib.image import imread
from numpy.lib.stride_tricks import as_strided
from math import floor
from skimage.measure import block_reduce



# Create patches and load into 6-dimension array with properties:
# (patch column, patches row, 0, patchSize, patchSize, depth of original image)
def extract_patches (originalPath, patchSize, overlap):
    print("Patching " + originalPath)
    # Load image and convert to numpy array
    im = imread(originalPath)
    arr = np.array(im)
    arr_ndim = arr.ndim

    # Create patches in same shape as original image, and determing step size for given overlap
    patchShape = (patchSize,patchSize,arr.shape[2])
    extractionStep = floor(patchSize*(1-overlap*.01))

    # Assure all objects are tuples of the same shape
    if isinstance(patchShape, numbers.Number):
        patchShape = tuple([patchShape] * arr_ndim)
    if isinstance(extractionStep, numbers.Number):
        extractionStep = tuple([extractionStep] * arr_ndim)

    # consistent stride values necessary for arrays of indeterminate size and shape
    patchStrides = arr.strides

    slices = [slice(None, None, st) for st in extractionStep]
    indexingStrides = arr[slices].strides

    patchIndicesShape = ((np.array(arr.shape) - np.array(patchShape)) //
                           np.array(extractionStep)) + 1

    shape = tuple(list(patchIndicesShape) + list(patchShape))
    strides = tuple(list(indexingStrides) + list(patchStrides))

    # Create views into the array with given shape and stride
    print('Extracting...')
    patches = as_strided(arr, shape=shape, strides=strides)
    
    return patches

# Create a folder to hold patches if one doesn't exist, then
# use it to save tiffs from the patches array
def save_patches (originalPath, patches, patchSize, overlap, sampleFactor):

    # Create enclosing folder to hold patch dump
    folderPath = os.path.abspath(os.path.join(originalPath, os.pardir)) + '/' + os.path.splitext(originalPath)[0] + '_patches'
    os.makedirs(folderPath, exist_ok=True)
    x_index = 0
    y_index = 0

    # Iterate through patches array, downsample the patch by sampleFactor,
    # and create individual tiffs of each downsampled patch
    print('Saving...')
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            name = folderPath + '/' +  os.path.splitext(originalPath)[0] + '_' + str(x_index) + '_' + str(y_index) + '.tif'
            downsample = block_reduce(patches[i,j,0,:,:,:], block_size=(sampleFactor,sampleFactor,1), func=np.mean)
            misc.imsave(name,downsample)
            #misc.imsave(name, patches[i,j,0,:,:,:])
            x_index += floor(patchSize*(1-overlap*.01))
        y_index += floor(patchSize*(1-overlap*.01))

    return patches.shape[0]*patches.shape[1]



if __name__ == '__main__':

    # Specify image to be patched, size of each path, and amount of overlap between patches
    originalPath = str(sys.argv[1])
    patchSize = int(sys.argv[2])
    overlap = int(sys.argv[3])
    sampleFactor = int(sys.argv[4])


    patches = extract_patches(originalPath, patchSize, overlap)

    numPatches = save_patches(originalPath, patches, patchSize, overlap, sampleFactor)

    print('Created ' + str(numPatches) + ' patches from ' + originalPath)

    sys.exit(0)
