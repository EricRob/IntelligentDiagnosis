#!/usr/bin/env python3

import numpy as np
import numbers
import os
import sys
from scipy import misc
from matplotlib.image import imread
from numpy.lib.stride_tricks import as_strided
from math import floor

def receive_inputs ():

	return


# Create patches and load into 6-dimension array with properties:
# (patch column, patches row, 0, patchSize, patchSize, depth of original image)
def extract_patches (originalPath, patchSize, overlap):
	
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
def save_patches (originalPath, patches):

	# Create enclosing folder to hold patch dump
	folderPath = os.path.abspath(os.path.join(originalPath, os.pardir)) + '/' + os.path.splitext(originalPath)[0] + '_patches'
	os.makedirs(folderPath, exist_ok=True)

	# Iterate through patches array, creating individual tiffs of each patch.
	print('Saving...')
	for i in range(patches.shape[0]):
		for j in range(patches.shape[1]):
			name = folderPath + '/' + os.path.splitext(originalPath)[0] + '_' + str(i) + '_' + str(j) + '.tif'
			misc.imsave(name,patches[i,j,0,:,:,:])

	return patches.shape[0]*patches.shape[1]


if __name__ == '__main__':

	# Specify image to be patched, size of each path, and amount of overlap between patches
	originalPath = str(sys.argv[1])
	patchSize = int(sys.argv[2])
	overlap = int(sys.argv[3])

	# originalPath = str(input('Image filepath: '))
	# patchSize = int(input('Patch size: '))
	# overlap = int(input('Overlap percentage: '))


	patches = extract_patches(originalPath, patchSize, overlap)

	numPatches = save_patches(originalPath, patches)

	print('Created ' + str(numPatches) + ' patches from ' + originalPath)

	sys.exit(0)



