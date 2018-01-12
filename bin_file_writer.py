import numpy as np
import os
import sys
from skimage import io
from IPython import embed
import math

# ATTN: must be run within parent folder for set of patches

if __name__ == '__main__':

	# pass in binary file name as first parameter
	bin_name = str(sys.argv[1])
	sequence_overlap_percentage = int(sys.argv[2]) / 100
	num_steps = int(sys.argv[3])

	# create a new byte file that can be appended
	bin_train_file = open(str(bin_name + "_train.bin"), "ab+")
	bin_valid_file = open(str(bin_name + "_valid.bin"), "ab+")
	bin_test_file = open(str(bin_name + "_test.bin"), "ab+")

	TRAIN_PORTION = 0.7
	VALID_PORTION = 0.2
	TEST_PORTION = 0.1

	IMAGE_BYTES = 10000

	write_array = np.zeros([num_steps, IMAGE_BYTES], dtype=np.uint8)
	write_stride = int(math.ceil(num_steps * (1-sequence_overlap_percentage)))

	total_files = sum([len(files) for r, d, files in os.walk(".")])
	counter = 0

	# walk over all files in starting directory and sub-directories
	for dirpath, dirnames, filenames in os.walk("."):
		lst = os.listdir(dirpath)
		lst.sort() # need files in order by index (see fix_patch_alpha) so time series is accurately fed to RNN]
		
		for filename in [f for f in lst if f.endswith(".tif")]:
			
			counter +=1

			# open tiff as grayscale, convert to unsiged 8-bit integer, and append to file
			img = io.imread(os.path.join(dirpath,filename), as_grey=True)
			arr = np.floor(img*255)
			arr = np.uint8(arr)
			arr = arr.flatten()

			# At start of iterations, first fill write_array with data until it contains one complete sequence
			if(counter <= num_steps):
				write_array[counter-1,:] = arr
			else: # Shift all images in the write_array to the left one step, then replace the last value with the new image data
				write_array = np.roll(write_array, -1, axis=0)
				write_array[num_steps - 1,:] = arr

			# After the a new portion of the sequence has been added, add the write_array to the binary file
			if(counter % write_stride == 0):
				print(filename + " -- writing " + str(counter))
				if(counter / total_files < TRAIN_PORTION):
					bin_train_file.write(write_array.tobytes())
				elif(counter / total_files < (TRAIN_PORTION + VALID_PORTION)):
					bin_valid_file.write(write_array.tobytes())
				else:
					bin_test_file.write(write_array.tobytes())

	bin_train_file.close()
	bin_valid_file.close()
	bin_test_file.close()

	print("stride: " + str(write_stride))

sys.exit(0)
