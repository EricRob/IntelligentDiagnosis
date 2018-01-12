import numpy as np
import os
import sys
from skimage import io

# ATTN: must be run within parent folder for set of patches

if __name__ == '__main__':

	# pass in binary file name as first parameter
	bin_name = str(sys.argv[1])

	# create a new byte file that can be appended
	bin_train_file = open(str(bin_name + "_train.bin"), "ab+")
	bin_valid_file = open(str(bin_name + "_valid.bin"), "ab+")
	bin_test_file = open(str(bin_name + "_test.bin"), "ab+")

	TRAIN_PORTION = 0.7
	VALID_PORTION = 0.2
	TEST_PORTION = 0.1

	total_files = sum([len(files) for r, d, files in os.walk(".")])
	counter = 1

	# walk over all files in starting directory and sub-directories
	for dirpath, dirnames, filenames in os.walk("."):
		lst = os.listdir(dirpath)
		print(dirpath)
		lst.sort() # need files in order by index (see fix_patch_alpha) so time series is accurately fed to RNN
		for filename in [f for f in lst if f.endswith(".tif")]:
			# open tiff as grayscale, convert to unsiged 8-bit integer, and append to file
			img = io.imread(os.path.join(dirpath,filename), as_grey=True)
			arr = np.floor(img*255)
			arr = np.uint8(arr)
			if(counter / total_files < TRAIN_PORTION):
				bin_train_file.write(arr.tobytes())
				counter += 1
			elif(counter / total_files < (TRAIN_PORTION + VALID_PORTION)):
				bin_valid_file.write(arr.tobytes())
				counter += 1
			else:
				bin_test_file.write(arr.tobytes())

	bin_train_file.close()
	bin_valid_file.close()
	bin_test_file.close()

sys.exit(0)
