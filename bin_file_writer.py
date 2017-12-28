import numpy as np
import os
import sys
from skimage import io

# ATTN: must be run within parent folder for set of patches

if __name__ == '__main__':

	# pass in binary file name as first parameter
	bin_name = str(sys.argv[1])

	# create a new byte file that can be appended
	bin_file = open(bin_name, "ab+")

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
			bin_file.write(arr.tobytes())

	bin_file.close()

	sys.exit(0)
