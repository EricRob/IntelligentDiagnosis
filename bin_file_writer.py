import numpy as np
import os
import sys
from skimage import io
from IPython import embed
import math
import pdb

# ATTN: must be run within parent folder for set of patches

if __name__ == '__main__':

	# pass in binary file name as first parameter
	bin_name = str(sys.argv[1])
	sequence_overlap_percentage = int(sys.argv[2]) / 100
	num_steps = int(sys.argv[3])
	mode = str(sys.argv[4])

	bin_file = open(bin_name + "_" + mode + ".bin", "ab+")

	IMAGE_HEIGHT = 100
	IMAGE_WIDTH = 100
	IMAGE_DEPTH = 3
	
	image_bytes = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH

	write_array = np.zeros([num_steps, image_bytes], dtype=np.uint8)
	write_stride = int(math.floor(num_steps * (1-sequence_overlap_percentage)))

	total_files = sum([len(files) for r, d, files in os.walk(".")])
	total_dirs = sum([len(d) for r, d, f in os.walk(".")])
	counter = 0
	dir_counter = 0

	
	# walk over all files in starting directory and sub-directories
	folder_file = open(bin_name + "_" + mode + "_subjects.txt")
	subjects_list = folder_file.read().splitlines()
	for subject in subjects_list:
		dir_counter +=1
		print("Writing " + subject + " -- %i/%i" % (dir_counter, len(subjects_list)))
		image_list = os.listdir(subject)
		#image_list.sort()
		#pdb.set_trace()
		img_dict = {}	
		img_dict_y = {}
		for image_name in image_list:
			if image_name.endswith(".tif"):
				#pdb.set_trace()
				image_name_noex = os.path.splitext(image_name)[0]
				image_name_chop = image_name_noex.split("_")
				x_coord = int(image_name_chop[len(image_name_chop)-2])
				y_coord = int(image_name_chop[len(image_name_chop)-1])
				#pdb.set_trace()
				img_dict[x_coord] = img_dict.get(x_coord, {})
				img_dict[x_coord][y_coord] = image_name
				#image_name
		#pdb.set_trace()

		# for key in sorted(img_dict.keys()): #.items():
		# 	print(key) #, value)
		# 	for key2 in sorted(img_dict[key].keys()):
		# 		#print(key2)
		# 		print(img_dict[key][key2])

		for filename in [f for f in image_list if f.endswith(".tif")]:
			counter +=1
			img = io.imread(os.path.join(subject,filename))
			arr = img.flatten()

			if(counter <= num_steps):
				write_array[counter-1,:] = arr
			else: # Shift all images in the write_array to the left one step, then replace the last value with the new image data
				write_array = np.roll(write_array, -1, axis=0)
				write_array[num_steps - 1,:] = arr
			
			# if counter == write_stride:		
			# 	pdb.set_trace()
			write_count = counter - num_steps
			# After the a new portion of the sequence has been added, add the write_array to the binary file
			if(write_count >= 0) & (write_count % write_stride == 0):
				np.save(bin_file, write_array)
				# pdb.set_trace()
	bin_file.close()

sys.exit(0)
