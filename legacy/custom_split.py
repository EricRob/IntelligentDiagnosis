
#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import os
import argparse
import pdb
import numpy as np
from termcolor import cprint
from skimage import io

# Global variables
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--minx', required=True, type=int, help='bottom x coordinate of rectangle to be cut')
parser.add_argument('--maxx', required=True, type=int, help='top x coordinate of rectangle to be cut')

parser.add_argument('--miny', required=True, type=int, help='bottom y coordinate of rectangle to be cut')
parser.add_argument('--maxy', required=True, type=int, help='top y coordinate of rectangle to be cut')

parser.add_argument('--image', required=True)
parser.add_argument('--suffix', required=True)

# Class declarations

# Function declarations

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def main():
	options = parser.parse_args(sys.argv[1:])
	image_path = options.image
	minx = options.minx
	miny = options.miny
	maxx = options.maxx
	maxy = options.maxy
	suffix = options.suffix
	ans = True
	cprint("Loading " + image_path + " into memory", 'yellow')
	image = io.imread(image_path)
	while ans:
		crop = image[miny:maxy, minx:maxx, :]
		cprint("Saving " + os.path.splitext(image_path)[0] + "_" + str(suffix) + ".tif...", 'green')
		io.imsave(os.path.splitext(image_path)[0] + "_" + str(suffix) + ".tif", crop)
		del crop
		ans = query_yes_no("More crops?", default="no")
		if ans:
			sys.stdout.write("minx: ")
			minx = int(input())

			sys.stdout.write("maxx: ")
			maxx = int(input())
			
			sys.stdout.write("miny: ")
			miny = int(input())
			
			sys.stdout.write("maxy: ")
			maxy = int(input())

			sys.stdout.write("suffix (int): ")
			suffix = int(input())



# Main body
if __name__ == '__main__':
	main()