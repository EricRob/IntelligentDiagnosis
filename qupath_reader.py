


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


# Global variables
DETECTIONS = '/data/QuPath/CellCounter/detections'

# Class declarations

# Function declarations

def main():
	d = {}
	n = 1
	tsvfile = '00_02_1_1 Detectionstxt.txt'
	tsvfile = os.path.join(DETECTIONS, tsvfile)
	with open(tsvfile, 'r') as f:
		reader = csv.DictReader(f, delimiter='\t')
		print(reader.fieldnames)
		for row in reader:
			d['{0:06}'.format(n)] = row
			n += 1
	for cell in d:
		d[cell].pop('Centroid Y px', None)
		d[cell].pop('Centroid X px', None)


# Main body
if __name__ == '__main__':
	main()