#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import os

# Global variables

# Class declarations

# Function declarations

def main():
	add_to_nr = ['/data/recurrence_seq_lstm/data_conditions/add_data/nonrecurrence_test.bin']
	add_to_r = ['/data/recurrence_seq_lstm/data_conditions/add_data/recurrence_test.bin']

	nr_bin = os.path.join('/data/recurrence_seq_lstm/data_conditions/yale_testing_data/nonrecurrence_test.bin')
	r_bin = os.path.join('/data/recurrence_seq_lstm/data_conditions/yale_testing_data/recurrence_test.bin')

	with open(nr_bin, 'ab+') as bin_file:
		for add_nr in add_to_nr:
			with open(add_nr, 'rb') as add:
				write_bytes = add.read(os.path.getsize(add_nr))
			bin_file.write(write_bytes)

	# with open(r_bin, 'ab+') as bin_file:
	# 	for add_r in add_to_r:
	# 		with open(add_r, 'rb') as add:
	# 			write_bytes = add.read(os.path.getsize(add_r))
	# 		bin_file.write(write_bytes)
# Main body
if __name__ == '__main__':
    main()