#!/usr/bin/python3 python3

# Imports
import sys
import csv
import os
import pdb
import random
import numpy as np
from termcolor import cprint
import argparse

parser = argparse.ArgumentParser(description='Unpack binary files to determine the subjects and images they contain')

# Global variables
parser.add_argument('--cond', default=None, required=True, type=str, help='Name of condition to unpack')

SEQ_LENGTH = 20
NUM_FEATURES = 7


SUBJ_BYTES = 5
NAME_BYTES = 92
COORD_BYTES = SEQ_LENGTH*2*6
FEAT_BYTES = NUM_FEATURES*8
IMG_BYTES = SEQ_LENGTH*100*100*3

DATA_CONDITIONS = os.path.join('/data', 'recurrence_seq_lstm', 'data_conditions')


FLAGS = parser.parse_args()
# Class declaration

def main():

    BINARIES = ['recurrence_train.bin', 'recurrence_test.bin', 'recurrence_valid.bin',
    'nonrecurrence_train.bin', 'nonrecurrence_test.bin', 'nonrecurrence_valid.bin']


    for binary in BINARIES:
            with open(os.path.join(DATA_CONDITIONS, ARGS.cond, binary), 'rb') as file:
                while (file.peek(1)):
                    subj = file.read(SUBJ_BYTES).decode("utf-8")
                    name = file.read(NAME_BYTES).decode("utf-8").strip()
                    _ = file.read(COORD_BYTES + FEAT_BYTES + IMG_BYTES)
                    if subj not in data:
                        data[subj] = []
                    if name not in data[subj]:
                        data[subj].append(name)
                        print('subject: ' + subj + ' -- image: ' + name)
    pdb.set_trace()

# Main body
if __name__ == '__main__':
    main()