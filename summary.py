#!/usr/bin/python3 python3

'''
##################

Written by Eric J Robinson (https://github.com/EricRob)

##################

Create graphs to display the sensitivity, specificity, accuracy, and loss of the recurrence_lstm network training.

##################
'''

# Imports
import sys
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import pdb
import csv
import math
from termcolor import cprint
import pickle
import numpy as np
from config import Config

# Global variables


# Class declarations
class Summary:
    def __init__(self, condition, model, TO_PLOT):
        self.condition = condition
        self.sensitivity = []
        self.specificity = []
        self.accuracy = []
        self.loss = []
        self.model = model
        self.file = os.path.join(model, '%s_results.txt' % condition)
        self.load_file()
        self.create_dicts(TO_PLOT)
        self.set_color()

    def add_row(self, row):
        self.sensitivity.append(float(row[0]))
        self.specificity.append(float(row[1]))
        self.accuracy.append(float(row[4]))
        self.loss.append(float(row[7]))

    def load_file(self):
        try:
            with open(self.file) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    self.add_row(row)
            self.broken = False
        except Exception as e:
            self.broken = True
            cprint('[ERROR] Encountered issue opening %s -- Does it exist at that location?' % self.file, 'red')

    def create_dicts(self, TO_PLOT):
        self.data = {}
        for item in TO_PLOT:
            self.data[item] = getattr(self, item)
    def set_color(self):
        if self.condition == 'train':
            self.color = 'b'
        elif self.condition == 'valid':
            self.color = 'r'
        elif self.condition == 'test':
            self.color = 'g'
        else:
            self.color = 'black'

# Function declarations

def factor_int(n):
    # Find the set of integer factors closest to a square. Hopefully n isn't prime, or this will be an image of a long row
    val = math.ceil(math.sqrt(n))
    solution = False
    while not solution:
        val2 = int(n/val)
        if val2 * val == float(n):
            solution = True
        else:
            val-=1
    return val, val2

def add_subplot(summaries, fig, plot_name, idx):
    subplot = fig.add_subplot(idx)
    epochs = None
    for summary in summaries:
        if summary.broken:
            continue
        epochs = np.arange(1, len(summary.data[plot_name]) + 1)
        print('%s -- cond: %s, epochs: %d, plot_len:%d' % (plot_name, summary.condition, len(epochs), len(summary.data[plot_name])))
        subplot.plot(epochs, summary.data[plot_name], label=summary.condition, color=summary.color)
    subplot.set_title(plot_name)

    if not len(epochs):
        cprint('[ERROR] Unable to produce %s subplot, no data available' % plot_name, 'red')
        return
    
    if not plot_name == 'loss':
        subplot.set_ylim([0,1.05]) 
        subplot.plot(epochs, np.repeat(0.5, len(epochs)), color='black', ls='--')
    if idx % 10 == 1:
        subplot.legend()
    return

def get_config(config_name):
    try:
        if config_name == 'default':
            with open('./default_config.file', 'rb') as f:
                config = pickle.load(f)
        else:
            if '.file' not in config_name:
                config_name += '.file'
            with open(os.path.join(config_name), 'rb') as f:
                config = pickle.load(f)
        return config
    except:
        print('[ERROR] No valid config file: %s.' % config_name)
        print('[INFO] Check --conf parameter and make sure you have run config.py for initial setup.')

def valid_summary_exists(summaries):
    for summary in summaries:
        if not summary.broken:
            return True
    return False

def main(ars):
    # Change this to add or remove data from the graphs.
    TO_PLOT = ['sensitivity', 'specificity', 'accuracy', 'loss']
    config = get_config(ars.conf)
    model_path = os.path.join(config.results_dir, ars.model)

    # matplotlib magic
    fig_x, fig_y = factor_int(len(TO_PLOT))
    subplot_idx = fig_y*100 + fig_x*10 + 1
    

    # This is what we will summarize
    summaries = [Summary('train', model_path, TO_PLOT), Summary('valid', model_path, TO_PLOT), Summary('test', model_path, TO_PLOT)]

    if not valid_summary_exists(summaries):
        cprint('[ERROR] No valid training summary files exist, exiting', 'red', 'on_white')
        sys.exit(1)
    # epochs = np.arange(1, len(summaries[0].data[TO_PLOT[0]]) + 1)
    fig = plt.figure()
    for plot in TO_PLOT:
        add_subplot(summaries, fig, plot, subplot_idx)
        subplot_idx += 1
    
    fig.tight_layout()
    fig.suptitle(ars.model, fontsize=20)
    fig.savefig(os.path.join(model_path,'%s_training_summary.jpg' % ars.model))


# Main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize training based on exported info from recurrence_lstm')
    parser.add_argument('--conf', default='default', type=str, help='Name of configuration file for processing and voting')
    parser.add_argument('--model', default='None', type=str, required=True, help='Model directory for training summary')
    ars = parser.parse_args()
    main(ars)
    sys.exit(0)