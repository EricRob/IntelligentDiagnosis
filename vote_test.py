#!/usr/bin/python3 python3

'''
##################

Written by Eric J Robinson (https://github.com/EricRob)

##################

Process output of recurrence_lstm into voting scores
for all subjects in a testing condition.

Outputs a csv with a status for each subject and some
basic summary statistics

##################

'''

# Imports
import sys
import os
import csv
from sklearn.metrics import roc_auc_score
import pdb
import argparse
import pickle
from art import tprint
from config import Config

# Function declarations


class VotingSummary:
    def __init__(self, row, category='subject'):
        self.subject = row['ID'].strip().upper()
        self.recur_votes = int(row['output'])
        self.nonrecur_votes = (int(row['output'])-1)*-1
        self.total_votes = 1
        self.truth_label = int(row['label'])
        self.str_truth_label = self.assign_str_label(self.truth_label)
        self.keys = [self.subject + row['coords']]
        self.category = category
        if category == 'subject':
            self.init_subject(row)
        elif category == 'image':
            self.init_image(row)

    def init_subject(self, row):
        self.images = {row['name'].strip().upper(): VotingSummary(row, category='image')}
        return
    
    def init_image(self, row):
        self.name = row['name'].strip().upper()

    def add_vote(self, row):
        if (self.subject + row['coords']) in self.keys:
            return
        else:
            self.recur_votes += int(row['output'])
            self.nonrecur_votes += (int(row['output'])-1)*-1
            self.total_votes += 1
            self.keys.append(self.subject + row['coords'])
            if self.category == 'subject':
                image = row['name'].strip().upper()
                if image not in self.images:
                    self.images[image] = VotingSummary(row, category='image')
                else:
                    self.images[image].add_vote(row)
    
    def assign_str_label(self, label):
        if label:
            return 'RECURRENT'
        else:
            return 'NONRECURRENT'

    def assign_network_label(self, cutoff):
        self.vote = self.recur_votes / self.total_votes
        if self.vote >= cutoff:
            self.net_label = 1
        else:
            self.net_label = 0
        self.str_net_label = self.assign_str_label(self.net_label)


def process_voting_input(voting_csv):
    subjects = {}
    count = 0
    line_art = '^o^ =========='
    ending = "((0))"
    line_index = 4
    with open(voting_csv) as csvfile:  
        csvreader = csv.DictReader(csvfile)
        row_count = sum(1 for row in csvreader)
        row_print = row_count // (len(line_art) - line_index)

    with open(voting_csv) as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            subject = row['ID'].strip().upper()
            if subject in subjects:
                subjects[subject].add_vote(row)
            else:
                subjects[subject] = VotingSummary(row, category='subject')
            if count % row_print == 0:
                print(line_art[:line_index] + ending, end='\r')
                line_index += 1
            count += 1
    print(line_art + '==> Votes counted!')
    return subjects

def success_label(subject):
    if subject.truth_label == subject.net_label:
        return 'PASS'
    else:
        return 'FAIL'

def prediction_label(subject):
    if subject.net_label == 0:
        return 'NONRECURRENT'
    else:
        return 'RECURRENT'
        
def accurate_votes(subject):
    if subject.truth_label:
        return subject.recur_votes
    else:
        return subject.nonrecur_votes

def output_voting_results(subjects, config, model):
    print('Writing output csv...')
    truth = []
    score = []
    rec_votes = []
    nrec_votes = []
    iter_list = sorted(subjects.items())
    for _, subject in iter_list:
        subject.assign_network_label(config.vote_cutoff_float)
        truth.append(subject.truth_label)
        score.append(subject.net_label)
        if subject.truth_label:
            rec_votes.append(subject.net_label)
        else:
            nrec_votes.append(subject.net_label)

    with open(os.path.join(config.results_dir, model, config.output_csv), 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['SUMMARY'])
        csvwriter.writerow([''])
        csvwriter.writerow(['Subject ID', 'Prediction',
            'Network Score','Accurate Votes', 'Total Votes', 'Image Count'])
        for _, subject in iter_list:
            csvwriter.writerow([subject.subject,
                prediction_label(subject),
                '%.2f' % subject.vote,
                accurate_votes(subject),
                subject.total_votes,
                len(subject.images)])
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

def main(ars):
    config = get_config(ars.conf)
    if not config:
        sys.exit(1)
    tprint('majo\nrity\nvote', font='sub-zero')
    subjects = process_voting_input(os.path.join(config.results_dir, ars.model, config.voting_csv))
    output_voting_results(subjects, config, ars.model)
    print('Done!')

# Main body
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create decisions based on output of recurrence_seq_lstm')
    parser.add_argument('--conf', default='default', type=str, help='Name of configuration file for processing and voting')
    parser.add_argument('--model', default='None', type=str, required=True, help='Directory location of voting_csv to test and summarize')
    ars = parser.parse_args()
    main(ars)
    sys.exit(0)
