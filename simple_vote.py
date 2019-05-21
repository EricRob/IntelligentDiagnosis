# Imports
import sys
import os
import csv
from sklearn.metrics import roc_auc_score
import pdb

# Function declarations

class Config(object):
    voting_filename = os.path.join('/data/recurrence_seq_lstm/results/tester_list/voting_file.csv')
    output_csv = os.path.join('./voting_results.csv')
    vote_cutoff = 0.5

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
        self.images = {row['names'].strip().upper(): VotingSummary(row, category='image')}
        return
    
    def init_image(self, row):
        self.name = row['names'].strip().upper()

    def add_vote(self, row):
        if (self.subject + row['coords']) in self.keys:
            return
        else:
            self.recur_votes += int(row['output'])
            self.nonrecur_votes += (int(row['output'])-1)*-1
            self.total_votes += 1
            self.keys.append(self.subject + row['coords'])
            if self.category == 'subject':
                image = row['names'].strip().upper()
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


def process_voting_input(voting_filename):
    subjects = {}
    
    with open(voting_filename) as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            subject = row['ID'].strip().upper()
            if subject in subjects:
                subjects[subject].add_vote(row)
            else:
                subjects[subject] = VotingSummary(row, category='subject')

    return subjects

def success_label(subject):
    if subject.truth_label == subject.net_label:
        return 'PASS'
    else:
        return 'FAIL'
def accurate_votes(subject):
    if subject.truth_label:
        return subject.recur_votes
    else:
        return subject.nonrecur_votes

def output_voting_results(subjects, config):
    truth = []
    score = []
    rec_votes = []
    nrec_votes = []
    iter_list = sorted(subjects.items())
    for _, subject in iter_list:
        subject.assign_network_label(config.vote_cutoff)
        truth.append(subject.truth_label)
        score.append(subject.net_label)
        if subject.truth_label:
            rec_votes.append(subject.net_label)
        else:
            nrec_votes.append(subject.net_label)
    sensitivity = sum(rec_votes) / len(rec_votes)
    specificity = (len(nrec_votes) - sum(nrec_votes)) / len(nrec_votes)
    accuracy = (len(nrec_votes) - sum(nrec_votes) + sum(rec_votes)) / (len(rec_votes) + len(nrec_votes))
    auc = roc_auc_score(truth, score)

    with open(config.output_csv, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['SUMMARY'])
        csvwriter.writerow(['Sensitivity:', '%.3f' % sensitivity])
        csvwriter.writerow(['Specificity:', '%.3f' % specificity])
        csvwriter.writerow(['Accuracy:', '%.3f' % accuracy])
        csvwriter.writerow(['AUC:', '%.3f' % auc])
        csvwriter.writerow(['Nonrecurrent passed:', len(nrec_votes) - sum(nrec_votes)])
        csvwriter.writerow(['Nonrecurrent count:', len(nrec_votes)])
        csvwriter.writerow(['Recurrent passed:', sum(rec_votes)])
        csvwriter.writerow(['Recurrent count:', len(rec_votes)])
        csvwriter.writerow(['Cutoff:', config.vote_cutoff])
        csvwriter.writerow([''])
        csvwriter.writerow(['Subject ID', 'Success', 'Ground Truth',
            'Network Score','Accurate Votes', 'Total Votes', 'Image Count'])
        for _, subject in iter_list:
            csvwriter.writerow([subject.subject,
                success_label(subject),
                subject.truth_label,
                '%.2f' % subject.vote,
                accurate_votes(subject),
                subject.total_votes,
                len(subject.images)])
    return

def main():
    config = Config()
    subjects = process_voting_input(config.voting_filename)
    output_voting_results(subjects, config)

# Main body
if __name__ == '__main__':
    main()