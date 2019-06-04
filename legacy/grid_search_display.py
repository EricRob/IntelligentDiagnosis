
#!/user/bin/env python3 -tt
"""
Module documentation.
"""

# Imports
import sys
import argparse
import os
import csv
import pdb
import numpy as np
import matplotlib.pyplot as plt
import ipdb

# Global variables

CSV_PATH = '/data/recurrence_seq_lstm/feature_testing/delaunay_grid_search.csv'
parser = argparse.ArgumentParser(description='Create a grid search display of the available delaunay data')

ARGS = parser.parse_args()

# Class declarations

# Function declarations

def main():
    data = {}
    feature_list = []
    x_ticks = []
    y_ticks = []
    name_dict = { 'imm small / total AUC': 'Unclustered Immune Count / Total Count AUC', \
    'imm large / all imm AUC': 'Clustered Immune Count / Immune Count AUC', \
    'imm large / tum large AUC': 'Clustered Immune Count / Clustered Tumor Count AUC', \
    'imm large / imm small AUC': 'Clustered Immune Count / Unclustered Immune Count AUC', \
    'imm small / total AUC': 'Unclustered Immune Count / Total Count AUC', \
    'tum large / all tum AUC': 'Clustered Tumor Count / Tumor Count AUC', \
    'imm large / total AUC': 'Clustered Immune Count / Total Count AUC'
    }
    with open(CSV_PATH, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rad_s = row.pop('delaunay radius')
            clust_s = row.pop('small cluster size')
            # if len(rad_s) < 3:
            #     rad_s = '0' + rad_s
            rad = int(rad_s)
            clust = int(clust_s)
            if rad >= 100:
                continue

            if rad_s not in x_ticks:
                x_ticks.append(rad_s)
            if clust_s not in y_ticks:
                y_ticks.append(clust_s)

            if rad not in data:
                data[rad] = {}
            
            if clust not in data[rad]:
                data[rad][clust] = {}
            else:
                continue
            
            for value in row:
                if 'AUC' in value:
                    data[rad][clust][value] = float(row[value])
                    if value not in feature_list:
                        feature_list.append(value)
    # ipdb.set_trace()
    x_ticks.sort()
    y_ticks.sort()
    y_ticks.reverse()
    
    cmap = plt.cm.inferno
    fig, axs = plt.subplots(3, 2)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    idx = 0
    for feature in feature_list:
        row = idx % 3
        col = idx // 3
        print(str((row, col)))
        feature_arr = np.zeros((7, len(data)))
        i = 0
        for radius in sorted(data):
            j = 0
            for cluster in sorted(data[radius]):
                feature_arr[j][i] = data[radius][cluster][feature]
                j += 1
            i += 1
        feature_arr = np.flip(feature_arr, axis=0)
        
        im = axs[row, col].imshow(feature_arr, interpolation='nearest', cmap=cmap)

        axs[row, col].set_xticks(np.arange(len(data)))
        axs[row, col].set_xticklabels(x_ticks)
        axs[row, col].set_yticks(np.arange(7), y_ticks)
        axs[row, col].set_yticklabels(y_ticks)

        axs[row, col].set_title(name_dict[feature])
        axs[row, col].set(xlabel='Delaunay Pixel Radius', ylabel='Small Cluster Size')
        fig.colorbar(im, ax=axs[row, col])
        idx += 1
    fig.tight_layout()
    plt.show()
    # pdb.set_trace()

# Main body
if __name__ == '__main__':
    main()