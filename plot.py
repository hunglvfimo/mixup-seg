import os
import argparse
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

parser     = argparse.ArgumentParser()
parser.add_argument('--csv_history', default="history.csv", help='Path to file containing training log')
parser.add_argument('--title', help='Plot title', type=str, default="")
parser = parser.parse_args()

def plotloss(csvfile):
    '''
    Args
        csvfile: name of the csv file
    Returns
        graph_loss: trend of loss values over epoch
    '''

    # Bring in the csv file
    history     = pd.read_csv(csvfile)

    # Initiation
    epoch       = np.asarray(history.iloc[:, 0])
    tr_acc      = np.asarray(history.iloc[:, 2])
    tr_loss     = np.asarray(history.iloc[:, 1])
    val_acc     = np.asarray(history.iloc[:, 4])
    val_loss    = np.asarray(history.iloc[:, 3])

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    # Label and color the axes
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16, color='black')
    ax2.set_ylabel('Accuracy', fontsize=16, color='black')

    # Plot valid/train losses
    ax1.plot(epoch, tr_loss, linewidth=2,
             ls='--', color='#c92508', label='Train loss')
    ax1.plot(epoch, val_loss, linewidth=2,
             ls='--', color='#2348ff', label='Val loss')
    ax1.spines['left'].set_color('#f23d1d')
    # Coloring the ticks
    for label in ax1.get_yticklabels():
        label.set_color('#c92508')
        label.set_size(12)

    # Plot valid/trian accuracy
    ax2.plot(epoch, tr_acc, linewidth=2,
             color='#c92508', label='Train Acc')
    ax2.plot(epoch, val_acc, linewidth=2,
             color='#2348ff', label='Val Acc')
    ax2.spines['right'].set_color('#2348ff')
    # Coloring the ticks
    for label in ax2.get_yticklabels():
        label.set_color('#2348ff')
        label.set_size(12)

    # Manually setting the y-axis ticks
    # yticks = np.arange(0, 1.1, 0.1)
    # ax1.set_yticks(yticks)
    # ax2.set_yticks(yticks)

    for label in ax1.get_xticklabels():
        label.set_size(12)

    # Modification of the overall graph
    fig.legend(ncol=4, loc=9, fontsize=12)
    plt.xlim(xmin=0)
    # ax2.set_ylim(ymax=1, ymin=0)
    # ax1.set_ylim(ymax=1, ymin=0)
    plt.title(parser.title, weight="bold")
    plt.grid(True, axis='y')

if __name__ == '__main__':
    plt.show(plotloss(parser.csv_history))
