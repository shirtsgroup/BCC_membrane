#!/usr/bin/env python

# Gathers and plots the pore size distribution over all the frames 
# --> current directory should have all frame??? directories

import argparse
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from os import listdir
from os.path import isdir, exists
from gromacs.formats import XVG

parser = argparse.ArgumentParser()
parser.add_argument('-o','--output',default='output.png',
                    help='output png filename')
args = parser.parse_args()

# Inputs
plot_every = False

# Get list of frames from the frame directories
frames = []
for f in listdir('./'):
    if isdir(f) and f.startswith('frame') and exists(f + '/Total_psd.txt'):
        frames.append(f)
tot_frames = len(frames)

# Compile all the data
tmp = np.loadtxt(frames[0] + '/Total_psd.txt')
PSD = np.zeros((tmp.shape[0], tot_frames))
d = np.zeros((tmp.shape[0], tot_frames))

n_frames = 0
for f in frames:
    data = np.loadtxt(f + '/Total_psd.txt')
    d[:, n_frames] = data[:,0]
    PSD[:, n_frames] = data[:,1]
    n_frames += 1
    if plot_every:
        plt.plot(data[:,0]/10, data[:,1], alpha=0.3, c='gray')

# Average all the data
avg_d = d.mean(axis=1) # average distance
avg_PSD = PSD.mean(axis=1) # average PSD
se_PSD = sem(PSD, axis=1) # standard error PSD
std_PSD = PSD.std(axis=1) # standard deviation PSD

mytitle = 'Pore size distribution averaged over %d frames with average SE %.4f' %(n_frames, se_PSD.mean())
print(mytitle)

# Plot the data
plt.errorbar(avg_d/10, avg_PSD, yerr=se_PSD, lw=1, label='PSD w/standard error')
plt.errorbar(avg_d/10, avg_PSD + 1, yerr=std_PSD, lw=1, label='PSD w/standard deviation')
plt.xlabel('distance (nm)')
plt.ylabel('PSD')
plt.legend()
plt.show()

data = np.vstack((avg_d, avg_PSD, se_PSD, std_PSD))
xvg = XVG(array=data, names=['pore size (Ang)', 'pore size distribution', 'standard error', 'standard deviation'])
xvg.write('trajectory_PSD.xvg')
