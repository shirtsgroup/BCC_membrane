# Some functions to run bootstraping error on MSDF

import numpy as np


def make_bootstraps(data, n_bootstraps):

    n_frames = data.shape[0]
    n_groups = data.shape[1]
    boot = np.zeros((n_bootstraps, n_frames, n_groups)) # n_bootstraps, n_frames, n_groups
    idx = [i for i in range(n_groups)]

    for b in range(n_bootstraps): # Loop through bootstraps
        boot_idx = np.random.choice(idx, replace=True, size=n_groups)
        boot[b, :, :] = data[:, boot_idx]

    return boot


def get_bootstrapped_histograms(boot, n_bins=200):

    n_bootstraps = boot.shape[0]

    histograms = np.zeros((n_bootstraps, n_bins-1))
    bin_edges = np.zeros((n_bootstraps, n_bins))
    bins = np.linspace(-5,5, num=n_bins)

    for b, b_data in enumerate(boot):
        histograms[b,:], bin_edges[b,:] = np.histogram(b_data, bins=bins)

    return histograms, bin_edges
