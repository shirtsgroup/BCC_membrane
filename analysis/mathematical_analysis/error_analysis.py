# Script to run bootstrapping error on the geometric analysis distributions

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity

def histogram_KDE(data, X, bandwidth=None):
    # K-fold cross validation for bandwidth choice with sklearn
    reshaped = data.reshape(-1,1)

    if bandwidth is None:
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=KFold(n_splits=5))
        grid.fit(reshaped)
        bandwidth = grid.best_params_['bandwidth']
        print('Best bandwidth: ', bandwidth)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(reshaped)
    log_dens = kde.score_samples(X.reshape(-1,1))

    return bandwidth, np.exp(log_dens)

# Load in data
data = np.loadtxt('data/pore2pore_schwarz_raw.txt')
idx = [i for i in range(data.shape[0])]

# Generate bootstrapped data sets of the raw pore-to-pore distances
n_bootstraps = 1000
boot = np.zeros((n_bootstraps, data.shape[0]))
for b in range(n_bootstraps):
    boot_idx = np.random.choice(idx,replace=True,size=data.shape[0])
    boot[b,:] = data[boot_idx]

# Calculate KDE histogram for each bootstrapped data sets
X_plot = np.linspace(0,10,1000)
histograms = np.zeros((n_bootstraps, X_plot.shape[0]))
bandwidths = np.zeros(n_bootstraps)
for b, b_data in enumerate(boot):
    bandwidths[b], histograms[b, :] = histogram_KDE(b_data, X_plot, bandwidth=0.1)
    print('Completed {} out of {} bootstraps'.format(b+1, n_bootstraps))

print('Average best bandwidth: {:.4f}'.format(bandwidths.mean()))

# Save the average KDE and CI from bootstrapping
ci = st.norm.interval(alpha=0.95, loc=histograms.mean(axis=0), scale=st.sem(histograms))
save_data = np.vstack([X_plot, histograms.mean(axis=0), ci[0], ci[1]])
np.savetxt('data/schwarz_with_error.txt', save_data.transpose(), header='Bootstrapped average pore-to-pore distance distribution (2nd column) with 95% confidence intervals (3rd and 4th columns) -- (1st column is x axis (nm))')

plt.plot(X_plot, histograms.mean(axis=0), c='k')
plt.plot(X_plot, ci[0], ls='--', c='r')
plt.plot(X_plot, ci[1], ls='--', c='r')
plt.show()
    
