# Script to perform a parametric bootstrap on the Ferry model fit

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Ferry_model(r_solute, r_pore):
    R = (1 - (1 - r_solute/r_pore)**2 )**2 * 100
    return R

data = np.loadtxt('rejection_data.csv', delimiter=',', skiprows=1, usecols=(1,2,3))
r_solute = data[:,0] / 2

# Generate n_bootstraps bootstrapped data sets
n_bootstraps = 1000
boot = np.zeros((n_bootstraps, data.shape[0]))
for b in range(n_bootstraps):

    for s,solute in enumerate(data): # get 4 data points for each bootstrap
        solute_distribution = np.random.normal(loc=solute[1], scale=solute[2]) # sample from a Normal distribution for each sample
        boot[b, s] = solute_distribution

# Fit each bootstrapped data set to the Ferry Equation
r_pore = np.zeros(n_bootstraps)
for b,b_data in enumerate(boot):

    r_pore[b], covariance = curve_fit(Ferry_model, r_solute, b_data)

r_diameter = r_pore*2
print('\nMean pore diameter from {} bootstraps: {:.4f} nm'.format(n_bootstraps, r_diameter.mean()))
print('Standard deviation of pore diameter from {} bootstraps: {:.4f} nm'.format(n_bootstraps, r_diameter.std()))
print('95% confidence interval for pore diameter from {} bootstraps: ({:.4f} nm - {:.4f} nm)\n'.format(n_bootstraps, np.percentile(r_diameter, 2.5), np.percentile(r_diameter, 97.5)))

plt.hist(r_diameter, bins=30, color='lightskyblue', alpha=0.75)
plt.axvline(r_diameter.mean(), ls='--', c='k', label='average')
plt.xlabel('Pore diameter (nm)')
plt.ylabel('Frequency')
plt.legend()
plt.show()