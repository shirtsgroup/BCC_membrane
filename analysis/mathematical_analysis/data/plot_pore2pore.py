# Script to plot pore-to-pore distance distributions from txt data files

import numpy as np
import matplotlib.pyplot as plt

filename = 'pore2pore'

gyroid = np.loadtxt(filename + '_gyroid.txt')
schwarz = np.loadtxt(filename + '_schwarz.txt')
primitive = np.loadtxt(filename + '_primitive.txt')

fig, ax = plt.subplots(1,1, figsize=(10,8))

# Plot the KDE plots for each space group
ax.plot(gyroid[0,:], gyroid[1,:], label='Gyroid')
ax.plot(schwarz[0,:], schwarz[1,:], label='SchwarzD')
ax.plot(primitive[0,:], primitive[1,:], label='Primitive')

# Add the bilayer thickness + pore size line
ax.axvline(4.6, color='black', linestyle='dashed', label='Expected pore-to-pore distance')
ax.axvspan(4.6 - 0.2, 4.6 + 0.2, color='gray', alpha=0.5)

# Some formatting
ax.set_xlim(0,10)
ax.set_xticks(np.arange(0,11,1))
ax.set_xlabel('distance (nm)',fontsize='large')
ax.set_ylabel('probability density',fontsize='large')
ax.legend(fontsize='x-large',loc=1)
plt.show()