# Script to save MSDFs from csv files with heavy atom distances

import mdtraj as md
import numpy as np
import os
import gromacs as gro
import matplotlib.pyplot as plt
from MSDF_error import make_bootstraps, get_bootstrapped_histograms

# Some input information
t = md.load('solvated_water.gro')

n_frames = 1001
n_bins = 200
n_bootstraps = 1000

# Get information on names and numbers from gro file
gly_atoms = []
mol_atoms = []
ion_atoms = []

need_gly = True
need_mol = True
need_ion = True

for res in t.top.residues:
    if res.name == 'GLY':
        if need_gly:
            need_gly = False
            for atom in res.atoms:
                if not atom.element.name == 'hydrogen':
                    gly_atoms.append(atom.name)

    elif res.name == 'MOL': 
        if need_mol:
            need_mol = False
            first_atom = None
            for atom in res.atoms:
                if not atom.element.name == 'hydrogen':
                    if not first_atom:
                        first_atom = atom.name
                    elif atom.name == first_atom:
                        break

                    mol_atoms.append(atom.name)

    elif res.name == 'BR':

        if need_ion:
            need_ion = False
            for atom in res.atoms:
                if not atom.element.name == 'hydrogen':
                    ion_atoms.append(atom.name)

print('Finished reading system info...')
# Save ions with all heavy atoms
selection = ['BR']
cum_data = np.zeros((n_bins-1, 3))
for f in os.listdir('./'):
    if f.startswith('ION'):
        file_number = int(f.split('.')[0].split('heavy')[1])
        atom_name = ion_atoms[file_number]
        if atom_name in selection:
            data = np.loadtxt(f, delimiter=',')
            boot = make_bootstraps(data, n_bootstraps=1000)
            hist, bin_edges = get_bootstrapped_histograms(boot, n_bins=n_bins)
            cum_data[:,0] = bin_edges.mean(axis=0)[:-1]
            cum_data[:,1] += hist.mean(axis=0) / n_frames
            cum_data[:,2] +=  (hist.std(axis=0)  / n_frames)**2

cum_data[:,2] = np.sqrt(cum_data[:,2])
xvg = gro.fileformats.XVG(array=cum_data.transpose(), names=['displacement bins (nm)', 'number density (counts)', 'standard deviation (counts)'])
xvg.write('ions_density.xvg')

plt.plot(cum_data[:,0], cum_data[:,1], c='blue', label='ions')
plt.fill_between(cum_data[:,0], cum_data[:,1] - cum_data[:,2], cum_data[:,1] + cum_data[:,2], color='blue', alpha=0.5)

print('Finished saving ions...')

# Plot water from previous calculations
xvg = gro.fileformats.XVG('deviation_water_density_cols.xvg')
data = xvg.array
plt.plot(data[0,:], data[1,:] / n_frames, c='green', label='water')

# # Save glycerol with all heavy atoms
# selection = ['O','C','C1','C2','O','OXT']
# cum_data = np.zeros((n_bins-1, 3))
# for f in os.listdir('./'):
#     if f.startswith('GLY'):
#         file_number = int(f.split('.')[0].split('heavy')[1])
#         atom_name = gly_atoms[file_number]
#         if atom_name in selection:
#             data = np.loadtxt(f, delimiter=',')
#             boot = make_bootstraps(data, n_bootstraps=1000)
#             hist, bin_edges = get_bootstrapped_histograms(boot, n_bins=n_bins)
#             cum_data[:,0] = bin_edges.mean(axis=0)[:-1]
#             cum_data[:,1] += hist.mean(axis=0) / n_frames
#             cum_data[:,2] +=  (hist.std(axis=0)  / n_frames)**2

# cum_data[:,2] = np.sqrt(cum_data[:,2])
# xvg = gro.fileformats.XVG(array=cum_data.transpose(), names=['displacement bins (nm)', 'number density (counts)', 'standard deviation (counts)'])
# xvg.write('glycerol_density.xvg')

# plt.plot(cum_data[:,0], cum_data[:,1], c='green', label='glycerol')
# plt.fill_between(cum_data[:,0], cum_data[:,1] - cum_data[:,2], cum_data[:,1] + cum_data[:,2], color='green', alpha=0.5)

# print('Finished saving glycerol...')

# Save inside head groups with all heavy atoms
selection = ['N','C18','C19','N1','C47','N2','C26','C27','N3','C46','C20','C21','C22','C23','C24','C25']
cum_data = np.zeros((n_bins-1, 3))
for f in os.listdir('./'):
    if f.startswith('MOL_in'):
        file_number = int(f.split('.')[0].split('heavy')[1])
        atom_name = mol_atoms[file_number]
        if atom_name in selection:
            data = np.loadtxt(f, delimiter=',')
            boot = make_bootstraps(data, n_bootstraps=1000)
            hist, bin_edges = get_bootstrapped_histograms(boot, n_bins=n_bins)
            cum_data[:,0] = bin_edges.mean(axis=0)[:-1]
            cum_data[:,1] += hist.mean(axis=0) / n_frames
            cum_data[:,2] +=  (hist.std(axis=0)  / n_frames)**2

cum_data[:,2] = np.sqrt(cum_data[:,2])
xvg = gro.fileformats.XVG(array=cum_data.transpose(), names=['displacement bins (nm)', 'number density (counts)', 'standard deviation (counts)'])
xvg.write('heads_in_density.xvg')

plt.plot(cum_data[:,0], cum_data[:,1], c='red', label='heads')
plt.fill_between(cum_data[:,0], cum_data[:,1] - cum_data[:,2], cum_data[:,1] + cum_data[:,2], color='red', alpha=0.5)

print('Finished saving inside head groups...')

# Save outside head groups with all heavy atoms
selection = ['N','C18','C19','N1','C47','N2','C26','C27','N3','C46','C20','C21','C22','C23','C24','C25']
cum_data = np.zeros((n_bins-1, 3))
for f in os.listdir('./'):
    if f.startswith('MOL_out'):
        file_number = int(f.split('.')[0].split('heavy')[1])
        atom_name = mol_atoms[file_number]
        if atom_name in selection:
            data = np.loadtxt(f, delimiter=',')
            boot = make_bootstraps(data, n_bootstraps=1000)
            hist, bin_edges = get_bootstrapped_histograms(boot, n_bins=n_bins)
            cum_data[:,0] = bin_edges.mean(axis=0)[:-1]
            cum_data[:,1] += hist.mean(axis=0) / n_frames
            cum_data[:,2] +=  (hist.std(axis=0)  / n_frames)**2

cum_data[:,2] = np.sqrt(cum_data[:,2])
xvg = gro.fileformats.XVG(array=cum_data.transpose(), names=['displacement bins (nm)', 'number density (counts)', 'standard deviation (counts)'])
xvg.write('heads_out_density.xvg')

plt.plot(cum_data[:,0], cum_data[:,1], c='red')
plt.fill_between(cum_data[:,0], cum_data[:,1] - cum_data[:,2], cum_data[:,1] + cum_data[:,2], color='red', alpha=0.5)

print('Finished saving outside head groups...')

# Save tail groups with all heavy atoms
selection = ['C','C1','C2','C3','C4','C5','C6','C7','C8',
             'C9','C10','C11','C12','C13','C14','C15','C16',
             'C17','C28','C29','C30','C31','C32','C33','C34',
             'C35','C36','C37','C38','C39','C40','C41','C42',
             'C43','C44','C45']
cum_data = np.zeros((n_bins-1, 3))
for f in os.listdir('./'):
    if f.startswith('MOL'):
        file_number = int(f.split('.')[0].split('heavy')[1])
        atom_name = mol_atoms[file_number]
        if atom_name in selection:
            data = np.loadtxt(f, delimiter=',')
            boot = make_bootstraps(data, n_bootstraps=1000)
            hist, bin_edges = get_bootstrapped_histograms(boot, n_bins=n_bins)
            cum_data[:,0] = bin_edges.mean(axis=0)[:-1]
            cum_data[:,1] += hist.mean(axis=0) / n_frames
            cum_data[:,2] +=  (hist.std(axis=0)  / n_frames)**2

cum_data[:,2] = np.sqrt(cum_data[:,2])
xvg = gro.fileformats.XVG(array=cum_data.transpose(), names=['displacement bins (nm)', 'number density (counts)', 'standard deviation (counts)'])
xvg.write('tails_density.xvg')

plt.plot(cum_data[:,0], cum_data[:,1], c='violet', label='tails')
plt.fill_between(cum_data[:,0], cum_data[:,1] - cum_data[:,2], cum_data[:,1] + cum_data[:,2], color='violet', alpha=0.5)

print('Finished saving tail groups...')

plt.legend()
plt.show()
