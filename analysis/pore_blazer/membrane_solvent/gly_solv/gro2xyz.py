#!/usr/bin/env python

# Convert gro file to xyz format for PoreBlazer, removing solvent molecules in the process

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-g','--gro',
                    help='gro file to convert for PoreBlazer')
parser.add_argument('-o','--output',default='output.xyz',
                    help='output xyz filename')
args = parser.parse_args()

# Inputs
solvent = 'GLY' # name of solvent molecule

gro = open(args.gro,'r')

# Read the two header lines
header = gro.readline()
skip = gro.readline() # skip number of atoms and keep track below

# Read the rest of the gro file and save atoms information
atoms = {}
n_atoms = 0
for line in gro:

    if len(line.split()) == 3: # box vector at end
        box_x = float(line.split()[0])
        box_y = float(line.split()[1])
        box_z = float(line.split()[2])

    else:

        mol = line.split()[0]
        mol_name = mol.strip('0123456789')
        atom = line.split()[1]
        element = atom.strip('0123456789')
        
        if len(line.split()) == 6:
            a_num = int(line.split()[2])
            x = float(line.split()[3]) # coordinates in nm
            y = float(line.split()[4])
            z = float(line.split()[5])
        elif len(line.split()) == 5:
            x = float(line.split()[2]) # coordinates in nm
            y = float(line.split()[3])
            z = float(line.split()[4])

        if element == 'BR': # hard code bromine
            element = 'Br'
        
        if not mol_name == solvent: # save atoms that are not solvent
            n_atoms += 1
            xyz_line = '%s\t%.6f\t%.6f\t%.6f\n' %(element, x*10, y*10, z*10)
            atoms[n_atoms] = xyz_line

# Write xyz file
out = open(args.output, 'w')

out.write(str(n_atoms) + '\n') # number of atoms
out.write(' ' + header) # header

for atom in atoms: # atoms and coordinates
    out.write(atoms[atom])


# Write PoreBlazer input.dat file
inp = open('input.dat','w')

inp.write(args.output + '\n')
inp.write('%.4f\t%.4f\t%.4f\n' %(box_x*10, box_y*10, box_z*10) )
inp.write('90\t90\t90')
            
    
    


