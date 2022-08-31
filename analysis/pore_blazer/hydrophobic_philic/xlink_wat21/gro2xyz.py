#!/usr/bin/env python

# Convert gro file to xyz format for PoreBlazer, removing desired atoms in the process

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-g','--gro',
                    help='gro file to convert for PoreBlazer')
parser.add_argument('-o','--output',default='output.xyz',
                    help='output xyz filename')
args = parser.parse_args()

# Inputs
solvent = 'SOL' # name of solvent molecule
head_groups = ['C18','C19','C20','C21', 
               'C22','C23','C24','C25',
               'C26','C27','N'  ,'N1' ,
               'N2' ,'N3' ,'C46','C47',
               'H33','H34','H35','H36',
               'H37','H38','H39','H40',
               'H41','H42','H43','H44',
               'H45','H46','H47','H48',
               'H82','H83']

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
            atom_name = atom
            a_num = int(line.split()[2])
            x = float(line.split()[3]) # coordinates in nm
            y = float(line.split()[4])
            z = float(line.split()[5])
        elif len(line.split()) == 5:
            atom_name = line[12:15].split()[0] # hard code which characters correspond to atom name
            x = float(line.split()[2]) # coordinates in nm
            y = float(line.split()[3])
            z = float(line.split()[4])

        if mol_name not in [solvent, 'BR']: # atoms that are not solvent or bromine
            
            if atom_name not in head_groups: # atoms not in head group
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
            
    
    


