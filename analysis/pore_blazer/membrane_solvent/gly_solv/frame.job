#!/bin/bash
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -p RM-shared
#SBATCH -o '%x.out'
#SBATCH --mail-type=NONE
#SBATCH --mail-user=nasc4134@colorado.edu

module load gcc
module load cuda/11.1.1
module load openmpi/3.1.6-gcc8.3.1
PATH=$PATH:/jet/home/schwinns/pkgs/gromacs/2020.5
source /jet/home/schwinns/pkgs/gromacs/2020.5/bin/GMXRC
source /jet/home/schwinns/.bashrc

