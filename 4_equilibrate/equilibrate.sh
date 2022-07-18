#!/bin/bash

#source your gromacs GMXRC file
source /usr/local/gromacs/2020.3_gpu/bin/GMXRC

export GMX_MAXBACKUP=-1
export GMX_MAXCONSTRWARN=-1

cp ../2_solvate/solvated_final.gro ./

# res 500
gmx_mpi grompp -f em.mdp -p topol_500.top -c solvated_final.gro -r solvated_final.gro  -o em_500 
gmx_mpi  mdrun -v -deffnm em_500

gmx_mpi grompp -f nvt.mdp -p topol_500.top -c em_500.gro -r em_500.gro -o nvt_500 
gmx_mpi  mdrun -v -deffnm nvt_500

# res 250
gmx_mpi grompp -f em.mdp -p topol_250.top -c nvt_500.gro -r nvt_500.gro  -o em_250 
gmx_mpi  mdrun -v -deffnm em_250

gmx_mpi grompp -f nvt.mdp -p topol_250.top -c em_250.gro -r em_250.gro -o nvt_250 
gmx_mpi  mdrun -v -deffnm nvt_250

# res 100
gmx_mpi grompp -f em.mdp -p topol_100.top -c nvt_250.gro -r nvt_250.gro  -o em_100 
gmx_mpi  mdrun -v -deffnm em_100
    
gmx_mpi grompp -f nvt.mdp -p topol_100.top -c em_100.gro -r em_100.gro -o nvt_100 
gmx_mpi  mdrun -v -deffnm nvt_100
    
# res 50
gmx_mpi grompp -f em.mdp -p topol_50.top -c nvt_100.gro -r nvt_100.gro  -o em_50 
gmx_mpi  mdrun -v -deffnm em_50

gmx_mpi grompp -f nvt.mdp -p topol_50.top -c em_50.gro -r em_50.gro -o nvt_50 
gmx_mpi  mdrun -v -deffnm nvt_50
    
# res 10
gmx_mpi grompp -f em.mdp -p topol_10.top -c nvt_50.gro -r nvt_50.gro  -o em_10 
gmx_mpi  mdrun -v -deffnm em_10

gmx_mpi grompp -f nvt.mdp -p topol_10.top -c em_10.gro -r em_10.gro -o nvt_10 
gmx_mpi  mdrun -v -deffnm nvt_10

# res 0
gmx_mpi grompp -f em.mdp -p topol_0.top -c nvt_10.gro -r nvt_10.gro  -o em_0 
gmx_mpi  mdrun -v -deffnm em_0

gmx_mpi grompp -f nvt.mdp -p topol_0.top -c em_0.gro -r em_0.gro -o nvt_0 
gmx_mpi  mdrun -v -deffnm nvt_0


# 10 ns long nvt
gmx_mpi grompp -f nvt_long.mdp -p topol_0.top -c nvt_0.gro -r nvt_0.gro -o nvt_long
gmx_mpi  mdrun -v -deffnm nvt_long

# 10 ns long npt
gmx_mpi grompp -f npt_long.mdp -p topol_0.top -c nvt_long.gro -r nvt_long.gro  -o npt_long
gmx_mpi  mdrun -v -deffnm npt_long
