#!/bin/bash

cp ../0_build/initial.gro ./
cp ../0_build/topol.top ./

#used restrained
sed -i 's/ITW.itp/ITW_res_no_dih.itp/g' topol.top


#minimize
export GMX_MAXBACKUP=-1
i=0
cp -f em_scaled.mdp min$i.mdp
sed -i "s/repl_itr/$i/g" min$i.mdp
gmx grompp -f min$i.mdp -c initial.gro -p topol.top -r initial.gro -o min$i -maxwarn 5
gmx mdrun -deffnm min$i

for i in `seq 1 9`
do
    j=$((i-1))
    cp -f em_scaled.mdp min$i.mdp
    sed -i "s/repl_itr/$i/g" min$i.mdp
    gmx grompp -f min$i.mdp -c min$j.gro -p topol.top -r  initial.gro -o min$i -maxwarn 5
    gmx mdrun -v --deffnm min$i
done

gmx grompp -f sd.mdp -c min9.gro -p topol.top -r initial.gro -o sd -maxwarn 5
gmx mdrun -deffnm sd

#clean
mkdir -p intermidiates
mv -f min?.* intermidiates


