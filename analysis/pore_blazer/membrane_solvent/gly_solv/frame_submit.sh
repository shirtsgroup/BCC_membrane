#!/bin/bash

for f in $(ls *.gro)
do

    cp frame.job "${f%.*}.job"

    echo "mkdir "${f%.*}"" >> "${f%.*}.job"
    echo "mv $f "${f%.*}"/$f" >> "${f%.*}.job"
    echo "cd "${f%.*}"" >> "${f%.*}.job"
    echo "python ../gro2xyz.py -g $f -o "${f%.*}.xyz"" >> "${f%.*}.job"
    echo "cp ../defaults.dat ./" >> "${f%.*}.job"
    echo "cp ../UFF.atoms ./" >> "${f%.*}.job"
    echo "cp ../poreblazer.exe ./" >> "${f%.*}.job"
    echo "./poreblazer.exe < input.dat" >> "${f%.*}.job"

    sbatch "${f%.*}.job"

done
