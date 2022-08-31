#!/bin/bash

for f in $(ls *.gro)
do

    cp frame.job "${f%.*}.job" # Copy frame.job to frameXXX.job

    # Add lines to frameXXX.job
    echo "mkdir "${f%.*}"" >> "${f%.*}.job"                                 # mkdir frameXXX
    echo "mv $f "${f%.*}"/$f" >> "${f%.*}.job"                              # mv frameXXX.gro frameXXX/frameXXX.gro
    echo "cd "${f%.*}"" >> "${f%.*}.job"                                    # cd frameXXX
    echo "python ../gro2xyz.py -g $f -o "${f%.*}.xyz"" >> "${f%.*}.job"     # python ../gro2xyz.py -g frameXXX.gro -o frameXXX.xyz
    echo "cp ../defaults.dat ./" >> "${f%.*}.job"                           # cp ../defaults.dat ./
    echo "cp ../UFF.atoms ./" >> "${f%.*}.job"                              # cp ../UFF.atoms ./
    echo "cp ../poreblazer.exe ./" >> "${f%.*}.job"                         # cp ../poreblazer.exe
    echo "./poreblazer.exe < input.dat" >> "${f%.*}.job"                    # ./poreblazer.exe < input.dat

    sbatch "${f%.*}.job" # Submit job to queue

done
