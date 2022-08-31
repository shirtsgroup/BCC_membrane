# Build bicontinuous cubic structure
### Follow these steps to build bicontinous cubic structure. 

#### Step 1 : Build initial coarse-grained structure:
* Go to the folder **1_build** and source **build.sh** which runs the python command:\
`python build_initial.py -y build.yaml`

#### Step 2: Minimize the initial structure:
* Go to the folder **2_minimize** and source the script **minimize.sh**

#### Step 3: Solvate with glycerol
* Go to the folder **3_solv_gly** and source the script **solvate.sh** which runs the python command:\
`python bcc_solvate.py -y bcc_solvate.yaml`

#### Step 4: Equilibrate 
* Go to the folder **4_equilibrate** and source the script **equilibrate.sh**

#### Step 5: Production run
* Go to the folder **5_prod/T300** and submit the batch job **cpu.job**

#### Step 6: Backmap to all-atom model
* Go to the folder **6_backmap/backmapping** and submit the batch job **backmap.job**
    * Note 1: cluster must have Gromacs 2018 installed. The batch job assumes it is installable via `module load gromacs/2018`
    * Note 2: we provide a yml file with the conda environment needed to run backmapping, so you should ensure you have a working installation of Anaconda3 sourced on your machine

#### Step 7: Equilibrate the all-atom model
* Go to the folder **7_equil_AA** and submit the batch job **equil_AA.job**

#### Step 8: Crosslink
* Go to the folder **8_Xlink** and source the script **run_xlink.sh** which runs the python command:\
`python bcc_xlink.py -y bcc_xlink.yaml >> bcc_xlink.log`

#### Step 9: Solvate with water
* Go to the folder **9_solv_water** and source the script **solvate.sh** which runs the python command:\
`py bcc_solvate.py -y bcc_solvate.yaml`

### Scripts to replicate our analysis of the pores and transport within the bicontinuous cubic structure.
* [MSDF](analysis/MSDF) - Minimal surface distribution function
* [bilayers](analysis/bilayers) - Comparisons between all-atom and CG bilayers
* [experimental_pore_size](analysis/experimental_pore_size) - Parametric bootstrap to get error on experimental pore size estimation
* [gyroid_stability](analysis/gyroid_stability) - time series of glycerol displacement from the minimal surface, serving as a measure of stability over the simulation
* [mathematical_analysis](analysis/mathematical_analysis) - Mathematical argument to test how bilayers fit within different bicontinuous cubic space groups
* [pore_blazer](analysis/pore_blazer) - Pore size distribution over a trajectory performed by [PoreBlazer](https://github.com/SarkisovGitHub/PoreBlazer)
* [transport](analysis/transport) - Mean square displacement and diffusion coefficient analysis
* [visualizations](analysis/visualizations) - VMD visualization states of the gyroid structure





