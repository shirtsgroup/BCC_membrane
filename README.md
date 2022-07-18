# Build bicontinuous cubic structure
### Follow these steps to build bicontinous cubic structure. 

#### Step 1 : Build initial coarse-grained structure:
* Goto the folder **1_build** and source **build.sh** which runs the python command:\
`python build_initial.py -y build.yaml`

#### Step 2: Minimize the initial structure:
* Goto the folder **2_minimize** and source the script **minimize.sh**

#### Step 3: Solvate with glycerol
* Goto the folder **3_solv_gly** and source the script **solvate.sh** which runs python command:\
`python bcc_solvate.py -y bcc_solvate.yaml`

#### Step 4: Equilibrate 
* Goto the folder **4_equilibrate** and source the script **equilibrate.sh**

#### Step 5: Production run
* Goto the folder **5_prod/T300** and submit the bactch job **cpu.job**

#### Step 6: Backmap to all-atom model
* Goto the folder **6_backmap**

#### Step 7: Crosslink
* Goto the folder **7_Xlink** and source the script **run_xlink.sh** which run python command:\
`python bcc_xlink.py -y bcc_xlink.yaml >> bcc_xlink.log`

#### Step 8: Solvate with water
* Goto the folder **8_solv_water** and source the script **solvate.sh** which run python command:\
`py bcc_solvate.py -y bcc_solvate.yaml`






