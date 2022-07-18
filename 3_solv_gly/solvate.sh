#!/bin/bash

cp ../1_minimize/sd.gro .
python bcc_solvate.py -y bcc_solvate.yaml
