#!/bin/bash

cp ../1_minimize/sd.gro .
py bcc_solvate.py -y bcc_solvate.yaml
