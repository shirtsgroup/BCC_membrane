#!/bin/bash

cp ../7_equil_AA/prod.gro ./


python bcc_xlink.py -y bcc_xlink.yaml >> bcc_xlink.log
