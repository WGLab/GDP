#!/bin/bash
qsub -l h_vmem=5G -e log/1.e -o log/1.o -cwd -V -N get_input_file run_script.sh 
