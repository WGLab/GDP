#!/bin/bash
qsub -l h_vmem=20G -e example/log/run_example.e -o example/log/run_example.o -cwd -V -N run_train run_train.sh
