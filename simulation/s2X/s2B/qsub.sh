#!/bin/bash
for i in $(seq 10) 
do
	e_f=log/"$i".e
	o_f=log/"$i".o
	qsub -l h_vmem=5G -e $e_f -o $o_f -cwd -V -N run_simu$i run_script.sh $i 
done
