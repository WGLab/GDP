#!/bin/bash
head -452 input.csv > data1.csv
head -2 input.csv > data2.csv
tail -50 input.csv >> data2.csv
