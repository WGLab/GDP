# GDNNP
This is a tutorial on how to generate files needed to plot D3.js for Atif.

## Prerequisite
1. tensorflow. 
This is already installed in Phoenix. Please use following steps to access it. Note that the current version has no GPU.
```
ssh phoenix
python3
import tensorflow as tf
```
2. numpy
If you are using my Python3, then probably you don't need it. If not, please do
```
pip3 install numpy
```

## Installation of the code or feel free to use mine
Installation is as follows
```
git clone https://github.com/WGLab/GDNNP.git
```
If you prefer to use my code directly, please run your code under this directory
```
cd /home/cocodong/project/deeplearningMentor/model/
```

## Example to run the model
An example to run the model is shown in this file:
```
runDeep.py
```
Please see annotation in this file for more details

## Output
There are four files in the output. 
1. A file ends with "predict.txt"
This file contains predicted hazard (probability that a patient is going to die) for each patient (one row represents one patient) at each step.

2. A file ends with "weight.txt"
This file contains a weight matrix (w2: for the first hidden layer) and a weight vector (w1: for the output layer) at each step.

3. A file ends with "groundTruth.txt"
This file contains empirical hazard for each patient (the index of 3 and 1 are the same) at each step.

4. The remaining file ends with ".txt"
This file contains cost, accuracy (c index) and weightSize (for diagnostics) at each step.

## Goal for Atif
Create a D3.js to visualize the dynatic changes of weight matrix, weight vector, patients' hazard prediction and prediction accuracy at each step of deep learning run, at various conditions.


