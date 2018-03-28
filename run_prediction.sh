#!/bin/bash
#notice:
#make sure:
#1. the parameter settings here is consistent with the training process
#2. the number of patients (or samples) in prediction_file equals to the batch size used in the training process (here both is 50)
#3. the format of prediction_file is the same as the format of the input file for training process, if columns "days" and "censors" in the prediction file is unknown, random number can be used here to make the format consistent
python gdp_prediction.py --prediction_file data2.csv --model "NN" --hidden_nodes "200,100" --prediction_dir "example" --saver_file_prefix "gdp.model-1999"
