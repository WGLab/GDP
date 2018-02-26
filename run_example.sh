#!/bin/bash
python train_model.py --train_file input.csv --reg_type "group_lasso" --alpha 0.9 --batch_size 50 --max_steps 5000 --model "NN" --evaluation_file "eval_output.tsv" --hidden_nodes "200,100" --scale 0.5 --evaluation_dir "example/output" --dropout_keep_rate 0.8 --train_dir "example"
