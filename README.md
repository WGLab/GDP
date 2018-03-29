# GDP 

GDP stands for Group lasso regularized Deep learning for cancer Prognosis from multi-omics and clinical features, and it is a python package based on tensorflow to do survival analysis for the cancer survival data with high-dimensional features but smaller sample size.

## PREREQUISITES

1. Python 2.7 or Python 3.6
2. Tensorflow 1.4.1
3. linux environment
4. CUDA (only for GPU version, skip it if using CPU)

## Quick Start
### Train GDP model

```
$bash run_train.sh
or under SGE high-performance cluster
$bash qsub_run_train.sh
```
### Make prediction
```
$bash run_prediction.sh
or under SGE high-performance cluster
$bash qsub_run_prediction.sh
```

## USAGE 

### Training GDP model
```
usage: train_model.py [-h] [--initial_learning_rate INITIAL_LEARNING_RATE]
                      [--alpha ALPHA] [--scale SCALE] [--delta DELTA]
                      [--reg_type REG_TYPE] [--activation ACTIVATION]
                      [--max_steps MAX_STEPS] [--hidden_nodes HIDDEN_NODES]
                      [--batch_size BATCH_SIZE] [--eval_steps EVAL_STEPS]
                      [--dropout_keep_rate DROPOUT_KEEP_RATE]
                      [--skip_eval SKIP_EVAL] [--model MODEL]
                      [--log_dir LOG_DIR] [--train_dir TRAIN_DIR]
                      [--train_file TRAIN_FILE]
                      [--evaluation_file EVALUATION_FILE]
                      [--evaluation_cindex_file EVALUATION_CINDEX_FILE]
                      [--output_evaluation_cindex OUTPUT_EVALUATION_CINDEX]
                      [--evaluation_dir EVALUATION_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial learning rate.
  --alpha ALPHA         alpha for group regularizer proportion control
  --scale SCALE         Lambda scale for regularization
  --delta DELTA         delta value used for log transformation in partial
                        likelihood calculation
  --reg_type REG_TYPE   types of regularization (available: lasso, l2,
                        group_lasso, sparse_group_lasso,none)
  --activation ACTIVATION
                        activation function (relu, sigmoid or tanh)
  --max_steps MAX_STEPS
                        Number of steps to run trainer.
  --hidden_nodes HIDDEN_NODES
                        Number of nodes in each hidden layer.
  --batch_size BATCH_SIZE
                        Batch size. Must divide evenly into the dataset sizes.
  --eval_steps EVAL_STEPS
                        number of steps used for evaluation average score
                        calculation
  --dropout_keep_rate DROPOUT_KEEP_RATE
                        dropout keep rate(default:0.6)
  --skip_eval SKIP_EVAL
                        YES or NO for if skipping evaluation
  --model MODEL         model used for the training, NN: neural network,
                        linear: linear regression (alpha should be set to 1
                        for group lasso)
  --log_dir LOG_DIR     Directory to put the log data.
  --train_dir TRAIN_DIR
                        Directory for the input file to do model
                        training,validation and testing
  --train_file TRAIN_FILE
                        input file for model training , validation and testing
  --evaluation_file EVALUATION_FILE
                        evaluation scores output file(cost, cindex changes in
                        each epoch after one batch of data feeding)
  --evaluation_cindex_file EVALUATION_CINDEX_FILE
                        evaluation cindex output file(cindex changes in each
                        training)
  --output_evaluation_cindex OUTPUT_EVALUATION_CINDEX
                        whether or not to output evaluation cindex detail
                        information, default(NO), other choice YES
  --evaluation_dir EVALUATION_DIR
                        evaluation scores output directory(cost, cindex
                        changes in each epoch after one batch of data feeding)

```
### Making survival hazard prediction
```
usage: gdp_prediction.py [-h] [--reg_type REG_TYPE] [--activation ACTIVATION]
                         [--hidden_nodes HIDDEN_NODES]
                         [--saver_file_dir SAVER_FILE_DIR]
                         [--saver_file_prefix SAVER_FILE_PREFIX]
                         [--model MODEL] [--prediction_dir PREDICTION_DIR]
                         [--prediction_file PREDICTION_FILE]
                         [--out_dir OUT_DIR] [--out_file OUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --reg_type REG_TYPE   types of regularization (available: lasso, l2,
                        group_lasso, sparse_group_lasso,none)
  --activation ACTIVATION
                        activation function (relu, sigmoid or tanh)
  --hidden_nodes HIDDEN_NODES
                        Number of nodes in each hidden layer.
  --saver_file_dir SAVER_FILE_DIR
                        Directory to put the files with saved variables.
  --saver_file_prefix SAVER_FILE_PREFIX
                        prefix of the saver files (saver files used for
                        reloading the model for prediction)
  --model MODEL         model used for the training, NN: neural network,
                        linear: linear regression (alpha should be set to 1
                        for group lasso)
  --prediction_dir PREDICTION_DIR
                        Directory for the input file to do model
                        training,validation and testing
  --prediction_file PREDICTION_FILE
                        input file for making prediction,the sample size in
                        this file should equal to the batch size in the
                        training process
  --out_dir OUT_DIR     output file directory
  --out_file OUT_FILE   output file name

```

## About Input Data

### simple example
```
Here is the first four lines of an example input file:
1,1,2,2,2,3,3,3,4,4,4 
f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,days,censors
0.1,1,1.5,2,2,3.3,3,3,9.2,2.1,1
2.3,1,2.4,2,2,4.2,3,7.8,4,1.3,0
```

### first line: providing group prior knowledge
```
eg: 1,1,2,2,2,3,3,3,4,4,4
The first line is the group prior knowledge line with group information provided as numbers. In this example, there are 4 groups,
and the group size of the 1 group is 2, and the group size for other groups (2,3,4) is 3 for each.
```

### second line: header information for the name of features and days and censors
```
eg: f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,days,censors
```

### other lines: feature matrix + survival days and censoring information
```
eg: 0.1,1,1.5,2,2,3.3,3,3,9.2,2.1,1
For censors column: if 1 then means the survival date of the patient is censored, otherwise non-censored.
```
### about the input file for making prediction
If the survival data and survival censoring status are not availabe for prediction data, then random numbers can be added to make the format consistent with the format used in the training process. And in order to load the model based on the training, the number of patients (samples) in this prediction input file should equal to the batch size used in the training process.


