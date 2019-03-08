# GDP 

GDP stands for Group lasso regularized Deep learning for cancer Prognosis from multi-omics and clinical features, and it is a python package based on tensorflow to do survival analysis for the cancer survival data with high-dimensional features but smaller sample size.

## PREREQUISITES

1. Python 2.7 or Python 3.6
2. Google Tensorflow 1.4.1
3. Linux environment
4. CUDA (only for GPU version, skip it if using CPU)
5. lifelines (https://lifelines.readthedocs.io/en/latest/Quickstart.html)

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
Here is a simple example with 2 groups of features and 10 patients input file:
1,1,1,2,2,2
X1,X2,X3,X4,X5,X6,days,censors
-1.85902783053298,0.0994494352091435,-0.807210866434911,0.303156984244015,-0.275110805413045,0.579571367421607,27.7573213813725,1
0.86859158824812,-0.618129783705067,-0.308273526302284,-0.86749748596954,-1.19636100811427,-0.745102422373332,118.705992121249,1
-0.525185612462805,0.241140196288319,-1.01799171699816,0.991736132807852,-1.29180454442748,0.101100728946668,47.2236119511256,0
0.651322038444394,0.624104844298306,0.74730472763471,0.815543899943677,0.771588304339387,-0.466081795713416,195.641828371416,0
1.09407882660999,-0.0184001549027565,-0.0283117031875402,-0.27400025250167,-1.02635165546504,-1.65681317376197,8.72296970337629,1
0.797017366622877,0.647989168058519,1.96239849878692,0.65190924214759,-0.154590911151089,-1.35643031260092,103.915377632488,1
-0.827356228955125,-1.03067954961316,1.32470461704247,1.21053565312543,0.189932779563431,0.0174347878628153,23.692097151167,1
-0.431529861637166,-0.167016525491645,-0.205556633566519,-1.12039115892178,-0.502304556662159,0.410028189118801,85.330219194293,1
-0.558018708249103,0.832222076104534,0.346448789650193,0.384576276352321,0.856890192421964,-1.05987085048325,104.801233112812,1
-0.481907510681125,0.644392832802829,0.762831469086699,1.09207762024784,-0.799866311327226,0.42173466089875,172.114941469374,1
```
![Alt text](example/GDP_Input_Format_Example.png?raw=true "GDP training input example")

### first line: providing group prior knowledge
```
eg: 1,1,1,2,2,2
The first line is the group prior knowledge line with group information provided as numbers. In this example, there are 4 groups,
and the group size of the 1 group is 2, and the group size for other groups (2,3,4) is 3 for each.
```

### second line: header information for the name of features and days and censors
```
eg: X1,X2,X3,X4,X5,X6,days,censors
```

### other lines: feature matrix + survival days and censoring information
```
eg: 0.1,1,1.5,2,2,3.3,3,3,9.2,2.1,1
For censors column: if 1 then means the survival date of the patient is censored, otherwise non-censored.
```
### about the input file for making prediction
If the survival data and survival censoring status are not availabe for prediction data, then random numbers can be added to make the format consistent with the format used in the training process. And in order to load the model based on the training, the number of patients (samples) in this prediction input file should equal to the batch size used in the training process.

## Reference

Xie et al, GDP: Group lasso regularized Deep learning for cancer Prognosis from multi-omics and clinical features. Submitted, 2019
