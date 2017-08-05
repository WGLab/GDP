from deepLassoPhen import lasso
fileName = "/home/cocodong/project/deeplearningMentor/processData/Brain_allData.txt"  ### use this file to model 

n_hidden = 100 ### number of nodes in each hidden layer, here for simplicity, we only used 1 hidden layer
num_steps = 100 ### number of steps to run the stochastic neural net

penaltyLambdaArray = [0] ### penalty for all regularization terms, including group penalty and L1 penalty
alphaArray = [0.01] ### balancing factor between L2 and L1 penalty, the smaller the alpha, the more sparse the weighting matrix it is
prefix = "/home/cocodong/project/deeplearningMentor/processData/test" ### prefix for the output
nonLinear = "tanh"  ### activation function for the hidden layer

lasso(fileName, num_steps, n_hidden, penaltyLambdaArray, alphaArray, prefix, nonLinear)

