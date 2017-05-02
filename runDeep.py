from deepGroupLassoPhen import lasso
fileName = "/home/cocodong/project/deeplearningMentor/processData/all.cgc.1000.txt"  ### use this file to model 

n_hidden = 50 ### number of nodes in each hidden layer, here for simplicity, we only used 1 hidden layer
num_steps = 50 ### number of steps to run the stochastic neural net

penaltyLambdaArray = [0.0001] ### step size for gradient descent
alphaArray = [0.5] ### balancing factor between L1 and L2 penalty
prefix = "/home/cocodong/project/deeplearningMentor/processData/test" ### prefix for the output
nonLinear = "sigmoid"  ### activation function for the hidden layer

lasso(fileName, num_steps, n_hidden, penaltyLambdaArray, alphaArray, prefix, nonLinear)

