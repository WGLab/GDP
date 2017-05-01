from deepGroupLassoPhen import lasso
fileName = "/home/cocodong/project/deeplearningMentor/processData/all.cgc.1000.txt"

n_hidden = 50
num_steps = 50

penaltyLambdaArray = [0.0001]
alphaArray = [0.5]
prefix = "/home/cocodong/project/deeplearningMentor/processData/test"
nonLinear = "sigmoid"

lasso(fileName, num_steps, n_hidden, penaltyLambdaArray, alphaArray, prefix, nonLinear)

