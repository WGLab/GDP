from SurvivalAnalysis import SurvivalAnalysis
from lifelines.utils import _naive_concordance_index
from scipy import stats
import numpy as np
import random
import scipy.io as sio
import os
import tensorflow as tf


np.set_printoptions(threshold=np.inf)

def lasso(fileName, num_steps, n_hidden, penaltyLambdaArray, alphaArray, prefix, nonLinear):
    
    """
        normal lasso cox regression implementation
        
        """
    
    random.seed(100)
    
    #### tensorflow implementation
    def cumsum(x, observations):
        x = tf.reshape(x, (1, observations))
        values = tf.split(x, observations, axis = 1)
        out = []
        prev = tf.zeros_like(values[0])
        for val in values:
            s = prev + val
            out.append(s)
            prev = s
        cumsum = tf.concat(out, axis = 1)
        cumsum = tf.reshape(cumsum, (observations, 1))
        return cumsum
    
    
    #### load data
    data = np.genfromtxt(fileName, dtype=float, missing_values="None", delimiter="\t")
    print("finished loading for the dataset")

    T = data[:, 1]
    np.place(T , T == 0, 1)
    O = data[:, 0]
    X = data[:, 2:]
    
    train_set = {}

    #caclulate the risk group for every patient i: patients who die after i
    sa = SurvivalAnalysis()

    train_set['X'], train_set['T'], train_set['O'], train_set['A'] = sa.calc_at_risk(X, T, O)



    print("finished calculating risks")

    ## initialization
    n_obs =  train_set['X'].shape[0] # 302
    n_in = train_set['X'].shape[1] # 201
        
    n_out = 1
    
    
    ###### normal lasso #######
    
    with tf.device('/cpu:0'):
        ## dropout
        keep_prob = tf.placeholder(tf.float32)
        
        ## penaltyLambda
        penaltyLambda = tf.placeholder(tf.float32)
        
        ## alpha
        alpha = tf.placeholder(tf.float32)
        
        ## data
        input = tf.placeholder(tf.float32, [n_obs, n_in])
        at_risk = tf.placeholder(tf.int32, [n_obs, ])
        observed = tf.placeholder(tf.float32, [n_obs, ])
                
            # weight
        w_6 = tf.Variable(tf.truncated_normal([n_in, n_hidden], dtype=tf.float32)/10)
        w_1 = tf.Variable(tf.truncated_normal([n_hidden, n_out], dtype=tf.float32)/10)
            
            ## output layer


        if (nonLinear == "tanh") :
            output_1 = tf.tanh(tf.matmul(input, w_6))
                
        elif (nonLinear == "sigmoid") :
            output_1 = tf.sigmoid(tf.matmul(input, w_6))
                
        elif (nonLinear == "linear") :
            output_1 = tf.matmul(input, w_6)

        elif (nonLinear == "relu"):
            output_1 = tf.nn.relu(tf.matmul(input, w_6))
            
        output = tf.matmul(output_1, w_1)

            
        ## optimization
        exp = tf.reverse(tf.exp(output), axis = [-2])
        partial_sum_a = cumsum(exp, n_obs)
        partial_sum = tf.reverse(partial_sum_a, axis = [-2]) + 1
        log_at_risk = tf.log(tf.gather(partial_sum, tf.reshape(at_risk, [-1])) + 1e-50)
        diff = tf.subtract(output, log_at_risk)
        times = tf.reshape(diff, [-1]) * observed
            
        # group lasso penalty
#        cost = - (tf.reduce_sum(times)) + alpha * tf.reduce_sum(penaltyLambda * tf.abs(w_6)) + alpha * tf.reduce_sum(penaltyLambda * tf.abs(w_1)) + alpha * tf.reduce_sum(penaltyLambda * tf.nn.l2_loss(w_6)) + alpha * tf.reduce_sum(penaltyLambda * tf.nn.l2_loss(w_1))
        cost =  -tf.reduce_sum(times)

        weightSize = tf.nn.l2_loss(w_6)
            
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.0001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.989, staircase=True)
        
        # optimizer
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        optimizer = tf.train.FtrlOptimizer(0.05).minimize(cost)


    print("finished loading graphs")

    for alphaArrayIndex in range(len(alphaArray)):
        print("alpha: " + str(alphaArray[alphaArrayIndex]))
        
        for penaltyLambdaIndex in range(len(penaltyLambdaArray)):
            print("lambda: " + str(penaltyLambdaArray[penaltyLambdaIndex]))
            

            gtFile = prefix + ".lambda." + str(penaltyLambdaArray[penaltyLambdaIndex]) + ".alpha." + str(alphaArray[alphaArrayIndex]) + ".groundTruth.txt"
            gtFH = open(gtFile, "w")
            gtFH.write(str(train_set['A']))
            

            targetFile = prefix + ".lambda." + str(penaltyLambdaArray[penaltyLambdaIndex]) + ".alpha." + str(alphaArray[alphaArrayIndex]) + ".txt"
            targetFH = open(targetFile, "w")
            targetFH.write("iteration" + "\t" + "trainCost"  + "\t"  + "\t" + "trainCIndex" + "\t"  + "weightSize" + "\t" + "\n")
            
            weightFile = prefix + ".lambda." + str(penaltyLambdaArray[penaltyLambdaIndex]) + ".alpha." + str(alphaArray[alphaArrayIndex]) + ".weight.txt"
            weightFH = open(weightFile, "w")
            
            probablityFile = prefix + ".lambda." + str(penaltyLambdaArray[penaltyLambdaIndex]) + ".alpha." + str(alphaArray[alphaArrayIndex]) + ".predict.txt"
            probablityFH = open(probablityFile, "w")
            
            
            config = tf.ConfigProto(allow_soft_placement=True)
            
            with tf.Session(config=config) as session:
                

                tf.global_variables_initializer().run()
                    

                for step in range(num_steps):
                    
                    
                    feed_dict = {input : train_set['X'], at_risk : train_set['A'], observed : train_set['O'], keep_prob : 1, penaltyLambda : penaltyLambdaArray[penaltyLambdaIndex], alpha : alphaArray[alphaArrayIndex]}
                        
                    _, partial_sumV, diffV, log_at_riskV, timesV, expV, outputV, costV, w1V, wV, weightSizeV= session.run([optimizer, partial_sum, diff, log_at_risk, times, exp, output, cost, w_1, w_6, weightSize], feed_dict = feed_dict)

                        
                    train_c_index = _naive_concordance_index(train_set['T'], -outputV, train_set['O'])
                        
                    targetFH.write(str(step) + "\t" + str(costV) + "\t"  + str(train_c_index)  + "\t" + str(weightSizeV)  + "\n")

                                        
                    hazardV = 0.535487532 * expV
                        

                    print("step: " + str(step) + ", cost: " + str(costV))
                    print("train cIndex: " + str(train_c_index))
                        

                    weightFH.write("step: " + str(step) + "\n")
                    weightFH.write("w1: --- \n" + str(w1V) + "\n" + "w2: --- " + "\n" + str(wV) + "\n" )
                
                    probablityFH.write("step: " + str(step) + "\n")
                    probablityFH.write(str(hazardV) + "\n" )
            
            weightFH.close()
            probablityFH.close()
            targetFH.close()
            gtFH.close()



