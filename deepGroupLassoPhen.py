from SurvivalAnalysis import SurvivalAnalysis
from lifelines.utils import _naive_concordance_index
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
    
    phenoFile = "/home/cocodong/project/deeplearningMentor/processData/out.final_gene_list"
    pFH = open(phenoFile)
    pHeader = pFH.readline()
    pContent = pFH.readlines()
    phenoGene = {}
    for line in pContent:
        pLineContent = line.rstrip().split("\t")
        phenoGene[pLineContent[1]] = float(pLineContent[3])
    
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
    data = np.genfromtxt(fileName, dtype=float, missing_values="None", delimiter="\t", skip_header=1)
    print("finished loading for the dataset")

    #### load header
    f = open(fileName)
    header = f.readline().rstrip().split("\t")
    f.close()
    
    nonClinicalIndex = []
    clinicalIndex = []
    for i in range(data.shape[1]):
        if '_' in header[i]:
            nonClinicalIndex.append(i)
        else :
            clinicalIndex.append(i)

    clinical = [header[index] for index in clinicalIndex]
    nonClinical = [header[index] for index in nonClinicalIndex]
    
    T = data[:, header.index('days')]
    np.place(T , T == 0, 1)
    O = data[:, header.index('dead')]
    X = data[:, [index for index in range(len(header)) if index not in [0, header.index('days'), header.index('dead')] ]]

    group = []
    startGroup = 0
    lastGene = ''

    groupWeight = {}
    for index in range(len(header)) :
        if (index == 0) :
            continue
        if (header[index] != 'days' and header[index] != 'dead'):
            if (index == 3) :
                group.append(startGroup)
                groupWeight[startGroup] = 0.1

            elif ('_' not in header[index] ) :
                # clinical features
                startGroup += 1
                group.append(startGroup)
                groupWeight[startGroup] = 0.1
                
            else :
                gene = header[index].split('_')[0]
                if (lastGene == '' or gene != lastGene) :
                    startGroup += 1

                lastGene = gene
                group.append(startGroup)
                if (gene in phenoGene):
                    groupWeight[startGroup] = 1 - phenoGene[gene]
                else :
                    groupWeight[startGroup] = 1


    print("finished parsing header")
    
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
        w_6 = tf.Variable(tf.truncated_normal([n_in, n_hidden], dtype=tf.float32)/20)
        w_1 = tf.Variable(tf.truncated_normal([n_hidden, n_out], dtype=tf.float32)/20)
            
            ## output layer


        if (nonLinear == "tanh") :
            output_1 = tf.tanh(tf.matmul(input, w_6))
                
        elif (nonLinear == "sigmoid") :
            output_1 = tf.sigmoid(tf.matmul(input, w_6))
                
        elif (nonLinear == "linear") :
            output_1 = tf.matmul(input, w_6)
            
        output = tf.matmul(output_1, w_1)

            
        ## optimization
        exp = tf.reverse(tf.exp(output), axis = [-1])
        partial_sum_a = cumsum(exp, n_obs)
        partial_sum = tf.reverse(partial_sum_a, axis = [-1]) + 1
        log_at_risk = tf.log(tf.gather(partial_sum, tf.reshape(at_risk, [-1])) + 1e-50)
        diff = output - log_at_risk
        times = tf.reshape(diff, [-1]) * observed
            
        # group lasso penalty
        regularization = 0.0
        for eachGroup in range(startGroup):
            groupIndex = [i for i, x in enumerate(group) if x == eachGroup]
            if (eachGroup in groupWeight) :
                regularization += tf.sqrt(groupWeight[eachGroup]  * groupWeight[eachGroup] * tf.nn.l2_loss(tf.gather(w_6, tf.to_int64(groupIndex))))
            
        cost = - (tf.reduce_sum(times)) * 1 / n_obs + alpha * penaltyLambda * regularization + (1 - alpha) * tf.reduce_sum(penaltyLambda * tf.abs(w_6))  + (1 - alpha) * tf.reduce_sum(penaltyLambda * tf.abs(w_1))

        weightSize = tf.nn.l2_loss(w_6)
            
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.0001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.989, staircase=True)
        
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


    print("finished loading graphs")

    for alphaArrayIndex in range(len(alphaArray)):
        print("alpha: " + str(alphaArray[alphaArrayIndex]))
        
        for penaltyLambdaIndex in range(len(penaltyLambdaArray)):
            print("lambda: " + str(penaltyLambdaArray[penaltyLambdaIndex]))
            
            targetFile = prefix + ".lambda." + str(penaltyLambdaArray[penaltyLambdaIndex]) + ".alpha." + str(alphaArray[alphaArrayIndex]) + ".txt"
            targetFH = open(targetFile, "w")
            targetFH.write("iteration" + "\t" + "trainCost"  + "\t"  + "\t" + "trainCIndex" + "\t"  + "weightSize" + "\t" + "\n")
            
            weightFile = prefix + ".lambda." + str(penaltyLambdaArray[penaltyLambdaIndex]) + ".alpha." + str(alphaArray[alphaArrayIndex]) + ".weight.txt"
            weightFH = open(weightFile, "w")
            
            probablityFile = prefix + ".lambda." + str(penaltyLambdaArray[penaltyLambdaIndex]) + ".alpha." + str(alphaArray[alphaArrayIndex]) + ".predict.txt"
            probablityFH = open(probablityFile, "w")
            
            
            config = tf.ConfigProto(allow_soft_placement=True)
            
            with tf.Session(config=config) as session:
                
                for step in range(num_steps):
                    
                    tf.global_variables_initializer().run()
                    
                    
                    feed_dict = {input : train_set['X'], at_risk : train_set['A'], observed : train_set['O'], keep_prob : 1, penaltyLambda : penaltyLambdaArray[penaltyLambdaIndex], alpha : alphaArray[alphaArrayIndex]}
                        
                    _, partial_sumV, diffV, log_at_riskV, timesV, expV, outputV, costV, w1V, wV, weightSizeV= session.run([optimizer, partial_sum, diff, log_at_risk, times, exp, output, cost, w_1, w_6, weightSize], feed_dict = feed_dict)

                    print("exp: " + str(expV))
                    print("partial_sum: " + str(partial_sumV))
                    print("log_at_risk: " + str(log_at_riskV))
                    print("diff: " + str(diffV))
                    print("times: " + str(timesV))
                    
                        
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




