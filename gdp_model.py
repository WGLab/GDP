import tensorflow as tf
import numpy as np
import re,math
# The TCGA feature size
FEATURE_SIZE=2931 #TODO, during training , this need be adjusted
NUM_CLASSES=1 # number of final output neurons

#TODO: use tf.cumsum instead
#def cumsum(x,observations):
#    x=tf.reshape(x,(1,observations))
#    values=tf.split(x,observations,axis=1)
#    out=[]
#    prev=tf.zeros_like(values[0])
#    for val in values:
#        s=prev+val
#        out.append(s)
#        prev=s
#    cumsum=tf.concat(out,axis=1)
#    cumsum=tf.reshape(cumsum,(observations,1))
#    return cumsum

def calc_at_risk(X, T, O):
    """
    Calculate the at risk group of all patients. For every patient i, this
    function returns the index of the first patient who died after i, after
    sorting the patients w.r.t. time of death.
    Refer to the definition of
    Cox proportional hazards log likelihood for details: https://goo.gl/k4TsEM

    Parameters
    ----------
    X: numpy.ndarray
        m*n matrix of expression data
    T: numpy.ndarray
        m sized vector of time of death
    O: numpy.ndarray
        m sized vector of observed status (1 - censoring status)

    Returns
    -------
    X: numpy.ndarray
        m*n matrix of expression data sorted w.r.t time of death
    T: numpy.ndarray
        m sized sorted vector of time of death
    O: numpy.ndarray
        m sized vector of observed status sorted w.r.t time of death
    at_risk: numpy.ndarray
        m sized vector of starting index of risk groups
    Output Examples
    ------
    (array([[ 0.        ,  0.        ,  0.        , ...,  0.        ,
        -0.53219468,  0.12068921],
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.33331188,  0.12068921],
       [ 1.        ,  0.        ,  0.        , ...,  0.        ,
         0.61579785,  0.12068921],
       ...,
       [ 1.        ,  0.        ,  0.        , ...,  0.        ,
         0.18626446,  0.12068921],
       [ 0.        ,  1.        ,  0.        , ...,  0.        ,
         1.80174805,  0.12068921],
       [ 1.        ,  0.        ,  0.        , ...,  0.        ,
         0.08993093,  0.12068921]]), array([    6.,    33.,   181.,   279.,   315.,   327.,   359.,   705.,
         734.,  1322.]), array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]), array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32))
    """
    tmp = list(T)
    T = np.asarray(tmp).astype('float64')
    order = np.argsort(T)
    sorted_T = T[order]
    at_risk = np.asarray([list(sorted_T).index(x) for x in sorted_T]).astype('int32')
    T = np.asarray(sorted_T)
    O = O[order]
    X = X[order]
    return X, T, O, at_risk

def activation(x,method):
	if method=='relu':
		return tf.nn.relu(x)
	elif method=='sigmoid':
		return tf.nn.sigmoid(x)
	elif method=='tanh':
		return tf.nn.tanh(x)
	else:
		print("Activation method can only be on of these, relu, sigmoid and tanh")
		return False


def inference(features,hidden_nodes,activation_type,keep_prob,isTrain):
    """ Build the survival model for further inference
    Args:
        features:  the survival co-variants data,
        hidden_nodes: a list, # of nodes in each hidden layer, eg: [20,30,20]
        activation_type: activation function, suggested default relu
	keep_prob: dropout keep_probability
	isTrain: if or not training optimization process
    Returns:
        output: the output value of the neural network
    """
    def f_train(h,k_p):
	    #add dropout to training optimization process
	    return tf.nn.dropout(h,k_p)
    def f_nontrain(h):
	    #doesn't add dropout
	    return h
    pre_hidden_units=0 #keep previous layer nodes number
    for i,hidden_units in enumerate(hidden_nodes):
        layer_index=i+1
        if layer_index==1:
            with tf.name_scope("hidden"+str(layer_index)):
                weights=tf.Variable(tf.truncated_normal([FEATURE_SIZE, hidden_units], dtype=tf.float32)/20,name="weights") # divided by 20 to generate smaller ouptut at initial, otherwise exp(output) will be inf
                #biases1=tf.Variable(tf.zeros([hidden_units]),name="biases")
                #hidden1=tf.nn.relu(tf.matmul(features,weights1)+biases1)
		hidden=activation(tf.matmul(features,weights),activation_type)
                #hidden=tf.nn.relu(tf.matmul(features,weights))
		hidden_dropout=tf.cond(isTrain,lambda: f_train(hidden,keep_prob),lambda: f_nontrain(hidden))
                pre_hidden_units=hidden_units
        else:
            with tf.name_scope("hidden"+str(layer_index)):
                weights=tf.Variable(tf.truncated_normal([pre_hidden_units, hidden_units], dtype=tf.float32)/20,name="weights") # divided by 20 to generate smaller ouptut at initial, otherwise exp(output) will be inf
                #biases1=tf.Variable(tf.zeros([hidden_units]),name="biases")
                #hidden1=tf.nn.relu(tf.matmul(features,weights1)+biases1)
		hidden=activation(tf.matmul(hidden_dropout,weights),activation_type)
		#if add_dropout:
	       #		hidden_dropout=tf.nn.dropout(hidden,keep_prob)
	#	else:
	#		hidden_dropout=hidden
		#hidden=tf.nn.relu(tf.matmul(hidden,weights))
		hidden_dropout=tf.cond(isTrain,lambda: f_train(hidden,keep_prob),lambda: f_nontrain(hidden))
                pre_hidden_units=hidden_units


    with tf.name_scope('output'):
        weights2=tf.Variable(tf.truncated_normal([hidden_units,NUM_CLASSES],dtype=tf.float32)/20,name="weights")
        #biases2=tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
        #output=tf.matmul(hidden1,weights2)+biases2
        output=tf.matmul(hidden_dropout,weights2)

    return output

def inference_linear(features):
    #theata=tf.Variable(tf.truncated_normal([FEATURE_SIZE,1],dtype=tf.float32)/20,name="theta")
    #keep the name scope the same as in NN (for group lasso and sparse_group_lasso weights filtering)
    with tf.name_scope("hidden1"):
	weights=tf.Variable(tf.truncated_normal([FEATURE_SIZE, 1], dtype=tf.float32)/20,name="weights")
        output=tf.matmul(features,weights)
    #output=tf.matmul(features,theata)
    return output

#def loss(inf_output,at_risk,censors,groups,batch_size,alpha,penaltyLambda):
#    """ Calculate the loss based on observed survival data and the predicted ones.
#
#    Args:
#        inf_weights1: the weights1 of the first hidden layer from inference
#        inf_output: the inference output
#        at_risk: #TODO
#        censors: censor status of the patients ([batch_size])
#        batch_size: number of samples/patients data used for each training/validation/testing
#        groups: corresponding group id for each feature  ([feature_size])
#    Returns:
#        loss
#    """
#    groups_size=len(set(groups)) # number of groups
#
#
#    partial_sum=tf.reverse(cumsum(tf.reverse(tf.exp(inf_output),axis=[-2]),batch_size),axis=[-2])+1
#    psum_at_risk=tf.gather(partial_sum,tf.reshape(at_risk,[-1])) # at_risk out of boundary?
#    log_at_risk=tf.log(tf.gather(partial_sum,tf.reshape(at_risk,[-1]))+1e-50)
#    diff=tf.reshape(tf.subtract(inf_output,log_at_risk),[-1])*(1-censors) #Only use non-censored data for difference calculation
#
#    all_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#    all_weights=[x for x in all_variables if re.search("weights",x.name)]
#    weights1=[x for x in all_weights if re.search("hidden1/weights",x.name)][0]
#
#
#    # group lasso regularization for the input-hidden1 weights
#    rg=0.0
#    group_index=0
#    for group_id in range(groups_size):
#        this_group_mask=[i for i,x in enumerate(groups) if x== group_id]
#        rg += tf.sqrt(tf.nn.l2_loss(tf.gather(weights1,tf.to_int64(this_group_mask))))
#
#
#
#    loss = - (tf.reduce_sum(diff)) + alpha*penaltyLambda*rg+(1-alpha)*tf.reduce_sum(penaltyLambda*tf.abs(inf_weights1))
#
#    return loss

#def loss(inf_output,at_risk,censors,groups,batch_size,alpha,scale,reg_type="lasso"):
#    """ Calculate the loss based on observed survival data and the predicted ones.
#
#    Args:
#        inf_output: the inference output
#        at_risk: #TODO
#        censors: censor status of the patients ([batch_size])
#        batch_size: number of samples/patients data used for each training/validation/testing
#        groups: corresponding group id for each feature  ([feature_size])
#        alpha: adjust the proportion of group penalty
#        scale: scale used for regularization adjust (float)
#    Returns:
#        loss
#    """
#    groups_size=len(set(groups)) # number of groups
#
#
#    #TODO: consider the situation of Tied times
#    partial_sum=tf.reverse(cumsum(tf.reverse(tf.exp(inf_output),axis=[-2]),batch_size),axis=[-2])+1
#    #psum_at_risk=tf.gather(partial_sum,tf.reshape(at_risk,[-1])) # at_risk out of boundary?
#    log_at_risk=tf.log(tf.gather(partial_sum,tf.reshape(at_risk,[-1]))+1e-50)
#    diff=tf.reshape(tf.subtract(inf_output,log_at_risk),[-1])*(1-censors) #Only use non-censored data for difference calculation
#
#    if reg_type=="lasso":
#        reg=lasso(scale)
#    elif reg_type=="l2":
#        reg=l2(scale)
#    elif reg_type=='group_lasso':
#        reg=group_lasso(alpha,scale,groups)
#    elif reg_type=='sparse_group_lasso':
#        reg=sparse_group_lasso(alpha,scale,groups)
#
#    loss=-(tf.reduce_sum(diff))+reg
#
#    return loss

def loss(inf_output,censors,groups,batch_size,alpha,scale,delta,reg_type="lasso"):
    """ Calculate the loss based on observed survival data and the predicted ones.

    Args:
        inf_output: the inference output
        censors: censor status of the patients ([batch_size])
        batch_size: number of samples/patients data used for each training/validation/testing
        groups: corresponding group id for each feature  ([feature_size])
        alpha: adjust the proportion of group penalty
        scale: scale used for regularization adjust (float)
        delta: a small value added in log transformed value to avoid log(0)
    Returns:
        loss (negative sum of log parital likelihood+regularizer)
    """
    #groups_size=len(set(groups)) # number of groups


    #TODO: consider the situation of Tied times
    hazard=tf.squeeze(tf.exp(inf_output)) # samples are ordered by dates in increasing way during batch fetching
    hazard_condition_sum=tf.reverse(tf.cumsum(tf.reverse(hazard,[0])),[0]) #for given i, get sum of theta_j where dates_j>=dates_i
    log_sum=tf.log(hazard_condition_sum+delta)
    diff=tf.subtract(tf.squeeze(inf_output),log_sum)*(1-censors) # only consider events Ci==1

    if reg_type=="lasso":
        reg=lasso(scale)
    elif reg_type=="l2":
        reg=l2(scale)
    elif reg_type=='group_lasso':
        reg=group_lasso(alpha,scale,groups)
    elif reg_type=='sparse_group_lasso':
        reg=sparse_group_lasso(alpha,scale,groups)
    elif reg_type=='none':
        reg=0

    loss=-(tf.reduce_sum(diff))+reg

    return loss

def hazard(inf_output):
    """ Calculate the survival hazard value

    Args:
        inf_output: the inference output
    Returns:
    	GDP-CPH hazard
    """

    hazard=tf.squeeze(tf.exp(inf_output)) 
    return hazard


#or l1
def lasso(scale):
    all_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_weights=[x for x in all_variables if re.search("weights",x.name)]
    rg=0.0
    regularizer = tf.contrib.layers.l1_regularizer(scale=scale)
    rg=tf.contrib.layers.apply_regularization(regularizer, all_weights)
    return rg

def l2(scale):
    all_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_weights=[x for x in all_variables if re.search("weights",x.name)]
    regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
    rg=tf.contrib.layers.apply_regularization(regularizer, all_weights)
    return rg

def group_lasso(alpha,scale,groups):
    #groups_size=len(set(groups)) # number of groups
    all_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_weights=[x for x in all_variables if re.search("weights",x.name)]
    weights1=[x for x in all_weights if re.search("hidden1/weights",x.name)][0] #only the input-hidden1 weights were grouped
    weights_others=[x for x in all_weights if not re.search("hidden1/weights",x.name)]

    # group lasso regularization for the input-hidden1 weights
    regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
    rg=0.0
    group_index=0
    for group_id in list(set(groups)):
        this_group_mask=[i for i,x in enumerate(groups) if x== group_id]
        pl=len(this_group_mask)
        rg+=math.sqrt(pl)*tf.sqrt(regularizer(tf.gather(weights1,tf.to_int64(this_group_mask))))
        #rg+=math.sqrt(pl)*tf.contrib.layers.l2_regularizer(tf.gather(weights1,tf.to_int64(this_group_mask)))
        #rg += tf.sqrt(tf.nn.l2_loss(tf.gather(weights1,tf.to_int64(this_group_mask))))
    regularizer2 = tf.contrib.layers.l1_regularizer(scale=scale)
    if(alpha==1):
	#for the case of linear regression, and there are no other weights
	pass
    else:
	rg=rg*alpha+(1-alpha)*tf.contrib.layers.apply_regularization(regularizer2, weights_others)
    return rg


def sparse_group_lasso(alpha,scale,groups):
    #groups_size=len(set(groups)) # number of groups
    all_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_weights=[x for x in all_variables if re.search("weights",x.name)]
    weights1=[x for x in all_weights if re.search("hidden1/weights",x.name)][0]

    # group lasso regularization for the input-hidden1 weights
    regularizer = tf.contrib.layers.l2_regularizer(scale=scale)
    rg=0.0
    group_index=0
    for group_id in list(set(groups)):
        this_group_mask=[i for i,x in enumerate(groups) if x== group_id]
        pl=len(this_group_mask)
        rg+=math.sqrt(pl)*tf.sqrt(regularizer(tf.gather(weights1,tf.to_int64(this_group_mask))))
    regularizer2 = tf.contrib.layers.l1_regularizer(scale=scale)
    rg=alpha*rg+(1-alpha)*tf.contrib.layers.apply_regularization(regularizer2, all_weights)
    return rg


def training(loss,initial_learning_rate):
    """model training and optimization

    Args:
        loss :  loss tensor form loss()
        initial_learning_rate: the initial learning rate for gradient descent

    Returns:
        train_op: Traning optimizer.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss',loss)
    # track global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Learning rate with decay
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 100000, 0.989, staircase=True)
    # gradient descent optimizer for given learning rate
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_op


#def evaluation(inf_output,dates,censors):
#    """Evaluate the quality of the trained model
#
#    Args:
#        inf_output: nerual network output by inference()
#        days_last_follow: days last follow up for each patients #TODO, is it accurate to use days last follow to represent event times
#        death_status: the status of death
#
#    Returns:
#        cindex: concordance index for the model prediction
#        0: absoluate opposite
#        1: absolute correct
#        0.5: same as random
#    """
#    #c_index= lu.concordance_index(dates,-inf_output,1-censors)
#    c_index=tf.reduce_sum([1,2,3]) #TODO
#    return c_index

