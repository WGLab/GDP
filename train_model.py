import gdp_model as dm
import numpy as np
import argparse
import lifelines.utils as lu
import load_data as ld
import tensorflow as tf
import sys
import time
import os

#TODO: allow batch fetching of the data
def get_data(filename):
    """
    Get the datasets from file
    """
    data = np.genfromtxt(filename, dtype=float, missing_values="None", delimiter="\t", skip_header=1)
    f = open(fileName)
    header = f.readline().rstrip().split("\t")
    f.close()

    T = data[:, header.index('days')] # Days to last follow up
    np.place(T , T == 0, 1)

    O = data[:, header.index('dead')] # censor: dead or live ; o or 1
    X = data[:, [index for index in range(len(header)) if index not in [header.index('days'), header.index('dead')] ]]

    #group the genes, same gene same group value
    group = []
    startGroup = 0
    lastGene = ''
    for index in range(len(header)) :
        if (header[index] != 'days' and header[index] != 'dead'):
            if (index == 0) :
                group.append(startGroup)
            elif ('_' not in header[index] ) :
                # clinical features
                startGroup += 1
                group.append(startGroup)
            else :
                gene = header[index].split('_')[0]
                if (lastGene == '' or gene != lastGene) :
                    startGroup += 1
                group.append(startGroup)
                lastGene = gene
    fold_size = int( len(X) / 10) # used to choose training , testing and final datasets
    datasets={"train":{},"eval":{},"test":{}}
    index_s=0
    index_e=0
    total_size=len(x)
    for dtype,frac in {"train":0.6,"eval":0.2,"test":0.2}:
        index_e=index_s+int(total_size*frac)
        datasets[dtype]['X'],datasets[dtype]['T'],datasets[dtype]['O'],datasets[dtype]['A']= dm.calc_at_risk(X[index_s:index_e],T[index_s:index_e],O[index_s:index_e])
        index_s=index_e

    return datasets


def placeholder_inputs(batch_size):
    """Placeholder variables for input tensors
    Args:
        batch_size: batch size for each step
    Returns:
        placeholders for features, at_risks and censors respectively
    """
    #pl stands for placeholder
    feature_pl = tf.placeholder(tf.float32, shape= [batch_size, dm.FEATURE_SIZE]) 
    at_risk_pl = tf.placeholder(tf.int32, shape= [batch_size, ])
    date_pl= tf.placeholder(tf.float32,shape= [batch_size, ])
    censor_pl = tf.placeholder(tf.float32,shape= [batch_size, ]) 
    return feature_pl, at_risk_pl, date_pl, censor_pl



def fill_feed_dict(data_set_next_batch,feature_pl,at_risk_pl,date_pl,censor_pl):
    """get feed_dict for data feeding

    """
#    features,dates,censors,at_risks=data_set.next_batch(FLAGS.batch_size)
    features,dates,censors,at_risks=data_set_next_batch
    feed_dict={
        feature_pl: features,
        at_risk_pl: at_risks,
        date_pl: dates,
        censor_pl: censors,
    }
#    feed_dict={
#        feature_pl: data_set.features,
#        at_risk_pl: data_set.at_risks,
#        date_pl: data_set.dates,
#        censor_pl: data_set.censors,
#    }
    return feed_dict


def do_eval(sess,eval_cindex_o,eval_o_file,eval_type,train_steps,loss_pl,inf_out_pl,feature_pl,at_risk_pl,date_pl,censor_pl,data_set,isTrain_pl,isTrain):
    """ Run evaluation
    Args:
        sess: tensorflow session object
        feature_pl: placeholder for feature
        at_risk_pl: placeholder for at_risk
        censor_pl: placeholder for censor
        data_set:  DataSet format data
    Returns:
        concordance index for the goodness of the model
    """
    cindex_sum=0
    loss_sum=0.0
    count=0
    #steps_per_epoch=data_set.patients_num
    eval_steps=FLAGS.eval_steps
    for step in range(eval_steps):
        data_set_next_batch=data_set.next_batch(FLAGS.batch_size)
        feed_dict=fill_feed_dict(data_set_next_batch,feature_pl,at_risk_pl,date_pl,censor_pl)
	feed_dict[isTrain_pl]=isTrain
        features,dates,censors,at_risks=data_set_next_batch
        loss,info_output=sess.run([loss_pl,inf_out_pl],feed_dict=feed_dict)
        cindex=lu.concordance_index(dates,-info_output,1-censors)
        if FLAGS.output_evaluation_cindex=='YES':
            for k in range(FLAGS.batch_size):
                eval_cindex_o.write(eval_type+"\t"+str(dates[k])+"\t"+str(-info_output[k][0])+"\t"+str(censors[k])+"\t"+str(cindex)+"\n") #TEST: check dates prediction
        cindex_sum+=cindex
        loss_sum+=loss
        count+=1

    cindex=float(cindex_sum)/float(count)
    loss=float(loss_sum)/float(count)



    eval_o_file.write(str(train_steps)+"\t"+str(eval_type)+"\t"+str(loss)+"\t"+str(cindex)+"\n")
    print("cindex:"+str(cindex))
    return cindex

def get_parameters():
    """
    get detailed parameter information
    Returns:
        args information(string)
    """
    flags_dict=vars(FLAGS)
    para_str=""
    for arg in flags_dict:
        value=flags_dict[arg]
        para_str=para_str+" "+str(arg)+":"+str(value)
    return para_str


def run_training():
    """Train, evaluate and test the model
    """
    #get the data and format it as DataSet tuples
    data_sets=ld.read_data_sets(FLAGS.train_dir,FLAGS.train_file)
    feature_groups=data_sets.train.feature_groups 
    feature_size=data_sets.train.feature_size
    dm.FEATURE_SIZE=feature_size #set the feature size of the datasets


    if FLAGS.skip_eval=='NO':
        #output the evaluation results
        if not os.path.exists(FLAGS.evaluation_dir):
            os.makedirs(FLAGS.evaluation_dir)
        eval_output=open(FLAGS.evaluation_dir+"/"+FLAGS.evaluation_file,"w")
        para_info=get_parameters()
        eval_output.write("#"+para_info+"\n")
        eval_output.write("train_steps\teval_data_type\tloss\tcindex\n")

        #output the evaluation cindex detail
        if not os.path.exists(FLAGS.evaluation_dir):
            os.makedirs(FLAGS.evaluation_dir)
        if FLAGS.output_evaluation_cindex=='YES':
            eval_cindex_output=open(FLAGS.evaluation_dir+"/"+FLAGS.evaluation_cindex_file,"w")
            eval_cindex_output.write("eval_type\tdates\tdates_predicted\tcensorship\tcindex\n")

    #Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        #Generate input placeholder
        feature_pl,at_risk_pl,date_pl,censor_pl=placeholder_inputs(FLAGS.batch_size)

        # Build the graph and get the prediction from the inference model

	isTrain_pl=tf.placeholder(tf.bool, shape=()) #boolean value to check if it is during training optimization process
        if FLAGS.model=='NN':
            hidden_nodes=[int(x) for x in FLAGS.hidden_nodes.split(",")]
	    #inf_output_pl=tf.cond(isTrain_pl,inf_train,inf_nontrain)
	    inf_output_pl=dm.inference(feature_pl,hidden_nodes,FLAGS.activation,FLAGS.dropout_keep_rate,isTrain_pl)
            #inf_output_pl=dm.inference(feature_pl,hidden_nodes,FLAGS.activation)
        elif FLAGS.model=='linear':
	    #alpha should equal to 1
            inf_output_pl=dm.inference_linear(feature_pl)

        # Add to the Graph the Ops for loss calculation.
        #loss=dm.loss(inf_output_pl,at_risk_pl,censor_pl,feature_groups,FLAGS.batch_size,FLAGS.alpha,FLAGS.scale,FLAGS.reg_type)
        loss=dm.loss(inf_output_pl,censor_pl,feature_groups,FLAGS.batch_size,FLAGS.alpha,FLAGS.scale,FLAGS.delta,FLAGS.reg_type)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = dm.training(loss, FLAGS.initial_learning_rate)

        # Add evaluation to the Graph
        #cindex_pl=dm.evaluation(inf_output,date_pl,censor_pl)

        # Build the summary Tensor based on the TF collection of Summaries.
        #summary=tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess=tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        #summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # Run the Op to initialize the variables.
        sess.run(init)

        # Begin the training loops.
        for step in range(FLAGS.max_steps):
            start_time=time.time()

            #fill the data
            feed_dict=fill_feed_dict(data_sets.train.next_batch(FLAGS.batch_size),feature_pl,at_risk_pl,date_pl,censor_pl)
	    feed_dict[isTrain_pl]=True
            #run training, update parameters
            _, loss_value, output= sess.run([train_op,loss,inf_output_pl],feed_dict=feed_dict)
            #_, loss_value, loss2_value , output= sess.run([train_op,loss, loss2 ,inf_output_pl],feed_dict=feed_dict) #TEST

            #print("output type:"+str(type(output)))
            #print("output shape:"+str(output.shape))

            duration = time.time() - start_time

            if step % 5 == 0:
		pass
                # Print status to stdout.
                #print('Step %d: loss = %.2f (%.3f sec) : output max = %.2f' % (step, loss_value, duration, output.max()))
                # Update the events file.
                #summary_str = sess.run(summary, feed_dict=feed_dict)
                #summary_writer.add_summary(summary_str, step)
                #summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 5 == 0 or (step + 1) == FLAGS.max_steps:
                print('Step: %d , duration: %.3f sec' % (step+1, duration))
		#Save the variables to disk
                checkpoint_file = os.path.join(FLAGS.saver_file_dir,FLAGS.saver_file_prefix)
                saver.save(sess, checkpoint_file, global_step=step)
                if FLAGS.skip_eval=='NO':
                    if FLAGS.output_evaluation_cindex!='YES':
                        eval_cindex_output=0
                    # Evaluate agianst the traing data.
                    print("Training data evaluation:")
                    do_eval(sess,eval_cindex_output,eval_output,"training_data",step+1,loss,inf_output_pl,feature_pl,at_risk_pl,date_pl,censor_pl,data_sets.train,isTrain_pl,False)

                    print("Validation data evaluation:")
                    do_eval(sess,eval_cindex_output,eval_output,"validation_data",step+1,loss,inf_output_pl,feature_pl,at_risk_pl,date_pl,censor_pl,data_sets.validation,isTrain_pl,False)

                    print("Testing data evaluation:")
                    do_eval(sess,eval_cindex_output,eval_output,"testing_data",step+1,loss,inf_output_pl,feature_pl,at_risk_pl,date_pl,censor_pl,data_sets.test,isTrain_pl,False)
    if FLAGS.skip_eval=='NO':
        eval_output.close()
        if FLAGS.output_evaluation_cindex=='YES':
            eval_cindex_output.close()


def main(_):
  if not tf.gfile.Exists(FLAGS.saver_file_dir):
  	tf.gfile.MakeDirs(FLAGS.saver_file_dir)
    #tf.gfile.DeleteRecursively(FLAGS.log_dir)
  #tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--initial_learning_rate',
      type=float,
      default=0.0001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--alpha',
      type=float,
      default=0.3,
      help='alpha for group regularizer proportion control'
  )
  parser.add_argument(
      '--scale',
      type=float,
      default=0.001,
      help='Lambda scale for regularization'
  )
  parser.add_argument(
      '--delta',
      type=float,
      default=0.01,
      help='delta value used for log transformation in partial likelihood calculation'
  )
  parser.add_argument(
      '--reg_type',
      type=str,
      default="lasso",
      help='types of  regularization (available: lasso, l2, group_lasso, sparse_group_lasso,none)'
  )
  parser.add_argument(
      '--activation',
      type=str,
      default="relu",
      help='activation function (relu, sigmoid or tanh)'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=1000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden_nodes',
      type=str,
      default="50,20",
      help='Number of nodes in each hidden layer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=50,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--eval_steps',
      type=int,
      default=10,
      help='number of steps used for evaluation average score calculation'
  )
  parser.add_argument(
      '--dropout_keep_rate',
      type=float,
      default=0.6,
	  help='dropout keep rate(default:0.6)'
  )
  parser.add_argument(
      '--skip_eval',
      type=str,
      default="NO",
      help='YES or NO for if skipping evaluation'
  )
  parser.add_argument(
      '--saver_file_dir',
      type=str,
      default="./model",
      help='Directory to put the files with saved variables.'
  )
  parser.add_argument(
      '--saver_file_prefix',
      type=str,
      default="gdp.model",
      help='prefix of the saver files (saver files used for reloading the model for prediction)'
  )
  parser.add_argument(
      '--model',
      type=str,
      default="NN",
      help='model used for the training, NN: neural network, linear: linear regression (alpha should be set to 1 for group lasso)'
  )
  parser.add_argument(
      '--train_dir',
      default="./example",
      help='Directory for the input file to do model training,validation and testing',
  )
  parser.add_argument(
      '--train_file',
      default="test.csv",
      help='input file for model training , validation and testing',
  )
  parser.add_argument(
      '--evaluation_file',
      default="test_eval.tsv",
      help='evaluation scores output file(cost, cindex changes in each epoch after one batch of data feeding)',
  )
  parser.add_argument(
      '--evaluation_cindex_file',
      default="test_eval_cindex.csv",
      help='evaluation cindex output file(cindex changes in each training)',
  )
  parser.add_argument(
      '--output_evaluation_cindex',
      type=str,
      default="NO",
      help='whether or not to output evaluation cindex detail information, default(NO), other choice YES',
  )
  parser.add_argument(
      '--evaluation_dir',
      default="eval_output",
      help='evaluation scores output directory(cost, cindex changes in each epoch after one batch of data feeding)',
  )
  if len(sys.argv)==1:
	  parser.print_help()
	  sys.exit(1)

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


