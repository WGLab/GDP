import gdp_model as dm
import numpy as np
import argparse
import lifelines.utils as lu
import load_data as ld
import tensorflow as tf
import sys
import time
import os

def placeholder_inputs(batch_size):
    """Placeholder variables for input tensors
    Args:
        batch_size: batch size for each step
    Returns:
        placeholders for features, at_risks and censors respectively
    """
    #pl stands for placeholder
    feature_pl = tf.placeholder(tf.float32, shape= [batch_size, dm.FEATURE_SIZE]) 
    return feature_pl


def run_prediction():
    """making prediction based on trained models
    """
    #get the data and format it as DataSet tuples
    data_sets=ld.read_prediction_data(FLAGS.prediction_dir,FLAGS.prediction_file)
    feature_groups=data_sets.feature_groups
    feature_size=data_sets.feature_size
    sample_size=data_sets.patients_num
    #features=data_sets.features
    features = np.asarray(list(data_sets.features)).astype('float32')

    dm.FEATURE_SIZE=feature_size #set the feature size of the datasets

    #Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        #Generate input placeholder
        feature_pl=placeholder_inputs(sample_size)

        # Build the graph and get the prediction from the inference model

	isTrain_pl=tf.placeholder(tf.bool, shape=()) #boolean value to check if it is during training optimization process
        if FLAGS.model=='NN':
            hidden_nodes=[int(x) for x in FLAGS.hidden_nodes.split(",")]
	    inf_output_pl=dm.inference(feature_pl,hidden_nodes,FLAGS.activation,1,isTrain_pl)
        elif FLAGS.model=='linear':
            inf_output_pl=dm.inference_linear(feature_pl)

	hazard_pl=dm.hazard(inf_output_pl)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess=tf.Session()

	#restore the variables
	checkpoint_file = os.path.join(FLAGS.saver_file_dir,FLAGS.saver_file_prefix)
	saver.restore(sess, checkpoint_file)
	feed_dict={
		feature_pl: features,
		isTrain_pl: False
	}
	hazard=sess.run([hazard_pl],feed_dict=feed_dict)
	outfile=open(FLAGS.out_dir+"/"+FLAGS.out_file,"w")
	for h in hazard[0]:
		h_=h.astype('|S32')
		outfile.write(h_+"\n")
	outfile.close()


def main(_):
  if not tf.gfile.Exists(FLAGS.saver_file_dir):
  	tf.gfile.MakeDirs(FLAGS.saver_file_dir)
  if not tf.gfile.Exists(FLAGS.out_dir):
	  tf.gfile.MakeDirs(FLAGS.out_dir)
  run_prediction()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
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
      '--hidden_nodes',
      type=str,
      default="50,20",
      help='Number of nodes in each hidden layer.'
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
      '--prediction_dir',
      default="./example",
      help='Directory for the input file to do model training,validation and testing',
  )
  parser.add_argument(
      '--prediction_file',
      default="data2.csv",
      help='input file for making prediction,the sample size in this file should equal to the batch size in the training process',
  )
  parser.add_argument(
      '--out_dir',
      default="./out",
      help='output file directory',
  )
  parser.add_argument(
      '--out_file',
      default="hazard.out.txt",
      help='output file name',
  )
  if len(sys.argv)==1:
          parser.print_help()
          sys.exit(1)

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
