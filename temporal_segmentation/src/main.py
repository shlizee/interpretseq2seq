import numpy as np
import os
import tensorflow as tf
import seq2seq_model
import training
import segmentation
import data_preprocessing
from forward_kinematics_cmu import get_gt
import viz
from generate_video import generate_video
#suppress tensorflow warning
tf.logging.set_verbosity(tf.logging.ERROR)
###Hyperparameters Arguments###

#repeatable action or not
tf.app.flags.DEFINE_boolean("unique",True,"repeatable actions or not.")

#actions list
tf.app.flags.DEFINE_integer("num_actions",3,"number of actions.")

tf.app.flags.DEFINE_string("actions","random","specific actions.")
#Learning rate
tf.app.flags.DEFINE_float("learning_rate",.0001,"Learning rate")
#batch size
tf.app.flags.DEFINE_integer("batch_size",16,"Batch size to use during training.")
#number of neurons
tf.app.flags.DEFINE_integer("rnn_size",1024,"Size of each layer")
#number of encoder frames
tf.app.flags.DEFINE_integer("source_seq_len",25,"Number of frames to feed into encoder")
#number of decoder frames
tf.app.flags.DEFINE_integer("target_seq_len",25,"Number of frames to predict from decoder")
#training iterations
tf.app.flags.DEFINE_integer("iterations",int(1e4),"Iterations to train.")

#data name
tf.app.flags.DEFINE_string("dataset","cmu","Which Data to use")

###Directories###
#model saving directory
tf.app.flags.DEFINE_string("train_dir",os.path.normpath("./training/"),"training directory")

#video saving directory
tf.app.flags.DEFINE_string("video_dir",os.path.normpath("./videos/"),"video directory")

###Load checkpoint###
tf.app.flags.DEFINE_integer("load",0,"Try to load a previous checkpoint.")

###generate video or not###
tf.app.flags.DEFINE_boolean("gen_video",False,"whether to generate video.")


FLAGS = tf.app.flags.FLAGS

if FLAGS.dataset=="cmu":
 	data_dir = ("./data/cmu_mocap")
elif FLAGS.dataset=="H3.6M":
	data_dir = ("./data/h3.6m")

train_dir = os.path.normpath(os.path.join(FLAGS.train_dir,FLAGS.dataset,"{0}_actions".format(FLAGS.num_actions),
	"in_{0}".format(FLAGS.source_seq_len),"out_{0}".format(FLAGS.target_seq_len),
	"iter_{0}".format(FLAGS.iterations),
	"size_{0}".format(FLAGS.rnn_size),
	"lr_{0}".format(FLAGS.learning_rate)))
if not os.path.exists(train_dir):
	os.makedirs(train_dir)


def main(_):
	print("Start preprocessing data...")
	if FLAGS.actions == "random":
		data = data_preprocessing.data_prep(FLAGS.dataset,FLAGS.num_actions,FLAGS.unique)
	else:
		actions = FLAGS.actions.split(",")
		data = data_preprocessing.data_prep(FLAGS.dataset,FLAGS.num_actions,FLAGS.unique,actions)
	video_dir = os.path.normpath(os.path.join(FLAGS.video_dir,FLAGS.dataset,"{0}_actions".format(FLAGS.num_actions),
			'_'.join(data.actions)))
	tf.reset_default_graph()
	print("Create model...")
	model = seq2seq_model.Seq2SeqModel(len(data.dim_use),FLAGS.source_seq_len,FLAGS.target_seq_len,
		FLAGS.rnn_size,FLAGS.batch_size,FLAGS.learning_rate)
	sess = training.get_session()
	#perform segmentations with pre-trained model
	if FLAGS.load>0:
		model = training.load_model(sess,model,train_dir,FLAGS.load)

	else:
		if FLAGS.load<=0:
			current_step = 0
			print("Creating model with fresh parameters.")
			sess.run(tf.global_variables_initializer())
		else:
			current_step = FLAGS.load+1
			model = training.load_model(sess,model,train_dir,FLAGS.load)
		print("Training is in process...")
		training.train_model(sess,model,train_dir,data,model.batch_size,current_step,FLAGS.iterations)
	print("Finding temporal segemnts...")
	test_states,all_states,cuts = segmentation.find_cuts(sess,model,data.norm_complete_train)
	print("Doing clustering...")
	pred_labels,reduced_states,labels_true,all_labels = segmentation.clustering(model,cuts,test_states,all_states,data.trainData,FLAGS.num_actions)
	labels_true,labels_pred = segmentation.order_labels(labels_true,all_labels)
	colors = viz.get_color(labels_true)
	xyz_gt = get_gt(data.complete_train)
	print("Generate results...")
	print("length of gt:{0}, pred:{1},reduced:{2}:".format(len(labels_true),len(labels_pred),len(reduced_states)))
	generate_video(video_dir,labels_true,labels_pred,xyz_gt,reduced_states,colors,FLAGS.gen_video)

if __name__ == "__main__":
	tf.app.run()