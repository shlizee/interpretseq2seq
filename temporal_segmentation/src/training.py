import tensorflow as tf
import numpy as np
import data_preprocessing
import os
def get_session():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  return session

def load_model(sess,model,train_dir,load):
  ckpt = tf.train.get_checkpoint_state(train_dir,latest_filename="checkpoint")
  print("train_dir",train_dir)
  if ckpt and ckpt.model_checkpoint_path:
    if load>0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(load))):
        ckpt_name = os.path.join(train_dir,"checkpoint-{0}".format(load))
      else:
        raise ValueError("asked to load checkpoint {0}, but it does not seem to exist.".format(load))
    else:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    print("loading model {0}".format(ckpt_name))
    model.saver.restore(sess,ckpt.model_checkpoint_path)
  else:
    print("could not find checkpoint. Aborting.")
    raise(ValueError, "checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))
  return model

def train_model(sess,model,train_dir,data,batch_size,current_step,iter):
  for i in range(1,iter+1):
    encoder_inputs,decoder_inputs,decoder_outputs = data.mini_batch(model,batch_size)
    _,_,train_loss,_ = model.step(sess,encoder_inputs,decoder_inputs,decoder_outputs,False)
    current_step +=1
    if i%100 == 0:
      print("step {0}:train loss:{1}".format(i,train_loss))
    if i%1000 == 0:
      model.saver.save(sess,os.path.normpath(os.path.join(train_dir,'checkpoint')),global_step=current_step)

