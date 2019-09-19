import tensorflow as tf
import numpy as np
import os
import copy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from AGED import Seq2SeqModel
from helper import get_gt, get_demo_checkpoint
import scipy.io
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore")
#suppress tensorflow warning
tf.logging.set_verbosity(tf.logging.ERROR)

tf.app.flags.DEFINE_boolean("demo",True,"use saved model.")

tf.app.flags.DEFINE_string("num_trial",'01',"the trial of the subject.")

tf.app.flags.DEFINE_string("train_dir",os.path.normpath("./training/cmu/sub_86"),"the directory of saved model")

FLAGS = tf.app.flags.FLAGS

data_dir = os.path.normpath("./data/cmu_mocap/subject_86/expmap")

train_dir = os.path.normpath(os.path.join(FLAGS.train_dir,"trial_{0}".format(FLAGS.num_trial)))

demo_dir = os.path.normpath("./demo/{0}".format(FLAGS.num_trial))

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

def main(_):  
    print("load data...")
    data = scipy.io.loadmat('{0}/{1}.mat'.format(data_dir,FLAGS.num_trial))
    data = data['expmapchannels']
    print("Starting preprocessing data...")
    data_mean, data_std, data_ignore,data_use = normalization_stats(data)
    seg,true_labels = get_gt(data,FLAGS.num_trial)
    for i in range(1,len(list(seg.keys()))+1):
        seg[i] = np.divide(seg[i]-data_mean,data_std)
        seg[i] = seg[i][:,data_use]
    #set hyperparameters
    source_seq_len=50
    target_seq_len=25
    rnn_size=1024
    batch_size=16
    lr = .0001
    train_keep_prob = 1.0
    iterations = 10000
    tf.reset_default_graph()
    sess = get_session()
    model = Seq2SeqModel(source_seq_len, target_seq_len, rnn_size, batch_size, lr,train_keep_prob)
    if FLAGS.demo:
        print("get Seq2Seq model...")
        checkpoint_dir = get_demo_checkpoint(demo_dir,FLAGS.num_trial)
        print(checkpoint_dir)
        model.saver.restore(sess,checkpoint_dir)
    else:
        print("training fresh Seq2Seq model...")
        sess.run(tf.global_variables_initializer())
        for i in range(1,iterations+1):
            encoder_inputs, decoder_inputs, decoder_outputs = mini_batch(seg,source_seq_len, target_seq_len, batch_size)
            _,gradient_norm,train_loss,_ = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, False)
            if i%100 == 0:
                print("step {0}: train loss:{1:.4f}, gradient norm:{2}".format(i, train_loss, gradient_norm))
            if i%500 == 0:
                model.saver.save(sess,os.path.normpath(os.path.join(train_dir,'checkpoint')),global_step=i)
    test_states = {}
    for i in range(1,len(list(seg.keys()))+1):
        test_states[i] = []
        start = 0
        while start<seg[i].shape[0]:
            if start+source_seq_len+target_seq_len>=seg[i].shape[0]:
                break
            end = start + source_seq_len+target_seq_len
            encoder_inputs = np.zeros((1, source_seq_len, 62),dtype=float)
            decoder_inputs = np.zeros((1, target_seq_len, 62),dtype=float)
            decoder_outputs = np.zeros((1,target_seq_len,62),dtype=float)
            data_sel = seg[i][start:end,:]
            encoder_inputs[0,:,:] = data_sel[:source_seq_len,:]
            decoder_inputs[0,:,:] = data_sel[source_seq_len-1:source_seq_len+target_seq_len-1,:]
            decoder_outputs[0,:,:] = data_sel[source_seq_len:,:]
            test_loss,dec_out,dec_states,_,enc_states = model.step(sess,encoder_inputs,decoder_inputs,decoder_outputs,True)
            test_states[i].append(dec_states)
            start+=target_seq_len
    d = {}
    for i in range(1,len(list(seg.keys()))+1):
        d[i] = np.vstack(np.reshape(np.asarray(test_states[i][j]),(-1,1024)) for j in range(len(test_states[i])))
    all_states = np.vstack(d[i] for i in range(1,len(list(seg.keys()))+1))
    clustering = AgglomerativeClustering(affinity='cosine',n_clusters=max(true_labels)+1,linkage='single').fit(all_states)
    labels = clustering.labels_
    all_labels = assign_miss_label(labels,model,seg)
    gt,pred = order_labels(true_labels,all_labels)
    acc = accuracy_score(gt,pred)
    print("the accuracy is {0}".format(acc))
    plot_results(gt,pred)

def plot_results(test_gt,test_pred):
    colors = {}
    cmap = plt.get_cmap('tab20')
    for label_idx,label in enumerate(np.unique(test_pred)):
        colors[label] = cmap(label_idx/len(np.unique(test_pred)))
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12,2),dpi=200)
    plt.axis("off")
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_xlim([0,len(test_pred)])
    ax1.set_ylim([0,0.05])
    ax1.set_ylabel("pred",rotation=0,labelpad=50)
    ax1.set_xticks([])
    ax1.set_yticks([])

    for start,end,label in bounds(test_pred):
        ax1.barh(0,end-start,color=colors[label],left=start,height=0.1)
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_xlim([0,len(test_gt)])
    ax2.set_ylim([0,0.05])
    ax2.set_ylabel("GT",rotation=0,labelpad=50)
    ax2.set_yticks([])

    for start,end,label in bounds(test_gt):
            ax2.barh(0,end-start,color=colors[label],left=start,height=0.1)
    plt.xlabel('frames',fontsize=32)
    plt.show()

def order_labels(true_labels,pred_labels):
    key = []
    key_ = []
    
    for i in true_labels:
        if i not in key:
            key.append(i)
    for i in pred_labels:
        if i not in key_:
            key_.append(i)
    labels_true = []
    labels_pred = []
    for i in true_labels:
        labels_true.append(key.index(i))
    for i in pred_labels:
        labels_pred.append(key_.index(i))
    return labels_true, labels_pred

def assign_miss_label(labels,model,sub):
    all_labels = labels
    all_labels = np.insert(all_labels,0,np.repeat(labels[0],model.target_seq_len))
    g = 0
    for i in range(1,len(list(sub.keys()))+1):
        start = 0
        while start<sub[i].shape[0]:
            if start + model.source_seq_len+model.target_seq_len>=sub[i].shape[0]:
                miss = sub[i].shape[0]-start
                all_labels = np.insert(all_labels,g+start,np.repeat(all_labels[g+start-1],miss-model.target_seq_len))
                g+=sub[i].shape[0]
                break
            start+=model.target_seq_len
        if i==len(list(sub.keys())):
            break
        else:
            all_labels = np.insert(all_labels,g,np.repeat(all_labels[g],model.target_seq_len))
    return all_labels

def bounds(segm):
    start_label = segm[0]
    start_idx = 0
    idx = 0
    while idx < len(segm):
        try:
            while start_label == segm[idx]:
                idx += 1
        except IndexError:
            yield start_idx, idx, start_label
            break

        yield start_idx, idx, start_label
        start_idx = idx
        start_label = segm[start_idx]

def normalization_stats(completeData):
    data_mean = np.mean(completeData,axis=0)
    data_std = np.std(completeData,axis=0)
    dimensions_to_ignore = []
    dimensions_to_use = []
    dimensions_to_ignore.extend(list(np.where(data_std<1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std>=1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0
    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use

def mini_batch(data, source_seq_len, target_seq_len, batch_size):
    total_frames = source_seq_len + target_seq_len
    encoder_inputs = np.zeros((batch_size, source_seq_len, 62),dtype=float)
    decoder_inputs = np.zeros((batch_size, target_seq_len, 62),dtype=float)
    decoder_outputs = np.zeros((batch_size,target_seq_len,62),dtype=float)
    
    for i in range(batch_size):
        key = np.random.randint(1,len(list(data.keys()))+1)
        r,c = data[key].shape
        idx = np.random.randint(0, r-total_frames)
        data_sel = data[key][idx:idx+total_frames, :]
        #data_sel = data[idx:idx+total_frames,:]
        encoder_inputs[i,:,:] = data_sel[0:source_seq_len,:]
        decoder_inputs[i,:,:] = data_sel[source_seq_len-1:source_seq_len+target_seq_len-1, :]
        decoder_outputs[i,:,:] = data_sel[source_seq_len:,:]
        
    return encoder_inputs, decoder_inputs, decoder_outputs

def get_session():
    """Create a session that dynamically allocates memory."""
    # See: https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

if __name__ == "__main__":
    tf.app.run()