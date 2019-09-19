import tensorflow as tf
import numpy as np
from modified_seq2seq import modified_rnn_decoder, modified_tied_rnn_seq2seq
from modified_static_rnn import modified_static_rnn
import rnn_cell_extensions

def df(x,rnn_size):
    with tf.variable_scope("discriminator_f"):
        gru_df = tf.contrib.rnn.GRUCell(rnn_size)
        outputs, final_state, all_states = modified_static_rnn(gru_df,x,dtype=tf.float32)
        logits = tf.layers.dense(final_state,1)
        return logits

def dc(x,rnn_size):
    with tf.variable_scope("discriminator_c"):
        gru_dc = tf.contrib.rnn.GRUCell(rnn_size)
        outputs, final_state,all_states = modified_static_rnn(gru_dc,x,dtype=tf.float32)
        logits = tf.layers.dense(final_state,1)
        return logits

class Seq2SeqModel(object):
    def __init__(self,source_seq_len, target_seq_len, rnn_size,batch_size, lr,train_keep_prob, dtype=tf.float32):
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.lr = lr
        
        self.keep_prob = tf.placeholder_with_default(1.0,shape=())
        
        self.global_step = tf.Variable(0, trainable=False)
        print('rnn_size = {0}'.format(rnn_size))
        
        cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        with tf.name_scope("inputs"):
            enc_in = tf.placeholder(dtype, shape=[None,self.source_seq_len, 62], name='enc_in')
            dec_in = tf.placeholder(dtype, shape=[None, self.target_seq_len, 62],name='dec_in')
            dec_out = tf.placeholder(dtype, shape=[None, self.target_seq_len, 62],name='dec_out')

            self.encoder_inputs = enc_in
            self.decoder_inputs = dec_in
            self.decoder_outputs = dec_out

            enc_in = tf.transpose(enc_in, [1,0,2])
            dec_in = tf.transpose(dec_in, [1,0,2])
            dec_out = tf.transpose(dec_out, [1,0,2])

            enc_in = tf.reshape(enc_in, [-1,62])
            dec_in = tf.reshape(dec_in,[-1,62])
            dec_out = tf.reshape(dec_out, [-1,62])

            enc_in = tf.split(enc_in, source_seq_len, axis=0)
            dec_in = tf.split(dec_in, target_seq_len, axis=0)
            dec_out = tf.split(dec_out, target_seq_len, axis=0)

        cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, 62)

        outputs = []
        lf = None
        def lf (prev, i):
            return prev

        outputs, self.states, self.enc_outputs, self.enc_states = modified_tied_rnn_seq2seq(enc_in, dec_in, cell, loop_function=lf)
        self.outputs = outputs

        with tf.variable_scope("df") as scope:
            df_real = df(dec_out,self.rnn_size)
            scope.reuse_variables()
            df_fake = df(outputs,self.rnn_size)
        loss_f1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=df_real,labels=tf.ones_like(df_real)))
        loss_f2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=df_fake,labels=tf.zeros_like(df_fake)))
        f_loss = loss_f1 + loss_f2
        
        with tf.variable_scope("dc") as scope:
            concat_real = tf.concat([enc_in,dec_out],axis=0)
            #print(concat_real)
            concat_real = tf.reshape(concat_real,[-1,62])
            concat_real = tf.split(concat_real,self.source_seq_len+self.target_seq_len,axis=0)
            #print(concat_real)
            dc_real = dc(concat_real,self.rnn_size)
            scope.reuse_variables()
            #print(dc_real)
            concat_fake = tf.concat([enc_in,outputs],axis=0)
            concat_fake = tf.reshape(concat_fake,[-1,62])
            concat_fake = tf.split(concat_fake,self.source_seq_len+self.target_seq_len,axis=0)
            dc_fake = dc(concat_fake,self.rnn_size)
        loss_c1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dc_real,labels=tf.ones_like(dc_real)))
        loss_c2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dc_fake,labels=tf.zeros_like(dc_fake)))
        c_loss = loss_c1 + loss_c2
        
        loss_g = tf.reduce_mean(tf.square(tf.subtract(dec_out,outputs)))
        self.loss = 0.6*(loss_g + f_loss) + c_loss

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.lr)
        gradients, params = zip(*opt.compute_gradients(self.loss))
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 1)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(zip(clipped_gradients,params),global_step = self.global_step)
        
        
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=12)
    
    def step(self, session, encoder_inputs, decoder_inputs, decoder_outputs,forward_only):
        
        if not forward_only:
            input_feed = {self.encoder_inputs:encoder_inputs, self.decoder_inputs:decoder_inputs,
                      self.decoder_outputs:decoder_outputs, self.keep_prob:train_keep_prob}
            output_feed = [self.updates, self.gradient_norms, self.loss,self.keep_prob]
            outputs = session.run(output_feed, input_feed)
            #session.run(self.reset_op)
            return outputs[0], outputs[1], outputs[2],outputs[3]
        else:
            input_feed = {self.encoder_inputs:encoder_inputs, self.decoder_inputs:decoder_inputs,
                      self.decoder_outputs:decoder_outputs}
            output_feed = [self.loss,self.outputs,self.states,self.enc_outputs, self.enc_states]
            outputs = session.run(output_feed, input_feed)
            return outputs[0],outputs[1],outputs[2],outputs[3],outputs[4]

