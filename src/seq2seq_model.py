import numpy as np
import tensorflow as tf
from modified_seq2seq import modified_rnn_decoder, modified_tied_rnn_seq2seq
import rnn_cell_extensions
class Seq2SeqModel(object):
    def __init__(self,number_features,source_seq_len, target_seq_len, rnn_size,batch_size, lr, dtype=tf.float32):
        self.features = number_features
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.lr = lr
        
        self.keep_prob = tf.placeholder_with_default(1.0,shape=())
        
        self.global_step = tf.Variable(0, trainable=False)
        print('rnn_size = {0}'.format(rnn_size))
        
        cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        cell_drop = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=self.keep_prob,output_keep_prob=self.keep_prob)
        with tf.name_scope("inputs"):
            enc_in = tf.placeholder(dtype, shape=[None,self.source_seq_len, self.features], name='enc_in')
            dec_in = tf.placeholder(dtype, shape=[None, self.target_seq_len, self.features],name='dec_in')
            dec_out = tf.placeholder(dtype, shape=[None, self.target_seq_len, self.features],name='dec_out')

            self.encoder_inputs = enc_in
            self.decoder_inputs = dec_in
            self.decoder_outputs = dec_out

            enc_in = tf.transpose(enc_in, [1,0,2])
            dec_in = tf.transpose(dec_in, [1,0,2])
            dec_out = tf.transpose(dec_out, [1,0,2])

            enc_in = tf.reshape(enc_in, [-1,self.features])
            dec_in = tf.reshape(dec_in,[-1,self.features])
            dec_out = tf.reshape(dec_out, [-1,self.features])

            enc_in = tf.split(enc_in, source_seq_len, axis=0)
            dec_in = tf.split(dec_in, target_seq_len, axis=0)
            dec_out = tf.split(dec_out, target_seq_len, axis=0)

        cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell_drop, self.features)

        outputs = []
        lf = None
        def lf (prev, i):
            return prev

        outputs, self.states, self.enc_outputs, self.enc_states = modified_tied_rnn_seq2seq(enc_in, dec_in, cell, loop_function=lf)
        self.outputs = outputs
        loss = tf.reduce_mean(tf.square(tf.subtract(dec_out,outputs)))
        self.loss = loss
        #l2_loss = tf.losses.get_regularization_loss()
        #self.loss = loss+l2_loss

        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.lr)
        gradients, params = zip(*opt.compute_gradients(self.loss))
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 1)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(zip(clipped_gradients,params),global_step = self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 12)
    
    def step(self, session, encoder_inputs, decoder_inputs, decoder_outputs,forward_only):
        
        if not forward_only:
            input_feed = {self.encoder_inputs:encoder_inputs, self.decoder_inputs:decoder_inputs,
                      self.decoder_outputs:decoder_outputs}
            output_feed = [self.updates, self.gradient_norms, self.loss,self.keep_prob]
            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2],outputs[3]
        else:
            input_feed = {self.encoder_inputs:encoder_inputs, self.decoder_inputs:decoder_inputs,
                      self.decoder_outputs:decoder_outputs}
            output_feed = [self.loss,self.outputs,self.states,self.enc_outputs, self.enc_states]
            outputs = session.run(output_feed, input_feed)
            return outputs[0],outputs[1],outputs[2],outputs[3],outputs[4]