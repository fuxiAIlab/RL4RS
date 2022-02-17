from __future__ import print_function
import tensorflow as tf

from .layers import *
from .modules import *
from .utils import *


class Generator:
    def __init__(self,
                 l1_mask,
                 l2_mask,
                 l3_mask,
                 l0_ssr_mask,
                 is_training=True,
                 lr=0.001,
                 temperature=1,
                 train_sample='random',
                 predict_sample='random',
                 seq_length=500,
                 res_length=9,
                 hidden_units=64,
                 dropout_rate=0.1,
                 num_heads=4,
                 num_layers=1,
                 num_glimpse=1,
                 num_blocks=2,
                 use_mha=True,
                 beam_size=3
                 ):

        self.user = tf.placeholder(tf.float32, shape=(None, 256), name='user')  # 779

        self.batch_size = tf.shape(self.user)[0]
        self.item_cand = tf.placeholder(tf.int32, shape=(None, seq_length), name='item_cand')

        self.decode_target_ids = tf.placeholder(dtype=tf.int32, shape=[None, res_length], name="decoder_target_ids")  # [batch_size, res_length]
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")  # [batch_size]

        # Encoder
        with tf.variable_scope("encoder"):
            # region emb
            self.enc_user = tf.layers.dense(self.user, hidden_units, activation=tf.nn.relu)  # (N, T_q, C)
            # enc_item = [batch_size, seq_len, hidden_units]
            self.enc_item = embedding(self.item_cand,
                                      vocab_size=500,
                                      num_units=hidden_units,
                                      zero_pad=False,
                                      scale=True,
                                      scope='enc_item_embed',
                                      # reuse=not is_training,
                                      reuse=False
                                      )
            self.enc = tf.concat([tf.stack(seq_length * [self.enc_user], axis=1), self.enc_item], axis=2)
            # endregion
            # region Dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=dropout_rate,
                                         training=tf.convert_to_tensor(is_training))
            # endregion
            # region squence
            if use_mha:
                ## Blocks
                for i in range(num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hidden_units * 2,
                                                       num_heads=num_heads,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       causality=False)

                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4 * hidden_units, hidden_units * 2])
            else:
                cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_units * 2)
                outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.enc, dtype=tf.float32)
                self.enc = outputs
            # endregion

        # Decoder
        with tf.variable_scope("decoder"):
            dec_cell = LSTMCell(hidden_units * 2)

            if num_layers > 1:
                cells = [dec_cell] * num_layers
                dec_cell = MultiRNNCell(cells)
            # ptr sampling
            enc_init_state = trainable_initial_state(self.batch_size, dec_cell.state_size)

            custom_logits, custom_path, _ = ptn_rnn_decoder(
                dec_cell, None,
                self.enc, enc_init_state,
                seq_length, res_length, hidden_units * 2,
                num_glimpse, self.batch_size,
                l1_mask, l2_mask, l3_mask, l0_ssr_mask,
                mode="CUSTOM", reuse=False, beam_size=None,
                temperature=temperature,
                train_sample=train_sample, predict_sample=predict_sample
            )
            # logits: [batch_size, res_length, seq_length]
            self.custom_logits = tf.identity(custom_logits, name="custom_logits")
            # sample_path: [batch_size, res_length]
            self.custom_path = tf.identity(custom_path, name="custom_path")
            self.custom_result = batch_gather(self.item_cand, self.custom_path)
            sampled_logits, sampled_path, _ = ptn_rnn_decoder(
                dec_cell, None,
                self.enc, enc_init_state,
                seq_length, res_length, hidden_units * 2,
                num_glimpse, self.batch_size,
                l1_mask, l2_mask, l3_mask, l0_ssr_mask,
                mode="SAMPLE", reuse=True, beam_size=None,
                temperature=temperature,
                train_sample=train_sample, predict_sample=predict_sample
            )
            # logits: [batch_size, res_length, seq_length]
            self.sampled_logits = tf.identity(sampled_logits, name="sampled_logits")
            # sample_path: [batch_size, res_length]
            self.sampled_path = tf.identity(sampled_path, name="sampled_path")
            self.sampled_result = batch_gather(self.item_cand, self.sampled_path)

            # self.decode_target_ids is placeholder
            decoder_logits, _ = ptn_rnn_decoder(
                dec_cell, self.decode_target_ids,
                self.enc, enc_init_state,
                seq_length, res_length, hidden_units * 2,
                num_glimpse, self.batch_size,
                l1_mask, l2_mask, l3_mask, l0_ssr_mask,
                mode="TRAIN", reuse=True, beam_size=None,
                temperature=temperature,
                train_sample=train_sample, predict_sample=predict_sample
            )
            self.dec_logits = tf.identity(decoder_logits, name="dec_logits")

            _, beam_path, _ = ptn_rnn_decoder(
                dec_cell, None,
                self.enc, enc_init_state,
                seq_length, res_length, hidden_units * 2,
                num_glimpse, self.batch_size,
                l1_mask, l2_mask, l3_mask, l0_ssr_mask,
                mode="BEAMSEARCH", reuse=True, beam_size=beam_size,
                temperature=temperature,
                train_sample=train_sample, predict_sample=predict_sample
            )
            self.beam_path = tf.identity(beam_path, name="beam_path")
            self.beam_result = batch_gather(self.item_cand, self.beam_path)

            _, greedy_path, _ = ptn_rnn_decoder(
                dec_cell, None,
                self.enc, enc_init_state,
                seq_length, res_length, hidden_units * 2,
                num_glimpse, self.batch_size,
                l1_mask, l2_mask, l3_mask, l0_ssr_mask,
                mode="GREEDY", reuse=True, beam_size=None,
                temperature=temperature,
                train_sample=train_sample, predict_sample=predict_sample
            )
            self.greedy_path = tf.identity(greedy_path, name="greedy_path")
            self.greedy_result = batch_gather(self.item_cand, self.greedy_path)

        if is_training:
            # Loss
            # self.y_smoothed = label_smoothing(tf.one_hot(self.decode_target_ids, depth=hp.data_length))
            self.r_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dec_logits,
                                                                         labels=self.decode_target_ids)
            # reinforcement
            self.policy_loss = tf.reduce_mean(tf.reduce_sum(self.r_loss, axis=1) * self.reward)
            # supervised loss
            self.loss = self.policy_loss

            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.variables = tf.global_variables()


class Discriminator:
    def __init__(self, lr=0.005, seq_length=500):
        self.user = tf.placeholder(tf.float32, shape=(None, 256), name='user')
        self.batch_size = tf.shape(self.user)[0]
        self.item_cand = tf.placeholder(tf.int32, shape=(None, seq_length), name='item_cand')

        self.reward_target = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")  # [batch_size]

        dense0 = self.user
        dense1 = tf.layers.dense(dense0, 128, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, 128, activation=tf.nn.relu)
        dense3 = tf.layers.dense(dense2, 128, activation=tf.nn.relu)

        self.reward = tf.layers.dense(dense3, 1)[:, 0]

        self.td_error = tf.abs(self.reward_target - self.reward)
        self.loss = tf.square(self.td_error)

        # Training Scheme
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
