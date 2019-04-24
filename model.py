# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sketch-RNN Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import rnn
import numpy as np
import tensorflow as tf

from subnet_tf_utils import generative_cnn_encoder, generative_cnn_decoder


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())


def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = tf.contrib.training.HParams(
        data_type='QMUL',  # 'QMUL' or 'QuickDraw'
        data_set=['shoes'],  # Our dataset.
        num_steps=10000000,  # Total number of steps of training. Keep large.
        batch_size=100,  # Minibatch size. Recommend leaving at 100.
        save_every=2000,  # Number of batches per checkpoint creation.
        max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
        dec_rnn_size=512,  # Size of decoder.
        dec_model='hyper',  # Decoder: lstm, layer_norm or hyper.
        enc_rnn_size=256,  # Size of encoder.
        enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
        z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.

        image_size=256,
        crop_size=224,
        resize_method='crop',  # 'crop' or 'scaling'
        pix_drop_kp=0.8,  # Dropout keep rate

        kl_weight=1.0,  # KL weight of loss equation. Recommend 0.5 or 1.0.
        kl_weight_start=0.01,  # KL start weight when annealing.
        kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL.
        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
        num_mixture=20,  # Number of mixtures in Gaussian mixture model.
        learning_rate=0.001,  # Learning rate.
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
        min_learning_rate=0.00001,  # Minimum learning rate.
        use_recurrent_dropout=True,  # Dropout with memory loss. Recommended
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
        use_input_dropout=False,  # Input dropout. Recommend leaving False.
        input_dropout_prob=0.90,  # Probability of input dropout keep.
        use_output_dropout=False,  # Output dropout. Recommend leaving False.
        output_dropout_prob=0.90,  # Probability of output dropout keep.
        random_scale_factor=0.15,  # Random scaling data augmentation proportion.
        augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
        is_training=True  # Is model training? Recommend keeping true.
    )
    return hparams


# below is where we need to do MDN (Mixture Density Network) splitting of
# distribution params
def get_mixture_coef(output):
    """Returns the tf slices containing mdn dist params."""
    # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
    z = output
    z_pen_logits = z[:, 0:3]  # pen states
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

    # process output z's into MDN parameters

    # softmax all the pi's and pen states:
    z_pi = tf.nn.softmax(z_pi)
    z_pen = tf.nn.softmax(z_pen_logits)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)

    r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
    return r


class Model(object):
    """Define a SketchRNN model."""

    def __init__(self, hps, gpu_mode=True, reuse=False):
        """Initializer for the SketchRNN model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
        self.hps = hps
        with tf.variable_scope('SCC', reuse=reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    print('Model using cpu.')
                    self.build_model()
            else:
                print('-' * 100)
                print('is_training:', hps.is_training)
                print('Model using gpu.')
                self.build_model()

    def build_model(self):
        """Define model architecture."""
        self.config_model()

        # First obtain the two z from pix_encoder and seq_encoder
        self.pix_h = self.build_pix_encoder(self.input_image)

        # Then for the 4 decoding branch
        self.build_pix2seq_branch(self.pix_h)

        if self.hps.is_training:
            self.seq_h = self.build_seq_encoder(self.output_x, self.sequence_lengths)  # last_h

            self.build_seq2pix_branch(self.seq_h)
            self.build_pix2pix_branch(self.pix_h, reuse=True)
            self.build_seq2seq_branch(self.seq_h, reuse=True)

            # Build losses
            self.build_losses()

            self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
            self.cost = self.r_cost_sum + self.kl_cost_sum * self.kl_weight

            self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.cost)
            g = self.hps.grad_clip
            capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(
                capped_gvs, global_step=self.global_step, name='train_step')

    def config_model(self):
        if self.hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.hps.dec_model == 'lstm':
            dec_cell_fn = rnn.LSTMCell
        elif self.hps.dec_model == 'layer_norm':
            dec_cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.dec_model == 'hyper':
            dec_cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        if self.hps.enc_model == 'lstm':
            enc_cell_fn = rnn.LSTMCell
        elif self.hps.enc_model == 'layer_norm':
            enc_cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.enc_model == 'hyper':
            enc_cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        use_recurrent_dropout = self.hps.use_recurrent_dropout
        use_input_dropout = self.hps.use_input_dropout
        use_output_dropout = self.hps.use_output_dropout

        dec_cell = dec_cell_fn(
            self.hps.dec_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

        self.enc_cell_fw = enc_cell_fn(
            self.hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            self.hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

        # dropout:
        print('Input dropout mode = %s.' % use_input_dropout)
        print('Output dropout mode = %s.' % use_output_dropout)
        print('Recurrent dropout mode = %s.' % use_recurrent_dropout)
        if use_input_dropout:
            print('Dropout to input w/ keep_prob = %4.4f.' % self.hps.input_dropout_prob)
            dec_cell = tf.contrib.rnn.DropoutWrapper(
                dec_cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            print('Dropout to output w/ keep_prob = %4.4f.' % self.hps.output_dropout_prob)
            dec_cell = tf.contrib.rnn.DropoutWrapper(
                dec_cell, output_keep_prob=self.hps.output_dropout_prob)
        self.dec_cell = dec_cell

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[self.hps.batch_size])
        self.input_sketch = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])
        self.input_photo = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.image_size, self.hps.image_size, 3])

        # The target/expected vectors of strokes
        self.output_x = self.input_sketch[:, 1:self.hps.max_seq_len + 1, :]  # [N, max_seq_len, 5]
        # vectors of strokes to be fed to decoder (same as above, but lagged behind
        # one step to include initial dummy value of (0, 0, 1, 0, 0))
        self.input_x = self.input_sketch[:, :self.hps.max_seq_len, :]  # [N, max_seq_len, 5]

        if self.hps.resize_method == 'crop':
            input_image = tf.random_crop(self.input_photo,
                                         size=[self.input_photo.shape[0], self.hps.crop_size,
                                               self.hps.crop_size, self.input_photo.shape[3]],
                                         name='input_random_crop')
        elif self.hps.resize_method == 'scaling':
            input_image = tf.image.resize_images(self.input_photo, (self.hps.crop_size, self.hps.crop_size))
        else:
            raise Exception('Unknown resize method', self.hps.resize_method)

        # Normalizing image
        input_image = tf.divide(input_image, 255.0)
        input_image = tf.multiply(input_image, 2.0)
        input_image = tf.subtract(input_image, 1.0)
        self.input_image = input_image  # [N, H, W, 3], [-1, 1]

    ###########################

    def build_pix2seq_branch(self, pix_h, reuse=False):
        output, last_state, mean, presig = self.build_seq_decoder(pix_h, 'p2s', reuse=reuse)
        self.rnn_output_p2s = output
        self.final_state_p2s = last_state
        self.mean_p2s = mean
        self.presig_p2s = presig

    def build_seq2pix_branch(self, seq_h, reuse=False):
        output, mean, presig = self.build_pix_decoder(seq_h, 's2p', reuse=reuse)
        self.gen_images_s2p = output
        self.mean_s2p = mean
        self.presig_s2p = presig

    def build_pix2pix_branch(self, pix_h, reuse=False):
        output, mean, presig = self.build_pix_decoder(pix_h, 'p2p', reuse=reuse)
        self.gen_images_p2p = output
        self.mean_p2p = mean
        self.presig_p2p = presig

    def build_seq2seq_branch(self, seq_h, reuse=False):
        output, last_state, mean, presig = self.build_seq_decoder(seq_h, 's2s', reuse=reuse)
        self.rnn_output_s2s = output
        self.final_state_s2s = last_state
        self.mean_s2s = mean
        self.presig_s2s = presig

    ###########################

    def build_pix_encoder(self, batch_input, reuse=False):
        if self.hps.is_training:
            is_training = True
            dropout_keep_prob = self.hps.pix_drop_kp
        else:
            is_training = False
            dropout_keep_prob = 1.0

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            image_embedding = generative_cnn_encoder(batch_input, is_training, dropout_keep_prob, reuse=reuse)
            return image_embedding

    def build_pix_decoder(self, encoded_h, name_scope, reuse=False):
        with tf.variable_scope(name_scope, reuse=False):
            batch_z, mean, presig = self.get_decoder_inputs(encoded_h, is_seq=False)

        if self.hps.is_training:
            is_training = True
            dropout_keep_prob = self.hps.pix_drop_kp
        else:
            is_training = False
            dropout_keep_prob = 1.0

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            output = generative_cnn_decoder(batch_z, is_training, dropout_keep_prob, reuse)

        return output, mean, presig

    def build_seq_encoder(self, input_strokes, sequence_lengths, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            strokes_embedding = self.rnn_encoder(input_strokes, sequence_lengths)

            return strokes_embedding

    def build_seq_decoder(self, encoded_h, name_scope, reuse=False):
        with tf.variable_scope(name_scope, reuse=False):
            batch_z, initial_state, actual_input_x, mean, presig = self.get_decoder_inputs(encoded_h, is_seq=True,
                                                                                           name_scope=name_scope)

        output, last_state = self.rnn_decoder(initial_state, actual_input_x, reuse)

        # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
        # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the mixture weight/probability (Pi_k)
        n_out = (3 + self.hps.num_mixture * 6)

        with tf.variable_scope('RNN_DEC_projection', reuse=reuse):
            output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
            output_b = tf.get_variable('output_b', [n_out])

        output = tf.reshape(output, [-1, self.hps.dec_rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)

        return output, last_state, mean, presig

    ###########################

    def get_decoder_inputs(self, encoded_h, is_seq=True, name_scope=None):
        mean, presig = self.get_mu_sig(encoded_h)
        sigma = tf.exp(presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
        eps = tf.random_normal((self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
        batch_z = mean + tf.multiply(sigma, eps)  # [N, z_size]

        if not is_seq:
            return batch_z, mean, presig

        pre_tile_y = tf.reshape(batch_z, [self.hps.batch_size, 1, self.hps.z_size])
        overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])  # [N, max_seq_len, z_size]
        actual_input_x = tf.concat([self.input_x, overlay_x], 2)

        initial_state = tf.nn.tanh(
            rnn.super_linear(
                batch_z,
                self.dec_cell.state_size,
                init_w='gaussian',
                weight_start=0.001,
                input_size=self.hps.z_size))

        if name_scope == 'p2s':
            # print('p2s seq decoder')
            self.initial_state_p2s = initial_state
            return batch_z, self.initial_state_p2s, actual_input_x, mean, presig
        elif name_scope == 's2s':
            # print('s2s seq decoder')
            self.initial_state_s2s = initial_state
            return batch_z, self.initial_state_s2s, actual_input_x, mean, presig
        else:
            raise Exception('Unknown name_scope', name_scope)

    def get_mu_sig(self, encoded_h):
        input_size = int(encoded_h.shape[-1])
        mu = rnn.super_linear(
            encoded_h,
            self.hps.z_size,
            input_size=input_size,
            scope='ENC_mu',
            init_w='gaussian',
            weight_start=0.001)
        presig = rnn.super_linear(
            encoded_h,
            self.hps.z_size,
            input_size=input_size,
            scope='ENC_sigma',
            init_w='gaussian',
            weight_start=0.001)
        return mu, presig

    ###########################

    def rnn_encoder(self, batch, sequence_lengths):
        """Define the bi-directional encoder module of sketch-rnn."""
        unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
            self.enc_cell_fw,
            self.enc_cell_bw,
            batch,
            sequence_length=sequence_lengths,
            time_major=False,
            swap_memory=True,
            dtype=tf.float32,
            scope='RNN_ENC')

        last_state_fw, last_state_bw = last_states
        last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
        last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
        last_h = tf.concat([last_h_fw, last_h_bw], 1)

        return last_h

    def rnn_decoder(self, initial_state, actual_input_x, reuse):
        with tf.variable_scope("RNN_DEC", reuse=reuse) as rnn_scope:
            output, last_state = tf.nn.dynamic_rnn(
                self.dec_cell,
                actual_input_x,
                initial_state=initial_state,
                time_major=False,
                swap_memory=True,
                dtype=tf.float32,
                scope=rnn_scope)
        return output, last_state

    ###########################

    def build_losses(self):
        # pix2seq_branch
        self.kl_cost_p2s = self.build_kl_loss(self.mean_p2s, self.presig_p2s)
        self.r_cost_p2s = self.build_seq_reconst_loss(self.rnn_output_p2s)

        # seq2pix_branch
        self.kl_cost_s2p = self.build_kl_loss(self.mean_s2p, self.presig_s2p)
        self.r_cost_s2p = self.build_pix_reconst_loss(self.input_image, self.gen_images_s2p)

        # pix2pix_branch
        self.kl_cost_p2p = self.build_kl_loss(self.mean_p2p, self.presig_p2p)
        self.r_cost_p2p = self.build_pix_reconst_loss(self.input_image, self.gen_images_p2p)

        # seq2seq_branch
        self.kl_cost_s2s = self.build_kl_loss(self.mean_s2s, self.presig_s2s)
        self.r_cost_s2s = self.build_seq_reconst_loss(self.rnn_output_s2s)

        # total
        self.kl_cost_sum = self.kl_cost_p2s + self.kl_cost_s2p + self.kl_cost_p2p + self.kl_cost_s2s
        self.r_cost_sum = self.r_cost_p2s + self.r_cost_s2p + self.r_cost_p2p + self.r_cost_s2s

    def build_kl_loss(self, mean, presig):
        kl_cost = -0.5 * tf.reduce_mean((1 + presig - tf.square(mean) - tf.exp(presig)))
        kl_cost = tf.maximum(kl_cost, self.hps.kl_tolerance)
        return kl_cost

    def build_pix_reconst_loss(self, real_images, gen_images):
        pixel_losses = tf.reduce_mean(tf.square(real_images - gen_images))
        return pixel_losses

    def build_seq_reconst_loss(self, rnn_output):

        # NB: the below are inner functions, not methods of Model
        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
            norm1 = tf.subtract(x1, mu1)
            norm2 = tf.subtract(x2, mu2)
            s1s2 = tf.multiply(s1, s2)
            # eq 25
            z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
                 2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
            neg_rho = 1 - tf.square(rho)
            result = tf.exp(tf.div(-z, 2 * neg_rho))
            denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
            result = tf.div(result, denom)
            return result

        def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                         z_pen_logits, x1_data, x2_data, pen_data):
            """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
            # This represents the L_R only (i.e. does not include the KL loss term).

            result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                                   z_corr)
            epsilon = 1e-6
            # result1 is the loss wrt pen offset (L_s in equation 9 of
            # https://arxiv.org/pdf/1704.03477.pdf)
            result1 = tf.multiply(result0, z_pi)
            result1 = tf.reduce_sum(result1, 1, keep_dims=True)
            result1 = -tf.log(result1 + epsilon)  # avoid log(0)

            fs = 1.0 - pen_data[:, 2]  # use training data for this
            fs = tf.reshape(fs, [-1, 1])
            # Zero out loss terms beyond N_s, the last actual stroke
            result1 = tf.multiply(result1, fs)

            # result2: loss wrt pen state, (L_p in equation 9)
            result2 = tf.nn.softmax_cross_entropy_with_logits(
                labels=pen_data, logits=z_pen_logits)
            result2 = tf.reshape(result2, [-1, 1])
            if not self.hps.is_training:  # eval mode, mask eos columns
                result2 = tf.multiply(result2, fs)

            result = result1 + result2
            return result

        out = get_mixture_coef(rnn_output)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

        # reshape target data so that it is compatible with prediction shape
        target = tf.reshape(self.output_x, [-1, 5])
        [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
        pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

        lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
                                o_pen_logits, x1_data, x2_data, pen_data)

        r_cost = tf.reduce_mean(lossfunc)
        return r_cost
