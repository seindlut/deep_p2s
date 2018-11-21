# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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

import os
import random

# internal imports

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_work import get_input_size
try:
    from magenta.models.sketch_rnn import rnn
except:
    import magenta_rnn as rnn

from build_subnet import *
from tf_data_work import *


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())


class Model(object):
    """Define a SketchRNN model."""

    def __init__(self, hps, reuse=False):
        """Initializer for the SketchRNN model.

        Args:
           hps: a HParams object containing model hyperparameters
           gpu_mode: a boolean that when True, uses GPU mode.
           reuse: a boolean that when true, attemps to reuse variables.
        """
        self.hps = hps
        with tf.variable_scope('vector_rnn', reuse=reuse):
            self.build_model(hps)

    def encoder(self, input_batch, sequence_lengths, reuse):
        if self.hps.enc_type == 'rnn':  # vae mode:
            image_embeddings = self.rnn_encoder(input_batch, sequence_lengths)
        elif self.hps.enc_type == 'cnn':
            image_embeddings = self.cnn_encoder(input_batch, reuse)
        elif self.hps.enc_type == 'feat':
            image_embeddings = input_batch
        else:
            raise Exception('Please choose a valid encoder type')
        return image_embeddings

    def rnn_encoder(self, batch, sequence_lengths):

        if self.hps.rnn_model == 'lstm':
            enc_cell_fn = rnn.LSTMCell
        elif self.hps.rnn_model == 'layer_norm':
            enc_cell_fn = rnn.LayerNormLSTMCell
        elif self.hps.rnn_model == 'hyper':
            enc_cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        if self.hps.rnn_model == 'hyper':
            self.enc_cell_fw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=self.hps.use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)
            self.enc_cell_bw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=self.hps.use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            self.enc_cell_fw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=self.hps.use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)
            self.enc_cell_bw = enc_cell_fn(
                self.hps.enc_rnn_size,
                use_recurrent_dropout=self.hps.use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)

        """Define the bi-directional encoder module of sketch-rnn."""
        unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
            self.enc_cell_fw,
            self.enc_cell_bw,
            batch,
            sequence_length=sequence_lengths,
            time_major=False,
            swap_memory=True,
            dtype=tf.float32,
            scope='ENC_RNN')

        last_state_fw, last_state_bw = last_states
        last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
        last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
        last_h = tf.concat([last_h_fw, last_h_bw], 1)
        return last_h

    def cnn_encoder(self, batch_input, reuse):
        if self.hps.is_train:
            is_train = True
            dropout_keep_prob = self.hps.drop_kp
        else:
            is_train = False
            dropout_keep_prob = 1.0
        tf_batch_input = tf_image_processing(batch_input, self.hps.basenet, self.hps.crop_size, self.hps.dist_aug, self.hps.hp_filter)
        self.tf_images = tf_batch_input
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            if self.hps.basenet == 'sketchanet':
                feature = sketch_a_net_slim(tf_batch_input)

            elif self.hps.basenet == 'gen_cnn':
                # feature = generative_cnn_encoder(tf_batch_input, is_train, dropout_keep_prob, reuse=reuse)
                feature = generative_cnn_encoder(tf_batch_input, True, dropout_keep_prob, reuse=reuse)

            elif FLAGS.basenet == 'alexnet':
                feature, end_points = tf_alexnet_single(tf_batch_input, dropout_keep_prob)

            elif FLAGS.basenet == 'vgg':
                _, feature = build_single_vggnet(tf_batch_input, is_train, dropout_keep_prob)

            elif FLAGS.basenet == 'resnet':
                print('Warning, resnet scope is not set')
                _, feature = build_single_resnet(tf_batch_input, is_train, name_scope='resnet_v1_50')

            elif FLAGS.basenet == 'inceptionv1':
                # _, feature = build_single_inceptionv1(tf_batch_input, is_train, dropout_keep_prob)
                _, feature = build_single_inceptionv1(tf_batch_input, True, dropout_keep_prob)
                # _, feature = build_single_inceptionv1(tf_batch_input, False, dropout_keep_prob)

            elif FLAGS.basenet == 'inceptionv3':
                # _, feature = build_single_inceptionv3(batch_input, is_train, dropout_keep_prob, reduce_dim=False)
                _, feature = build_single_inceptionv3(tf_batch_input, True, dropout_keep_prob, reduce_dim=False)
                # _, feature = build_single_inceptionv3(tf_batch_input, False, dropout_keep_prob, reduce_dim=False)

            else:
                raise Exception('basenet error')
            return feature

    def decoder(self, actual_input_x, initial_state, reuse):

        # decoder module of sketch-rnn is below
        with tf.variable_scope("RNN", reuse=reuse) as rnn_scope:
            output, last_state = tf.nn.dynamic_rnn(
                self.cell,
                actual_input_x,
                initial_state=initial_state,
                time_major=False,
                swap_memory=True,
                dtype=tf.float32,
                scope=rnn_scope)
        return output, last_state

    def cnn_decoder(self, z_input, reuse):

        if self.hps.is_train:
            is_train = True
            dropout_keep_prob = self.hps.drop_kp
        else:
            is_train = False
            dropout_keep_prob = 1.0

        output = generative_cnn_decoder(z_input, is_train, dropout_keep_prob, reuse)

        return output

    def get_mu_sig(self, image_embedding):
        enc_size = int(image_embedding.shape[-1])
        mu = rnn.super_linear(
            image_embedding,
            self.hps.z_size,
            input_size=enc_size,
            scope='ENC_RNN_mu',
            init_w='gaussian',
            weight_start=0.001)
        presig = rnn.super_linear(
            image_embedding,
            self.hps.z_size,
            input_size=enc_size,
            scope='ENC_RNN_sigma',
            init_w='gaussian',
            weight_start=0.001)
        return mu, presig

    def build_kl_for_vae(self, image_embedding, scope_name, with_state=True, reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            if with_state:
                return self.get_init_state(image_embedding)
            else:
                return self.get_kl_cost(image_embedding)

    def get_init_state(self, image_embedding):
        self.mean, self.presig = self.get_mu_sig(image_embedding)
        self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
        eps = tf.random_normal(
            (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
        # batch_z = self.mean + tf.multiply(self.sigma, eps)
        if self.hps.is_train:
            batch_z = self.mean + tf.multiply(self.sigma, eps)
        else:
            batch_z = self.mean
            if self.hps.inter_z:
                batch_z = self.mean + tf.multiply(self.sigma, self.sample_gussian)
        # KL cost
        kl_cost = -0.5 * tf.reduce_mean(
            (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
        kl_cost = tf.maximum(kl_cost, self.hps.kl_tolerance)

        # get initial state based on batch_z
        initial_state = tf.nn.tanh(
            rnn.super_linear(
                batch_z,
                self.cell.state_size,
                init_w='gaussian',
                weight_start=0.001,
                input_size=self.hps.z_size))
        pre_tile_y = tf.reshape(batch_z, [self.hps.batch_size, 1, self.hps.z_size])
        overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
        actual_input_x = tf.concat([self.input_x, overlay_x], 2)

        return initial_state, actual_input_x, batch_z, kl_cost

    def get_kl_cost(self, image_embedding):
        self.mean, self.presig = self.get_mu_sig(image_embedding)
        self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
        eps = tf.random_normal(
            (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
        batch_z = self.mean + tf.multiply(self.sigma, eps)
        # KL cost
        kl_cost = -0.5 * tf.reduce_mean(
            (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
        kl_cost = tf.maximum(kl_cost, self.hps.kl_tolerance)

        return batch_z, kl_cost

    def config_model(self, hps):
        """Define model architecture."""
        if hps.is_train:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            # self.global_step = tf.get_variable('global_step', trainable=False)

        if hps.rnn_model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif hps.rnn_model == 'layer_norm':
            cell_fn = rnn.LayerNormLSTMCell
        elif hps.rnn_model == 'hyper':
            cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        self.hps.crop_size, self.hps.chn_size = get_input_size()

        use_recurrent_dropout = self.hps.use_recurrent_dropout
        rnn_input_dropout = self.hps.rnn_input_dropout
        rnn_output_dropout = self.hps.rnn_output_dropout

        if hps.rnn_model == 'hyper':
            cell = cell_fn(
                hps.dec_rnn_size,
                use_recurrent_dropout=use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(
                hps.dec_rnn_size,
                use_recurrent_dropout=use_recurrent_dropout,
                dropout_keep_prob=self.hps.recurrent_dropout_prob)

        # dropout:
        if rnn_input_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
        if rnn_output_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
        self.cell = cell

        batch_size = self.hps.batch_size
        image_size = self.hps.image_size

        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[batch_size], name='seq_len')
        self.input_sketch = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, self.hps.max_seq_len + 1, 5], name='input_sketch')
        self.target_sketch = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, self.hps.max_seq_len + 1, 5], name='target_sketch')

        if self.hps.chn_size == 1:
            image_shape = [batch_size, image_size, image_size]
        else:
            image_shape = [batch_size, image_size, image_size, self.hps.chn_size]

        sketch_shape = [batch_size, image_size, image_size]

        if self.hps.vae_type == 's2s':
            self.input_photo = tf.placeholder(dtype=tf.float32, shape=sketch_shape, name='input_photo')
        else:
            self.input_photo = tf.placeholder(dtype=tf.float32, shape=image_shape, name='input_photo')
        if self.hps.vae_type in ['ps2s', 'sp2s']:
            self.input_sketch_photo = tf.placeholder(dtype=tf.float32, shape=sketch_shape, name='input_sketch_photo')

        self.input_label = tf.placeholder(dtype=tf.int32, shape=batch_size, name='label')

        if self.hps.inter_z:
            self.sample_gussian = tf.placeholder(dtype=tf.float32, shape=batch_size, name='sample_gussian')

        # The target/expected vectors of strokes
        self.output_x = self.input_sketch[:, 1:self.hps.max_seq_len + 1, :]
        # vectors of strokes to be fed to decoder (same as above, but lagged behind
        # one step to include initial dummy value of (0, 0, 1, 0, 0))
        self.input_x = self.input_sketch[:, :self.hps.max_seq_len, :]

    def build_pix_encoder(self, input_image, reuse=False):

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            image_embedding = self.cnn_encoder(input_image, reuse)

            return image_embedding

    def build_seq_encoder(self, input_strokes, reuse=False):

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            strokes_embedding = self.rnn_encoder(input_strokes, self.sequence_lengths)

            return strokes_embedding

    # ######################

    def build_seq_decoder(self, feat_embedding, kl_name_scope, reuse = False):

        initial_state, actual_input_x, batch_z, kl_cost = self.build_kl_for_vae(feat_embedding, kl_name_scope, reuse=False)
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            output, last_state = self.decoder(actual_input_x, initial_state, reuse)

            return output, initial_state, last_state, actual_input_x, batch_z, kl_cost

    def build_pix_decoder(self, feat_embedding, kl_name_scope, reuse = False):

        batch_z, kl_cost = self.build_kl_for_vae(feat_embedding, kl_name_scope, with_state=False, reuse=False)
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            output = self.cnn_decoder(batch_z, reuse)

            return output, batch_z, kl_cost

    def build_pix2seq_embedding(self, input_image, encode_pix=True, reuse = False):

        if encode_pix:
            self.pix_embedding = self.build_pix_encoder(input_image)

        return self.build_seq_decoder(self.pix_embedding, 'p2s', reuse=reuse)

    def build_seq2pix_embedding(self, input_strokes, encode_seq=True, reuse = False):

        if encode_seq:
            self.seq_embedding = self.build_seq_encoder(input_strokes)

        return self.build_pix_decoder(self.seq_embedding, 's2p', reuse=reuse)

    def build_pix2pix_embedding(self, input_image, encode_pix=True, reuse = False):

        if encode_pix:
            self.pix_embedding = self.build_pix_encoder(input_image)

        return self.build_pix_decoder(self.pix_embedding, 'p2p', reuse=reuse)

    def build_seq2seq_embedding(self, input_strokes, encode_seq=True, reuse=False):

        if encode_seq:
            self.seq_embedding = self.build_seq_encoder(input_strokes)

        return self.build_seq_decoder(self.seq_embedding, 's2s', reuse=reuse)

    def build_seq_loss(self, output, initial_state, final_state, batch_z, kl_cost, vae_type, reuse=False):
        # code for output, pi miu
        pi, mu1, mu2, sigma1, sigma2, corr, pen_logits, pen, y1_data, y2_data, r_cost, r_score, gen_strokes = self.build_strokes_rcons(output, reuse=reuse)
        # self.pen: pen state probabilities (result of applying softmax to self.pen_logits)

        r_cost = self.hps.seq_lw * r_cost

        cost_dict = {'rcons': r_cost, 'kl': kl_cost}

        end_points = {'init_s': initial_state, 'fin_s': final_state, 'pi': pi, 'mu1': mu1, 'mu2': mu2, 'sigma1': sigma1,
                      'sigma2': sigma2, 'corr': corr, 'pen': pen, 'batch_z': batch_z}

        cost_dict_vae_type = {'%s_%s' % (vae_type, key): cost_dict[key] for key in cost_dict.keys()}
        end_points_vae_type = {'%s_%s' % (vae_type, key): end_points[key] for key in end_points.keys()}

        return gen_strokes, cost_dict_vae_type, end_points_vae_type

    def build_pix_loss(self, gen_photo, batch_z, kl_cost, vae_type, reuse=False):
        r_cost = self.build_photo_rcons(self.target_photo, gen_photo)
        # self.pen: pen state probabilities (result of applying softmax to self.pen_logits)

        r_cost = self.hps.pix_lw * r_cost

        cost_dict = {'rcons': r_cost, 'kl': kl_cost}

        gen_photo_rgb = tf.cast((gen_photo + 1) * 127.5, tf.int16)

        end_points = {'gen_photo': gen_photo, 'gen_photo_rgb': gen_photo_rgb, 'batch_z': batch_z}

        cost_dict_vae_type = {'%s_%s' % (vae_type, key): cost_dict[key] for key in cost_dict.keys()}
        end_points_vae_type = {'%s_%s' % (vae_type, key): end_points[key] for key in end_points.keys()}

        return gen_photo_rgb, cost_dict_vae_type, end_points_vae_type

    def build_strokes_rcons(self, output, reuse=False):

        # target data
        x1_data, x2_data = self.x1_data, self.x2_data
        eos_data, eoc_data, cont_data = self.eos_data, self.eoc_data, self.cont_data

        # TODO(deck): Better understand this comment.
        # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
        # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the
        # mixture weight/probability (Pi_k)
        n_out = (3 + self.hps.num_mixture * 6)

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            with tf.variable_scope('RNN'):
                output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
                output_b = tf.get_variable('output_b', [n_out])

            output_reshape = tf.reshape(output, [-1, self.hps.dec_rnn_size])

        output_mdn = tf.nn.xw_plus_b(output_reshape, output_w, output_b)

        o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits = get_mixture_coef(output_mdn)

        o_id = tf.stack([tf.range(0, tf.shape(o_mu1)[0]), tf.cast(tf.argmax(o_pi, 1), tf.int32)], axis=1)
        y1_data = tf.gather_nd(o_mu1, o_id)
        y2_data = tf.gather_nd(o_mu2, o_id)

        y3_data = tf.cast(tf.greater(tf.argmax(o_pen_logits, 1), 0), tf.float32)
        gen_strokes = tf.stack([y1_data, y2_data, y3_data], 1)
        gen_strokes = tf.reshape(gen_strokes, [self.hps.batch_size, -1, 3])
        start_points_np = np.zeros((self.hps.batch_size, 1, 3))
        start_points_tf = tf.constant(start_points_np, dtype=tf.float32)
        gen_strokes = tf.concat([start_points_tf, gen_strokes], 1)

        pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

        rcons_loss = get_rcons_loss_pen_state(o_pen_logits, pen_data, self.hps.is_train)

        rcons_loss += get_rcons_loss_mdn(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x1_data, x2_data, pen_data)

        r_cost = tf.reduce_mean(rcons_loss)
        r_score = -tf.reduce_sum(tf.reshape(rcons_loss, [self.hps.batch_size, -1]), axis=1)
        return o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits, o_pen, y1_data, y2_data, r_cost, r_score, gen_strokes

    def build_photo_rcons(self, real_images, output_images):

        # target data
        image_shape = real_images.get_shape()
        output_images_reshaped = tf.reshape(output_images, image_shape)
        pixel_losses = tf.reduce_mean(tf.square(real_images - output_images_reshaped))
        return pixel_losses

    def build_l2_loss(self):

        self.rnn_l2 = tf.reduce_mean(tf.square(self.end_points['p2s_batch_z'] - self.end_points['s2s_batch_z']))
        self.cnn_l2 = tf.reduce_mean(tf.square(self.end_points['s2p_batch_z'] - self.end_points['p2p_batch_z']))

        self.l2_cost = self.rnn_l2 + self.cnn_l2

    def build_seq_discriminator(self, x, y, l, reuse):

        # set label for orig and gen
        label_r = tf.ones([self.hps.batch_size, 1], tf.int32)
        label_f = tf.zeros([self.hps.batch_size, 1], tf.int32)

        # build domain classifier
        cell_type, n_hidden, num_layers = self.hps.dis_model, self.hps.dis_num_hidden, self.hps.dis_num_layers
        in_dp, out_dp, batch_size = self.hps.dis_input_dropout, self.hps.dis_output_dropout, self.hps.batch_size
        pred_r, logits_r = rnn_discriminator(x, l, cell_type, n_hidden, num_layers, in_dp, out_dp, batch_size, reuse=reuse)
        pred_f, logits_f = rnn_discriminator(y, l, cell_type, n_hidden, num_layers, in_dp, out_dp, batch_size, reuse=True)

        # if self.hps.w_gan:
        #     dis_loss, gen_loss = wgan_gp_loss(logits_r, logits_f, None, use_gradients=False)
        #     dis_acc = tf.constant(-1.0)
        # else:
        dis_loss, gen_loss, dis_acc = get_adv_loss(logits_r, logits_f, label_r, label_f)

        dis_loss *= self.hps.rnn_dis_lw
        gen_loss *= self.hps.rnn_gen_lw

        return dis_loss, gen_loss, dis_acc

    def build_pix_discriminator(self, x, y, reuse):
        # set label for orig and gen
        label_r = tf.ones([self.hps.batch_size, 1], tf.int32)
        label_f = tf.zeros([self.hps.batch_size, 1], tf.int32)

        # build domain classifier
        batch_size = self.hps.batch_size
        pred_r, logits_r = cnn_discriminator(x, batch_size, reuse=reuse)
        pred_f, logits_f = cnn_discriminator(y, batch_size, reuse=True)

        if self.hps.gp_gan:
            alpha = tf.random_uniform(shape=[self.hps.batch_size, 1, 1, 1], minval=0., maxval=1.)
            differences = y - x
            interpolates = x + (alpha*differences)
            gradients = tf.gradients(cnn_discriminator(interpolates, batch_size, reuse=True)[1], [interpolates])[0]
            dis_loss, gen_loss, dis_acc = get_adv_gp_loss(logits_r, logits_f, label_r, label_f, gradients)
        else:
            dis_loss, gen_loss, dis_acc = get_adv_loss(logits_r, logits_f, label_r, label_f)

        dis_loss *= self.hps.cnn_dis_lw
        gen_loss *= self.hps.cnn_gen_lw

        return dis_loss, gen_loss, dis_acc

    def build_wgan_seq_discriminator(self, x, y, l, reuse):

        print("Build wgan seq discriminator")

        # build domain classifier
        logits_r = wgan_gp_rnn_discriminator(x, reuse=reuse)
        logits_f = wgan_gp_rnn_discriminator(y, reuse=True)

        alpha = tf.random_uniform(shape=[self.hps.batch_size, 1, 1], minval=0., maxval=1.)
        differences = y - x
        interpolates = x + (alpha*differences)
        gradients = tf.gradients(wgan_gp_rnn_discriminator(interpolates, reuse=True), [interpolates])[0]
        dis_loss, gen_loss = wgan_gp_loss(logits_f, logits_r, gradients)
        dis_acc = tf.constant(-1.0)

        dis_loss *= self.hps.rnn_dis_lw
        gen_loss *= self.hps.rnn_gen_lw

        return dis_loss, gen_loss, dis_acc

    def build_wgan_pix_discriminator(self, x, y, reuse):

        print("Build wgan pix discriminator")

        # build domain classifier
        logits_r = wgan_gp_cnn_discriminator(x, reuse=reuse)
        logits_f = wgan_gp_cnn_discriminator(y, reuse=True)

        alpha = tf.random_uniform(shape=[self.hps.batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = y - x
        interpolates = x + (alpha*differences)
        gradients = tf.gradients(wgan_gp_cnn_discriminator(interpolates, reuse=True), [interpolates])[0]
        dis_loss, gen_loss = wgan_gp_loss(logits_f, logits_r, gradients)
        dis_acc = tf.constant(-1.0)

        dis_loss *= self.hps.cnn_dis_lw
        gen_loss *= self.hps.cnn_gen_lw

        return dis_loss, gen_loss, dis_acc

    def get_train_vars(self):

        self.t_vars = tf.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
        self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]

    def get_train_op(self):

        self.apply_decay()

        # get total loss
        self.get_total_loss()

        # get train vars
        self.get_train_vars()

        optimizer = tf.train.AdamOptimizer(self.lr)
        gvs = optimizer.compute_gradients(self.cost)

        capped_gvs = clip_gradients(gvs, self.hps.grad_clip)
        self.train_op = optimizer.apply_gradients(
            capped_gvs, global_step=self.global_step, name='train_step')

    def apply_decay(self):
        if self.hps.lr_decay:
            self.lr = tf.train.exponential_decay(self.hps.lr, self.global_step, self.hps.decay_step, self.hps.decay_rate, staircase=True)
        else:
            self.lr = self.hps.lr

        # self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
        if self.hps.kl_weight_decay:
            self.kl_weight = tf.train.exponential_decay(self.hps.kl_weight_start, self.global_step, self.hps.kl_decay_step, self.hps.kl_decay_rate, staircase=True)
        else:
            self.kl_weight = self.hps.kl_weight_start

        if self.hps.l2_weight_decay:
            self.l2_weight = tf.train.exponential_decay(self.hps.l2_weight_start, self.global_step, self.hps.l2_decay_step, self.hps.l2_decay_rate, staircase=True)
        else:
            self.l2_weight = self.hps.l2_weight_start

    def get_total_loss(self):
        self.p2s_kl, self.s2p_kl = self.cost_dict['p2s_kl'], self.cost_dict['s2p_kl']
        self.p2p_kl, self.s2s_kl = self.cost_dict['p2p_kl'], self.cost_dict['s2s_kl']
        self.kl_cost = self.p2s_kl + self.s2p_kl + self.p2p_kl + self.s2s_kl
        self.cost = self.kl_cost * self.kl_weight

        # get reconstruction loss
        self.p2s_r, self.s2p_r = self.cost_dict['p2s_rcons'], self.cost_dict['s2p_rcons']
        self.p2p_r, self.s2s_r = self.cost_dict['p2p_rcons'], self.cost_dict['s2s_rcons']
        self.r_cost = self.p2s_r + self.s2p_r + self.p2p_r + self.s2s_r
        self.cost += self.r_cost

    def get_target_strokes(self):
        target = tf.reshape(self.output_x, [-1, 5])
        # reshape target data so that it is compatible with prediction shape
        [self.x1_data, self.x2_data, self.eos_data, self.eoc_data, self.cont_data] = tf.split(target, 5, 1)
        start_points_np = np.zeros((self.hps.batch_size, 1, 3))
        start_points_tf = tf.constant(start_points_np, dtype=tf.float32)
        self.target_strokes = tf.concat([self.x1_data, self.x2_data, 1 - self.eos_data], 1)
        self.target_strokes = tf.reshape(self.target_strokes, [self.hps.batch_size, -1, 3])
        self.target_strokes = tf.concat([start_points_tf, self.target_strokes], 1)

    def get_target_photo(self):
        self.target_photo = \
            tf_image_processing(self.input_photo, self.hps.basenet, self.hps.crop_size, self.hps.dist_aug, self.hps.hp_filter)

    def build_model(self, hps):
        self.config_model(hps)

        # get target data
        self.get_target_strokes()
        self.get_target_photo()

        # build photo to stroke-level synthesis part
        self.gen_strokes, cost_dict_p2s, end_points_p2s = self.build_pix2seq_branch(self.input_photo)

        # build stroke-level to photo synthesis part
        self.gen_photo, cost_dict_s2p, end_points_s2p = self.build_seq2pix_branch(self.input_sketch)

        # build photo to photo reconstruction part
        self.recon_photo, cost_dict_p2p, end_points_p2p = self.build_pix2pix_branch(self.input_photo, encode_pix=False, reuse=True)

        # build sketch to sketch reconstruction part
        self.recon_sketch, cost_dict_s2s, end_points_s2s = self.build_seq2seq_branch(self.input_sketch, encode_seq=False, reuse=True)

        self.cost_dict = dict(cost_dict_p2s.items() + cost_dict_s2p.items() + cost_dict_p2p.items() + cost_dict_s2s.items())
        self.end_points = dict(end_points_p2s.items() + end_points_s2p.items() + end_points_p2p.items() + end_points_s2s.items())

        self.initial_state, self.final_state = self.end_points['p2s_init_s'], self.end_points['p2s_fin_s']
        self.pi, self.corr = self.end_points['p2s_pi'], self.end_points['p2s_corr']
        self.mu1, self.mu2 = self.end_points['p2s_mu1'], self.end_points['p2s_mu2']
        self.sigma1, self.sigma2 = self.end_points['p2s_sigma1'], self.end_points['p2s_sigma2']
        self.pen = self.end_points['p2s_pen']
        self.batch_z = self.end_points['p2s_batch_z']

        self.recon_initial_state, self.recon_final_state = self.end_points['s2s_init_s'], self.end_points['s2s_fin_s']
        self.recon_pi, self.recon_corr = self.end_points['s2s_pi'], self.end_points['s2s_corr']
        self.recon_mu1, self.recon_mu2 = self.end_points['s2s_mu1'], self.end_points['s2s_mu2']
        self.recon_sigma1, self.recon_sigma2 = self.end_points['s2s_sigma1'], self.end_points['s2s_sigma2']
        self.recon_pen = self.end_points['s2s_pen']
        self.recon_batch_z = self.end_points['s2s_batch_z']

        if self.hps.is_train:
            # self.get_train_op_with_bn() # dosen't work
            self.get_train_op()

    def build_pix2seq_branch(self, input_photo, encode_pix=True, reuse=False):
        # pixel to sequence
        output, initial_state, final_state, actual_input_x, batch_z, kl_cost = \
            self.build_pix2seq_embedding(input_photo, encode_pix=encode_pix, reuse=reuse)

        return self.build_seq_loss(output, initial_state, final_state, batch_z, kl_cost, 'p2s', reuse=reuse)
    
    def build_seq2pix_branch(self, input_strokes, encode_seq=True, reuse=False):
        # sequence to pixel
        gen_photo, batch_z, kl_cost = self.build_seq2pix_embedding(input_strokes, encode_seq=encode_seq, reuse=reuse)

        return self.build_pix_loss(gen_photo, batch_z, kl_cost, 's2p', reuse=reuse)

    def build_pix2pix_branch(self, input_photo, encode_pix=False, reuse=False):
        # pixel to pixel
        gen_photo, batch_z, kl_cost = self.build_pix2pix_embedding(input_photo, encode_pix=encode_pix, reuse=reuse)

        return self.build_pix_loss(gen_photo, batch_z, kl_cost, 'p2p', reuse=reuse)

    def build_seq2seq_branch(self, input_strokes, encode_seq=False, reuse=False):
        output, initial_state, final_state, actual_input_x, batch_z, kl_cost = \
            self.build_seq2seq_embedding(input_strokes, encode_seq=encode_seq, reuse=reuse)

        return self.build_seq_loss(output, initial_state, final_state, batch_z, kl_cost, 's2s', reuse=reuse)


def get_pi_idx(pdf):
        """Samples from a pdf."""
        return np.argmax(pdf)


def sample(sess, model, input_image, sketch=None, seq_len=250, temperature=0.5, with_sketch=False, rnn_enc_seq_len = None, cond_sketch=False, inter_z=False, inter_z_sample=0):
    """Samples a sequence from a pre-trained model."""

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
    # print("enter the function of sample")

    if cond_sketch:
        if int(model.input_photo.get_shape()[-1]) == 3:
            input_image = input_image[:,:,:,np.newaxis]
            input_image = np.concatenate([input_image, input_image, input_image], -1)

    if rnn_enc_seq_len is None:
        if inter_z:
            prev_state = sess.run(model.initial_state, feed_dict={model.input_photo: input_image, model.sample_gussian: inter_z_sample})
        else:
            prev_state = sess.run(model.initial_state, feed_dict={model.input_photo: input_image})
        # image_embedding = sess.run(model.image_embedding, feed_dict={model.input_photo: input_image})
        # batch_z = sess.run(model.batch_z, feed_dict={model.input_photo: input_image})
    else:
        if inter_z:
            prev_state = sess.run(model.initial_state, feed_dict={model.input_photo: input_image, model.sequence_lengths: rnn_enc_seq_len, model.sample_gussian: inter_z_sample})
        else:
            prev_state = sess.run(model.initial_state, feed_dict={model.input_photo: input_image, model.sequence_lengths: rnn_enc_seq_len})
        # image_embedding = sess.run(model.image_embedding, feed_dict={model.input_photo: input_image, model.sequence_lengths: rnn_enc_seq_len})
        # batch_z = sess.run(model.batch_z, feed_dict={model.input_photo: input_image, model.sequence_lengths: rnn_enc_seq_len})

    strokes = np.zeros((seq_len, 5), dtype=np.float32)
    mixture_params = []

    for i in range(seq_len):
        if inter_z:
            feed = {
                model.input_x: prev_x,
                model.sequence_lengths: [1],
                model.initial_state: prev_state,
                model.input_photo: input_image,
                model.sample_gussian: inter_z_sample
            }
        else:
            feed = {
                model.input_x: prev_x,
                model.sequence_lengths: [1],
                model.initial_state: prev_state,
                model.input_photo: input_image
            }

        params = sess.run([
            model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
            model.pen, model.final_state
        ], feed)

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

        idx = get_pi_idx(o_pi[0])

        idx_eos = get_pi_idx(o_pen[0])
        eos = [0, 0, 0]
        eos[idx_eos] = 1

        next_x1, next_x2 = o_mu1[0][idx], o_mu2[0][idx]

        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

        params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_pen[0]]

        mixture_params.append(params)

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        if with_sketch:
            prev_x[0][0] = sketch[0][i+1]
        else:
            prev_x[0][0] = np.array(
                [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
        prev_state = next_state

    return strokes, mixture_params


def sample_recons(sess, model, gen_model, input_sketch, sketch=None, seq_len=250, temperature=0.5, with_sketch=False, cond_sketch=False, inter_z=False, inter_z_sample=0):
    """Samples a sequence from a pre-trained model."""

    prev_x = np.zeros((1, 1, 5), dtype=np.float32)
    prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
    # print("enter the function of sample")

    if inter_z:
        feed_dict = {
            gen_model.input_sketch: input_sketch,
            gen_model.sequence_lengths: [seq_len],
            gen_model.sample_gussian: inter_z_sample
        }
        prev_state, batch_z = sess.run([gen_model.recon_initial_state, gen_model.recon_batch_z], feed_dict=feed_dict)
    else:
        feed_dict = {
            gen_model.input_sketch: input_sketch,
            gen_model.sequence_lengths: [seq_len]
        }
        prev_state, batch_z = sess.run([gen_model.recon_initial_state, gen_model.recon_batch_z], feed_dict=feed_dict)

    strokes = np.zeros((seq_len, 5), dtype=np.float32)
    mixture_params = []

    for i in range(seq_len):
        if not model.hps.concat_z:
            feed = {
                model.input_x: prev_x,
                model.sequence_lengths: [1],
                model.recon_initial_state: prev_state
            }
        elif inter_z:
            feed = {
                model.input_x: prev_x,
                model.sequence_lengths: [1],
                model.recon_initial_state: prev_state,
                model.sample_gussian: inter_z_sample,
                model.recon_batch_z: batch_z
            }
        else:
            feed = {
                model.input_x: prev_x,
                model.sequence_lengths: [1],
                model.recon_initial_state: prev_state,
                model.recon_batch_z: batch_z
            }

        params = sess.run([
            model.recon_pi, model.recon_mu1, model.recon_mu2, model.recon_sigma1, model.recon_sigma2, model.recon_corr,
            model.recon_pen, model.recon_final_state
        ], feed)

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

        idx = get_pi_idx(o_pi[0])

        idx_eos = get_pi_idx(o_pen[0])
        eos = [0, 0, 0]
        eos[idx_eos] = 1

        next_x1, next_x2 = o_mu1[0][idx], o_mu2[0][idx]

        strokes[i, :] = [next_x1, next_x2, eos[0], eos[1], eos[2]]

        params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], o_pen[0]]

        mixture_params.append(params)

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        if with_sketch:
            prev_x[0][0] = sketch[0][i+1]
        else:
            prev_x[0][0] = np.array(
                [next_x1, next_x2, eos[0], eos[1], eos[2]], dtype=np.float32)
        prev_state = next_state

    return strokes, mixture_params


def get_init_fn(pretrain_model, checkpoint_exclude_scopes):
    """Returns a function run by the chief worker to warm-start the training."""
    print("load pretrained model from %s" % pretrain_model)
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    # for var in slim.get_model_variables():
    for var in tf.trainable_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            print(var.name)
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(pretrain_model, variables_to_restore)


def get_init_fn_rm(checkpoint_dir, checkpoint_exclude_scopes, prefix, pretrain_model):
    """Returns a function run by the chief worker to warm-start the training."""
    print("load pretrained model from %s and remove header %s" % (pretrain_model, prefix))
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    var_list = tf.trainable_variables()
    # remove the prefix to restore the pretrained inception model
    variables_to_restore = {var.name.split(prefix)[-1].split(':')[0]:var for var in var_list if prefix in var.name}
    variables_to_restore = pop_restore_keys(variables_to_restore, exclusions)

    saver = tf.train.Saver(variables_to_restore)

    def callback(session):
        saver.restore(session, os.path.join(checkpoint_dir, pretrain_model))
    return callback


def get_init_fn_with_sep_models(checkpoint_dir, checkpoint_exclude_scopes, p2s_pretrain_model, s2p_pretrain_model):
    """Returns a function run by the chief worker to warm-start the training."""
    print("load sep pretrained model from %s" % p2s_pretrain_model)
    print("load inv pretrained model from %s" % s2p_pretrain_model)
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    var_list = []
    # for var in slim.get_model_variables():
    for var in tf.trainable_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            print(var.name)
            var_list.append(var)

    p2s_var_list, s2p_var_list, other_var_list = [], [], []
    for var in var_list:
        if any(sub_str in var.name for sub_str in ['p2s', 's2p', 'p2p', 's2s']):
            other_var_list.append(var)
        elif any(sub_str in var.name for sub_str in ['vector_rnn/ENC_RNN', 'vector_rnn/CNN_DEC']):
            s2p_var_list.append(var)
        elif any(sub_str in var.name for sub_str in ['vector_rnn/CNN', 'vector_rnn/RNN']):
            p2s_var_list.append(var)
        else:
            other_var_list.append(var)

    p2s_saver = tf.train.Saver(p2s_var_list)
    s2p_saver = tf.train.Saver(s2p_var_list)

    def callback(session):
        p2s_saver.restore(session, os.path.join(checkpoint_dir, p2s_pretrain_model))
        s2p_saver.restore(session, os.path.join(checkpoint_dir, s2p_pretrain_model))
    return callback


def pop_restore_keys(variables_to_restore, exclusions):
    for key in variables_to_restore.keys():
        for exclusion in exclusions:
            if exclusion in key:
                variables_to_restore.pop(key, None)

    for key in variables_to_restore.keys():
        print("%s <== %s" % (variables_to_restore[key].name, key))

    return variables_to_restore


def init_vars(sess, model_prefix, basenet):
    # init model
    pre_trained_model_path = os.path.join(model_prefix.split('runs')[0], 'pretrained_model')
    if basenet == 'sketchanet':
        init_model_name = os.path.join(pre_trained_model_path, 'sketchnet_init.npy')
        # init_model_name = os.path.join(pre_trained_model_path, 'step1.npy')
        init_ops = init_variables(init_model_name)
        sess.run(init_ops)
    elif basenet == 'inceptionv1':
        init_fn = get_init_fn_rm(pre_trained_model_path, ['linear/', 'RNN', 'Logits'], 'vector_rnn/', 'inception_v1.ckpt')
        init_fn(sess)
    elif basenet == 'inceptionv3':
        init_fn = get_init_fn_rm(pre_trained_model_path, ['linear/', 'RNN', 'Logits'], 'vector_rnn/', 'inception_v3.ckpt')
        init_fn(sess)
    else:
        raise Exception('Input file error')


def init_dis_vars(sess, model_prefix, model_name, include_scope_str, exclude_scopes, include_scope_str_ref):
    print("Initialising discriminator variables")
    model_path = os.path.join(model_prefix.split('runs')[0], 'pretrained_model/%s' % model_name)
    init_ops = load_npy_model(model_path, include_scope_str, exclude_scopes, include_scope_str_ref)
    sess.run(init_ops)
