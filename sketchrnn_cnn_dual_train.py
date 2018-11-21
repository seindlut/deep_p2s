"""Model training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from cStringIO import StringIO
import json
import os
import sys
import time
import urllib
import zipfile

# internal imports

import numpy as np
import tensorflow as tf

from model import sample, get_init_fn, init_vars, init_dis_vars, get_init_fn_with_sep_models
import model as sketch_rnn_model
import utils
import data_work
import cv2


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir', './data', 'The root directory for the data')
tf.app.flags.DEFINE_string('dataset', 'shoesv2', 'The dataset for classification')
tf.app.flags.DEFINE_string('log_root', './models/runs', 'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_float('lr', 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float('decay_rate', 0.9, "Learning rate decay for certain minibatches.")
tf.app.flags.DEFINE_float('decay_step', 5000, "Learning rate decay after how many training steps.")
tf.app.flags.DEFINE_boolean('lr_decay', False, "Learning rate decay.")
tf.app.flags.DEFINE_float('grad_clip', 1.0, 'Gradient clipping')
tf.app.flags.DEFINE_float('kl_weight_start', 0.01, "KL start weight when annealing.")
tf.app.flags.DEFINE_float('kl_decay_rate', 0.99995, "KL annealing decay rate per minibatch")
tf.app.flags.DEFINE_boolean('kl_weight_decay', False, "KL weight decay.")
tf.app.flags.DEFINE_boolean('kl_decay_step', 5000, "KL weight decay after how many steps.")
tf.app.flags.DEFINE_boolean('kl_tolerance', 0.2, "Level of KL loss at which to stop optimizing for KL.")
tf.app.flags.DEFINE_float('l2_weight_start', 0.1, "start weight for l2.")
tf.app.flags.DEFINE_float('l2_decay_rate', 0.99995, "l2 decay rate per minibatch")
tf.app.flags.DEFINE_boolean('l2_weight_decay', False, "l2 weight decay.")
tf.app.flags.DEFINE_boolean('l2_decay_step', 5000, "l2 weight decay after how many steps.")
tf.app.flags.DEFINE_integer('max_seq_len', 250, "max length of sequential data.")
tf.app.flags.DEFINE_float('seq_lw', 1.0, "Loss weight for sequence reconstruction.")
tf.app.flags.DEFINE_float('pix_lw', 1.0, "Loss weight for pixel reconstruction.")
tf.app.flags.DEFINE_boolean('tune_cnn', True, 'finetune the cnn part or not, this is trying to ')
tf.app.flags.DEFINE_string('vae_type', 'p2s', 'variational autoencoder type: s2s, sketch2sketch, p2s, photo2sketch, '
                                              'ps2s/sp2s, photo2sketch & sketch2sketch')
tf.app.flags.DEFINE_string('enc_type', 'cnn', 'type of encoder')
tf.app.flags.DEFINE_boolean('image_size', 256, 'image size for cnn')
tf.app.flags.DEFINE_boolean('crop_size', 224, 'crop size for cnn')
tf.app.flags.DEFINE_boolean('chn_size', 1, 'number of channel for cnn')
tf.app.flags.DEFINE_string('basenet', 'gen_cnn', 'basenet for cnn encoder')
tf.app.flags.DEFINE_float('margin', 0.1, 'Margin for contrastive/triplet loss')
tf.app.flags.DEFINE_boolean('load_pretrain', True, 'Load pretrain model for the p2s model')
tf.app.flags.DEFINE_boolean('resume_training', False, 'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string('load_dir', '', 'Directory to load the pretrained model')
tf.app.flags.DEFINE_string('img_dir', '', 'Directory to save the images')

# hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 100, 'Number of images to process in a batch.')
tf.app.flags.DEFINE_boolean('is_train', True, 'In the training stage or not')
tf.app.flags.DEFINE_float('drop_kp', 0.8, 'Dropout keep rate')

# data augmentation
tf.app.flags.DEFINE_boolean('flip_aug', False, 'Whether to flip the sketch and photo or not')
tf.app.flags.DEFINE_boolean('dist_aug', False, 'Whether to distort the images')
tf.app.flags.DEFINE_boolean('hp_filter', False, 'Whether to add high pass filter')

tf.app.flags.DEFINE_integer('k_dis', 5, 'train k_dis dis_op for 1 gen steps')
tf.app.flags.DEFINE_integer('k_gen', 3, 'train 1 dis_op among k_gen gen steps')
tf.app.flags.DEFINE_integer("max_steps", 100000, "Max training steps")
tf.app.flags.DEFINE_integer("print_every", 100, "print training loss after this many steps (default: 20)")
tf.app.flags.DEFINE_integer("save_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_boolean('debug_test', False, 'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_boolean('tee_log', True, 'Create log file to save the print info')
tf.app.flags.DEFINE_boolean('inter_z', False, 'Interpolate latent vector of batch z')
tf.app.flags.DEFINE_string("saved_flags", None, "Save all flags for printing")

# hyperparameters succeded from sketch-rnn
tf.app.flags.DEFINE_float('random_scale_factor', 0.15, 'Random scaling data augmention proportion.')
tf.app.flags.DEFINE_float('augment_stroke_prob', 0.10, 'Point dropping augmentation proportion.')
tf.app.flags.DEFINE_string('rnn_model', 'lstm', 'lstm, layer_norm or hyper.')
tf.app.flags.DEFINE_boolean('use_recurrent_dropout', True, 'Dropout with memory loss.')
tf.app.flags.DEFINE_float('recurrent_dropout_prob', 0.90, 'Probability of recurrent dropout keep.')
tf.app.flags.DEFINE_boolean('rnn_input_dropout', False, 'RNN input dropout.')
tf.app.flags.DEFINE_boolean('rnn_output_dropout', False, 'RNN output droput.')
tf.app.flags.DEFINE_integer('enc_rnn_size', 256, 'Size of RNN when used as encoder')
tf.app.flags.DEFINE_integer('dec_rnn_size', 512, 'Size of RNN when used as decoder')
tf.app.flags.DEFINE_integer('z_size', 128, 'Size of latent vector z')
tf.app.flags.DEFINE_integer('num_mixture', 20, 'Size of latent vector z')


def reset_graph():
    """Closes the current default session and resets the graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def load_model(model_dir):
    """Loads model for inference mode, used in jupyter notebook."""
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())

    model_params.batch_size = 1  # only sample one at a time
    eval_model_params = sketch_rnn_model.copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0
    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    sample_model_params.max_seq_len = 1  # sample one point at a time
    return [model_params, eval_model_params, sample_model_params]


def evaluate_model(sess, model, data_set):
    """Returns the average weighted cost, reconstruction cost and KL cost."""
    batch_size = data_set.batch_size
    sketch_size, photo_size = data_set.sketch_size, data_set.image_size
    dmat = np.zeros((photo_size, sketch_size))

    # extract sketch-photo score mat
    for idx in range(photo_size):
        sys.stdout.write('\r>> evaluation iter: (%d/%d)' % (idx, photo_size))
        y = data_set.image_data[np.repeat(idx, batch_size), ::]
        for idy in range(data_set.num_batches_for_sketch):
            s, x = data_set.get_sketch_batch(idy)
            start_idy = idy * batch_size
            end_idy = min(start_idy + batch_size, sketch_size)
            feed_dict = {
                model.input_sketch: x,
                model.sequence_lengths: s,
                model.input_pos_photo: y
                }

            dist_batch = sess.run([model.r_score], feed_dict)[0]
            dmat[idx, start_idy:end_idy] = dist_batch[:end_idy-start_idy]

    dmat = -np.transpose(dmat)

    acc1, acc10, _ = data_work.score_feat_eval_with_label(dmat, data_set.s_c_label, data_set.s_id_label, data_set.p_id_label)

    return acc1, acc10


def sampling_model(sess, model, gen_model, data_set, step, seq_len, subset_str=''):
    """Returns the average weighted cost, reconstruction cost and KL cost."""
    sketch_size, photo_size = data_set.sketch_size, data_set.image_size

    image_index = np.random.randint(0, photo_size)
    sketch_index = data_set.get_corr_sketch_id(image_index)
    gt_strokes = data_set.sketch_strokes[sketch_index]

    image_feat, rnn_enc_seq_len = data_set.get_input_image(image_index)
    sample_strokes, m = sample(sess, model, image_feat, seq_len=seq_len, rnn_enc_seq_len=rnn_enc_seq_len)
    strokes = utils.to_normal_strokes(sample_strokes)
    svg_gen_sketch = os.path.join(FLAGS.img_dir, '%s/%s/gensketch_for_photo%d_step%d.svg' % (data_set.dataset, subset_str, image_index, step))
    utils.draw_strokes(strokes, svg_filename=svg_gen_sketch)
    svg_gt_sketch = os.path.join(FLAGS.img_dir, '%s/%s/gt_sketch%d_for_photo%d.svg' % (data_set.dataset, subset_str, sketch_index, image_index))
    utils.draw_strokes(gt_strokes, svg_filename=svg_gt_sketch)
    input_sketch = data_set.pad_single_sketch(image_index)
    feed = {gen_model.input_sketch: input_sketch, gen_model.input_photo: image_feat, gen_model.sequence_lengths: [seq_len]}
    gen_photo = sess.run(gen_model.gen_photo, feed)
    gen_photo_file = os.path.join(FLAGS.img_dir, '%s/%s/gen_photo%d_step%d.png' % (data_set.dataset, subset_str, image_index, step))
    cv2.imwrite(gen_photo_file, cv2.cvtColor(gen_photo[0, ::].astype(np.uint8), cv2.COLOR_RGB2BGR))
    gt_photo = os.path.join(FLAGS.img_dir, '%s/%s/gt_photo%d.png' % (data_set.dataset, subset_str, image_index))
    if len(image_feat[0].shape) == 2:
        cv2.imwrite(gt_photo, image_feat[0])
    else:
        cv2.imwrite(gt_photo, cv2.cvtColor(image_feat[0].astype(np.uint8), cv2.COLOR_RGB2BGR))


def load_pretrain(sess, vae_type, enc_type, dataset, basenet, log_root):
    if vae_type in ['ps2s', 'sp2s'] or dataset in ['shoesv2', 'chairsv2']:
        if 'shoe' in dataset:
            sv_str = 'shoe'
        elif 'chair' in dataset:
            sv_str = 'chair'
        pretrain_dir = log_root.split('runs')[0] + 'pretrained_model/%s/' % sv_str
        ckpt = tf.train.get_checkpoint_state(pretrain_dir)
        if ckpt is not None:
            pretrained_model = ckpt.model_checkpoint_path
            print('Loading model %s.' % pretrained_model)
            checkpoint_exclude_scopes = []
            init_fn = get_init_fn(pretrained_model, checkpoint_exclude_scopes)
            init_fn(sess)
        else:
            print('Warning: pretrained model not found at %s' % pretrain_dir)
    else:
        if enc_type == 'cnn':
            init_vars(sess, FLAGS.log_root, basenet)


def resume_train(sess, load_dir, dataset, enc_type, basenet, feat_type, log_root):
    if not load_dir:
        if 'shoe' in dataset:
            sv_str = 'shoe'
        elif 'chair' in dataset:
            sv_str = 'chair'
        load_dir = log_root.split('runs')[0] + 'save_models/%s/' % sv_str
        # set dir to load the model for resume training
        load_dir = load_dir + basenet

    load_checkpoint(sess, load_dir)


def load_checkpoint(sess, checkpoint_path):

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt is None:
        raise Exception('Pretrained model not found at %s' % checkpoint_path)
    print('Loading model %s.' % ckpt.model_checkpoint_path)
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, saver, model_save_path, global_step):
    checkpoint_path = os.path.join(model_save_path, 'p2s')
    print('saving model %s at global_step %i.' % (checkpoint_path, global_step))
    saver.save(sess, checkpoint_path, global_step=global_step)


def train(sess, model, train_set):
    """Train a sketch-rnn model."""

    # print log
    if FLAGS.tee_log:
        utils.config_and_print_log(FLAGS.log_root, FLAGS)

    # set image dir
    FLAGS.img_dir = FLAGS.log_root.split('runs')[0] + 'sv_imgs/'

    # main train loop
    hps = model.hps
    start = time.time()

    # create saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    curr_step = sess.run(model.global_step)

    for step_id in range(curr_step, FLAGS.max_steps + FLAGS.save_every):

        step = sess.run(model.global_step)

        if hps.vae_type in ['p2s', 's2s']:
            s, a, p = train_set.random_batch()
            feed = {
                model.input_sketch: a,
                model.input_photo: p,
                model.sequence_lengths: s,
            }
        else:
            s, a, p, n = train_set.random_batch()
            feed = {
                model.input_sketch: a,
                model.input_photo: p,
                model.input_sketch_photo: n,
                model.sequence_lengths: s,
            }

        (train_cost, r_cost, kl_cost, p2s_r, p2s_kl, s2p_r, s2p_kl, p2p_r, p2p_kl, s2s_r, s2s_kl, _, train_step, _) = \
            sess.run([model.cost, model.r_cost, model.kl_cost, model.p2s_r, model.p2s_kl, model.s2p_r, model.s2p_kl,
                      model.p2p_r, model.p2p_kl, model.s2s_r, model.s2s_kl, model.final_state, model.global_step,
                      model.train_op], feed)

        if step % hps.print_every == 0 and step > 0:
            end = time.time()
            time_taken = end - start

            output_format = ('step: %d, ALL (cost: %.4f, recon: %.4f, kl: %.4f), P2S (recons: %.4f, kl: %.4f), '
                             'S2P (recons: %.4f, kl: %.4f), P2P (recons: %.4f, kl: %.4f), S2S (recons: %.4f, kl: %.4f), '
                             'train_time_taken: %.4f')
            output_values = (step, train_cost, r_cost, kl_cost, p2s_r, p2s_kl,
                             s2p_r, s2p_kl, p2p_r, p2p_kl, s2s_r, s2s_kl, time_taken)
            output_log = output_format % output_values

            print(output_log)

            start = time.time()

        if step % hps.save_every == 0 and step > 0:

            save_model(sess, saver, FLAGS.log_root, step)

    print("Finished training stage with %d steps" % FLAGS.max_steps)


def trainer(model_params):
    """Train a sketch-rnn model."""
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    print('Loading data files.')
    train_set, model_params = utils.load_dataset(FLAGS.root_dir, FLAGS.dataset, model_params)

    reset_graph()
    model = sketch_rnn_model.Model(model_params)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if FLAGS.load_pretrain:
        load_pretrain(sess, FLAGS.vae_type, FLAGS.enc_type, FLAGS.dataset, FLAGS.basenet, FLAGS.log_root)

    if FLAGS.resume_training:
        resume_train(sess, FLAGS.load_dir, FLAGS.dataset, FLAGS.enc_type, FLAGS.basenet, FLAGS.feat_type, FLAGS.log_root)

    train(sess, model, train_set)


def main(unused_argv):
    """Load model params, save config file and start trainer."""
    model_params = tf.contrib.training.HParams()
    # merge FLAGS to hps
    for attr, value in sorted(FLAGS.__flags.items()):
        model_params.add_hparam(attr, value)

    trainer(model_params)


if __name__ == '__main__':
    tf.app.run(main)
