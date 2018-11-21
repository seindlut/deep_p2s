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

import cv2
import numpy as np
import requests
import tensorflow as tf

from model import sample, sample_recons, get_init_fn, init_vars
import model as sketch_rnn_model
import utils
import data_work

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('root_dir', './data', 'The root directory for the data')
# tf.app.flags.DEFINE_string('data_dir', data_dir, 'The directory to find the dataset')
# tf.app.flags.DEFINE_string('dataset', 'quickdraw', 'The dataset for classification')
# tf.app.flags.DEFINE_string('dataset', 'shoes', 'The dataset for classification')
tf.app.flags.DEFINE_string('dataset', 'shoesv2', 'The dataset for classification')
# tf.app.flags.DEFINE_string('dataset', 'quickdraw_shoe', 'The dataset for classification')
tf.app.flags.DEFINE_boolean('simplify_flag', True, 'use simplified dataset')
tf.app.flags.DEFINE_boolean('use_vae', True, 'use vae or ae only')
tf.app.flags.DEFINE_boolean('concat_z', True, 'concatenate z with x')
tf.app.flags.DEFINE_string('log_root', './models/runs', 'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_float('lr', 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float('decay_rate', 0.9999, "Learning rate decay for certain minibatches.")
tf.app.flags.DEFINE_boolean('lr_decay', False, "Learning rate decay.")
tf.app.flags.DEFINE_boolean('nkl', False, "if True, keep vae architecture but remove kl loss.")
tf.app.flags.DEFINE_float('kl_weight_start', 0.01, "KL start weight when annealing.")
tf.app.flags.DEFINE_float('kl_decay_rate', 0.99995, "KL annealing decay rate per minibatch")
tf.app.flags.DEFINE_boolean('kl_weight_decay', False, "KL weight decay.")
tf.app.flags.DEFINE_boolean('kl_tolerance', 0.2, "Level of KL loss at which to stop optimizing for KL.")
tf.app.flags.DEFINE_float('l2_weight_start', 0.1, "start weight for l2.")
tf.app.flags.DEFINE_float('l2_decay_rate', 0.99995, "l2 decay rate per minibatch")
tf.app.flags.DEFINE_boolean('l2_weight_decay', False, "l2 weight decay.")
tf.app.flags.DEFINE_boolean('l2_decay_step', 5000, "l2 weight decay after how many steps.")
tf.app.flags.DEFINE_integer('max_seq_len', 250, "max length of sequential data.")
tf.app.flags.DEFINE_float('seq_lw', 1.0, "Loss weight for sequence reconstruction.")
tf.app.flags.DEFINE_float('pix_lw', 1.0, "Loss weight for pixel reconstruction.")
tf.app.flags.DEFINE_float('tri_weight', 1.0, "Triplet loss weight.")
tf.app.flags.DEFINE_boolean('tune_cnn', True, 'finetune the cnn part or not, this is trying to ')
tf.app.flags.DEFINE_string('vae_type', 'p2s', 'variational autoencoder type: s2s, sketch2sketch, p2s, photo2sketch, '
                                              'ps2s/sp2s, photo2sketch & sketch2sketch')
tf.app.flags.DEFINE_string('enc_type', 'cnn', 'type of encoder')
tf.app.flags.DEFINE_string('rcons_type', 'mdn', 'type of reconstruction loss')
tf.app.flags.DEFINE_boolean('rd_dim', 512, 'embedding dim after mlp or other subnet')
# tf.app.flags.DEFINE_boolean('reduce_dim', False, 'add fc layer before the embedding loss')
tf.app.flags.DEFINE_boolean('image_size', 256, 'image size for cnn')
tf.app.flags.DEFINE_boolean('crop_size', 224, 'crop size for cnn')
tf.app.flags.DEFINE_boolean('chn_size', 1, 'number of channel for cnn')
tf.app.flags.DEFINE_string('basenet', 'gen_cnn', 'basenet for cnn encoder')
tf.app.flags.DEFINE_string('feat_type', 'inceptionv3', 'feature size for the extracted photo feature')
tf.app.flags.DEFINE_integer('feat_size', 2048, 'feature size for the extracted photo feature')
tf.app.flags.DEFINE_float('margin', 0.1, 'Margin for contrastive/triplet loss')
tf.app.flags.DEFINE_boolean('load_pretrain', False, 'Load pretrain model for CBB')
tf.app.flags.DEFINE_boolean('resume_training', True, 'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string('load_dir', '', 'Directory to load the pretrained model')
tf.app.flags.DEFINE_string('img_dir', '', 'Directory to save the images')
tf.app.flags.DEFINE_string('add_str', '', 'add str to the image save directory and checkpoints')

# hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 100, 'Number of images to process in a batch.')
tf.app.flags.DEFINE_boolean('is_train', False, 'In the training stage or not')
tf.app.flags.DEFINE_float('drop_kp', 1.0, 'Dropout keep rate')

# data augmentation
tf.app.flags.DEFINE_boolean('flip_aug', False, 'Whether to flip the sketch and photo or not')
tf.app.flags.DEFINE_boolean('dist_aug', False, 'Whether to distort the images')
tf.app.flags.DEFINE_boolean('hp_filter', False, 'Whether to add high pass filter')

tf.app.flags.DEFINE_integer("print_every", 100, "print training loss after this many steps (default: 20)")
tf.app.flags.DEFINE_integer("save_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_boolean('debug_test', False, 'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_boolean('use_jade', False, 'Locate data and model in the current dir')
tf.app.flags.DEFINE_string("saved_flags", None, "Save all flags for printing")
tf.app.flags.DEFINE_string('hparams', '',
                           'Pass in comma-separated key=value pairs such as \'save_every=40,decay_rate=0.99\''
                           '(no whitespace) to be read into the HParams object defined in model.py')

# save settings for sampling in testing stage
tf.app.flags.DEFINE_boolean('sample_sketch', True, 'Set to true to save ground truth sketch')
tf.app.flags.DEFINE_boolean('save_gt_sketch', True, 'Set to true to save ground truth sketch')
tf.app.flags.DEFINE_boolean('save_photo', False, 'Set to true to save ground truth photo')
tf.app.flags.DEFINE_boolean('cond_sketch', False, 'Set to true to generate sketch conditioned on sketch')
tf.app.flags.DEFINE_boolean('inter_z', False, 'Interpolate latent vector of batch z')
tf.app.flags.DEFINE_boolean('recon_sketch', False, 'Set to true to reconstruct sketch')
tf.app.flags.DEFINE_boolean('recon_photo', False, 'Set to true to reconstruct photo')

# hyperparameters succeded from sketch-rnn
tf.app.flags.DEFINE_float('random_scale_factor', 0.15, 'Random scaling data augmention proportion.')
tf.app.flags.DEFINE_float('augment_stroke_prob', 0.10, 'Point dropping augmentation proportion.')
tf.app.flags.DEFINE_string('rnn_model', 'lstm', 'lstm, layer_norm or hyper.')
tf.app.flags.DEFINE_boolean('use_recurrent_dropout', True, 'Dropout with memory loss.')
tf.app.flags.DEFINE_float('recurrent_dropout_prob', 1.0, 'Probability of recurrent dropout keep.')
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


def sampling_model_eval(sess, model, gen_model, data_set, seq_len):
    """Returns the average weighted cost, reconstruction cost and KL cost."""
    sketch_size, photo_size = data_set.sketch_size, data_set.image_size

    folders_to_create = ['gen_test', 'gen_test_png', 'gt_test', 'gt_test_png', 'gt_test_photo', 'gt_test_sketch_image',
                         'gen_test_s', 'gen_test_s_png', 'gen_test_inter', 'gen_test_inter_png', 'gen_test_inter_sep',
                         'gen_test_inter_sep_png', 'gen_photo', 'gen_test_inter_with_photo', 'recon_test',
                         'recon_test_png', 'recon_photo']
    for folder_to_create in folders_to_create:
        folder_path = os.path.join(FLAGS.img_dir, '%s/%s' % (data_set.dataset, folder_to_create))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    for image_index in range(photo_size):

        sys.stdout.write('\x1b[2K\r>> Sampling test set, [%d/%d]' % (image_index + 1, photo_size))
        sys.stdout.flush()

        image_feat, rnn_enc_seq_len = data_set.get_input_image(image_index)
        sample_strokes, m = sample(sess, model, image_feat, seq_len=seq_len, rnn_enc_seq_len=rnn_enc_seq_len)
        strokes = utils.to_normal_strokes(sample_strokes)
        svg_gen_sketch = os.path.join(FLAGS.img_dir, '%s/gen_test/gen_sketch%d.svg' % (data_set.dataset, image_index))
        png_gen_sketch = os.path.join(FLAGS.img_dir, '%s/gen_test_png/gen_sketch%d.png' % (data_set.dataset, image_index))
        utils.sv_svg_png_from_strokes(strokes, svg_filename=svg_gen_sketch, png_filename=png_gen_sketch)

    print("\nSampling finished")


def load_checkpoint(sess, checkpoint_path):

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt is None:
        raise Exception('Pretrained model not found at %s' % checkpoint_path)
    print('Loading model %s.' % ckpt.model_checkpoint_path)
    init_op = get_init_fn(checkpoint_path, [''], ckpt.model_checkpoint_path)
    init_op(sess)


def save_model(sess, model_save_path, global_step):
    saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'vector')
    print('saving model %s.' % checkpoint_path)
    print('global_step %i.' % global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)


def sample_test(sess, sample_model, gen_model, test_set, max_seq_len):

    # set image dir
    # FLAGS.img_dir = FLAGS.log_root.split('runs')[0] + 'sv_imgs/%s/' % FLAGS.dataset
    FLAGS.img_dir = FLAGS.log_root.split('runs')[0] + 'sv_imgs/'

    FLAGS.img_dir += 'dual/' + FLAGS.basenet

    sampling_model_eval(sess, sample_model, gen_model, test_set, max_seq_len, sample_model.hps.rcons_type)


def tester(model_params):
    """Test model."""
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    print('Hyperparams:')
    for key, val in model_params.values().iteritems():
        print('%s = %s' % (key, str(val)))
    print('Loading data files.')
    test_set, sample_model_params, gen_model_params  = utils.load_dataset(FLAGS.root_dir, FLAGS.dataset, model_params, inference_mode=True)

    reset_graph()
    sample_model = sketch_rnn_model.Model(sample_model_params)
    gen_model = sketch_rnn_model.Model(gen_model_params, reuse=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if FLAGS.dataset in ['shoesv2f_sup', 'shoesv2f_train']:
        dataset = 'shoesv2'
    else:
        dataset = FLAGS.dataset

    if FLAGS.resume_training:
        if FLAGS.load_dir == '':
            FLAGS.load_dir = FLAGS.log_root.split('runs')[0] + 'model_to_test/%s/' % dataset
            # set dir to load the model for testing
            FLAGS.load_dir = os.path.join(FLAGS.load_dir, FLAGS.basenet)
        load_checkpoint(sess, FLAGS.load_dir)

    # Write config file to json file.
    tf.gfile.MakeDirs(FLAGS.log_root)
    with tf.gfile.Open(
            os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)

    sample_test(sess, sample_model, gen_model, test_set, model_params.max_seq_len)


def main(unused_argv):
    """Load model params, save config file and start trainer."""
    model_params = tf.contrib.training.HParams()
    # merge FLAGS to hps
    for attr, value in sorted(FLAGS.__flags.items()):
        model_params.add_hparam(attr, value)

    tester(model_params)


if __name__ == '__main__':
    tf.app.run(main)