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
"""SketchRNN data loading and image manipulation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import random
import svgwrite
import numpy as np
import scipy.io as sio
import tensorflow as tf
from cairosvg import svg2png
import model as sketch_rnn_model


def load_hdf5(fname):
    hf = h5py.File(fname, 'r')
    d = {key: np.array(hf.get(key)) for key in hf.keys()}
    hf.close()
    return d


def get_sketch_data(fname):
    meta_data = load_hdf5(fname)
    return reassemble_data(meta_data['image_data'], meta_data['data_offset'])


def reassemble_data(concated_data, offsets):
    data = []
    num_items = offsets.shape[0]
    for index in range(num_items):
        data.append(concated_data[offsets[index, 0]:offsets[index,1]])
    return data


def config_and_print_log(log_dir, FLAGS):
    log_prefix = FLAGS.dataset.lower() + '_dual_' + '%s_' % FLAGS.basenet
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_prefix = os.path.join(log_dir, log_prefix)
    print_log(log_prefix, FLAGS)


def print_log(log_prefix, FLAGS):
    import sys
    import datetime
    try:
        from tabulate import tabulate
    except:
        print("tabulate lib not installed")
    # log direcory and file
    net_name = 'Deep-Photo-to-Sketch-Synthesis-Net'
    log_file = log_prefix + os.uname()[1].split('.')[0] + '_' + datetime.datetime.now().isoformat().split('.')[0].replace('-','_').replace(':', '_') + '_log.txt'
    f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, f)
    print("logging to ", log_file, "...")

    # print header
    print("===============================================")
    print("Trainning ", net_name, " in this framework")
    print("===============================================")

    print("Tensorflow flags:")

    flags_list = []
    for attr, value in sorted(FLAGS.__flags.items()):
        flags_list.append(attr)
    FLAGS.saved_flags = " ".join(flags_list)

    flag_table = {}
    flag_table['FLAG_NAME'] = []
    flag_table['Value'] = []
    flag_lists = FLAGS.saved_flags.split()
    # print self.FLAGS.__flags
    for attr in flag_lists:
        if attr not in ['saved_flags', 'net_name', 'log_root']:
            flag_table['FLAG_NAME'].append(attr.upper())
            flag_table['Value'].append(getattr(FLAGS, attr))
    flag_table['FLAG_NAME'].append('NET_NAME')
    flag_table['Value'].append(net_name)
    flag_table['FLAG_NAME'].append('HOST_NAME')
    flag_table['Value'].append(os.uname()[1].split('.')[0])
    try:
        print(tabulate(flag_table, headers="keys", tablefmt="fancy_grid").encode('utf-8'))
    except:
        for attr in flag_lists:
            print("attr name, ", attr.upper())
            print("attr value, ", getattr(FLAGS, attr))


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()


def get_skh_img_ids(image_id, instance_id):
    skh_img_ids = image_id
    list_image_id = image_id.tolist()
    unique_image_id = np.unique(image_id).tolist()
    # img_skh_ids = np.arange(len(instance_id))[instance_id == 0]
    img_skh_ids = [list_image_id.index(idx) for idx in unique_image_id]
    return skh_img_ids, img_skh_ids


def load_dataset(root_dir, dataset, model_params, inference_mode=False):
    """Loads the .npz file, and splits the set into train/valid/test."""

    # normalizes the x and y columns usint the training set.
    # applies same scaling factor to valid and test set.

    if dataset in ['shoesv2', 'chairsv2', 'shoesv2f_sup', 'shoesv2f_train']:
        data_dir = '%s/%s/' % (root_dir, dataset.split('v')[0])
    elif 'quickdraw' in str(dataset).lower():
        data_dir = os.path.join(root_dir, 'quickdraw')
    else:
        raise Exception('Dataset error')

    print(data_dir)

    sketch_data, sketch_png_data, image_data, sbir_data, skh_img_id, img_skh_id = {}, {}, {}, {}, {}, {}
    subset = 'train' if not inference_mode else 'test'

    rgb_str = ''

    view_type = 'photo'
    if dataset in ['shoesv2', 'chairsv2']:
        rgb_str = '_rgb'

    print('Prepared data for %s set' % subset)

    photo_png_dir = os.path.join(data_dir, '%s/%s%s.h5' % (view_type, subset, rgb_str))
    photo_png_dir_train = os.path.join(data_dir, '%s/train%s.h5' % (view_type, rgb_str))
    photo_png_dir_test = os.path.join(data_dir, '%s/test%s.h5' % (view_type, rgb_str))

    if 'quickdraw' in str(dataset).lower():
        sketch_data_dir = os.path.join(data_dir, 'npz_data/%s.npz' % category)
        sketch_png_dir_train = os.path.join(data_dir, 'hdf5_data/%s_train.h5' % category)
        sketch_png_dir_test = os.path.join(data_dir, 'hdf5_data/%s_test.h5' % category)

        # load data w/o label into dictionary
        # sketch_data[subset] = np.copy(np.load(sketch_data_dir)[subset])
        sketch_data['train'] = np.copy(np.load(sketch_data_dir)['train'])
        sketch_data['valid'] = np.copy(np.load(sketch_data_dir)['valid'])
        sketch_data['test'] = np.copy(np.load(sketch_data_dir)['test'])
        sketch_png_data['train'] = load_hdf5(sketch_png_dir_train)['image_data']
        sketch_png_data['test'] = load_hdf5(sketch_png_dir_test)['image_data']
        # image_data[subset] = None
        image_data['train'], image_data['test'] = None, None
        skh_img_id['train'], img_skh_id['train'] = None, None
        skh_img_id['test'], img_skh_id['test'] = None, None

    elif dataset in ['shoesv2', 'chairsv2']:

        sketch_data_dir_train = os.path.join(data_dir, 'svg_trimmed/train_svg_sim_spa_png.h5')
        sketch_data_dir_test = os.path.join(data_dir, 'svg_trimmed/test_svg_sim_spa_png.h5')

        # load data w/o label into dictionary
        # sketch_data[subset] = get_sketch_data(sketch_data_dir)
        sketch_data['train'] = get_sketch_data(sketch_data_dir_train)
        sketch_data['test'] = get_sketch_data(sketch_data_dir_test)
        sketch_data_info_train = load_hdf5(sketch_data_dir_train)
        sketch_data_info_test = load_hdf5(sketch_data_dir_test)
        sketch_png_data['train'] = sketch_data_info_train['png_data']
        sketch_png_data['test'] = sketch_data_info_test['png_data']
        # image_data[subset] = load_hdf5(photo_png_dir)['image_data']
        image_data['train'] = load_hdf5(photo_png_dir_train)['image_data']
        image_data['test'] = load_hdf5(photo_png_dir_test)['image_data']
        # skh_img_id[subset], img_skh_id[subset] = get_skh_img_ids(sketch_data_info['image_id'], sketch_data_info['instance_id'])
        skh_img_id['train'], img_skh_id['train'] = get_skh_img_ids(sketch_data_info_train['image_id'], sketch_data_info_train['instance_id'])
        skh_img_id['test'], img_skh_id['test'] = get_skh_img_ids(sketch_data_info_test['image_id'], sketch_data_info_test['instance_id'])

    if model_params.enc_type != 'cnn':
        # free the memory if not need
        # sketch_png_data[subset], image_data[subset] = None, None
        sketch_png_data['train'], image_data['train'] = None, None
        sketch_png_data['test'], image_data['test'] = None, None

    if sketch_data[subset] is not None:
        sketch_size = len(sketch_data[subset])
    else:
        sketch_size = 0
    if image_data[subset] is not None:
        photo_size = len(image_data[subset])
    else:
        photo_size = sketch_size

    print('Loaded {} set, {} sketches and {} images'.format(subset, sketch_size, photo_size))

    if 'quickdraw' in str(dataset).lower():
        all_strokes = np.concatenate((sketch_data['train'], sketch_data['valid'], sketch_data['test']))
    else:
        all_strokes = np.concatenate((sketch_data['train'], sketch_data['test']))
    num_points = 0
    for stroke in all_strokes:
        num_points += len(stroke)
    avg_len = num_points / len(all_strokes)
    print('Dataset train/valid/test, avg len {}'.format(int(avg_len)))

    # calculate the max strokes we need.
    max_seq_len = get_max_len(all_strokes)

    # overwrite the hps with this calculation.
    model_params.max_seq_len = max_seq_len
    model_params.rnn_enc_max_seq_len = max_seq_len

    print('model_params.max_seq_len %d' % model_params.max_seq_len)

    eval_model_params = sketch_rnn_model.copy_hparams(model_params)

    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    # eval_model_params.is_training = 1
    eval_model_params.is_training = 0

    if inference_mode:
        eval_model_params.batch_size = 1
        eval_model_params.is_training = 0

    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    sample_model_params.batch_size = 1  # only sample one at a time
    sample_model_params.max_seq_len = 1  # sample one point at a time

    gen_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    gen_model_params.batch_size = 1  # only sample one at a time

    print("Create DataLoader for %s subset" % subset)
    sbir_data['train_set'] = DataLoader(
        sketch_data['train'],
        sketch_png_data['train'],
        image_data['train'],
        skh_img_ids = skh_img_id['train'],
        img_skh_ids = img_skh_id['train'],
        dataset=dataset,
        enc_type=model_params.enc_type,
        vae_type=model_params.vae_type,
        batch_size=model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor,
        augment_stroke_prob=model_params.augment_stroke_prob,
        augment_flipr_flag=model_params.flip_aug)

    if inference_mode:
        sbir_data['test_set'] = DataLoader(
            sketch_data['test'],
            sketch_png_data['test'],
            image_data['test'],
            skh_img_ids=skh_img_id['test'],
            img_skh_ids=img_skh_id['test'],
            dataset=dataset,
            enc_type=model_params.enc_type,
            vae_type=model_params.vae_type,
            batch_size=model_params.batch_size,
            max_seq_length=model_params.max_seq_len,
            random_scale_factor=model_params.random_scale_factor,
            augment_stroke_prob=model_params.augment_stroke_prob,
            augment_flipr_flag=model_params.flip_aug)

    normalizing_scale_factor = sbir_data['train_set'].calculate_normalizing_scale_factor()

    sbir_data['%s_set' % subset].normalize(normalizing_scale_factor)

    print('normalizing_scale_factor %4.4f.' % normalizing_scale_factor)

    # return_model_params = eval_model_params if inference_mode else model_params

    # return sbir_data['%s_set' % subset], return_model_params
    if not inference_mode:
        return sbir_data['%s_set' % subset], model_params
    else:
        return sbir_data['%s_set' % subset], sample_model_params, gen_model_params


def get_bounds(data, factor=10):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def slerp(p0, p1, t):
    """Spherical interpolation."""
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
    """Linear interpolation."""
    return (1.0 - t) * p0 + t * p1


# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.
def strokes_to_lines(strokes):
    """Convert stroke-3 format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def lines_to_strokes(lines):
    """Convert polyline format to stroke-3 format."""
    eos = 0
    strokes = [[0, 0, 0]]
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)


def scale_bound(stroke, average_dimension=10.0):
    """Scale an entire image to be less than a certain size."""
    # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
    # modifies stroke directly.
    bounds = get_bounds(stroke, 1)
    max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def clean_strokes(sample_strokes, factor=100):
    """Cut irrelevant end points, scale to pixel space and store as integer."""
    # Useful function for exporting data to .json format.
    copy_stroke = []
    added_final = False
    for j in range(len(sample_strokes)):
        finish_flag = int(sample_strokes[j][4])
        if finish_flag == 0:
            copy_stroke.append([
                int(round(sample_strokes[j][0] * factor)),
                int(round(sample_strokes[j][1] * factor)),
                int(sample_strokes[j][2]),
                int(sample_strokes[j][3]), finish_flag
            ])
        else:
            copy_stroke.append([0, 0, 0, 0, 1])
            added_final = True
            break
    if not added_final:
        copy_stroke.append([0, 0, 0, 0, 1])
    return copy_stroke


def to_big_strokes(stroke, max_len=250):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).

    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result


def draw_strokes(data, factor=0.2, svg_filename='./sample.svg'):
    # little function that displays vector images and saves them to .svg
    tf.gfile.MakeDirs(os.path.dirname(svg_filename))
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in xrange(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    # display(SVG(dwg.tostring()))
    # png_filename = svg_filename.split('.svg')[0] + '.png'
    # svg2png(bytestring=dwg.tostring(),write_to=png_filename)


def sv_svg_png_from_strokes(data, svg_filename, png_filename, im_height=256, aspect_ratio=1, margin=0.1):
    # little function that displays vector images and saves them to .svg
    im_width = im_height * aspect_ratio
    dims = (im_width, im_height)
    min_x, max_x, min_y, max_y = get_bounds(data, factor=1)
    bound_x = max_x - min_x
    bound_y = max_y - min_y
    # bound = max(bound_x, bound_y)
    # factor = bound / ((1 - 2 * margin) * im_size)
    factor_x = bound_x / ((1 - 2 * margin) * im_width)
    factor_y = bound_y / ((1 - 2 * margin) * im_height)
    factor = max(factor_x, factor_y)
    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
    lift_pen = 1
    abs_x = (im_width - max_x/factor - min_x/factor) / 2
    abs_y = (im_height - max_y/factor - min_y/factor) / 2
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in xrange(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "
    the_color = "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
    dwg.save()
    # display(SVG(dwg.tostring()))
    svg2png(bytestring=dwg.tostring(),write_to=png_filename)
    # try:
    #     svg2png(bytestring=dwg.tostring(),write_to=png_filename)
    # except:
    #     print("Cannot save png file: %s" % png_filename)


def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max + x_min) * 0.5
        return x_start - center_loc, x_end

    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        grid_loc = sample[1]
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]
    return np.array(result)


def test_flip_dist(sess, model, sample_model, hps, feed, s, a, p, n):
    tf_images = sess.run(model.tf_images, feed)
    import matplotlib.pyplot as plt
    if hps.vae_type == 'p2s':
        orig_sketch_png = np.array(p[0,::], np.uint8)
    else:
        orig_sketch_png = np.array(np.stack([n[0,::], n[0,::], n[0,::]], axis=-1), np.uint8)
    if hps.basenet in ['inceptionv1', 'inceptionv3']:
        dist_sketch_png = np.array((tf_images[0,::]+1)*255/2, np.uint8)
    elif hps.basenet in ['sketchanet']:
        dist_sketch_png = np.array(tf_images[0,::]+250.42, np.uint8)
    else:
        raise Exception('basenet type error')
    import cv2
    cv2.imwrite('./orig_image.png', orig_sketch_png)
    cv2.imwrite('./dist_image.png', dist_sketch_png)

    import pdb
    pdb.set_trace()

    flipr_stroke = a[0, :, :4]
    flipr_stroke[:, 2] = a[0, :, 3]
    flipr_stroke[:, 2] += a[0, :, 4]
    flipr_stroke[:, 3] = a[0, :, 4]
    sv_svg_png_from_strokes(flipr_stroke, svg_filename='./flipr_stroke.svg', png_filename='./flipr_stroke.png')
    gen_strokes = sess.run(model.gen_strokes, feed)
    sv_svg_png_from_strokes(gen_strokes[0, :s[0], :], svg_filename='./gen_stroke.svg', png_filename='./gen_stroke.png')
    tar_strokes = sess.run(model.target_strokes, feed)
    sv_svg_png_from_strokes(tar_strokes[0, :s[0], :], svg_filename='./tar_stroke.svg', png_filename='./tar_stroke.png')
    # sampel via the eval model
    from model import sample
    sample_strokes, m = sample(sess, sample_model, p[:1, ::], seq_len=hps.max_seq_len, greedy_mode=True)
    sample_strokes_normal = to_normal_strokes(sample_strokes)
    sv_svg_png_from_strokes(sample_strokes_normal, svg_filename='./sample_stroke.svg', png_filename='./sample_stroke.png')


class DataLoader(object):
    """Class for loading data."""

    def __init__(self,
                 sketch_strokes,
                 sketch_data,
                 image_data,
                 skh_img_ids = None,
                 img_skh_ids = None,
                 dataset='shoesv2',
                 enc_type='cnn',
                 vae_type='p2s',
                 batch_size=100,
                 max_seq_length=250,
                 scale_factor=1.0,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 augment_flipr_flag=False,
                 limit=1000):
        self.batch_size = batch_size  # minibatch size
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.random_scale_factor = random_scale_factor  # data augmentation method
        # Removes large gaps in the data. x and y offsets are clamped to have
        # absolute value no greater than this limit.
        self.limit = limit
        self.dataset = dataset
        self.enc_type = enc_type
        self.vae_type = vae_type
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.augment_flipr_flag = augment_flipr_flag  # data augmentation method
        self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        self.convert_indices = False
        # sets self.sketch_strokes (list of ndarrays, one per sketch, in stroke-3 format,
        # sorted by size)
        self.preprocess_sketches(sketch_strokes, sketch_data)
        self.preprocess_images(image_data)
        self.preprocess_labels(skh_img_ids, img_skh_ids)
        self.data_size = len(self.sketch_strokes)
        self.num_batches = int(self.data_size / self.batch_size)
        self._assign_batches_according_to_vae_type()

    def preprocess_sketches(self, sketch_strokes, sketch_data):
        """Remove entries from strokes having > max_seq_length points."""
        raw_data = []
        seq_len = []
        count_data = 0
        self.sketch_strokes = []

        if sketch_strokes is not None and sketch_data is not None:
            for i in range(len(sketch_strokes)):
                data = sketch_strokes[i]
                if len(data) <= (self.max_seq_length):
                    count_data += 1
                    # removes large gaps from the data
                    data = np.minimum(data, self.limit)
                    data = np.maximum(data, -self.limit)
                    data = np.array(data, dtype=np.float32)
                    data[:, 0:2] /= self.scale_factor
                    raw_data.append(data)
                    seq_len.append(len(data))
            seq_len = np.array(seq_len)  # nstrokes for each sketch
            # idx = np.argsort(seq_len) # it is weird to sort the strokes here, just comment
            if len(sketch_strokes) != len(seq_len):
                print("Warning: some strokes has been removed")
            for i in range(len(seq_len)):
                # self.sketch_strokes.append(raw_data[idx[i]])
                self.sketch_strokes.append(raw_data[i])
            # print("total images <= max_seq_len is %d" % count_data)
            print("total images: %d" % count_data)

        self.sketch_size = count_data
        self.num_batches_for_sketch = int(np.ceil(self.sketch_size*1.0/self.batch_size))
        self.sketch_data = sketch_data

    def preprocess_images(self, image_data):
        self.image_data = image_data
        # crop images
        self.origin_size = 256
        self.crop_size, self.channel_size = 224, 3
        self.margin_size = self.origin_size - self.crop_size
        if self.image_data is None:
            print('Assume sketch and photo have same data size')
            self.image_size = self.sketch_size
        else:
            self.image_size = self.image_data.shape[0]

    def preprocess_labels(self, skh_img_ids, img_skh_ids):
        if skh_img_ids is not None and img_skh_ids is not None:
            self.convert_indices = True
        self.skh_img_ids, self.img_skh_ids = skh_img_ids, img_skh_ids

    def random_sample(self):
        """Return a random sample, in stroke-3 format as used by draw_strokes."""
        sample = np.copy(random.choice(self.sketch_strokes))
        return sample

    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(self.sketch_strokes)):
            if len(self.sketch_strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.sketch_strokes[i])):
                data.append(self.sketch_strokes[i][j, 0])
                data.append(self.sketch_strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.sketch_strokes)):
            self.sketch_strokes[i][:, 0:2] /= self.scale_factor

    def _get_sketch_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        x_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale(self.sketch_strokes[i])
            data_copy = np.copy(data)
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            x_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
        seq_len = np.array(seq_len, dtype=int)
        # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
        return seq_len, self.pad_batch(x_batch, self.max_seq_length)

    def _get_sketch_batch_from_indices_for_eval(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        x_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.sketch_strokes[i]
            data_copy = np.copy(data)
            x_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
        seq_len = np.array(seq_len, dtype=int)
        # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
        return seq_len, self.pad_batch(x_batch, self.max_seq_length)

    def _get_photo_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        return self.image_data[indices, ::]

    def _get_batch_from_indices_with_strokes_rnn(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        anc_batch_stroke_len, anc_batch_stroke = self._get_sketch_batch_from_indices(indices)

        # return sketch stroke with length and sketch bmp data batch.
        return anc_batch_stroke_len, anc_batch_stroke, anc_batch_stroke

    def _get_batch_from_indices_with_strokes_sketches(self, indices):
        """Given a list of indices, return the potentially augmented batch."""

        anc_batch_stroke_len, anc_batch_stroke = self._get_sketch_batch_from_indices(indices)
        sketch_batch = self.sketch_data[indices, ::]

        # return sketch stroke with length and sketch bmp data batch.
        return anc_batch_stroke_len, anc_batch_stroke, sketch_batch

    def _get_batch_from_indices_with_strokes_photos(self, indices):
        """Given a list of indices, return the potentially augmented batch."""

        anc_batch_stroke_len, anc_batch_stroke = self._get_sketch_batch_from_indices(indices)
        photo_batch = self._get_photo_batch_from_indices(indices)

        # augment sketch by random flip
        anc_batch_stroke, photo_batch = self.flip_aug_strokes_photos(anc_batch_stroke, photo_batch)

        # return sketch stroke with length and photo bmp data batch.
        return anc_batch_stroke_len, anc_batch_stroke, photo_batch

    def _get_batch_from_indices_with_strokes_photos_with_ids(self, indices):
        """Given a list of indices, return the potentially augmented batch."""

        anc_batch_stroke_len, anc_batch_stroke = self._get_sketch_batch_from_indices(indices)
        photo_batch = self._get_photo_batch_from_indices(self.skh_img_ids[indices])

        # augment sketch by random flip
        anc_batch_stroke, photo_batch = self.flip_aug_strokes_photos(anc_batch_stroke, photo_batch)

        # return sketch stroke with length and photo bmp data batch.
        return anc_batch_stroke_len, anc_batch_stroke, photo_batch

    def _get_batch_from_indices_with_strokes_photo_feats(self, indices):
        """Given a list of indices, return the potentially augmented batch."""

        anc_batch_stroke_len, anc_batch_stroke = self._get_sketch_batch_from_indices(indices)
        photo_batch = self._get_photo_feat_batch_from_indices(indices)

        # return sketch stroke with length and photo bmp data batch.
        return anc_batch_stroke_len, anc_batch_stroke, photo_batch

    def _get_batch_from_indices_with_strokes_photo_feats_with_ids(self, indices):
        """Given a list of indices, return the potentially augmented batch."""

        anc_batch_stroke_len, anc_batch_stroke = self._get_sketch_batch_from_indices(indices)
        photo_batch = self._get_photo_feat_batch_from_indices(self.skh_img_ids[indices])

        # return sketch stroke with length and photo bmp data batch.
        return anc_batch_stroke_len, anc_batch_stroke, photo_batch

    def _get_batch_from_indices_with_strokes_sketches_photos(self, indices):
        """Given a list of indices, return the potentially augmented batch."""

        anc_batch_stroke_len, anc_batch_stroke = self._get_sketch_batch_from_indices(indices)
        photo_batch = self._get_photo_batch_from_indices(indices)
        sketch_batch = self.sketch_data[indices, ::]

        # augment sketch by random flip
        anc_batch_stroke, photo_batch, sketch_batch = self.flip_aug_strokes_photos_sketches(anc_batch_stroke, photo_batch, sketch_batch)

        # return sketch stroke with length and photo bmp data batch.
        return anc_batch_stroke_len, anc_batch_stroke, photo_batch, sketch_batch

    def _get_batch_from_indices_with_strokes_sketches_photos_with_ids(self, indices):
        """Given a list of indices, return the potentially augmented batch."""

        anc_batch_stroke_len, anc_batch_stroke = self._get_sketch_batch_from_indices(indices)
        photo_batch = self._get_photo_batch_from_indices(self.skh_img_ids[indices])
        sketch_batch = self.sketch_data[indices, ::]

        # augment sketch by random flip
        anc_batch_stroke, photo_batch, sketch_batch = self.flip_aug_strokes_photos_sketches(anc_batch_stroke, photo_batch, sketch_batch)

        # return sketch stroke with length and photo bmp data batch.
        return anc_batch_stroke_len, anc_batch_stroke, photo_batch, sketch_batch

    def _assign_batches_according_to_vae_type(self):
        if self.enc_type == 'rnn':
            self._get_batch_from_indices = self._get_batch_from_indices_with_strokes_rnn
        elif self.enc_type == 'feat':
            if self.convert_indices:
                self._get_batch_from_indices = self._get_batch_from_indices_with_strokes_photo_feats_with_ids
            else:
                self._get_batch_from_indices = self._get_batch_from_indices_with_strokes_photo_feats
        else:
            if self.vae_type in ['p2s']:
                if self.convert_indices:
                    self._get_batch_from_indices = self._get_batch_from_indices_with_strokes_photos_with_ids
                else:
                    self._get_batch_from_indices = self._get_batch_from_indices_with_strokes_photos
            elif self.vae_type in ['s2s']:
                self._get_batch_from_indices = self._get_batch_from_indices_with_strokes_sketches
            else:
                if self.convert_indices:
                    self._get_batch_from_indices = self._get_batch_from_indices_with_strokes_sketches_photos_with_ids
                else:
                    self._get_batch_from_indices = self._get_batch_from_indices_with_strokes_sketches_photos

    def random_batch(self):
        """Return a randomised portion of the training data."""
        idx = np.random.permutation(range(0, self.data_size))[0:self.batch_size]
        return self._get_batch_from_indices(idx)

    def get_sketch_batch(self, idx):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0, "idx must be non negative"
        assert idx < self.num_batches_for_sketch, "idx must be less than the number of batches"
        start_idx = idx * self.batch_size
        indices = np.remainder(np.arange(start_idx, start_idx + self.batch_size), self.sketch_size)
        return self._get_sketch_batch_from_indices_for_eval(indices)

    def pad_batch(self, batch, max_len):
        """Pad the batch to be stroke-5 bigger format as described in paper."""
        result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        # t_result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        assert len(batch) == self.batch_size
        for i in range(self.batch_size):
            l = len(batch[i])
            assert l <= max_len
            result[i, 0:l, 0:2] = batch[i][:, 0:2]
            result[i, 0:l, 3] = batch[i][:, 2]
            result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
            result[i, l:, 4] = 1
            # put in the first token, as described in sketch-rnn methodology
            result[i, 1:, :] = result[i, :-1, :]
            result[i, 0, :] = 0
            result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
            result[i, 0, 3] = self.start_stroke_token[3]
            result[i, 0, 4] = self.start_stroke_token[4]
        # t_result[:, :-1, :] = result[:, 1:, :]
        # t_result[:, -1, 4] = 1
        return result

    def pad_single_sketch(self, index):
        max_len = self.max_seq_length
        sketch_strokes = self.sketch_strokes[index]
        """Pad the batch to be stroke-5 bigger format as described in paper."""
        result = np.zeros((1, max_len + 1, 5), dtype=float)
        # t_result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        l = len(sketch_strokes)
        assert l <= max_len
        result[0, 0:l, 0:2] = sketch_strokes[:, 0:2]
        result[0, 0:l, 3] = sketch_strokes[:, 2]
        result[0, 0:l, 2] = 1 - result[0, 0:l, 3]
        result[0, l:, 4] = 1
        # put in the first token, as described in sketch-rnn methodology
        result[0, 1:, :] = result[0, :-1, :]
        result[0, 0, :] = 0
        result[0, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
        result[0, 0, 3] = self.start_stroke_token[3]
        result[0, 0, 4] = self.start_stroke_token[4]
        # t_result[:, :-1, :] = result[:, 1:, :]
        # t_result[:, -1, 4] = 1
        return result

    def get_corr_sketch_id(self, photo_id):
        if self.dataset in ['shoes', 'chairs'] or 'quickdraw' in str(self.dataset.lower()):
            sketch_id = photo_id
        else:
            # one photo corresponding to a lot sketches, here we only select the first one
            sketch_id = self.img_skh_ids[photo_id]
        return sketch_id

    def get_corr_photo_id(self, sketch_id):
        if self.dataset in ['shoes', 'chairs'] or 'quickdraw' in str(self.dataset.lower()):
            photo_id = sketch_id
        else:
            # one photo corresponding to a lot sketches, here we only select the first one
            photo_id = self.skh_img_ids[sketch_id]
        return photo_id

    def get_input_image(self, index):
        rnn_enc_seq_len = None
        if self.enc_type == 'cnn':
            if self.image_data is None:
                image_data = self.sketch_data[index, ::][np.newaxis, :]
            else:
                image_data = self.image_data[index, ::][np.newaxis, :]
        elif self.enc_type == 'rnn':
            image_data = self.pad_single_sketch(index)
            rnn_enc_seq_len = [len(self.sketch_strokes[index])]
        else:
            raise Exception('Encode type error')

        return image_data, rnn_enc_seq_len

    def get_rgb_image(self, index):
        rgb_image = self.image_data[index, ::]
        if len(rgb_image.shape) == 2:
            rgb_image = np.repeat(rgb_image[:, :, np.newaxis], 3, axis=-1)
        return rgb_image

    def flip_aug_strokes_photos(self, stroke_batch, photo_batch):
        if self.augment_flipr_flag and random.random() >= 0.5:
            # print("Flip strokes and photo")
            stroke_batch[:, :, 0] = -stroke_batch[:, :, 0]
            photo_batch = photo_batch[:, :, ::-1, :]
        return stroke_batch, photo_batch

    def flip_aug_strokes_photos_sketches(self, stroke_batch, photo_batch, sketch_batch):
        if self.augment_flipr_flag and random.random() >= 0.5:
            # print("Flip strokes, photo and sketches")
            stroke_batch[:, :, 0] = -stroke_batch[:, :, 0]
            # photo_batch = photo_batch[:, :, ::-1, :]
            # sketch_batch = sketch_batch[:, :, ::-1]
            photo_batch = np.flip(photo_batch, 2)
            sketch_batch = sketch_batch[:, :, ::-1]
        return stroke_batch, photo_batch, sketch_batch