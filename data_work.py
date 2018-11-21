import os
import numpy as np
import scipy.io as sio
import re
import pickle

import itertools
from collections import Counter
from tensorflow.contrib import learn
from scipy.misc import imread
import random
import cv2
import scipy.spatial.distance as ssd
import sys
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import scipy

# from sbir_sampling import triplet_data_fetcher
# from sbir_data_util import *

import model

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def load_hdf5(fname):
    hf = h5py.File(fname, 'r')
    d = {key: np.array(hf.get(key)) for key in hf.keys()}
    hf.close()
    return d


def load_hdf5_uint8(fname):
    hf = h5py.File(fname, 'r')
    d = {key: np.array(hf.get(key), dtype=np.uint8) for key in hf.keys()}
    hf.close()
    return d


def multi_view_data(src_data, center_margin, crop_size, multi_view=True, crop_flag = 1):
    if len(src_data.shape) == 3:
        if src_data.shape[0] == 256 and src_data.shape[1] == 256:
            src_data = src_data[np.newaxis, :, :, :]
        else:
            src_data = src_data[:,:,:,np.newaxis]
    if multi_view:
        if crop_flag == 0:
            # 9 crop
            crop_xs = [0, 0, 0, 1, 1, 1, 2, 2, 2]
            crop_ys = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        elif crop_flag == 1:
            # 5 crop
            # crop_xs = [0, 0, 1, 2, 2]
            # crop_ys = [0, 2, 1, 0, 2]
            crop_xs = [0, center_margin + center_margin + 1, 0, center_margin + center_margin + 1, center_margin]
            crop_ys = [0, 0, center_margin + center_margin + 1, center_margin + center_margin + 1, center_margin]
            # crop_xs = crop_xs + crop_xs
            # crop_ys = crop_ys + crop_ys
            # crop_xs = [0, 2, 0, 2, 1, 0, 2, 0, 2, 1]
            # crop_ys = [0, 0, 2, 2, 1, 0, 0, 2, 2, 1]
        else:
            crop_xs = [0, 1, 1, 1, 2]
            crop_ys = [1, 0, 1, 2, 1]

        nums = src_data.shape[0]
        chns = src_data.shape[-1]
        repeat_size = len(crop_xs)*2
        # repeat_size = len(crop_xs)
        dst_data = np.zeros((repeat_size*nums, crop_size, crop_size, chns))
        index = 0
        for idx in xrange(nums):
            for crop_x, crop_y in zip(crop_xs, crop_ys):
                # crop_x = crop_x * center_margin
                # crop_y = crop_y * center_margin
                dst_data[index, ::] = src_data[idx, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
                dst_data[index+1, :, :, :] = dst_data[index, :, ::-1, :]
                index += 2
                # index += 1
        return dst_data
        # return dst_data[::-1, :, :, :]
    else:
        return src_data[:, center_margin:center_margin+crop_size, center_margin:center_margin+crop_size]


def train_batch_iter(sketch_data, image_data, trip_data, label_data, batch_size):
    """
    Generates a batch iterator for a dataset.
    """

    skh_data = sketch_data['train']
    skh_label, img_label = None, None
    if not image_data is None:
        img_data = image_data['train']
    if not label_data is None:
        if 'sketch' in label_data.keys():
            skh_label = label_data['sketch']['train']
        if 'sketch_id' in label_data.keys():
            skh_id = label_data['sketch_id']['train']
        if not image_data is None:
            if 'image' in label_data.keys():
                img_label = label_data['image']['train']
            if 'image_id' in label_data.keys():
                img_id = label_data['image_id']['train']
    if not trip_data is None:
        trips = trip_data['triplet']

    if FLAGS.dataset in ['shoes', 'chairs', 'handbags', 'shoesv2', 'chairsv2', 'sketchy', 'sch_com', 'step3', 'shoes1K', 'flickr15k']:
        if 'id' in FLAGS.loss_type:
            if FLAGS.dataset in ['sketchy', 'step3', 'flickr15k']:
                batches = tri_data_label_id_batch_iter(skh_data, img_data, skh_label, img_label, skh_id, img_id, trips, batch_size, FLAGS.num_epochs)
            elif FLAGS.dataset in ['shoesv2', 'chairsv2']:
                batches = tri_data_id_batch_iter_for_sbirv2(skh_data, img_data, skh_id, img_id, trips, batch_size, FLAGS.num_epochs)
            else:
                batches = trip_batch_id_iter(skh_data, img_data, skh_id, img_id, trips, batch_size, FLAGS.num_epochs)
        else:
            if FLAGS.use_triplet_sampler:
                queue_paras = model.get_input_paras()[2]
                data_fetcher = triplet_data_fetcher.TripletQueueRunner()
                data_fetcher.setup(skh_data, img_data, skh_label, img_label, trips, queue_paras)
                return data_fetcher
            else:
                if FLAGS.dataset in ['sketchy', 'step3', 'flickr15k']:
                    # batches = data_tid_batch_iter(skh_data, img_data, trips, hard_trips, easy_batch_size, hard_batch_size, 1000)
                    batches = tri_data_label_batch_iter(skh_data, img_data, skh_label, img_label, trips, batch_size, FLAGS.num_epochs)
                    # if 'ver' in FLAGS.loss_type:
                    #     batches = tri_data_label_ver_batch_iter(skh_data, img_data, skh_label, img_label, trips, batch_size, FLAGS.num_epochs)
                elif FLAGS.dataset == 'sch_com':
                    batches = trip_batch_iter_with_label(skh_data, img_data, skh_label, img_label, trips, batch_size, FLAGS.num_epochs)
                elif FLAGS.dataset in ['shoesv2', 'chairsv2']:
                    batches = tri_data_batch_iter_for_sbirv2(skh_data, img_data, trips, batch_size, FLAGS.num_epochs)
                else:
                    batches = trip_batch_iter(skh_data, img_data, trips, batch_size, FLAGS.num_epochs)
    elif FLAGS.dataset == 'TU-Berlin':
        batches = data_label_batch_iter(skh_data, skh_label, batch_size, FLAGS.num_epochs)
    else:
        raise Exception('Dataset type error')
    return batches


def get_input_size():
    if FLAGS.basenet == 'alexnet':
        crop_size = 227
        channel_size = 3
    elif FLAGS.basenet in ['resnet', 'inceptionv3']:
        crop_size = 299
        channel_size = 3
    elif FLAGS.basenet in ['sketchynet', 'inceptionv1', 'resnet', 'vgg', 'mobilenet', 'gen_cnn']:
        # assert FLAGS.image_type == 'rgb', "sketchynet use image as input"
        # FLAGS.image_type = 'rgb'
        crop_size = 224
        channel_size = 3
    else:
        crop_size = 225
        channel_size = 1
    return crop_size, channel_size


def trip_batch_iter(sketch_data, image_data, triplets,  batch_size, num_epochs=1000):
    """
    Generates a batch iterator for a dataset.
    """
    sketch_data_size = sketch_data.shape[0]
    if len(sketch_data.shape) == 3:
        sketch_data = sketch_data[:,:,:,np.newaxis]
        image_data = image_data[:,:,:,np.newaxis]

    # Generate triplet tuples
    pos_inds = triplets[::2, :].astype(np.int32) - np.min(triplets[:]).astype(np.int32)  # starts from 0
    neg_inds = triplets[1::2, :].astype(np.int32) - np.min(triplets[:]).astype(np.int32)  # starts from 0
    trip_width = pos_inds.shape[1]

    num_batches_per_epoch = int(np.ceil(sketch_data_size*1.0/batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch
    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # crop images
    origin_size = sketch_data.shape[2]
    crop_size, channel_size = get_input_size()
    margin_size = origin_size - crop_size

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(sketch_data_size))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            indices = np.remainder(np.arange(start_index, start_index + batch_size), sketch_data_size)
            indices = shuffle_indices[indices]

            anc_batch = sketch_data[indices, ::]
            pos_batch = image_data[pos_inds[indices,:][np.arange(batch_size)[:, None], np.random.choice(trip_width, batch_size)[:, None]][:, 0], ::]
            neg_batch = image_data[neg_inds[indices,:][np.arange(batch_size)[:, None], np.random.choice(trip_width, batch_size)[:, None]][:, 0], ::]

            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            anc_batch = random_flip(anc_batch)
            pos_batch = random_flip(pos_batch)
            neg_batch = random_flip(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch)


def trip_batch_id_iter(sketch_data, image_data, sketch_id, image_id, triplets,  batch_size, num_epochs=1000):
    """
    Generates a batch iterator for a dataset.
    """
    sketch_data_size = sketch_data.shape[0]
    if len(sketch_data.shape) == 3:
        sketch_data = sketch_data[:,:,:,np.newaxis]
        image_data = image_data[:,:,:,np.newaxis]

    # Generate triplet tuples
    pos_inds = triplets[::2, :].astype(np.int32) - np.min(triplets[:]).astype(np.int32)  # starts from 0
    neg_inds = triplets[1::2, :].astype(np.int32) - np.min(triplets[:]).astype(np.int32)  # starts from 0
    trip_width = pos_inds.shape[1]

    num_batches_per_epoch = int(np.ceil(sketch_data_size*1.0/batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch
    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # crop images
    origin_size = sketch_data.shape[2]
    crop_size, channel_size = get_input_size()
    margin_size = origin_size - crop_size

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(sketch_data_size))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            indices = np.remainder(np.arange(start_index, start_index + batch_size), sketch_data_size)
            indices = shuffle_indices[indices]

            anc_batch = sketch_data[indices, ::]
            pos_batch = image_data[pos_inds[indices,:][np.arange(batch_size)[:, None], np.random.choice(trip_width, batch_size)[:, None]][:, 0], ::]
            neg_batch = image_data[neg_inds[indices,:][np.arange(batch_size)[:, None], np.random.choice(trip_width, batch_size)[:, None]][:, 0], ::]

            anc_batch_id = sketch_id[indices]
            pos_batch_id = image_id[pos_inds[indices,:][np.arange(batch_size)[:, None], np.random.choice(trip_width, batch_size)[:, None]][:, 0]]
            neg_batch_id = image_id[neg_inds[indices,:][np.arange(batch_size)[:, None], np.random.choice(trip_width, batch_size)[:, None]][:, 0]]

            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            anc_batch = random_flip(anc_batch)
            pos_batch = random_flip(pos_batch)
            neg_batch = random_flip(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_id, pos_batch_id, neg_batch_id)


def trip_batch_iter_with_label(sketch_data, image_data, sketch_label, image_label, triplets,  batch_size, num_epochs=1000):
    """
    Generates a batch iterator for a dataset.
    """
    sketch_data_size = sketch_data.shape[0]
    if len(sketch_data.shape) == 3:
        sketch_data = sketch_data[:,:,:,np.newaxis]
        image_data = image_data[:,:,:,np.newaxis]

    # Generate triplet tuples
    pos_inds = triplets[::2, :].astype(np.int32) - np.min(triplets[:]).astype(np.int32)  # starts from 0
    neg_inds = triplets[1::2, :].astype(np.int32) - np.min(triplets[:]).astype(np.int32)  # starts from 0
    trip_width = pos_inds.shape[1]

    num_batches_per_epoch = int(np.ceil(sketch_data_size*1.0/batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch
    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # crop images
    origin_size = sketch_data.shape[2]
    crop_size, channel_size = get_input_size()
    margin_size = origin_size - crop_size

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(sketch_data_size))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            indices = np.remainder(np.arange(start_index, start_index + batch_size), sketch_data_size)
            indices = shuffle_indices[indices]
            pos_indices = pos_inds[indices,:][np.arange(batch_size)[:, None], np.random.choice(trip_width, batch_size)[:, None]][:, 0]
            neg_indices = neg_inds[indices,:][np.arange(batch_size)[:, None], np.random.choice(trip_width, batch_size)[:, None]][:, 0]

            anc_batch = sketch_data[indices, ::]
            pos_batch = image_data[pos_indices, ::]
            neg_batch = image_data[neg_indices, ::]

            anc_batch_label = sketch_label[indices]
            pos_batch_label = image_label[pos_indices]
            neg_batch_label = image_label[neg_indices]

            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            anc_batch = random_flip(anc_batch)
            pos_batch = random_flip(pos_batch)
            neg_batch = random_flip(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_label, pos_batch_label, neg_batch_label)


def trip_batch_iter_bk(sketch_data, image_data, triplets,  batch_size, num_epochs=1000, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    sketch_data_size = sketch_data.shape[0]
    trip_width = triplets.shape[1]
    if len(sketch_data.shape) == 3:
        sketch_data = sketch_data[:,:,:,np.newaxis]
        image_data = image_data[:,:,:,np.newaxis]
    # sketch_data = np.transpose(sketch_data, (1, 2, 3, 0))    # cb01 to b01c
    # image_data = np.transpose(image_data, (1, 2, 3, 0))    # cb01 to b01c
    # sketch_data = np.concatenate([sketch_data, sketch_data], axis=0)
    # image_data = np.concatenate([image_data, image_data], axis=0)
    start_index = 0

    # Generate triplet tuples
    triplet_data_size = sketch_data_size * trip_width
    trip_pairs = np.zeros((triplet_data_size, 3), np.int32)
    pos_inds = triplets[::2, :].astype(np.int32) - np.min(triplets[:])  # starts from 0
    neg_inds = triplets[1::2, :].astype(np.int32) - np.min(triplets[:])  # starts from 0
    trip_id = 0
    for idx in xrange(sketch_data_size):
        for idy in xrange(trip_width):
            trip_pairs[trip_id, 0] = idx                       # anc id
            trip_pairs[trip_id, 1] = pos_inds[idx, idy]        # pos id
            trip_pairs[trip_id, 2] = neg_inds[idx, idy]        # neg id
            trip_id += 1
    trip_pairs = np.concatenate([trip_pairs, trip_pairs])

    num_batches_per_epoch = int(np.ceil(triplet_data_size*1.0/batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    # crop images
    origin_size = sketch_data.shape[2]
    crop_size, channel_size = get_input_size()
    margin_size = origin_size - crop_size
    center_margin = abs(origin_size - crop_size - 1) / 2
    crop_rand = True

    crop_xs = [0, center_margin + center_margin + 1, 0, center_margin + center_margin + 1, center_margin]
    crop_ys = [0, 0, center_margin + center_margin + 1, center_margin + center_margin + 1, center_margin]

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        for batch_num in range(num_batches_per_epoch):
            if start_index >= triplet_data_size:
                start_index -= triplet_data_size
            end_index = start_index + batch_size
            indices = np.arange(start_index, end_index)
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(batch_size))
                indices = indices[shuffle_indices]

            crop_id = random.randint(0, 4)
            crop_x = crop_xs[crop_id]; crop_y = crop_ys[crop_id]
            # crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            # print "(crop_x, crop_y) = (", crop_x, " ,", crop_y, " )"
            # batch_anc = sketch_data[trip_pairs[indices, 0], crop_xs:crop_xs+crop_size, crop_ys:crop_ys+crop_size, :]
            # batch_pos = image_data[trip_pairs[indices, 1], crop_xs:crop_xs+crop_size, crop_ys:crop_ys+crop_size, :]
            # batch_neg = image_data[trip_pairs[indices, 2], crop_xs:crop_xs+crop_size, crop_ys:crop_ys+crop_size, :]

            batch_anc = sketch_data[trip_pairs[indices, 0], crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            batch_pos = image_data[trip_pairs[indices, 1], crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            batch_neg = image_data[trip_pairs[indices, 2], crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            if crop_rand and random.random() >= 0.5:
                # print "flip"
                batch_anc = batch_anc[:, :, ::-1, :]
                batch_pos = batch_pos[:, :, ::-1, :]
                batch_neg = batch_neg[:, :, ::-1, :]

            # batch_anc = sketch_data[trip_pairs[indices, 0], :, :, :]
            # batch_pos = image_data[trip_pairs[indices, 1], :, :, :]
            # batch_neg = image_data[trip_pairs[indices, 2], :, :, :]

            start_index += batch_size

            yield zip(batch_anc, batch_pos, batch_neg)


def pair_batch_iter_dev(sketch_data, image_data, batch_size, multi_view = True):
    """
    Generates a batch iterator for a dataset.
    """

    start_index = 0

    # crop images
    origin_size = image_data.shape[2]
    crop_size = 225
    if 'basenet' in FLAGS.__flags.keys():
        crop_size = get_input_size()[0]
    center_margin = abs(origin_size - crop_size - 1) / 2
    crop_flag = 1

    if multi_view:
        sketch_data = multi_view_data(sketch_data, center_margin, crop_size, multi_view, crop_flag)
        image_data = multi_view_data(image_data, center_margin, crop_size, multi_view, crop_flag)
    else:
        sketch_data = multi_view_data(sketch_data, center_margin, crop_size, False, crop_flag)
        image_data = multi_view_data(image_data, center_margin, crop_size, False, crop_flag)

    sketch_data_size = sketch_data.shape[0]

    num_batches_per_epoch = int(np.ceil(sketch_data_size*1.0/batch_size))
    # print "num_batches_in_val_set: ", num_batches_per_epoch

    for batch_num in range(num_batches_per_epoch):
        indices = np.remainder(np.arange(start_index, start_index + batch_size), sketch_data_size)

        sketch_batch = sketch_data[indices, ::]
        image_batch = image_data[indices, ::]

        start_index += batch_size

        # sketch_batch = remove_mean(sketch_batch)
        # image_batch = remove_mean(image_batch)

        yield zip(sketch_batch, image_batch)


def batch_iter_dev(image_data, batch_size, multi_view = True):
    """
    Generates a batch iterator for a dataset.
    """

    start_index = 0

    # crop images
    origin_size = image_data.shape[2]
    crop_size = 225
    if 'basenet' in FLAGS.__flags.keys():
        crop_size = get_input_size()[0]
    center_margin = abs(origin_size - crop_size - 1) / 2
    crop_flag = 1

    if multi_view:
        image_data = multi_view_data(image_data, center_margin, crop_size, multi_view, crop_flag)
    else:
        image_data = multi_view_data(image_data, center_margin, crop_size, False, crop_flag)

    image_data_size = image_data.shape[0]

    num_batches_per_epoch = int(np.ceil(image_data_size*1.0/batch_size))
    # print "num_batches_in_val_set: ", num_batches_per_epoch

    for batch_num in range(num_batches_per_epoch):
        indices = np.remainder(np.arange(start_index, start_index + batch_size), image_data_size)
        image_batch = image_data[indices, ::]
        start_index += batch_size
        yield image_batch


def batch_iter_dev_minibatch(image_data, batch_size, multi_view = True):
    """
    Generates a batch iterator for a dataset.
    """

    start_index = 0
    # crop images
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    center_margin = abs(origin_size - crop_size - 1) / 2
    crop_flag = 1
    if multi_view:
        num_views = 10
    else:
        num_views = 1

    image_data_size = image_data.shape[0]

    num_batches_per_epoch = int(np.ceil(image_data_size*1.0/batch_size))
    # print "num_batches_in_val_set: ", num_batches_per_epoch

    for batch_num in range(num_batches_per_epoch):
        # print 'batch_num', batch_num
        indices = np.remainder(np.arange(start_index, start_index + batch_size), image_data_size)
        image_batch = image_data[indices, ::]
        start_index += batch_size
        image_batch = remove_mean(image_batch)

        if multi_view:
            image_batch = multi_view_data(image_batch, center_margin, crop_size, multi_view, crop_flag)
        else:
            image_batch = multi_view_data(image_batch, center_margin, crop_size, False, crop_flag)
        for mini_batch_num in range(num_views):
            # print 'mini_batch_num', mini_batch_num
            mini_indices = np.arange(mini_batch_num*batch_size, mini_batch_num*batch_size + batch_size)
            image_mini_batch = image_batch[mini_indices, ::]

            yield image_mini_batch


def pair_batch_iter_dev_minibatch(sketch_data, image_data, batch_size, multi_view = True):
    """
    Generates a batch iterator for a dataset.
    """

    start_index = 0

    # crop images
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    center_margin = abs(origin_size - crop_size - 1) / 2
    crop_flag = 1
    if multi_view:
        num_views = 10
    else:
        num_views = 1

    sketch_data_size = sketch_data.shape[0]

    num_batches_per_epoch = int(np.ceil(sketch_data_size*1.0/batch_size))
    # print "num_batches_in_val_set: ", num_batches_per_epoch

    for batch_num in range(num_batches_per_epoch):
        # print 'batch_num', batch_num
        indices = np.remainder(np.arange(start_index, start_index + batch_size), sketch_data_size)

        sketch_batch = sketch_data[indices, ::]
        image_batch = image_data[indices, ::]

        start_index += batch_size

        sketch_batch = remove_mean(sketch_batch)
        image_batch = remove_mean(image_batch)

        if multi_view:
            sketch_batch = multi_view_data(sketch_batch, center_margin, crop_size, multi_view, crop_flag)
            image_batch = multi_view_data(image_batch, center_margin, crop_size, multi_view, crop_flag)
        else:
            sketch_batch = multi_view_data(sketch_batch, center_margin, crop_size, False, crop_flag)
            image_batch = multi_view_data(image_batch, center_margin, crop_size, False, crop_flag)
        for mini_batch_num in range(num_views):
            # print 'mini_batch_num', mini_batch_num
            mini_indices = np.arange(mini_batch_num*batch_size, mini_batch_num*batch_size + batch_size)
            sketch_mini_batch = sketch_batch[mini_indices, ::]
            image_mini_batch = image_batch[mini_indices, ::]

            yield zip(sketch_mini_batch, image_mini_batch)


def pair_batch_iter_dev_score_mini_batch(sketch_data, image_data,  batch_size, multi_view = False):
    """
    Generates a batch iterator for a dataset.
    """

    if multi_view:
        num_views = 10
    else:
        num_views = 1
    sketch_data_size = sketch_data.shape[0]
    image_data_size = image_data.shape[0]

    # crop images
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    center_margin = abs(origin_size - crop_size - 1) / 2
    crop_flag = 1

    if not multi_view:
        sketch_data = multi_view_data(sketch_data, center_margin, crop_size, False, crop_flag)
        image_data = multi_view_data(image_data, center_margin, crop_size, False, crop_flag)

    num_batches_per_sketch = int(np.ceil(sketch_data_size * 1.0 / batch_size))

    sketch_start_index = 0
    for sketch_batch_num in range(num_batches_per_sketch):
        sketch_indices = np.remainder(np.arange(sketch_start_index, sketch_start_index + batch_size), sketch_data_size)
        sketch_batch = sketch_data[sketch_indices, ::]
        sketch_batch = remove_mean(sketch_batch)
        sketch_start_index += batch_size
        image_start_index = 0
        if multi_view:
            sketch_batch = multi_view_data(sketch_batch, center_margin, crop_size, multi_view, crop_flag)
        for image_batch_num in range(num_batches_per_sketch):
            # print "(idx, idy) = (%d, %d)" % (sketch_start_index, image_start_index)
            image_indices = np.remainder(np.arange(image_start_index, image_start_index + batch_size), image_data_size)
            image_batch = image_data[image_indices, ::]
            image_start_index += batch_size
            image_batch = remove_mean(image_batch)
            if multi_view:
                image_batch = multi_view_data(image_batch, center_margin, crop_size, multi_view, crop_flag)
                for view in range(num_views):
                    sketch_mini_batch = sketch_batch[view::num_views, ::]
                    image_mini_batch = image_batch[view::num_views, ::]
                    yield zip(sketch_mini_batch, image_mini_batch)
            else:
                yield zip(sketch_batch, image_batch)


def pair_batch_iter_dev_score(sketch_data, image_data,  batch_size, multi_view = True):
    """
    Generates a batch iterator for a dataset.
    """

    if multi_view:
        num_views = 10
    else:
        num_views = 1
    sketch_data_size = sketch_data.shape[0]
    multi_view_sketch_data_size = sketch_data_size * num_views

    # crop images
    origin_size = image_data.shape[2]
    crop_size = 225
    if 'basenet' in FLAGS.__flags.keys():
        crop_size = get_input_size()[0]

    center_margin = abs(origin_size - crop_size - 1) / 2
    crop_flag = 1


    if multi_view:
        sketch_data = multi_view_data(sketch_data, center_margin, crop_size, multi_view, crop_flag)
        image_data = multi_view_data(image_data, center_margin, crop_size, multi_view, crop_flag)

    num_batches_per_sketch = int(np.ceil(sketch_data_size * 1.0 / batch_size))

    for view in range(num_views):
        sketch_start_index = view
        for sketch_batch_num in range(num_batches_per_sketch):
            sketch_indices = np.remainder(np.arange(sketch_start_index, sketch_start_index + batch_size*num_views, num_views), multi_view_sketch_data_size)
            sketch_batch = sketch_data[sketch_indices, ::]
            # sketch_batch = remove_mean(sketch_batch)
            sketch_start_index += batch_size*num_views
            image_start_index = view
            for image_batch_num in range(num_batches_per_sketch):
                # print "(idx, idy) = (%d, %d)" % (sketch_start_index, image_start_index)
                image_indices = np.remainder(np.arange(image_start_index, image_start_index + batch_size*num_views, num_views), multi_view_sketch_data_size)
                image_batch = image_data[image_indices, ::]
                image_start_index += batch_size*num_views
                # image_batch = remove_mean(image_batch)
                yield zip(sketch_batch, image_batch)


# load data
def load_data():
    dataset = FLAGS.dataset
    image_type = FLAGS.image_type
    sketch_data, image_data, trip_data, label_data = None, None, None, None
    if dataset == 'sketchy':
        sketch_data, image_data, trip_data, label_data = get_sketchy_data(image_type)

    elif dataset == 'TU-Berlin':
        sketch_data, image_data, trip_data, label_data = get_tuberlin_data(image_type)

    elif dataset in ['shoes', 'chairs', 'handbags']:
        sketch_data, image_data, trip_data, label_data = get_sbir_data()

    elif dataset in ['shoesv2', 'chairsv2', 'handbagsv2']:
        sketch_data, image_data, trip_data, label_data = get_sbirv2_data()

    elif dataset == 'sch_com':
        sketch_data, image_data, trip_data, label_data = get_sch_data()

    elif dataset == 'step3':
        sketch_data, image_data, trip_data, label_data = get_step3_data(image_type)

    elif dataset == 'shoes1K':
        sketch_data, image_data, trip_data = get_shoes1K_data(image_type)

    elif dataset == 'flickr15k':
        sketch_data, image_data, trip_data, label_data = get_flickr15k_data(image_type)

    else:
        raise Exception('Dataset type error')
    return sketch_data, image_data, trip_data, label_data


# load shoe-chair data
def get_sbir_data(dataset = None, img_str = None):

    if dataset is None:
        dataset = FLAGS.dataset
    if img_str is None:
        if FLAGS.image_type == 'rgb':
            img_str = 'image'
        else:
            img_str = 'edge'
    data_dir = '/import/vision-ephemeral/Jifei/sbirattmodel/data'
    sketch_dir = os.path.join(data_dir, '%s_sketch_db_train.mat' % dataset)
    image_dir = os.path.join(data_dir, '%s_%s_db_train.mat' % (dataset, img_str))
    # image_dir = os.path.join(data_dir, '%s_%s_db_train_orig.mat' % (dataset, img_str))
    triplet_path = os.path.join(data_dir, '%s_annotation.json' % dataset)
    sketch_dir_te = os.path.join(data_dir, '%s_sketch_db_val.mat' % dataset)
    image_dir_te = os.path.join(data_dir, '%s_%s_db_val.mat' % (dataset, img_str))
    # image_dir_te = os.path.join(data_dir, '%s_%s_db_val_orig.mat' % (dataset, img_str))
    tripet_mat_path = triplet_path.split('_annotation')[0] + '_triplets_train.mat'

    skh_data, img_data, trip = read_train_data(sketch_dir, image_dir, tripet_mat_path)
    # load testing data
    skh_data_te, img_data_te = read_test_data(sketch_dir_te, image_dir_te)

    # repmat sketch data to 3 channels and transpose image data, also remove mean
    skh_data, img_data = convert_sbir_data(skh_data, img_data)
    skh_data_te, img_data_te = convert_sbir_data(skh_data_te, img_data_te, test=True)

    sketch_data = {}
    image_data = {}
    sketch_data['train'] = skh_data
    sketch_data['test'] = skh_data_te
    image_data['train'] = img_data
    image_data['test'] = img_data_te

    FLAGS.num_id_class = img_data.shape[0]

    trip_data = {}
    trip_data['triplet'] = trip
    if FLAGS.use_triplet_sampler:
        trip_data['triplet'] = triplet_path

    label_data = {}
    label_data['sketch_id'] = {}
    label_data['sketch_id']['train'] = np.arange(sketch_data['train'].shape[0])
    label_data['sketch_id']['test'] = np.arange(sketch_data['test'].shape[0])
    label_data['image_id'] = {}
    label_data['image_id']['train'] = np.arange(image_data['train'].shape[0])
    label_data['image_id']['test'] = np.arange(image_data['test'].shape[0])

    return sketch_data, image_data, trip_data, label_data


def read_train_data(sketch_dir, image_dir, triplet_dir):
    # read data
    sketch_data = sio.loadmat(sketch_dir)['data']
    image_data = sio.loadmat(image_dir)['data']

    if os.path.exists(triplet_dir):
        trip = sio.loadmat(triplet_dir)['triplets']
    else:
        trip = gen_full_trip(triplet_dir, sketch_data.shape[0])

    return sketch_data, image_data, trip


def read_test_data(sketch_dir, image_dir):
    # read data
    sketch_data = sio.loadmat(sketch_dir)['data']
    image_data = sio.loadmat(image_dir)['data']
    return sketch_data, image_data


def convert_sbir_data(sketch_data, image_data, test = False):
    if FLAGS.image_type == 'rgb':
        sketch_data = sketch_data[:, :, :, np.newaxis]
        # sketch_data = np.concatenate((sketch_data, sketch_data, sketch_data), axis=3)
        sketch_data = repeat_channel(sketch_data)
        # image_data = np.transpose(image_data, (0, 2, 3, 1))
    if not  FLAGS.use_triplet_sampler or test or FLAGS.basenet in ['inceptionv1', 'inceptionv3', 'mobilenet']:
        if FLAGS.dataset != 'sch_com' or not test:
            sketch_data = remove_mean(sketch_data)
            image_data = remove_mean(image_data)
    return sketch_data, image_data


def repeat_channel(image_data):
    if len(image_data.shape) == 3:
        image_data = image_data[:, :, :, np.newaxis]
    if image_data.shape[-1] == 1:
        image_data = np.concatenate((image_data, image_data, image_data), axis=3)
    return image_data


def gen_full_trip(triplet_dir, nums, cols = None):
    # set cols to nums - 1, if not given
    if cols is None:
        cols = nums - 1
    trips = np.zeros((nums*2, cols), dtype=int)
    range_list = range(nums)
    for index in range(nums):
        row_id = index + index
        # set positive always true match
        trips[row_id, :] = index
        # negative candidates are all the rest except the true match
        temp_range_list = np.array(range_list)
        trips[row_id + 1, :] = np.delete(temp_range_list, index)
    sio.savemat(triplet_dir, {'triplets': trips})
    return trips


# load sch-com data
def get_sch_data():

    detect_flag = 1
    data_dir = '/import/vision-ephemeral/Jifei/sbirattmodel/data'
    meta_data_dir = os.path.join(data_dir, 'sch_com_%s_meta.npy' % FLAGS.image_type)

    if os.path.exists(meta_data_dir) and detect_flag:
        meta_data = np.load(meta_data_dir).item()
        sketch_data = meta_data['sketch_data']
        image_data = meta_data['image_data']
        trip_data = meta_data['trip_data']
        label_data = meta_data['label_data']

    else:
        sketch_data = {}
        image_data = {}
        trip_data = {}

        combined_datasets = ['shoes', 'chairs', 'handbags']
        # read data from each dataset
        for dataset in combined_datasets:
            sketch_data[dataset], image_data[dataset], trip_data[dataset] = get_sbir_data(dataset)

        subsets = ['train', 'test']
        # get data size and combine the sketch/image/triplet data
        for subset in subsets:
            sketch_data[subset] = np.concatenate([sketch_data[dataset][subset] for dataset in combined_datasets])
            image_data[subset] = np.concatenate([image_data[dataset][subset] for dataset in combined_datasets])
            print "In %s data, sketch size: " % subset, sketch_data[subset].shape[0]
            print "In %s data, image size: " % subset, image_data[subset].shape[0]

        trip_data['triplet'] = trip_concat(trip_data, combined_datasets)

        label_data = {'sketch': {}, 'image': {}}
        for subset in subsets:
            label_data['sketch'][subset] = np.concatenate([class_id * np.ones(sketch_data[dataset][subset].shape[0]) for
                                                           class_id, dataset in enumerate(combined_datasets)])
            label_data['image'][subset] = np.concatenate([class_id * np.ones(image_data[dataset][subset].shape[0]) for
                                                          class_id, dataset in enumerate(combined_datasets)])

        # pop origninal data after combined
        for dataset in combined_datasets:
            sketch_data.pop(dataset, None)
            image_data.pop(dataset, None)

    return sketch_data, image_data, trip_data, label_data


def trip_concat(trip_data, datasets):

    offset = {}
    data_size = {}
    trip_width = {}
    pre_data_size = 0
    cur_offset = 0
    for dataset in datasets:
        cur_offset += pre_data_size
        offset[dataset] = cur_offset
        pre_data_size = trip_data[dataset]['triplet'].shape[0]/2
        data_size[dataset] = pre_data_size
        trip_width[dataset] = trip_data[dataset]['triplet'].shape[1]

    max_trip_width = max([trip_width[dataset] for dataset in datasets])
    for dataset in datasets:
        if trip_width[dataset] < max_trip_width:
            repeat_size = max_trip_width/trip_width[dataset] + 1
            trip_data[dataset]['triplet'] = np.tile(trip_data[dataset]['triplet'], (1, repeat_size))
            trip_data[dataset]['triplet'] = trip_data[dataset]['triplet'][:, :max_trip_width]
        trip_data[dataset]['triplet'] += offset[dataset]

    trip_data = np.concatenate([trip_data[dataset]['triplet'] for dataset in datasets])

    return trip_data


# load shoe-1K data
def get_shoes1K_data(image_type = 'grey'):

    dataset = FLAGS.dataset
    if FLAGS.image_type == 'rgb':
        img_str = 'rgb'
    else:
        img_str = 'image'
    data_dir = '/import/vision-ephemeral/Jifei/SBIRDataset'
    sketch_dir = os.path.join(data_dir, 'shoes_2k_dev2_sketch_train.mat')
    image_dir = os.path.join(data_dir, 'shoes_2k_dev2_%s_train.mat' % img_str)
    tripet_mat_path = os.path.join(data_dir, 'shoes_2k_trip_dev2_train.mat')
    sketch_dir_te = os.path.join(data_dir, 'shoes_2k_dev2_sketch_val.mat')
    image_dir_te = os.path.join(data_dir, 'shoes_2k_dev2_%s_val.mat' % img_str)

    skh_data, img_data, trip = read_train_data(sketch_dir, image_dir, tripet_mat_path)
    # load testing data
    skh_data_te, img_data_te = read_test_data(sketch_dir_te, image_dir_te)

    # repmat sketch data to 3 channels and transpose image data, also remove mean
    skh_data, img_data = convert_sbir_data(skh_data, img_data)
    skh_data_te, img_data_te = convert_sbir_data(skh_data_te, img_data_te, test=True)

    sketch_data = {}
    image_data = {}
    sketch_data['train'] = skh_data
    sketch_data['test'] = skh_data_te
    image_data['train'] = img_data
    image_data['test'] = img_data_te

    trip_data = {}
    trip_data['triplet'] = trip

    return sketch_data, image_data, trip_data


# load sketchy data
def get_sketchy_data(image_type = 'grey'):

    # load mean value
    mean_value = 250.42    # default mean value for triplet

    net_data = {}
    if FLAGS.use_scratch_dir:
        prefix = '/scratch/Jifei/sketchy/'
    else:
        prefix = '/import/vision-datasets001/Jifei/sbir/sketchy/'

    sub_types = ['train', 'test']
    net_data['sketch_dir'] = {}
    net_data['image_dir'] = {}

    append_str = ''

    if FLAGS.image_type == 'grey':
        print "loading grey image"
        image_type_str = 'image'
    elif FLAGS.image_type == 'rgb':
        print "loading rgb image"
        image_type_str = 'image'
        append_str = '_rgb'
        # append_str = '_rgb224'
    else:
        print "loading edge for image"
        image_type_str = 'edge'

    for sub_type in sub_types:
        if FLAGS.debug_test:
            # this is only for debug, not loading traing data to save time
            net_data['sketch_dir'][sub_type] = prefix + 'sketch/%s%s.h5' % ('test', append_str)
            net_data['image_dir'][sub_type] = prefix + '%s/%s%s.h5' % (image_type_str, 'test', append_str)
        else:
            net_data['sketch_dir'][sub_type] = prefix + 'sketch/%s%s.h5' % (sub_type, append_str)
            net_data['image_dir'][sub_type] = prefix + '%s/%s%s.h5' % (image_type_str, sub_type, append_str)

    if FLAGS.full_test:
        net_data['sketch_dir']['test'] = prefix + 'sketch/test_probe%s.h5' % append_str

    sketch_train_info = load_hdf5(net_data['sketch_dir']['train'])
    sketch_test_info = load_hdf5(net_data['sketch_dir']['test'])

    image_train_info = load_hdf5(net_data['image_dir']['train'])
    image_test_info = load_hdf5(net_data['image_dir']['test'])

    sketch_data = {}
    image_data = {}

    print "In train data, sketch size: ", sketch_train_info['image_data'].shape[0]
    print "In train data, gallery size: ", image_train_info['image_data'].shape[0]

    if FLAGS.full_test:
        print "use full testing set and test via prob and gallery"
        print "probe size: ", sketch_test_info['image_data'].shape[0]
        print "gallery size: ", image_test_info['image_data'].shape[0]
    else:
        # get_subset_test(sketch_test_info.copy(), split = 1)
        # get_subset_test(sketch_test_info.copy(), split = 2)
        # get_subset_test(sketch_test_info.copy(), split = 3)
        # get_subset_test(sketch_test_info.copy(), split = 4)
        # get_subset_test(sketch_test_info.copy(), split = 5)
        # get_subset_test(sketch_test_info.copy(), split = 6)
        # get_subset_test(sketch_test_info.copy(), split = 7)
        sketch_test_info = get_subset_test(sketch_test_info, split = FLAGS.test_split)

    FLAGS.num_id_class = image_train_info['image_data'].shape[0]

    sketch_data['train'], image_data['train'], trips, easy_trips, hard_trips = load_data_and_pair(sketch_train_info, image_train_info, 'train')
    sketch_data['test'], image_data['test'] = load_data_and_pair(sketch_test_info, image_test_info, 'test')

    if len(sketch_data['train'].shape) == 3:
        sketch_data['train'] = sketch_data['train'][:, :, :, np.newaxis]
        sketch_data['test'] = sketch_data['test'][:, :, :, np.newaxis]
        image_data['train'] = image_data['train'][:, :, :, np.newaxis]
        image_data['test'] = image_data['test'][:, :, :, np.newaxis]

    trip_data = {}
    trip_data['triplet'] = trips
    trip_data['easy_triplet'] = easy_trips
    trip_data['hard_triplet'] = hard_trips

    label_data = {}
    label_data['sketch'] = {}
    label_data['sketch']['train'] = sketch_train_info['class_id']
    label_data['sketch']['test'] = sketch_test_info['class_id']
    label_data['image'] = {}
    label_data['image']['train'] = image_train_info['class_id']
    label_data['image']['test'] = image_test_info['class_id']
    label_data['sketch_id'] = {}
    label_data['sketch_id']['train'] = sketch_train_info['image_id']
    label_data['sketch_id']['test'] = sketch_test_info['image_id']
    label_data['image_id'] = {}
    label_data['image_id']['train'] = image_train_info['image_id']
    label_data['image_id']['test'] = image_test_info['image_id']
    label_data['sketch_class'] = sketch_test_info['class_id']
    label_data['sketch_label'] = sketch_test_info['image_id']
    label_data['image_label'] = image_test_info['image_id']

    return sketch_data, image_data, trip_data, label_data


# load sketchy data
def get_sbirv2_data(dataset = None, img_str = None):

    if dataset is None:
        dataset = FLAGS.dataset.split('v2')[0]
    net_data = {}
    if FLAGS.use_scratch_dir:
        prefix = '/scratch/Jifei/sketchy/'
    else:
        prefix = '/import/vision-datasets001/Jifei/sbir/sch_com_v2/'

    sub_types = ['train', 'test']
    net_data['sketch_dir'] = {}
    net_data['image_dir'] = {}

    append_str = ''

    if FLAGS.image_type == 'rgb':
        print "loading rgb image"
        image_type_str = 'photo'
        append_str = '_rgb'
        # append_str = '_rgb224'
    else:
        print "loading edge for image"
        image_type_str = 'edge'

    for sub_type in sub_types:
        net_data['sketch_dir'][sub_type] = prefix + '%s/sketch/%s%s.h5' % (dataset, sub_type, append_str)
        net_data['image_dir'][sub_type] = prefix + '%s/%s/%s%s.h5' % (dataset, image_type_str, sub_type, append_str)

    sketch_train_info = load_hdf5(net_data['sketch_dir']['train'])
    sketch_test_info = load_hdf5(net_data['sketch_dir']['test'])

    image_train_info = load_hdf5(net_data['image_dir']['train'])
    image_test_info = load_hdf5(net_data['image_dir']['test'])

    sketch_data = {}
    image_data = {}

    print "In train data, sketch size: ", sketch_train_info['image_data'].shape[0]
    print "In train data, gallery size: ", image_train_info['image_data'].shape[0]

    if FLAGS.full_test:
        print "use full testing set and test via prob and gallery"
    else:
        sketch_test_info = get_subset_test(sketch_test_info, split = FLAGS.test_split)

    print "In test data, probe size: ", sketch_test_info['image_data'].shape[0]
    print "In test data, gallery size: ", image_test_info['image_data'].shape[0]

    FLAGS.num_id_class = image_train_info['image_data'].shape[0]

    sketch_data['train'], image_data['train'], trips, easy_trips, hard_trips = load_data_and_pair(sketch_train_info, image_train_info, 'train')
    sketch_data['test'], image_data['test'] = load_data_and_pair(sketch_test_info, image_test_info, 'test')

    if len(sketch_data['train'].shape) == 3:
        sketch_data['train'] = sketch_data['train'][:, :, :, np.newaxis]
        sketch_data['test'] = sketch_data['test'][:, :, :, np.newaxis]
        image_data['train'] = image_data['train'][:, :, :, np.newaxis]
        image_data['test'] = image_data['test'][:, :, :, np.newaxis]

    trip_data = {}
    trip_data['triplet'] = trips
    trip_data['easy_triplet'] = easy_trips
    trip_data['hard_triplet'] = hard_trips

    label_data = {}
    label_data['sketch'] = {}
    label_data['sketch']['train'] = sketch_train_info['class_id']
    label_data['sketch']['test'] = sketch_test_info['class_id']
    label_data['image'] = {}
    label_data['image']['train'] = image_train_info['class_id']
    label_data['image']['test'] = image_test_info['class_id']
    label_data['sketch_id'] = {}
    label_data['sketch_id']['train'] = sketch_train_info['image_id']
    label_data['sketch_id']['test'] = sketch_test_info['image_id']
    label_data['image_id'] = {}
    label_data['image_id']['train'] = image_train_info['image_id']
    label_data['image_id']['test'] = image_test_info['image_id']
    label_data['sketch_class'] = sketch_test_info['class_id']
    label_data['sketch_label'] = sketch_test_info['image_id']
    label_data['image_label'] = image_test_info['image_id']

    return sketch_data, image_data, trip_data, label_data


# load tuberlin data
def get_tuberlin_data(image_type = 'grey'):

    net_data = {}
    if FLAGS.use_scratch_dir:
        prefix = '/scratch/Jifei/TU_Berlin/'
    else:
        prefix = '/import/vision-datasets001/Jifei/TU_Berlin/'

    sub_types = ['train', 'test']
    net_data['sketch_dir'] = {}
    net_data['image_dir'] = {}

    append_str = ''

    if FLAGS.image_type == 'grey':
        print "loading grey image"
    elif FLAGS.image_type == 'rgb':
        print "loading rgb image"
        append_str = '_rgb'
    else:
        print "loading edge for image"

    for sub_type in sub_types:
        if FLAGS.debug_test:
            # this is only for debug, not loading traing data to save time
            net_data['sketch_dir'][sub_type] = prefix + '%s%s.h5' % ('test', append_str)
        else:
            net_data['sketch_dir'][sub_type] = prefix + '%s%s.h5' % (sub_type, append_str)

    sketch_train_info = load_hdf5(net_data['sketch_dir']['train'])
    sketch_test_info = load_hdf5(net_data['sketch_dir']['test'])

    sketch_data = {}
    image_data = None

    print "In train data, sketch size: ", sketch_train_info['image_data'].shape[0]
    print "In test data, sketch size: ", sketch_test_info['image_data'].shape[0]

    sketch_data['train'] = sketch_train_info['image_data']
    sketch_data['test'] = sketch_test_info['image_data']

    if len(sketch_data['train'].shape) == 3:
        sketch_data['train'] = sketch_data['train'][:, :, :, np.newaxis]
        sketch_data['test'] = sketch_data['test'][:, :, :, np.newaxis]

    trip_data = None

    label_data = {}
    label_data['sketch'] = {}
    label_data['sketch']['train'] = sketch_train_info['class_id']
    label_data['sketch']['test'] = sketch_test_info['class_id']
    label_data['sketch_class'] = sketch_test_info['class_id']
    label_data['sketch_label'] = sketch_test_info['image_id']

    return sketch_data, image_data, trip_data, label_data


# load step3 data
def get_step3_data(image_type = 'grey'):

    net_data = {}
    step3_prefix = '/import/vision-datasets001/Jifei/sbir/step3_data/'
    shoes_prefix = '/import/vision-ephemeral/Jifei/sbirattmodel/data/'

    net_data['sketch_dir'] = {}
    net_data['image_dir'] = {}

    append_str = ''

    if FLAGS.image_type == 'grey':
        print "loading grey image"
        image_type_str = 'image'
    elif FLAGS.image_type == 'rgb':
        print "loading rgb image"
        image_type_str = 'image'
        append_str = '_rgb'
    else:
        print "loading edge for image"
        image_type_str = 'edge'

    # load train data, actually, we use all step 3 data
    net_data['sketch_dir']['train'] = step3_prefix + 'sketch/%s%s.h5' % ('all', append_str)
    net_data['image_dir']['train'] = step3_prefix + '%s/%s%s.h5' % (image_type_str, 'all', append_str)

    sketch_train_info = load_hdf5(net_data['sketch_dir']['train'])
    image_train_info = load_hdf5(net_data['image_dir']['train'])

    sketch_data = {}
    image_data = {}

    print "In train data, prob size: ", sketch_train_info['image_data'].shape[0]
    print "In train data, gallery size: ", image_train_info['image_data'].shape[0]

    sketch_data['train'], image_data['train'], trips = load_data_and_pair(sketch_train_info, image_train_info, 'step3')

    net_data['sketch_dir']['test'] = shoes_prefix + 'shoes_sketch_db_val.mat'
    net_data['image_dir']['test'] = shoes_prefix + 'shoes_%s_db_val.mat' % image_type_str
    sketch_data['test'], image_data['test'] = read_test_data(net_data['sketch_dir']['test'], net_data['image_dir']['test'])

    if len(sketch_data['train'].shape) == 3:
        sketch_data['train'] = sketch_data['train'][:, :, :, np.newaxis]
        sketch_data['test'] = sketch_data['test'][:, :, :, np.newaxis]
        image_data['train'] = image_data['train'][:, :, :, np.newaxis]
        image_data['test'] = image_data['test'][:, :, :, np.newaxis]

    # repmat sketch data to 3 channels and transpose image data, also remove mean
    sketch_data['test'], image_data['test'] = convert_sbir_data(sketch_data['test'], image_data['test'], test=True)

    trip_data = {'triplet': trips}

    label_data = {}
    label_data['sketch'] = {}
    label_data['sketch']['train'] = sketch_train_info['class_id']
    label_data['image'] = {}
    label_data['image']['train'] = image_train_info['class_id']

    return sketch_data, image_data, trip_data, label_data


# load flickr15k data
def get_flickr15k_data(image_type = 'grey'):

    net_data = {}
    flickr15k_prefix = '/import/vision-datasets001/Jifei/Flickr15K/'

    net_data['sketch_dir'] = {}
    net_data['image_dir'] = {}

    append_str = ''

    if FLAGS.image_type == 'grey':
        print "loading grey image"
        image_type_str = 'image'
    elif FLAGS.image_type == 'rgb':
        print "loading rgb image"
        image_type_str = 'image'
        append_str = '_rgb'
    else:
        print "loading edge for image"
        image_type_str = 'edge'

    # load train data, actually, we use all step 3 data
    net_data['sketch_dir']['train'] = flickr15k_prefix + 'sketch/%s%s.h5' % ('all', append_str)
    net_data['image_dir']['train'] = flickr15k_prefix + '%s/%s%s.h5' % (image_type_str, 'all', append_str)

    sketch_train_info = load_hdf5(net_data['sketch_dir']['train'])
    image_train_info = load_hdf5(net_data['image_dir']['train'])

    sketch_data = {}
    image_data = {}

    print "In train data, prob size: ", sketch_train_info['image_data'].shape[0]
    print "In train data, gallery size: ", image_train_info['image_data'].shape[0]

    sketch_data['train'], image_data['train'], trips = load_data_and_pair(sketch_train_info, image_train_info, 'step3')

    # load train data, actually, we use all step 3 data
    net_data['sketch_dir']['test'] = flickr15k_prefix + 'sketch/%s%s.h5' % ('all', append_str)
    net_data['image_dir']['test'] = flickr15k_prefix + '%s/%s%s.h5' % (image_type_str, 'all', append_str)

    sketch_test_info = load_hdf5(net_data['sketch_dir']['test'])
    image_test_info = load_hdf5(net_data['image_dir']['test'])
    sketch_data['test'], image_data['test'] = load_data_and_pair(sketch_test_info, image_test_info, 'test')

    trip_data = {'triplet': trips}

    label_data = {}
    label_data['sketch'] = {}
    label_data['sketch']['train'] = sketch_train_info['class_id']
    label_data['sketch']['test'] = sketch_test_info['class_id']
    label_data['image'] = {}
    label_data['image']['train'] = image_train_info['class_id']
    label_data['image']['test'] = image_test_info['class_id']
    label_data['sketch_class'] = sketch_test_info['class_id']
    # use class id instead of instance id, as this is a class level dataset
    label_data['sketch_label'] = sketch_test_info['class_id']
    label_data['image_label'] = image_test_info['class_id']

    return sketch_data, image_data, trip_data, label_data


def remove_mean(data):
    if 'image_type' in FLAGS.__flags.keys():
        # image_type = FLAGS.image_type
        if FLAGS.basenet in ['sketchynet', 'alexnet']:
            # assert image_type == 'rgb', "image type error"
            if FLAGS.image_type in ['edge', 'grey']:
                data = repeat_channel(data)
            mean_value = [104.0069879317889, 116.66876761696767, 122.6789143406786]  # mean_bgr
        elif FLAGS.basenet in ['resnet', 'vgg']:
            # assert image_type == 'rgb', "image type error"
            if FLAGS.image_type in ['edge', 'grey']:
                data = repeat_channel(data)
            mean_value = [123.68, 116.78, 103.94]  # mean_rgb
        else:
            mean_value = 250.42

        if FLAGS.basenet in ['ss_alex_net', 'inceptionv1', 'inceptionv3', 'mobilenet']:
            # print "for the alexnet with batch normaliztion, normalize the image and use 0 mean value"
            if FLAGS.image_type in ['edge', 'grey']:
                data = repeat_channel(data)
            data = data / 255.0 * 2.0 - 1
        else:
            data = data - mean_value
    return data


def random_flip(data):
    if random.random() >= 0.5:
        data = data[:, :, ::-1, :]
    return data


def refine_epochs(num_epochs, num_batches_per_epoch):
    if 'num_batches_per_epoch' in FLAGS.__flags.keys():
        FLAGS.num_batches_per_epoch = num_batches_per_epoch
    if 'max_steps' in FLAGS.__flags.keys():
        est_steps = FLAGS.num_epochs * num_batches_per_epoch
        if FLAGS.max_steps != est_steps:
            if FLAGS.max_steps > est_steps:
                print "Given max_steps is not corresponding to num_epochs"
                FLAGS.num_epochs = FLAGS.max_steps / FLAGS.num_batches_per_epoch
                print "Change num_epochs to ", FLAGS.num_epochs
            num_epochs = FLAGS.num_epochs
    return num_epochs


def load_data_and_pair(sketch_data_info, image_data_info, stage = 'train'):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    sketch_data = sketch_data_info['image_data']
    image_data = image_data_info['image_data']

    if stage == 'train':
        easy_trips = gen_triplet_pairs(sketch_data_info['class_id'], image_data_info['class_id'])
        hard_trips = gen_triplet_pairs(sketch_data_info['image_id'], image_data_info['image_id'])

        hard_trips = refine_hard_trips(easy_trips, hard_trips)
        trips = hard_trips

        return sketch_data, image_data, trips, easy_trips, hard_trips

    elif stage == 'step3':
        trips = gen_triplet_pairs(sketch_data_info['class_id'], image_data_info['class_id'])

        return sketch_data, image_data, trips

    else:

        return sketch_data, image_data


def gen_triplet_pairs(anc_pair_data, pos_pair_data):
    num_class = max(anc_pair_data) + 1
    sketch_data_list = anc_pair_data.tolist()
    image_data_list = pos_pair_data.tolist()
    unique_ids = np.sort(np.unique(np.array(sketch_data_list))).tolist()

    anc_start_ids = [0]
    pos_start_ids = [0]
    pair_infos = []
    max_len_per_sketch = 0
    max_len_per_photo = 0
    min_len_per_sketch = len(sketch_data_list)
    min_len_per_photo = len(image_data_list)
    range_list = range(len(image_data_list))

    # len_unique_ids = len(unique_ids)
    # if num_class != len_unique_ids:
    #     print "the length of unique_ids is not equal to number of class"

    for index in xrange(num_class):
        len_for_this_sketch = sketch_data_list.count(unique_ids[index])
        len_for_this_photo = image_data_list.count(unique_ids[index])
        if len_for_this_photo == 0:
            import pdb
            pdb.set_trace()
        assert len_for_this_photo != 0, "No positive photo found for this anchor sketch"
        anc_start_ids.append(anc_start_ids[-1] + len_for_this_sketch)
        pos_start_ids.append(pos_start_ids[-1] + len_for_this_photo)
        max_len_per_sketch = max(max_len_per_sketch, len_for_this_sketch)
        max_len_per_photo = max(max_len_per_photo, len_for_this_photo)
        min_len_per_sketch = min(min_len_per_sketch, len_for_this_sketch)
        min_len_per_photo = max(min_len_per_photo, len_for_this_photo)

    for index in xrange(num_class):
        pair_info = {}
        pair_info['anc'] = range(anc_start_ids[index], anc_start_ids[index+1])
        pair_info['pos'] = range(pos_start_ids[index], pos_start_ids[index+1])
        pair_info['neg'] = [x for x in range_list if x not in pair_info['pos']] # not apply to hard triplet pair
        pair_infos.append(pair_info)

    return pair_infos


def refine_hard_trips(trips, hard_trips):
    pos_id = 0
    for trip in trips:
        for pos_id in trip['pos']:
            try:
                hard_trips[pos_id]['neg'] = [x for x in trip['pos'] if x not in hard_trips[pos_id]['pos']]
            except:
                import pdb
                pdb.set_trace()

    assert pos_id + 1 == len(hard_trips), "refine hard trips failed, check whether your triplet pairs is right"
    return hard_trips


def get_subset_test(test_data_info, split = 1):
    orig_nums = test_data_info['instance_id'].shape[0]
    indices = test_data_info['instance_id'] == split
    for key in test_data_info.keys():
        if len(test_data_info[key].shape) == 1:
            test_data_info[key] = test_data_info[key][indices]
        else:
            test_data_info[key] = test_data_info[key][indices, ::]
    subset_nums = test_data_info['instance_id'].shape[0]
    print "Split %d has %d sketches from %d sketches" % (split, subset_nums, orig_nums)
    return test_data_info


def gen_image_pairs(easy_batch_size, hard_batch_size):

    anc_pos_ids = list(itertools.permutations(range(easy_batch_size), 2))
    anc_pos_neg_ids = list(itertools.product(range(len(anc_pos_ids)), range(easy_batch_size)))
    anc_ids = []
    pos_ids = []
    neg_ids = []

    for anc_pos_neg_id in anc_pos_neg_ids:
        anc_ids.append(anc_pos_ids[anc_pos_neg_id[0]][0])
        pos_ids.append(anc_pos_ids[anc_pos_neg_id[0]][1])
        neg_ids.append(anc_pos_neg_id[1])

    hard_anc_pos_neg_ids = list(itertools.permutations(range(hard_batch_size), 2))

    for hard_anc_pos_neg_id in hard_anc_pos_neg_ids:
        anc_ids.append(hard_anc_pos_neg_id[0])
        pos_ids.append(hard_anc_pos_neg_id[0])
        neg_ids.append(hard_anc_pos_neg_id[1])

    return anc_ids, pos_ids, neg_ids


def gen_hard_image_pairs(mini_batch_size, num_mini_batch_size):

    anc_ids = []
    pos_ids = []
    neg_ids = []

    mini_batch_anc_ids = []
    mini_batch_pos_ids = []
    mini_batch_neg_ids = []

    hard_anc_pos_neg_ids = list(itertools.permutations(range(mini_batch_size), 2))

    for hard_anc_pos_neg_id in hard_anc_pos_neg_ids:
        mini_batch_anc_ids.append(hard_anc_pos_neg_id[0])
        mini_batch_pos_ids.append(hard_anc_pos_neg_id[0])
        mini_batch_neg_ids.append(hard_anc_pos_neg_id[1])

    for index in range(num_mini_batch_size):
        anc_ids.extend(np.array(mini_batch_anc_ids) + index * mini_batch_size)
        pos_ids.extend(np.array(mini_batch_pos_ids) + index * mini_batch_size)
        neg_ids.extend(np.array(mini_batch_neg_ids) + index * mini_batch_size)

    return anc_ids, pos_ids, neg_ids


# generate training batches for att_trip network
def data_tid_batch_iter(sketch_data, image_data, trips, hard_trips, easy_batch_size, hard_batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    trip_size = len(trips)
    hard_trip_size = len(hard_trips)
    origin_size = image_data.shape[2]
    crop_size = 225
    margin_size = origin_size - crop_size

    num_batches_per_epoch = trip_size
    print "num_batches_per_epoch: ", num_batches_per_epoch

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices_trip = np.random.permutation(np.arange(trip_size))
            shuffled_trip = np.array(trips)[shuffle_indices_trip]
            shuffle_indices_hard_trip = np.random.permutation(np.arange(hard_trip_size))
            shuffled_hard_trip = np.array(hard_trips)[shuffle_indices_hard_trip]
        for batch_num in range(num_batches_per_epoch):
            anc_ids_trip = np.array(shuffled_trip[batch_num]['anc'])
            pos_ids_trip = np.array(shuffled_trip[batch_num]['pos'])
            neg_ids_trip = np.array(shuffled_trip[batch_num]['neg'])

            batch_trip_anc_indices = np.random.choice(len(anc_ids_trip), size = easy_batch_size)
            batch_trip_pos_indices = np.random.choice(len(pos_ids_trip), size = easy_batch_size)
            batch_trip_neg_indices = np.random.choice(len(neg_ids_trip), size = easy_batch_size)

            anc_batch_easy = sketch_data[anc_ids_trip[batch_trip_anc_indices], ::]
            pos_batch_easy = image_data[pos_ids_trip[batch_trip_pos_indices], ::]
            neg_batch_easy = image_data[neg_ids_trip[batch_trip_neg_indices], ::]

            anc_ids_hard_trip = np.array(shuffled_hard_trip[batch_num]['anc'])
            pos_ids_hard_trip = np.array(shuffled_hard_trip[batch_num]['pos'])
            neg_ids_hard_trip = np.array(shuffled_hard_trip[batch_num]['neg'])

            batch_hard_trip_anc_indices = np.random.choice(len(anc_ids_hard_trip), size = hard_batch_size)
            batch_hard_trip_pos_indices = np.random.choice(len(pos_ids_hard_trip), size = hard_batch_size)
            batch_hard_trip_neg_indices = np.random.choice(len(neg_ids_hard_trip), size = hard_batch_size)

            anc_batch_hard = sketch_data[anc_ids_hard_trip[batch_hard_trip_anc_indices], ::]
            pos_batch_hard = image_data[pos_ids_hard_trip[batch_hard_trip_pos_indices], ::]
            neg_batch_hard = image_data[neg_ids_hard_trip[batch_hard_trip_neg_indices], ::]

            anc_batch = np.concatenate((anc_batch_easy, anc_batch_hard), axis=0)
            pos_batch = np.concatenate((pos_batch_easy, pos_batch_hard), axis=0)
            neg_batch = np.concatenate((neg_batch_easy, neg_batch_hard), axis=0)

            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]

            yield zip(anc_batch, pos_batch, neg_batch)


# generate training batches for att_trip network
def data_label_hard_tid_batch_iter(sketch_data, image_data, sketch_label, image_label, trips, hard_trips, num_mini_batch, mini_batch_size, num_epochs, flip_flag = False):
    """
    Generates a batch iterator for a dataset.
    """
    num_class = len(trips)
    origin_size = image_data.shape[2]
    crop_size, origin_channel = get_input_size()
    margin_size = origin_size - crop_size

    num_batches_per_epoch = int(np.ceil(num_class * 1.0 / num_mini_batch))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        shuffle_indices_trip = np.random.permutation(np.arange(num_class))
        shuffled_trip = np.array(trips)[shuffle_indices_trip]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*num_batches_per_epoch
            batch_indices = np.remainder(np.arange(start_index, start_index + num_mini_batch), num_class)
            shuffle_batch_indices = batch_indices[np.random.permutation(np.arange(num_mini_batch))]

            anc_batch = np.empty((0, origin_size, origin_size, origin_channel))
            pos_batch = np.empty((0, origin_size, origin_size, origin_channel))
            neg_batch = np.empty((0, origin_size, origin_size, origin_channel))

            anc_batch_label = []
            pos_batch_label = []
            neg_batch_label = []

            # batch_index represent the index of shuffled class queue
            for batch_index in shuffle_batch_indices:

                # random get one instance's hard trips according to pos image id in trips list, given this class
                hard_trip_index = np.random.choice(shuffled_trip[batch_index]['pos'])

                anc_ids_hard_trip = np.array(hard_trips[hard_trip_index]['anc'])

                pos_ids_hard_trip = np.array(hard_trips[hard_trip_index]['pos'])
                neg_ids_hard_trip = np.array(hard_trips[hard_trip_index]['neg'])

                batch_hard_trip_anc_indices = np.random.choice(len(anc_ids_hard_trip), size = mini_batch_size)
                batch_hard_trip_pos_indices = np.random.choice(len(pos_ids_hard_trip), size = mini_batch_size)
                batch_hard_trip_neg_indices = np.random.choice(len(neg_ids_hard_trip), size = mini_batch_size)

                anc_batch = np.concatenate((anc_batch, sketch_data[anc_ids_hard_trip[batch_hard_trip_anc_indices], ::]), axis=0)
                pos_batch = np.concatenate((pos_batch, image_data[pos_ids_hard_trip[batch_hard_trip_pos_indices], ::]), axis=0)
                neg_batch = np.concatenate((neg_batch, image_data[neg_ids_hard_trip[batch_hard_trip_neg_indices], ::]), axis=0)

                anc_batch_label.extend(sketch_label[anc_ids_hard_trip[batch_hard_trip_anc_indices]])
                pos_batch_label.extend(image_label[pos_ids_hard_trip[batch_hard_trip_pos_indices]])
                neg_batch_label.extend(image_label[neg_ids_hard_trip[batch_hard_trip_neg_indices]])

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, ::]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, ::]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, ::]

            if flip_flag:
                if  np.random.random() >= 0.5:
                    # print "flip"
                    anc_batch = anc_batch[:, :, ::-1, :]
                    pos_batch = pos_batch[:, :, ::-1, :]

            anc_batch = remove_mean(anc_batch)
            pos_batch = remove_mean(pos_batch)
            neg_batch = remove_mean(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_label, pos_batch_label, neg_batch_label)


# generate training batches for att_trip network
def data_label_tid_batch_iter(sketch_data, image_data, sketch_label, image_label, trips, hard_trips, easy_batch_size, hard_batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    trip_size = len(trips)
    hard_trip_size = len(hard_trips)
    origin_size = image_data.shape[2]
    crop_size = 225
    margin_size = origin_size - crop_size

    num_batches_per_epoch = trip_size
    print "num_batches_per_epoch: ", num_batches_per_epoch

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices_trip = np.random.permutation(np.arange(trip_size))
            shuffled_trip = np.array(trips)[shuffle_indices_trip]
            shuffle_indices_hard_trip = np.random.permutation(np.arange(hard_trip_size))
            shuffled_hard_trip = np.array(hard_trips)[shuffle_indices_hard_trip]
        for batch_num in range(num_batches_per_epoch):
            anc_ids_trip = np.array(shuffled_trip[batch_num]['anc'])
            pos_ids_trip = np.array(shuffled_trip[batch_num]['pos'])
            neg_ids_trip = np.array(shuffled_trip[batch_num]['neg'])

            batch_trip_anc_indices = np.random.choice(len(anc_ids_trip), size = easy_batch_size)
            batch_trip_pos_indices = np.random.choice(len(pos_ids_trip), size = easy_batch_size)
            batch_trip_neg_indices = np.random.choice(len(neg_ids_trip), size = easy_batch_size)

            anc_batch_easy = sketch_data[anc_ids_trip[batch_trip_anc_indices], ::]
            pos_batch_easy = image_data[pos_ids_trip[batch_trip_pos_indices], ::]
            neg_batch_easy = image_data[neg_ids_trip[batch_trip_neg_indices], ::]

            anc_batch_easy_label = sketch_label[anc_ids_trip[batch_trip_anc_indices]]
            pos_batch_easy_label = image_label[pos_ids_trip[batch_trip_pos_indices]]
            neg_batch_easy_label = image_label[neg_ids_trip[batch_trip_neg_indices]]

            anc_ids_hard_trip = np.array(shuffled_hard_trip[batch_num]['anc'])
            pos_ids_hard_trip = np.array(shuffled_hard_trip[batch_num]['pos'])
            neg_ids_hard_trip = np.array(shuffled_hard_trip[batch_num]['neg'])

            batch_hard_trip_anc_indices = np.random.choice(len(anc_ids_hard_trip), size = hard_batch_size)
            batch_hard_trip_pos_indices = np.random.choice(len(pos_ids_hard_trip), size = hard_batch_size)
            batch_hard_trip_neg_indices = np.random.choice(len(neg_ids_hard_trip), size = hard_batch_size)

            anc_batch_hard = sketch_data[anc_ids_hard_trip[batch_hard_trip_anc_indices], ::]
            pos_batch_hard = image_data[pos_ids_hard_trip[batch_hard_trip_pos_indices], ::]
            neg_batch_hard = image_data[neg_ids_hard_trip[batch_hard_trip_neg_indices], ::]

            anc_batch_hard_label = sketch_label[anc_ids_hard_trip[batch_hard_trip_anc_indices]]
            pos_batch_hard_label = image_label[pos_ids_hard_trip[batch_hard_trip_pos_indices]]
            neg_batch_hard_label = image_label[neg_ids_hard_trip[batch_hard_trip_neg_indices]]

            anc_batch = np.concatenate((anc_batch_easy, anc_batch_hard), axis=0)
            pos_batch = np.concatenate((pos_batch_easy, pos_batch_hard), axis=0)
            neg_batch = np.concatenate((neg_batch_easy, neg_batch_hard), axis=0)

            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]

            anc_batch_label = np.concatenate((anc_batch_easy_label, anc_batch_hard_label), axis=0)
            pos_batch_label = np.concatenate((pos_batch_easy_label, pos_batch_hard_label), axis=0)
            neg_batch_label = np.concatenate((neg_batch_easy_label, neg_batch_hard_label), axis=0)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_label, pos_batch_label, neg_batch_label)


# generate training batches for att_trip network
def data_label_easy_tid_batch_iter(sketch_data, image_data, sketch_label, image_label, trips, num_mini_batch, mini_batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    num_class = len(trips)
    origin_size = image_data.shape[2]
    crop_size = 225
    margin_size = origin_size - crop_size

    num_batches_per_epoch = int(np.ceil(num_class * 1.0 / num_mini_batch))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        shuffle_indices_trip = np.random.permutation(np.arange(num_class))
        shuffled_trip = np.array(trips)[shuffle_indices_trip]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*num_batches_per_epoch
            batch_indices = np.remainder(np.arange(start_index, start_index + num_mini_batch), num_class)
            shuffle_batch_indices = batch_indices[np.random.permutation(np.arange(num_mini_batch))]

            anc_batch = np.empty((0, origin_size, origin_size))
            pos_batch = np.empty((0, origin_size, origin_size))
            neg_batch = np.empty((0, origin_size, origin_size))

            anc_batch_label = []
            pos_batch_label = []
            neg_batch_label = []

            # batch_index represent the index of shuffled class queue
            for batch_index in shuffle_batch_indices:

                anc_ids_easy_trip = np.array(shuffled_trip[batch_index]['anc'])

                pos_ids_easy_trip = np.array(shuffled_trip[batch_index]['pos'])
                neg_ids_easy_trip = np.array(shuffled_trip[batch_index]['neg'])

                batch_easy_trip_anc_indices = np.random.choice(len(anc_ids_easy_trip), size = mini_batch_size)
                batch_easy_trip_pos_indices = np.random.choice(len(pos_ids_easy_trip), size = mini_batch_size)
                batch_easy_trip_neg_indices = np.random.choice(len(neg_ids_easy_trip), size = mini_batch_size)

                anc_batch = np.concatenate((anc_batch, sketch_data[anc_ids_hard_trip[batch_hard_trip_anc_indices], ::]), axis=0)
                pos_batch = np.concatenate((pos_batch, image_data[pos_ids_hard_trip[batch_hard_trip_pos_indices], ::]), axis=0)
                neg_batch = np.concatenate((neg_batch, image_data[neg_ids_hard_trip[batch_hard_trip_neg_indices], ::]), axis=0)

                anc_batch_label.extend(sketch_label[anc_ids_easy_trip[batch_easy_trip_anc_indices]])
                pos_batch_label.extend(image_label[pos_ids_easy_trip[batch_easy_trip_pos_indices]])
                neg_batch_label.extend(image_label[neg_ids_easy_trip[batch_easy_trip_neg_indices]])

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, np.newaxis]

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_label, pos_batch_label, neg_batch_label)


# generate training batches for att_trip network
def tri_data_label_batch_iter(sketch_data, image_data, sketch_label, image_label, triplets, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    margin_size = origin_size - crop_size

    triplet_size = len(triplets)
    num_batches_per_epoch = int(np.ceil(triplet_size * 1.0 / batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        indices = np.random.permutation(np.arange(triplet_size))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            batch_indices = np.remainder(np.arange(start_index, start_index + batch_size), triplet_size)
            anc_inds = []
            pos_inds = []
            neg_inds = []
            for index in batch_indices:
                triplet = triplets[indices[index]]
                anc_id = np.random.choice(triplet['anc'])
                pos_id = np.random.choice(triplet['pos'])
                neg_id = np.random.choice(triplet['neg'])
                anc_inds.append(anc_id)
                pos_inds.append(pos_id)
                neg_inds.append(neg_id)

            anc_batch = sketch_data[anc_inds, ::]
            pos_batch = image_data[pos_inds, ::]
            neg_batch = image_data[neg_inds, ::]

            anc_batch_label = sketch_label[anc_inds]
            pos_batch_label = image_label[pos_inds]
            neg_batch_label = image_label[neg_inds]

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            if random.random() >= 0.5:
                # print "flip"
                anc_batch = anc_batch[:, :, ::-1, :]
                pos_batch = pos_batch[:, :, ::-1, :]
                neg_batch = neg_batch[:, :, ::-1, :]

            anc_batch = remove_mean(anc_batch)
            pos_batch = remove_mean(pos_batch)
            neg_batch = remove_mean(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_label, pos_batch_label, neg_batch_label)


# generate training batches for att_trip network
def tri_data_label_id_batch_iter(sketch_data, image_data, sketch_label, image_label, sketch_id, image_id, triplets, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    margin_size = origin_size - crop_size

    triplet_size = len(triplets)
    num_batches_per_epoch = int(np.ceil(triplet_size * 1.0 / batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        indices = np.random.permutation(np.arange(triplet_size))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            batch_indices = np.remainder(np.arange(start_index, start_index + batch_size), triplet_size)
            anc_inds = []
            pos_inds = []
            neg_inds = []
            for index in batch_indices:
                triplet = triplets[indices[index]]
                anc_id = np.random.choice(triplet['anc'])
                pos_id = np.random.choice(triplet['pos'])
                neg_id = np.random.choice(triplet['neg'])
                anc_inds.append(anc_id)
                pos_inds.append(pos_id)
                neg_inds.append(neg_id)

            anc_batch = sketch_data[anc_inds, ::]
            pos_batch = image_data[pos_inds, ::]
            neg_batch = image_data[neg_inds, ::]

            anc_batch_label = sketch_label[anc_inds]
            pos_batch_label = image_label[pos_inds]
            neg_batch_label = image_label[neg_inds]

            anc_batch_id = sketch_id[anc_inds]
            pos_batch_id = image_id[pos_inds]
            neg_batch_id = image_id[neg_inds]

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            if random.random() >= 0.5:
                # print "flip"
                anc_batch = anc_batch[:, :, ::-1, :]
                pos_batch = pos_batch[:, :, ::-1, :]
                neg_batch = neg_batch[:, :, ::-1, :]

            anc_batch = remove_mean(anc_batch)
            pos_batch = remove_mean(pos_batch)
            neg_batch = remove_mean(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_label, pos_batch_label, neg_batch_label, anc_batch_id, pos_batch_id, neg_batch_id)


# generate training batches for sketchy dataset, verification loss
def tri_data_label_ver_batch_iter(sketch_data, image_data, sketch_label, image_label, triplets, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    margin_size = origin_size - crop_size

    triplet_size = len(triplets)
    num_batches_per_epoch = int(np.ceil(triplet_size * 1.0 / batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    gallery_size = triplets[-1]['pos'][0]

    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        indices = np.random.permutation(np.arange(triplet_size))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            batch_indices = np.remainder(np.arange(start_index, start_index + batch_size), triplet_size)
            anc_inds = []
            pos_inds = []
            neg_inds = []
            for index in batch_indices:
                triplet = triplets[indices[index]]
                anc_id = np.random.choice(triplet['anc'])
                pos_id = np.random.choice(triplet['pos'])
                rand_id = np.random.randint(gallery_size)
                neg_id = rand_id + int(rand_id >= pos_id)
                anc_inds.append(anc_id)
                pos_inds.append(pos_id)
                neg_inds.append(neg_id)

            anc_batch = sketch_data[anc_inds, ::]
            pos_batch = image_data[pos_inds, ::]
            neg_batch = image_data[neg_inds, ::]

            anc_batch_label = sketch_label[anc_inds]
            pos_batch_label = image_label[pos_inds]
            neg_batch_label = image_label[neg_inds]

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            if random.random() >= 0.5:
                # print "flip"
                anc_batch = anc_batch[:, :, ::-1, :]
                pos_batch = pos_batch[:, :, ::-1, :]
                neg_batch = neg_batch[:, :, ::-1, :]

            anc_batch = remove_mean(anc_batch)
            pos_batch = remove_mean(pos_batch)
            neg_batch = remove_mean(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_label, pos_batch_label, neg_batch_label)


# generate training batches for att_trip network
def tri_data_batch_iter_for_sbirv2(sketch_data, image_data, triplets, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    margin_size = origin_size - crop_size

    triplet_size = len(triplets)
    num_batches_per_epoch = int(np.ceil(triplet_size * 1.0 / batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        indices = np.random.permutation(np.arange(triplet_size))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            batch_indices = np.remainder(np.arange(start_index, start_index + batch_size), triplet_size)
            anc_inds = []
            pos_inds = []
            neg_inds = []
            for index in batch_indices:
                triplet = triplets[indices[index]]
                anc_id = np.random.choice(triplet['anc'])
                pos_id = np.random.choice(triplet['pos'])
                neg_id = np.random.choice(triplet['neg'])
                anc_inds.append(anc_id)
                pos_inds.append(pos_id)
                neg_inds.append(neg_id)

            anc_batch = sketch_data[anc_inds, ::]
            pos_batch = image_data[pos_inds, ::]
            neg_batch = image_data[neg_inds, ::]

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            if random.random() >= 0.5:
                # print "flip"
                anc_batch = anc_batch[:, :, ::-1, :]
                pos_batch = pos_batch[:, :, ::-1, :]
                neg_batch = neg_batch[:, :, ::-1, :]

            anc_batch = remove_mean(anc_batch)
            pos_batch = remove_mean(pos_batch)
            neg_batch = remove_mean(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch)


# generate training batches for att_trip network
def tri_data_id_batch_iter_for_sbirv2(sketch_data, image_data, sketch_id, image_id, triplets, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    margin_size = origin_size - crop_size

    triplet_size = len(triplets)
    num_batches_per_epoch = int(np.ceil(triplet_size * 1.0 / batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        indices = np.random.permutation(np.arange(triplet_size))
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            batch_indices = np.remainder(np.arange(start_index, start_index + batch_size), triplet_size)
            anc_inds = []
            pos_inds = []
            neg_inds = []
            for index in batch_indices:
                triplet = triplets[indices[index]]
                anc_id = np.random.choice(triplet['anc'])
                pos_id = np.random.choice(triplet['pos'])
                neg_id = np.random.choice(triplet['neg'])
                anc_inds.append(anc_id)
                pos_inds.append(pos_id)
                neg_inds.append(neg_id)

            anc_batch = sketch_data[anc_inds, ::]
            pos_batch = image_data[pos_inds, ::]
            neg_batch = image_data[neg_inds, ::]

            anc_batch_id = sketch_id[anc_inds]
            pos_batch_id = image_id[pos_inds]
            neg_batch_id = image_id[neg_inds]

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            if random.random() >= 0.5:
                # print "flip"
                anc_batch = anc_batch[:, :, ::-1, :]
                pos_batch = pos_batch[:, :, ::-1, :]
                neg_batch = neg_batch[:, :, ::-1, :]

            anc_batch = remove_mean(anc_batch)
            pos_batch = remove_mean(pos_batch)
            neg_batch = remove_mean(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_id, pos_batch_id, neg_batch_id)


# generate training batches for att_trip network
def tri_data_label_batch_iter_without_trips(sketch_data, image_data, sketch_label, image_label, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    sketch_data_size = sketch_data.shape[0]
    image_data_size = image_data.shape[0]
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    margin_size = origin_size - crop_size

    num_batches_per_epoch = int(np.ceil(sketch_data_size * 1.0 / batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        shuffle_indices_anc = np.random.permutation(np.arange(sketch_data_size))
        shuffle_indices_pos = np.random.permutation(np.arange(image_data_size))
        shuffle_indices_neg = np.random.permutation(np.arange(image_data_size))

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*num_batches_per_epoch
            batch_indices_anc = np.remainder(np.arange(start_index, start_index + batch_size), sketch_data_size)
            shuffle_batch_indices_anc = batch_indices_anc[np.random.permutation(np.arange(batch_size))]
            batch_indices_pos = np.remainder(np.arange(start_index, start_index + batch_size), image_data_size)
            shuffle_batch_indices_pos = batch_indices_pos[np.random.permutation(np.arange(batch_size))]

            anc_batch = sketch_data[shuffle_indices_anc[shuffle_batch_indices_anc], ::]
            pos_batch = image_data[shuffle_indices_pos[shuffle_batch_indices_pos], ::]
            neg_batch = image_data[shuffle_indices_neg[shuffle_batch_indices_pos], ::]

            anc_batch_label = sketch_label[shuffle_indices_anc[shuffle_batch_indices_anc]]
            pos_batch_label = image_label[shuffle_indices_pos[shuffle_batch_indices_pos]]
            neg_batch_label = image_label[shuffle_indices_neg[shuffle_batch_indices_pos]]

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            anc_batch = anc_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            pos_batch = pos_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
            neg_batch = neg_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            if random.random() >= 0.5:
                # print "flip"
                anc_batch = anc_batch[:, :, ::-1, :]
                pos_batch = pos_batch[:, :, ::-1, :]
                neg_batch = neg_batch[:, :, ::-1, :]

            anc_batch = remove_mean(anc_batch)
            pos_batch = remove_mean(pos_batch)
            neg_batch = remove_mean(neg_batch)

            yield zip(anc_batch, pos_batch, neg_batch, anc_batch_label, pos_batch_label, neg_batch_label)


# generate training batches for att_trip network
def data_label_batch_iter(image_data, image_label, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = image_data.shape[0]
    origin_size = image_data.shape[2]
    crop_size = get_input_size()[0]
    margin_size = origin_size - crop_size

    num_batches_per_epoch = int(np.ceil(data_size * 1.0 / batch_size))
    print "num_batches_per_epoch: ", num_batches_per_epoch
    num_epochs = refine_epochs(num_epochs, num_batches_per_epoch)

    # Shuffle the data at each epoch
    for epoch in range(num_epochs):

        # shuffle trip array, this equals to shuffle the data list, but can keep the relation between data list and hard trip list
        shuffle_indices = np.random.permutation(np.arange(data_size))

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*num_batches_per_epoch
            batch_indices = np.remainder(np.arange(start_index, start_index + batch_size), data_size)
            shuffle_batch_indices = batch_indices[np.random.permutation(np.arange(batch_size))]

            data_batch = image_data[shuffle_indices[shuffle_batch_indices], ::]

            label_batch = image_label[shuffle_indices[shuffle_batch_indices]]

            # crop the concatenated photo
            crop_x = np.random.randint(margin_size); crop_y = np.random.randint(margin_size)
            data_batch = data_batch[:, crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]

            if random.random() >= 0.5:
                # print "flip"
                data_batch = data_batch[:, :, ::-1, :]

            data_batch = remove_mean(data_batch)

            yield zip(data_batch, label_batch)


def feat_tid_batch_iter_for_test(prob_data, gallary_data, batch_size=None, crop=True):
    """
    Generates a batch iterator for a dataset.
    """
    probe_data_size = prob_data.shape[0]
    gallary_data_size = gallary_data.shape[0]
    anc_start_index = 0

    if crop:
        # crop images
        origin_size = prob_data.shape[2]
        crop_size = get_input_size()[0]
        center_margin = abs(origin_size - crop_size - 1) / 2

    num_batches_anc = int(np.ceil(probe_data_size * 1.0 / batch_size))
    num_batches_pos = int(np.ceil(gallary_data_size * 1.0 / batch_size))
    eval_batch_size_anc = batch_size
    eval_batch_size_pos = batch_size

    for anc_batch_num in range(num_batches_anc):
        anc_indices = np.remainder(np.arange(anc_start_index, anc_start_index + eval_batch_size_anc), probe_data_size)
        anc_feat_batch = prob_data[anc_indices, :]
        if eval_batch_size_anc == 1:
            anc_feat_batch = np.tile(anc_feat_batch, (eval_batch_size_pos, 1))
        if crop:
            anc_feat_batch = multi_view_data(anc_feat_batch, center_margin, crop_size, False)
        anc_feat_batch = remove_mean(anc_feat_batch)
        anc_start_index += eval_batch_size_anc
        pos_start_index = 0
        for pos_batch_num in range(num_batches_pos):
            pos_indices = np.remainder(np.arange(pos_start_index, pos_start_index + eval_batch_size_pos), gallary_data_size)
            pos_feat_batch = gallary_data[pos_indices, :]
            if crop:
                pos_feat_batch = multi_view_data(pos_feat_batch, center_margin, crop_size, False)
            pos_feat_batch = remove_mean(pos_feat_batch)
            pos_start_index += eval_batch_size_pos
            yield zip(anc_feat_batch, pos_feat_batch)


def feat_tid_batch_iter_for_test_mini_batch(prob_data, gallary_data, batch_size=None, multi_view=True):
    """
    Generates a batch iterator for a dataset.
    """
    probe_data_size = prob_data.shape[0]
    gallary_data_size = gallary_data.shape[0]
    anc_start_index = 0

    # crop images
    origin_size = prob_data.shape[2]
    crop_size = get_input_size()[0]
    center_margin = abs(origin_size - crop_size - 1) / 2
    crop_flag = 1

    if multi_view:
        num_views = 10
    else:
        num_views = 1
        prob_data = multi_view_data(prob_data, center_margin, crop_size, False, crop_flag)
        gallary_data = multi_view_data(gallary_data, center_margin, crop_size, False, crop_flag)

    num_batches_anc = int(np.ceil(probe_data_size * 1.0 / batch_size))
    num_batches_pos = int(np.ceil(gallary_data_size * 1.0 / batch_size))
    eval_batch_size_anc = batch_size
    eval_batch_size_pos = batch_size

    for anc_batch_num in range(num_batches_anc):
        anc_indices = np.remainder(np.arange(anc_start_index, anc_start_index + eval_batch_size_anc), probe_data_size)
        anc_feat_batch = prob_data[anc_indices, :]
        anc_feat_batch = remove_mean(anc_feat_batch)
        anc_start_index += eval_batch_size_anc
        pos_start_index = 0
        if multi_view:
            anc_feat_batch = multi_view_data(anc_feat_batch, center_margin, crop_size, multi_view, crop_flag)
        for pos_batch_num in range(num_batches_pos):
            pos_indices = np.remainder(np.arange(pos_start_index, pos_start_index + eval_batch_size_pos), gallary_data_size)
            pos_feat_batch = gallary_data[pos_indices, :]
            pos_feat_batch = remove_mean(pos_feat_batch)
            pos_start_index += eval_batch_size_pos
            if multi_view:
                pos_feat_batch = multi_view_data(pos_feat_batch, center_margin, crop_size, multi_view, crop_flag)
                for view in range(num_views):
                    anc_feat_mini_batch = anc_feat_batch[view::num_views, ::]
                    pos_feat_mini_batch = pos_feat_batch[view::num_views, ::]
                    yield zip(anc_feat_mini_batch, pos_feat_mini_batch)
            else:
                yield zip(anc_feat_batch, pos_feat_batch)


def data_iter_for_test(data, batch_size=None, multi_view = False):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = data.shape[0]
    start_index = 0

    if multi_view:
        num_views = 10
    else:
        num_views = 1

    # crop images
    origin_size = data.shape[2]
    crop_size = get_input_size()[0]
    center_margin = abs(origin_size - crop_size - 1) / 2

    num_batches = int(np.ceil(data_size * 1.0 / batch_size))

    for batch_num in range(num_batches):
        indices = np.remainder(np.arange(start_index, start_index + batch_size), data_size)
        start_index += batch_size
        data_batch = data[indices, :]
        data_batch = remove_mean(data_batch)
        if multi_view:
            data_batch = multi_view_data(data_batch, center_margin, crop_size, True)
            for view in range(num_views):
                yield data_batch[view::num_views, :]
        else:
            data_batch = multi_view_data(data_batch, center_margin, crop_size, False)
            yield data_batch


def eval_outer_metric(sketch_data, image_data, train_anchor_data, train_positive_data, train_negative_data, dropout_keep_prob, anc_feat_outer,
                      pos_feat_outer, outer_dists, session, batch_size, multi_view = None):

    if multi_view is None:
        multi_view = FLAGS.multi_view
    if multi_view:
        num_views = 10
    else:
        num_views = 1
    # feat_a, feat_p = [], []
    val_size = sketch_data.shape[0]
    dmat = np.zeros((num_views, val_size, val_size))
    eval_batch_size = batch_size

    num_batches_anc = int(np.ceil(val_size * 1.0 / eval_batch_size))
    num_batches_pos = int(np.ceil(val_size * 1.0 / eval_batch_size))
    eval_batch_size_anc = eval_batch_size
    eval_batch_size_pos = eval_batch_size

    if FLAGS.dataset in ['sketchy', 'sch_com']:
        batches = pair_batch_iter_dev_score_mini_batch(sketch_data, image_data, batch_size, multi_view)

        for idx in range(num_batches_anc):
            start_idx = idx * eval_batch_size_anc
            end_idx = min(start_idx + eval_batch_size_anc, val_size)
            for idy in range(num_batches_pos):
                start_idy = idy * eval_batch_size_pos
                end_idy = min(start_idy + eval_batch_size_pos, val_size)
                for view in range(num_views):
                    # sys.stdout.write('\r>> evaluation iter: (%d/%d, %d/%d, %d/%d)' % (start_idx, val_size, start_idy, val_size, view, num_views))
                    batch = batches.next()
                    anc_batch, pos_batch = zip(*batch)
                    feed_dict = {
                        train_anchor_data: anc_batch,
                        train_positive_data: pos_batch,
                        train_negative_data: pos_batch,
                        dropout_keep_prob: 1.0
                        }
                    feat_a_batch, feat_p_batch, dist_batch = session.run(
                        [anc_feat_outer, pos_feat_outer, outer_dists], feed_dict)
                    dmat[view, start_idx:end_idx, start_idy:end_idy] = dist_batch[:end_idx-start_idx, :end_idy-start_idy]

    else:
        batches = pair_batch_iter_dev_score(sketch_data, image_data, batch_size)

        for view in range(num_views):
            for idx in range(num_batches_anc):
                # sys.stdout.write('\r>> evaluation iter: %d/%d/%d' % (idx, num_batches_anc, view))
                start_idx = idx * eval_batch_size_anc
                end_idx = min(start_idx + eval_batch_size_anc, val_size)
                for idy in range(num_batches_pos):
                    start_idy = idy * eval_batch_size_pos
                    end_idy = min(start_idy + eval_batch_size_pos, val_size)
                    batch = batches.next()
                    anc_batch, pos_batch = zip(*batch)
                    feed_dict = {
                        train_anchor_data: anc_batch,
                        train_positive_data: pos_batch,
                        train_negative_data: pos_batch,
                        dropout_keep_prob: 1.0
                        }

                    feat_a_batch, feat_p_batch, dist_batch = session.run([anc_feat_outer, pos_feat_outer, outer_dists], feed_dict)

                    dmat[view, start_idx:end_idx, start_idy:end_idy] = dist_batch[:end_idx-start_idx, :end_idy-start_idy]
                # if start_idx == 0:
                #     feat_p.extend(feat_p_batch[:end_idy-start_idy, :])
            # feat_a.extend(feat_a_batch[:end_idx-start_idx, :])


    # feat_a = np.array(feat_a)
    # feat_p = np.array(feat_p)

    scores = np.mean(dmat, axis=0)
    if FLAGS.debug_test:
        save_list(scores)
    acc1, acc10 = score_feat_eval(scores)

    return acc1, acc10, scores


def eval_ws_maha_metric(sketch_data, image_data, train_anchor_data, train_positive_data, dropout_keep_prob, anc_feat, pos_feat, weights, batch_size, session, multiview = None):
    feat_a = []
    feat_p = []

    if multiview is None:
        multiview = FLAGS.multi_view
    if multiview:
        num_views = 10
    else:
        num_views = 1

    val_size = sketch_data.shape[0]
    start_iter = 0

    num_batches = num_views * val_size /batch_size


    # Evaluation loop. For each batch...
    if FLAGS.dataset in ['sketchy', 'sch_com']:
        print "Use mini batch for testing, as sketchy is quite large"
        batches = pair_batch_iter_dev_minibatch(sketch_data, image_data, batch_size, multi_view=multiview)
    else:
        batches = pair_batch_iter_dev(sketch_data, image_data, batch_size)

    for batch in batches:

        sys.stdout.write('\r>> evaluation iter: (%d/%d)' % (start_iter, num_batches))
        start_iter += 1


        anc_batch, pos_batch = zip(*batch)

        feed_dict = {
            train_anchor_data: anc_batch,
            train_positive_data: pos_batch,
            dropout_keep_prob: 1.0
        }

        # feat_a_batch, feat_p_batch = session.run([train_anchor, train_positive], feed_dict=feed_dict)
        feat_a_batch, feat_p_batch = session.run([anc_feat, pos_feat], feed_dict=feed_dict)

        feat_a.extend(feat_a_batch)
        feat_p.extend(feat_p_batch)

    feat_a = np.array(feat_a)[:val_size*num_views]
    feat_p = np.array(feat_p)[:val_size*num_views]

    weight = session.run([weights])[0]

    if FLAGS.dataset in ['sketchy', 'sch_com']:
        # use cosine distance here because during training, we trained with normalized feature with tripelt loss
        feat_a, feat_p, dmat, acc1, acc10 = feat_eval(feat_a, feat_p, multi_view=multiview, metric='cosine')
    else:
        if FLAGS.mc == 'maha':
            feat_a, feat_p, dmat, acc1, acc10 = feat_eval(feat_a, feat_p, weight, multi_view=multiview)
        elif FLAGS.mc == 'ws':
            feat_a, feat_p, dmat, acc1, acc10 = feat_eval(feat_a, feat_p, np.diag(weight), multi_view=multiview)


        # sio.savemat('./dist.mat', {'dmat': dmat})

    return acc1, acc10, dmat


def eval_prob_gallery_score(sketch_data, image_data, label_data, train_anchor_data, train_positive_data, dropout_keep_prob, dists, batch_size, session, multi_view = None):

    if multi_view is None:
        multi_view = FLAGS.multi_view

    if multi_view:
        num_views = 10
    else:
        num_views = 1

    if 'dataset' in FLAGS.__flags.keys():
        dataset = FLAGS.dataset
    else:
        dataset = 'sketchy'

    prob_size = sketch_data.shape[0]
    gallery_size = image_data.shape[0]

    # Evaluation loop. For each batch...

    # batch_id = 0
    dmat = np.zeros((num_views, prob_size, gallery_size))
    num_batches_anc = int(np.ceil(prob_size * 1.0 / batch_size))
    num_batches_pos = int(np.ceil(gallery_size * 1.0 / batch_size))
    eval_batch_size_anc = batch_size
    eval_batch_size_pos = batch_size
    anc_size = prob_size
    pos_size = gallery_size

    if multi_view:
        batches = feat_tid_batch_iter_for_test_mini_batch(sketch_data, image_data, batch_size)
    else:
        batches = feat_tid_batch_iter_for_test(sketch_data, image_data, batch_size)
    # Evaluation loop. For each batch...

    for idx in range(num_batches_anc):
        start_idx = idx * eval_batch_size_anc
        end_idx = min(start_idx + eval_batch_size_anc, anc_size)
        for idy in range(num_batches_pos):
            start_idy = idy * eval_batch_size_pos
            end_idy = min(start_idy + eval_batch_size_pos, pos_size)
            for view in range(num_views):
                sys.stdout.write('\r>> evaluation iter: (%d/%d, %d/%d, %d/%d)' % (start_idx, anc_size, start_idy, pos_size, view, num_views))
                sys.stdout.flush()
                batch = batches.next()
                anc_batch, pos_batch = zip(*batch)
                feed_dict = {
                    train_anchor_data: anc_batch,
                    train_positive_data: pos_batch,
                    dropout_keep_prob: 1.0
                }
                dist_batch = session.run([dists], feed_dict)[0]
                dmat[view, start_idx:end_idx, start_idy:end_idy] = dist_batch[:end_idx-start_idx, :end_idy-start_idy]

    s_cid_test = label_data['sketch_class']
    s_label = label_data['sketch_label']
    p_label = label_data['image_label']
    dmat = np.mean(dmat, axis=0)

    acc1, acc10, acc_cls = score_feat_eval_with_label(dmat, s_cid_test, s_label, p_label, dataset=dataset)

    return acc1, acc10, dict({'dmat': dmat}.items() + acc_cls.items())


def eval_classification_accuracy(data, label, model_data, is_train, dropout_keep_prob, prob, batch_size, session, multi_view=False, class_detail = False):

    data_size = data.shape[0]
    if multi_view:
        num_views = 10
    else:
        num_views = 1
    prob_mat = np.zeros((num_views, data_size, FLAGS.num_class))
    num_batches = int(np.ceil(data_size * 1.0 / batch_size))

    batches = data_iter_for_test(data, batch_size, multi_view=multi_view)

    # Evaluation loop. For each batch...
    for idx in range(num_batches):
        start_idx = idx * batch_size
        end_idx = min(start_idx + batch_size, data_size)
        for view in range(num_views):
            data_batch = batches.next()
            feed_dict = {
                model_data: data_batch,
                is_train: False,
                dropout_keep_prob: 1.0
            }
            prob_batch = session.run([prob], feed_dict)[0]
            # est_label[view, start_idx:end_idx] = np.argmax(prob_batch[:end_idx-start_idx, :], axis=1)
            prob_mat[view, start_idx:end_idx, :] = prob_batch[:end_idx-start_idx, :]

    if multi_view:
        prob_mat = np.mean(prob_mat, axis=0)
    else:
        prob_mat = prob_mat[0, :, :]

    est_label = np.argmax(prob_mat, axis=1)

    acc_clf = 100.0 * np.sum((est_label == label).astype(int))/label.shape[0]

    if class_detail:
        acc_clf = []
        num_class = max(label) + 1
        for class_id in range(num_class):
            acc_clf.append(100.0 * np.sum(np.logical_and(est_label == label, label == class_id).astype(int))/np.sum((label == class_id).astype(int)))
    return acc_clf


def eval_model(sketch_data, image_data, model_data, label_data, batch_size):
    # evaluate the performance, not using anymore

    sketch_data = sketch_data['test']
    train_anchor_data = model_data['anc_input']
    session = model_data['session']
    dropout_keep_prob = model_data['dropout_keep_prob']
    if FLAGS.dataset in ['sketchy', 'shoes', 'chairs', 'handbags']:
        image_data = image_data['test']
        train_positive_data = model_data['pos_input']
        train_negative_data = model_data['neg_input']
        anc_feat = model_data['anc_feat']
        pos_feat = model_data['pos_feat']
        anc_feat_outer = model_data['anc_feat_outer']
        pos_feat_outer = model_data['pos_feat_outer']
        weights = model_data['weights']
        dists = model_data['dists']
    acc1, acc10, acc_anc_clf, acc_pos_clf, dmat = -1, -1, -1, -1, -1

    if FLAGS.dataset in ['shoes', 'chairs', 'handbags']:
        # batches = data_work.pair_batch_iter_dev(skh_data_te, img_data_te, batch_size)
        if FLAGS.mc == 'os':
            acc1, acc10, dmat = eval_outer_metric(sketch_data, image_data, train_anchor_data, train_positive_data, train_negative_data, dropout_keep_prob, anc_feat_outer,
                      pos_feat_outer, dists, session, batch_size)
        elif FLAGS.mc == 'ws' or FLAGS.mc == 'maha':
            acc1, acc10, dmat = data_work.eval_ws_maha_metric(sketch_data, image_data, train_anchor_data, train_positive_data, dropout_keep_prob, anc_feat, pos_feat, weights, batch_size, session)
        else:
            acc1, acc10, dmat = eval_euc_metric(sketch_data, image_data, train_anchor_data, train_positive_data, dropout_keep_prob, anc_feat, pos_feat, batch_size, session)
    elif FLAGS.dataset == 'sketchy':
        if 'anc_prob' in model_data.keys() and 'pos_prob' in model_data.keys():
            anc_prob = model_data['anc_prob']
            pos_prob = model_data['pos_prob']
            if not anc_prob is None and not pos_prob is None:
                acc_anc_clf = eval_classification_accuracy(sketch_data, label_data['sketch']['test'], train_anchor_data, dropout_keep_prob, anc_prob, batch_size, session)
                acc_pos_clf = eval_classification_accuracy(image_data, label_data['image']['test'], train_positive_data, dropout_keep_prob, pos_prob, batch_size, session)

        if FLAGS.debug_test and FLAGS.full_test:
            acc1, acc10, dmat = eval_prob_gallery_score(sketch_data, image_data, label_data, train_anchor_data, train_positive_data, dropout_keep_prob, dists, batch_size, session)
        elif FLAGS.mc == 'os':
            acc1, acc10, dmat = eval_outer_metric(sketch_data, image_data, train_anchor_data, train_positive_data, train_negative_data, dropout_keep_prob, anc_feat_outer,
                      pos_feat_outer, dists, session, batch_size)
        elif FLAGS.mc == 'ws' or FLAGS.mc == 'maha':
            acc1, acc10, dmat = data_work.eval_ws_maha_metric(sketch_data, image_data, train_anchor_data, train_positive_data, dropout_keep_prob, anc_feat, pos_feat, weights, batch_size, session)
        elif FLAGS.debug_test and FLAGS.sema_test and 'anc_prob' in model_data.keys() and 'pos_prob' in model_data.keys():
            anc_prob = model_data['anc_prob']
            pos_prob = model_data['pos_prob']
            if not anc_prob is None and not pos_prob is None:
                acc1, acc10, dmat = eval_euc_metric_with_prob(sketch_data, image_data, train_anchor_data, train_positive_data, dropout_keep_prob, anc_feat, pos_feat, anc_prob, pos_prob, batch_size, session)
        else:
            acc1, acc10, dmat = eval_euc_metric(sketch_data, image_data, train_anchor_data, train_positive_data, dropout_keep_prob, anc_feat, pos_feat, batch_size, session)


    elif FLAGS.dataset == 'TU-Berlin':
        if 'anc_prob' in model_data.keys():
            anc_prob = model_data['anc_prob']
            if not anc_prob is None:
                acc_anc_clf = eval_classification_accuracy(sketch_data, label_data['sketch']['test'], train_anchor_data, dropout_keep_prob, anc_prob, batch_size, session, False)
                acc_pos_clf = eval_classification_accuracy(sketch_data, label_data['sketch']['test'], train_anchor_data, dropout_keep_prob, anc_prob, batch_size, session, True)

    if FLAGS.dataset == 'sketchy':
        print("\nAccuracy, Top 1: {0:.2f}%, Classification sketch: {1:.2f}%, Classification image: {2:.2f}%".format(acc1, acc_anc_clf, acc_pos_clf))
    elif FLAGS.dataset == 'TU-Berlin':
        print("\nAccuracy, Single-Acc: {0:.2f}%, Multi-Acc: {1:.2f}%".format(acc_anc_clf, acc_pos_clf))
    elif FLAGS.dataset in ['shoes', 'chairs', 'handbags']:
        print("\nAccuracy, Top 1: {0:.2f}%, Top 10: {1:.2f}%".format(acc1, acc10))
        # acc1 = acc_pos_clf

    return acc1, acc10, acc_anc_clf, acc_pos_clf, dmat


def eval_euc_metric(sketch_data, image_data, train_anchor_data, train_positive_data, is_train, dropout_keep_prob, anc_feat, pos_feat, batch_size, session, subset = 'test', multiview = None):
    feat_a = []
    feat_p = []

    if multiview is None:
        multiview = FLAGS.multi_view

    if multiview:
        num_views = 10
    else:
        num_views = 1

    val_size = sketch_data.shape[0]
    start_iter = 0

    num_batches = int(np.ceil(num_views * val_size*1.0 /batch_size))
    # print "num_batches", num_batches

    # Evaluation loop. For each batch...
    if FLAGS.dataset in ['sketchy', 'sch_com', 'shoesv2', 'chairsv2']:
        print "Use mini batch for testing, and remove mean at each mini batch"
        batches = pair_batch_iter_dev_minibatch(sketch_data, image_data, batch_size, multi_view=multiview)
    else:
        batches = pair_batch_iter_dev(sketch_data, image_data, batch_size, multi_view=multiview)

    for batch in batches:

        sys.stdout.write('\r>> evaluation iter: (%d/%d)' % (start_iter, num_batches))
        start_iter += 1

        anc_batch, pos_batch = zip(*batch)

        feed_dict = {
            train_anchor_data: anc_batch,
            train_positive_data: pos_batch,
            is_train: False,
            dropout_keep_prob: 1.0
        }

        # feat_a_batch, feat_p_batch = session.run([train_anchor, train_positive], feed_dict=feed_dict)
        feat_a_batch, feat_p_batch = session.run([anc_feat, pos_feat], feed_dict=feed_dict)

        feat_a.extend(feat_a_batch)
        feat_p.extend(feat_p_batch)

    feat_a = np.array(feat_a)[:val_size*num_views]
    feat_p = np.array(feat_p)[:val_size*num_views]

    # featdir = '/import/vision-datasets001/Jifei/sbirattmodel/feats/%s' % FLAGS.basenet
    # if FLAGS.pretrained_model == FLAGS.dataset:
    #     featdir += '_finetune'
    #
    # sio.savemat('%s/%s_sketch_feats_%s.mat' % (featdir, FLAGS.dataset, subset), {'feats': feat_a})
    # sio.savemat('%s/%s_photo_feats_%s.mat' % (featdir, FLAGS.dataset, subset), {'feats': feat_p})
    #
    # import pdb
    # pdb.set_trace()

    if FLAGS.dataset in ['sketchy', 'sch_com']:
        # use cosine distance here because during training, we trained with normalized feature with tripelt loss
        feat_a, feat_p, dmat, acc1, acc10 = feat_eval(feat_a, feat_p, multi_view=multiview, metric='cosine')
    else:
        feat_a, feat_p, dmat, acc1, acc10 = feat_eval(feat_a, feat_p, multi_view=multiview)

    # sio.savemat('./feat_a.mat', {'feat_a': feat_a})
    # sio.savemat('./feat_p.mat', {'feat_p': feat_p})
    # sio.savemat('./dist.mat', {'dmat': dmat})

    return acc1, acc10, dmat


def eval_euc_metric_with_prob(sketch_data, image_data, train_anchor_data, train_positive_data, dropout_keep_prob, anc_feat, pos_feat, anc_prob, pos_prob, batch_size, session, multiview = None):
    feat_a = []
    feat_p = []
    prob_a = []
    prob_p = []

    if multiview is None:
        multiview = FLAGS.multi_view

    if multiview:
        num_views = 10
    else:
        num_views = 1

    val_size = sketch_data.shape[0]
    start_iter = 0

    num_batches = int(np.ceil(num_views * val_size *1.0/batch_size))

    # Evaluation loop. For each batch...
    if FLAGS.dataset in ['sketchy', 'sch_com']:
        print "Use mini batch for testing, as sketchy is quite large"
        batches = pair_batch_iter_dev_minibatch(sketch_data, image_data, batch_size, multi_view=multiview)
    else:
        batches = pair_batch_iter_dev(sketch_data, image_data, batch_size)

    for batch in batches:

        sys.stdout.write('\r>> evaluation iter: (%d/%d)' % (start_iter, num_batches))
        start_iter += 1

        anc_batch, pos_batch = zip(*batch)

        feed_dict = {
            train_anchor_data: anc_batch,
            train_positive_data: pos_batch,
            dropout_keep_prob: 1.0
        }

        # feat_a_batch, feat_p_batch = session.run([train_anchor, train_positive], feed_dict=feed_dict)
        feat_a_batch, feat_p_batch, prob_a_batch, prob_p_batch = session.run([anc_feat, pos_feat, anc_prob, pos_prob], feed_dict=feed_dict)

        feat_a.extend(feat_a_batch)
        feat_p.extend(feat_p_batch)

        prob_a.extend(np.argmax(prob_a_batch, axis=1))
        prob_p.extend(np.argmax(prob_p_batch, axis=1))

    feat_a = np.array(feat_a)[:val_size*num_views]
    feat_p = np.array(feat_p)[:val_size*num_views]
    prob_a = np.array(prob_a)[:val_size*num_views]
    prob_p = np.array(prob_p)[:val_size*num_views]

    if FLAGS.dataset in ['sketchy', 'sch_com']:
        # use cosine distance here because during training, we trained with normalized feature with tripelt loss
        feat_a, feat_p, dmat, acc1, acc10 = feat_eval_with_prob(feat_a, feat_p, prob_a, prob_p, multi_view=multiview, metric='cosine')
    else:
        feat_a, feat_p, dmat, acc1, acc10 = feat_eval_with_prob(feat_a, feat_p, prob_a, prob_p, multi_view=multiview)

    # sio.savemat('./dist.mat', {'dmat': dmat})

    return acc1, acc10, dmat


def eval_euc_prob_gallery(sketch_data, image_data, label_data, train_anchor_data, train_positive_data, is_train, dropout_keep_prob, anc_feat, pos_feat, batch_size, session, subset = 'test', multiview = None):

    if multiview is None:
        multiview = FLAGS.multi_view

    if multiview:
        num_views = 10
    else:
        num_views = 1

    if 'dataset' in FLAGS.__flags.keys():
        dataset = FLAGS.dataset
    else:
        dataset = 'sketchy'

    sketch_size = sketch_data.shape[0]
    image_size = image_data.shape[0]

    num_batches_sketch = int(np.ceil(num_views * sketch_size * 1.0 / batch_size))
    num_batches_image = int(np.ceil(num_views * image_size * 1.0 / batch_size))

    # Evaluation loop. For each batch...
    if FLAGS.dataset in ['sketchy', 'sch_com', 'flickr15k', 'shoesv2', 'chairsv2']:
        print "Use mini batch for testing, as sketchy is quite large"
        sketch_batches = batch_iter_dev_minibatch(sketch_data, batch_size, multi_view=multiview)
        image_batches = batch_iter_dev_minibatch(image_data, batch_size, multi_view=multiview)
    else:
        sketch_batches = batch_iter_dev(sketch_data, batch_size, multi_view=multiview)
        image_batches = batch_iter_dev(image_data, batch_size, multi_view=multiview)

    sketch_feat = extract_image_feature(sketch_batches, num_batches_sketch, train_anchor_data, anc_feat, is_train, dropout_keep_prob, session, 'sketch')
    image_feat = extract_image_feature(image_batches, num_batches_image, train_positive_data, pos_feat, is_train, dropout_keep_prob, session, 'image')

    sketch_feat = np.array(sketch_feat)[:sketch_size*num_views]
    image_feat = np.array(image_feat)[:image_size*num_views]

    # featdir = '/import/vision-datasets001/Jifei/sbirattmodel/feats/%s' % FLAGS.basenet
    # if FLAGS.pretrained_model == FLAGS.dataset:
    #     featdir += '_finetune'
    #
    # sio.savemat('%s/%s_sketch_feats_%s.mat' % (featdir, FLAGS.dataset, subset), {'feats': sketch_feat})
    # sio.savemat('%s/%s_photo_feats_%s.mat' % (featdir, FLAGS.dataset, subset), {'feats': image_feat})
    #
    # import pdb
    # pdb.set_trace()

    if dataset == 'sketchy':
        dmat = get_prob_gallery_dmat(sketch_feat, image_feat, multiview, 'cosine')
    else:
        dmat = get_prob_gallery_dmat(sketch_feat, image_feat, multiview)

    s_cid_test = label_data['sketch_class']
    s_label = label_data['sketch_label']
    p_label = label_data['image_label']

    acc1, acc10, acc_cls = score_feat_eval_with_label(dmat, s_cid_test, s_label, p_label, dataset=dataset)

    return acc1, acc10, dict({'dmat': dmat}.items() + acc_cls.items())


def extract_image_feature(batches, num_batches, tf_data, tf_feat, tf_is_train, tf_drop_kp, session, data_type = 'image'):
    feat = []
    start_iter = 0

    for batch in batches:

        start_iter += 1
        sys.stdout.write('\r>> extract %s feature iter: (%d/%d)' % (data_type, start_iter, num_batches))

        feed_dict = {
            tf_data: batch,
            tf_is_train: False,
            tf_drop_kp: 1.0
        }

        feat_batch = session.run([tf_feat], feed_dict=feed_dict)[0]

        feat.extend(feat_batch)

    print ""

    return feat


def get_prob_gallery_dmat(feat_s, feat_p, multi_view = True, metric = 'sqeuclidean'):
    # Retrieval
    if multi_view:
        num_views = 10
    else:
        num_views = 1
    feat_s_size = feat_s.shape[0]
    feat_p_size = feat_p.shape[0]

    s_val_size = int(feat_s_size/num_views)
    p_val_size = int(feat_p_size/num_views)

    feat_dims = feat_s.shape[1]
    feat_s = np.reshape(feat_s, (feat_s_size/num_views, num_views, feat_dims))
    feat_p = np.reshape(feat_p, (feat_p_size/num_views, num_views, feat_dims))
    multi_dmat = np.zeros((num_views, s_val_size, p_val_size))
    for i in xrange(num_views):
        multi_dmat[i,::] = ssd.cdist(feat_s[:, i, :], feat_p[:, i, :], metric)

    dmat = multi_dmat.mean(axis=0)

    return dmat


def feat_eval(feat_s, feat_p, weight = None, multi_view = True, metric = 'sqeuclidean'):
    # Retrieval

    if multi_view:
        num_views = 10
    else:
        num_views = 1
    feat_size = feat_s.shape[0]

    val_size = int(feat_size/num_views)
    # print "feat_size: ", feat_size
    # print "val_size: ", val_size

    feat_dims = feat_s.shape[1]
    feat_s = np.reshape(feat_s, (feat_size/num_views, num_views, feat_dims))
    feat_p = np.reshape(feat_p, (feat_size/num_views, num_views, feat_dims))
    multi_dmat = np.zeros((num_views, val_size, val_size))
    for i in xrange(num_views):
        if weight is None:
            multi_dmat[i,::] = ssd.cdist(feat_s[:, i, :], feat_p[:, i, :], metric)
        else:
            multi_dmat[i,::] = ssd.cdist(feat_s[:, i, :], feat_p[:, i, :], 'mahalanobis', VI=weight)
    dmat = multi_dmat.mean(axis=0)
    dmat_self = np.matrix(np.diag(dmat))
    dmat_self = dmat_self.T.repeat(dmat_self.size, axis=1)

    dist = np.matrix(np.sort(dmat)) - dmat_self
    dist_flag = (dist >= 0).astype(np.single)

    acc1 = np.mean(dist_flag[:, 0], axis=0).item()
    acc10 = np.mean(dist_flag[:, 9], axis=0).item()

    return feat_s, feat_p, dmat, acc1*100, acc10*100


def feat_eval_with_prob(feat_s, feat_p, prob_s, prob_p, weight = None, multi_view = True, metric = 'sqeuclidean'):
    # Retrieval

    if multi_view:
        num_views = 10
    else:
        num_views = 1
    feat_size = feat_s.shape[0]

    val_size = int(feat_size/num_views)
    # print "feat_size: ", feat_size
    # print "val_size: ", val_size

    feat_dims = feat_s.shape[1]
    feat_s = np.reshape(feat_s, (feat_size/num_views, num_views, feat_dims))
    feat_p = np.reshape(feat_p, (feat_size/num_views, num_views, feat_dims))
    prob_s = np.reshape(prob_s, (feat_size/num_views, num_views))
    prob_p = np.reshape(prob_p, (feat_size/num_views, num_views))
    prob_mask = np.abs(prob_s - prob_p)
    multi_dmat = np.zeros((num_views, val_size, val_size))
    for i in xrange(num_views):
        if weight is None:
            multi_dmat[i,::] = ssd.cdist(feat_s[:, i, :], feat_p[:, i, :], metric)
        else:
            multi_dmat[i,::] = ssd.cdist(feat_s[:, i, :], feat_p[:, i, :], 'mahalanobis', VI=weight)
        multi_dmat[i,::] += 10*prob_mask[:, i]

    dmat = multi_dmat.mean(axis=0)
    dmat_self = np.matrix(np.diag(dmat))
    dmat_self = dmat_self.T.repeat(dmat_self.size, axis=1)

    dist = np.matrix(np.sort(dmat)) - dmat_self
    dist_flag = (dist >= 0).astype(np.single)

    acc1 = np.mean(dist_flag[:, 0], axis=0).item()
    acc10 = np.mean(dist_flag[:, 9], axis=0).item()

    return feat_s, feat_p, dmat, acc1*100, acc10*100


def score_feat_eval(dmat):
    # Retrieval

    dmat_self = np.matrix(np.diag(dmat))
    dmat_self = dmat_self.T.repeat(dmat_self.size, axis=1)

    dist = np.matrix(np.sort(dmat)) - dmat_self
    dist_flag = (dist >= 0).astype(np.single)

    acc1 = np.mean(dist_flag[:, 0], axis=0).item()
    acc10 = np.mean(dist_flag[:, 9], axis=0).item()
    return acc1*100, acc10*100


def score_feat_eval_with_label(dmat, src_class, src_label, dst_label, dataset=''):
    # Retrieval

    num_src = len(src_label)

    est_lbs = np.repeat(np.array(dst_label)[np.newaxis, :], num_src, axis=0)
    est_lbs = est_lbs[np.arange(num_src)[:, None], np.argsort(dmat)]

    # check if it is a rank 1
    rank_1_mat = ((est_lbs[:, 0] - src_label) == 0).astype(int)
    rank_1_count = np.sum(rank_1_mat)

    rank_1_all = 100.0 * rank_1_count / num_src

    # check rank 10
    rank_10_mat = ((est_lbs[:, :10] - src_label[:, np.newaxis]) == 0).astype(int)
    rank_10_count = np.sum(rank_10_mat)

    rank_10_all = 100.0 * rank_10_count / num_src

    # calculate acc for each class
    rank_1s = []
    rank_1s_ids = []
    gt_ids = []
    if dataset in ['sketchy', 'flickr15k']:
        if dataset == 'sketchy':
            gt_class_id = src_label/10
        else:
            gt_class_id = src_label
        num_class = max(src_class) + 1
        for class_id in range(num_class):
            match_mat = np.logical_and((est_lbs[:, 0] - src_label) == 0, gt_class_id == class_id)
            rank_1_count_for_this_class = np.sum(match_mat).astype(int)
            num_for_this_class = np.sum(gt_class_id == class_id)
            rank_1s.append(100.0 * rank_1_count_for_this_class / num_for_this_class)
            rank_1s_id = np.arange(match_mat.shape[0])[gt_class_id == class_id][0]
            gt_id = src_label[rank_1s_id]
            rank_1s_ids.append(rank_1s_id)
            gt_ids.append(gt_id)

    return rank_1_all, rank_10_all, {'rank_1s': rank_1s, 'rank_1s_id': rank_1s_ids, 'gt_id': gt_ids}


def save_feats(sketch_data, image_data, model_data, batch_size, dataset, subset = 'test'):

    sketch_data = sketch_data[subset]
    image_data = image_data[subset]
    train_anchor_data = model_data['anc_input']
    train_positive_data = model_data['pos_input']
    anc_feat = model_data['anc_feat']
    pos_feat = model_data['pos_feat']
    session = model_data['session']

    num_views = 10
    feat_size = sketch_data.shape[0] * num_views
    batches = pair_batch_iter_dev(sketch_data, image_data, batch_size)

    feat_a = []
    feat_p = []

    for batch in batches:

        anc_batch, pos_batch = zip(*batch)

        feed_dict = {
            train_anchor_data: anc_batch,
            train_positive_data: pos_batch
        }


        feat_a_batch, feat_p_batch = session.run([anc_feat, pos_feat], feed_dict=feed_dict)

        feat_a.extend(feat_a_batch)
        feat_p.extend(feat_p_batch)

    feat_a = np.array(feat_a)[:feat_size, :]
    feat_p = np.array(feat_p)[:feat_size, :]

    _, _, _, acc1, acc10 = feat_eval(feat_a, feat_p)

    sio.savemat('./feats/%s/%s_sketch_feats.mat' % (subset, dataset), {'im_feats': feat_a})
    sio.savemat('./feats/%s/%s_image_feats.mat' % (subset, dataset), {'im_feats': feat_p})

    return acc1, acc10


def vis_model(sketch_data, image_data, dmat):

    sketch_data = sketch_data['test']
    image_data = image_data['test']

    test_size = 1250

    rand_test = True

    if test_size > dmat.shape[0]:
        test_size = dmat.shape[0]

    if rand_test:
        demo_size = 10
        rand_indices = np.random.choice(test_size, demo_size)
    else:
        rand_indices = np.arange(test_size)
    test_id = 0
    rank_lists = []

    for rand_index in rand_indices:
        print "Case ", test_id, " :"
        print "Query sketch: ", rand_index
        target_id = np.argmin(dmat[rand_index][:])
        rank_list = np.argsort(dmat[rand_index][:])
        rank_lists.append(rank_list[:10])
        print "Top 5 similar photo: ", target_id
        test_id += 1

    vis_rank(sketch_data, image_data, rand_indices, rank_lists, './vis_rank_sketchy.jpg')


def vis_rank(sketch_data, image_data, query_ids, rank_lists, file_name, ground_truth_flag = 1, title_name = None):

    if len(image_data.shape) == 4:
        image_data = np.transpose(image_data, [0, 3, 1, 2])  #b01c to bc01
        sketch_data = np.transpose(sketch_data, [0, 3, 1, 2])  #b01c to bc01S

    if np.max(sketch_data) < 2:
        sketch_data = ((sketch_data / 2.0 + 0.5) * 255).astype(np.uint8)

    if np.max(image_data) < 2:
        image_data = ((image_data / 2.0 + 0.5) * 255).astype(np.uint8)

    nums, chns, rows, cols = image_data.shape
    rank_nums = len(rank_lists[0])
    row_nums = len(query_ids)
    colour_pad = 5

    data = 255 - np.zeros((chns, row_nums * rows, (rank_nums+1) * cols))

    for idx, query_id, rank_list in zip(xrange(row_nums), query_ids, rank_lists):
        startx = idx * rows
        data[:, startx : startx + rows, 0 : cols] = sketch_data[query_id, :, :, :]
        for idy in xrange(1,rank_nums+1):
            starty = idy * cols
            image_id = rank_list[idy-1]
            data[:, startx : startx + rows, starty : starty + cols] = image_data[image_id, :, :, :]
        if query_id in rank_list:
            index = int(np.where(rank_list == query_id)[0])
            startgy = (index  + 1) * cols
            data[:, startx : startx + rows, startgy : startgy + colour_pad] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(rows, axis = 1).repeat(colour_pad, axis = 2)
            data[:, startx : startx + rows, startgy + cols - colour_pad : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(rows, axis = 1).repeat(colour_pad, axis = 2)
            data[:, startx : startx + colour_pad, startgy : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(colour_pad, axis = 1).repeat(rows, axis = 2)
            data[:, startx + rows - colour_pad : startx + rows, startgy : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(colour_pad, axis = 1).repeat(rows, axis = 2)

    if data.dtype != np.uint8:
        if sketch_data is None:
            data = data[:, :, cols:].astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    # fig = plt.figure()
    # if title_name is None:
    #     title_name = 'retrieval result: ' + file_name.split('./')[1].split('rank_')[1]
    # fig.canvas.set_window_title(title_name)
    # plt.axis('off')
    # plt.title(title_name)
    # data = np.asarray(im)
    # transpose, and transfer bgr to rgb
    # im = Image.fromarray(np.roll(data.transpose(1, 2, 0)[...,[1,0,2]], 1, axis=-1))
    # transpose only
    im = Image.fromarray(data.transpose(1, 2, 0))
    # plt.imshow(im)
    # fig.savefig(file_name)
    scipy.misc.imsave(file_name.split('.jpg')[0] + '_scipy.png',im)


def vis_model_dic(sketch_data, image_data, dmat):

    sort_by_rank1 = True
    sort_by_acc = False
    gt_ids = None

    if type(dmat) is dict:
        rank_1s = dmat['rank_1s']
        class_accs = dmat['class_acc']
        rank_1s_ids = dmat['rank_1s_id']
        gt_ids = dmat['gt_id']
        dmat = dmat['dmat']

    sketch_data = sketch_data['test']
    image_data = image_data['test']

    test_size = sketch_data.shape[0]
    demo_size = 5
    rank_size = 10

    # rand_test = True
    rand_test = False

    if rand_test:
        rand_indices = np.random.choice(test_size, demo_size)
    else:
        if sort_by_rank1:
            rand_indices = np.array(rank_1s_ids)[np.argsort(rank_1s)[::-1]][:demo_size]
            print "rank_1s", np.array(rank_1s)[np.argsort(rank_1s)[::-1]][:demo_size]
            gt_ids = np.array(gt_ids)[np.argsort(rank_1s)[::-1]][:demo_size]
        elif sort_by_acc:
            rand_indices = np.array(rank_1s_ids)[np.argsort(class_accs)[::-1]][:demo_size]
            print "class_accs", np.array(class_accs)[np.argsort(class_accs)[::-1]][:demo_size]
            gt_ids = np.array(gt_ids)[np.argsort(class_accs)[::-1]][:demo_size]
        else:
            rand_indices = np.arange(test_size)
    test_id = 0
    rank_lists = []

    for rand_index in rand_indices:
        print "Case ", test_id, " :"
        print "Query sketch: ", rand_index
        target_id = np.argmin(dmat[rand_index][:])
        rank_list = np.argsort(dmat[rand_index][:])
        rank_lists.append(rank_list[:rank_size])
        print "Top 5 similar photo: ", target_id
        test_id += 1

    vis_rank_dic(sketch_data, image_data, rand_indices, rank_lists, './vis_rank_sketchy.jpg', gt_ids)
    print "Visualization result saved in vis_rank_sketchy.jpg"


def vis_rank_dic(sketch_data, image_data, query_ids, rank_lists, file_name, gt_ids = None):

    if len(image_data.shape) == 4:
        image_data = np.transpose(image_data, [0, 3, 1, 2])  #b01c to bc01
        sketch_data = np.transpose(sketch_data, [0, 3, 1, 2])  #b01c to bc01S

    nums, chns, rows, cols = image_data.shape
    rank_nums = len(rank_lists[0])
    row_nums = len(query_ids)
    colour_pad = 5

    if gt_ids is None:
        gt_ids = query_ids # assume the gt_id corresponding to the query id, if not given

    data = 255 - np.zeros((chns, row_nums * rows, (rank_nums+1) * cols))

    for idx, query_id, gt_id, rank_list in zip(xrange(row_nums), query_ids, gt_ids, rank_lists):
        startx = idx * rows
        data[:, startx : startx + rows, 0 : cols] = sketch_data[query_id, :, :, :]
        for idy in xrange(1,rank_nums+1):
            starty = idy * cols
            image_id = rank_list[idy-1]
            data[:, startx : startx + rows, starty : starty + cols] = image_data[image_id, :, :, :]
        if gt_id in rank_list:
            index = int(np.where(rank_list == gt_id)[0])
            startgy = (index  + 1) * cols
            data[:, startx : startx + rows, startgy : startgy + colour_pad] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(rows, axis = 1).repeat(colour_pad, axis = 2)
            data[:, startx : startx + rows, startgy + cols - colour_pad : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(rows, axis = 1).repeat(colour_pad, axis = 2)
            data[:, startx : startx + colour_pad, startgy : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(colour_pad, axis = 1).repeat(rows, axis = 2)
            data[:, startx + rows - colour_pad : startx + rows, startgy : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(colour_pad, axis = 1).repeat(rows, axis = 2)

    if data.dtype != np.uint8:
        if sketch_data is None:
            data = data[:, :, cols:].astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    # fig = plt.figure()
    # if title_name is None:
    #     title_name = 'retrieval result: ' + file_name.split('./')[1].split('rank_')[1]
    # fig.canvas.set_window_title(title_name)
    # plt.axis('off')
    # plt.title(title_name)
    # data = np.asarray(im)
    # transpose, and transfer bgr to rgb
    # im = Image.fromarray(np.roll(data.transpose(1, 2, 0)[...,[1,0,2]], 1, axis=-1))
    # transpose only
    im = Image.fromarray(data.transpose(1, 2, 0))
    # plt.imshow(im)
    # fig.savefig(file_name)
    # scipy.misc.imsave(file_name.split('.jpg')[0] + '_scipy.png',im)
    scipy.misc.imsave(file_name, im)


def vis_rank_single(sketch_data, image_data, query_id, rank_list, file_name, ground_truth_flag = 1, title_name = None):

    if len(image_data.shape) == 4:
        image_data = np.transpose(image_data, [0, 3, 1, 2])  #b01c to bc01
        sketch_data = np.transpose(sketch_data, [0, 3, 1, 2])  #b01c to bc01S

    nums, chns, rows, cols = image_data.shape
    rank_nums = len(rank_list)
    row_rank_nums = rank_nums / 2
    colour_pad = 5

    data = 255 - np.zeros((chns, 2 * rows, (row_rank_nums+1) * cols))
    if not sketch_data is None:
        data[:, :rows, 0 : cols] = sketch_data[query_id, :, :, :]
    for idx in xrange(1,rank_nums+1):
        startx = int((idx - 1) / row_rank_nums) * rows
        starty = ((idx - 1) % row_rank_nums + 1) * cols
        image_id = rank_list[idx-1]
        try:
            data[:, startx : startx + rows, starty : starty + cols] = image_data[image_id, :, :, :]
        except:
            print "e"
    if ground_truth_flag == 1:
        if query_id in rank_list:
            index = np.where(rank_list == query_id)
            startgx = int(index[0] / row_rank_nums) * cols
            startgy = (index[0] % row_rank_nums + 1) * cols
            data[:, startgx : startgx + rows, startgy : startgy + colour_pad] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(rows, axis = 1).repeat(colour_pad, axis = 2)
            data[:, startgx : startgx + rows, startgy + cols - colour_pad : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(rows, axis = 1).repeat(colour_pad, axis = 2)
            data[:, startgx : startgx + colour_pad, startgy : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(colour_pad, axis = 1).repeat(rows, axis = 2)
            data[:, startgx + rows - colour_pad : startgx + rows, startgy : startgy + cols] = \
                np.array([0,255,0])[:,np.newaxis,np.newaxis].repeat(colour_pad, axis = 1).repeat(rows, axis = 2)

    if data.dtype != np.uint8:
        if sketch_data is None:
            data = data[:, :, cols:].astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    # fig = plt.figure()
    # if title_name is None:
    #     title_name = 'retrieval result: ' + file_name.split('./')[1].split('rank_')[1]
    # fig.canvas.set_window_title(title_name)
    # plt.axis('off')
    # plt.title(title_name)
    # data = np.asarray(im)
    im = Image.fromarray(np.roll(data.transpose(1, 2, 0)[...,[1,0,2]], 1, axis=-1))
    # plt.imshow(im)
    # fig.savefig(file_name)
    scipy.misc.imsave(file_name.split('.jpg')[0] + '_scipy.png',im)


def vis_filter(data, padsize=1, padval=0, file_name = './test.png'):
    # this is to visualize conv1 filter only
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data)
    # png.from_array(data).save(file_name)
    scipy.misc.imsave(file_name, data)


def save_list(scores):
    rank_list_name = './rank_list_%s.mat' % FLAGS.dataset
    if not os.path.exists(rank_list_name):
        ranks = np.argsort(scores)[:,:10]
        sio.savemat(rank_list_name, {'rank': ranks})


def merge_sketchy_npy():
    sketch_model = '/import/vision-ephemeral/Jifei/sketchy/pre_trained_model/sketch_convert/sketch_init.npy'
    image_model = '/import/vision-ephemeral/Jifei/sketchy/pre_trained_model/image_convert/image_init.npy'
    merge_model = '/import/vision-ephemeral/Jifei/sketchy/pre_trained_model/triplet_googlenet_sketchy.npy'
    sketch_dict = np.load(sketch_model).item()
    image_dict = np.load(image_model).item()
    merge_dict = sketch_dict
    for key in image_dict.keys():
        if key not in merge_dict.keys():
            merge_dict[key] = image_dict[key]
        else:
            raise Exception('find repeat entriy for two models')
    np.save(merge_model, merge_dict)
    print "Save merged model to %s" % merge_model


def transpose_image_db(image_db_file):
    image_db = sio.loadmat(image_db_file)
    image_db['data'] = np.transpose(image_db['data'], (0, 3, 1, 2))
    sio.savemat(image_db_file, image_db)


if __name__ == "__main__":
    # merge_folder = '/import/vision-ephemeral/Jifei/textclassification/data/office_shoe_merge'
    # # merge_folder = '/import/vision-ephemeral/Jifei/textclassification/data/office_shoe_merge_test'
    # # data_collect(merge_folder)
    #
    # data_path = '/import/vision-ephemeral/Jifei/textclassification/data/shoes.mat'
    # load_data_and_labels(data_path)
    # prepare_full_data()
    # prepare_dev_data()
    # test_pair_cons_id_label()
    merge_sketchy_npy()
