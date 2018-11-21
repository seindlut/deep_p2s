import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception, resnet_v1, vgg

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def get_input_paras():
    input_paras = {'batch_size': FLAGS.batch_size, 'num_epochs': FLAGS.num_epochs}
    crop_size, channel_size = data_work.get_input_size()
    if FLAGS.image_type == 'rgb':
        if FLAGS.basenet in ['inceptionv1', 'inceptionv3', 'mobilenet']:
            input_paras['mean'] = [0, 0, 0]  # mean_bgr
        else:
            input_paras['mean'] = [104.0069879317889, 116.66876761696767, 122.6789143406786]  # mean_bgr
    else:
        input_paras['mean'] = 250.42
    input_paras['im_size'] = 256
    input_paras['cp_size'] = crop_size
    input_paras['chns'] = channel_size
    return crop_size, channel_size, input_paras


def build_classifier(identity, label, num_class, add_hidden = False, add_logits_layer=True, reuse=False):

    # classifier
    with tf.variable_scope("sketch_image_classifier", reuse=reuse):
        # calculate loss
        if add_logits_layer:
            print('Created additional logits layer')
            logit, pred = classifier(identity, num_class, add_hidden)
        else:
            print('Havn''t create additional logits layer')
            logit = identity
            pred = tf.nn.softmax(logit)
        # calculate acc
        clf_loss, clf_acc = compute_classification_loss_acc(logit, pred, label, num_class, 'single')
        return logit, pred, clf_loss, clf_acc


def classifier(inputs, num_classes, add_hidden = False, scope='classifier'):
    """classifier.
     Args:
       inputs: a tensor of size [batch_size, feature_dims].
       num_classes: number of predicted classes.
     Returns:
       logits code.
     """
    with tf.name_scope(scope):
        with tf.variable_scope('classifier'):
            if add_hidden:
                fc = slim.fully_connected(inputs, num_classes*2, activation_fn=None, scope='fc')
                logits = slim.fully_connected(fc, num_classes, activation_fn=None, scope='clf')
            else:
                logits = slim.fully_connected(inputs, num_classes, activation_fn=None, scope='clf')
            preds = tf.nn.softmax(logits, name='predictions')
            return logits, preds


def compute_classification_loss_acc(logits, preds, label_ids, num_classes, str = ''):
    with tf.name_scope("%s_classification_loss" % str):
        labels = convert_labels_to_dense(label_ids, num_classes)
        # import pdb
        # pdb.set_trace()
        # clf_loss = slim.losses.cross_entropy_loss(logits, labels, label_smoothing=0.1, weight=1.0)
        # clf_loss = slim.losses.softmax_cross_entropy(logits, labels, label_smoothing=0.1, weight=1.0)
        clf_loss = slim.losses.softmax_cross_entropy(logits, labels, label_smoothing=0.1)
        clf_acc = slim.metrics.accuracy(tf.argmax(preds, 1), tf.argmax(labels, 1))

    return clf_loss, clf_acc


def convert_labels_to_dense(labels, num_classes):
    # Reshape the labels into a dense Tensor of
    # shape [FLAGS.batch_size, num_classes].
    batch_size = labels.get_shape().as_list()[0]
    labels = tf.cast(labels, tf.int32)
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat([indices, sparse_labels], axis=1)
    output_labels = tf.sparse_to_dense(concated, [batch_size, num_classes], 1.0, 0.0)  # dense labels

    return output_labels


def build_triplets(anc_feat, pos_feat, neg_feat, margin):

    # norm feature and calculate the triplet loss
    anc_feat, pos_feat, neg_feat = norm_feats(anc_feat, pos_feat, neg_feat)
    tri_loss = compute_triplet_loss_with_norm(anc_feat, pos_feat, neg_feat, margin)[0]

    return tri_loss


def build_triplets_with_dists(d_p_squared, d_n_squared, margin):

    with tf.name_scope("triplet_loss_with_dists"):
        tri_loss = tf.maximum(0., margin + d_p_squared - d_n_squared)
        return tf.reduce_mean(tri_loss)


def norm_feats(anchor_feature, positive_feature, negative_feature):
    norm_anchor_feature = tf.nn.l2_normalize(anchor_feature, dim=1)
    norm_positive_feature = tf.nn.l2_normalize(positive_feature, dim=1)
    norm_negative_feature = tf.nn.l2_normalize(negative_feature, dim=1)
    return norm_anchor_feature, norm_positive_feature, norm_negative_feature


def compute_triplet_loss_with_norm(anchor_feature, positive_feature, negative_feature, margin):
    with tf.name_scope("triplet_loss"):
        # tf.reduce_sum(tf.square(anchor_feature - positive_feature), axis=1)
        norm_anchor_feature = tf.nn.l2_normalize(anchor_feature, dim=1)
        norm_positive_feature = tf.nn.l2_normalize(positive_feature, dim=1)
        norm_negative_feature = tf.nn.l2_normalize(negative_feature, dim=1)
        d_p_squared = square_distance(norm_anchor_feature, norm_positive_feature)
        d_n_squared = square_distance(norm_anchor_feature, norm_negative_feature)

        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)


def square_distance(x, y):
    return tf.reduce_sum(tf.square(x - y), axis=1)


def build_mlp(anc_input, pos_input, neg_input):

    anc_feat = slim.fully_connected(anc_input, 512, activation_fn=None, scope='sketch_mlp')
    pos_feat = slim.fully_connected(pos_input, 512, activation_fn=None, scope='image_mlp')
    neg_feat = slim.fully_connected(neg_input, 512, activation_fn=None, scope='image_mlp', reuse=True)
    return anc_feat, pos_feat, neg_feat


def build_mlp_for_photo(pos_input, neg_input, rd_dim):

    pos_feat = slim.fully_connected(pos_input, rd_dim, activation_fn=None, scope='image_mlp')
    neg_feat = slim.fully_connected(neg_input, rd_dim, activation_fn=None, scope='image_mlp', reuse=True)
    return pos_feat, neg_feat


def sketch_a_net_slim(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=True):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            # x = tf.reshape(inputs, shape=[-1, 225, 225, 1])
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', scope='conv5_s1')
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            att_f = slim.flatten(conv5)
            fc6 = slim.fully_connected(att_f, 512, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, scope='fc7_sketch')
    return fc7


def build_single_vggnet(train_tfdata, is_train, dropout_keep_prob):
    if not FLAGS.is_train or FLAGS.debug_test:
        is_train = False
    with slim.arg_scope(vgg.vgg_arg_scope()):
        identity, end_points = vgg.vgg_16(train_tfdata, num_classes=FLAGS.num_class, is_training=is_train, dropout_keep_prob = dropout_keep_prob)
        # identity, end_points = vgg.vgg_19(train_tfdata, num_classes=FLAGS.num_class, is_training=is_train, dropout_keep_prob = dropout_keep_prob)
        for key in end_points.keys():
            if 'fc7' in key:
                feature = tf.squeeze(end_points[key], [1, 2])
    return identity, feature


def build_single_resnet(train_tfdata, is_train, name_scope = 'resnet_v1_50', variable_scope = ''):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=is_train)):
        identity, end_points = resnet_v1.resnet_v1_50(train_tfdata, num_classes=FLAGS.num_class, global_pool = True)
        feature = slim.flatten(tf.get_default_graph().get_tensor_by_name('%s%s/pool5:0' % (variable_scope, name_scope)))
    return identity, feature


def build_single_inceptionv1(train_tfdata, is_train, dropout_keep_prob):
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        identity, end_points = inception.inception_v1(train_tfdata, dropout_keep_prob = dropout_keep_prob, is_training=is_train)
        net = slim.avg_pool2d(end_points['Mixed_5c'], [7, 7], stride=1, scope='MaxPool_0a_7x7')
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout_0b')
        feature = tf.squeeze(net, [1, 2])
    return identity, feature


def build_single_inceptionv3(train_tfdata, is_train, dropout_keep_prob, reduce_dim = False):
    train_tfdata_resize = tf.image.resize_images(train_tfdata, (299, 299))
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        identity, end_points = inception.inception_v3(train_tfdata_resize, dropout_keep_prob = dropout_keep_prob, is_training=is_train)
        feature = slim.flatten(end_points['Mixed_7c'])
        if reduce_dim:
            feature = slim.fully_connected(feature, 256, scope='feat')
    return identity, feature


def generative_cnn_encoder(inputs, is_training=True, drop_keep_prob=0.5, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
        o_c1 = general_conv2d(inputs, 32, is_training=is_training, name="CNN_1")
        o_c2 = general_conv2d(o_c1, 64, is_training=is_training, name="CNN_2")
        o_c3 = general_conv2d(o_c2, 128, is_training=is_training, name="CNN_3")
        o_c4 = general_conv2d(o_c3, 256, is_training=is_training, name="CNN_4")
        o_c5 = general_conv2d(o_c4, 256, is_training=is_training, name="CNN_5")
        o_c5 = tf.reshape(o_c5, (-1, 256 * 7 * 7))
        o_c6 = linear1d(o_c5, 256 * 7 * 7, 512, name='CNN_FC')
        # o_c6 = tf.cond(is_training, lambda: tf.nn.dropout(o_c6, 0.5), lambda: o_c6)
        o_c6 = tf.nn.dropout(o_c6, drop_keep_prob)

        return o_c6


def generative_cnn_decoder(inputs, is_training=True, drop_keep_prob=0.5, reuse=False):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
        o_d1 = linear1d(inputs, 128, 256 * 7 * 7, name='CNN_DEC_FC')
        # o_d1 = tf.cond(is_training, lambda: tf.nn.dropout(o_d1, 0.5), lambda: o_d1)
        o_d1 = tf.nn.dropout(o_d1, drop_keep_prob)
        o_d1 = tf.reshape(o_d1, [-1, 7, 7, 256])
        o_d2 = general_deconv2d(o_d1, 256, is_training=is_training, name="CNN_DEC_1")
        o_d3 = general_deconv2d(o_d2, 128, is_training=is_training, name="CNN_DEC_2")
        o_d4 = general_deconv2d(o_d3, 64, is_training=is_training, name="CNN_DEC_3")
        o_d5 = general_deconv2d(o_d4, 32, is_training=is_training, name="CNN_DEC_4")
        # o_d6 = general_deconv2d(o_d5, 3, is_training=is_training, name="CNN_DEC_5", do_relu=False, do_tanh=True)
        o_d6 = general_deconv2d(o_d5, 3, name="CNN_DEC_5", do_norm=False, do_relu=False, do_tanh=True)

        return o_d6


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name) as scope:
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def instance_norm_bk(x):
    with tf.variable_scope("instance_norm") as scope:
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return out


def linear1d(inputlin, inputdim, outputdim, name="linear1d", std=0.02, mn=0.0):
    with tf.variable_scope(name) as scope:
        weight = tf.get_variable("weight", [inputdim, outputdim])
        bias = tf.get_variable("bias", [outputdim], dtype=np.float32, initializer=tf.constant_initializer(0.0))

        return tf.matmul(inputlin, weight) + bias


def general_conv2d(inputconv, output_dim=64, filter_height=4, filter_width=4, stride_height=2, stride_width=2,
                   stddev=0.02, padding="SAME", name="conv2d", do_norm=True, norm_type='instance_norm', do_relu=True,
                   relufactor=0, is_training=True):
    with tf.variable_scope(name) as scope:

        conv = tf.contrib.layers.conv2d(inputconv, output_dim, [filter_width, filter_height],
                                        [stride_width, stride_height], padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            if norm_type == 'instance_norm':
                conv = instance_norm(conv)
            elif norm_type == 'batch_norm':
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=is_training, updates_collections=None,
                                                    epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_conv2d_bk(inputconv, output_dim=64, filter_height=4, filter_width=4, stride_height=2, stride_width=2,
                   stddev=0.02, padding="SAME", name="conv2d", do_norm=True, norm_type='batch_norm', do_relu=True,
                   relufactor=0, is_training=True):
    with tf.variable_scope(name) as scope:

        conv = tf.contrib.layers.conv2d(inputconv, output_dim, [filter_width, filter_height],
                                        [stride_width, stride_height], padding, activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            if norm_type == 'instance_norm':
                conv = instance_norm(conv)
            elif norm_type == 'batch_norm':
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=is_training, updates_collections=None,
                                                    epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_deconv2d(inputconv, output_dim=64, filter_height=4, filter_width=4, stride_height=2, stride_width=2,
                     stddev=0.02, padding="SAME", name="deconv2d", do_norm=True, norm_type='instance_norm', do_relu=True,
                     relufactor=0, do_tanh=False, is_training=True):
    with tf.variable_scope(name) as scope:

        conv = tf.contrib.layers.conv2d_transpose(inputconv, output_dim, [filter_height, filter_width],
                                                  [stride_height, stride_width], padding, activation_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                  biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            if norm_type == 'instance_norm':
                conv = instance_norm(conv)
            elif norm_type == 'batch_norm':
                conv = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=is_training, updates_collections=None,
                                                    epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if (relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        if do_tanh:
            conv = tf.nn.tanh(conv, "tanh")

        return conv


def rnn_discriminator(x, x_l, cell_type, n_hidden, num_layers, in_dp, out_dp, batch_size, reuse=False):
    with tf.variable_scope('RNN_DIS', reuse=reuse) as rnn_dis_scope:
        # encode sketch
        temp_cells = []
        for idx in range(num_layers):
            if cell_type == "lstm":
                temp_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            elif cell_type == "gru":
                temp_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
            elif cell_type == "lstm-layerNorm":
                temp_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden)
            else:
                temp_cell = tf.nn.rnn_cell.RNNCell(n_hidden)
            temp_cells.append(temp_cell)
        rnn_cell = tf.contrib.rnn.MultiRNNCell(temp_cells)

        if out_dp != 1.0:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=out_dp)

        init_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        outputs, stateFinal = tf.nn.dynamic_rnn(rnn_cell, x, sequence_length=x_l, initial_state=init_state, dtype=tf.float32, scope=rnn_dis_scope)

        batch_range = tf.range(batch_size)
        indices = tf.stack([batch_range, x_l-1], axis=1)
        last_output = tf.gather_nd(outputs, indices)

        # classifier
        logits = slim.fully_connected(last_output, 1, activation_fn=None, scope='clf')
        preds = tf.nn.softmax(logits)
        return preds, logits


def cnn_discriminator(inputs, batch_size, is_training=True, reuse=False):
    with tf.variable_scope('CNN_DIS', reuse=reuse):

        o_h1 = general_conv2d(inputs, 32, is_training=is_training, do_norm=False, name="CNN_DIS_1")
        o_h2 = general_conv2d(o_h1, 64, is_training=is_training, name="CNN_DIS_2")
        o_h3 = general_conv2d(o_h2, 128, is_training=is_training, name="CNN_DIS_3")
        o_h4 = general_conv2d(o_h3, 256, stride_height=1, stride_width=1, is_training=is_training, name="CNN_DIS_4")
        o_h5 = general_conv2d(o_h4, 1, stride_height=1, stride_width=1, do_norm=False, is_training=is_training,
                              name="CNN_DIS_5")

        # classifier
        last_output = tf.reshape(o_h5, [batch_size, -1])
        logits = slim.fully_connected(last_output, 1, activation_fn=None, scope='clf')
        preds = tf.nn.softmax(logits)

        return preds, logits


def wgan_gp_rnn_discriminator(inputs, dim=32, reuse=False):
    "inputs shape: batch_size*max_sequence*2"
    with tf.variable_scope('RNN_DIS', reuse=reuse) as scope:
        input = tf.expand_dims(tf.reshape(inputs, [-1, int(inputs.get_shape()[1]) * int(inputs.get_shape()[2])]), 2)
        input = general_resBlock(input, name='WGAN_GP1D_CONV1', dim=dim)
        input = general_resBlock(input, name='WGAN_GP1D_CONV2', dim=dim)
        input = general_resBlock(input, name='WGAN_GP1D_CONV3', dim=dim)
        input = general_resBlock(input, name='WGAN_GP1D_CONV4', dim=dim)
        input = general_resBlock(input, name='WGAN_GP1D_CONV5', dim=dim)
        input = tf.reshape(input, [-1, int(inputs.get_shape()[1]) * dim])
        input = linear1d(input, int(inputs.get_shape()[1]) * dim, 1, name='WGAN_GP1D_LIN')
        return input


def wgan_gp_cnn_discriminator(inputs, dim=32, reuse=False):
    with tf.variable_scope('CNN_DIS', reuse=reuse) as scope:
        o_h1 = general_conv2d(inputs, dim, do_norm=False, relufactor=0.2, name="WGAN_GP2D_CONV1")  # 112
        o_h2 = general_conv2d(o_h1, dim * 2, do_norm=False, relufactor=0.2, name="WGAN_GP2D_CONV2")  # 56
        o_h3 = general_conv2d(o_h2, dim * 4, do_norm=False, relufactor=0.2, name="WGAN_GP2D_CONV3")  # 28
        o_h4 = general_conv2d(o_h3, dim * 8, do_norm=False, relufactor=0.2, name="WGAN_GP2D_CONV4")  # 14
        o_h5 = general_conv2d(o_h4, dim * 8, do_norm=False, relufactor=0.2, name="WGAN_GP2D_CONV5")  # 7
        o_h6 = tf.reshape(o_h5, [-1, 7 * 7 * dim * 8])
        o_h7 = linear1d(o_h6, 7 * 7 * dim * 8, 1, name='WGAN_GP2D_LIN')

        return o_h7


def wgan_gp_loss(fake_logits, real_logits, gradients, LAMBDA=10, use_gradients=True):
    """"
    how to get gradient:
        alpha = tf.random_uniform(
        shape=[batch_size, 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_input-real_input
    interpolates = real_input + (alpha*differences)
    gradients = tf.gradients(wgan_gp_rnn_discriminator(interpolates, reuse=reuse), [interpolates])[0]

    """
    fake_logits = tf.nn.sigmoid(fake_logits)
    real_logits = tf.nn.sigmoid(real_logits)
    disc_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
    gen_cost = -tf.reduce_mean(fake_logits)

    if use_gradients:
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        disc_cost += LAMBDA * gradient_penalty

    return disc_cost, gen_cost


def general_resBlock(inputs, name='res', dim=64):
    output = inputs
    output = tf.nn.relu(output)
    output = general_conv1d(output, dim, 10, name=name + '_1')
    output = tf.nn.relu(output)
    output = general_conv1d(output, dim, 10, name=name + '_2')
    return inputs + (0.3 * output)


def general_conv1d(inputconv, output_dim, filter_size, stride=1, stddev=0.02, name='conv1d'):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', [filter_size, inputconv.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv1d(inputconv, w, stride=stride, padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def get_other_op(global_step):
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    batchnorm_updates_op = tf.group(*batchnorm_updates)
    return variables_averages_op, batchnorm_updates_op


def init_variables(model_file='/import/vision-ephemeral/Jifei/sbirattmodel/pretrained_model/model-iter9000.npy'):
    d = np.load(model_file).item()
    pretrained_paras = d.keys()
    init_ops = []  # a list of operations
    for var in tf.global_variables():
        for w_name in pretrained_paras:
            if w_name in var.name:
                print('Initialise var %s with weight %s' % (var.name, w_name))
                try:
                    if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        # init_ops.append(var.assign(d[w_name+'/weights:0']))
                        init_ops.append(var.assign(d[w_name]['weights']))
                    elif 'biases' in var.name:
                        # init_ops.append(var.assign(d[w_name+'/biases:0']))
                        init_ops.append(var.assign(d[w_name]['biases']))
                except KeyError:
                     if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        init_ops.append(var.assign(d[w_name+'/weights:0']))
                        # init_ops.append(var.assign(d[w_name]['weights']))
                     elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name+'/biases:0']))
                        # init_ops.append(var.assign(d[w_name]['biases']))
                except:
                     if 'weights' in var.name:
                        # using assign(src, dst) to assign the weights of pre-trained model to current network
                        init_ops.append(var.assign(d[w_name][0]))
                        # init_ops.append(var.assign(d[w_name]['weights']))
                     elif 'biases' in var.name:
                        init_ops.append(var.assign(d[w_name][1]))
                        # init_ops.append(var.assign(d[w_name]['biases']))
    return init_ops


def init_variables_v2(model_file):
    # pretrained_paras = ['conv1_s1', 'conv2_s1', 'conv3_s1', 'conv4_s1', 'conv5_s1', 'fc6_s1', 'fc7_sketch',
                        # 'att_conv1', 'att_conv2']
    d = np.load(model_file).item()
    pretrained_paras = d.keys()
    # print(pretrained_paras)
    init_ops = []  # a list of operations
    var_initialized = []
    for var in tf.global_variables():
        for w_name in pretrained_paras:
            if w_name in var.name:
                print('Initialise var %s with weight %s' % (var.name, w_name))
                init_ops.append(var.assign(d[w_name]))
                var_initialized.append(var)
    for var in tf.global_variables():
        if var not in var_initialized:
            print "Variable %s not initialized ", var.name
    if not init_ops:
        print "Warning: no variable is initialized"
    return init_ops


def load_npy_model(model_file, include_scope_str, exclude_scopes, include_scope_str_ref = False):
    d = np.load(model_file).item()
    init_ops = []
    model_variables = [var for var in tf.trainable_variables() if include_scope_str in var.name]
    for var in model_variables:
        if not include_scope_str_ref:
            varName = var.name
        else:
            varName_appendix = var.name.split(include_scope_str)[-1]
            ref_keys = [key for key in d.keys() if varName_appendix in key]
            if len(ref_keys) == 1:
                varName = ref_keys[0]
            else:
                raise Exception('More than one refer keys has been found')
        excluded = False
        for exclude_scope in exclude_scopes:
            if var.op.name.startswith(exclude_scope):
                excluded = True
                break
        if not excluded:
            init_ops.append(var.assign(d[varName]))
            print (varName)
        else:
            print ("%s not initialized due to exclusion" % varName)
    return init_ops


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


def get_rcons_loss_l2(y1_data, y2_data, t1_data, t2_data, pen_data):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).

    result_xy_1 = tf.square(t1_data - y1_data)
    result_xy_2 = tf.square(t2_data - y2_data)

    result = result_xy_1 + result_xy_2

    fs = 1.0 - pen_data[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    # Zero out loss terms beyond N_s, the last actual stroke
    result = tf.multiply(result, fs)

    return result


def get_rcons_loss_mdn(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data, pen_data):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).

    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                           z_corr)
    epsilon = 1e-6
    # result1 is the loss wrt pen offset (L_s in equation 9 of
    # https://arxiv.org/pdf/1704.03477.pdf)
    result = tf.multiply(result0, z_pi)
    result = tf.reduce_sum(result, 1, keep_dims=True)
    result = -tf.log(result + epsilon)  # avoid log(0)

    fs = 1.0 - pen_data[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    # Zero out loss terms beyond N_s, the last actual stroke
    result = tf.multiply(result, fs)

    return result


def get_rcons_loss_pen_state(z_pen_logits, pen_data, hps_is_training):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).

    # result2: loss wrt pen state, (L_p in equation 9)
    result = tf.nn.softmax_cross_entropy_with_logits(
        labels=pen_data, logits=z_pen_logits)
    result = tf.reshape(result, [-1, 1])
    # if not hps_is_training:  # eval mode, mask eos columns
    #     fs = 1.0 - pen_data[:, 2]
    #     fs = tf.reshape(fs, [-1, 1])
    #     result = tf.multiply(result, fs)

    return result


# below is where we need to do MDN (Mixture Density Network) splitting of
# distribution params
def get_mixture_coef(output):
    """Returns the tf slices containing mdn dist params."""
    # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
    z = output
    z_pen_logits = z[:, 0:3]  # pen states
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

    # process output z's into MDN paramters

    # softmax all the pi's and pen states:
    z_pi = tf.nn.softmax(z_pi)
    z_pen = tf.nn.softmax(z_pen_logits)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)  # \rho

    r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
    return r


# clip gradients
def clip_gradients(grad_and_vars, clip_value):
    # g = self.hps.grad_clip
    # import pdb
    # pdb.set_trace()
    capped_gvs = []
    # capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
    for grad, var in grad_and_vars:
        if grad is not None:
            capped_gvs.append((tf.clip_by_value(grad, -clip_value, clip_value), var))
        else:
            capped_gvs.append((grad, var))
    return capped_gvs


# below is how to build the adversarial training loss
def adversarial_loss_dis_real(logits_r, weight=1.0):
    loss = weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_r, labels=tf.ones_like(logits_r)))
    return loss


def adversarial_loss_dis_fake(logits_f, weight=1.0):
    loss = weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_f, labels=tf.zeros_like(logits_f)))
    return loss


def adversarial_loss_gen_fake(logits_f, weight=1.0):
    loss = weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_f, labels=tf.ones_like(logits_f)))
    return loss


def get_adv_loss(logits_r, logits_f, labels_r, labels_f):
    dis_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_r, labels=tf.cast(labels_r, tf.float32)))
    dis_f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_f, labels=tf.cast(labels_f, tf.float32)))

    dis_loss = dis_r_loss + dis_f_loss

    dis_r_acc = slim.metrics.accuracy(tf.cast(tf.round(tf.nn.sigmoid(logits_r)), tf.int32), tf.round(labels_r))
    dis_f_acc = slim.metrics.accuracy(tf.cast(tf.round(tf.nn.sigmoid(logits_f)), tf.int32), tf.round(labels_f))

    dis_acc = (dis_r_acc + dis_f_acc) / 2

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_f, labels=tf.cast(labels_r, tf.float32)))

    return dis_loss, gen_loss, dis_acc


def get_adv_gp_loss(logits_r, logits_f, labels_r, labels_f, gradients, LAMBDA=10, use_gradients=True):
    dis_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_r, labels=tf.cast(labels_r, tf.float32)))
    dis_f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_f, labels=tf.cast(labels_f, tf.float32)))

    dis_loss = dis_r_loss + dis_f_loss

    dis_r_acc = slim.metrics.accuracy(tf.cast(tf.round(tf.nn.sigmoid(logits_r)), tf.int32), tf.round(labels_r))
    dis_f_acc = slim.metrics.accuracy(tf.cast(tf.round(tf.nn.sigmoid(logits_f)), tf.int32), tf.round(labels_f))

    dis_acc = (dis_r_acc + dis_f_acc) / 2

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_f, labels=tf.cast(labels_r, tf.float32)))

    if use_gradients:
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        dis_loss += LAMBDA * gradient_penalty

    return dis_loss, gen_loss, dis_acc