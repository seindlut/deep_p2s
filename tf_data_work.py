# modeified from image processing in inceptionv3
# revised at 01/10/2017, By Jeffrey

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import pdb
# from model import image_processing_test

FLAGS = tf.app.flags.FLAGS


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.
    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.
    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.
    Returns:
      the cropped (and resized) image.
    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    cropped_shape = control_flow_ops.with_dependencies(
        [rank_assertion],
        tf.pack([crop_height, crop_width, original_shape[2]]))

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.pack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = control_flow_ops.with_dependencies(
        [size_assertion],
        tf.slice(image, offsets, cropped_shape))
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.
    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:
      image, depths, normals = _random_crop([image, depths, normals], 120, 150)
    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.
    Returns:
      the image_list with cropped images.
    Raises:
      ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    image_shape = control_flow_ops.with_dependencies(
        [rank_assertions[0]],
        tf.shape(image_list[0]))
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                                   tf.shape(image))
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    max_offset_height = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_height - crop_height + 1, []))
    max_offset_width = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_width - crop_width + 1, []))
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.
    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.
    Returns:
      the list of cropped images.
    """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def decode_png(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    # with tf.name_scope([image_buffer], scope, 'decode_png'):
    # Decode the string as an RGB JPEG.
    image = tf.image.decode_png(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
    Returns:
    color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def distort_image(image, height, width, chn_size=3, bbox=None, flipr=False, scope=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
    Returns:
    3-D float Tensor of distorted image used for training.
    """
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
    with tf.op_scope([image, height, width, bbox], scope, 'distort_image'):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an allowed
        # range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=[[[0.0, 0.0, 1.0, 1.0]]],
            min_object_covered=0.1,
            aspect_ratio_range=[0.9, 1.1],
            area_range=[0.8, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        # if not thread_id:
        #     image_with_distorted_box = tf.image.draw_bounding_boxes(
        #       tf.expand_dims(image, 0), distort_bbox)
        #     tf.image_summary('images_with_distorted_bounding_box',
        #                    image_with_distorted_box)

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        distorted_image = tf.image.resize_images(distorted_image, [height, width])
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, chn_size])

        # Randomly flip the image horizontally.
        if flipr:
            distorted_image = tf.image.random_flip_left_right(distorted_image)

        # # Randomly distort the colors.
        # distorted_image = distort_color(distorted_image)

        return distorted_image


def processing_image(image_buffer, thread_id=0, data_augmentation_flag=True):

    image = decode_png(image_buffer)

    if data_augmentation_flag:
        image = distort_image(image, FLAGS.output_height, FLAGS.output_width, thread_id)
    # else:
    #     image = eval_image(image, height, width)

    image.set_shape([FLAGS.output_height, FLAGS.output_width, 3])
    image = tf.image.resize_images(image, [FLAGS.resize_height, FLAGS.resize_width])
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    # image = _central_crop([image], FLAGS.crop_height, FLAGS.crop_width)[0]
    image.set_shape([FLAGS.resize_height, FLAGS.resize_width, 3])
    image = tf.to_float(image)

    return image


def call_distort_image(image):
    crop_size = FLAGS.crop_size
    dist_chn_size = FLAGS.dist_chn_size
    return distort_image(image, crop_size, crop_size, dist_chn_size)


def data_augmentation(raw_images):
    # processed_images = tf.map_fn(lambda inputs: call_distort_image(*inputs), elems=raw_images, dtype=tf.float32)
    processed_images = tf.map_fn(lambda inputs: call_distort_image(inputs), raw_images)
    return processed_images


def tf_high_pass_filter(tf_images):
    # implementation of high pass filter according to the "Sketch-pix2seq"
    filter_w = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], tf.float32)
    filter_w = tf.expand_dims(tf.expand_dims(filter_w, -1), -1, name='hp_w')
    filtered_images = tf.nn.conv2d(tf_images, filter_w, strides=[1, 1, 1, 1], padding='SAME')
    return filtered_images


def tf_image_processing(tf_images, basenet, crop_size, distort=False, hp_filter=False):
    if len(tf_images.shape) == 3:
        tf_images = tf.expand_dims(tf_images, -1)
    if basenet == 'sketchanet':
        mean_value = 250.42
        tf_images = tf.subtract(tf_images, mean_value)
        if distort:
            print("Distorting photos")
            FLAGS.crop_size = crop_size
            FLAGS.dist_chn_size = 1
            tf_images = data_augmentation(tf_images)
        else:
            tf_images = tf.image.resize_images(tf_images, (crop_size, crop_size))
    elif basenet in ['inceptionv1', 'inceptionv3', 'gen_cnn']:
        tf_images = tf.divide(tf_images, 255.0)
        tf_images = tf.subtract(tf_images, 0.5)
        tf_images = tf.multiply(tf_images, 2.0)
        if int(tf_images.shape[-1]) != 3:
            tf_images = tf.concat([tf_images, tf_images, tf_images], axis=-1)
        if distort:
            print("Distorting photos")
            FLAGS.crop_size = crop_size
            FLAGS.dist_chn_size = 3
            tf_images = data_augmentation(tf_images)
            # Display the training images in the visualizer.
            # tf.image_summary('input_images', input_images)
        else:
            tf_images = tf.image.resize_images(tf_images, (crop_size, crop_size))

    if hp_filter:
        tf_images = tf_high_pass_filter(tf_images)

    return tf_images
