"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave, imresize

import tensorflow as tf
from nets import vgg

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_vgg', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 0.0, 'Momentum.')

FLAGS = tf.flags.FLAGS

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel

kernel = gkern(15, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)



def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB')
    image = imresize(image, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
    image[:,:,0] -= _R_MEAN
    image[:,:,1] -= _G_MEAN
    image[:,:,2] -= _B_MEAN
    images[idx, :, :, :] = image
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      image = images[i, :, :, :]
      image[:,:,0] += _R_MEAN
      image[:,:,1] += _G_MEAN
      image[:,:,2] += _B_MEAN
      image = imresize(image, [299, 299])
      imsave(f, image, format='png')


def graph(x, y, i, x_max, x_min, grad):
  eps = FLAGS.max_epsilon
  num_iter = FLAGS.num_iter
  alpha = eps / num_iter
  momentum = FLAGS.momentum
  num_classes = 1000

  with slim.arg_scope(vgg.vgg_arg_scope()):
      logits, end_points = vgg.vgg_16(
          x, num_classes=num_classes, is_training=False)
            
  pred = tf.argmax(logits, 1)

  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)

  cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                  logits,
                                                  label_smoothing=0.0,
                                                  weights=1.0)
  noise = tf.gradients(cross_entropy, x)[0]
  noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = momentum * grad + noise
  x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise

def stop(x, y, i, x_max, x_min, grad):
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  start_time = time.time()

  eps = FLAGS.max_epsilon

  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = x_input + eps
    x_min = x_input - eps

    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    i = tf.constant(0)
    grad = tf.zeros(shape=batch_shape)
    x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])

    # Run computation
    s1 = tf.train.Saver(slim.get_model_variables())

    print('Building Graph Done', time.time() - start_time)

    with tf.Session() as sess:
      s1.restore(sess, FLAGS.checkpoint_path_vgg)

      print('Load Parameters Done', time.time() - start_time)
      tot_images = 0

      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)

        tot_images += len(filenames)
        print(tot_images, time.time() - start_time)

  print(time.time() - start_time)


if __name__ == '__main__':
  tf.app.run()
