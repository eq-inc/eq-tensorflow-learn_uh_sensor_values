# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import csv

from six.moves import xrange  # pylint: disable=redefined-builtin
import mnist as mnist
import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes

# Basic model parameters as external flags.
FLAGS = None

def run_training():
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # load saved model
    try:
      meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], FLAGS.saved_data_dir)
    except IOError:
      pass

    tempArray = []

def main(_):
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--saved_data_dir',
      type=str,
      # default='/tmp/saved_model_half_plus_two',
      default='./saved_model_half_plus_two',
      help='Directory to restore the saved data.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
