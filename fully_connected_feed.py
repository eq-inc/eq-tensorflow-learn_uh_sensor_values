# coding: utf-8
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Trains and Evaluates the Unlimited Hand - Finger condition network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import csv
import glob
import os.path
import random
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import uh_sensor_values as uh_sensor_values
import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph

# Basic model parameters as external flags.
FLAGS = None

DATA_INDEX_ACCEL = 0
DATA_INDEX_GYRO = 1
DATA_INDEX_PHOTO_REFLECTOR = 2
DATA_INDEX_ANGLE = 3
DATA_INDEX_TEMPERATURE = 4
DATA_INDEX_QUATERNION = 5

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    sensor_values_placeholder: Sensor values placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # sensor values and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  sensor_values_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         get_parameter_data_count()),
                                                         name="sensor_values_placeholder")
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name="labels_placeholder")
  return sensor_values_placeholder, labels_placeholder


def fill_feed_dict(data_set, sensor_values_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    sensor_values_pl: The sensor values placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  sensor_values_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      sensor_values_pl: sensor_values_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            sensor_values_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    sensor_values_placeholder: The sensor values placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of sensor values and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               sensor_values_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
    """Train sensor data for a number of steps."""
    # Get the sets of images and labels for training, validation, and test on uh_sensor_values.
    read_step = FLAGS.batch_size
    max_read_step = FLAGS.max_steps

    with tf.Graph().as_default() as graph:
        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Generate placeholders for the images and labels.
        sensor_values_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = uh_sensor_values.inference(
                        FLAGS.max_finger_condition ** 5,
                        sensor_values_placeholder,
                        get_parameter_data_count(),
                        FLAGS.hidden1,
                        FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = uh_sensor_values.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = uh_sensor_values.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = uh_sensor_values.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(max_to_keep=FLAGS.max_save_checkpoint)

        # load checkpoint
        checkpoint = ''
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        try:
            saver.restore(sess, checkpoint_file)
        except errors.NotFoundError:
            pass

        data_files = []
        if FLAGS.random_learning:
            max_read_step, out_file_name = create_random_data_file()
            data_files = [out_file_name]
        else:
            data_files = glob.glob(FLAGS.input_data_dir + "/sensor_data_*")

        total_read_step = len(data_files) * FLAGS.offset

        for data_file in data_files:
            print('%s: ' % data_file)
            start_offset_step = offset_step = FLAGS.offset

            while True:
                # read data_sets from CVS
                data_sets = read_sensor_data_sets(data_file, offset_step=offset_step, read_step=read_step)

                if data_sets != None:
                    # Start the training loop.
                    start_time = time.time()

                    # Fill a feed dictionary with the actual set of images and labels
                    # for this particular training step.
                    feed_dict = fill_feed_dict(data_sets.train,
                                         sensor_values_placeholder,
                                         labels_placeholder)

                    # Run one step of the model.  The return values are the activations
                    # from the `train_op` (which is discarded) and the `loss` Op.  To
                    # inspect the values of your Ops or variables, you may include them
                    # in the list passed to sess.run() and the value tensors will be
                    # returned in the tuple from the call.
                    _, loss_value = sess.run([train_op, loss],
                                       feed_dict=feed_dict)

                    duration = time.time() - start_time

                    # Write the summaries and print an overview fairly often.
                    # Print status to stdout.
                    print('Step %d - %d: loss = %.2f (%.3f sec)' % (offset_step, offset_step + read_step, loss_value, duration))
                    # Update the events file.
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, offset_step)
                    summary_writer.flush()

                    # Save a checkpoint and evaluate the model periodically.
                    checkpoint = saver.save(sess, checkpoint_file, meta_graph_suffix='meta', write_meta_graph=True, global_step=total_read_step, write_state=True, latest_filename='checkpoint_state_name')

                    offset_step += read_step
                    total_read_step += read_step

                    if (max_read_step != 0) and (offset_step >= (start_offset_step + max_read_step) or (offset_step + FLAGS.combine_data_line_count >= max_read_step)):
                        # Evaluate against the training set.
                        print('Training Data Eval:')
                        do_eval(sess,
                              eval_correct,
                              sensor_values_placeholder,
                              labels_placeholder,
                              data_sets.train)
                        # Evaluate against the validation set.
                        print('Validation Data Eval:')
                        do_eval(sess,
                              eval_correct,
                              sensor_values_placeholder,
                              labels_placeholder,
                              data_sets.validation)
                        # Evaluate against the test set.
                        print('Test Data Eval:')
                        do_eval(sess,
                              eval_correct,
                              sensor_values_placeholder,
                              labels_placeholder,
                              data_sets.test)

                        break
                else:
                    break;

        graph_io.write_graph(sess.graph, FLAGS.saved_data_dir, "saved_data.pb", as_text=False)
        input_binary = True

        input_graph_path = os.path.join(FLAGS.saved_data_dir, "saved_data.pb")
        input_saver = ""
        output_node_names = "eval_correct"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_graph_path = os.path.join(FLAGS.saved_data_dir, "saved_data_out.pb")
        clear_devices = False

        freeze_graph.freeze_graph(input_graph_path, input_saver,
                                  input_binary, checkpoint, output_node_names,
                                  restore_op_name, filename_tensor_name,
                                  output_graph_path, clear_devices, "", "xentropy_mean")

def create_random_data_file():
    read_line = 0
    out_file_name = 'temp_saved_data.csv'
    data_files = glob.glob(FLAGS.input_data_dir + "/sensor_data_*")
    index_list = list(range(len(data_files)))

    with open(out_file_name, "wb") as fout:
        read_offset = 0
        for line in xrange(FLAGS.max_steps):
            random.shuffle(index_list)
            for file_index in index_list:
                with open(data_files[file_index], 'r') as fin:
                    # 指定されているoffset+これまでに読み込んだ分、読み捨てる
                    for remove_line in xrange(FLAGS.offset + read_offset):
                        fin.readline()

                    read_buffer = fin.readline()
                    if read_buffer != None:
                        fout.write(bytes(read_buffer, 'UTF-8'))
                        read_line += 1

            read_offset += 1

    return (read_line, out_file_name)


def read_sensor_data_sets(train_data_file,
                   dtype=dtypes.uint8,
                   reshape=False,
                   training=True,
                   offset_step=0,
                   read_step=500):

    sensor_data_sets = np.array([], dtype=np.float32)
    value_data_sets = np.array([], dtype=np.float32)
    no_data = True
    combine_data_line_count = FLAGS.combine_data_line_count
    combine_data_line_array = []

    with open(train_data_file, 'r') as f:
        csv_data_sets = csv.reader(f)

        step_count = 0
        read_step_count = 0

        for row in csv_data_sets:
            if step_count < offset_step:
                step_count+=1
            else:
                no_data = False

                # save combination data in list
                combine_data_line_array.append(row)
                if read_step_count < read_step:
                    if len(combine_data_line_array) == combine_data_line_count:
                        for data_index in xrange(len(combine_data_line_array)):
                            sensor_data_set_array = combine_data_line_array[data_index][0].split('+')

                            for sensor_data_set_index in xrange(len(sensor_data_set_array)):
                                data_array = None

                                if sensor_data_set_index == DATA_INDEX_ACCEL and FLAGS.use_accel:
                                    data_array = sensor_data_set_array[sensor_data_set_index].split('_')
                                elif sensor_data_set_index == DATA_INDEX_GYRO and FLAGS.use_gyro:
                                    data_array = sensor_data_set_array[sensor_data_set_index].split('_')
                                elif sensor_data_set_index == DATA_INDEX_PHOTO_REFLECTOR and FLAGS.use_photo:
                                    data_array = sensor_data_set_array[sensor_data_set_index].split('_')
                                elif sensor_data_set_index == DATA_INDEX_ANGLE and FLAGS.use_angle:
                                    data_array = sensor_data_set_array[sensor_data_set_index].split('_')
                                elif sensor_data_set_index == DATA_INDEX_TEMPERATURE and FLAGS.use_temperature:
                                    data_array = sensor_data_set_array[sensor_data_set_index].split('_')
                                elif sensor_data_set_index == DATA_INDEX_QUATERNION and FLAGS.use_quaternion:
                                    data_array = sensor_data_set_array[sensor_data_set_index].split('_')

                                if data_array != None:
                                    for data in data_array:
                                        if data != "null":
                                            data_byte_array = data.encode()
                                            for data_byte in data_byte_array:
                                                sensor_data_sets = np.append(sensor_data_sets, float(data_byte))

                            if data_index == (len(combine_data_line_array) - 1):
                                if training == True:
                                    # use value
                                    value_data_sets = np.append(value_data_sets, combine_data_line_array[data_index][1])

                        step_count+=1
                        read_step_count+=1
                        sys.stdout.write("read line: %d\r\b" % (step_count))
                        sys.stdout.flush()

                        # remove first data, because it is not of range for combination
                        del combine_data_line_array[0]
                else:
                    break

        if len(combine_data_line_array) < combine_data_line_count:
            no_data = True

    if not no_data:
        new_shape = (read_step_count, get_parameter_data_count())
        sensor_data_sets = np.reshape(sensor_data_sets, new_shape)
        sensor_data_sets.astype(np.float32)
        train = DataSet(sensor_data_sets, value_data_sets, dtype=dtype, reshape=reshape)

        return base.Datasets(train=train, validation=train, test=train)
    else:
        return None

def get_parameter_data_count():
    ret_unit = 0

    if FLAGS.use_accel:
        ret_unit += 3
    if FLAGS.use_gyro:
        ret_unit += 3
    if FLAGS.use_photo:
        ret_unit += 8
    if FLAGS.use_angle:
        ret_unit += 3
    if FLAGS.use_temperature:
        ret_unit += 1
    if FLAGS.use_quaternion:
        ret_unit += 4

    return (ret_unit * FLAGS.combine_data_line_count)

def main(_):
    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    # tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--debug_log',
      default=False,
      help='enable debug log'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=0,
      help='0: unlimited'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=8,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=4,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='./checkpoint',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--max_save_checkpoint',
      type=int,
      default=1000,
      help=''
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )
  parser.add_argument(
      '--saved_data_dir',
      type=str,
      default='./saved_train_data',
      help='Directory to restore the saved data.'
  )
  parser.add_argument(
      '--load_saved_data',
      default=False,
      help='load saved data and evaluate with it'
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      help=''
  )
  parser.add_argument(
      '--combine_data_line_count',
      type=int,
      default=1,
      help=''
  )
  parser.add_argument(
      '--max_finger_condition',
      type=int,
      default=2,
      help='0: straight, 1: curve'
  )
  parser.add_argument(
      '--random_learning',
      default=False,
      help=''
  )
  parser.add_argument(
      '--use_accel',
      default=False,
      help='use accel for learning'
  )
  parser.add_argument(
      '--use_gyro',
      default=False,
      help='use gyro for learning'
  )
  parser.add_argument(
      '--use_photo',
      default=True,
      help='use photo reflector for learning'
  )
  parser.add_argument(
      '--use_angle',
      default=True,
      help='use angle for learning'
  )
  parser.add_argument(
      '--use_temperature',
      default=False,
      help='use temperature for learning'
  )
  parser.add_argument(
      '--use_quaternion',
      default=False,
      help='use quaternion for learning'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
