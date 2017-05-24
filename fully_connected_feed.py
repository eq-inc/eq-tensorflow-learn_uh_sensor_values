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
import math
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
DATA_INDEX_AMBIENT_LIGHT = 6
DUMMY_FILE_NAME = "dummy_sensor_data.csv"
READ_SAVED_DATA_BUFFER = []
MAX_FINGER_COUNT = 5
ENABLE_FINGER_COUNT = MAX_FINGER_COUNT
VALIDATE_SENSOR_DATA_SETS = np.array([], dtype=np.float32)
VALIDATE_VALUE_DATA_SETS = np.array([], dtype=np.float32)

class SensorDataFile:
    def __init__(self, sensor_data_file):
        self.sensor_data_file = sensor_data_file
        self.sensor_data_file_desc = None
        self.reach_eof = False
        self.sub_sensor_data_file_array = []

    def __del__(self):
        self.fileClose()

    def __str__(self):
        return self.sensor_data_file + ": " + str(self.reach_eof)

    def readLine(self):
        if self.sensor_data_file_desc == None:
            # open file
            self.sensor_data_file_desc = open(self.sensor_data_file, 'r')

        line = self.sensor_data_file_desc.readline()
        if line == None or len(line) == 0:
            # 本当は正しくないけど、簡易的に判断するようにする
            self.reach_eof = True

        return line

    def isEndOfFile(self):
        return self.reach_eof

    def fileClose(self):
        if self.sensor_data_file_desc != None:
            self.sensor_data_file_desc.close()
            self.sensor_data_file_desc = None
            self.reach_eof = False

        if len(self.sub_sensor_data_file_array) > 0:
            for sub_sensor_data_file in self.sub_sensor_data_file_array:
                sub_sensor_data_file.fileClose()

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

  if num_examples == 0:
      precision = float(true_count) / data_set.num_examples
      print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (data_set.num_examples, true_count, precision))
  else:
      precision = float(true_count) / num_examples
      print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))


def run_training():
    """Train sensor data for a number of steps."""
    # check enable finger count from FLAGS.enable_finger_flags
    ENABLE_FINGER_COUNT = get_enable_finger_count()

    # Get the sets of images and labels for training, validation, and test on uh_sensor_values.
    read_step = FLAGS.batch_size
    max_read_step = FLAGS.max_steps

    with tf.Graph().as_default() as graph:
        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Generate placeholders for the images and labels.
        sensor_values_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        layer_units_array = [get_parameter_data_count()]
        hidden_layer_units_array = FLAGS.hidden_layrer_units.split(',')
        for hidden_layer_units in hidden_layer_units_array:
            layer_units_array.append(int(hidden_layer_units))
        layer_units_array.append(FLAGS.max_finger_condition ** ENABLE_FINGER_COUNT)
        logits = uh_sensor_values.inference(sensor_values_placeholder, layer_units_array)

        # Add to the Graph the Ops for loss calculation.
        loss = uh_sensor_values.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = uh_sensor_values.training(FLAGS.optimizer, loss, FLAGS.learning_rate)

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
        checkpoint = ''
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')

        eof_dict = {}
        data_files = []
        data_file_paths = []
        if FLAGS.random_learning:
            max_read_step, out_file_name = create_random_data_file()
            data_file_paths = [out_file_name]
        else:
            data_file_paths = glob.glob(FLAGS.input_data_dir + "/sensor_data_*")

        total_read_step = 0

        # ファイルパスからSensorDataFileインスタンスへ変更
        for data_file_path in data_file_paths:
            data_files.append(SensorDataFile(data_file_path))

        for data_file in data_files:
            print('%s: ' % data_file)
            offset_step = 0

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
                    if total_read_step % 100 == 0:
                        # Print status to stdout.
                        print('Step %d - %d: loss = %.2f (%.3f sec)' % (total_read_step, total_read_step + read_step, loss_value, duration))
                        # Update the events file.
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, total_read_step)
                        summary_writer.flush()

                    if (FLAGS.max_steps > 0) and ((total_read_step + read_step) % FLAGS.max_steps == 0):
                        # Save a checkpoint and evaluate the model periodically.
                        checkpoint = saver.save(sess, checkpoint_file, global_step=total_read_step)

                    offset_step += read_step
                    total_read_step += read_step
                else:
                    break;

        # Save a checkpoint and evaluate the model periodically.
        checkpoint = saver.save(sess, checkpoint_file, global_step=total_read_step)

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
                                  output_graph_path, clear_devices, "", "")

        # Evaluate against the training set.
        print('Validation Data Eval:')
        global VALIDATE_SENSOR_DATA_SETS
        global VALIDATE_VALUE_DATA_SETS
        new_shape = (int(len(VALIDATE_SENSOR_DATA_SETS) / get_parameter_data_count()), get_parameter_data_count())
        VALIDATE_SENSOR_DATA_SETS = np.reshape(VALIDATE_SENSOR_DATA_SETS, new_shape)
        VALIDATE_SENSOR_DATA_SETS.astype(np.float32)

        train = DataSet(VALIDATE_SENSOR_DATA_SETS, VALIDATE_VALUE_DATA_SETS, dtype=dtypes.uint8, reshape=False)
        data_sets = base.Datasets(train=train, validation=train, test=train)
        if data_sets != None:
            do_eval(sess,
                  eval_correct,
                  sensor_values_placeholder,
                  labels_placeholder,
                  data_sets.train)


def get_enable_finger_count():
    enable_finger_count = 0
    max_finger_count = MAX_FINGER_COUNT
    enable_finger_flags = FLAGS.enable_finger_flags

    for exponent in xrange(max_finger_count):
        if enable_finger_flags % 10 != 0:
            enable_finger_count += 1

        # 桁を下げる
        enable_finger_flags = enable_finger_flags // 10

    return enable_finger_count

def create_random_data_file():
    # ランダムデータを集めたファイルは作成に時間が掛かるので、それは作成せずに、ダミーファイルパスを返却し、コール元でパスにより判断できるようにする
    data_files = glob.glob(FLAGS.input_data_dir + "/sensor_data_*")
    return (FLAGS.max_steps * len(data_files), DUMMY_FILE_NAME)

def read_sensor_data_sets(
                train_data_file,
                dtype=dtypes.uint8,
                reshape=False,
                training=True,
                offset_step=0,
                read_step=500):

    global VALIDATE_SENSOR_DATA_SETS
    global VALIDATE_VALUE_DATA_SETS
    global READ_SAVED_DATA_BUFFER

    sensor_data_sets = np.array([], dtype=np.float32)
    value_data_sets = np.array([], dtype=np.float32)
    no_data = True
    combine_data_line_array = []

    if train_data_file.sensor_data_file == DUMMY_FILE_NAME:
        if (FLAGS.max_steps == 0) or (offset_step < FLAGS.max_steps):
            if len(train_data_file.sub_sensor_data_file_array) == 0:
                data_files = glob.glob(FLAGS.input_data_dir + "/sensor_data_*")
                for data_file in data_files:
                    data_file_flags = data_file[len(FLAGS.input_data_dir + "/sensor_data_"):]
                    enable_finger_flags = FLAGS.enable_finger_flags
                    enable_data_file = True

                    try:
                        data_file_flags_int = int(data_file_flags)
                        for finger_flag_count in xrange(MAX_FINGER_COUNT):
                            if enable_finger_flags % 10 == 0:
                                if data_file_flags_int % 10 != 0:
                                    enable_data_file = False
                                    break
                            data_file_flags_int = data_file_flags_int // 10
                            enable_finger_flags = enable_finger_flags // 10
                    except:
                        enable_data_file = False
                        pass

                    if enable_data_file:
                        train_data_file.sub_sensor_data_file_array.append(SensorDataFile(data_file))

            data_files = train_data_file.sub_sensor_data_file_array
            index_list = list(range(len(data_files)))

            need_read_count = math.ceil((FLAGS.batch_size - len(READ_SAVED_DATA_BUFFER)) / len(data_files))
            while len(READ_SAVED_DATA_BUFFER) < FLAGS.batch_size:
                all_file_empty = False

                for read_count in xrange(need_read_count):
                    empty_file_count = 0
                    random.shuffle(index_list)
                    for file_index in index_list:
                        if (data_files[file_index].isEndOfFile()) == False:
                            read_buffer = data_files[file_index].readLine()
                            if (read_buffer != None) and (len(read_buffer) > 0):
                                READ_SAVED_DATA_BUFFER.append(read_buffer.rstrip("\n").split(','))
                            else:
                                empty_file_count += 1
                        else:
                            empty_file_count += 1

                    if FLAGS.use_same_data_count and empty_file_count > 0:
                        # 全てのファイルからデータを取得できなくなったので、学習は終了させる
                        all_file_empty = True
                        break
                    elif empty_file_count == len(data_files):
                        # 全てのファイルから読み込めなくなったときはあきらめる
                        all_file_empty = True
                        break

                if all_file_empty == True:
                    break

            step_count = 0
            read_step_count = 0
            used_buffer_index = 0
            if len(READ_SAVED_DATA_BUFFER) >= FLAGS.batch_size:
                for line_index in xrange(FLAGS.batch_size):
                    used_buffer_index = line_index
                    if len(READ_SAVED_DATA_BUFFER) <= line_index:
                        break
                    else:
                        no_data = False

                        if read_step_count < read_step:
                            temp_data_array = []
                            temp_data_array.append(READ_SAVED_DATA_BUFFER[line_index])
                            sensor_data_sets, value_data_sets = insert_sensor_data(sensor_data_sets, value_data_sets, temp_data_array, training)
                            step_count+=1
                            read_step_count+=1
                        else:
                            break

                # 使った分は削除する
                READ_SAVED_DATA_BUFFER = []

    else:
        data_file_flags = train_data_file.sensor_data_file[len(FLAGS.input_data_dir + "/sensor_data_"):]
        enable_finger_flags = FLAGS.enable_finger_flags
        enable_data_file = True

        try:
            data_file_flags_int = int(data_file_flags)
            for finger_flag_count in xrange(MAX_FINGER_COUNT):
                if enable_finger_flags % 10 == 0:
                    if data_file_flags_int % 10 != 0:
                        enable_data_file = False
                        break
                data_file_flags_int = data_file_flags_int // 10
                enable_finger_flags = enable_finger_flags // 10
        except:
            enable_data_file = False
            pass

        if enable_data_file:
            while len(READ_SAVED_DATA_BUFFER) < FLAGS.batch_size:
                if (train_data_file.isEndOfFile()) == False:
                    read_buffer = train_data_file.readLine()
                    if (read_buffer != None) and (len(read_buffer) > 0):
                        READ_SAVED_DATA_BUFFER.append(read_buffer.rstrip("\n").split(','))
                    else:
                        break
                else:
                    break

            step_count = 0
            read_step_count = 0
            used_buffer_index = 0
            if len(READ_SAVED_DATA_BUFFER) >= FLAGS.batch_size:
                for line_index in xrange(FLAGS.batch_size):
                    used_buffer_index = line_index
                    if len(READ_SAVED_DATA_BUFFER) <= line_index:
                        break
                    else:
                        no_data = False

                        if read_step_count < read_step:
                            temp_data_array = []
                            temp_data_array.append(READ_SAVED_DATA_BUFFER[line_index])
                            sensor_data_sets, value_data_sets = insert_sensor_data(sensor_data_sets, value_data_sets, temp_data_array, training)
                            step_count+=1
                            read_step_count+=1
                        else:
                            break

                # 使った分は削除する
                READ_SAVED_DATA_BUFFER = []

    if not no_data:
        new_shape = (read_step_count, get_parameter_data_count())
        sensor_data_sets = np.reshape(sensor_data_sets, new_shape)
        sensor_data_sets.astype(np.float32)
        if len(np.atleast_1d(VALIDATE_SENSOR_DATA_SETS)) < (FLAGS.validation_count * get_parameter_data_count()):
            use_data_index = random.randint(0, len(sensor_data_sets) - 1)
            VALIDATE_SENSOR_DATA_SETS = np.append(VALIDATE_SENSOR_DATA_SETS, sensor_data_sets[use_data_index])
            VALIDATE_VALUE_DATA_SETS = np.append(VALIDATE_VALUE_DATA_SETS, value_data_sets[use_data_index])

        train = DataSet(sensor_data_sets, value_data_sets, dtype=dtype, reshape=reshape)

        return base.Datasets(train=train, validation=train, test=train)
    else:
        return None

def insert_sensor_data(sensor_data_sets, value_data_sets, combine_data_line_array, training):
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
            elif sensor_data_set_index == DATA_INDEX_AMBIENT_LIGHT and FLAGS.use_ambient_light:
                data_array = sensor_data_set_array[sensor_data_set_index].split('_')

            if data_array != None:
                for data in data_array:
                    if data != "null":
                        if FLAGS.expand_data_size > 0:
                            # dataをFLAGS.expand_data_size分まで0詰め
                            data = data.zfill(FLAGS.expand_data_size)
                            data_byte_array = data.encode()
                            for data_byte in data_byte_array:
                                sensor_data_sets = np.append(sensor_data_sets, float(data_byte))
                        else:
                            # dataをそのまま使用
                            sensor_data_sets = np.append(sensor_data_sets, data)

        if data_index == (len(combine_data_line_array) - 1):
            if training == True:
                # use value
                try:
                    value_data_sets = np.append(value_data_sets, combine_data_line_array[data_index][1])
                except IndexError:
                    print("combine_data_line_array: %s" % combine_data_line_array)

    # remove first data, because it is not of range for combination
    del combine_data_line_array[0]

    return (sensor_data_sets, value_data_sets)

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
    if FLAGS.use_ambient_light:
        ret_unit += 1

    if FLAGS.expand_data_size > 0:
        return (ret_unit * FLAGS.expand_data_size * FLAGS.combine_data_line_count)
    else:
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
        '--validation_count',
        type=int,
        default=1000,
        help=''
    )
    parser.add_argument(
        '--hidden_layrer_units',
        type=str,
        default='8,4',
        help='Number of units in hidden layers.'
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
        type=int,
        default=1,
        help='0: no random learning, none 0: random learning'
    )
    parser.add_argument(
        '--use_accel',
        default=False,
        action='store_true',
        help='use accel for learning'
    )
    parser.add_argument(
        '--use_gyro',
        default=False,
        action='store_true',
        help='use gyro for learning'
    )
    parser.add_argument(
        '--use_photo',
        default=False,
        action='store_true',
        help='use photo reflector for learning'
    )
    parser.add_argument(
        '--use_angle',
        default=False,
        action='store_true',
        help='use angle for learning'
    )
    parser.add_argument(
        '--use_temperature',
        default=False,
        action='store_true',
        help='use temperature for learning'
    )
    parser.add_argument(
        '--use_quaternion',
        default=False,
        action='store_true',
        help='use quaternion for learning'
    )
    parser.add_argument(
        '--use_ambient_light',
        default=False,
        action='store_true',
        help='use ambient light for learning'
    )
    parser.add_argument(
        '--expand_data_size',
        type=int,
        default=0,
        help='0: As is'
    )
    parser.add_argument(
        '--use_same_data_count',
        default=False,
        action='store_true',
        help='use same data from each data files'
    )
    parser.add_argument(
        '--enable_finger_flags',
        type=int,
        default=11111,
        help='0: disable, none 0: enable, PRMIT order'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='tf.train.AdamOptimizer',
        help=''
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
