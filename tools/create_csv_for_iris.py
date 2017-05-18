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

# Basic model parameters as external flags.
FLAGS = None

DATA_INDEX_ACCEL = 0
DATA_INDEX_GYRO = 1
DATA_INDEX_PHOTO_REFLECTOR = 2
DATA_INDEX_ANGLE = 3
DATA_INDEX_TEMPERATURE = 4
DATA_INDEX_QUATERNION = 5

def create_csv_file():
    data_file_paths = glob.glob(FLAGS.input_data_dir + "/sensor_data_*")

    with open('out.csv', 'w') as fout:
        for data_file_path in data_file_paths:
            with open(data_file_path, 'r') as fin:
                csv_data_sets = csv.reader(fin)

                for row in csv_data_sets:
                    no_data = False

                    sensor_data_set_array = row[0].split('+')
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
                                fout.write(data)
                                fout.write(',')

                    fout.write(row[1])
                    fout.write('\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='../data',
      help='Directory to put the input data.'
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

  FLAGS, unparsed = parser.parse_known_args()
  create_csv_file()
