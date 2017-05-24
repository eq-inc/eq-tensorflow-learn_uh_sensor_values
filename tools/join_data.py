# coding: utf-8
"""Join sensor data files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import glob
import os
import sys

# Basic model parameters as external flags.
FLAGS = None

def join_sensor_data_file():
    target_directory_paths = glob.glob(FLAGS.join_dir_prefix + '*')

    if len(target_directory_paths) > 0:
        # 結合結果を保存するディレクトリが存在しない場合は作成
        if not os.path.exists(FLAGS.out_dir):
            print('make out dir: ' + FLAGS.out_dir)
            os.makedirs(FLAGS.out_dir)

        # 結合対象のディレクトリに対してループ
        for target_directory_path in target_directory_paths:
            target_file_paths = glob.glob(target_directory_path + os.sep + 'sensor_data_*')

            if len(target_file_paths) > 0:
                for target_file_path in target_file_paths:
                    # 結合結果を保存するディレクトリに同一のファイル名称のファイルを作成し、追記していく
                    split_target_file_path = target_file_path.split(os.sep)
                    target_file_name = split_target_file_path[len(split_target_file_path) - 1]
                    with open(FLAGS.out_dir + os.sep + target_file_name, 'a') as fout:
                        for line in open(target_file_path, 'r'):
                            fout.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--join_dir_prefix',
        type=str,
        default='..' + os.sep + 'data' + os.sep + 'cond_',
        help='target join sensor data directory name prefix'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='..' + os.sep + 'data' + os.sep + 'join_out',
        help='output directory for joined files'
    )

    FLAGS, unparsed = parser.parse_known_args()
    join_sensor_data_file()
