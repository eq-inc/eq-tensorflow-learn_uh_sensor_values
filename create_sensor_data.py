# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import os.path

def concat_data_files():
    finger_max_value = 3

    with open(FLAGS.out_file, "w") as fout:
        for thumb_value in range(finger_max_value):
            for index_value in range(finger_max_value):
                for middle_value in range(finger_max_value):
                    for ring_value in range(finger_max_value):
                        for pinkey_value in range(finger_max_value):
                            file_path = "data/sensor_data_" + str(pinkey_value) + str(ring_value) + str(middle_value) + str(index_value) + str(thumb_value)
                            if os.path.exists(file_path):
                                read_line_count = 0
                                with open(file_path, "r") as fin:
                                    while read_line_count < FLAGS.use_lines:
                                        read_line_count += 1
                                        read_line = fin.readline()
                                        if read_line:
                                            fout.write(read_line)
                                        else:
                                             break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_lines',
        type=int,
        default=5000,
        help='use max lines from 1 data file'
    )
    parser.add_argument(
        '--out_file',
        type=str,
        default="sensor_data.csv",
        help='output file name'
    )

    FLAGS, unparsed = parser.parse_known_args()
    concat_data_files()
