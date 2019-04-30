#!/usr/bin/env python
# python script to transform python files

import argparse
import os
import utils

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/midi',
                        help='data directory containing .mid files to use for' \
                             'training')
    parser.parse_args()
    args = vars(parser.parse_args())
    data_dir = 'midi_files/reformated'
    midi_paths = utils.get_midi_paths(data_dir)
    for p in midi_paths:
        try:
            file_name = os.path.splitext(p)[0]
            dir_path = os.path.dirname(p)
            format0_file = file_name + "_format0.mid"
            if os.path.exists(format0_file):
                os.remove(p)
        except Exception as e:
            print("Unable to reformat {}. Deleting it".format(p))
            os.remove(p)

if __name__ == '__main__':
    main()
