#!/usr/bin/env python
# python script to transform python files
import utils
from pymidifile import *
import argparse

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/midi',
                        help='data directory containing .mid files to use for' \
                             'training')
    parser.parse_args()
    args = vars(parser.parse_args())
    data_dir = args["data_dir"]
    midi_paths = utils.get_midi_paths(data_dir)
    for p in midi_paths:
        reformat_midi(p,verbose=False,write_to_file=True)



if __name__ == '__main__':
    main()

