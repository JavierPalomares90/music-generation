#!/usr/bin/env python
import utils
import os, argparse, time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# use 20 percent for validation
VALIDATION_SPLIT_RATE = 0.2


def main():
    args = utils.parse_args()
    data_dir = args.data_dir
    midi_files = utils.get_midi_files_path(data_dir)
    experiment_dir = utils.create_experiment_dir(args.experiment_dir)

    val_split_index = int(midi_files * VALIDATION_SPLIT_RATE)



if __name__ == '__main__':
    main()