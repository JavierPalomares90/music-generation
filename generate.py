#!/usr/bin/env python
import argparse, os, pdb
import train
import utils


def main():
    args = utils.parse_generate_args()
        # prime file validation
    prime_file = args.prime_file
    data_dir = args.data_dir
    
    experiment_dir = args.experiment_dir
    utils.log('Using {} as --experiment_dir'.format(experiment_dir), args.verbose)

    if prime_file and not os.path.exists(prime_file):
        utils.log('Error: prime file {} does not exist. Exiting.'.format(prime_file))
        exit(1)
    elif not os.path.isdir(data_dir):
        utils.log('Error: data dir {} does not exist. Exiting.'.format(data_dir))
        exit(1)

    if prime_file:
        midi_files = prime_file
    else:
        midi_files = utils.get_midi_paths(data_dir)
    
    save_dir = args.save_dir
    if not save_dir:
        save_dir = os.path.join(experiment_dir,'generated_midi')
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        utis.log("Created save dir {}".format(save_dir))

    utils.log("Saving generated_midis to {}".format(save_dir))
    model, epoch = train.get_model(args, experiment_dir=experiment_dir)
    utils.log("Loaded model from {}".format(os.path.join(experiment_dir,"model.json")))

    window_size = model.layers[0].get_input_shape_at(0)[1]
    seed_generator = utils.get_midi_data_generator(midi_files, 
                                              window_size=window_size,
                                              batch_size=32,
                                              num_threads=1,
                                              max_files_in_ram=10)

    # generate tracks using random seeds
    utils.log('Loading seed files {}'.format(midi_files))
    X, y = next(seed_generator)
    generated = utils.generate(model, X, window_size, args.file_length, args.num_files)
    for i, midi in enumerate(generated):
        file = os.path.join(args.save_dir, '{}.mid'.format(i + 1))
        midi.write(file.format(i + 1))
        utils.log('wrote midi file to {}'.format(file), True)

if __name__ == '__main__':
    main()