#!/usr/bin/env python
import os, argparse, time

def parse_args():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/midi',
                        help='data directory containing .mid files to use for' \
                             'training')
    parser.add_argument('--experiment_dir', type=str,
                        default='experiments/default',
                        help='directory to store checkpointed models and tensorboard logs.' \
                             'if omitted, will create a new numbered folder in experiments/.')
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='learning rate. If not specified, the recommended learning '\
                        'rate for the chosen optimizer is used.')
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for RNN input per step.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs before stopping training.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='percentage of weights that are turned off every training '\
                        'set step. This is a popular regularization that can help with '\
                        'overfitting. Recommended values are 0.2-0.5')
    parser.add_argument('--optimizer', 
                        choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 
                                 'adam', 'adamax', 'nadam'], default='adam',
                        help='The optimization algorithm to use. '\
                        'See https://keras.io/optimizers for a full list of optimizers.')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='clip gradients at this value.')
    parser.add_argument('--message', '-m', type=str,
                        help='a note to self about the experiment saved to message.txt '\
                        'in --experiment_dir.')
    parser.add_argument('--n_jobs', '-j', type=int, default=1, 
                        help='Number of CPUs to use when loading and parsing midi files.')
    parser.add_argument('--max_files_in_ram', default=25, type=int,
                        help='The maximum number of midi files to load into RAM at once.'\
                        ' A higher value trains faster but uses more RAM. A lower value '\
                        'uses less RAM but takes significantly longer to train.')
    return parser.parse_args()

def get_midi_paths(dir):
    # Find all the midi files in the directory
    paths = []
    for root,dirs,files in os.walk(dir):
        for file in files:
            if file.endswith(".mid"):
                paths.append(os.path.join(root,file))
    return paths

# if the experiment dir doesn't exist create it and its subfolders
def create_experiment_dir(experiment_dir, verbose=False):
    
    # if the experiment directory was specified and already exists
    if experiment_dir != 'experiments/default' and \
       os.path.exists(experiment_dir):
    	# raise an error
    	raise Exception('Error: Invalid --experiemnt_dir, {} already exists' \
    		            .format(experiment_dir))

    # if the experiment directory was not specified, create a new numeric folder
    if experiment_dir == 'experiments/default':
    	
    	experiments = os.listdir('experiments')
    	experiments = [dir_ for dir_ in experiments \
    	               if os.path.isdir(os.path.join('experiments', dir_))]
    	
    	most_recent_exp = 0
    	for dir_ in experiments:
    		try:
    			most_recent_exp = max(int(dir_), most_recent_exp)
    		except ValueError as e:
    			# ignrore non-numeric folders in experiments/
    			pass

    	experiment_dir = os.path.join('experiments', 
    		                          str(most_recent_exp + 1).rjust(2, '0'))

    os.mkdir(experiment_dir)
    log('Created experiment directory {}'.format(experiment_dir), verbose)
    os.mkdir(os.path.join(experiment_dir, 'checkpoints'))
    log('Created checkpoint directory {}'.format(os.path.join(experiment_dir, 'checkpoints')),
    	verbose)
    os.mkdir(os.path.join(experiment_dir, 'tensorboard-logs'))
    log('Created log directory {}'.format(os.path.join(experiment_dir, 'tensorboard-logs')), 
    	verbose)

    return experiment_dir
