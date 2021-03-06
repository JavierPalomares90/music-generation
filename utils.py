#!/usr/bin/env python
import os, argparse, time,glob,random
from multiprocessing import Pool as ThreadPool
import pandas as pd
from pymidifile import *
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage
from keras.models import model_from_json
from pretty_midi import *

#NUM_NOTES = 128
NUM_NOTES = 129
#NUM_VELOCITIES = 128
NUM_VELOCITIES = 0
DEFAULT_VELOCITY = 64
REST_NOTE = 128

def log(msg,verbose = 1):
    if verbose:
        print('LOG: {}'.format(msg))

def parse_args():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--data_dir', type=str, default='data/midi',
                        help='data directory containing .mid files to use for' \
                             'training')
    parser.add_argument('-e','--experiment_dir', type=str,
                        default='experiments/default',
                        help='directory to store checkpointed models and tensorboard logs.' \
                             'if omitted, will create a new numbered folder in experiments/.')
    parser.add_argument('-r','--rnn_size', type=int, default=64,
                        help='size of RNN hidden state')
    parser.add_argument('-n','--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('-l','--learning_rate', type=float, default=None,
                        help='learning rate. If not specified, the recommended learning '\
                        'rate for the chosen optimizer is used.')
    parser.add_argument('-w','--window_size', type=int, default=20,
                        help='Window size for RNN input per step.')
    parser.add_argument('-b','--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('-N','--num_epochs', type=int, default=10,
                        help='number of epochs before stopping training.')
    parser.add_argument('-D','--dropout', type=float, default=0.2,
                        help='percentage of weights that are turned off every training '\
                        'set step. This is a popular regularization that can help with '\
                        'overfitting. Recommended values are 0.2-0.5')
    parser.add_argument('-o','--optimizer', 
                        choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 
                                 'adam', 'adamax', 'nadam'], default='adam',
                        help='The optimization algorithm to use. '\
                        'See https://keras.io/optimizers for a full list of optimizers.')
    parser.add_argument('-g','--grad_clip', type=float, default=5.0,
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

def parse_generate_args():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e','--experiment_dir', type=str,
                        default='experiments/default',
                        help='directory to load saved model from. ' \
                             'If omitted, it will use the most recent directory from ' \
                             'experiments/.')
    parser.add_argument('-s','--save_dir', type=str,
    					help='directory to save generated files to. Directory will be ' \
    					'created if it doesn\'t already exist. If not specified, ' \
    					'files will be saved to generated/ inside --experiment_dir.')
    parser.add_argument('-n','--num_files', type=int, default=10,
                        help='number of midi files to sample.')
    parser.add_argument('-f','--file_length', type=int, default=1000,
    					help='Length of each file, measured in 16th notes.')
    parser.add_argument('-p','--prime_file', type=str,
                        help='prime generated files from midi file. If not specified ' \
                        'random windows from the validation dataset will be used for ' \
                        'for seeding.')
    parser.add_argument('-d','--data_dir', type=str, default='data/midi',
                        help='data directory containing .mid files to use for' \
                             'seeding/priming. Required if --prime_file is not specified')
    parser.add_argument('-t','--threshold', type=float, default=.7,
                        help='Threshold for turning notes on. Notes with sigmoid value higher than this will be activated')
    return parser.parse_args()


def load_model_from_checkpoint(model_dir):
    
    '''Loads the best performing model from checkpoint_dir'''
    with open(os.path.join(model_dir, 'model.json'), 'r') as f:
        model = model_from_json(f.read())

    epoch = 0
    newest_checkpoint = max(glob.iglob(model_dir + 
    	                    '/checkpoints/*.hdf5'), 
                            key=os.path.getctime)

    if newest_checkpoint: 
       epoch = int(newest_checkpoint[-22:-19])
       model.load_weights(newest_checkpoint)

    return model, epoch

def get_midi_data(midi_paths,max_num_dfs = 100):
    num_files = len(midi_paths)
    num_dfs = num_files
    if(num_files > max_num_dfs):
        num_dfs = max_num_dfs
    dfs = [] * num_dfs
    for i in range(num_dfs):
        midi_file = midi_paths[i]

        df = get_midi_as_pandas(midi_file)
        dfs.append(df)
    return dfs;

def _parse_msg(msg, on_notes = None):
    # keep track of which notes have been on in the past
    if(on_notes == None):
        on_notes = [0] * NUM_NOTES
    #velocities = [0] * NUM_VELOCITIES
    #on_notes.extend(velocities)
    msg_type = msg.type
    if msg_type == 'note_on' or msg_type == 'note_off':
        velocity = msg.velocity
        note = msg.note
        if(msg_type == 'note_on'):
            val = 1
        elif msg_type == 'note_off':
            val = 0
        on_notes[note] = val
        #on_notes[NUM_NOTES + velocity] = 1
    return on_notes


def _get_windows_from_midis(midis,window_size):
    X, y = [],[]
    for midi in midis:
        if midi is not None:
            windows = _get_windows_from_midi(midi,window_size)
            for w in windows:
                X.append(w[0])
                y.append(w[1])
    return (np.asarray(X), np.asarray(y))


def _get_windows_from_midi(midi,window_size):
    pmidi = midi
    pmidi.remove_invalid_notes()
    windows = []
    for instrument in pmidi.instruments:
        if len(instrument.notes) > window_size:
            windows = _encode_sliding_windows(instrument, window_size)
    return windows


# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach uses the piano roll method, where each step in the sliding
# window represents a constant unit of time (fs=4, or 1 sec / 4 = 250ms).
# This allows us to encode rests.
# pm_instrument can be polyphonic.
def _encode_sliding_windows(pm_instrument, window_size):
    roll = np.copy(pm_instrument.get_piano_roll(fs=4).T)

    # trim beginning silence
    summed = np.sum(roll, axis=1)
    mask = (summed > 0).astype(float)
    roll = roll[np.argmax(mask):]

    # transform note velocities into 1s
    roll = (roll > 0).astype(float)

    # calculate the percentage of the events that are rests
    # s = np.sum(roll, axis=1)
    # num_silence = len(np.where(s == 0)[0])
    # print('{}/{} {:.2f} events are rests'.format(num_silence, len(roll), float(num_silence)/float(len(roll))))

    # append a feature: 1 to rests and 0 to notes
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)

    windows = []
    for i in range(0, roll.shape[0] - window_size - 1):
        windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
    return windows


# lazily load the midi data
def get_midi_data_generator(midi_paths, window_size=20, batch_size=32, num_threads=8,max_files_in_ram=170):
    if num_threads > 1:
        pool = ThreadPool(num_threads)

    load_index = 0

    while True:
        load_files = midi_paths[load_index:load_index + max_files_in_ram]
        load_index = (load_index + max_files_in_ram) % len(midi_paths)

        # print('loading large batch: {}'.format(max_files_in_ram))
        # print('Parsing midi files...')
        # start_time = time.time()
        if num_threads > 1:
       		midi_pandas = pool.map(get_midi, load_files)
       	else:
       		midi_pandas = map(get_midi, load_files)
        # print('Finished in {:.2f} seconds'.format(time.time() - start_time))
        # print('parsed, now extracting data')
        data = _get_windows_from_midis(midi_pandas,window_size)
        batch_index = 0
        while batch_index + batch_size < len(data[0]):
            # print('getting data...')
            # print('yielding small batch: {}'.format(batch_size))
            
            res = (data[0][batch_index: batch_index + batch_size], 
                   data[1][batch_index: batch_index + batch_size])
            yield res
            batch_index = batch_index + batch_size
        
        # probably unneeded but why not
        del midi_pandas # free the mem
        del data # free the mem


def get_midi_as_pandas(midi_file):
    midi_pandas = pymidifile.mid_to_matrix(midi_file,output='pandas')
    return midi_pandas

def get_midi(path):
    midi = None
    try:
        midi = pretty_midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
    except Exception as e:
        raise Exception(("%s\nerror readying midi file %s" % (e, path)))
    return midi

def get_midi_paths(dir):
    # Find all the midi files in the directory
    paths = []
    for root,dirs,files in os.walk(dir):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
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

def save_model(model,directory):
    with open(os.path.join(directory, 'model.json'), 'w') as f:
        f.write(model.to_json())

def _get_notes_on(prev_notes,notes):
    if(prev_notes is None):
        new_notes = notes;
    else:
        new_notes = np.subtract(notes,prev_notes)
    return np.where(new_notes == 1)[0]

def _get_notes_off(prev_notes,notes):
    if(prev_notes is None):
        return []
    old_notes = np.subtract(prev_notes,notes)
    return np.where(old_notes == 1)[0]

# Parse the encoded notes to a midos MidiFile
def _get_midi_from_model_output(seed, generated):
    # save all message in one track
    midi = MidiFile(type=0)
    track = MidiTrack()
    midi.tracks.append(track)
    prev_notes = None
    for arr in seed:
        # the notes are encoded as indicators for 
        # 128 notes, followed by indicators for 128 velocities
        notes = arr[0:NUM_NOTES]
        #velocities = arr[NUM_NOTES:]
        notes_on = _get_notes_on(prev_notes,notes)
        notes_off = _get_notes_off(prev_notes,notes)
        for note in notes_on:
            msg = Message('note_on',note=note,velocity = DEFAULT_VELOCITY )
            track.append(msg)
        for note in notes_off:
            msg = Message('note_off',note=note,velocity = DEFAULT_VELOCITY)
            track.append(msg)
        prev_notes = notes

    for arr in generated:
        # the notes are encoded as indicators for 
        # 128 notes, followed by indicators for 128 velocities
        notes = arr[0:NUM_NOTES]
        #velocities = arr[NUM_NOTES:]
        notes_on = _get_notes_on(prev_notes,notes)
        notes_off = _get_notes_off(prev_notes,notes)
        for note in notes_on:
            msg = Message('note_on',note=note,velocity = DEFAULT_VELOCITY)
            track.append(msg)
        for note in notes_off:
            msg = Message('note_off',note=note,velocity = DEFAULT_VELOCITY)
            track.append(msg)
        prev_notes = notes
    return midi;


# Parse the encoded notes to a midos MidiFile
def _get_pretty_midi_from_model_output(generated_notes,
                                       instrument_name='Acoustic Grand Piano',
                                       allow_represses=False):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    # Create an Instrument instance for a piano
    instrument_program = instrument_name_to_program(instrument_name)
    instrument_track = Instrument(program=instrument_program)

    cur_note = None  # an invalid note to start with
    cur_note_start = None
    clock = 0

    for note in generated_notes:
        note_num = np.argmax(note) - 1

        # a note has changed
        if allow_represses or note_num != cur_note:

            # if a note has been played before and it wasn't a rest
            if cur_note is not None and cur_note >= 0 and cur_note is not REST_NOTE:
                # add the last note, now that we have its end time
                vel = 127 #Highest note striking velocity
                if (cur_note is REST_NOTE): # If rest is predicted, set zero pitch & velocity midi note
                    vel = 0
                    cur_note = 0
                note = pretty_midi.Note(velocity=vel,
                                        pitch=int(cur_note),
                                        start=cur_note_start,
                                        end=clock)
                instrument_track.notes.append(note)

            # update the current note
            cur_note = note_num
            cur_note_start = clock

        # update the clock
        clock = clock + 1.0 / 4

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument_track)
    return midi


def _get_notes_from_pred(pred_probs):
    num_notes = len(pred_probs)
    index = np.random.choice(range(0,num_notes), p = pred_probs)
    notes = np.zeros(num_notes)
    notes[index] = 1
    return notes

# generate note encodings from a model using a seed
def _gen(model, seed,window_size,length,threshold):
    generated  = []
    # ring buffer
    buf = np.copy(seed).tolist()
    while len(generated) < length:
        arr = np.expand_dims(np.asarray(buf), 0)
        pred = model.predict(arr)
        pred_probs = pred[0]
        
        # get the notes by from bernoulli samples
        notes = _get_notes_from_pred(pred_probs)
        generated.append(notes)
        buf.pop(0)
        buf.append(notes)
    return generated
    
    

def generate(model, seeds, window_size, length, num_to_gen,threshold):
    midis = []
    for i in range(num_to_gen):
        log("generating midi {} of {}".format(i,num_to_gen))
        # get a random seed
        seed = seeds[random.randint(0,len(seeds) - 1)]
        generated = _gen(model,seed,window_size,length,threshold)
        midi = _get_pretty_midi_from_model_output(generated)
        midis.append(midi)
    return midis
