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

OUTPUT_SIZE = 129 # 0-127 notes + 1 for rests
#create or load a saved model
# returns the model and the epoch number (>1 if loaded from a checkpoint)
def get_model(args,experiment_dir=None):
    epoch = 0

    if not experiment_dir:
        model = Sequential()
        num_layers = args.num_layers
        for layer in range(num_layers):
            kwargs = dict()
            kwargs['units'] = args.rnn_size
            
            # first layer
            if layer == 0:
                kwargs['input_shape'] = (args.window_size,OUTPUT_SIZE)
                if num_layers == 1:
                    kwargs['return_sequences'] = False
                else:
                    kwargs['return_sequences'] = True
                model.add(LSTM(**kwargs))
            else:
                # if this is a middle layer
                if not layer_index == args.num_layers - 1:
                    kwargs['return_sequences'] = True
                    model.add(LSTM(**kwargs))
                else: # this is the last layer
                    kwargs['return_sequences'] = False
                    model.add(LSTM(**kwargs))
        model.add(Dropout(args.dropout))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('softmax'))
    else:
        model, epoch = utils.load_model_from_checkpoint(experiment_dir)
    # these cli args aren't specified if get_model() is being
    # being called from sample.py
    if 'grad_clip' in args and 'optimizer' in args:
        kwargs = { 'clipvalue': args.grad_clip }

        if args.learning_rate:
            kwargs['lr'] = args.learning_rate

        # select the optimizers
        if args.optimizer == 'sgd':
            optimizer = SGD(**kwargs)
        elif args.optimizer == 'rmsprop':
            optimizer = RMSprop(**kwargs)
        elif args.optimizer == 'adagrad':
            optimizer = Adagrad(**kwargs)
        elif args.optimizer == 'adadelta':
            optimizer = Adadelta(**kwargs)
        elif args.optimizer == 'adam':
            optimizer = Adam(**kwargs)
        elif args.optimizer == 'adamax':
            optimizer = Adamax(**kwargs)
        elif args.optimizer == 'nadam':
            optimizer = Nadam(**kwargs)
        else:
            utils.log(
                'Error: {} is not a supported optimizer. Exiting.'.format(args.optimizer),
                True)
            exit(1)
    else: # so instead lets use a default (no training occurs anyway)
        optimizer = Adam()

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model, epoch

def get_tf_callbacks(experiment_dir, checkpoint_monitor='val_acc'):
    
    callbacks = []
    
    # save model checkpoints
    filepath = os.path.join(experiment_dir, 
                            'checkpoints', 
                            'checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

    callbacks.append(ModelCheckpoint(filepath, 
                                     monitor=checkpoint_monitor, 
                                     verbose=1, 
                                     save_best_only=False, 
                                     mode='max'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=3, 
                                       verbose=1, 
                                       mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=0, 
                                       min_lr=0))

    callbacks.append(TensorBoard(log_dir=os.path.join(experiment_dir, 'tensorboard-logs'), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False))

    return callbacks

def main():
    args = utils.parse_args()
    data_dir = args.data_dir
    utils.log("looking for midi files from {}".format(data_dir))
    midi_files = utils.get_midi_paths(data_dir)
    experiment_dir = utils.create_experiment_dir(args.experiment_dir)
    utils.log("Created experiment directory {}".format(experiment_dir))

    val_split_index = len(midi_files) - int(len(midi_files) * VALIDATION_SPLIT_RATE)
    utils.log("Loading midi files from {}".format(data_dir))
    train_data = utils.get_midi_data(midi_files[0:val_split_index])
    val_data = utils.get_midi_data(midi_files[val_split_index:])
    utils.log("Split training and validation data")

    model,epoch = get_model(args)
    utils.log("Loaded model on epoch={}".format(epoch))
    if args.verbose:
        print(model.summary())

    utils.save_model(model, experiment_dir)
    utils.log('Saved model to {}'.format(os.path.join(experiment_dir, 'model.json')))

    callbacks = get_tf_callbacks(experiment_dir)
    utils.log("fitting model")
    num_windows = 827
    #average number of length-20 windows
    start_time = time.time()
    model.fit_generator(train_data,
                        steps_per_epoch=len(midi_files) * num_windows / args.batch_size, 
                        epochs=args.num_epochs,
                        validation_data=val_data,
                        validation_steps=len(midi_files) * VALIDATION_SPLIT_RATE * num_windows/ args.batch_size,
                        verbose=1, 
                        callbacks=callbacks,
                        initial_epoch=epoch)
    utils.log('Finished in {:.2f} seconds'.format(time.time() - start_time), args.verbose)



if __name__ == '__main__':
    main()