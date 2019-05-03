
==========================================
Music-generation "orig_non-format0--dataset" Model Training Experiment
==========================================

------------------------------------------
Platform Info:
------------------------------------------
Paperspace "Core" GPU+ VM created with "ML-in-a-box" public template, running Ubuntu Linux v18.04.


------------------------------------------
Steps to reproduce experiment:
------------------------------------------

1. From top level of music-generation git repo, create midi data folder ('midi_files/orig_non-format0') and copy the entire original (non-reformated) midi files directory:

  >$ mkdir midi_files/orig_non-format0
  >$ cp -rf midi_files/originals/* midi_files/orig_non-format0/.


2. Remove dupliate midi files with "format" in the file name, so that this experiment wil train on original midi files that may have multiple tracks (format0 files have one and only one track/channel).

  >$ rm midi_files/orig_non-format0--dataset/*/format*.mid


3. Train a new model with mostly default values with the newly created "pperry-01" midi data folder, and set the experiment to the same name:

  >$ python train.py --experiment_dir experiments/original-non-format0--dataset --data_dir midi_files/orig_non-format0/ --num_layers 3 --num_epochs 9 --num_windows 20 --n_jobs 8 --message 'Trained with 3 layers, 9 epochs, 8 jobs'

  [NOTE: Training took about 3 hours]

The training produces the following output:


Using TensorFlow backend.
LOG: looking for midi files from midi_files/orig_non-format0/
LOG: Created experiment directory experiments/original-non-format0--dataset
LOG: Loading midi files from midi_files/orig_non-format0/
LOG: getting data generators
LOG: Getting model
WARNING:tensorflow:From /home/paperspace/.local/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3368: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
LOG: Loaded model on epoch=0
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_1 (LSTM)                (None, 20, 64)            49664     
_________________________________________________________________
lstm_2 (LSTM)                (None, 20, 64)            33024     
_________________________________________________________________
lstm_3 (LSTM)                (None, 64)                33024     
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 129)               8385      
_________________________________________________________________
activation_1 (Activation)    (None, 129)               0         
=================================================================
Total params: 124,097
Trainable params: 124,097
Non-trainable params: 0
_________________________________________________________________
None
LOG: Saved model to experiments/original-non-format0--dataset/model.json
/usr/local/lib/python3.6/dist-packages/keras/callbacks.py:999: UserWarning: `epsilon` argument is deprecated and will be removed, use `min_delta` instead.
  warnings.warn('`epsilon` argument is deprecated and '
LOG: fitting model
WARNING:tensorflow:From /home/paperspace/.local/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
2019-05-03 11:10:00.699079: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-05-03 11:10:00.801370: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-05-03 11:10:00.801817: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x1852830 executing computations on platform CUDA. Devices:
2019-05-03 11:10:00.801844: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): Quadro M4000, Compute Capability 5.2
2019-05-03 11:10:00.804450: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2599995000 Hz
2019-05-03 11:10:00.805229: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x3838400 executing computations on platform Host. Devices:
2019-05-03 11:10:00.805271: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-05-03 11:10:00.805696: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: Quadro M4000 major: 5 minor: 2 memoryClockRate(GHz): 0.7725
pciBusID: 0000:00:05.0
totalMemory: 7.94GiB freeMemory: 7.58GiB
2019-05-03 11:10:00.805731: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-05-03 11:10:00.806634: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-05-03 11:10:00.806663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-05-03 11:10:00.806688: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-05-03 11:10:00.806932: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7374 MB memory) -> physical GPU (device: 0, name: Quadro M4000, pci bus id: 0000:00:05.0, compute capability: 5.2)
Epoch 1/9
2019-05-03 11:11:05.616390: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
8658/8657 [==============================] - 1261s 146ms/step - loss: 9.8348 - acc: 0.6327 - val_loss: 7.8224 - val_acc: 0.7356

Epoch 00001: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_001-val_acc_0.736.hdf5
Epoch 2/9
8658/8657 [==============================] - 1209s 140ms/step - loss: 7.8818 - acc: 0.6352 - val_loss: 6.9303 - val_acc: 0.7219

Epoch 00002: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_002-val_acc_0.722.hdf5
Epoch 3/9
8658/8657 [==============================] - 1177s 136ms/step - loss: 7.9771 - acc: 0.5988 - val_loss: 5.8133 - val_acc: 0.7322

Epoch 00003: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_003-val_acc_0.732.hdf5
Epoch 4/9
8658/8657 [==============================] - 1175s 136ms/step - loss: 8.8320 - acc: 0.5896 - val_loss: 6.0336 - val_acc: 0.7449

Epoch 00004: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_004-val_acc_0.745.hdf5
Epoch 5/9
8658/8657 [==============================] - 1213s 140ms/step - loss: 8.3174 - acc: 0.5989 - val_loss: 6.3749 - val_acc: 0.7302

Epoch 00005: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_005-val_acc_0.730.hdf5
Epoch 6/9
8658/8657 [==============================] - 1162s 134ms/step - loss: 6.8464 - acc: 0.5948 - val_loss: 6.0122 - val_acc: 0.7497

Epoch 00006: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_006-val_acc_0.750.hdf5

Epoch 00006: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
Epoch 7/9
8658/8657 [==============================] - 1201s 139ms/step - loss: 8.5010 - acc: 0.5860 - val_loss: 6.3343 - val_acc: 0.7675

Epoch 00007: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_007-val_acc_0.767.hdf5
Epoch 8/9
8658/8657 [==============================] - 1175s 136ms/step - loss: 7.6954 - acc: 0.5922 - val_loss: 5.6445 - val_acc: 0.7303

Epoch 00008: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_008-val_acc_0.730.hdf5
Epoch 9/9
8658/8657 [==============================] - 1188s 137ms/step - loss: 7.1476 - acc: 0.5865 - val_loss: 6.5812 - val_acc: 0.7194

Epoch 00009: saving model to experiments/original-non-format0--dataset/checkpoints/checkpoint-epoch_009-val_acc_0.719.hdf5
LOG: Finished in 10767.10 seconds

