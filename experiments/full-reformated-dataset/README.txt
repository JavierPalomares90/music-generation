
==========================================
Music-generation "full-reformated-dataset" Experiment
==========================================

------------------------------------------
Platform Info:
------------------------------------------
Paperspace "Core" GPU+ VM created with "ML-in-a-box" public template, running Ubuntu Linux v18.04.


------------------------------------------
Steps to reproduce experiment:
------------------------------------------

Train a new model with mostly default values with the newly created "pperry-01" midi data folder, and set the experiment to the same name:

  >$ python train.py --experiment_dir experiments/full-reformated-dataset --data_dir midi_files/reformated/ --num_layers 3 --num_epochs 30 --window_size 20 --n_jobs 8 --message 'Trained with 3 layers, 45 epochs, 8 jobs'


  [NOTE: Training tookjsut over 10 hours]

The following was output to the console:


LOG: looking for midi files from midi_files/reformated/
LOG: Created experiment directory experiments/full-reformated-dataset
LOG: Loading midi files from midi_files/reformated/
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
LOG: Saved model to experiments/full-reformated-dataset/model.json
/usr/local/lib/python3.6/dist-packages/keras/callbacks.py:999: UserWarning: `epsilon` argument is deprecated and will be removed, use `min_delta` instead.
  warnings.warn('`epsilon` argument is deprecated and '
LOG: fitting model
WARNING:tensorflow:From /home/paperspace/.local/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
2019-05-02 21:07:02.922552: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-05-02 21:07:03.033874: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-05-02 21:07:03.034425: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x2d0d890 executing computations on platform CUDA. Devices:
2019-05-02 21:07:03.034510: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): Quadro M4000, Compute Capability 5.2
2019-05-02 21:07:03.037613: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2599995000 Hz
2019-05-02 21:07:03.038697: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x4cf26c0 executing computations on platform Host. Devices:
2019-05-02 21:07:03.038787: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-05-02 21:07:03.039158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: Quadro M4000 major: 5 minor: 2 memoryClockRate(GHz): 0.7725
pciBusID: 0000:00:05.0
totalMemory: 7.94GiB freeMemory: 7.58GiB
2019-05-02 21:07:03.039349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-05-02 21:07:03.040366: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-05-02 21:07:03.040451: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-05-02 21:07:03.040511: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-05-02 21:07:03.040792: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7374 MB memory) -> physical GPU (device: 0, name: Quadro M4000, pci bus id: 0000:00:05.0, compute capability: 5.2)
Epoch 1/30
2019-05-02 21:07:53.387185: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
8658/8657 [==============================] - 1234s 143ms/step - loss: 15.3313 - acc: 0.8849 - val_loss: 15.0986 - val_acc: 0.8549

Epoch 00001: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_001-val_acc_0.855.hdf5
Epoch 2/30
8658/8657 [==============================] - 1228s 142ms/step - loss: 14.1696 - acc: 0.8770 - val_loss: 15.4994 - val_acc: 0.8659

Epoch 00002: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_002-val_acc_0.866.hdf5
Epoch 3/30
8658/8657 [==============================] - 1224s 141ms/step - loss: 14.2101 - acc: 0.8739 - val_loss: 14.4511 - val_acc: 0.8530

Epoch 00003: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_003-val_acc_0.853.hdf5
Epoch 4/30
8658/8657 [==============================] - 1218s 141ms/step - loss: 13.9040 - acc: 0.8599 - val_loss: 14.6405 - val_acc: 0.8390

Epoch 00004: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_004-val_acc_0.839.hdf5
Epoch 5/30
8658/8657 [==============================] - 1214s 140ms/step - loss: 13.4362 - acc: 0.8471 - val_loss: 13.8271 - val_acc: 0.8304

Epoch 00005: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_005-val_acc_0.830.hdf5
Epoch 6/30
8658/8657 [==============================] - 1211s 140ms/step - loss: 13.9074 - acc: 0.8439 - val_loss: 15.1694 - val_acc: 0.8139

Epoch 00006: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_006-val_acc_0.814.hdf5
Epoch 7/30
8658/8657 [==============================] - 1213s 140ms/step - loss: 13.4332 - acc: 0.8273 - val_loss: 15.0629 - val_acc: 0.8261

Epoch 00007: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_007-val_acc_0.826.hdf5
Epoch 8/30
8658/8657 [==============================] - 1208s 139ms/step - loss: 13.3554 - acc: 0.8296 - val_loss: 15.9213 - val_acc: 0.8380

Epoch 00008: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_008-val_acc_0.838.hdf5

Epoch 00008: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
Epoch 9/30
8658/8657 [==============================] - 1210s 140ms/step - loss: 12.9134 - acc: 0.8216 - val_loss: 15.5587 - val_acc: 0.8406

Epoch 00009: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_009-val_acc_0.841.hdf5
Epoch 10/30
8658/8657 [==============================] - 1209s 140ms/step - loss: 13.6047 - acc: 0.8089 - val_loss: 13.9545 - val_acc: 0.8230

Epoch 00010: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_010-val_acc_0.823.hdf5
Epoch 11/30
8658/8657 [==============================] - 1211s 140ms/step - loss: 13.0060 - acc: 0.8090 - val_loss: 13.8810 - val_acc: 0.8339

Epoch 00011: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_011-val_acc_0.834.hdf5

Epoch 00011: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
Epoch 12/30
8658/8657 [==============================] - 1192s 138ms/step - loss: 13.0701 - acc: 0.8074 - val_loss: 15.9981 - val_acc: 0.8549

Epoch 00012: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_012-val_acc_0.855.hdf5
Epoch 13/30
8658/8657 [==============================] - 1192s 138ms/step - loss: 13.0392 - acc: 0.8003 - val_loss: 16.3953 - val_acc: 0.8206

Epoch 00013: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_013-val_acc_0.821.hdf5
Epoch 14/30
8658/8657 [==============================] - 1215s 140ms/step - loss: 12.9952 - acc: 0.8031 - val_loss: 16.4000 - val_acc: 0.8222

Epoch 00014: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_014-val_acc_0.822.hdf5

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
Epoch 15/30
8658/8657 [==============================] - 1190s 137ms/step - loss: 12.6948 - acc: 0.8007 - val_loss: 14.3544 - val_acc: 0.8079

Epoch 00015: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_015-val_acc_0.808.hdf5
Epoch 16/30
8658/8657 [==============================] - 1214s 140ms/step - loss: 13.3129 - acc: 0.7942 - val_loss: 13.8340 - val_acc: 0.8273

Epoch 00016: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_016-val_acc_0.827.hdf5
Epoch 17/30
8658/8657 [==============================] - 1196s 138ms/step - loss: 12.6538 - acc: 0.8049 - val_loss: 16.2143 - val_acc: 0.8255

Epoch 00017: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_017-val_acc_0.825.hdf5

Epoch 00017: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
Epoch 18/30
8658/8657 [==============================] - 1220s 141ms/step - loss: 12.9945 - acc: 0.7919 - val_loss: 16.6087 - val_acc: 0.8316

Epoch 00018: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_018-val_acc_0.832.hdf5
Epoch 19/30
8658/8657 [==============================] - 1245s 144ms/step - loss: 12.8967 - acc: 0.7969 - val_loss: 15.8870 - val_acc: 0.8295

Epoch 00019: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_019-val_acc_0.829.hdf5
Epoch 20/30
8658/8657 [==============================] - 1196s 138ms/step - loss: 12.9804 - acc: 0.7947 - val_loss: 14.3974 - val_acc: 0.8165

Epoch 00020: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_020-val_acc_0.817.hdf5

Epoch 00020: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
Epoch 21/30
8658/8657 [==============================] - 1211s 140ms/step - loss: 12.8010 - acc: 0.7991 - val_loss: 14.9700 - val_acc: 0.8302

Epoch 00021: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_021-val_acc_0.830.hdf5
Epoch 22/30
8658/8657 [==============================] - 1207s 139ms/step - loss: 13.1830 - acc: 0.7905 - val_loss: 15.3231 - val_acc: 0.8411

Epoch 00022: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_022-val_acc_0.841.hdf5
Epoch 23/30
8658/8657 [==============================] - 1199s 139ms/step - loss: 12.7146 - acc: 0.7983 - val_loss: 16.0185 - val_acc: 0.8300

Epoch 00023: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_023-val_acc_0.830.hdf5

Epoch 00023: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
Epoch 24/30
8658/8657 [==============================] - 1185s 137ms/step - loss: 12.8794 - acc: 0.7950 - val_loss: 16.2326 - val_acc: 0.8337

Epoch 00024: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_024-val_acc_0.834.hdf5
Epoch 25/30
8658/8657 [==============================] - 1201s 139ms/step - loss: 12.7497 - acc: 0.7992 - val_loss: 14.8105 - val_acc: 0.8234

Epoch 00025: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_025-val_acc_0.823.hdf5
Epoch 26/30
8658/8657 [==============================] - 1209s 140ms/step - loss: 12.7777 - acc: 0.7957 - val_loss: 15.3634 - val_acc: 0.8291

Epoch 00026: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_026-val_acc_0.829.hdf5

Epoch 00026: ReduceLROnPlateau reducing learning rate to 7.812500371073838e-06.
Epoch 27/30
8658/8657 [==============================] - 1210s 140ms/step - loss: 12.9854 - acc: 0.8018 - val_loss: 15.3420 - val_acc: 0.8298

Epoch 00027: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_027-val_acc_0.830.hdf5
Epoch 28/30
8658/8657 [==============================] - 1203s 139ms/step - loss: 12.8545 - acc: 0.7994 - val_loss: 16.2165 - val_acc: 0.8460

Epoch 00028: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_028-val_acc_0.846.hdf5
Epoch 29/30
8658/8657 [==============================] - 1204s 139ms/step - loss: 12.7733 - acc: 0.7966 - val_loss: 14.5772 - val_acc: 0.8186

Epoch 00029: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_029-val_acc_0.819.hdf5

Epoch 00029: ReduceLROnPlateau reducing learning rate to 3.906250185536919e-06.
Epoch 30/30
8658/8657 [==============================] - 1200s 139ms/step - loss: 12.9162 - acc: 0.7992 - val_loss: 15.9850 - val_acc: 0.8320

Epoch 00030: saving model to experiments/full-reformated-dataset/checkpoints/checkpoint-epoch_030-val_acc_0.832.hdf5
LOG: Finished in 36274.50 seconds



