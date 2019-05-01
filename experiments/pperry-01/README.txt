
==========================================
Music-generation "pperry-01" Experiment
==========================================

------------------------------------------
Platform Info:
------------------------------------------
Paperspace "Core" GPU+ VM created with "ML-in-a-box" public template, running Ubuntu Linux v18.04.


------------------------------------------
Steps to reproduce experiment:
------------------------------------------

1. From top level of music-generation git repo, create  midi data folder ('midi_files/pperry-01') and copy the original (non-reformated) midi files from composers Bach, Beethoven, and Mozart:

  >$ mkdir midi_files/pperry-01
  >$ cp midi_files/originals/bach/*.mid midi_files/pperry-01/.
  >$ cp midi_files/originals/beethoven/*.mid midi_files/pperry-01/.
  >$ cp midi_files/originals/mozart/*.mid midi_files/pperry-01/.


2. Remove dupliate midi files with "format" in the file name, so that this experiment wil train on original midi files that may have multiple tracks (format0 files have one and only one track/channel).

  >$ rm midi_files/pperry-01/*format*.mid


3. Train a new model with mostly default values with the newly created "pperry-01" midi data folder, and set the experiment to the same name:

  >$ python train.py --data_dir midi_files/pperry-01/ --experiment_dir experiments/pperry-01/ --message "pperry first attempt"

  [NOTE: Training took about 15 minutes]


4. After model has been trained (and saved), use the newly trained model to generate new music:

  >$ python generate.py --experiment_dir experiments/pperry-01/ --data_dir midi_files/reformated/ravel/


5. The defauuuuult ten generated midi files should be located in the "experiments/pperry-01/generated_midi" folder.

