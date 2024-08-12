# MindEasy-EEG
Scripts and functions for EEG analysis in MindEasy app.

For more information check: 

https://github.com/DawidPietrykowski/MindEasy/tree/main

## Description

`classificator.py` calculates model from the recorded EEG data used to predict the Mind Wandering and Focus scores.

`stream.py` connects to LSL stream of EEG data and predicts the scores in realtime.

`config.py` : Contains configuration arguments used by `classificator.py` and `stream.py`:


#### EEG files
* *FNAME_CALIBFILE*: name of the EEG data file used to calibrate ASR for detection of artifacts
* *FNAME_REST*: name of the EEG data file used as mind wandering condition
* *FNAME_TASK*: name of the EEG data file used as a lesson condtition

#### classification mode
* *SAVE_MODEL*: logical, whether to save the model
* *SAVED_MODELS_PATH*: path for saving the model
* *CLASSIFICATION_METHOD*: name of the method used to calculate features for classification, "features" for bandpowers and complexity metrics and "csp" for common spatial patterns 
* *PLOT_FEATURES*: logical, whether to plot the features

#### loading saved model 
* *MODEL_FNAME*: name of the model

#### streaming
* *PRERECORDED*: logical, whether to simulate the stream from the prerecored EEG data collected during the lesson
* *FIF_FNAME*: name of the EEG data file recorded during the lesson
* *SUBJECT*: name of the subject
* *CONDITION*: name of the condition (lesson)
* *STREAM_TIME*: length of interval (in seconds) of the streamed data for which the analysis is performed 
* *INTERVAL*: length of interval (in seconds) for which average Mind Wandering and Focus scores are calculated and saved as .json file


