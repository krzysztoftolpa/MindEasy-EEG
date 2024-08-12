# EEG files
FNAME_CALIBFILE = 'data/sample_calibfile.set'
FNAME_REST = 'data/sample-eo-rest.xdf'
FNAME_TASK = 'data/sample-task.xdf'

# classification mode
SAVE_MODEL = True
SAVED_MODELS_PATH = 'models'
CLASSIFICATION_METHOD = 'features'
PLOT_FEATURES = False

# loading saved model 
MODEL_FNAME = 'Logistic_Regression_asr_2024-08-10_21-52-38.pkl' # logistic_regression_test_model_asr.pkl

# streaming
FIF_FNAME = 'data/sub-sample_task_ses-S001_task-Default_run-001_eeg.fif'
SUBJECT = 'sub-sample'
CONDITION = ''
STREAM_TIME = 61
INTERVAL = 10
PRERECORDED = True #False