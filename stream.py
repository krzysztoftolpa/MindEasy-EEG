import pickle
import os
import time
import numpy as np
import pandas as pd
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.lsl import resolve_streams

import config
from power_features import calculate_powers_epoch, calculate_complexity_epoch
from utils import generate_eeg_metrics_json

# reading model and ASR
with open(os.path.join(config.SAVED_MODELS_PATH, config.MODEL_FNAME), 'rb') as handle:
    (best_models, asr) = pickle.load(handle)


clf = best_models[list(best_models.keys())[0]]['model']

if config.PRERECORDED:
    player = Player(config.FIF_FNAME).start()
    print(player.info)
    streams = resolve_streams()

stream = Stream(bufsize=1).connect()
print(stream.info)
sfreq = stream.info['sfreq']
channels = stream.info['ch_names']
if len(channels):
    channels = channels[:-1]
# stream.set_eeg_reference("average") # setting reference to average of all channels found in buffer
stream.filter(0.5, 40.0)  # filter 1.5, 40 Hz


scores_list = []
start_time = time.time()
loop_start_time = time.time()

##  MAIN LOOP   ##
while time.time() - start_time <= config.STREAM_TIME:
   
    # calculate power and in few seconds forwards them
    
    data, ts = stream.get_data(winsize=None, picks=['TP9', 'AF7', 'AF8', 'TP10'])
    data = data*1e-6 
    
    # remove mean
    # data = data - np.mean(data, axis=1, keepdims=True)
  
    if config.CLASSIFICATION_METHOD=='features':
        results_epoch = calculate_powers_epoch(data, sfreq, channels, asr)
        complexity_epoch = calculate_complexity_epoch(data, channels, asr)
        df_powers = pd.DataFrame(results_epoch, columns=['channel', 'delta', 'delta_rel', 'theta', 'theta_rel', 'alpha',
                                                        'alpha_rel', 'beta', 'beta_rel', 'gamma', 'gamma_rel'])
        
        df_complexity = pd.DataFrame(complexity_epoch, columns=['channel','sampen', 'appen', 'higuchi', 'katz'])

        df_epoch = df_powers.copy()
        # df_epoch = df_powers.merge(df_complexity, on='channel')

        X_epoch = df_epoch.drop(['channel'], axis=1)

        scores = clf.predict_proba(X_epoch)
    
    
    if config.CLASSIFICATION_METHOD=='csp':
        data = data.reshape(1, data.shape[0], data.shape[1])
        features = clf['CSP'].transform(data)
        scores = clf['classifier'].predict_proba(features)

    scores = np.mean(scores, axis=0)
    scores_list.append(scores)
    print(scores)
    # Collect scores every 10 seconds
    if (time.time() - loop_start_time) >= config.INTERVAL:

        loop_start_time = time.time()
        scores_arr = np.array(scores_list)
        avg_scores = np.round(np.mean(scores_arr, axis=0), 2)
        # print(avg_scores)
        print('Mind wandering: ', avg_scores[0])
        print('Focus: ', avg_scores[1])

        generate_eeg_metrics_json('Focus_MW_scores.json', avg_scores)
        scores_list = []
