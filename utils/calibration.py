import mne
import pandas as pd
from meegkit.asr import ASR

from utils.power_features import calculate_powers_xdf, calculate_complexity_xdf
from utils.helpers import extract_epochs_array
import config
import numpy as np

def asr_calibration(fname_calibfile: str = config.FNAME_CALIBFILE):
    raw_calib = mne.io.read_raw_eeglab(fname_calibfile, preload=True)
    raw_calib = raw_calib.filter(l_freq=0.5, h_freq=40)

    asr = ASR(method='euclid', sfreq=raw_calib.info['sfreq'], win_len=1)
    asr.fit(raw_calib.get_data())

    return asr


def asr_for_condition(asr, fname: str, condition: str):
    results = calculate_powers_xdf(fname, asr)
    df_powers = pd.DataFrame(results,
                              columns=['channel', 'delta', 'delta_rel', 'theta', 'theta_rel', 'alpha', 'alpha_rel',
                                       'beta', 'beta_rel', 'gamma', 'gamma_rel'])
    
    complexity_results = calculate_complexity_xdf(fname, asr)
    
    df_complexity = pd.DataFrame(complexity_results, columns=['channel', 'sampen', 'appen', 'higuchi', 'katz'])
    
    df_results = pd.concat([df_powers, df_complexity], axis=1)
    df_results.insert(0, 'condition', condition)

   
    return df_results


def asr_for_both_conditions(asr, fname_rest: str, fname_task: str):

    df_task = asr_for_condition(asr, fname=fname_task, condition='task')
    df_rest = asr_for_condition(asr, fname=fname_rest, condition='rest')
    df_all = pd.concat([df_rest, df_task])

    return df_all


def asr_epoched_for_condition(asr, fname: str, condition: str):
    
    epochs = extract_epochs_array(fname, asr)
    labels = [condition] * len(epochs)

    return epochs, labels

def asr_epoched_for_both_conditions(asr, fname_rest: str, fname_task: str):
    
    epoched_task, labels_task = asr_epoched_for_condition(asr, fname=fname_task, condition='task')
    epoched_rest, labels_rest = asr_epoched_for_condition(asr, fname=fname_rest, condition='rest')

    # balancing number of epochs, could be done other way?
    epoched_task = epoched_task[10:len(epoched_rest)+10,:,:]
    labels_task = labels_task[10:len(epoched_rest)+10]

    epoched_all = np.concatenate([epoched_rest, epoched_task])
    labels_all = labels_rest + labels_task
    labels_all = np.array(labels_all)

    return epoched_all, labels_all