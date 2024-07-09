import mne
import pandas as pd
from meegkit.asr import ASR

from power_features import calculate_powers_xdf
import config


def asr_calibration(fname_calibfile: str = config.FNAME_CALIBFILE):
    raw_calib = mne.io.read_raw_eeglab(fname_calibfile, preload=True)
    raw_calib = raw_calib.filter(l_freq=0.5, h_freq=40)

    asr = ASR(method='euclid', sfreq=raw_calib.info['sfreq'], win_len=1)
    asr.fit(raw_calib.get_data())

    return asr


def asr_for_condition(asr, fname: str, condition: str):
    results = calculate_powers_xdf(fname, asr)
    df_results = pd.DataFrame(results,
                              columns=['channel', 'delta', 'delta_rel', 'theta', 'theta_rel', 'alpha', 'alpha_rel',
                                       'beta', 'beta_rel', 'gamma', 'gamma_rel'])
    df_results.insert(0, 'condition', condition)

    return df_results


def asr_for_both_conditions(asr, fname_rest: str, fname_task: str):

    df_task = asr_for_condition(asr, fname=fname_task, condition='task')
    df_rest = asr_for_condition(asr, fname=fname_rest, condition='rest')
    df_all = pd.concat([df_rest, df_task])

    return df_all
