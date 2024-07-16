import numpy as np
import scipy
import pyxdf
import mne
from mne_features.univariate import (compute_samp_entropy, compute_app_entropy, compute_higuchi_fd, compute_katz_fd)

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    # print(freqs)
    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def calculate_powers_xdf(fname, asr):

    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    # load data
    streams, header = pyxdf.load_xdf(fname)
    data = streams[0]["time_series"].T
    data = data[:-1,:]  # last channel is AUX
    data = data*1e-6  # convert to V
    sfreq = float(streams[0]["info"]["nominal_srate"][0])
    info = mne.create_info(channels, sfreq, 'eeg')
    raw = mne.io.RawArray(data, info)
    raw.set_montage('standard_1020')

    # filter data
    raw = raw.filter(l_freq=0.5, h_freq = 40)

    # make fixed length epochs
    epochs = mne.make_fixed_length_epochs(raw, duration = 1.0, overlap = 0)
    # epochs.plot()

    results = []
    for epoch in epochs:

        # plt.plot(epoch[0,:])
        epoch = asr.transform(epoch)
        # plt.plot(epoch[0,:])
        # plt.show()
        for idx, chan in enumerate(epoch):

            delta = bandpower(chan, sfreq, [0.5, 4], 1, relative=False)
            delta_rel = bandpower(chan, sfreq, [0.5, 4], 1, relative=True)

            theta = bandpower(chan, sfreq, [4, 8], 1, relative=False)
            theta_rel = bandpower(chan, sfreq, [4, 8], 1, relative=True)

            alpha = bandpower(chan, sfreq, [8, 12], 1, relative=False)
            alpha_rel = bandpower(chan, sfreq, [8, 12], 1, relative=True)

            beta = bandpower(chan, sfreq, [15, 30], 1, relative=False)
            beta_rel = bandpower(chan, sfreq, [15, 30], 1, relative=True)

            gamma = bandpower(chan, sfreq, [30, 40], 1, relative=False)
            gamma_rel = bandpower(chan, sfreq, [30, 40], 1, relative=True)

            results.append([channels[idx], delta, delta_rel, theta, theta_rel, alpha, alpha_rel, beta, beta_rel, gamma, gamma_rel])

    return results


def calculate_powers_epoch(epoch, sfreq, channels, asr):

    results = []
    epoch = asr.transform(epoch)
    # plt.plot(epoch[0,:])
    # plt.show()
    for idx, chan in enumerate(epoch):

        delta = bandpower(chan, sfreq, [0.5, 4], 1, relative=False)
        delta_rel = bandpower(chan, sfreq, [0.5, 4], 1, relative=True)

        theta = bandpower(chan, sfreq, [4, 8], 1, relative=False)
        theta_rel = bandpower(chan, sfreq, [4, 8], 1, relative=True)

        alpha = bandpower(chan, sfreq, [8, 12], 1, relative=False)
        alpha_rel = bandpower(chan, sfreq, [8, 12], 1, relative=True)

        beta = bandpower(chan, sfreq, [15, 30], 1, relative=False)
        beta_rel = bandpower(chan, sfreq, [15, 30], 1, relative=True)

        gamma = bandpower(chan, sfreq, [30, 40], 1, relative=False)
        gamma_rel = bandpower(chan, sfreq, [30, 40], 1, relative=True)

        results.append([channels[idx], delta, delta_rel, theta, theta_rel, alpha, alpha_rel, beta, beta_rel, gamma, gamma_rel])

    return results



def calculate_complexity_epoch(epoch, channels, asr):

   
    epoch = asr.transform(epoch)


    sampen = compute_samp_entropy(epoch, emb=2)
    appen = compute_app_entropy(epoch, emb = 2)
    higuchi = compute_higuchi_fd(epoch)
    katz = compute_katz_fd(epoch)
    
    metrics = np.vstack((sampen, appen, higuchi, katz)).T
    results = np.hstack((np.array(channels).reshape(-1, 1), metrics))
    return results





def calculate_complexity_xdf(fname, asr):

    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    # load data
    streams, header = pyxdf.load_xdf(fname)
    data = streams[0]["time_series"].T
    data = data[:-1,:]  # last channel is AUX
    data = data*1e-6  # convert to V
    sfreq = float(streams[0]["info"]["nominal_srate"][0])
    info = mne.create_info(channels, sfreq, 'eeg')
    raw = mne.io.RawArray(data, info)
    raw.set_montage('standard_1020')

    # filter data
    raw = raw.filter(l_freq=0.5, h_freq = 40)

    # make fixed length epochs
    epochs = mne.make_fixed_length_epochs(raw, duration = 1.0, overlap = 0)
    # epochs.plot()

    results = []
    for epoch in epochs:

        # plt.plot(epoch[0,:])
        epoch = asr.transform(epoch)
        # plt.plot(epoch[0,:])
        # plt.show()
        sampen = compute_samp_entropy(epoch, emb=2)
        appen = compute_app_entropy(epoch, emb = 2)
        higuchi = compute_higuchi_fd(epoch)
        katz = compute_katz_fd(epoch)
        
        metrics = np.vstack((sampen, appen, higuchi, katz)).T
        tmp_results = np.hstack((np.array(channels).reshape(-1, 1), metrics))


        results.append(tmp_results)

    results = np.vstack(results)

    return results