import json
import mne
import pyxdf
import numpy as np

def balance_conditions(df, condition_col):
    """
    Subsets the DataFrame so that each category in the specified condition column has an equal number of rows.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    condition_col (str): The name of the column containing the condition categories.

    Returns:
    pd.DataFrame: The balanced DataFrame.
    """
    # Get the minimum count of rows across all unique values in the condition column
    min_count = df[condition_col].value_counts().min()

    # Sample the minimum count of rows for each unique value in the condition column
    balanced_df = df.groupby(condition_col).apply(lambda x: x.sample(min_count)).reset_index(drop=True)

    return balanced_df


def generate_eeg_metrics_json(file_path, values):
    # Check if the values array has exactly two elements
    if len(values) != 2:
        raise ValueError("The input array must contain exactly two values.")

    # Data to be written to JSON
    data = {
        "eeg_metrics": {
            "engagement": {
                "focus": values[1],
                "mind_wandering": values[0]
            }
        }
    }

    # Writing JSON data to a file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file created at: {file_path}")



def extract_epochs_array(fname, asr):

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

        results.append(epoch)

    return np.array(results)


def get_muse_info(fname):
    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    # load data
    streams, header = pyxdf.load_xdf(fname)
    data = streams[0]["time_series"].T
    data = data[:-1,:]  # last channel is AUX
    data = data*1e-6  # convert to V
    sfreq = float(streams[0]["info"]["nominal_srate"][0])
    info = mne.create_info(channels, sfreq, 'eeg')
    info.set_montage('standard_1020')

    return info