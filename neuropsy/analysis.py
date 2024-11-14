import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import math
import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test
import pywt


# [REMOVE] drop this function???
def get_trial_markers(df, col) -> list:
    markers = []
    pass


# [REMOVE] now in data.py module -> Class(DataHandler)->method(create_epochs)
def create_epochs(raw, tmin: int | float, tmax: int | float, markers: list | tuple | np.ndarray, event_id: int = 10, event_desc: str = 'event'):
    """create_epochs Create epochs from raw data.

    Args:
        raw (mne.Raw): mne Raw object from which epochs will be created.
        tmin (int | float): Time before onset of event to include in epoch.
        tmax (int | float): Time after onset of event to include in epoch.
        markers (list | tuple | np.ndarray): Event markers to identify epochs.
        event_id (int, optional): Event id to assign to epochs. Defaults to 10.
        event_desc (str, optional): Event description to assign to epochs. Defaults to 'event'.

    Returns:
        epochs (mne.Epochs): mne epochs object.
    """
    # create 2D array with zeros of same length as channel data array,
    # this will be the stimulus channel
    stim_arr = np.zeros((1, len(raw)))

    # set indices to 10 where the markers are
    stim_arr[0, markers] = event_id

    # create info object for stimulus channel
    info = mne.create_info(
        ch_names=['STI_TMP'],
        sfreq=raw.info['sfreq'],
        ch_types=['stim'])

    # create raw array with trigger channel
    stim_raw = mne.io.RawArray(stim_arr, info)

    # add stimulus channel to raw object
    raw.add_channels([stim_raw], force_update_info=True)

    # find events from stimulus channel
    events = mne.find_events(raw, stim_channel="STI_TMP")
    raw.drop_channels(['STI_TMP'])

    # create mapping between event_id to event description
    event_ids = {f'{event_desc}': event_id}

    # create epochs including both first and last trial, with baseline correction
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_ids,
        tmin=tmin,
        tmax=tmax,
        picks=['seeg'],
        reject=None,
        baseline=None,
        preload=True
    )
    return epochs


def apply_baseline(epoch_power: np.ndarray, baseline_power: np.ndarray, method: str = 'mean') -> np.ndarray:
    """apply_baseline Baseline correct a single epoch.

    Apply baseline correction on time-frequency power array for a single epoch, given the time-frequency power
    for a baseline period.

    Available methods include:
        - 'mean'
        - 'ratio'
        - 'logratio'
        - 'zscore'
        - 'zlogratrio'

    [FIXME]
    - add the option to apply baseline correction using raw signal, ie. mne Epochs object.

    Args:
        epoch_power (np.ndarray): The epoch to baseline correct. Should have shape [len(frequencies), len(time)]
        baseline_power (np.ndarray): the baseline to use for correction. Should have shape [len(frequencies), len(time)]
        method (str, optional): The method to use for baseline correcting the epoch. Defaults to 'mean'.

    Returns:
        epoch_corrected (nd.array): Has same dimensions as input epoch_power [len(frequencies), len(time)].
    """
    try:
        # use desired baseline correction method
        if method == 'mean':
            def func(d, m):
                d = (d.T - m).T  # subtract mean of baseline
                return d
        elif method == 'ratio':
            def func(d, m):
                d = (d.T / m).T  # divide by mean of baseline
                return d
        elif method == 'logratio':
            def func(d, m):
                d = (d.T / m).T  # divide by standard deviation of baseline
                # log transform the epoch [FIXME] perhaps log10 should be used?
                d = np.log(d)
                return d
        elif method == "zscore":
            def func(d, m, s):
                d = (d.T - m).T  # subtract mean of baseline
                d = (d.T / s).T  # divide by standard deviation of baseline
                return d
        elif method == "zlogratio":
            def func(d, m, s):
                d = (d.T / m).T  # divide by mean of baseline
                # log transform the epoch [FIXME] perhaps log10 should be used?
                d = np.log(d)
                d = (d.T / s).T  # divide by standard deviation of baseline
                return d
        else:
            raise ValueError(
                "method should be 'mean', 'ratio', 'logratio', 'zscore', or 'zlogratio'"
            )

        # calculate mean of baseline across time, vector output = [freq_dim,]
        mean = np.mean(baseline_power, axis=1)

        # baseline correct according to the desired method
        if method == 'zscore' or method == 'zlogratio':
            # calculate the standard deviation of baseline across time. Vector output = [freq_dim,]
            std = np.std(baseline_power, axis=1)
            epoch_corrected = func(d=epoch_power, m=mean, s=std)
        else:
            epoch_corrected = func(d=epoch_power, m=mean)
        return epoch_corrected

    except Exception as exc:
        raise Exception(
            f"Encountered an exception during baseline correction, error [{exc}]")


def compute_fft(signal, fs, output: str = 'mag'):
    """compute_fft Compute the Fast Fourier Transform of a signal.

    Args:
        signal (_type_): Signal to apply FFT.
        fs (_type_): Sampling frequency.
        output (str): Format of the output, choices are 'psd' for power spectral density,
                      and 'mag' for magnitude spectrum. Defaults to 'mag'. 

    Returns:
        x (np.array): Frequency vector, one-sided.
        y (np.ndarray): The complex discrete Fourier transform. 
    """
    from scipy.fft import fft, fftfreq

    y = fft(signal)                 # FFT
    xf = fftfreq(signal.size, 1/fs)  # Frequency vector
    xf = xf[xf >= 0]
    y = np.abs(np.ravel(y))[:len(xf)]

    if output == 'mag':
        return xf, y
    elif output == 'psd':
        psd = 20*np.log10(y)
        return xf, psd


# own implementation of morlet wavelet
def morlet(f0, n, t, return_gaussian=False):
    """
    Complex Morlet wavelet with frequency f and number of cycles n.
    """
    import warnings
    warnings.filterwarnings(action="error", category=np.ComplexWarning)
    with np.errstate(invalid='raise'):
        try:
            # create complex sine wave
            sine_wave = np.exp(2 * 1j * np.pi * f0 * t, dtype=np.complex128)

            # create gaussian envelope
            sigma = n / (2 * np.pi * f0)  # standard deviation for gaussian
            gaussian = np.exp(-t**2 / (2 * sigma**2))

            # create wavelet
            wavelet = np.multiply(sine_wave, gaussian, dtype=np.complex128)

            if return_gaussian:
                return wavelet, gaussian
            else:
                return wavelet
        except FloatingPointError as e:
            print(f"NumPy warning: {e}")
        except Exception as e:
            print(f"Exception: {e}")


# used for PyWavelets
def get_scales(freqs, wavelet, fs):
    frequencies = np.array(freqs) / fs  # normalise frequencies
    return pywt.frequency2scale(wavelet, frequencies)


def check_trials(df, col_name, tmin, tmax, baseline, verbose=False):
    # Check if any trials are too short for the given tmax
    if verbose:
        print(f"Checking if any trials are too short for tmax = {tmax} ...")
    idx_too_short = df[df['Reaction Time (computed)'] <= tmax].index.to_list()
    if len(idx_too_short) > 0:
        labels_too_short = df.loc[idx_too_short, col_name].to_list()
        dict_too_short = {}
        for i, label in zip(idx_too_short, labels_too_short):
            if label not in dict_too_short.keys():
                dict_too_short[label] = 1
            else:
                dict_too_short[label] += 1
        for key, value in dict_too_short.items():
            if verbose:
                print(f"{value} trials in condition {repr(key)} too short.")
    else:
        if verbose:
            print("All trials are longer than tmax.")

    # check if any trials are too close to each other wrt. tmin and baseline
    if verbose:
        print(
            f"Checking if any trials are too close to each other for tmin = {tmin} s and baseline tmin = {np.min(baseline)} s ...")
    idx_too_close = []
    for i, time_placed in enumerate(df['Timestamp (s) for Picture Placed']):
        if i == len(df) - 1:
            break
        time_between = (df['Timestamp (s) for Picture Shown']
                        [i+1] - time_placed)
        if time_between < abs(tmin) + .2:  # add 200 ms
            # print(f"{time_between:.2f} s is not enough time between trials {repr(data.df_exp['Trial Identifier'][i+1])} and {repr(data.df_exp['Trial Identifier'][i])} for the chosen tmin ({np.min(baseline)} s), removing trial {repr(data.df_exp['Trial Identifier'][i+1])} with condition {repr(data.df_exp['Condition'][i+1])}.")
            idx_too_close.append(i+1)
        elif time_between < abs(np.min(baseline)) + .2:  # add 200 ms
            # print(f"{time_between:.2f} s is not enough time between trials {repr(data.df_exp['Trial Identifier'][i+1])} and {repr(data.df_exp['Trial Identifier'][i])} for the chosen baseline tmin ({np.min(baseline)} s), removing trial {repr(data.df_exp['Trial Identifier'][i+1])} with condition {repr(data.df_exp['Condition'][i+1])}.")
            idx_too_close.append(i+1)
    if len(idx_too_close) > 0:
        labels_too_close = df.loc[idx_too_close, col_name].to_list()
        dict_too_close = {}
        for i, label in zip(idx_too_close, labels_too_close):
            if label not in dict_too_close.keys():
                dict_too_close[label] = 1
            else:
                dict_too_close[label] += 1
        for key, value in dict_too_short.items():
            if verbose:
                print(f"{value} trials in condition {repr(key)} too close.")
    else:
        if verbose:
            print("All trials are far enough apart.")

    # make sure that idx_too_close and idx_too_short are disjoint
    for i in idx_too_short:
        if i in idx_too_close:
            idx_too_close.remove(i)

    return idx_too_short, idx_too_close


def get_conditions_avg_power(power_data, conditions, verbose=False):
    """get_conditions_avg_power Average power across all channels (also across subjects) for each condition.

    Args:
        power_data (dict): Dictionary with all subjects' power data for each condition and channel.
        conditions (tuple, list): Tuple with conditions to average power across.

    Returns:
        power (dict): Dictionary with averaged power data for each condition.
    """
    power = {}
    for condition in conditions:
        if verbose:
            print(f"condition: {repr(condition)}")
        power[condition] = []
        for subject_id in power_data.keys():
            for channel in power_data[subject_id][condition].keys():
                # save channel data
                power[condition].append(
                    power_data[subject_id][condition].get(channel))
        if verbose:
            print("Shape before average:")
        if verbose:
            print(f"\t{np.array(power[condition]).shape}")
        # compute mean of all channels across subjects for condition
        power[condition] = np.mean(power[condition], axis=0)
        if verbose:
            print("Shape after average:")
        if verbose:
            print(f"\t{np.array(power[condition]).shape}\n")
    return power


def get_total_avg_power(power_data, conditions):
    """get_total_avg_power Average power across all conditions and channels (also across subjects).

    Args:
        power_data (dict): Dictionary with all subjects' power data for each condition and channel.
        conditions (tuple, list): Tuple with conditions to average power across.

    Returns:
        power (np.ndarray): Total averaged power.
    """
    power = []
    for condition in conditions:
        print(f"condition: {repr(condition)}")
        for subject_id in power_data.keys():
            for channel in power_data[subject_id][condition].keys():
                # save channel data
                power.append(power_data[subject_id][condition].get(channel))
    print("Shape before average:")
    print(f"\t{np.array(power).shape}")
    # compute mean of all channels across subjects
    power = np.mean(power, axis=0)
    print("Shape after average:")
    print(f"\t{np.array(power).shape}\n")
    return power


def check_period_for_ieds(ch_name, df_ied, i_start, i_end):
    """check_period_for_ieds Check the inter-event intervals between two events.

    Args:
        ch_name (str): Channel name.
        df_ied (pd.DataFrame): DataFrame with IED timepoints.
        i_start (int): Index of start of period to check.
        i_end (int): Index of end of period to check.

    Returns:
        bool: True if IED is within the desired range, False otherwise.
    """
    bool_ied = False
    ied_timepoints = df_ied.loc[df_ied['chan name']
                                == ch_name]['time point'].to_numpy()
    period_to_check = np.arange(start=i_start, stop=i_end+1, step=1)

    for ied in ied_timepoints:
        if ied in period_to_check:
            bool_ied = True
            break
    return bool_ied

# [INFO] Not finished


def compute_tfr(
        data,
        fs: int = 512,
        freqs: list | np.ndarray = np.arange(1, 40)):
    """compute_tfr _summary_

    _extended_summary_

    Args:
        data (_type_): _description_
        fs (int, optional): _description_. Defaults to 512.
        freqs (list | np.ndarray, optional): _description_. Defaults to np.arange(1, 40).
    """
    # ********** CHECK INPUTS **********#
    if data is None:
        raise ValueError('data must be provided.')
    elif not isinstance(data, (list, np.ndarray)):
        raise ValueError(
            f'data must be a list or numpy array, got {type(data)}.')
    if fs is None:
        raise ValueError('fs must be provided.')
    elif not isinstance(fs, int):
        raise ValueError(f'fs must be an integer, got {type(fs)}.')
    if freqs is None:
        raise ValueError('freqs must be provided.')
    elif not isinstance(freqs, (list, np.ndarray)):
        raise ValueError(
            f'freqs must be a list or numpy array, got {type(freqs)}.')
    elif not all(isinstance(f, (int, float)) for f in freqs):
        raise ValueError(
            'freqs must be a list or numpy array of integers or floats.')

    # ********** COMPUTE TFR **********#

# [REMOVE] Not used


def tfr_clust_perm_test(
    epochs_a,
    raw,
    df_exp: pd.DataFrame,
    freqs: list | np.ndarray,
    n_cycles: int,
    baseline: list = [None, None],
    baseline_mode: str = 'mean',
    epochs_b=None,
    fs: int = 512,
    n_permutations: int = 1000,
    p_value: int = 0.05,
    tail: int = 1,
    n_jobs: int = -1,
    decim: int = 5,
    random_seed: int = 23,
    plot_results: bool = True,
    copy: bool = True,
    path: str = None,
    subject_id: str = None
):
    """tfr_clust_perm_test _summary_

    _extended_summary_

    Args:
        epochs_a (_type_): _description_
        freqs (list | np.ndarray): _description_
        n_cycles (int): _description_
        baseline (list, optional): _description_. Defaults to [None, None].
        epochs_b (_type_, optional): _description_. Defaults to None.
        n_permutations (int, optional): _description_. Defaults to 1000.
        p_value (int, optional): _description_. Defaults to 0.05.
        n_jobs (int, optional): _description_. Defaults to -1.
        decim (int, optional): Factor to down-sample the temporal dimension of the TFR computed by
            tfr_morlet. Decimation occurs after frequency decomposition and can
            be used to reduce memory usage (and possibly computational time of downstream
            operations such as nonparametric statistics) if you don't need high
            spectrotemporal resolution.. Defaults to 5.

    steps:
        1. compute tfr for epochs_a and epochs_b (if provided)
        2. baseline correct tfrs
        3. perform cluster permutation test using baseline if no condition b, otherwise condition a vs condition b
        4. plot results

    """
    # ********** CHECK INPUTS **********#
    if epochs_a is None:
        raise ValueError('epochs_a must be provided.')
    if epochs_b is not None:
        if epochs_a._data.shape != epochs_b._data.shape:
            raise ValueError('epochs_a and epochs_b must have the same shape.')
    if fs is None:
        raise ValueError('fs must be provided.')
    if n_cycles is None:
        raise ValueError('n_cycles must be provided.')
    if freqs is None:
        raise ValueError('freqs must be provided.')
    if baseline is None:
        raise ValueError('baseline must be provided.')

    # ********** FUNCTIONS **********#
    def _create_baseline_epochs(raw, df_exp, tmin, tmax, event_str):
        """_create_baseline_epochs _summary_

        _extended_summary_

        Args:
            raw (_type_): _description_
            df_exp (_type_): _description_
            tmin (_type_): _description_
            tmax (_type_): _description_
            event_str (_type_): _description_

        Returns:
            _type_: _description_
        """
        # create 2D array with zeros of same length as channel data array
        stim_arr = np.zeros((1, len(raw)))

        # get indices for first and last trials from experiment dataframe
        idx_first_trial = []
        idx_last_trial = []
        for pic_num in sorted(df_exp['Picture Number'].unique()):
            idx_first_trial.append(
                df_exp[df_exp['Picture Number'] == pic_num][event_str].astype(int).to_numpy()[0])
            idx_last_trial.append(
                df_exp[df_exp['Picture Number'] == pic_num][event_str].astype(int).to_numpy()[-1])

        # set indices to 1 where the picture was first placed and 2 where the picture was correctly placed, this will be the stimulus channel
        stim_arr[0, idx_first_trial] = 10
        stim_arr[0, idx_last_trial] = 20

        # create info object for stimulus channel
        info = mne.create_info(
            ch_names=['STI_TMP'],
            sfreq=raw.info['sfreq'],
            ch_types=['stim'])

        # create raw array with trigger channel
        stim_raw = mne.io.RawArray(stim_arr, info)

        # add stimulus channel to raw object
        raw.add_channels([stim_raw], force_update_info=True)

        # find events from stimulus channel
        events = mne.find_events(raw, stim_channel="STI_TMP")
        raw.drop_channels(['STI_TMP'])

        # create mapping between event_id to event description
        event_ids = dict(first_trial=10, last_trial=20)

        # create epochs including both first and last trial, with baseline correction
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_ids,
            tmin=tmin,
            tmax=tmax,
            picks=['eeg'],
            reject=None,
            baseline=None,
            preload=True
        )

        return epochs

    def _apply_baseline(epochs, raw, markers: list, baseline: list = baseline, fs: int = fs, mode: str = baseline_mode, trial_type: str = 'first', copy: bool = copy):
        """_apply_baseline _summary_

        _extended_summary_

        Args:
            epochs (_type_): _description_
            raw (_type_): _description_
            baseline (_type_): _description_
            markers (_type_): _description_
            mode (str, optional): _description_. Defaults to 'mean'.
            copy (bool, optional): _description_. Defaults to True.
        """
        if copy:
            epochs = epochs.copy()

        if mode == 'mean':
            def func(d, m):
                d -= m
        elif mode == 'ratio':
            def func(d, m):
                d /= m
        elif mode == 'logratio':
            def func(d, m):
                d /= m
                d = np.log(d)
        elif mode == "zscore":
            def func(d, m, s):
                d -= m
                d /= s
        elif mode == "zlogratio":
            def func(d, m, s):
                d /= m
                d = np.log(d)
                d /= s
        else:
            raise ValueError(
                "mode should be 'mean', 'ratio', 'logratio', 'zscore', or 'zlogratio'"
            )

        # create baseline period epochs from raw data
        baseline_epochs = _create_baseline_epochs(
            raw=raw, df_exp=df_exp, tmin=baseline[0], tmax=baseline[1], event_str='Mark for Picture Shown')

        # compute time-frequency representation for baseline epochs
        baseline_tfr = tfr_morlet(
            inst=baseline_epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=decim,
            average=False,
            return_itc=False,
            n_jobs=n_jobs
        )

        # get power for baseline period wrt. trial type
        num_observations = epochs.data.shape[0]
        if trial_type == 'first':
            baseline_power = baseline_tfr.data[:num_observations, ...]
        elif trial_type == 'last':
            baseline_power = baseline_tfr.data[num_observations:, ...]

        # get number of trials and number of channels
        num_trials, num_channels = epochs.data.shape[:-2]

        # iterate over trials
        for i, trial in enumerate(range(num_trials)):

            # compute baseline period for trial i
            onset = markers[i]
            start = round(onset + (fs * baseline[0]))
            end = round(onset + (fs * baseline[1]))

            # iterate over all channels (ie. contacts)
            for channel, channel_name in zip(range(num_channels), epochs.ch_names):

                # get baseline period data
                baseline_data = baseline_power[i, channel, ...]

                # calculate mean of baseline period
                mean = baseline_data.mean()

                # baseline correct according to the desired mode
                if mode == 'zscore' or mode == 'zlogratio':
                    std = np.std(baseline_data, axis=-1, keepdims=True)
                    func(d=epochs.data[trial, channel, ...], m=mean, s=std)
                else:
                    func(d=epochs.data[trial, channel, ...], m=mean)

        return epochs

    def _get_adjaceny(epochs, tfr_epochs):
        """_get_adjaceny Compute the adjacency matrix for permutation cluster test. 

        Args:
            epochs (mne.Epochs): The epochs object for which to compute the adjacency matrix.
            tfr_epochs (mne.time_frequency.tfr.EpochsTFR): The tfr epochs object for which to compute the adjacency matrix.

        Returns:
            adjecency (ndarray): The adjacency matrix for the given epochs.
        """

        # we need to prepare adjacency information for the time-frequency
        # plane. For that, we use "combine_adjacency", and pass dimensions
        # as in the data we want to test (excluding observations). Here:
        # epochs × channels × frequencies × times
        assert epochs.data.shape == (
            len(epochs),
            len(tfr_epochs.ch_names),
            len(tfr_epochs.freqs),
            len(tfr_epochs.times),
        )
        adjacency = mne.stats.combine_adjacency(
            len(tfr_epochs.ch_names), len(
                tfr_epochs.freqs), len(tfr_epochs.times)
        )
        # The overall adjacency we end up with is a square matrix with each
        # dimension matching the data size (excluding observations) in an
        # "unrolled" format, so: len(channels × frequencies × times)
        assert (
            adjacency.shape[0]
            == adjacency.shape[1]
            == len(tfr_epochs.ch_names) * len(tfr_epochs.freqs) * len(tfr_epochs.times)
        )
        return adjacency

    # ********** GET DATA **********#
    evoked = epochs_a.average()

    # ********** TIME-FREQUENCY COMPUTATION AND BASELINE CORRECTION **********#
    # Condition A
    tfr_epochs_a = tfr_morlet(
        inst=epochs_a,
        freqs=freqs,
        n_cycles=n_cycles,
        decim=decim,
        average=False,
        return_itc=False,
        n_jobs=n_jobs,
    )
    # Baseline correct TFR for condition A
    markers = []
    for pic_num in df_exp['Picture Number'].unique():
        markers.append(df_exp[df_exp['Picture Number'] ==
                       pic_num]['Mark for Picture Shown'].to_numpy()[0])
    markers = np.sort(markers)
    tfr_epochs_a = _apply_baseline(epochs=tfr_epochs_a, raw=raw, baseline=baseline,
                                   fs=fs, markers=markers, mode=baseline_mode, trial_type='first', copy=copy)
    # get TFR power for condition A
    epochs_power_a = tfr_epochs_a.data

    # Condition B (if provided)
    if epochs_b is not None:
        tfr_epochs_b = tfr_morlet(
            inst=epochs_b,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=decim,
            average=False,
            return_itc=False,
            n_jobs=n_jobs
        )
        # Baseline correct TFR for condition B
        markers = []
        for pic_num in df_exp['Picture Number'].unique():
            markers.append(df_exp[df_exp['Picture Number'] == pic_num]
                           ['Mark for Picture Shown'].to_numpy()[-1])
        markers = np.sort(markers)
        tfr_epochs_b = _apply_baseline(epochs=tfr_epochs_b, raw=raw, baseline=baseline,
                                       markers=markers, mode=baseline_mode, trial_type='last', copy=copy)
        # get TFR power for condition B
        epochs_power_b = tfr_epochs_b.data

    if epochs_b is None:
        epochs_power = epochs_power_a
    elif epochs_b is not None:
        epochs_power = epochs_power_a - epochs_power_b

    # ********** ORIGINAL CLUSTERS **********#
    T_obs_orig = np.nan * np.ones_like(epochs_power[0, ...])
    clusters_p_value_orig = np.zeros_like(epochs_power[0, ...])

    hypothesis_vector = np.zeros(epochs_power.shape[0])

    # shape: (epochs/observations, channels, frequencies, times)
    for ch in range(epochs_power.shape[1]):

        # vector to hold all observations to perform paired t-test
        # note:
        #   vector will be of same length as there are observations,
        #   in this case observations are the number of epochs
        obs_vector = np.zeros(epochs_power.shape[0])

        for f in range(epochs_power.shape[2]):
            for t in range(epochs_power.shape[3]):
                for i, e in enumerate(range(epochs_power.shape[0])):
                    obs_vector[i] = epochs_power[e][ch][f][t]

                    # finished building obs_vector
                    if i == epochs_power.shape[0] - 1:

                        # perform paired t-test
                        t_stat, p_val = scipy.stats.ttest_rel(
                            a=obs_vector, b=hypothesis_vector, axis=0, alternative='greater')

                        # save results
                        T_obs_orig[ch][f][t] = t_stat
                        clusters_p_value_orig[ch][f][t] = p_val

    # ********** PLOT ORIGINAL CLUSTERS **********#

    # plt.figure(figsize=(20, 3))

    # create cluster matrix of T-statistics value for plotting
    T_obs_orig_plot = np.nan * np.ones_like(T_obs_orig)
    for ch in range(T_obs_orig.shape[0]):
        for f in range(T_obs_orig.shape[1]):
            for t in range(T_obs_orig.shape[2]):
                if clusters_p_value_orig[ch][f][t] <= p_value:
                    T_obs_orig_plot[ch][f][t] = T_obs_orig[ch][f][t]

    vmax_ft = np.max(np.abs(T_obs_orig))
    vmin_ft = -vmax_ft

    times = 1e3 * epochs_a.times  # times by 1e3 to change unit to ms

    if len(epochs_a.ch_names) <= 4:
        nrows_plot = 1
        ncols_plot = len(epochs_a.ch_names)
        figsize = (20, 3)
    elif len(epochs_a.ch_names) > 4 and len(epochs_a.ch_names) <= 8:
        nrows_plot = 2
        ncols_plot = math.ceil(len(epochs_a.ch_names) / 2)
        figsize = (20, 6)
    else:
        nrows_plot = 3
        ncols_plot = math.ceil(len(epochs_a.ch_names) / 3)
        figsize = (20, 9)

    fig, axs = plt.subplots(
        nrows=nrows_plot, ncols=ncols_plot, figsize=figsize, sharex=True, sharey=True)
    axs = axs.flatten()
    for _ch in range(len(epochs_a.ch_names)):
        # plot grayscale TFR
        axs[_ch].imshow(
            T_obs_orig[_ch],
            cmap=plt.cm.gray,
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            vmin=vmin_ft,
            vmax=vmax_ft,
        )
        # plot significant clusters in colour
        axs[_ch].imshow(
            T_obs_orig_plot[_ch],
            cmap=plt.cm.RdBu_r,
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            aspect="auto",
            origin="lower",
            vmin=vmin_ft,
            vmax=vmax_ft,
        )
        axs[_ch].set_xlabel("Time (ms)", fontsize=12)
        axs[_ch].set_ylabel("Frequency (Hz)", fontsize=12)
        axs[_ch].set_title(
            f"Channel {epochs_a.ch_names[_ch]}", fontsize=14)

    # for _ch in range(epochs_power.shape[1]):
    #     plt.subplot(1, len(epochs_a.ch_names), _ch + 1)
    #     # plot grayscale TFR
    #     plt.imshow(
    #         T_obs_orig[_ch],
    #         cmap=plt.cm.gray,
    #         extent=[times[0], times[-1], freqs[0], freqs[-1]],
    #         aspect="auto",
    #         origin="lower",
    #         vmin=vmin_ft,
    #         vmax=vmax_ft,
    #     )
    #     # plot significant clusters in colour
    #     plt.imshow(
    #         T_obs_orig_plot[_ch],
    #         cmap=plt.cm.RdBu_r,
    #         extent=[times[0], times[-1], freqs[0], freqs[-1]],
    #         aspect="auto",
    #         origin="lower",
    #         vmin=vmin_ft,
    #         vmax=vmax_ft,
    #     )
    #     plt.colorbar()
    #     plt.xlabel("Time (ms)")
    #     plt.ylabel("Frequency (Hz)")
    #     plt.title(f"Channel {epochs_a.ch_names[_ch]}")
    plt.suptitle(
        f"Subject {subject_id} Original Clusters (p-value = {p_value})", y=1.05, fontsize=16)
    # don't show empty axes
    if len(axs) != len(epochs_a.ch_names):
        for _ch in range(len(axs) - len(epochs_a.ch_names)):
            axs[-(_ch + 1)].set_visible(False)

    save_path = ''.join(
        (path, '/', f'sub{subject_id}_original_clusters', '.png'))
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    # ********** CLUSTER PERMUTATION TEST **********#
    # Define adjacency matrix for permutation test results
    adjacency = _get_adjaceny(epochs=epochs_power, tfr_epochs=tfr_epochs_a)

    # Compute cluster permutation test

    # In this example, we wish to set the threshold for including data bins in
    # the cluster forming process to the t-value corresponding to p=0.001 for the
    # given data.
    #
    # Because we conduct a two-tailed test, we divide the p-value by 2 (which means
    # we're making use of both tails of the distribution).
    # As the degrees of freedom, we specify the number of observations
    # (here epochs) minus 1.
    # Finally, we subtract 0.001 / 2 from 1, to get the critical t-value
    # on the right tail (this is needed for MNE-Python internals)
    degrees_of_freedom = len(epochs_power) - 1
    t_thresh = scipy.stats.t.ppf(1 - p_value, df=degrees_of_freedom)

    # Run the analysis
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        epochs_power,
        n_permutations=n_permutations,
        threshold=t_thresh,
        tail=tail,
        adjacency=adjacency,
        out_type="mask",
        n_jobs=n_jobs,
        seed=random_seed,
        verbose=True
    )

    # ********** PLOT PERMUTATION RESULTS **********#
    if plot_results:
        import matplotlib
        matplotlib.use('Agg')

        # ********** PLOT PERMUTATION RESULTS **********#
        T_obs_plot = np.nan * np.ones_like(T_obs)
        for c, p_val in zip(clusters, cluster_p_values):
            if p_val <= p_value:
                T_obs_plot[c] = T_obs[c]

        vmax_ft = np.max(np.abs(T_obs))
        vmin_ft = -vmax_ft

        times = 1e3 * evoked.times  # times by 1e3 to change unit to ms

        if len(epochs_a.ch_names) <= 4:
            nrows_plot = 1
            ncols_plot = len(epochs_a.ch_names)
            figsize = (20, 3)
            y_suptitle = 1.05
        elif len(epochs_a.ch_names) > 4 and len(epochs_a.ch_names) <= 8:
            nrows_plot = 2
            ncols_plot = math.ceil(len(epochs_a.ch_names) / 2)
            figsize = (20, 6)
            y_suptitle = 1.03
        else:
            nrows_plot = 3
            ncols_plot = math.ceil(len(epochs_a.ch_names) / 3)
            figsize = (20, 9)
            y_suptitle = 1.01

        fig, axs = plt.subplots(
            nrows=nrows_plot, ncols=ncols_plot, figsize=figsize, sharex=True, sharey=True)
        axs = axs.flatten()
        for _ch in range(len(epochs_a.ch_names)):
            # [INFO] if it should be a blue (vmin) and red (vmax) plot, then change graymap to RdBu_r and second plot only plots contour lines (eg. yellow) around significant clusters
            # [INFO] maybe change to plt.subplot(s) and use plt.add_subplot() for each channel
            # plot grayscale TFR
            axs[_ch].imshow(
                T_obs[_ch],
                cmap=plt.cm.gray,
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                aspect="auto",
                origin="lower",
                vmin=vmin_ft,
                vmax=vmax_ft,
            )
            # plot significant clusters in colour
            axs[_ch].imshow(
                T_obs_plot[_ch],
                cmap=plt.cm.RdBu_r,
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                aspect="auto",
                origin="lower",
                vmin=vmin_ft,
                vmax=vmax_ft,
            )
            axs[_ch].set_xlabel("Time (ms)", fontsize=12)
            axs[_ch].set_ylabel("Frequency (Hz)", fontsize=12)
            axs[_ch].set_title(
                f"Channel {epochs_a.ch_names[_ch]}", fontsize=14)
        plt.suptitle(
            f"Subject {subject_id} Cluster Permutation Test (p-value = {p_value})", fontsize=16, y=y_suptitle)
        # don't show empty axes
        if len(axs) != len(epochs_a.ch_names):
            for _ch in range(len(axs) - len(epochs_a.ch_names)):
                axs[-(_ch + 1)].set_visible(False)

        save_path = ''.join(
            (path, '/', f'sub{subject_id}_permutation_clusters', '.png'))
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # ********** RETURN RESULTS **********#
    results = {
        'T_obs_orig': T_obs_orig,
        'T_obs': T_obs,
        'clusters': clusters,
        'clusters_p_value_orig': clusters_p_value_orig,
        'cluster_p_values': cluster_p_values,
        'H0': H0
    }
    return results
