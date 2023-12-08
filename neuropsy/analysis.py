import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import math
import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_1samp_test


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
        # for _ch in range(epochs_power.shape[1]):
        #     plt.subplot(1, len(epochs_a.ch_names), _ch + 1)
        #     # plot grayscale TFR
        #     plt.imshow(
        #         T_obs[_ch],
        #         cmap=plt.cm.gray,
        #         extent=[times[0], times[-1], freqs[0], freqs[-1]],
        #         aspect="auto",
        #         origin="lower",
        #         vmin=vmin_ft,
        #         vmax=vmax_ft,
        #     )
        #     # plot significant clusters in colour
        #     plt.imshow(
        #         T_obs_plot[_ch],
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
        #     plt.title(f"Channel {tfr_epochs_a.ch_names[_ch]}")
        plt.suptitle(
            f"Subject {subject_id} Cluster Permutation Test (p-value = {p_value})", fontsize=16, y=1.05)
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
