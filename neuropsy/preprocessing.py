import pandas as pd
import numpy as np


def clean_outliers(df: pd.DataFrame, cols: (list, tuple, np.ndarray), num_std: int = 3, verbose: bool = False):
    """clean_outliers Remove outliers in given columns.

    Args:
        df (pd.DataFrame): Dataframe with data to clean.
        cols (list | tuple | np.array): Columns to inspect for outliers.
        num_std (int, optional): Number of standard deviations to use to determine outliers. Defaults to 3.
        verbose (bool, optional): Print verbose to stdout. Defaults to False.

    Returns:
        df (pd.DataFrame): Dataframe with removed outliers.
    """
    def _determine_outlier_thresholds_std(df, col_name, num_std):
        upper_boundary = df[col_name].mean(
        ) + num_std * df[col_name].std()
        lower_boundary = df[col_name].mean(
        ) - num_std * df[col_name].std()
        return lower_boundary, upper_boundary

    def _check_outliers_std(df, col_name, num_std):
        lower_boundary, upper_boundary = _determine_outlier_thresholds_std(
            df, col_name, num_std)
        if df[(df[col_name] > upper_boundary) | (df[col_name] < lower_boundary)].any(axis=None):
            return True
        else:
            return False

    def _get_outlier_indices(df, cols, num_std):
        data = []
        for col_name in cols:
            outliers_ = _check_outliers_std(df, col_name, num_std)
            count = None
            lower_limit, upper_limit = _determine_outlier_thresholds_std(
                df, col_name, num_std)
            if outliers_:
                count = df[(df[col_name] > upper_limit) | (
                    df[col_name] < lower_limit)][col_name].count()
                indices = df[(df[col_name] > upper_limit) | (
                    df[col_name] < lower_limit)].index.to_numpy()
            outliers_status = _check_outliers_std(
                df, col_name, num_std)
            data.append([outliers_, outliers_status, count,
                        col_name, lower_limit, upper_limit])
        if verbose:
            from tabulate import tabulate
            print(f"Outliers (using {num_std} Standard Deviation)")
            table = tabulate(data, headers=['Outlier (Previously)', 'Outliers', 'Count',
                                            'Column', 'Lower Limit', 'Upper Limit'], tablefmt='rst', numalign='right')
            print(table)
        return indices

    # check arguments
    if not (isinstance(verbose, bool) or verbose is None):
        raise ValueError(
            f"verbose must be True or False or None, got {type(verbose)}")
    elif verbose is None:
        verbose = False
    else:
        pass
    if isinstance(df, pd.DataFrame):
        if df.shape[0] == 0:
            raise ValueError(
                f"df must not be empty, got {df.shape[0]} rows")
    else:
        raise ValueError(
            f"df must be a pandas DataFrame, got {type(df)}")
    if isinstance(cols, list) or isinstance(cols, tuple) or isinstance(cols, np.ndarray):
        if len(cols) == 0:
            raise ValueError(
                f"cols must not be empty, got {len(cols)} columns")
        else:
            for col in cols:
                if not col in df.columns:
                    raise ValueError(
                        f"column {repr(col)} not found in df")
    else:
        raise ValueError(
            f"cols must be a list, tuple, or array, got {type(cols)}")
    if isinstance(num_std, int):
        if num_std <= 0:
            raise ValueError(
                f"num_std must be greater than 0, got {num_std}")
    else:
        raise ValueError(
            f"num_std must be an integer, got {type(num_std)}")

    # start cleaning
    len_before = len(df)
    if verbose:
        print(f"cleaning outliers in {len_before} trials")

    # remove outliers
    idx = _get_outlier_indices(
        df=df,
        cols=cols,
        num_std=num_std
    )
    if verbose:
        print(f"found outliers in indices {idx}")
    df = df.drop(idx, inplace=False).reset_index(drop=True)

    # if verbose:
    #     print(f"removed {len_before - len(df)} trials")
    #     print(f"remaining {len(df)} trials")

    return df, idx


def clean_nan(df: pd.DataFrame, cols: (list, tuple, np.ndarray), verbose: bool = False):
    """clean_nan Remove NaNs in given columns.

    Args:
        df (pd.DataFrame): Dataframe with data to clean.
        cols (list | tuple | np.array): Columns to inspect for NaNs.
        verbose (bool, optional): Print verbose to stdout. Defaults to False.

    Returns:
        df (pd.DataFrame): Dataframe with removed NaNs.
    """
    # check arguments
    if not (isinstance(verbose, bool) or verbose is None):
        raise ValueError(
            f"verbose must be True or False or None, got {type(verbose)}")
    elif verbose is None:
        verbose = False
    else:
        pass
    if isinstance(df, pd.DataFrame):
        if df.shape[0] == 0:
            raise ValueError(
                f"df must not be empty, got {df.shape[0]} rows")
    else:
        raise ValueError(
            f"df must be a pandas DataFrame, got {type(df)}")
    if isinstance(cols, list) or isinstance(cols, tuple) or isinstance(cols, np.ndarray):
        if len(cols) == 0:
            raise ValueError(
                f"cols must not be empty, got {len(cols)} columns")
        else:
            for col in cols:
                if not col in df.columns:
                    raise ValueError(
                        f"column {repr(col)} not found in df")
    else:
        raise ValueError(
            f"cols must be a list, tuple, or array, got {type(cols)}")

    # start cleaning
    len_before = len(df[cols])
    if verbose:
        print(f"cleaning NaNs in {len_before} trials")

    # find NaNs in provided columns
    idx = df[df[cols].isna().any(axis=1)].index.to_numpy()
    if verbose:
        print(f"found NaNs at indices {idx}")
    df = df.drop(idx, inplace=False).reset_index(drop=True)

    if verbose:
        print(f"removed {len_before - len(df)} trials")
        print(f"remaining {len(df)} trials")

    return df, idx


def remove_artifacts_manually():
    """remove_artifacts_manually Remove artifacts in trials manually by inspecting plots for interictal epileptic discharges (IEDs).

    [TODO]
    """
    pass


def filter(data: np.ndarray, filter_dict: dict = {"order": 4, "cutoff": 500, "fs": 512, "filter_type": 'low', "apply_filter": True, "show_bodeplot": False, "show_impulse_response": False, "len_impulse": 1000}, use_qt: bool = False):
    """filter Filter the data.

    [TODO]
    """
    from neuropsy.utils.filters import digital_filter

    # check arguments
    if isinstance(data, np.ndarray):
        if data.ndim > 2:
            raise ValueError(
                f"data must be 1D or 2D array, got {data.ndim}D array")
        elif data.ndim == 1:
            data = data.reshape(1, -1)
    else:
        raise ValueError(
            f"data must be a numpy array, got {type(data)}")
    if data.ndim == 2:
        if data.shape[0] == 0:
            raise ValueError(
                f"data must not be empty, got {data.shape[0]} rows")
        elif data.shape[1] == 0:
            raise ValueError(
                f"data must not be empty, got {data.shape[1]} columns")
        else:
            pass
    if not isinstance(filter_dict, dict):
        raise ValueError(
            f"filter_dict must be a dictionary, got {type(filter_dict)}")

    for ch in range(data.shape[0]):
        # create filter and return second order sections
        if filter_dict['filter_mode'] == 'sos':
            sos = digital_filter(in_signal=data[ch, :],
                                 order=filter_dict['order'],
                                 cutoff=filter_dict['cutoff'],
                                 fs=filter_dict['fs'],
                                 filter_type=filter_dict['filter_type'],
                                 apply_filter=filter_dict['apply_filter'],
                                 show_bodeplot=filter_dict['show_bodeplot'],
                                 show_impulse_response=filter_dict['show_impulse_response'],
                                 len_impulse=filter_dict['len_impulse'],
                                 filter_mode=filter_dict['filter_mode'],
                                 use_qt=use_qt)
            break

        # create 2nd order cascaded filter and return numerator/denominator filter coefficients
        elif filter_dict['filter_mode'] == 'ba':
            b, a = digital_filter(in_signal=data[ch, :],
                                  order=filter_dict['order'],
                                  cutoff=filter_dict['cutoff'],
                                  fs=filter_dict['fs'],
                                  filter_type=filter_dict['filter_type'],
                                  apply_filter=filter_dict['apply_filter'],
                                  show_bodeplot=filter_dict['show_bodeplot'],
                                  show_impulse_response=filter_dict['show_impulse_response'],
                                  len_impulse=filter_dict['len_impulse'],
                                  filter_mode=filter_dict['filter_mode'],
                                  use_qt=use_qt)
            break

        # create and apply filter and return filtered data
        else:
            data[ch, :] = digital_filter(in_signal=data[ch, :],
                                         order=filter_dict['order'],
                                         cutoff=filter_dict['cutoff'],
                                         fs=filter_dict['fs'],
                                         filter_type=filter_dict['filter_type'],
                                         apply_filter=filter_dict['apply_filter'],
                                         show_bodeplot=filter_dict['show_bodeplot'],
                                         show_impulse_response=filter_dict['show_impulse_response'],
                                         len_impulse=filter_dict['len_impulse'],
                                         filter_mode=filter_dict['filter_mode'],
                                         use_qt=use_qt)

    if filter_dict['apply_filter']:
        return data
    elif filter_dict['filter_mode'] == 'sos':
        return sos
    elif filter_dict['filter_mode'] == 'ba':
        return b, a


def reference(data: np.ndarray, method: str, ch_names: (list, tuple, np.ndarray) = None, ref_channel: int = None, direction: str = None, verbose: bool = False):
    """reference Reference the data using one of the available methods.

    [TODO]
    """

    def _monopolar(data, ref_channel, verbose):
        """_monopolar Reference all channel to a single reference channel.

        Referencing all channels to a single reference channel is done by subtracting the reference channel
        from each of the other channels. The reference channel is usually the mastoid channel.

        Args:
            data (np.array): 2D array with channel data.
            ref_channel (int): The index to use as reference channel. Default is 0.

        Returns:
            new_data (ndarray): Re-referenced data.
        """

        # check arguments
        if isinstance(ref_channel, int):
            if ref_channel >= len(data[0]) - 1:
                raise ValueError(
                    f"ref_channel must be smaller than the number of channels in the data. Got {ref_channel} but expected {data.shape[0]}")
        else:
            raise ValueError(
                f"ref_channel must be an integer, got {repr(ref_channel)}")
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(
                    f"data must be a 2D array (channels, data), got {data.ndim}D array")
        else:
            raise ValueError(f"data must be a numpy array, got {type(data)}")

        # start re-referencing
        if verbose:
            print("applying re-referencing method: 'monopolar'")
        if verbose:
            print(f"reference channel: {ref_channel}")
        # get all channels except the reference channel
        selector = [i for i in range(data.shape[0]) if i != ref_channel]
        # subtract the reference channel from all other channels
        data[selector, :] = data[selector, :] - data[ref_channel, :]
        if verbose:
            print("successfully re-referenced data!")
        return data

    def _bipolar(data: np.array, ch_names: (list, np.array), direction: str, verbose: bool):
        """_bipolar Bipolar Reference channels using neighbouring channel as reference.

        Args:
            data (np.array): 2D array with channel data.
            ch_names (list, np.array): 1D list or array of channel names. Must be same order as the
                                       channels appear in the data. Default is None.
            direction (str): The direction of the bipolar referencing. If 'right' the n+1 channel will
                             substracted from the n channel. If 'left' the n channel will be 
                             substracted from the n+1 channel. Default is 'right'.

        Returns:
            new_data (ndarray): Re-referenced data with N-1 channels.
        """
        # check arguments
        if isinstance(direction, str):
            if direction == 'right' or direction == 'left':
                pass
            else:
                raise ValueError(
                    f"direction must be either 'right' or 'left', got {repr(direction)}")
        else:
            raise ValueError(
                f"direction must be a string, got {repr(direction)}")
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(
                    f"data must be a 2D array (channels, data), got {data.ndim}D array")
        else:
            raise ValueError(f"data must be a numpy array, got {type(data)}")
        if isinstance(ch_names, (list, np.ndarray)):
            if len(ch_names) != data.shape[0]:
                raise ValueError(
                    f"ch_names must be the same length as the number of channels in the data. Got {len(ch_names)} but expected {data.shape[0]}")
        else:
            raise ValueError(
                f"ch_names must be a list or numpy array, got {type(ch_names)}")

        # start re-referencing
        if verbose:
            print("applying re-referencing method: 'bipolar'")
        # get the unique channels names from ch_names
        unique_electrodes = _get_unique_channels(ch_names)
        if verbose:
            print(
                f"found {len(unique_electrodes)} unique electrodes and {len(ch_names)} channels")
        # tmp list find indices in list matching electrode name
        tmp_ch_names = np.array([ch.split(' ')[0] for ch in ch_names])

        # iterate over unique electrodes
        removed_indices = []
        removed_ch_names = []
        for electrode in unique_electrodes:
            indices = np.where(tmp_ch_names == electrode)[0]
            # iterate over N-1 contacts for each electrode
            for i in range(len(indices) - 1):
                if direction == 'right':
                    # subtract the next channel from the current channel
                    data[indices[i], :] = data[indices[i], :] - \
                        data[indices[i + 1], :]
                elif direction == 'left':
                    # subtract the previous channel from the current channel
                    data[indices[i + 1], :] = data[indices[i + 1], :] - \
                        data[indices[i], :]
            # remove the last channel depending on the direction
            if direction == 'right':
                removed_indices.append(indices[-1])
                removed_ch_names.append(ch_names[indices[-1]])
            elif direction == 'left':
                removed_indices.append(indices[0])
                removed_ch_names.append(ch_names[indices[0]])
        # remove the first or last contact (channel) in each electrode depending on the direction of the subtraction
        if verbose:
            print(f"removing {len(removed_indices)} channels from data")
            print(f"removing channels: {removed_ch_names}")
        data = np.delete(data, removed_indices, axis=0)
        if verbose:
            print("successfully re-referenced data!")
        return data, removed_ch_names, removed_indices

    def _laplacian(data: np.ndarray, ch_names: (list, np.array), verbose: bool):
        """Laplacian Reference channels using the average of neighbouring channels as reference.

        Args:
            data (np.ndarray): 2D array with channel data.
            ch_names (list, np.array): 1D list or array of channel names. Must be same order as the
                                       channels appear in the data.

        Returns:
            new_data (ndarray): Re-referenced data with N-2 channels.
            removed_indices (list): List of indices that could not be re-referenced.
        """
        # check arguments
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(
                    f"data must be a 2D array (channels, data), got {data.ndim}D array")
        else:
            raise ValueError(f"data must be a numpy array, got {type(data)}")
        if isinstance(ch_names, (list, np.ndarray)):
            if len(ch_names) != data.shape[0]:
                raise ValueError(
                    f"ch_names must be the same length as the number of channels in the data. Got {len(ch_names)} but expected {data.shape[0]}")
        else:
            raise ValueError(
                f"ch_names must be a list or numpy array, got {type(ch_names)}")

        # start re-referencing
        if verbose:
            print("applying re-referencing method: 'laplacian'")
        # get the unique channels names from ch_names
        unique_electrodes = _get_unique_channels(ch_names)
        if verbose:
            print(
                f"found {len(unique_electrodes)} unique electrodes and {len(ch_names)} channels")
        # tmp list of channel names to find indices in list matching electrode name
        tmp_ch_names = np.array([ch.split(' ')[0] for ch in ch_names])

        # iterate over unique electrodes
        removed_indices = []
        removed_ch_names = []
        for electrode in unique_electrodes:
            indices = np.where(tmp_ch_names == electrode)[0]
            # iterate over N-2 contacts for each electrode
            for i in range(len(indices) - 2):
                # subtract the mean of the previous and next channel from the current channel
                data[indices[i + 1], :] = data[indices[i + 1], :] - \
                    0.5 * (data[indices[i], :] + data[indices[i + 1], :])
            # remove the first and last channels
            removed_indices.append(indices[0])
            removed_indices.append(indices[-1])
            removed_ch_names.append(ch_names[indices[0]])
            removed_ch_names.append(ch_names[indices[-1]])
        # remove the first and last channels in each electrode
        if verbose:
            print(f"removing {len(removed_indices)} channels from data")
            print(f"removing channels: {removed_ch_names}")
        data = np.delete(data, removed_indices, axis=0)
        if verbose:
            print("successfully re-referenced data!")
        return data, removed_ch_names, removed_indices

    def _average(data: np.ndarray):
        """average Reference the data to the average of all channels.

        This method is also referred to as the Common Average Reference (CAR) method.
        Other similar methods include the median referencing.

        [TODO]
        """
        print("The method 'average' is not yet implemented. Try another method.")
        return data

    def _median():
        """median Reference the data to the median of all channels.

        Similar methods include Common Average Referencing (CAR), see average method in this Class.

        [TODO]
        """
        print("The method 'median' is not yet implemented. Try another method.")
        return data

    def _get_unique_channels(x: np.array = None) -> np.array:
        """_get_unique_channels Find unique electrodes based on names in a numpy array

        Args:
            x (np.array): Array with electrode names. Default is None.

        Returns:
            unique_arr (np.array): Array with unique electrode names.
        """
        tmp_list = []
        for i in range(len(x)):
            tmp_list.append(x[i].split(' ')[0])
        unique_arr = np.sort(np.unique(tmp_list))
        return unique_arr

    # check arguments
    if isinstance(verbose, bool):
        verbose = verbose
    else:
        raise ValueError(
            f"verbose must be True or False, got {repr(verbose)}")
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(
                f"data must be a 2D array (channels, data), got {data.ndim}D array")
    else:
        raise ValueError(f"data must be a numpy array, got {type(data)}")
    if isinstance(method, str):
        if method == 'monopolar' or method == 'bipolar' or method == 'laplacian' or method == 'average' or method == 'median':
            if method == 'bipolar' or method == 'laplacian':
                if isinstance(ch_names, (list, np.ndarray)):
                    if len(ch_names) != data.shape[0]:
                        raise ValueError(
                            f"ch_names must be the same length as the number of channels in the data. Got {len(ch_names)} but expected {data.shape[0]}")
                elif ch_names is None:
                    raise ValueError(
                        f"ch_names must be provided when method is {repr(method)}, but got {ch_names}")
                else:
                    raise ValueError(
                        f"ch_names must be list, tuple or numpy array, got {type(ch_names)}")
                if method == 'bipolar':
                    if isinstance(direction, str):
                        if direction == 'right' or direction == 'left':
                            pass
                        else:
                            raise ValueError(
                                f"direction must be either 'right' or 'left', got {repr(direction)}")
                    elif direction is None:
                        raise ValueError(
                            f"direction must be provided when method is {repr(method)}, but got {repr(direction)}")
                    else:
                        raise ValueError(
                            f"direction must be a string, got {type(direction)}")
            elif method == 'monopolar':
                if isinstance(ref_channel, int):
                    if ref_channel >= len(data[0]) - 1:
                        raise ValueError(
                            f"ref_channel must be smaller than the number of channels in the data. Got {ref_channel} but expected {data.shape[0]}")
                elif ref_channel is None:
                    raise ValueError(
                        f"ref_channel must be provided when method is {repr(method)}, but got {ref_channel}")
                else:
                    raise ValueError(
                        f"ref_channel must be an integer, got {repr(ref_channel)}")
        else:
            raise ValueError(
                f"method must be either 'monopolar', 'bipolar', 'laplacian', 'average', or 'median', got {repr(method)}")
    elif method is None:
        raise ValueError(
            f"method must be either 'monopolar', 'bipolar', 'laplacian', 'average', or 'median', got {repr(method)}")
    else:
        raise ValueError(
            f"method must be a string, got {repr(method)}")

    # start re-referencing
    if method == 'monopolar':
        data = _monopolar(data=data,
                          ref_channel=ref_channel,
                          verbose=verbose)
    elif method == 'bipolar':
        data, removed_ch_names, removed_indices = _bipolar(data=data,
                                                           ch_names=ch_names,
                                                           direction=direction,
                                                           verbose=verbose)
    elif method == 'laplacian':
        data, removed_ch_names, removed_indices = _laplacian(data=data,
                                                             ch_names=ch_names,
                                                             verbose=verbose)
    elif method == 'average':
        data = _average(data=data)
    elif method == 'median':
        data = _median(data=data)

    if method == 'bipolar' or method == 'laplacian':
        return data, removed_ch_names, removed_indices
    else:
        return data
