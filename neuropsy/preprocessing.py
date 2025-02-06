import pandas as pd
import numpy as np
import numpy.typing as npt
from tabulate import tabulate
import copy


def clean_outliers(df: pd.DataFrame, cols: list | tuple | npt.ArrayLike, num_std: int = 3, verbose: bool = False):
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


def clean_nan(df: pd.DataFrame, cols: list | tuple | npt.ArrayLike, verbose: bool = False):
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


def reference(data: npt.ArrayLike, method: str, ch_names: list | tuple | npt.ArrayLike = None, ref_channel: int = None, direction: str = None, verbose: bool = False):
    """reference Reference channels in one electrode using one of the available methods.

    Available methods are:
    - 'monopolar': Reference all channels to a single reference channel.
    - 'bipolar': Reference channels using neighbouring channel as reference.
    - 'laplacian': Reference channels using the average of neighbouring channels as reference.
    - 'average': Reference the data to the average of all channels. (Not yet implemented)
    - 'median': Reference the data to the median of all channels. (Not yet implemented)

    Args:
        data (np.ndarray): 2D array with channel data.
        method (str): The method to use for re-referencing. Options are 'monopolar', 'bipolar', 'laplacian', 'average', or 'median'.
        ch_names (list, tuple, np.array): List of channel names for one electrode. Default is None.
        ref_channel (int): The index to use as reference channel when method='monopolar'. Default is None.
        direction (str): The direction of the bipolar referencing. If 'right' the n+1 channel will substracted from the n channel. If 'left' the n channel will be substracted from the n+1 channel. Default is None.
        verbose (bool): Print verbose to stdout. Default is False.

    Returns:
        data (np.ndarray): Re-referenced data with data.shape[0] - removed channels. 
                           Remove the channels that cannot be referenced if method is 'bipolar' (elec_n_channels - 1) or 'laplacian' (elec_n_channels - 2)).
        ch_names_referenced (list): List of channel names that were re-referenced. None if no channels were re-referenced.
        ch_names_to_remove (list): List of channel names that could not be re-referenced (bipolar will always result in rank N-1, laplacian N-2).
        ch_names_not_re_referenced (list): ch_names if not possible to re-reference. None if re-referencing is applied.
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
        # if isinstance(ref_channel, int):
        #     if ref_channel >= len(data[0]) - 1:
        #         raise ValueError(
        #             f"ref_channel must be smaller than the number of channels in the data. Got {ref_channel} but expected {data.shape[0]}")
        # else:
        #     raise ValueError(
        #         f"ref_channel must be an integer, got {repr(ref_channel)}")
        # if isinstance(data, np.ndarray):
        #     if data.ndim != 2:
        #         raise ValueError(
        #             f"data must be a 2D array (channels, data), got {data.ndim}D array")
        # else:
        #     raise ValueError(f"data must be a numpy array, got {type(data)}")

        # # start re-referencing
        # if verbose:
        #     print("applying re-referencing method: 'monopolar'")
        # if verbose:
        #     print(f"reference channel: {ref_channel}")
        # # get all channels except the reference channel
        # selector = [i for i in range(data.shape[0]) if i != ref_channel]
        # # subtract the reference channel from all other channels
        # data[selector, :] = data[selector, :] - data[ref_channel, :]
        # if verbose:
        #     print("successfully re-referenced data!")
        print("The method 'monopolar' is not yet implemented. Try another method.")
        return data

    def _bipolar(data: npt.ArrayLike, ch_names: list | npt.ArrayLike, direction: str, verbose: bool):
        """_bipolar Bipolar Reference channels using neighbouring channel as reference.

        Args:
            data (np.array): 2D array with channel data.
            ch_names (list, np.array): 1D list or array of channel names. Must be same order as the
                                       channels appear in the data. Default is None.
            direction (str): The direction of the bipolar referencing. If 'right' the n+1 channel will
                             substracted from the n channel. If 'left' the n channel will be 
                             substracted from the n+1 channel.

        Returns:
            data (ndarray): Re-referenced data (all channels included, also not re-referenced).
            removed_ch_names (list): List of channel names that could not be re-referenced.
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

        # check if there are enough channels to apply re-referencing method
        if data.shape[0] < 2:
            data_ref = data
            ch_names_referenced = None
            ch_names_to_remove = None
            ch_names_not_re_referenced = ch_names
        else:
            data_ref = copy.deepcopy(data)
            # start re-referencing
            ch_names_to_remove = []
            # iterate over N-1 channels
            for i in range(data.shape[0] - 1):
                if direction == 'right':
                    # subtract the n+1 channel from the n channel
                    data_ref[i, :] = data[i, :] - data[i + 1, :]
                elif direction == 'left':
                    # subtract the n channel from the n+1 channel
                    data_ref[i + 1, :] = data[i + 1, :] - data[i, :]
            # assign return variables
            # return channel that cannot be re-referenced depending on the direction
            if direction == 'right':
                ch_names_to_remove.append(ch_names[-1])
            elif direction == 'left':
                ch_names_to_remove.append(ch_names[0])
            ch_names_referenced = [
                ch for ch in ch_names if ch not in ch_names_to_remove]
            ch_names_not_re_referenced = None
        return data_ref, ch_names_referenced, ch_names_to_remove, ch_names_not_re_referenced

    def _laplacian(data: npt.ArrayLike, ch_names: list | npt.ArrayLike, verbose: bool):
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

        # check if there are enough channels to apply re-referencing method
        if data.shape[0] < 3:
            data_ref = data
            ch_names_referenced = None
            ch_names_to_remove = None
            ch_names_not_re_referenced = ch_names
        else:
            data_ref = copy.deepcopy(data)
            # start re-referencing
            ch_names_to_remove = []
            # iterate over N-2 channels
            for i in range(data.shape[0] - 2):
                # for readability
                n = i + 1
                # subtract the mean of the previous and next channel from the current channel
                data_ref[n, :] = data[n, :] - \
                    0.5 * (data[n - 1, :] + data[n + 1, :])
            # return channels that cannot be re-referenced (each electrode/channel-group has 2 channels that cannot be re-referenced)
            ch_names_to_remove.append(ch_names[0])
            ch_names_to_remove.append(ch_names[-1])
            ch_names_referenced = [
                ch for ch in ch_names if ch not in ch_names_to_remove]
            ch_names_not_re_referenced = None
        return data_ref, ch_names_referenced, ch_names_to_remove, ch_names_not_re_referenced

    def _average(data: npt.ArrayLike):
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

    # def _get_ch_groups_in_electrode(ch_names):
    #     """_get_ch_groups_in_electrode Get channel groups based on an electrode's name (eg. "A'").

    #     One electrode might have multiple groups of channels, this function will return the groups
    #     based on the channel names.

    #     Args:
    #         ch_names (list, np.array): List or array of channel names for one electrode.

    #     Returns:
    #         elec_ch_groups (list): Nested list of channel names in each group, each sublist represents one group.

    #     ´´´
    #     # Example:
    #     ch_names = ['A 01', 'A 02', 'A 03', 'A 10', 'A 11', 'A 12']
    #     elec_ch_groups = _get_ch_groups_in_electrode(ch_names)
    #     print(elec_ch_groups)

    #     # Output:
    #     [['A 01', 'A 02', 'A 03'], ['A 10', 'A 11', 'A 12']]
    #     ´´´
    #     """
    #     elec_ch_groups = []

    #     # get the difference between the channel numbers
    #     diff = np.diff([int(ch.split(' ')[1]) for ch in ch_names])
    #     # get the indices where the difference is greater than 1
    #     idx = np.where(diff > 1)[0]
    #     # if there are multiple groups
    #     if len(idx) > 0:
    #         # get the first group
    #         elec_ch_groups.append(ch_names[:idx[0]+1])
    #         # get the rest of the groups
    #         for i in range(1, len(idx)):
    #             elec_ch_groups.append(ch_names[idx[i-1]+1:idx[i]+1])
    #         # get the last group
    #         elec_ch_groups.append(ch_names[idx[-1]+1:])
    #     # if there is only one group
    #     else:
    #         elec_ch_groups.append(ch_names)
    #     return elec_ch_groups

    def _apply_method(method, ch_data, ch_names, ref_channel, direction, verbose):
        """_apply_method Apply the selected method for re-referencing.

        Args:
            method (str): The method to use for re-referencing.

        Returns:
            data_ref (np.ndarray): Re-referenced channel data.
            ch_names_to_remove (list): List of channel names that could not be re-referenced.
        """
        ch_names_referenced = None
        ch_names_to_remove = None
        ch_names_not_re_referenced = None

        if method == 'monopolar':
            # check arguments
            data_ref = _monopolar(data=ch_data,
                                  ref_channel=ref_channel,
                                  verbose=verbose)
        elif method == 'bipolar':
            data_ref, ch_names_referenced, ch_names_to_remove, ch_names_not_re_referenced = _bipolar(data=ch_data,
                                                                                                     ch_names=ch_names,
                                                                                                     direction=direction,
                                                                                                     verbose=verbose)
        elif method == 'laplacian':
            data_ref, ch_names_referenced, ch_names_to_remove, ch_names_not_re_referenced = _laplacian(data=ch_data,
                                                                                                       ch_names=ch_names,
                                                                                                       verbose=verbose)
        elif method == 'average':
            data_ref = _average(data=ch_data)
        elif method == 'median':
            data_ref = _median(data=ch_data)
        return data_ref, ch_names_referenced, ch_names_to_remove, ch_names_not_re_referenced

    ############################## MAIN FUNCTION ##############################
    # Check arguments
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

    elec_name = ch_names[0].split(' ')[0]
    # check if all provided channels are from the same electrode
    assert all([ch.split(' ')[0] == elec_name for ch in ch_names]
               ), 'All channels must be from the same electrode!'

    # apply re-referencing method
    data_ref, ch_names_referenced, ch_names_to_remove, ch_names_not_re_referenced = _apply_method(method=method,
                                                                                                  ch_data=data,
                                                                                                  ch_names=ch_names,
                                                                                                  ref_channel=ref_channel,
                                                                                                  direction=direction,
                                                                                                  verbose=verbose)

    return data_ref, ch_names_referenced, ch_names_to_remove, ch_names_not_re_referenced
