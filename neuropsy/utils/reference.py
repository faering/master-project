import numpy as np


class Referencer():
    """Referencer Reference the channel data using an available method.

    Args:
        Vk (array): Vector with channel data.
        Vref (array): Must be a 1D-array. Used for 'monopolar' re-referencing. Default is None.
        Vk_nb (ndarray): The neighbouring channel or channels data. Used for neighbouring referencing such
                         as 'bipolar' and 'laplacian'. Must be 1D if bipolar method is used and 2D if 
                         laplacian is used. Default is None.
        Vk_new (array): Vector with re-referenced channel data.
    """

    def __init__(self, verbose: bool = False):
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")

    def monopolar(self, data: np.array, ref_channel: int = 0, verbose: bool = False):
        """monopolar Reference all channel to a single reference channel.

        Referencing all channels to a single reference channel is done by subtracting the reference channel
        from each of the other channels. The reference channel is usually the mastoid channel.

        Args:
            data (np.array): 2D array with channel data.
            ref_channel (int): The index to use as reference channel. Default is 0.

        Returns:
            new_data (ndarray): Re-referenced data.
        """

        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {repr(verbose)}")
        if isinstance(ref_channel, int):
            if ref_channel >= data.shape[0]:
                raise ValueError(
                    f"ref_channel must be smaller than the number of channels in the data. Got {ref_channel} but expected {data.shape[0]}")
        else:
            raise ValueError(
                f"ref_channel must be an integer, got {repr(ref_channel)}")
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(
                    f"data must be a 2D array, got {data.ndim}D array")
        else:
            raise ValueError(f"data must be a numpy array, got {type(data)}")

        # start re-referencing
        if self._verbose:
            print("applying re-referencing method: 'monopolar'")
        if self._verbose:
            print(f"reference channel: {ref_channel}")
        # get all channels except the reference channel
        selector = [i for i in range(data.shape[0]) if i != ref_channel]
        # subtract the reference channel from all other channels
        data[selector, :] = data[selector, :] - data[ref_channel, :]
        if self._verbose:
            print("successfully re-referenced data!")
        return data

    def bipolar(self, data: np.array, ch_names: (list, np.array) = None, direction: str = 'right', verbose: bool = False):
        """Bipolar Reference channels using neighbouring channel as reference.

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
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {repr(verbose)}")
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
                    f"data must be a 2D array, got {data.ndim}D array")
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
        if self._verbose:
            print("applying re-referencing method: 'bipolar'")
        # get the unique channels names from ch_names
        unique_electrodes = self._get_unique_channels(ch_names)
        if self._verbose:
            print(
                f"found {len(unique_electrodes)} unique electrodes and {len(ch_names)} channels")
        # tmp list find indices in list matching electrode name
        tmp_ch_names = np.array([ch.split(' ')[0] for ch in ch_names])

        # iterate over unique electrodes
        remove_indices = []
        remove_ch_names = []
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
                remove_indices.append(indices[-1])
                remove_ch_names.append(ch_names[indices[-1]])
            elif direction == 'left':
                remove_indices.append(indices[0])
                remove_ch_names.append(ch_names[indices[0]])
        # remove the first or last contact (channel) in each electrode depending on the direction of the subtraction
        if self._verbose:
            print(f"removing {len(remove_indices)} channels from data")
            print(f"removing channels: {remove_ch_names}")
        data = np.delete(data, remove_indices, axis=0)
        if self._verbose:
            print("successfully re-referenced data!")
        return data, remove_indices

    def laplacian(self, data: np.ndarray, ch_names: (list, np.array) = None, verbose: bool = False):
        """Laplacian Reference channels using the average of neighbouring channels as reference.

        Args:
            data (np.ndarray): 2D array with channel data.
            ch_names (list, np.array): 1D list or array of channel names. Must be same order as the
                                       channels appear in the data. Default is None.

        Returns:
            new_data (ndarray): Re-referenced data with N-2 channels.
            remove_indices (list): List of indices that could not be re-referenced.
        """
        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {repr(verbose)}")
        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(
                    f"data must be a 2D array, got {data.ndim}D array")
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
        if self._verbose:
            print("applying re-referencing method: 'laplacian'")
        # get the unique channels names from ch_names
        unique_electrodes = self._get_unique_channels(ch_names)
        if self._verbose:
            print(
                f"found {len(unique_electrodes)} unique electrodes and {len(ch_names)} channels")
        # tmp list of channel names to find indices in list matching electrode name
        tmp_ch_names = np.array([ch.split(' ')[0] for ch in ch_names])

        # iterate over unique electrodes
        remove_indices = []
        remove_ch_names = []
        for electrode in unique_electrodes:
            indices = np.where(tmp_ch_names == electrode)[0]
            # iterate over N-2 contacts for each electrode
            for i in range(len(indices) - 2):
                # subtract the mean of the previous and next channel from the current channel
                data[indices[i + 1], :] = data[indices[i + 1], :] - \
                    0.5 * (data[indices[i], :] + data[indices[i + 1], :])
            # remove the first and last channels
            remove_indices.append(indices[0])
            remove_indices.append(indices[-1])
            remove_ch_names.append(ch_names[indices[0]])
            remove_ch_names.append(ch_names[indices[-1]])
        # remove the first and last channels in each electrode
        if self._verbose:
            print(f"removing {len(remove_indices)} channels from data")
            print(f"removing channels: {remove_ch_names}")
        data = np.delete(data, remove_indices, axis=0)
        if self._verbose:
            print("successfully re-referenced data!")
        return data, remove_indices

    def average(self):
        """average Re-reference the data by averaging all channels.

        This method is also referred to as the Common Average Reference (CAR) method.
        Other similar methods include the median referencing.

        [TODO]
        """

    def median(self):
        """median Re-reference the data by substracting the median of all channels from each channel.

        Similar methods include Common Average Referencing (CAR), see average method in this Class.

        [TODO]
        """

    def _get_unique_channels(self, x: np.array = None) -> np.array:
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
