import numpy as np


class Rereference():
    """Re-reference the channel data.

    Args:
        Vk (array): Vector with channel data.
        Vref (array): Must be a 1D-array. Used for 'monopolar' re-referencing. Default is None.
        Vk_nb (ndarray): The neighbouring channel or channels data. Used for neighbouring referencing such
                         as 'bipolar' and 'laplacian'. Must be 1D if bipolar method is used and 2D if 
                         laplacian is used. Default is None.
        Vk_new (array): Vector with re-referenced channel data.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def monopolar(self, data: np.array, ref_channel=0):
        """monopolar Reference all channel to a single reference channel.
        
        Referencing all channels to a single reference channel is done by subtracting the reference channel
        from each of the other channels. The reference channel is usually the mastoid channel.
        
        Args:
            data (np.array): 2D array with channel data.
            ref_channel (int): The index to use as reference channel. Default is 0.
            
        Returns:
            new_data (ndarray): Re-referenced data.
        """
        if self.verbose: print("re-referencing method: 'monopolar'")
        if self.verbose: print(f"reference channel: {ref_channel}")
        # get all channels except the reference channel
        selector = [i for i in range(data.shape[0]) if i != ref_channel]
        # subtract the reference channel from all other channels
        data[selector, :] = data[selector, :] - data[ref_channel, :]
        if self.verbose: print("successfully re-referenced data.")
        return data

    def bipolar(self, data: np.array, ch_names: (list, np.array) = None, direction: str = 'right'):
        """Bipolar re-referencing of channel data uses a neighbouring contact as reference.
        
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
        if self.verbose: print("re-referencing method: 'bipolar'")
        if self.verbose: print(f"direction for substraction operation: {direction}")
        if ch_names is not None:
            if len(ch_names) != data.shape[0]:
                raise ValueError(f"ch_names must be the same length as the number of channels in the data. Got {len(ch_names)} but expected {data.shape[0]}")
        else:
            raise ValueError("ch_names must be provided.")  
        # get the unique channels names from ch_names
        print("ch_names: ", ch_names)
        unique_electrodes = self._get_unique(ch_names)
        print("unique_electrodes: ", unique_electrodes)
        # tmp list find indices in list matching electrode name
        tmp_ch_names = np.array([s.split(' ')[0] for s in ch_names])
        print("tmp_ch_names: ", tmp_ch_names)
        
        remove_indices = []
        # iterate over unique electrodes
        for electrode in unique_electrodes:
            print(electrode)
            indices = np.where(tmp_ch_names == electrode)[0]
            print("indices: ", indices)
            # iterate over N-1 contacts for each electrode
            # iterate over all channels - 1
            for i in range(len(indices) - 1):
                if direction == 'right':
                    # subtract the next channel from the current channel
                    data[indices[i], :] = data[indices[i], :] - data[indices[i + 1], :]
                elif direction == 'left':
                    # subtract the previous channel from the current channel
                    data[indices[i + 1], :] = data[indices[i + 1], :] - data[indices[i], :]
            # remove the last channel depending on the direction
            if direction == 'right':
                remove_indices.append(indices[-1])
            elif direction == 'left':
                remove_indices.append(indices[0])
            print("remove_indices: ", remove_indices)
        # remove the first or last contact (channel) in each electrode depending on the direction of the subtraction
        data = np.delete(data, remove_indices, axis=0)
        if self.verbose: print("successfully re-referenced data.")
        return data, remove_indices

    def laplacian(self, data: np.array, ch_names: (list, np.array) = None):
        """Laplacian re-referencing of channel data uses the average of neighbouring contacts as reference.
        
        Args:
            data (np.array): 2D array with channel data.
            ch_names (list, np.array): 1D list or array of channel names. Must be same order as the
                                       channels appear in the data. Default is None.
            
        Returns:
            new_data (ndarray): Re-referenced data with N-2 channels.
        """
        if self.verbose: print("re-referencing method: 'laplacian'")
        if ch_names is not None:
            if len(ch_names) != data.shape[0]:
                raise ValueError(f"ch_names must be the same length as the number of channels in the data. Got {len(ch_names)} but expected {data.shape[0]}")
        else:
            raise ValueError("ch_names must be provided.")
        
        # get the unique channels names from ch_names
        unique_electrodes = self._get_unique(ch_names)
        # tmp list of channel names to find indices in list matching electrode name
        tmp_ch_names = np.array([s.split(' ')[0] for s in ch_names])
        
        remove_indices = []
        # iterate over unique electrodes
        for electrode in unique_electrodes:
            indices = np.where(tmp_ch_names == electrode)[0]
            # iterate over N-2 contacts for each electrode
            for i in range(len(indices) - 2):
                # subtract the mean of the previous and next channel from the current channel
                data[indices[i + 1], :] = data[indices[i + 1], :] - 0.5 * (data[indices[i], :] + data[indices[i + 1], :])
            # remove the first and last channels
            remove_indices.append(indices[0])
            remove_indices.append(indices[-1])
            print("remove_indices: ", remove_indices)
        # remove the first and last channels in each electrode
        data = np.delete(data, remove_indices, axis=0)
        if self.verbose: print("successfully re-referenced data.")
        return data, remove_indices

    def average(self):
        """average Re-reference the data by averaging all channels.

        This method is also referred to as the Common Average Reference (CAR) method.
        Other similar methods include the median referencing.
        """
        
        
    def median(self):
        """median Re-reference the data by substracting the median of all channels from each channel.

        Similar methods include Common Average Referencing (CAR), see average method in this Class.
        """
        
    def _get_unique(self, x: np.array = None) -> np.array:
        """get_unique Find unique electrodes based on names in a numpy array
        
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
    
    
class Data():
    def __init__(self, path_to_filepaths_file, experiment_phase_of_interest, fs, verbose=False):
        self.path_to_filepaths_file = path_to_filepaths_file
        self.experiment_phase_of_interest = experiment_phase_of_interest
        self.fs = fs
        self.verbose = verbose
        self.full = self.ieeg = self.df_exp = self.df_chan = None
        
        # initiate variables
        self.dataone_filtered = None
        self.df_exp = None
        self.df_chan = None
        
        if self.verbose: print("initializing Data object...")
        if self.verbose: print(f"path_to_filepaths_file: {repr(self.path_to_filepaths_file)}")
        if self.verbose: print(f"experiment_phase_of_interest: {repr(self.experiment_phase_of_interest)}")
        if self.verbose: print(f"fs: {repr(self.fs)} Hz")
        
        # get filepaths and single subject ID
        if self.verbose: print("getting filepaths...")
        self.filepaths, self.subject_id = self.get_filepaths()
        if self.verbose: 
            for key, value in self.filepaths.items():
                print(f"{key}: {repr(value)}")
        if self.verbose: print(f"subject ID is {repr(self.subject_id)}")
        
        # get single subject data
        if self.verbose: print(f"loading subject {self.subject_id} data...")
        self.full = self.get_ieegdataone()
        if self.verbose: print("use <Class>.full to access full data (including metadata)")
        
        # create own instance of data of interest
        self.ieeg = self.full['data'][f'ieeg_data{self.experiment_phase_of_interest}']
        if self.verbose: print("use <Class>.ieeg to only access iEEG data")
        
        # create dataframe of metadata
        if self.verbose: print("creating dataframe of experiment metadata...")
        self.df_exp = self.get_experiment_meta()
        
        # create dataframe of channel metadata
        if self.verbose: print("creating dataframe of channel metadata...")
        self.df_chan = self.get_channel_meta()
        
        if self.verbose: print("successfully initialized Data object\n")
        
    def get_filepaths(self):
        filepaths = json.loads(open(self.path_to_filepaths_file).read())
        subject_id = filepaths['single_subject'].split('sub')[-1][0:2]
        return filepaths, subject_id
    
    def get_ieegdataone(self):
        return read_mat(self.filepaths['single_subject'])
    
    def get_experiment_meta(self):
        """Parse subject experiment metadata as Pandas dataframe"""
        df = pd.DataFrame(self.full['data'][f'file{self.experiment_phase_of_interest}'], columns=['x_coordinate', 'y_coordinate', 'Picture Number', 'Reaction Time (RT)'])
        df['Picture Number'] = df['Picture Number'].map(int)
        if self.verbose: print("creating new columns 'Mark for Picture Shown' and 'Mark for Picture Placed'...")
        df['Mark for Picture Shown'] = pd.Series(self.full['data'][f'mark{self.experiment_phase_of_interest}'][0::2]).map(int)
        df['Mark for Picture Placed'] = pd.Series(self.full['data'][f'mark{self.experiment_phase_of_interest}'][1::2]).map(int)
        if self.verbose: print("creating new columns 'Timestamp (s) for Picture Shown' and 'Timestamp (s) for Picture Placed'...")
        df['Timestamp (s) for Picture Shown'] = df['Mark for Picture Shown'].apply(lambda x: x/self.fs)
        df['Timestamp (s) for Picture Placed'] = df['Mark for Picture Placed'].apply(lambda x: x/self.fs)
        if self.verbose: print("creating new column 'Reaction Time (computed)'...")
        df['Reaction Time (computed)'] = df['Timestamp (s) for Picture Placed'] - df['Timestamp (s) for Picture Shown']
        return df
    
    def get_channel_meta(self):
        """Get channel metadata as Pandas dataframe"""
        return pd.read_excel(self.filepaths['single_subject_ieeg_chan'], engine='openpyxl')