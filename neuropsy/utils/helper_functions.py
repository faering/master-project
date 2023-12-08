import os
import numpy as np
import re
import pandas as pd
import pickle
from pymatreader import read_mat


class DataHandler():
    """DataHandler Load or save the data from a single subject.

    Load all the data including experiment and iEEG channel metadata into a data object.

    Args:
        exp_phase (int): The experiment phase of interest. Must be integer.
        fs (int): The sampling frequency of the EEG signal data.
        path (str): Path to the single subject data file. Default is None.
        path_json (str): Path to the JSON file with all filepaths. Default is None.
        load_saved (str): Path to a saved dataframe file. Default is None.
        verbose (bool): If True print out more information. Default is False.

    Returns:
        data (Data): Data object with all the data and metadata.
    """

    def __init__(
            self,
            path: str,
            subject_id: str,
            exp_phase: int,
            fs: int,
            verbose: bool = False):
        """__init__ _summary_

        _extended_summary_

        Args:
            path (str): _description_
            subject_id (str): _description_
            exp_phase (int): _description_
            fs (int): _description_
            verbose (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """

        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")
        if isinstance(path, str):
            self._path = path
            if not os.path.isdir(os.path.abspath(self._path)):
                raise ValueError(
                    f"path {os.path.abspath(self._path)} does not exist, check path and try again!")
        else:
            raise ValueError(f"path must be a string, got {type(path)}")
        if isinstance(subject_id, str):
            self._subject_id = subject_id
        else:
            raise ValueError(
                f"subject_id must be a string, got {type(subject_id)}")
        if isinstance(exp_phase, int):
            self._exp_phase = exp_phase
            if self._exp_phase < 1 or self._exp_phase > 4:
                raise ValueError(
                    f"exp_phase must be 1, 2, 3 or 4, got {self._exp_phase}")
        else:
            raise ValueError(
                f"exp_phase must be an integer, got {type(exp_phase)}")
        if isinstance(fs, int):
            self._fs = fs
        else:
            raise ValueError(f"fs must be an integer, got {type(fs)}")

        # start initialization
        if self._verbose:
            print("initializing...")
            print(f"path:\t\t\t{repr(self._path)}")
            print(f"subject ID:\t\t{self._subject_id}")
            print(
                f"experiment phase:\t{self._exp_phase} ({self._get_exp_phase_name()})")
            print(f"sampling frequency:\t{self._fs} Hz")
        # initiate variables
        self.full = self.ieeg = self.df_exp = self.df_chan = self.df_target = None
        # set filepath to load data from
        # note:
        #   needed to load full raw .mat file if data is not saved
        #   and experiment and channel dataframes need to be created
        self.filepath_matlab_raw = ''.join(
            (self._path, "/sub", self._subject_id, ".mat"))
        if self._verbose:
            print("done")

    def __len__(self):
        """__len__ Return the number of channels in the data"""
        if self.ieeg is None:
            return None
        else:
            return self.ieeg.shape[0]

    def _load_matlab_raw(self):
        try:
            if self._verbose:
                print("loading raw data...")
            return read_mat(self.filepath_matlab_raw)
        except FileNotFoundError:
            print(
                f"provided path does not contain any raw .mat file for subject {self._subject_id}, please provide a correct path.")

    def _get_ieeg(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """Load intracranial EEG data"""
        try:
            # set load path
            if path is None or path == '':
                load_path = self._path
            else:
                load_path = path
            # load saved data
            if load_saved:
                path_ieeg = ''.join(
                    (load_path, "/sub", self._subject_id + "_ieeg", f"{'_' if postfix != '' else ''}{postfix}", ".data"))
                if self._verbose:
                    print(
                        f"loading saved data file {repr(path_ieeg.split('/')[-1])}")
                with open(path_ieeg, 'rb') as f:
                    data_ieeg = pickle.load(f)
            # load raw data
            else:
                if self._verbose:
                    print(
                        f"loading raw iEEG data from {repr(self.filepath_matlab_raw)}")
                if self.full is None:
                    self.full = self._load_matlab_raw()
                data_ieeg = self.full['data'][f'ieeg_data{self._exp_phase}']
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_ieeg)}. Make sure you have saved the iEEG data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load the raw data instead? ([y]/n): ")
            if arg == 'y' or arg == 'Y' or arg == '':
                if self.full is None:
                    self.full = self._load_matlab_raw()
                data_ieeg = self.full['data'][f'ieeg_data{self._exp_phase}']
            else:
                data_ieeg = None
        finally:
            return data_ieeg

    def _get_experiment_meta(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """Parse subject experiment data as Pandas dataframe"""
        try:
            # set load path
            if path is None or path == '':
                load_path = self._path
            else:
                load_path = path
            # load saved experiment data
            if load_saved:
                path_df = ''.join(
                    (load_path, "/sub", self._subject_id + "_exp", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self._verbose:
                    print(
                        f"loading saved experiment data file {repr(path_df)}")
                df = pd.read_csv(path_df)
            # create from raw data
            else:
                if self._verbose:
                    print("creating dataframe with experiment data from raw data")
                # cast data to dataframe
                df = pd.DataFrame(self.full['data'][f'file{self._exp_phase}'], columns=[
                    'x_coordinate', 'y_coordinate', 'Picture Number', 'Reaction Time (RT)'])
                df['Picture Number'] = df['Picture Number'].astype(int)
                # create new columns
                # add subject ID
                if self._verbose:
                    print("creating new column 'Subject ID'")
                df['Subject ID'] = [self._subject_id] * df.shape[0]
                # create columns for picture marks
                if self._verbose:
                    print(
                        "creating new columns 'Mark for Picture Shown' and 'Mark for Picture Placed'")
                df['Mark for Picture Shown'] = pd.Series(
                    self.full['data'][f'mark{self._exp_phase}'][0::2]).astype(int)
                df['Mark for Picture Placed'] = pd.Series(
                    self.full['data'][f'mark{self._exp_phase}'][1::2]).astype(int)
                # create columns for picture marks in seconds
                if self._verbose:
                    print(
                        "creating new columns 'Timestamp (s) for Picture Shown' and 'Timestamp (s) for Picture Placed'")
                df['Timestamp (s) for Picture Shown'] = df['Mark for Picture Shown'].apply(
                    lambda x: x/self._fs)
                df['Timestamp (s) for Picture Placed'] = df['Mark for Picture Placed'].apply(
                    lambda x: x/self._fs)
                # create column for reaction time
                if self._verbose:
                    print("creating new column 'Reaction Time (computed)'")
                df['Reaction Time (computed)'] = df['Timestamp (s) for Picture Placed'] - \
                    df['Timestamp (s) for Picture Shown']
                # create column holding trial identifier
                if self._verbose:
                    print("creating new column 'Trial Identifier'")
                # important to set datatype correctly, array-protocol string allows whole string to be set at element level
                trial_identifiers_arr = np.zeros(df.shape[0], dtype='a5')
                for pic_num in df['Picture Number'].unique():
                    idx = df['Picture Number'][df['Picture Number']
                                               == pic_num].index.to_list()
                    for i in range(len(idx)):
                        trial_identifiers_arr[idx[i]] = str(
                            f'{str(i+1)}-{str(pic_num)}')
                df['Trial Identifier'] = trial_identifiers_arr
                df['Trial Identifier'] = df['Trial Identifier'].str.decode(
                    'utf-8')
                if self._verbose:
                    print("created experiment dataframe")
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_df)}. Make sure you have saved the experiment data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load the raw data instead? ([y]/n): ")
            if arg == 'y' or arg == '':
                if self.full is None:
                    self.full = self._load_matlab_raw()
                df = self._get_experiment_meta(load_saved=False)
            else:
                df = None
        finally:
            return df

    def _get_channel_meta(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """Get channel data as Pandas dataframe"""
        try:
            # set load path
            if path is None or path == '':
                load_path = self._path
            else:
                load_path = path
            # load saved data
            if load_saved:
                path_df = ''.join(
                    (load_path, "/sub", self._subject_id + "_chan", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self._verbose:
                    print(
                        f"loading saved channel data from {repr(path_df)}")
                    df = pd.read_csv(path_df)
            # create from raw data
            else:
                if self._verbose:
                    print("creating dataframe with channel data from raw data")
                filepath = ''.join(
                    (self._path, "/sub", self._subject_id + "_chan.xlsx"))
                df = pd.read_excel(filepath, engine='openpyxl')
                # create True/False column for left hippocampal electrode
                if self._verbose:
                    print("creating column 'HC left'")
                regex = re.compile(r'^(?=.*hippo)(?=.*left).*$', re.IGNORECASE)
                idx = np.where(df['DK_ROI'].str.contains(regex))[0]
                df['HC left'] = False
                df.loc[idx, 'HC left'] = True
                # create True/False column for right hippocampal electrode
                if self._verbose:
                    print("creating column 'HC right'")
                regex = re.compile(
                    r'^(?=.*hippo)(?=.*right).*$', re.IGNORECASE)
                idx = np.where(df['DK_ROI'].str.contains(regex))[0]
                df['HC right'] = False
                df.loc[idx, 'HC right'] = True
                if self._verbose:
                    print("created channel dataframe")
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_df)}. Make sure you have saved the channel data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load the raw data instead? ([y]/n): ")
            if arg == 'y' or arg == '':
                if self.full is None:
                    self.full = self._load_matlab_raw()
                df = self._get_channel_meta(load_saved=False)
            else:
                df = None
        finally:
            return df

    def _get_targets(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """Get target locations for each picture as Pandas dataframe"""
        try:
            # set load path
            if path is None or path == '':
                load_path = self._path
            else:
                load_path = path
            # load saved data
            if load_saved:
                path_df = ''.join(
                    (load_path, "/sub", self._subject_id + "_targets", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self._verbose:
                    print(
                        f"loading saved targets data from {repr(path_df)}")
                    df = pd.read_csv(path_df)
            # create from raw data
            else:
                if self._verbose:
                    print("creating dataframe with target data from raw data")
                df = pd.DataFrame(self.full['data'][f'file1'], columns=[
                    'x_coordinate', 'y_coordinate', 'Unnamed: 1', 'Unnamed: 2']).drop(columns=['Unnamed: 1', 'Unnamed: 2'])
                df['picture number'] = np.arange(1, 51)
                if self._verbose:
                    print("created targets dataframe")
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_df)}. Make sure you have saved the targets data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load the raw data instead? ([y]/n): ")
            if arg == 'y' or arg == '':
                if self.full is None:
                    self.full = self._load_matlab_raw()
                df = self._get_targets(load_saved=False)
            else:
                df = None
        finally:
            return df

    def _get_exp_phase_name(self) -> str:
        """_get_exp_phase_name Get the name of the experiment phase

        Returns:
            str: Name of the experiment phase.
        """
        if self._exp_phase == 1:
            return 'preview'
        elif self._exp_phase == 2:
            return 'learning'
        elif self._exp_phase == 3:
            return 'pre-sleep test'
        elif self._exp_phase == 4:
            return 'post-sleep test'
        else:
            raise ValueError(
                f"exp_phase must be 1, 2, 3 or 4, got {self._exp_phase}")

    def load(self, path: str, load_saved: bool = False, postfix: str = '', load_ieeg: bool = True, load_exp: bool = True, load_chan: bool = True, load_targets: bool = True,   verbose: bool = False):
        """Load data as defined by path, subject_id, and exp_phase"""
        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")
        if not (isinstance(path, str) or path is None):
            raise ValueError(
                f"path must be a string or None, got {type(path)}")
        if not isinstance(load_saved, bool):
            raise ValueError(
                f"load_saved must be True or False, got {type(load_saved)}")
        if not (isinstance(postfix, str) or postfix is None):
            raise ValueError(f"postfix must be a string, got {type(postfix)}")
        elif postfix == None:
            postfix = ''
        if not isinstance(load_ieeg, bool):
            raise ValueError(
                f"load_ieeg must be True or False, got {type(load_ieeg)}")
        if not isinstance(load_exp, bool):
            raise ValueError(
                f"load_exp must be True or False, got {type(load_exp)}")
        if not isinstance(load_chan, bool):
            raise ValueError(
                f"load_chan must be True or False, got {type(load_chan)}")
        if not isinstance(load_targets, bool):
            raise ValueError(
                f"load_targets must be True or False, got {type(load_targets)}")
        if not (load_ieeg or load_exp or load_chan or load_targets):
            raise ValueError(
                "one of load_ieeg, load_exp, load_chan, or load_targets must be True to load data")

        # start loading data
        if self._verbose:
            print("loading data...")
        if path is None or path == '':
            load_path = self._path
            if self._verbose:
                print(
                    f"no save path provided. Using path {repr(load_path)} from initialization")
        else:
            load_path = path
            if not os.path.isdir(os.path.abspath(load_path)):
                raise ValueError(
                    f"path {os.path.abspath(load_path)} does not exist, check path and try again!")
        # get subject iEEG data
        if load_ieeg:
            if self._verbose:
                print("loading iEEG data...")
            self.ieeg = self._get_ieeg(
                path=load_path, load_saved=load_saved, postfix=postfix)
            if self._verbose:
                if self.ieeg is not None:
                    print("use <DataHandler>.ieeg to access intracranial EEG data")
                if self.ieeg is None:
                    print("iEEG data not loaded")
                print("done")
        # load experiment metadata
        if load_exp:
            if self._verbose:
                print("loading experiment metadata...")
            self.df_exp = self._get_experiment_meta(
                path=load_path, load_saved=load_saved, postfix=postfix)
            if self._verbose:
                if self.df_exp is not None:
                    print("use <DataHandler>.df_exp to access experiment dataframe")
                if self.df_exp is None:
                    print("experiment data not loaded")
                print("done")
        # load channel metadata
        if load_chan:
            if self._verbose:
                print("loading channel metadata...")
            self.df_chan = self._get_channel_meta(
                path=load_path, load_saved=load_saved, postfix=postfix)
            if self._verbose:
                if self.df_chan is not None:
                    print("use <DataHandler>.df_chan to access channel dataframe")
                if self.df_chan is None:
                    print("channel data not loaded")
                print("done")
        # load picture targets metadata
        if load_targets:
            if self._verbose:
                print("loading picture target data...")
            self.df_targets = self._get_targets(
                path=load_path, load_saved=load_saved, postfix=postfix)
            if self._verbose:
                if self.df_targets is not None:
                    print("use <DataHandler>.df_targets to access targets dataframe")
                if self.df_targets is None:
                    print("target data not loaded")
                print("done")
        # clean up
        if self.full is not None:
            if self._verbose:
                print(
                    f"cleaning up to free memory...")
            del self.full
            self.full = None
        if self._verbose:
            if self.ieeg is None and self.df_exp is None and self.df_chan is None and self.df_targets is None:
                print("no data loaded!")
            else:
                print("loaded data successfully!")

    def save(self, path: str = None, postfix: str = '', save_ieeg: bool = True, save_exp: bool = True, save_chan: bool = True, save_targets: bool = True, verbose: bool = False):
        """Save iEEG data, experiment metadata, channel metadata and target data as csv files"""
        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")
        if not (isinstance(path, str) or path is None):
            raise ValueError(
                f"path must be a string or None, got {type(path)}")
        if not isinstance(postfix, str):
            raise ValueError(f"postfix must be a string, got {type(postfix)}")
        if not isinstance(save_ieeg, bool):
            raise ValueError(
                f"save_ieeg must be True or False, got {type(save_ieeg)}")
        if not isinstance(save_exp, bool):
            raise ValueError(
                f"save_exp must be True or False, got {type(save_exp)}")
        if not isinstance(save_chan, bool):
            raise ValueError(
                f"save_chan must be True or False, got {type(save_chan)}")
        if not isinstance(save_targets, bool):
            raise ValueError(
                f"save_targets must be True or False, got {type(save_targets)}")
        if not (save_ieeg or save_exp or save_chan or save_targets):
            raise ValueError(
                "one of save_ieeg, save_exp, save_chan, or save_targets must be True to save data")

        # start saving data
        if self._verbose:
            print("saving data...")
        if path is None or path == '':
            save_path = self._path
            if self._verbose:
                print(
                    f"no save path provided. Using path {repr(save_path)} from initialization")
        else:
            save_path = path
            if not os.path.isdir(os.path.abspath(save_path)):
                if self._verbose:
                    print(
                        f"path {os.path.abspath(save_path)} does not exist, creating...")
                os.mkdir(os.path.abspath(save_path))
        # save iEEG data
        if save_ieeg:
            if not self.ieeg is None:
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_ieeg", f"{'_' if postfix != '' else ''}{postfix}", ".data"))
                if self._verbose:
                    print(f"saving iEEG data as {repr(filename)}")
                    if os.path.isfile(filename):
                        print(
                            f"file {repr(filename)} already exists, overwriting...")
                with open(filename, 'wb') as f:
                    pickle.dump(self.ieeg, f)
                if self._verbose:
                    print(f"done")
            else:
                if self._verbose:
                    print(f"<DataHandler>.ieeg is None, skipping...")
        # save experiment dataframe
        if save_exp:
            if not self.df_exp is None:
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_exp", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self._verbose:
                    print(f"saving experiment data as {repr(filename)}")
                    if os.path.isfile(filename):
                        print(
                            f"file {repr(filename)} already exists, overwriting...")
                self.df_exp.to_csv(filename, index=False)
                if self._verbose:
                    print(f"done")
            else:
                if self._verbose:
                    print(f"<DataHandler>.df_exp is None, skipping...")
        # save channel dataframe
        if save_chan:
            if not self.df_chan is None:
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_chan", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self._verbose:
                    print(f"saving channel data as {repr(filename)}")
                    if os.path.isfile(filename):
                        print(
                            f"file {repr(filename)} already exists, overwriting...")
                self.df_chan.to_csv(filename, index=False)
                if self._verbose:
                    print(f"done")
            else:
                if self._verbose:
                    print(f"<DataHandler>.df_chan is None, skipping...")
        # save target data
        if save_targets:
            if not self.df_targets is None:
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_targets", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self._verbose:
                    print(f"saving target data as {repr(filename)}")
                    if os.path.isfile(filename):
                        print(
                            f"file {repr(filename)} already exists, overwriting...")
                self.df_targets.to_csv(filename, index=False)
                if self._verbose:
                    print(f"done")
            else:
                if self._verbose:
                    print(f"<DataHandler>.df_targets is None, skipping...")
        if self._verbose:
            if self.ieeg is None and self.df_exp is None and self.df_chan is None and self.df_targets is None:
                print("no data saved, load data first!")
            else:
                print("saved data successfully!")

    def plot(self, use_qt: bool = False):

        if use_qt:
            import matplotlib
            matplotlib.use('Qt5Agg')

        import mne
        import matplotlib.pyplot as plt

        # MNE info object will be used to create MNE Raw object
        info = mne.create_info(
            ch_names=self.df_chan['name'].to_list(),
            ch_types=['eeg'] * len(self.df_chan['name'].to_list()),
            sfreq=self._fs
        )

        # create events array (n_events, 3) with (sample, signal_value_preceding_sample, event_id)
        events = np.array(
            [np.ravel(np.array([self.df_exp['Mark for Picture Shown'].astype(int).to_numpy(), self.df_exp['Mark for Picture Placed'].astype(int).to_numpy()]).T),
             np.zeros(len(self.df_exp) * 2, dtype=int),
             np.array([1, 2] * len(self.df_exp), dtype=int)]
        ).T

        # create MNE Raw object
        raw = mne.io.RawArray(self.ieeg, info)

        # create mapping between event_id to event description
        event_dict = {
            1: "picture shown",
            2: "picture placed"
        }

        # create annotations from events
        annotations_from_events = mne.annotations_from_events(
            events=events,
            event_desc=event_dict,
            sfreq=raw.info["sfreq"],
            orig_time=raw.info["meas_date"],
        )

        # Create annotations for onset, duration and description arrays for trial shading
        onset_arr = self.df_exp['Timestamp (s) for Picture Shown'].to_numpy()
        duration_arr = self.df_exp['Timestamp (s) for Picture Placed'].to_numpy(
        ) - onset_arr
        description_arr = self.df_exp['Trial Identifier'].to_numpy()
        annotations_trial_shading = mne.Annotations(
            onset=onset_arr,
            duration=duration_arr,
            description=description_arr
        )
        annotations = mne.Annotations.__add__(
            annotations_from_events, annotations_trial_shading)

        # Add annotations to raw object
        raw.set_annotations(annotations)

        # plot continues data with annotations
        if self.ieeg.shape[0] > 20:
            n_channels = 20
        else:
            n_channels = self.ieeg.shape[0]
        raw.plot(n_channels=n_channels,
                 scalings='auto',
                 show=True,
                 title=f"Subject {self._subject_id} - Experiment Phase {self._exp_phase}",
                 start=9,
                 duration=9
                 )
        plt.show()


class Preprocess():

    def __init__(self, data: DataHandler, fs: int = None, verbose: bool = False) -> None:
        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")
        if isinstance(data, DataHandler):
            self._data = data
            self._subject_id = self._data._subject_id
            self._exp_phase = self._data._exp_phase
            self.ieeg = self._data.ieeg
            self.df_exp = self._data.df_exp
            self.df_chan = self._data.df_chan
            self.df_targets = self._data.df_targets
        if isinstance(self.ieeg, np.ndarray):
            if self.ieeg.ndim != 2:
                raise ValueError(
                    f"data.ieeg must be a 2D array, got {self.ieeg.ndim}D array")
        else:
            raise ValueError(
                f"data.ieeg must be a numpy array, got {type(self.ieeg)}")
        if not isinstance(self.df_exp, pd.DataFrame):
            raise ValueError(
                f"data.df_exp must be a pandas DataFrame, got {type(self.df_exp)}")
        if isinstance(self.df_chan, pd.DataFrame):
            if self.df_chan.shape[0] != self.ieeg.shape[0]:
                raise ValueError(
                    f"data.df_chan must have the same number of rows as the number of channels in the data. Got {self.df_chan.shape[0]} but expected {self.ieeg.shape[0]}")
        else:
            raise ValueError(
                f"data.df_chan must be a pandas DataFrame, got {type(self.df_chan)}")
        if not isinstance(self.df_targets, pd.DataFrame) or self.df_targets is None:
            raise ValueError(
                f"data.df_targets must be a pandas DataFrame or None, got {type(self.df_targets)}")
        if isinstance(fs, int) or fs is None:
            if fs is not None and fs != data._fs:
                raise ValueError(
                    f"fs must be the same as the sampling frequency of the data. Got {fs} but expected {data._fs}")
            elif fs is None:
                self._fs = data._fs
            else:
                self._fs = fs
        else:
            raise ValueError(f"fs must be an integer or None, got {type(fs)}")

        # start initialization
        if self._verbose:
            print("initializing...")
            print(f"subject ID:\t\t{self._subject_id}")
            print(f"sampling frequency:\t{self._fs} Hz")
            print(f"number of channels:\t{self.ieeg.shape[0]}")
        # initiate variables
        # [TODO]

        if self._verbose:
            print("done")

    def clean(self, df: pd.DataFrame, cols: (list, tuple, np.ndarray) = ['Reaction Time (RT)', 'Reaction Time (computed)'], num_std: int = 3, inplace: bool = True, verbose: bool = False):
        """clean Remove outliers in trials.

        Args:
            df (pd.DataFrame): Dataframe with data to clean.
            cols (list | tuple | np.array, optional): _description_. Defaults to ['Reaction Time (RT)', 'Reaction Time (computed)'].
            num_std (int, optional): _description_. Defaults to 3.
            inplace (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """        """clean Remove outliers.

        [TODO]
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
                if col_name != 'Outcome':
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
            if self._verbose:
                from tabulate import tabulate
                print(f"Outliers (using {num_std} Standard Deviation)")
                table = tabulate(data, headers=['Outlier (Previously)', 'Outliers', 'Count',
                                                'Column', 'Lower Limit', 'Upper Limit'], tablefmt='rst', numalign='right')
                print(table)
            return indices

        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")
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
                f"cols must be a list or numpy array, got {type(cols)}")
        if isinstance(num_std, int):
            if num_std <= 0:
                raise ValueError(
                    f"num_std must be greater than 0, got {num_std}")
        else:
            raise ValueError(
                f"num_std must be an integer, got {type(num_std)}")
        if not isinstance(inplace, bool):
            raise ValueError(
                f"inplace must be True or False, got {type(inplace)}")

        # start cleaning
        idx = _get_outlier_indices(
            df=df,
            cols=cols,
            num_std=num_std
        )
        if inplace:
            if self._verbose:
                print(f"dropping indices inplace {idx}")
            df.drop(idx, inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            if self._verbose:
                print(f"returning new dataframe without indices {idx}")
            return df.drop(idx, inplace=False).reset_index(drop=True)

    def remove_bad_channels(self):
        """remove_bad_channels Remove bad channels based on a threshold.

        [TODO]
        """
        pass

    def remove_ied_artifacts(self):
        """remove_ied_artifacts Remove interictal epileptic discharge artifacts (IEDs).

        [TODO]
        """
        pass

    def filter(self, in_signal: np.ndarray, filter_dict: dict = dict("order": 4, "cutoff": 500, "fs": 1000, "filter_type": 'high', "apply_filter": False, "show_bodeplot": True, "show_impulse_response": True, "len_impulse": 1000)):
        """filter Filter the data.

        [TODO]
        """

    def get_epochs(self, tmin, tmax):
        """get_epochs Get epochs from the data.

        [TODO]
        """
        pass

    class Reference():
        """Reference the channel data.

        [TODO]
        """
        pass


class Rereferencer():
    """Re-reference the channel data.

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
        """Laplacian re-referencing of channel data uses the average of neighbouring contacts as reference.

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
