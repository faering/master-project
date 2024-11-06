import matplotlib.pyplot as plt
from mne.io.constants import FIFF
import neuropsy as npsy
import os
import numpy as np
import re
import pandas as pd
import pickle
from pymatreader import read_mat
import mne
import math


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

    def __init__(self, path: str, subject_id: str, exp_phase: int, fs: int, verbose: bool = False):
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
            if len(subject_id) == 1:
                subject_id = '0' + subject_id
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
            print("initializing data...")
            print(f"path:\t\t\t{repr(self._path)}")
            print(f"subject ID:\t\t{self._subject_id}")
            print(
                f"experiment phase:\t{self._exp_phase} ({self._get_exp_phase_name()})")
            print(f"sampling frequency:\t{self._fs} Hz")
        # initiate variables
        self.full = self.ieeg = self.df_exp = self.df_chan = self.df_target = None
        self.ch_names = None
        self.raw = None
        # set filepath to load data from
        # note:
        #   needed to load full raw .mat file if data is not saved
        #   and experiment and channel dataframes need to be created
        # self.filepath_matlab_raw = ''.join( [REMOVE] obsoloete since path is now explicitly set in load method
        #     (self._path, "/sub", self._subject_id, ".mat"))
        if self._verbose:
            print("done")

    def __len__(self):
        """__len__ Return the number of channels in the data"""
        if self.ieeg is None:
            return None
        else:
            return self.ieeg.shape[0]

    def _load_matlab_raw(self, path: str):
        try:
            # if path is None or path == '':
            #     load_path = self.filepath_matlab_raw
            # else:
            #     load_path = path
            if self._verbose:
                print(f"loading raw data from {path}...")
            return read_mat(path)
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
                    (load_path, "/sub", self._subject_id + "_ieeg", f"{'_' if postfix != '' else ''}{postfix}", ".pkl"))
                if self._verbose:
                    print(
                        f"loading saved data file {repr(path_ieeg.split('/')[-1])}")
                with open(path_ieeg, 'rb') as f:
                    data_ieeg = pickle.load(f)
            # load raw data
            else:
                if self.full is None:
                    path_raw = ''.join(
                        (load_path, "/sub", self._subject_id, ".mat"))
                    self.full = self._load_matlab_raw(path=path_raw)
                data_ieeg = self.full['data'][f'ieeg_data{self._exp_phase}']
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_ieeg)}. Make sure you have saved the iEEG data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load the raw data instead? ([y]/n): ").strip().lower()
            if arg == 'y' or arg == 'yes' or arg == '':
                if self.full is None:
                    path_raw = ''.join(
                        (load_path, "/sub", self._subject_id, ".mat"))
                    self.full = self._load_matlab_raw(path=path_raw)
                data_ieeg = self.full['data'][f'ieeg_data{self._exp_phase}']
            else:
                data_ieeg = None
        finally:
            return data_ieeg

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
                "Do you want to load the raw data instead? ([y]/n): ").strip().lower()
            if arg == 'y' or arg == 'yes' or arg == '':
                if self.full is None:
                    path_raw = ''.join(
                        (load_path, "/sub", self._subject_id, ".mat"))
                    self.full = self._load_matlab_raw(path=path_raw)
                df = self._get_targets(load_saved=False)
            else:
                df = None
        finally:
            return df

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
                # reaction time is computed
                df = df.drop(columns=['Reaction Time (RT)'])
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
                            f'{str(pic_num)}-{str(i+1)}')
                df['Trial Identifier'] = trial_identifiers_arr
                df['Trial Identifier'] = df['Trial Identifier'].str.decode(
                    'utf-8')
                # create new column for trial error
                if self._verbose:
                    print("creating new column 'Trial Error'")
                if self.df_targets is None:
                    print("targets dataframe not found, loading...")
                    self.df_targets = self._get_targets(load_saved=False)
                euclidean_distances = np.empty(df.shape[0])
                euclidean_distances.fill(np.nan)
                for pic_num in df['Picture Number'].unique():
                    # get indices for picture number
                    idx = df['Picture Number'][df['Picture Number']
                                               == pic_num].index.to_list()
                    for i in idx:
                        p = [df['x_coordinate'].iloc[i],
                             df['y_coordinate'].iloc[i]]
                        q = [self.df_targets['x_coordinate'].iloc[pic_num-1],
                             self.df_targets['y_coordinate'].iloc[pic_num-1]]
                        distance = round(math.dist(p, q), 2)
                        euclidean_distances[i] = distance
                df['Trial Error'] = euclidean_distances
                if self._verbose:
                    print("created experiment dataframe")
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_df)}. Make sure you have saved the experiment data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load the raw data instead? ([y]/n): ").strip().lower()
            if arg == 'y' or arg == 'yes' or arg == '':
                if self.full is None:
                    path_raw = ''.join(
                        (load_path, "/sub", self._subject_id, ".mat"))
                    self.full = self._load_matlab_raw(path=path_raw)
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
                    (load_path, "/sub", self._subject_id + "_chan.xlsx"))
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
                # add channel names to self.ch_names
                self.ch_names = df['name'].to_list()
                if self._verbose:
                    print("created channel dataframe")
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_df)}. Make sure you have saved the channel data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load the raw data instead? ([y]/n): ").strip().lower()
            if arg == 'y' or arg == 'yes' or arg == '':
                if self.full is None:
                    path_raw = ''.join(
                        (load_path, "/sub", self._subject_id, ".mat"))
                    self.full = self._load_matlab_raw(path=path_raw)
                df = self._get_channel_meta(load_saved=False)
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

    def _get_ied_df(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """_get_ied_df Create a new DataFrame with information on IEDs in the data.

        Create a dataframe with time points of IEDs, channel numbers, channel names, and time in seconds.
        """
        try:
            # set load path
            if path is None or path == '':
                load_path = self._path
            else:
                load_path = path
            # load saved data
            if load_saved:
                path_df = ''.join(
                    (path, "/sub", self._subject_id, "_ied", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self._verbose:
                    print(
                        f"loading saved IED data from {repr(path_df)}")
                df_ied = pd.read_csv(path_df)
            # create from initial ied file
            else:
                # load ied data from initial file, no extra columns
                path_df = ''.join(
                    (path, "/ied", "/sub", self._subject_id, "_ied.csv"))
                df_ied = pd.read_csv(path_df)

                # get the channel name from df_chan and add it to df_ied (if channel is in df_ied)
                series_chan_name = pd.Series(
                    [None] * len(df_ied), name='chan name')
                for i in range(len(self.df_chan)):
                    if i+1 in df_ied['chan']:
                        idx = np.where(df_ied['chan'] == i+1)[0]
                        series_chan_name[idx] = self.df_chan['name'][i]

                # add new column with channel names to df_ied
                df_ied['chan name'] = series_chan_name

                # rename time column to 'time (s)'
                df_ied.rename(columns={'time': 'time (s)'}, inplace=True)

                # add new column with time points, converted from 'time' column which is in seconds, to df_ied
                df_ied['time point'] = np.floor(
                    df_ied['time (s)'] * self._fs).astype(int)
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_df)}. Make sure you have saved the IED data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load from the initial IED data instead? ([y]/n): ").strip().lower()
            if arg == 'y' or arg == 'yes' or arg == '':
                arg = input("Path to initial IED data directory: ").strip()
                path_df = ''.join((arg, "/sub", self._subject_id, "_ied.csv"))
                df_ied = self._get_ied_df(
                    path=path_df, load_saved=False, postfix=None)
            else:
                df_ied = None
        finally:
            return df_ied

    def load(self, path: str = None, load_saved: bool = False, postfix: str = '', load_ieeg: bool = True, load_exp: bool = True, load_chan: bool = True, load_targets: bool = True, load_ied: bool = True, postfix_ieeg: str = None, postfix_exp: str = None, postfix_chan: str = None, postfix_targets: str = None, postfix_ied: str = None, verbose: bool = None):
        """Load data as defined by path, subject_id, and exp_phase"""
        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        elif verbose is None:
            pass
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
        if not (isinstance(postfix_ieeg, str) or postfix_ieeg is None):
            raise ValueError(
                f"postfix_ieeg must be a string, got {type(postfix_ieeg)}")
        if not (isinstance(postfix_exp, str) or postfix_exp is None):
            raise ValueError(
                f"postfix_exp must be a string, got {type(postfix_exp)}")
        if not (isinstance(postfix_chan, str) or postfix_chan is None):
            raise ValueError(
                f"postfix_chan must be a string, got {type(postfix_chan)}")
        if not (isinstance(postfix_targets, str) or postfix_targets is None):
            raise ValueError(
                f"postfix_targets must be a string, got {type(postfix_targets)}")
        if not (isinstance(postfix_ied, str) or postfix_ied is None):
            raise ValueError(
                f"postfix_ied must be a string, got {type(postfix_ied)}")
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
        if not isinstance(load_ied, bool):
            raise ValueError(
                f"load_ied must be True or False, got {type(load_ied)}")
        if not (load_ieeg or load_exp or load_chan or load_targets or load_ied):
            raise ValueError(
                "one of load_ieeg, load_exp, load_chan, load_targets, or load_ied must be True to load data")

        # start loading data
        if self._verbose:
            print("loading data...")
        if path is None or path == '':
            load_path = self._path
            if self._verbose:
                print(
                    f"no path provided. Using path {repr(load_path)} from initialization")
        else:
            load_path = path
            if not os.path.isdir(os.path.abspath(load_path)):
                raise ValueError(
                    f"path {os.path.abspath(load_path)} does not exist, check path and try again!")
        # get subject iEEG data
        if load_ieeg:
            if self._verbose:
                print("loading iEEG data...")
            if postfix_ieeg is not None:
                pf = postfix_ieeg
            else:
                pf = postfix
            if self._verbose:
                print("trying to load file: ", ''.join(
                    (load_path, "/sub", self._subject_id, "_ieeg", f"{'_' if pf != '' else ''}{pf}", ".pkl")))
            try:
                self.ieeg = self._get_ieeg(
                    path=load_path, load_saved=load_saved, postfix=pf)
                if self._verbose:
                    if self.ieeg is not None:
                        print(
                            "use <DataHandler>.ieeg to access intracranial EEG data")
                    if self.ieeg is None:
                        print("iEEG data not loaded")
                    print("done")
            except FileNotFoundError as fe:
                print(
                    f"Could not find file with iEEG data! {fe}")
                self.ieeg = None
        # load picture targets metadata
        if load_targets:
            if self._verbose:
                print("loading picture target data...")
            if postfix_targets is not None:
                pf = postfix_targets
            else:
                pf = postfix
            if self._verbose:
                print("trying to load file: ", ''.join(
                    (load_path, "/sub", self._subject_id, "_targets", f"{'_' if pf != '' else ''}{pf}", ".csv")))
            try:
                self.df_targets = self._get_targets(
                    path=load_path, load_saved=load_saved, postfix=pf)
                if self._verbose:
                    if self.df_targets is not None:
                        print(
                            "use <DataHandler>.df_targets to access targets dataframe")
                    if self.df_targets is None:
                        print("target data not loaded")
                    print("done")
            except FileNotFoundError as fe:
                print(
                    f"Could not find file with target data! {fe}")
                self.df_targets = None
        # load experiment metadata
        if load_exp:
            if self._verbose:
                print("loading experiment metadata...")
            if postfix_exp is not None:
                pf = postfix_exp
            else:
                pf = postfix
            if self._verbose:
                print("trying to load file: ", ''.join(
                    (load_path, "/sub", self._subject_id, "_exp", f"{'_' if pf != '' else ''}{pf}", ".csv")))
            try:
                self.df_exp = self._get_experiment_meta(
                    path=load_path, load_saved=load_saved, postfix=pf)
                if self._verbose:
                    if self.df_exp is not None:
                        print(
                            "use <DataHandler>.df_exp to access experiment dataframe")
                    if self.df_exp is None:
                        print("experiment data not loaded")
                    print("done")

                if self.df_exp is not None:
                    if 'Trial Category' not in self.df_exp.columns.to_list():
                        try:
                            filename = ''.join(
                                (load_path, "/sub", self._subject_id, "_trial_labels.csv"))
                            if self._verbose:
                                print(
                                    f"Trying to load trial category from location {filename}...")
                            series = pd.read_csv(filename)
                            self.df_exp = pd.concat(
                                [self.df_exp, series], axis=1)
                            if self._verbose:
                                print("Loaded trial category from saved file")
                        except FileNotFoundError as fe:
                            if self._verbose:
                                print("Could not find file with trial category!")
                            print(fe)
            except FileNotFoundError as fe:
                print(
                    f"Could not find file with experiment metadata! {fe}")
                self.df_exp = None
        # load channel metadata
        if load_chan:
            if self._verbose:
                print("loading channel metadata...")
            if postfix_chan is not None:
                pf = postfix_chan
            else:
                pf = postfix
            if self._verbose:
                print("trying to load file: ", ''.join(
                    (load_path, "/sub", self._subject_id, "_chan", f"{'_' if pf != '' else ''}{pf}", ".csv")))
            try:
                self.df_chan = self._get_channel_meta(
                    path=load_path, load_saved=load_saved, postfix=pf)
                if self._verbose:
                    if self.df_chan is not None:
                        print("use <DataHandler>.df_chan to access channel dataframe")
                    if self.df_chan is None:
                        print("channel data not loaded")
                    print("done")
            except FileNotFoundError as fe:
                print(
                    f"Could not find file with channel metadata! {fe}")
                self.df_chan = None
        # load ied data
        if load_ied:
            if self._verbose:
                print("loading IED data...")
            if postfix_ied is not None:
                pf = postfix_ied
            else:
                pf = postfix
            try:
                self.df_ied = self._get_ied_df(
                    path=load_path, load_saved=load_saved, postfix=pf)
                if self._verbose:
                    if self.df_ied is not None:
                        print("use <DataHandler>.df_ied to access IED dataframe")
                    if self.df_ied is None:
                        print("IED data not loaded")
                    print("done")
            except FileNotFoundError as fe:
                print(
                    f"Could not find file with IED data! {fe}")
                self.df_ied = None
        # clean up
        if self.full is not None:
            if self._verbose:
                print(
                    f"cleaning up to free memory...")
            del self.full
            self.full = None
        if self._verbose:
            if self.ieeg is None and self.df_exp is None and self.df_chan is None and self.df_targets is None and self.df_ied is None:
                print("no data loaded!")
            else:
                print("finished loading data!")

    def save(self, path: str = None, postfix: str = '', save_ieeg: bool = True, save_exp: bool = True, save_chan: bool = True, save_targets: bool = True, save_ied: bool = True, postfix_ieeg: str = None, postfix_exp: str = None, postfix_chan: str = None, postfix_targets: str = None, postfix_ied: str = None, verbose: bool = None):
        """Save iEEG data, experiment metadata, channel metadata and target data as csv files"""

        # check arguments
        if isinstance(verbose, bool):
            self._verbose = verbose
        elif verbose is None:
            pass
        else:
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")
        if not (isinstance(path, str) or path is None):
            raise ValueError(
                f"path must be a string or None, got {type(path)}")
        if not isinstance(postfix, str):
            raise ValueError(f"postfix must be a string, got {type(postfix)}")
        if not (isinstance(postfix_ieeg, str) or postfix_ieeg is None):
            raise ValueError(
                f"postfix_ieeg must be a string, got {type(postfix_ieeg)}")
        if not (isinstance(postfix_exp, str) or postfix_exp is None):
            raise ValueError(
                f"postfix_exp must be a string, got {type(postfix_exp)}")
        if not (isinstance(postfix_chan, str) or postfix_chan is None):
            raise ValueError(
                f"postfix_chan must be a string, got {type(postfix_chan)}")
        if not (isinstance(postfix_targets, str) or postfix_targets is None):
            raise ValueError(
                f"postfix_targets must be a string, got {type(postfix_targets)}")
        if not (isinstance(postfix_ied, str) or postfix_ied is None):
            raise ValueError(
                f"postfix_ied must be a string, got {type(postfix_ied)}")
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
        if not isinstance(save_ied, bool):
            raise ValueError(
                f"save_ied must be True or False, got {type(save_ied)}")
        if not (save_ieeg or save_exp or save_chan or save_targets or save_ied):
            raise ValueError(
                "one of save_ieeg, save_exp, save_chan, save_targets. or save_ied must be True to save data")

        # start saving data
        if self._verbose:
            print("saving data...")
        if path is None or path == '':
            save_path = self._path
            if self._verbose:
                print(
                    f"no path provided. Using path {repr(save_path)} from initialization")
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
                if postfix_ieeg is not None:
                    pf = postfix_ieeg
                else:
                    pf = postfix
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_ieeg", f"{'_' if pf != '' else ''}{pf}", ".pkl"))
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
        # save target data
        if save_targets:
            if not self.df_targets is None:
                if postfix_targets is not None:
                    pf = postfix_targets
                else:
                    pf = postfix
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_targets", f"{'_' if pf != '' else ''}{pf}", ".csv"))
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
        # save experiment dataframe
        if save_exp:
            if not self.df_exp is None:
                if postfix_exp is not None:
                    pf = postfix_exp
                else:
                    pf = postfix
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_exp", f"{'_' if pf != '' else ''}{pf}", ".csv"))
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
                if postfix_chan is not None:
                    pf = postfix_chan
                else:
                    pf = postfix
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_chan", f"{'_' if pf != '' else ''}{pf}", ".csv"))
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
        # save ied dataframe
        if save_ied:
            if not self.df_ied is None:
                if postfix_ied is not None:
                    pf = postfix_ied
                else:
                    pf = postfix
                filename = ''.join(
                    (save_path, "/sub", self._subject_id, "_ied", f"{'_' if pf != '' else ''}{pf}", ".csv"))
                if self._verbose:
                    print(f"saving IED data as {repr(filename)}")
                    if os.path.isfile(filename):
                        print(
                            f"file {repr(filename)} already exists, overwriting...")
                self.df_ied.to_csv(filename, index=False)
                if self._verbose:
                    print(f"done")
            else:
                if self._verbose:
                    print(f"<DataHandler>.df_ied is None, skipping...")
        # finish
        if self._verbose:
            if self.ieeg is None and self.df_exp is None and self.df_chan is None and self.df_targets is None:
                print("no data saved, load data first!")
            else:
                print("finished saving data!")

    def plot(self, use_qt: bool = False):

        if self.ieeg is None:
            print(f"No data loaded, use <DataHandler>.load() to load data.")
        else:
            import matplotlib
            from neuropsy.viz.plot import plot_raw

            events = np.array(
                [self.df_exp['Mark for Picture Shown'].astype(int).to_numpy(),
                 self.df_exp['Mark for Picture Placed'].astype(int).to_numpy()])

            event_ids = {
                1: "picture shown",
                2: "picture placed"
            }

            shading_onset = self.df_exp['Timestamp (s) for Picture Shown'].to_numpy(
            )
            shading_duration = self.df_exp['Timestamp (s) for Picture Placed'].to_numpy(
            ) - self.df_exp['Timestamp (s) for Picture Shown'].to_numpy()

            plot_raw(data=self.ieeg,
                     events=events,
                     event_ids=event_ids,
                     ch_names=self.df_chan['name'].to_list(),
                     fs=self._fs,
                     shading=True,
                     shading_onset=shading_onset,
                     shading_duration=shading_duration,
                     shading_desc=self.df_exp['Trial Identifier'].to_numpy(),
                     use_qt=use_qt,
                     title=f"Subject {self._subject_id} - {self._get_exp_phase_name()}",
                     start=0,
                     duration=9)

            if use_qt:
                matplotlib.pyplot.show(block=True)
                matplotlib.use('Agg')
            else:
                matplotlib.pyplot.show()

    def copy(self):
        """Create a copy of the DataHandler object"""
        import copy
        return copy.deepcopy(self)

    def select_channels(self, ch_names: (list, tuple, np.ndarray) = None, ch_index: (list, tuple, np.ndarray) = None, inplace: bool = False, verbose: bool = False):
        """select_channels Cut the data to the selected channels.

        Cut the data to the selected channels based on the list provided
        in ch_names or ch_index. A good practice is to copy the DataHandler object
        before using this method. See the method copy() for more information.

        Args:
            ch_names (list, tuple, np.ndarray, optional): List of channel names to keep. Defaults to None.
            ch_index (list, tuple, np.ndarray, optional): List of channel indices to keep. Defaults to None.
            inplace (bool, optional): Delete unselected channels inplace in both self.ieeg and self.df_chan. Defaults to False.
            verbose (bool, optional): Print process. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            if inplace: None
            data (DataHandler): Copy of DataHandler object with selected channels.
        """
        if isinstance(ch_names, (list, tuple, np.ndarray)) and isinstance(ch_index, (list, tuple, np.ndarray)):
            print(
                "only one of ch_names or ch_index should be provided, defaulting to ch_names")
            ch_index = None
        if ch_names is None and ch_index is None:
            raise ValueError(
                "one of ch_names or ch_index must be provided")
        if ch_names is not None and isinstance(ch_names, (list, tuple, np.ndarray)):
            if np.array(ch_names).ndim != 1:
                raise ValueError(
                    f"ch_names must be 1D, got {np.array(ch_names).ndim}D")
            if not all(isinstance(x, str) for x in ch_names):
                wrong_types = [type(x)
                               for x in ch_names if not isinstance(x, str)]
                raise ValueError(
                    f"ch_names must be a list of strings, got wrong types {wrong_types}")
            if not all(x in self.df_chan['name'].to_list() for x in ch_names):
                wrong_names = [
                    x for x in ch_names if x not in self.df_chan['name'].to_list()]
                raise ValueError(
                    f"ch_names must be present in existing channel names, got wrong names {wrong_names} expected one or more of {self.df_chan['name'].to_list()}")
        if ch_index is not None and isinstance(ch_index, (list, tuple, np.ndarray)):
            if np.array(ch_index).ndim != 1:
                raise ValueError(
                    f"ch_index must be 1D, got {np.array(ch_index).ndim}D")
            if not all(isinstance(x, int) for x in ch_index):
                wrong_types = [type(x)
                               for x in ch_index if not isinstance(x, int)]
                raise ValueError(
                    f"ch_index must be a list of integers, got wrong types {wrong_types}")
            if not all(x in np.arange(self.__len__()) for x in ch_index):
                wrong_indices = [
                    x for x in ch_index if x not in np.arange(self.__len__())]
                raise ValueError(
                    f"ch_index must be present in existing channel indices, got wrong indices {wrong_indices}")
        if not isinstance(inplace, bool):
            raise ValueError(
                f"inplace must be True or False, got {type(inplace)}")
        if not isinstance(verbose, bool):
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")

        # get indices of channels to remove
        if ch_names is not None:
            idx = self.df_chan.loc[~self.df_chan['name'].isin(
                ch_names)].index
        elif ch_index is not None:
            idx = self.df_chan.loc[~self.df_chan.index.isin(
                ch_index)].index

        if inplace:
            if verbose:
                print("removing channels inplace...")
                print(
                    f"removing channels {self.df_chan.loc[idx]['name'].to_list()}")
                print(
                    f"keeping channels {self.df_chan.loc[~self.df_chan.index.isin(idx)]['name'].to_list()}")
            self.ieeg = np.delete(self.ieeg, idx, axis=0)
            self.df_chan = self.df_chan.drop(idx).reset_index(drop=True)
            self.ch_names = self.df_chan['name'].to_list()
        else:
            if verbose:
                print("returning copy of DataHandler object with selected channels...")
                print(
                    f"removing channels {self.df_chan.loc[idx]['name'].to_list()}")
                print(
                    f"keeping channels {self.df_chan.loc[~self.df_chan.index.isin(idx)]['name'].to_list()}")
            data = self.copy()  # deepcopy
            data.ieeg = np.delete(data.ieeg, idx, axis=0)
            data.df_chan = data.df_chan.drop(idx).reset_index(drop=True)
            self.ch_names = self.df_chan['name'].to_list()
            return data

    def create_raw(self, return_raw: bool = False):
        # MNE info object will be used to create MNE Raw object
        info = mne.create_info(
            ch_names=self.df_chan['name'].to_list(),
            ch_types=['seeg'] * len(self.df_chan['name'].to_list()),
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
        onset_arr = self.df_exp['Timestamp (s) for Picture Shown'].to_numpy(
        )
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

        if return_raw:
            return raw
        else:
            self.raw = raw

    def read_raw(self, fname: str, preload: bool = False, return_raw: bool = False):
        """read_raw Read a raw file using MNE.

        Args:
            fname (str): Path to the raw file.
            preload (bool, optional): Preload the data into memory. Defaults to False.

        Returns:
            raw (mne.io.BaseRaw): MNE raw object.
        """
        raw = mne.io.read_raw_fif(fname, preload=preload)
        if return_raw:
            return raw
        else:
            self.raw = raw

    def create_epochs(self,
                      raw: mne.io.BaseRaw = None,
                      tmin: int | float = -1,
                      tmax: int | float = 1,
                      markers: list | tuple | np.ndarray = None,
                      event_id: int = 10,
                      event_desc: str = 'event'):
        """create_epochs Create epochs from Raw object.

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
        if raw == None:
            if self.raw == None:
                self.raw = self.create_raw()
                raw = self.raw
            else:
                raw = self.raw

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

    # [REMOVE] this will be done in a separate script "electrode_coordinates.ipynb"
    def _add_montage(self, subjects_dir: str):
        """_add_montage Add a montage for the electrode contacts to the raw object in place.

        Args:
            subjects_dir (str): The path to the folder containing all subjects' freesurfer output files.
        """
        if self.raw == None:
            self.raw = self.create_raw()

        subject_dir_name = f'sub{self._subject_id}'

        # get fiducials for subject in MNI_TAL space using subject's Freesurfer MRI brian
        fiducials = mne.coreg.get_mni_fiducials(
            subject=subject_dir_name, subjects_dir=subjects_dir)

        for d in fiducials:
            if d["ident"] == FIFF.FIFFV_POINT_NASION:
                nas = d['r']  # * 1000
            elif d["ident"] == FIFF.FIFFV_POINT_LPA:
                lpa = d['r']  # * 1000
            elif d["ident"] == FIFF.FIFFV_POINT_RPA:
                rpa = d['r']  # * 1000

        # get channel locations from channel meta information dataframe
        ch_locations = {}
        for ch in self.df_chan['name']:
            ch_locations[ch] = tuple(self.df_chan.loc[self.df_chan['name'] == ch, [
                'loc_1', 'loc_2', 'loc_3']].values[0])

        # location values are in millimetres, but mne expects the values in meters
        def convert_to_meters(values):
            modified_values = np.zeros(len(values))
            for i, v in enumerate(values):
                modified_values[i] = v / 1000
            return modified_values

        ch_locations_in_meters = {key: convert_to_meters(
            values) for key, values in ch_locations.items()}

        # create electrode locations montage
        montage = mne.channels.make_dig_montage(
            ch_pos=ch_locations_in_meters, nasion=nas, lpa=lpa, rpa=rpa, hsp=None, hpi=None, coord_frame="unknown")
        # apply the montage to the raw data, mne will transform the coordinates to "head" space (under the hood)
        self.raw.set_montage(montage)

        # retrieve the montage from the raw data
        montage = self.raw.get_montage()

        # transform electrode locations from mne's "head" to "mri" space
        head_mri_t = mne.coreg.estimate_head_mri_t(
            subject_dir_name, subjects_dir)
        montage.apply_trans(head_mri_t)

        # load our Talairach transform and apply it
        mri_mni_t = mne.read_talxfm(subject_dir_name, subjects_dir)
        montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)

        # for fsaverage, "mri" and "mni_tal" are equivalent and, since
        # we want to plot in fsaverage "mri" space, we need to use an
        # identity transform to equate these coordinate frames
        montage.apply_trans(mne.transforms.Transform(
            fro="mni_tal", to="mri", trans=np.eye(4)))
        self.raw.set_montage(montage)
        return montage

    # [REMOVE] this will be done in a separate script "electrode_labelling.ipynb"
    def _get_montage_volume_labels(self, montage, subject, subjects_dir: str = None, aseg: str = "auto", dist: (float | int) = .2, fname: str = None):
        """_get_montage_volume_labels Get regions of interest near channels from a Freesurfer parcellation.

        This is applicable for channels inside the brain (intracranial electrodes).

        Args:
            montage (mne.channels.DigMontage): The montage object containing the channel positions.
            subject (str): The subject folder name (eg. "sub03").
            subjects_dir (str): The path to the folder containing all subjects' freesurfer output files.
            aseg (str): The name of the aseg file in the subject's mri folder. Defaults to "auto".
            dist (float): The distance in mm to use for identifying regions of interest. Defaults to 2.
            fname (str): The name of the freesurfer lookup table file. Defaults to None.

        Raises:
            ValueError: Distance given in the wrong range.
            RuntimeError: Coordinate frame not supported.

        Returns:
            labels (dict): The regions of interest labels within dist of each channel.
            colors (dict): The lookup table colors for the labels.
        """
        from mne._freesurfer import _get_aseg, read_freesurfer_lut
        from collections import OrderedDict

        _VOXELS_MAX = 1000  # define constant to avoid runtime issues

        if dist < 0 or dist > 10:
            raise ValueError("`dist` must be between 0 and 10")

        aseg, aseg_data = _get_aseg(aseg, subject, subjects_dir)

        # read freesurfer lookup table
        lut, fs_colors = read_freesurfer_lut(fname=fname)
        label_lut = {v: k for k, v in lut.items()}

        # assert that all the values in the aseg are in the labels
        assert all([idx in label_lut for idx in np.unique(aseg_data)])

        # get transform to surface RAS for distance units instead of voxels
        vox2ras_tkr = aseg.header.get_vox2ras_tkr()

        ch_dict = montage.get_positions()
        if ch_dict["coord_frame"] != "mri":
            raise RuntimeError(
                "Coordinate frame not supported, expected "
                '"mri", got ' + str(ch_dict["coord_frame"])
            )
        ch_coords = np.array(list(ch_dict["ch_pos"].values()))

        # convert to freesurfer voxel space
        ch_coords = mne.surface.apply_trans(
            np.linalg.inv(aseg.header.get_vox2ras_tkr()), ch_coords * 1000
        )
        labels = OrderedDict()
        for ch_name, ch_coord in zip(montage.ch_names, ch_coords):
            if np.isnan(ch_coord).any():
                labels[ch_name] = list()
            else:
                voxels = mne.surface._voxel_neighbors(
                    ch_coord,
                    aseg_data,
                    dist=dist,
                    vox2ras_tkr=vox2ras_tkr,
                    voxels_max=_VOXELS_MAX,
                )
                label_idxs = set([aseg_data[tuple(voxel)].astype(int)
                                 for voxel in voxels])
                labels[ch_name] = [label_lut[idx] for idx in label_idxs]

        all_labels = set([label for val in labels.values() for label in val])
        colors = {label: tuple(
            fs_colors[label][:3] / 255) + (1.0,) for label in all_labels}
        return labels, colors

    # [REMOVE] this will be done in a separate script "electrode_labelling.ipynb"
    def add_vep_atlas_labels_to_df_chan(self, subjects_dir: str = None, aseg: str = "auto", dist: (float | int) = .2, fname: str = None, return_labels: bool = False):

        if self.df_chan is None:
            print("No channel data found, use <DataHandler>.load() to load data.")
            return
        if self.raw == None:
            self.raw = self.create_raw()

        # apply montage and add it to the raw object
        montage = self._add_montage(subjects_dir)

        # get labels and colors for each channel
        labels, colors = self._get_montage_volume_labels(
            montage=montage,
            subject='sub' + self._subject_id,
            subjects_dir=subjects_dir,
            aseg=aseg,
            dist=dist,
            fname=fname)

        # get label strings from list of labels per contact, if more than one label has been found for a contact
        # keep all the labels, otherwise just keep the single label.
        labels = {k: v[0] if len(v) == 1 else '|'.join(v)
                  for k, v in labels.items()}

        # add labels to channel dataframe
        self.df_chan['VEP_atlas'] = self.df_chan['name'].map(labels)

        if return_labels:
            return labels, colors

    def average(self, method: str = 'epochs'):
        """average Compute average of ieeg data

        Args:
            method (str, optional): Compute across epochs or across entire self.ieeg object. Defaults to 'epochs'.
        """
        if self.ieeg is None:
            print("No ieeg data found, use <DataHandler>.load() to load data.")
            return
        else:
            if method == 'epochs':
                if self.epochs:
                    # [FIXME] perhaps wrong to compute across all epochs
                    averages = self.epochs.average()
                else:
                    print(
                        "No epochs found, use <DataHandler>.create_epochs() to create epochs.")
                    return
            else:
                averages = self.ieeg.mean(axis=0)
            return averages
