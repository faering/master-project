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
            self.verbose = verbose
        else:
            raise ValueError(
                f"verbose must be True or False, got {type(verbose)}")
        if isinstance(path, str):
            self.path = path
            if not os.path.isdir(os.path.abspath(self.path)):
                raise ValueError(
                    f"path {os.path.abspath(self.path)} does not exist, check path and try again!")
        else:
            raise ValueError(f"path must be a string, got {type(path)}")
        if isinstance(subject_id, str):
            self.subject_id = subject_id
        else:
            raise ValueError(
                f"subject_id must be a string, got {type(subject_id)}")
        if isinstance(exp_phase, int):
            self.exp_phase = exp_phase
            if self.exp_phase < 1 or self.exp_phase > 4:
                raise ValueError(
                    f"exp_phase must be 1, 2, 3 or 4, got {self.exp_phase}")
        else:
            raise ValueError(
                f"exp_phase must be an integer, got {type(exp_phase)}")
        if isinstance(fs, int):
            self.fs = fs
        else:
            raise ValueError(f"fs must be an integer, got {type(fs)}")

        # start initialization
        if self.verbose:
            print("initializing...")
            print(f"path:\t\t\t{repr(self.path)}")
            print(f"subject ID:\t\t{self.subject_id}")
            print(
                f"experiment phase:\t{self.exp_phase} ({self._get_exp_phase_name()})")
            print(f"sampling frequency:\t{self.fs} Hz")
        # initiate variables
        self.full = self.ieeg = self.df_exp = self.df_chan = self.df_target = None
        # set filepath to load data from
        # note:
        #   needed to load full raw .mat file if data is not saved
        #   and experiment and channel dataframes need to be created
        self.filepath_matlab_raw = ''.join(
            (self.path, "/sub", self.subject_id, ".mat"))
        if self.verbose:
            print("done")

    def __len__(self):
        """__len__ Return the number of channels in the data"""
        if self.ieeg is None:
            return None
        else:
            return self.ieeg.shape[0]

    def _load_matlab_raw(self):
        try:
            if self.verbose:
                print("loading raw data...")
            return read_mat(self.filepath_matlab_raw)
        except FileNotFoundError:
            print(
                f"provided path does not contain any raw .mat file for subject {self.subject_id}, please provide a correct path.")

    def _get_ieeg(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """Load intracranial EEG data"""
        try:
            # set load path
            if path is None or path == '':
                load_path = self.path
            else:
                load_path = path
            # load saved data
            if load_saved:
                path_ieeg = ''.join(
                    (load_path, "/sub", self.subject_id + "_ieeg", f"{'_' if postfix != '' else ''}{postfix}", ".data"))
                if self.verbose:
                    print(
                        f"loading saved data file {repr(path_ieeg.split('/')[-1])}")
                with open(path_ieeg, 'rb') as f:
                    data_ieeg = pickle.load(f)
            # load raw data
            else:
                if self.verbose:
                    print(
                        f"loading raw iEEG data from {repr(self.filepath_matlab_raw)}")
                if self.full is None:
                    self.full = self._load_matlab_raw()
                data_ieeg = self.full['data'][f'ieeg_data{self.exp_phase}']
        except FileNotFoundError:
            print(
                f"Could not find file {repr(path_ieeg)}. Make sure you have saved the iEEG data first and the postfix is set correctly.")
            arg = input(
                "Do you want to load the raw data instead? ([y]/n): ")
            if arg == 'y' or arg == 'Y' or arg == '':
                if self.full is None:
                    self.full = self._load_matlab_raw()
                data_ieeg = self.full['data'][f'ieeg_data{self.exp_phase}']
            else:
                data_ieeg = None
        finally:
            return data_ieeg

        """
        if load_saved:
            path_ieeg = ''.join(
                (self.path, "/sub", self.subject_id + "_ieeg", f"{'_' if postfix != '' else ''}{postfix}", ".data"))
            if self.verbose:
                print(
                    f"loading saved data file {repr(path_ieeg.split('/')[-1])}")
            try:
                with open(path_ieeg, 'rb') as f:
                    data_ieeg = pickle.load(f)
            except FileNotFoundError:
                print(
                    f"Could not find file {repr(path_ieeg)}. Make sure you have saved the iEEG data first and the postfix is set correctly.")
                arg = input(
                    "Do you want to load the raw data instead? ([y]/n): ")
                if arg == 'y' or arg == '':
                    if self.full is None:
                        self.full = self._load_matlab_raw()
                    data_ieeg = self.full['data'][f'ieeg_data{self.exp_phase}']
                else:
                    data_ieeg = None
            finally:
                if data_ieeg is not None:
                    if self.verbose:
                        print(
                            "use <DataHandler>.ieeg to access intracranial EEG data")
                        print("done")
                return data_ieeg
        else:
            if self.verbose:
                print(
                    f"loading raw iEEG data from {repr(self.filepath_matlab_raw)}")
            if self.full is None:
                self.full = self._load_matlab_raw()
            data_ieeg = self.full['data'][f'ieeg_data{self.exp_phase}']
            return data_ieeg
        """

    def _get_experiment_meta(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """Parse subject experiment data as Pandas dataframe"""
        try:
            # set load path
            if path is None or path == '':
                load_path = self.path
            else:
                load_path = path
            # load saved experiment data
            if load_saved:
                path_df = ''.join(
                    (load_path, "/sub", self.subject_id + "_exp", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self.verbose:
                    print(
                        f"loading saved experiment data file {repr(path_df)}")
                df = pd.read_csv(path_df)
            # create from raw data
            else:
                if self.verbose:
                    print("creating dataframe with experiment data from raw data")
                # cast data to dataframe
                df = pd.DataFrame(self.full['data'][f'file{self.exp_phase}'], columns=[
                    'x_coordinate', 'y_coordinate', 'Picture Number', 'Reaction Time (RT)'])
                df['Picture Number'] = df['Picture Number'].astype(int)
                # create new columns
                # add subject ID
                if self.verbose:
                    print("creating new column 'Subject ID'")
                df['Subject ID'] = [self.subject_id] * df.shape[0]
                # create columns for picture marks
                if self.verbose:
                    print(
                        "creating new columns 'Mark for Picture Shown' and 'Mark for Picture Placed'")
                df['Mark for Picture Shown'] = pd.Series(
                    self.full['data'][f'mark{self.exp_phase}'][0::2]).astype(int)
                df['Mark for Picture Placed'] = pd.Series(
                    self.full['data'][f'mark{self.exp_phase}'][1::2]).astype(int)
                # create columns for picture marks in seconds
                if self.verbose:
                    print(
                        "creating new columns 'Timestamp (s) for Picture Shown' and 'Timestamp (s) for Picture Placed'")
                df['Timestamp (s) for Picture Shown'] = df['Mark for Picture Shown'].apply(
                    lambda x: x/self.fs)
                df['Timestamp (s) for Picture Placed'] = df['Mark for Picture Placed'].apply(
                    lambda x: x/self.fs)
                # create column for reaction time
                if self.verbose:
                    print("creating new column 'Reaction Time (computed)'")
                df['Reaction Time (computed)'] = df['Timestamp (s) for Picture Placed'] - \
                    df['Timestamp (s) for Picture Shown']
                # create column holding trial identifier
                if self.verbose:
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
                if self.verbose:
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

        """
        if load_saved:
            path_df = ''.join(
                (self.path, "/sub", self.subject_id + "_exp", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
            if self.verbose:
                print(
                    f"loading saved experiment data file {repr(path_df)}")
            try:
                df = pd.read_csv(path_df)
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
                    if self.verbose:
                        print("experiment data not loaded")
                    df = None
            finally:
                return df
        else:
            if self.verbose:
                print("creating dataframe with experiment metadata from raw data")
            # cast data to dataframe
            df = pd.DataFrame(self.full['data'][f'file{self.exp_phase}'], columns=[
                              'x_coordinate', 'y_coordinate', 'Picture Number', 'Reaction Time (RT)'])
            df['Picture Number'] = df['Picture Number'].astype(int)
            # create new columns
            # add subject ID
            if self.verbose:
                print("creating new column 'Subject ID'")
            df['Subject ID'] = [self.subject_id] * df.shape[0]
            # create columns for picture marks
            if self.verbose:
                print(
                    "creating new columns 'Mark for Picture Shown' and 'Mark for Picture Placed'")
            df['Mark for Picture Shown'] = pd.Series(
                self.full['data'][f'mark{self.exp_phase}'][0::2]).astype(int)
            df['Mark for Picture Placed'] = pd.Series(
                self.full['data'][f'mark{self.exp_phase}'][1::2]).astype(int)
            # create columns for picture marks in seconds
            if self.verbose:
                print(
                    "creating new columns 'Timestamp (s) for Picture Shown' and 'Timestamp (s) for Picture Placed'")
            df['Timestamp (s) for Picture Shown'] = df['Mark for Picture Shown'].apply(
                lambda x: x/self.fs)
            df['Timestamp (s) for Picture Placed'] = df['Mark for Picture Placed'].apply(
                lambda x: x/self.fs)
            # create column for reaction time
            if self.verbose:
                print("creating new column 'Reaction Time (computed)'")
            df['Reaction Time (computed)'] = df['Timestamp (s) for Picture Placed'] - \
                df['Timestamp (s) for Picture Shown']
            # create column holding trial identifier
            if self.verbose:
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
            df['Trial Identifier'] = df['Trial Identifier'].str.decode('utf-8')
            if self.verbose:
                print("created experiment metadata dataframe")
            return df
        """

    def _get_channel_meta(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """Get channel data as Pandas dataframe"""
        try:
            # set load path
            if path is None or path == '':
                load_path = self.path
            else:
                load_path = path
            # load saved data
            if load_saved:
                path_df = ''.join(
                    (load_path, "/sub", self.subject_id + "_chan", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self.verbose:
                    print(
                        f"loading saved channel data from {repr(path_df)}")
                    df = pd.read_csv(path_df)
            # create from raw data
            else:
                if self.verbose:
                    print("creating dataframe with channel data from raw data")
                filepath = ''.join(
                    (self.path, "/sub", self.subject_id + "_chan.xlsx"))
                df = pd.read_excel(filepath, engine='openpyxl')
                # create True/False column for left hippocampal electrode
                if self.verbose:
                    print("creating column 'HC left'")
                regex = re.compile(r'^(?=.*hippo)(?=.*left).*$', re.IGNORECASE)
                idx = np.where(df['DK_ROI'].str.contains(regex))[0]
                df['HC left'] = False
                df.loc[idx, 'HC left'] = True
                # create True/False column for right hippocampal electrode
                if self.verbose:
                    print("creating column 'HC right'")
                regex = re.compile(
                    r'^(?=.*hippo)(?=.*right).*$', re.IGNORECASE)
                idx = np.where(df['DK_ROI'].str.contains(regex))[0]
                df['HC right'] = False
                df.loc[idx, 'HC right'] = True
                if self.verbose:
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

        """
        if load_saved:
            path_df = ''.join(
                (self.path, "/sub", self.subject_id + "_chan", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
            if self.verbose:
                print(
                    f"loading saved channel data from {repr(path_df)}")
            try:
                df = pd.read_csv(path_df)
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
                    if self.verbose:
                        print("channel data not loaded")
                    df = None
            finally:
                return df
        else:
            if self.verbose:
                print("creating dataframe with channel metadata from raw data")
            filepath = ''.join(
                (self.path, "/sub", self.subject_id + "_chan.xlsx"))
            df = pd.read_excel(filepath, engine='openpyxl')

            # create True/False column for left hippocampal electrode
            if self.verbose:
                print("creating column 'HC left'")
            regex = re.compile(r'^(?=.*hippo)(?=.*left).*$', re.IGNORECASE)
            idx = np.where(df['DK_ROI'].str.contains(regex))[0]
            df['HC left'] = False
            df.loc[idx, 'HC left'] = True
            # df.loc[~idx, 'HC left'] = False
            # create True/False column for right hippocampal electrode
            if self.verbose:
                print("creating column 'HC right'")
            regex = re.compile(r'^(?=.*hippo)(?=.*right).*$', re.IGNORECASE)
            idx = np.where(df['DK_ROI'].str.contains(regex))[0]
            df['HC right'] = False
            df.loc[idx, 'HC right'] = True
            # df.loc[~idx, 'HC right'] = False
            if self.verbose:
                print("created channel metadata dataframe")
            return df
        """

    def _get_targets(self, path: str = '', load_saved: bool = False, postfix: str = ''):
        """Get target locations for each picture as Pandas dataframe"""
        try:
            # set load path
            if path is None or path == '':
                load_path = self.path
            else:
                load_path = path
            # load saved data
            if load_saved:
                path_df = ''.join(
                    (load_path, "/sub", self.subject_id + "_targets", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self.verbose:
                    print(
                        f"loading saved targets data from {repr(path_df)}")
                    df = pd.read_csv(path_df)
            # create from raw data
            else:
                if self.verbose:
                    print("creating dataframe with target data from raw data")
                df = pd.DataFrame(self.full['data'][f'file1'], columns=[
                    'x_coordinate', 'y_coordinate', 'Unnamed: 1', 'Unnamed: 2']).drop(columns=['Unnamed: 1', 'Unnamed: 2'])
                df['picture number'] = np.arange(1, 51)
                if self.verbose:
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

        """
        if load_saved:
            path_df = ''.join(
                (self.path, "/sub", self.subject_id + "_targets", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
            if self.verbose:
                print(
                    f"loading saved targets data from {repr(path_df)}")
            try:
                df = pd.read_csv(path_df)
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
                    if self.verbose:
                        print("targets data not loaded")
                    df = None
            finally:
                return df
        else:
            if self.verbose:
                print("creating dataframe with targets metadata from raw data")
            df = pd.DataFrame(self.full['data'][f'file1'], columns=[
                'x_coordinate', 'y_coordinate', 'Unnamed: 1', 'Unnamed: 2']).drop(columns=['Unnamed: 1', 'Unnamed: 2'])
            df['picture number'] = np.arange(1, 51)
            if self.verbose:
                print("created targets metadata dataframe")
            return df
        """

    def _get_exp_phase_name(self) -> str:
        """_get_exp_phase_name Get the name of the experiment phase

        Returns:
            str: Name of the experiment phase.
        """
        if self.exp_phase == 1:
            return 'preview'
        elif self.exp_phase == 2:
            return 'learning'
        elif self.exp_phase == 3:
            return 'pre-sleep test'
        elif self.exp_phase == 4:
            return 'post-sleep test'
        else:
            raise ValueError(
                f"exp_phase must be 1, 2, 3 or 4, got {self.exp_phase}")

    def load(self, path: str, load_saved: bool = False, postfix: str = '', load_ieeg: bool = True, load_exp: bool = True, load_chan: bool = True, load_targets: bool = True,   verbose: bool = False):
        """Load data as defined by path, subject_id, and exp_phase"""
        # check arguments
        if isinstance(verbose, bool):
            self.verbose = verbose
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
        if self.verbose:
            print("loading data...")
        if path is None or path == '':
            load_path = self.path
            if self.verbose:
                print(
                    f"no save path provided. Using path {repr(load_path)} from initialization")
        else:
            load_path = path
            if not os.path.isdir(os.path.abspath(load_path)):
                raise ValueError(
                    f"path {os.path.abspath(load_path)} does not exist, check path and try again!")
        # get subject iEEG data
        if load_ieeg:
            if self.verbose:
                print("loading iEEG data...")
            self.ieeg = self._get_ieeg(
                path=load_path, load_saved=load_saved, postfix=postfix)
            if self.verbose:
                if self.ieeg is not None:
                    print("use <DataHandler>.ieeg to access intracranial EEG data")
                if self.ieeg is None:
                    print("iEEG data not loaded")
                print("done")
        # load experiment metadata
        if load_exp:
            if self.verbose:
                print("loading experiment metadata...")
            self.df_exp = self._get_experiment_meta(
                path=load_path, load_saved=load_saved, postfix=postfix)
            if self.verbose:
                if self.df_exp is not None:
                    print("use <DataHandler>.df_exp to access experiment dataframe")
                if self.df_exp is None:
                    print("experiment data not loaded")
                print("done")
        # load channel metadata
        if load_chan:
            if self.verbose:
                print("loading channel metadata...")
            self.df_chan = self._get_channel_meta(
                path=load_path, load_saved=load_saved, postfix=postfix)
            if self.verbose:
                if self.df_chan is not None:
                    print("use <DataHandler>.df_chan to access channel dataframe")
                if self.df_chan is None:
                    print("channel data not loaded")
                print("done")
        # load picture targets metadata
        if load_targets:
            if self.verbose:
                print("loading picture target data...")
            self.df_targets = self._get_targets(
                path=load_path, load_saved=load_saved, postfix=postfix)
            if self.verbose:
                if self.df_targets is not None:
                    print("use <DataHandler>.df_targets to access targets dataframe")
                if self.df_targets is None:
                    print("target data not loaded")
                print("done")
        # clean up
        if self.full is not None:
            if self.verbose:
                print(
                    f"cleaning up to free memory...")
            del self.full
            self.full = None
        if self.verbose:
            if self.ieeg is None and self.df_exp is None and self.df_chan is None and self.df_targets is None:
                print("no data loaded!")
            else:
                print("loaded data successfully!")

    def save(self, path: str = None, postfix: str = '', save_ieeg: bool = True, save_exp: bool = True, save_chan: bool = True, save_targets: bool = True, verbose: bool = False):
        """Save iEEG data, experiment metadata, channel metadata and target data as csv files"""
        # check arguments
        if isinstance(verbose, bool):
            self.verbose = verbose
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
        if self.verbose:
            print("saving data...")
        if path is None or path == '':
            save_path = self.path
            if self.verbose:
                print(
                    f"no save path provided. Using path {repr(save_path)} from initialization")
        else:
            save_path = path
            if not os.path.isdir(os.path.abspath(save_path)):
                if self.verbose:
                    print(
                        f"path {os.path.abspath(save_path)} does not exist, creating...")
                os.mkdir(os.path.abspath(save_path))
        # save iEEG data
        if save_ieeg:
            if not self.ieeg is None:
                filename = ''.join(
                    (save_path, "/sub", self.subject_id, "_ieeg", f"{'_' if postfix != '' else ''}{postfix}", ".data"))
                if self.verbose:
                    print(f"saving iEEG data as {repr(filename)}")
                with open(filename, 'wb') as f:
                    pickle.dump(self.ieeg, f)
                if self.verbose:
                    print(f"done")
            else:
                if self.verbose:
                    print(f"<DataHandler>.ieeg is None, skipping...")
        # save experiment dataframe
        if save_exp:
            if not self.df_exp is None:
                filename = ''.join(
                    (save_path, "/sub", self.subject_id, "_exp", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self.verbose:
                    print(f"saving experiment data as {repr(filename)}")
                self.df_exp.to_csv(filename, index=False)
                if self.verbose:
                    print(f"done")
            else:
                if self.verbose:
                    print(f"<DataHandler>.df_exp is None, skipping...")
        # save channel dataframe
        if save_chan:
            if not self.df_chan is None:
                filename = ''.join(
                    (save_path, "/sub", self.subject_id, "_chan", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self.verbose:
                    print(f"saving channel data as {repr(filename)}")
                self.df_chan.to_csv(filename, index=False)
                if self.verbose:
                    print(f"done")
            else:
                if self.verbose:
                    print(f"<DataHandler>.df_chan is None, skipping...")
        # save target data
        if save_targets:
            if not self.df_targets is None:
                filename = ''.join(
                    (save_path, "/sub", self.subject_id, "_targets", f"{'_' if postfix != '' else ''}{postfix}", ".csv"))
                if self.verbose:
                    print(f"saving target data as {repr(filename)}")
                self.df_targets.to_csv(filename, index=False)
                if self.verbose:
                    print(f"done")
            else:
                if self.verbose:
                    print(f"<DataHandler>.df_targets is None, skipping...")
        if self.verbose:
            if self.ieeg is None and self.df_exp is None and self.df_chan is None and self.df_targets is None:
                print("no data saved, load data first!")
            else:
                print("saved data successfully!")


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
            self.verbose = verbose
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
            self.verbose = verbose
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
        if self.verbose:
            print("applying re-referencing method: 'monopolar'")
        if self.verbose:
            print(f"reference channel: {ref_channel}")
        # get all channels except the reference channel
        selector = [i for i in range(data.shape[0]) if i != ref_channel]
        # subtract the reference channel from all other channels
        data[selector, :] = data[selector, :] - data[ref_channel, :]
        if self.verbose:
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
            self.verbose = verbose
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
        if self.verbose:
            print("applying re-referencing method: 'bipolar'")
        # get the unique channels names from ch_names
        unique_electrodes = self._get_unique_channels(ch_names)
        if self.verbose:
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
        if self.verbose:
            print(f"removing {len(remove_indices)} channels from data")
            print(f"removing channels: {remove_ch_names}")
        data = np.delete(data, remove_indices, axis=0)
        if self.verbose:
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
            self.verbose = verbose
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
        if self.verbose:
            print("applying re-referencing method: 'laplacian'")
        # get the unique channels names from ch_names
        unique_electrodes = self._get_unique_channels(ch_names)
        if self.verbose:
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
        if self.verbose:
            print(f"removing {len(remove_indices)} channels from data")
            print(f"removing channels: {remove_ch_names}")
        data = np.delete(data, remove_indices, axis=0)
        if self.verbose:
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
