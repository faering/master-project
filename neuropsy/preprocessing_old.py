from .data import DataHandler
from .utils import digital_filter

import numpy as np
import pandas as pd


class Preprocessor():

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

    def clean(self, df: pd.DataFrame = None, cols: (list, tuple, np.ndarray) = ['Reaction Time (RT)', 'Reaction Time (computed)'], num_std: int = 3, inplace: bool = True, verbose: bool = None):
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
        if not (isinstance(verbose, bool) or verbose is None):
            raise ValueError(
                f"verbose must be True or False or None, got {type(verbose)}")
        elif verbose is None:
            verbose = self._verbose
        else:
            pass
        if isinstance(df, pd.DataFrame):
            if df.shape[0] == 0:
                raise ValueError(
                    f"df must not be empty, got {df.shape[0]} rows")
        elif df is None:
            df = self.df_exp
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
            if verbose:
                print(f"dropping indices inplace {idx}")
            df.drop(idx, inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            if verbose:
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

    def filter(self, data: np.ndarray, filter_dict: dict = {"order": 4, "cutoff": 500, "fs": 512, "filter_type": 'low', "apply_filter": False, "show_bodeplot": True, "show_impulse_response": True, "len_impulse": 1000}):
        """filter Filter the data.

        [TODO]
        """
        # check arguments
        if isinstance(data, np.ndarray):
            if data.ndim > 2:
                raise ValueError(
                    f"data must be 1D or 2D array, got {data.ndim}D array")
        else:
            raise ValueError(
                f"data must be a numpy array, got {type(data)}")
        if not isinstance(filter_dict, dict):
            raise ValueError(
                f"filter_dict must be a dictionary, got {type(filter_dict)}")

        # filter data
        filtered_data = digital_filter(in_signal=data,
                                       order=filter_dict['order'],
                                       cutoff=filter_dict['cutoff'],
                                       fs=filter_dict['fs'],
                                       filter_type=filter_dict['filter_type'],
                                       apply_filter=filter_dict['apply_filter'],
                                       show_bodeplot=filter_dict['show_bodeplot'],
                                       show_impulse_response=filter_dict['show_impulse_response'],
                                       len_impulse=filter_dict['len_impulse'])
        return filtered_data

    def select_channels(self, ch_names: (list, tuple, np.ndarray)):
        # get indices of channels to remove
        idx = self.df_chan.loc[~self.df_chan['name'].isin(
            ch_names)].index
        # remove from self.ieeg and return resulting array
        self.ieeg = np.delete(self.ieeg, idx, axis=0)
        # remove from self.df_chan
        self.df_chan = self.df_chan.drop(idx).reset_index(drop=True)
        return self.ieeg, self.df_chan

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
