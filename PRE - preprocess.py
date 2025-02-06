#!/home/mha/.pyvenv/sdcmaster/bin/python

from argparse import ArgumentParser
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import traceback
import warnings
import neuropsy as npsy
import neuropsy.preprocessing as prep
from neuropsy import utils
import copy


# ******** FUNCTIONS ********
def parse_args():
    parser = ArgumentParser(prog='preprocess.py',
                            description='Preprocess data for further analysis',
                            epilog="Preprocess steps include: cleaning, filtering, re-referencing, and saving data")
    parser.add_argument('-d', '--path', type=str,
                        help='Path to data folder with subject data')
    parser.add_argument('-ep', '--exp-phase', nargs='?', const=1, type=int, default=2,
                        help='Experiment phase (1, 2, 3 or 4). Defaults to 2')
    parser.add_argument('-fs', '--sampling-freq', type=int, default=512,
                        help='Sampling frequency (Hz). Defaults to 512 Hz')
    parser.add_argument('-qt', '--use-qt', action='store_true',
                        help='Use Qt backend for matplotlib (flag)')
    parser.add_argument('-v', '--verbose',
                        action='store_true', help='Verbose output (flag)')
    args = parser.parse_args()

    # check if path exists
    if args.path is not None:
        args.path = os.path.abspath(args.path)
        if not os.path.isdir(args.path):
            raise ValueError(f"Path {repr(args.path)} does not exist")
    return args


def get_ch_groups_in_electrode(elec_ch_names):
    """get_ch_groups_in_electrode Get channel groups based on an electrode's name (eg. "A'").

    One electrode might have multiple groups of channels, this function will return the groups
    based on the channel names.

    Args:
        elec_ch_names (list, np.array): List or array of channel names for one electrode.

    Returns:
        elec_ch_groups (list): Nested list of channel names in each group, each sublist represents one group.

    ´´´
    # Example:
    elec_ch_names = ['A 01', 'A 02', 'A 03', 'A 10', 'A 11', 'A 12']
    elec_ch_groups = get_ch_groups_in_electrode(elec_ch_names)
    print(elec_ch_groups)

    # Output:
    [['A 01', 'A 02', 'A 03'], ['A 10', 'A 11', 'A 12']]
    ´´´
    """
    elec_ch_groups = []

    # get the difference between the channel numbers
    diff = np.diff([int(ch.split(' ')[1]) for ch in elec_ch_names])
    # get the indices where the difference is greater than 1
    idx = np.where(diff > 1)[0]
    # if there are multiple groups
    if len(idx) > 0:
        # get the first group
        elec_ch_groups.append(elec_ch_names[:idx[0]+1])
        # get the rest of the groups
        for i in range(1, len(idx)):
            elec_ch_groups.append(elec_ch_names[idx[i-1]+1:idx[i]+1])
        # get the last group
        elec_ch_groups.append(elec_ch_names[idx[-1]+1:])
    # if there is only one group
    else:
        elec_ch_groups.append(elec_ch_names)
    return elec_ch_groups


# ******** MAIN ********
if __name__ == '__main__':

    # init variables
    continue_preprocessing = True  # while loop
    subject_id = None  # defined by user
    postfix = ''
    info_dict = {}   # dictionary to store preprocessing information
    info_subjects_saved = []  # used to determine which subjects to save to JSON info file
    use_same_filters = False
    used_filter_dict = {}   # stores the previously applied filters
    save_path = None
    # if a subject has been preprocessed before
    bool_previous_subject_preprocessed = False

    # parse arguments
    try:
        args = parse_args()
    except ValueError as e:
        print(e)

    # main loop
    try:
        while continue_preprocessing:
            try:
                # check if arguments has been parsed, otherwise prompt for user input
                if args.path is None:
                    args.path = input("Path to data folder: ")
                    if args.path == '':
                        raise ValueError("Path cannot be empty")
                    elif not os.path.isdir(os.path.abspath(args.path)):
                        raise ValueError(
                            f"Path {repr(os.path.abspath(args.path))} does not exist")
                    args.path = os.path.abspath(args.path)
                if subject_id is None:
                    # Read subject IDs from path
                    subject_id_list = utils.get_subject_ids_from_path(
                        args.path)
                    # Create string to print to inform user
                    subject_id_str = '\n'.join(
                        [f"{i+1}. {subject_id_list[i]}" for i in range(len(subject_id_list))])
                    print(f"Available subject IDs:\n{subject_id_str}")
                    subject_id = input("Subject ID: ")
                    if subject_id == '':
                        raise ValueError("Subject ID cannot be empty")
                    else:
                        # check if subject ID is valid
                        if not subject_id.isdigit():
                            raise ValueError(
                                f"Invalid character for subject ID {subject_id}")
                        else:
                            # check if should zero pad
                            subject_id = int(subject_id)
                            if subject_id < 10:
                                subject_id = '0' + str(subject_id)
                            else:
                                subject_id = str(subject_id)
                            # check if subject ID is available
                            if subject_id not in subject_id_list:
                                raise ValueError(
                                    f"Subject ID {subject_id} not available")
                if isinstance(args.exp_phase, str):
                    if int(args.exp_phase) not in [1, 2, 3, 4]:
                        raise ValueError(
                            f"Experiment phase must be either 1, 2, 3 or 4. Got {args.exp_phase}")
                    else:
                        args.exp_phase = int(args.exp_phase)
                elif args.exp_phase is None or args.exp_phase == '':
                    args.exp_phase = input("Experiment phase: ")
                    if args.exp_phase == '':
                        raise ValueError("Experiment phase cannot be empty")
                    if int(args.exp_phase) not in [1, 2, 3, 4]:
                        raise ValueError(
                            f"Experiment phase must be either 1, 2, 3 or 4. Got {args.exp_phase}")
                    else:
                        args.exp_phase = int(args.exp_phase)
                if isinstance(args.sampling_freq, str):
                    if int(args.sampling_freq) < 0:
                        args.sampling_freq = None
                        raise ValueError(
                            f"Sampling frequency must be a positive integer. Got {args.sampling_freq}")
                    elif args.sampling_freq == '':
                        args.sampling_freq = None
                        raise ValueError("Sampling frequency cannot be empty")
                    else:
                        args.sampling_freq = int(args.sampling_freq)
                elif args.sampling_freq is None:
                    args.sampling_freq = input("Sampling frequency (Hz): ")
                    if not args.sampling_freq.isdigit():
                        args.sampling_freq = None
                        raise ValueError(
                            f"Sampling frequency must be a positive integer. Got {args.sampling_freq}")
                    elif args.sampling_freq == '':
                        raise ValueError("Sampling frequency cannot be empty")
                    if int(args.sampling_freq) < 0:
                        raise ValueError(
                            f"Sampling frequency must be a positive integer. Got {args.sampling_freq}")
                    else:
                        args.sampling_freq = int(args.sampling_freq)
                if isinstance(args.use_qt, bool):
                    if args.use_qt:
                        matplotlib.use('Qt5Agg')
                        print("Using Qt5Agg backend for matplotlib")
                    else:
                        matplotlib.use('Agg')
                else:
                    raise ValueError(
                        f"Unrecognized value for flag argument use_qt (see --help), expected True or False, got {type(args.use_qt)}")
                if isinstance(args.verbose, bool):
                    if args.verbose:
                        print("Displaying verbose output...")
                else:
                    raise ValueError(
                        f"Unrecognized value for flag argument verbose (see --help), expected True or False, got {type(args.verbose)}")

                ################# INITIALISE DATAHANDLER #################
                data = npsy.DataHandler(path=args.path,
                                        subject_id=subject_id,
                                        exp_phase=args.exp_phase,
                                        fs=args.sampling_freq,
                                        verbose=args.verbose)

                ################# LOAD DATA #################
                # prompt user to load saved data
                load_saved = input("Do you wish to load saved data? (y/[n]): ")
                if load_saved == 'y':
                    load_saved = True
                    change_path = input(
                        "Do you wish to change the path? (y/[n]): ")
                    if change_path == 'y':
                        args.path = input("Path to saved data folder: ")
                        if args.path == '':
                            raise ValueError("Path cannot be empty")
                        if args.path[-1] != '/':
                            args.path += '/'
                        elif not os.path.isdir(os.path.abspath(args.path)):
                            raise ValueError(
                                f"Path {args.path} does not exist")
                    input_load_postfix = input(
                        "Postfix of saved files to load: ")
                    if input_load_postfix == '':
                        warnings.warn(
                            "Continuing without a postfix!", UserWarning)
                else:
                    load_saved = False
                    input_load_postfix = ''

                data.load(path=args.path,
                          load_saved=load_saved,
                          postfix=input_load_postfix,
                          verbose=args.verbose)

                # Initiate preprocessing information dictionary
                info_dict[f'subject {subject_id}'] = {
                    'cleaning': {
                        'nan': {
                            'applied': False,
                            'columns': None,
                            'indices': [],
                            'count': 0
                        },
                        'outliers': {
                            'applied': False,
                            'columns': None,
                            'indices': [],
                            'count': 0
                        }
                    },
                    'filtering': {
                        'applied': False,
                        'filters': {}
                    },
                    'referencing': {
                        'applied': False,
                        'method': {}
                    },
                    'metadata': {
                        'subject_id': subject_id,
                        'exp_phase': args.exp_phase,
                        'fs': args.sampling_freq,
                        'load': {
                            'loaded': 'saved' if load_saved else 'raw',
                            'path': args.path,
                            'postfix': None if not load_saved else postfix
                        },
                        'save': {
                            'saved': False,
                            'path': None,
                            'postfix': None,
                        }
                    }
                }

                ################# START PREPROCESSING #################
                # prompt user to preprocess data
                arg = input("Continuing to preprocess data? ([y]/n): ").strip()
                if arg == 'n' or arg == 'N':
                    print("Exiting...")
                    exit(0)
                else:

                    ################# CLEANING EXPERIMENT DATA #################
                    # clean NaNs in trials (experiment metadata)
                    arg = input(
                        "Do you wish to clean the data for NaNs? ([y]/n): ").strip()
                    if arg == 'n' or arg == 'N':
                        print("Skipping NaN cleaning...")
                        inp_cols = None
                    else:
                        inp_cols = input(
                            "Columns to clean for NaNs (separated by comma or 'all' for all columns): ")
                        if inp_cols == '':
                            inp_cols = None
                            print(
                                "No columns specified, skipping NaN cleaning...")
                        else:
                            if inp_cols == 'all':
                                if args.verbose:
                                    print("Cleaning NaNs in all columns...")
                                data.df_exp, idx_nans = prep.clean_nan(df=data.df_exp,
                                                                       cols=data.df_exp.columns.to_list(),
                                                                       verbose=args.verbose)
                                cols = 'all'
                            else:
                                if args.verbose:
                                    print(
                                        f"Cleaning NaNs in columns {inp_cols}...")
                                cols = inp_cols.strip().split(',')
                                cols = [col.strip() for col in cols]
                                data.df_exp, idx_nans = prep.clean_nan(df=data.df_exp,
                                                                       cols=cols,
                                                                       verbose=args.verbose)

                            # save preprocessing information
                            info_dict[f'subject {subject_id}']['cleaning']['nan']['applied'] = True
                            info_dict[f'subject {subject_id}']['cleaning']['nan']['columns'] = cols
                            info_dict[f'subject {subject_id}']['cleaning']['nan']['indices'] = [
                                int(idx) for idx in idx_nans]
                            info_dict[f'subject {subject_id}']['cleaning']['nan']['count'] = len(
                                idx_nans)

                    # clean outliers in trials (experiment metadata)
                    arg = input(
                        "Do you wish to clean the data for outliers? ([y]/n): ").strip()
                    if arg == 'n' or arg == 'N':
                        print("Skipping outlier cleaning...")
                    else:
                        inp_cols = input(
                            "Columns to clean for outliers (separated by comma): ")
                        if inp_cols == '':
                            inp_cols = None
                            print(
                                "No columns specified, skipping outlier cleaning...")
                        else:
                            num_std = input(
                                "Number of standard deviations to use for cleaning outliers (default=3): ")
                            if num_std == '':
                                num_std = 3
                            else:
                                num_std = int(num_std)
                            if args.verbose:
                                print(
                                    f"Cleaning outliers in columns {inp_cols} with {num_std} standard deviations...")
                            cols = inp_cols.strip().split(',')
                            cols = [col.strip() for col in cols]
                            _, idx_outliers = prep.clean_outliers(df=data.df_exp,
                                                                  cols=cols,
                                                                  num_std=3,
                                                                  verbose=args.verbose)
                            # add new column to dataframe indicating if trial is an outlier
                            data.df_exp['outlier'] = False
                            data.df_exp.loc[idx_outliers, 'outlier'] = True
                            # save preprocessing information
                            info_dict[f'subject {subject_id}']['cleaning']['outliers']['applied'] = True
                            info_dict[f'subject {subject_id}']['cleaning']['outliers']['columns'] = cols
                            info_dict[f'subject {subject_id}']['cleaning']['outliers']['indices'] = [
                                int(idx) for idx in idx_outliers]
                            info_dict[f'subject {subject_id}']['cleaning']['outliers']['count'] = len(
                                idx_outliers)

                    ################# FILTERING #################
                    continue_filtering = True
                    arg = input(
                        "Do you wish to filter the data? ([y]/n): ").strip()
                    if arg == 'n' or arg == 'N':
                        print("Skipping filtering...")
                        continue_filtering = False

                    # continue to filter data until all desired filters are applied
                    while continue_filtering:
                        if len(used_filter_dict) > 0 and bool_previous_subject_preprocessed == True:
                            print(
                                "Previous subject has been preprocessed with the following filters:")
                            for filter_type, filter_dict in used_filter_dict.items():
                                print(
                                    f"\t{filter_type} filter with order {filter_dict['order']} and cutoff frequency {filter_dict['cutoff']}")
                            arg = input(
                                "Do you wish to use the same filters as the previous subject? ([y]/n): ").strip()
                            if arg == 'n' or arg == 'no':
                                use_same_filters = False
                            else:
                                use_same_filters = True

                        if use_same_filters:
                            for filter_type, filter_dict in used_filter_dict.items():
                                print(
                                    f"Applying {filter_type} filter with order {filter_dict['order']} and cutoff frequency {filter_dict['cutoff']}...")
                                data.ieeg = prep.filter(
                                    data=data.ieeg, filter_dict=filter_dict)
                                print(f"{filter_type} filter applied!")
                                # save preprocessing information
                                info_dict[f'subject {subject_id}']['filtering']['applied'] = True
                                info_dict[f'subject {subject_id}']['filtering']['filters'][filter_type] = {
                                    'order': filter_dict['order'],
                                    'cutoff': filter_dict['cutoff']
                                }
                            continue_filtering = False
                            use_same_filters = False
                        else:
                            print("Available filters:")
                            print("\thighpass/hp")
                            print("\tlowpass/lp")
                            print("\tbandpass/bp")
                            print("\tbandstop/bs")
                            avail_filters = ['highpass', 'lowpass',
                                             'bandpass', 'bandstop', 'hp', 'lp', 'bp', 'bs']
                            input_filter = input(
                                "Filter type to apply: ").strip()
                            # wrong input, skip or continue filtering
                            if input_filter == '':
                                arg = input(
                                    f"No filter specified, still want to continue filtering? ([y]/n): ").strip().lower()
                                if arg == 'n' or arg == 'no':
                                    continue_filtering = False
                                else:
                                    continue
                            # valid filter specified, filter process
                            elif input_filter in avail_filters:
                                if input_filter == 'hp':
                                    input_filter = 'highpass'
                                elif input_filter == 'lp':
                                    input_filter = 'lowpass'
                                elif input_filter == 'bp':
                                    input_filter = 'bandpass'
                                elif input_filter == 'bs':
                                    input_filter = 'bandstop'

                                # filter order
                                input_order = input(
                                    f"Order of {input_filter} filter (default=4): ")
                                if input_order == '':
                                    input_order = 4
                                elif input_order:
                                    input_order = int(input_order)

                                # filter cutoff frequency
                                input_cutoff = input(
                                    f"Cutoff frequency of {input_filter} filter (separated by comma if bandstop or bandpass): ")
                                if input_cutoff == '':
                                    warnings.warn(
                                        f"No cutoff frequency specified, skipping {input_filter} filter...", UserWarning)
                                    arg = input(
                                        f"No cutoff frequency specified, still want to continue filtering? ([y]/n): ").strip().lower()
                                    if arg == 'n' or arg == 'no':
                                        continue_filtering = False
                                    else:
                                        continue
                                else:
                                    input_cutoff = input_cutoff.strip().split(',')
                                    if len(input_cutoff) == 1:
                                        input_cutoff = float(input_cutoff[0])
                                    elif len(input_cutoff) == 2:
                                        input_cutoff = [
                                            float(input_cutoff[0]), float(input_cutoff[1])]
                                    else:
                                        warnings.warn(
                                            f"Invalid number of cutoff frequencies specified, skipping {input_filter} filter...", UserWarning)
                                        continue

                                # ask to show filter response and bodeplot
                                arg = input(
                                    "Do you wish to plot the filter response and bodeplot before applying filter? ([y]/n): ").strip().lower()
                                if arg == 'n' or arg == 'no':
                                    # only show bodeplot and filter response, do not apply filter yet
                                    filter_dict = {
                                        'fs': args.sampling_freq,
                                        'order': input_order,
                                        'cutoff': input_cutoff,
                                        'filter_type': input_filter,
                                        'apply_filter': True,
                                        'show_impulse_response': False,
                                        'show_bodeplot': False,
                                        'len_impulse': 1000,
                                        'filter_mode': 'sos'
                                    }
                                    data.ieeg = prep.filter(
                                        data=data.ieeg, filter_dict=filter_dict)
                                    print(f"{input_filter} filter applied!")
                                    # save preprocessing information
                                    info_dict[f'subject {subject_id}']['filtering']['applied'] = True
                                    info_dict[f'subject {subject_id}']['filtering']['filters'][input_filter] = {
                                        'order': input_order,
                                        'cutoff': input_cutoff
                                    }
                                    # save filter information in case user wants to use the same filters for the next subject
                                    used_filter_dict[input_filter] = filter_dict
                                else:
                                    # only show bodeplot and filter response, do not apply filter yet
                                    filter_dict = {
                                        'fs': args.sampling_freq,
                                        'order': input_order,
                                        'cutoff': input_cutoff,
                                        'filter_type': input_filter,
                                        'apply_filter': False,
                                        'show_impulse_response': True,
                                        'show_bodeplot': True,
                                        'len_impulse': 1000,
                                        'filter_mode': 'sos'
                                    }
                                    sos = prep.filter(
                                        data=data.ieeg, filter_dict=filter_dict, use_qt=args.use_qt)

                                    arg = input(
                                        "Do you wish to apply this filter? ([y]/n): ").strip().lower()
                                    plt.close()
                                    if arg == 'n' or arg == 'no':
                                        arg = input(
                                            f"Filter not applied, still want to continue filtering? ([y]/n): ").strip().lower()
                                        if arg == 'n' or arg == 'no':
                                            continue_filtering = False
                                        else:
                                            continue
                                    # apply filter
                                    else:
                                        filter_dict = {
                                            'fs': args.sampling_freq,
                                            'order': input_order,
                                            'cutoff': input_cutoff,
                                            'filter_type': input_filter,
                                            'apply_filter': True,
                                            'show_impulse_response': False,
                                            'show_bodeplot': False,
                                            'len_impulse': 1000,
                                            'filter_mode': 'sos'
                                        }
                                        data.ieeg = prep.filter(
                                            data=data.ieeg, filter_dict=filter_dict)
                                        print(
                                            f"{input_filter} filter applied!")
                                        # save preprocessing information
                                        info_dict[f'subject {subject_id}']['filtering']['applied'] = True
                                        info_dict[f'subject {subject_id}']['filtering']['filters'][input_filter] = {
                                            'order': input_order,
                                            'cutoff': input_cutoff
                                        }
                                        # save filter information in case user wants to use the same filters for the next subject
                                        used_filter_dict[input_filter] = filter_dict
                            # wrong input, skip or continue filtering
                            else:
                                warnings.warn(
                                    f"Filter {input_filter} not available. Available filters: {avail_filters}", UserWarning)
                                arg = input(
                                    f"Filter {input_filter} not available, still want to continue filtering? ([y]/n): ").strip().lower()
                                if arg == 'n' or arg == 'no':
                                    continue_filtering = False
                                else:
                                    continue
                            # ask if user wants to continue filtering
                            arg = input(
                                f"Want to continue filtering? ([y]/n): ").strip().lower()
                            if arg == 'n' or arg == 'no':
                                continue_filtering = False
                            else:
                                continue

                    ################# RE-REFERENCING #################
                    # prompt user to re-reference data
                    continue_referencing = True
                    while continue_referencing:
                        arg = input(
                            "Do you wish to re-reference the data? ([y]/n): ").strip().lower()
                        if arg == 'n' or arg == 'no':
                            print("Skipping re-referencing...")
                            continue_referencing = False
                            continue
                        # ask for reference method
                        print("Available reference methods:")
                        print("\tmonopolar")
                        print("\tbipolar")
                        print("\tlaplacian")
                        print("\taverage")
                        print("\tmedian")
                        ref_method = input(
                            "Reference method: ").strip()
                        if ref_method == '':
                            warnings.warn(
                                "No reference method specified, choose one of the available methods", UserWarning)
                            arg = input(
                                f"Want to continue referencing? ([y]/n): ").strip().lower()
                            if arg == 'n' or arg == 'no':
                                continue_referencing = False
                            else:
                                continue
                        elif ref_method not in ['monopolar', 'bipolar', 'laplacian', 'average', 'median']:
                            warnings.warn(
                                f"Reference method {ref_method} not available, choose one of the available methods", UserWarning)
                            arg = input(
                                f"Want to continue referencing? ([y]/n): ").strip().lower()
                            if arg == 'n' or arg == 'no':
                                continue_referencing = False
                            else:
                                continue
                        # user has chosen a reference method
                        else:
                            # monopolar: get necessary user arguments
                            if ref_method == 'monopolar':
                                ref_channel = input(
                                    "Reference channel: ").strip()
                                if ref_channel == '':
                                    warnings.warn(
                                        "No reference channel specified, try again!", UserWarning)
                                    continue
                                # assign unused arguments to None
                                direction = None
                            # bipolar
                            elif ref_method == 'bipolar':
                                # get necessary user arguments
                                direction = input(
                                    "Direction of bipolar referencing (left/right): ").strip().lower()
                                if direction == '':
                                    warnings.warn(
                                        "No direction specified, try again!", UserWarning)
                                    continue
                                elif direction not in ['left', 'right']:
                                    warnings.warn(
                                        f"Direction {direction} not available, try again!", UserWarning)
                                    continue
                                # assign unused arguments to None
                                ref_channel = None
                            # laplacian
                            elif ref_method == 'laplacian':
                                # no further user input needed
                                ref_channel = None
                                direction = None
                            # average
                            elif ref_method == 'average':
                                # no further user input needed
                                ref_channel = None
                                direction = None
                            # median
                            elif ref_method == 'median':
                                # no further user input needed
                                ref_channel = None
                                direction = None

                            try:
                                if args.verbose:
                                    print(
                                        f"Starting {repr(ref_method)} re-referencing...")
                                # copy ieeg data in case an exception occurs during the re-referencing the original data is preserved
                                ieeg = copy.deepcopy(data.ieeg)
                                # placeholder for all channel names that could not be re-referenced (important to remove them from channel dataframe and the iEEG data)
                                all_ch_names_to_remove = []
                                # get all channel names for subject
                                ch_names = data.df_chan['name'].to_list()
                                # get unique electrode names
                                elec_names = np.unique(
                                    [ch.split(' ')[0] for ch in ch_names])
                                # loop through all electrode names
                                for elec in elec_names:
                                    if args.verbose:
                                        print(
                                            f"Starting re-referencing for electrode {repr(elec)}...")
                                    # get all channels for the electrode
                                    elec_ch_names = [ch for ch in ch_names if ch.split(' ')[
                                        0] == elec]
                                    # get channel groups based on electrode name
                                    elec_ch_groups = get_ch_groups_in_electrode(
                                        elec_ch_names)
                                    if args.verbose:
                                        # inform user about the channel groups
                                        print(
                                            f"Found {'multiple groups' if len(elec_ch_groups) > 1 else 'a single group'} of channels:")
                                    for i, g in enumerate([g for g in elec_ch_groups]):
                                        print(f"\t{i+1}: {g}")
                                    # loop over all channel groups in electrode (works also if only one group)
                                    for ch_group in elec_ch_groups:
                                        if args.verbose:
                                            print(
                                                f"Applying re-referencing on channel group: {ch_group}")
                                        # get all channel indices for the channels in group from the channel dataframe
                                        elec_ch_indices = data.df_chan[data.df_chan['name'].isin(
                                            ch_group)].index.to_list()
                                        # apply re-referencing to electrode channels
                                        ieeg[elec_ch_indices, :], ch_names_referenced, ch_names_to_remove, ch_names_not_referenced = prep.reference(
                                            data=ieeg[elec_ch_indices, :],
                                            method=ref_method,
                                            ch_names=ch_group,                      # only one group at a time
                                            ref_channel=ref_channel,                # only when monopolar
                                            direction=direction,                    # only when bipolar
                                            verbose=args.verbose)

                                        # [FIXME] this is a quick fix to accommodate the case where the user wants to remove channels that could not be re-referenced
                                        # example is when only 1 channel is in a group but bipolar requires at least 2 channels (laplacian requires 3) in a group to apply re-referencing.
                                        if ch_names_not_referenced is not None:
                                            print(
                                                f"Could not apply {repr(ref_method)} re-referencing to channel(s) {repr(ch_names_not_referenced)}")
                                            print(
                                                f"{repr(ref_method)} needs at least {2 if ref_method == 'bipolar' else 3} channels to apply re-referencing, got {len(ch_names_not_referenced)}")
                                            arg = input(
                                                f"Remove channel(s) {repr(ch_names_not_referenced)}? ([y]/n): ").strip().lower()
                                            if arg == 'y' or arg == 'yes' or arg == '':
                                                print(
                                                    "No channels re-referenced")
                                                print(
                                                    f"Channels to remove: {repr(ch_names_not_referenced)}")
                                                all_ch_names_to_remove.extend(
                                                    ch_names_not_referenced)
                                            else:
                                                raise ValueError(
                                                    f"Could not re-reference channels {repr(ch_names_not_referenced)}")
                                        else:
                                            print(
                                                f"Channels successfully re-referenced: {repr(ch_names_referenced)}")
                                            print(
                                                f"Channels to remove: {repr(ch_names_to_remove)}")
                                            # save information
                                            all_ch_names_to_remove.extend(
                                                ch_names_to_remove)

                                    if args.verbose:
                                        print(
                                            f"Finished re-referencing electrode {repr(elec)}")

                                if args.verbose:
                                    print(
                                        f"Removing {len(all_ch_names_to_remove)} channels that could not be re-referenced...")
                                # get all channel indices for channels that could not be re-referenced and should be removed from the iEEG data
                                all_removed_indices = data.df_chan[data.df_chan['name'].isin(
                                    all_ch_names_to_remove)].index.to_list()
                                # remove channels that could not be re-referenced from channel dataframe
                                if args.verbose:
                                    print(
                                        f"Removing unreferenced channels from channel dataframe...")
                                data.df_chan = data.df_chan.drop(
                                    all_removed_indices)
                                data.df_chan = data.df_chan.reset_index(
                                    drop=True)
                                if args.verbose:
                                    print("Done")
                                # remove channels that could not be re-referenced from iEEG data
                                if args.verbose:
                                    print(
                                        f"Removing unreferenced channels from iEEG data...")
                                ieeg = np.delete(
                                    ieeg, all_removed_indices, axis=0)
                                if args.verbose:
                                    print("Done")
                                # assign re-referenced data to data object after successful re-referencing
                                data.ieeg = ieeg

                                # save preprocessing information
                                info_dict[f'subject {subject_id}']['referencing']['applied'] = True
                                info_dict[f'subject {subject_id}']['referencing']['method']['name'] = ref_method
                                if ref_method == 'monopolar':
                                    info_dict[f'subject {subject_id}']['referencing']['method']['ref_channel'] = ref_channel
                                if ref_method == 'bipolar':
                                    info_dict[f'subject {subject_id}']['referencing']['method']['direction'] = direction
                                info_dict[f'subject {subject_id}']['referencing']['method']['electrodes'] = {
                                }
                                for elec in elec_names:
                                    info_dict[f'subject {subject_id}']['referencing']['method']['electrodes'][elec] = {
                                        "channels": [ch for ch in data.df_chan['name'].to_list() if ch.split(' ')[0] == elec],
                                        "removed": [ch for ch in all_ch_names_to_remove if ch.split(' ')[0] == elec]
                                    }

                                if args.verbose:
                                    print(
                                        f"Finished {repr(ref_method)} re-referencing!")
                                continue_referencing = False
                            except ValueError as e:
                                print(e)
                                print(
                                    "Reverting to previous state (before re-referencing)...")
                                continue

                    ################# SAVE DATA #################
                    # prompt user to save data
                    arg = input(
                        "Do you wish to save the data? ([y]/n): ").strip().lower()
                    if arg == 'y' or arg == 'yes' or arg == '':
                        if save_path:
                            arg = input(
                                f"Previous save path: {repr(save_path)}\nPrevious postfix: {input_save_postfix}\nDo you wish to use the same directory and postfix as previous subject? ([y]/n): ").strip().lower()
                            if arg == 'n' or arg == 'no':
                                # prompt user for save path
                                input_save_path = input(
                                    "Path to save data folder (default is data folder): ").strip()
                                save_path = os.path.abspath(input_save_path)
                                if save_path == '':
                                    print(f"Saving data to {args.path}")
                                    save_path = args.path
                                elif os.path.isdir(save_path):
                                    print(f"Saving data to {save_path}")
                                else:
                                    # create directory if it does not exist
                                    print(f"Creating directory {save_path}")
                                    os.mkdir(save_path)

                                # prompt user for postfix
                                input_save_postfix = input(
                                    "Postfix to append when saving files: ").strip()
                                if input_save_postfix == '':
                                    warnings.warn(
                                        "Continuing without a postfix!", UserWarning)
                            else:
                                pass
                        else:
                            # prompt user for save path
                            input_save_path = input(
                                "Path to save data folder (default is data folder): ").strip()
                            save_path = os.path.abspath(input_save_path)
                            if save_path == '':
                                print(f"Saving data to {args.path}")
                                save_path = args.path
                            elif os.path.isdir(save_path):
                                print(f"Saving data to {save_path}")
                            else:
                                # create directory if it does not exist
                                print(f"Creating directory {save_path}")
                                os.mkdir(save_path)

                            # prompt user for postfix
                            input_save_postfix = input(
                                "Postfix to append when saving files: ").strip()
                            if input_save_postfix == '':
                                warnings.warn(
                                    "Continuing without a postfix!", UserWarning)

                        # save
                        # - experiment metadata
                        # - iEEG data
                        # - channel information
                        # - IED data
                        # - target data
                        data.save(path=save_path,
                                  postfix=input_save_postfix,
                                  verbose=args.verbose)

                        arg = input(
                            "Do you wish to save the iEEG data as a .fif file? ([y]/n): ").strip().lower()
                        if arg == 'y' or arg == 'yes' or arg == '':
                            data.create_mne_raw()
                            # save
                            # - MNE Raw object as .fif file
                            data.write_raw_fif(path=save_path,
                                               overwrite=True)
                        elif arg == 'n' or arg == 'no':
                            print("Skipping MNE Raw object creation...")
                        else:
                            warnings.warn(
                                "Invalid input, skipping MNE Raw object creation...", UserWarning)
                        print("data saved!")
                        # save preprocessing information
                        info_dict[f'subject {subject_id}']['metadata']['save']['saved'] = True
                        info_dict[f'subject {subject_id}']['metadata']['save']['path'] = save_path
                        info_dict[f'subject {subject_id}']['metadata']['save']['postfix'] = input_save_postfix

                        # save preprocessing information to file
                        full_file_path = f'{save_path}/preprocess_info_{input_save_postfix}.json'
                        # append to existing file if it exists
                        if os.path.isfile(full_file_path):
                            # read existing JSON file
                            with open(full_file_path, 'r') as openfile:
                                existing_info_dict = json.load(openfile)
                                # modify dictionary, add new subject info
                                existing_info_dict[f'subject {subject_id}'] = info_dict[f'subject {subject_id}']
                            # write modified dictionary to JSON file
                            with open(full_file_path, 'w') as outfile:
                                json.dump(existing_info_dict, outfile)
                            if args.verbose:
                                print(
                                    f"info file modified {full_file_path}")
                        # create new file if it does not exist
                        else:
                            # save preprocessing information to file
                            with open(f'{save_path}/preprocess_info_{input_save_postfix}.json', 'w') as outfile:
                                json.dump(info_dict, outfile)
                            if args.verbose:
                                print(f"info file saved {full_file_path}")

                        # info_subjects_saved.append(subject_id)
                    elif arg == 'n' or arg == 'no':
                        print("skipping save...")
                    else:
                        warnings.warn(
                            "Invalid input, skipping save...", UserWarning)

                ################# END / CONTINUE PREPROCESSING #################
                # prompt user to continue preprocessing
                arg = input(
                    f"Do you want to continue preprocessing other subject data? ([y]/n): ").strip().lower()
                if arg == 'n' or arg == 'no':
                    continue_preprocessing = False
                else:
                    subject_id = None
                    bool_previous_subject_preprocessed = True
                    continue

            except KeyboardInterrupt:
                print(
                    f"\nSubject {subject_id} data preprocessing interrupted by user!")
                arg = input(
                    "Do you want to continue preprocessing other subject data? (y/[n]): ").strip().lower()
                if arg == 'y' or arg == 'yes':
                    subject_id = None
                    continue
                else:
                    break
            except ValueError as valerr:
                print(valerr)
                print(traceback.format_exc())
                subject_id = None
                continue
    except KeyboardInterrupt:
        print("\nProgram interrupted by user!")
    except Exception as exc:
        print(exc)
        print(traceback.format_exc())
    finally:
        print("Exiting...")
        exit(0)
