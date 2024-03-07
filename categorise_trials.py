import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from argparse import ArgumentParser
import traceback
import numpy as np
from pathlib import Path

import neuropsy as nps


# ******** FUNCTIONS ********
def parse_args():
    parser = ArgumentParser(prog='categorise_trials.py',
                            description='Categorise trials from experiment for further analysis.',
                            epilog="Done per subject individually, the user is asked to define the categories and then categorise trial by trial.")
    parser.add_argument('-d', '--path', type=str,
                        help='Path to data folder with subject data')
    parser.add_argument('-ep', '--exp-phase', nargs='?', const=1, type=int, default=2,
                        help='Experiment phase (1, 2, 3 or 4). Defaults to 2')
    parser.add_argument('-fs', '--sampling-freq', type=int, default=512,
                        help='Sampling frequency (Hz). Defaults to 512 Hz')
    parser.add_argument('-l', '--load-saved',
                        action='store_true', help='Load saved data (flag)')
    parser.add_argument('-qt', '--use-qt', action='store_true',
                        help='Use Qt backend for matplotlib (flag)')
    parser.add_argument('-v', '--verbose',
                        action='store_true', help='Verbose output (flag)')
    args = parser.parse_args()

    # check if path exists
    if args.path is not None:
        args.path = os.path.abspath(args.path)
        if not os.path.isdir(args.path):
            raise ValueError(
                f"Path {repr(args.path)} does not exist, make sure to provide a valid path to the data folder!")
    return args


# ******** MAIN ********
if __name__ == '__main__':

    # init variables
    continue_selecting = True
    subject_id = None
    category_names_list = []
    category_dict = {}
    stop_asking_category_names = False
    postfix = ''
    stop_asking_postfix = False
    # dict of items and category for each trial in item
    # {item_1:
    #   {trial_1: category,
    #   ...
    #   trial_n: category},
    # item_m:
    #   trial_1: category,
    #   ...
    #   trial_n: category}
    dict_trial_categories = {}
    # plot colors
    plot_colors = {
        'outlier': 'salmon',
        'error': 'dodgerblue',
        'reaction_time': 'darkorange',
        'threshold': 'forestgreen',
        'current_trial': 'deeppink'
    }
    dict_category_colors = None
    stop_asking_cat_colors = False

    # parse arguments
    try:
        # [BUG] Try to surpress the Qt5 warning for "failed to get the current screen resources"
        os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.glx.*=false;qt.qpa.xcb.warning=false;qt.accessibility.cache.warning=false;qt.qpa.events.warning=false;qt.qpa.fonts.warning=false;qt.qpa.gl.warning=false;qt.qpa.input.devices.warning=false;qt.qpa.screen.warning=false;qt.text.font.db.warning=false;qt.xkb.compose.warning=false"
        # parse arguments
        args = parse_args()
    except ValueError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        exit(1)

    try:
        if isinstance(args.use_qt, bool):
            if args.use_qt:
                import matplotlib
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

        while continue_selecting:
            try:
                # ********** Check arguments **********#
                # check if arguments has been parsed, otherwise prompt for user input
                if args.path is None:
                    args.path = input("Path to data folder: ")
                    if args.path == '':
                        raise ValueError("Path cannot be empty")
                    elif not os.path.isdir(os.path.abspath(args.path)):
                        raise ValueError(
                            f"Path {repr(os.path.abspath(args.path))} does not exist, make sure to provide a valid path to the data folder!")
                    args.path = os.path.abspath(args.path)
                if args.exp_phase is None:
                    args.exp_phase = input("Experiment phase: ")
                    if args.exp_phase == '':
                        raise ValueError("Experiment phase cannot be empty")
                    ep_list = [1, 2, 3, 4]
                    if int(args.exp_phase) not in ep_list:
                        raise ValueError(
                            f"Experiment phase must be one of {repr(ep_list)}. Got {args.exp_phase}")
                    else:
                        args.exp_phase = int(args.exp_phase)
                if args.sampling_freq is None:
                    args.sampling_freq = input("Sampling frequency: ")
                    if args.sampling_freq == '':
                        raise ValueError("Sampling frequency cannot be empty")
                    else:
                        args.sampling_freq = int(args.sampling_freq)

                # ********** Define categories (one time) **********#
                if category_names_list == [] and stop_asking_category_names == False:
                    category_names_list = input(
                        "Enter category names and identifiers (example, certain:1,uncertain:2): ").strip().split(',')
                    if category_names_list == ['']:
                        category_names_list = []
                        raise ValueError("Category names cannot be empty!")
                    elif len(category_names_list) > 0:
                        category_names_list = [x.strip()
                                               for x in category_names_list]
                        for i, cat in enumerate(category_names_list):
                            if cat == '':
                                category_names_list = []
                                raise ValueError(
                                    f"Category {i+1} is empty, please provide valid category names")
                            elif ':' not in cat:
                                category_names_list = []
                                raise ValueError(
                                    f"Category {i+1} ({cat}) does not have an identifier, please provide identifiers along with category names (category_name:identifier, ...)")
                            elif len(cat.split(':')) != 2:
                                category_names_list = []
                                raise ValueError(
                                    f"Category {i+1} ({cat}) has more than one identifier, please provide only one identifier per category name")
                            elif not cat.split(':')[1].isdigit():
                                category_names_list = []
                                raise ValueError(
                                    f"Category {i+1} ({cat}) has an invalid identifier, please provide an integer identifier")
                        # reach here if no errors in category names and identifiers
                        category_dict = {x.split(':')[1].strip(): x.split(
                            ':')[0].strip() for x in category_names_list}
                        # sort after identifiers
                        category_names_list = sorted(
                            category_names_list, key=lambda x: int(x.split(':')[1]))
                        # create string to print to inform user
                        category_names_str = '\n'.join(
                            [f"{x.split(':')[1]} : {x.split(':')[0]}" for x in category_names_list])
                        print(f"ID : Category\n{category_names_str}")
                        stop_asking_category_names = True
                    else:
                        category_names_list = []
                        raise ValueError(
                            "Invalid input, please provide valid category names and identifiers")

                # ********** Get user input SUBJECT ID **********#
                if subject_id is None:
                    # Read subject IDs from path
                    subject_id_list = nps.utils.get_subject_ids_from_path(
                        args.path)
                    # Create string to print to inform user
                    subject_id_str = '\n'.join(
                        [f"{i+1}. {subject_id_list[i]}" for i in range(len(subject_id_list))])
                    print(f"Available subject IDs:\n{subject_id_str}")
                    subject_id = input("Subject ID: ")
                    if subject_id == '':
                        raise ValueError("Subject ID cannot be empty")

                # ********** Load data **********#
                # init data instance
                data = nps.DataHandler(path=args.path,
                                       subject_id=subject_id,
                                       exp_phase=args.exp_phase,
                                       fs=args.sampling_freq,
                                       verbose=args.verbose)

                # ask for postfix if loading saved data
                if args.load_saved == True and postfix == '' and stop_asking_postfix == False:
                    postfix = input(
                        "Postfix of saved files to load: ").strip()
                    if postfix == '':
                        warnings.warn(
                            "Continuing without a postfix!", UserWarning)
                        stop_asking_postfix = True
                    print(
                        f"loading saved data from {args.path} with postfix {repr(postfix)}...")

                # load data
                data.load(path=args.path,
                          load_saved=args.load_saved,
                          postfix=postfix,
                          verbose=args.verbose)

                # ********** Get Euclidean distances (Trial error) **********#
                dict_eucl_distances = {}
                for pic in data.df_targets['picture number'].unique():
                    dict_eucl_distances[pic] = data.df_exp[data.df_exp['Picture Number']
                                                           == pic]['Trial Error'].to_list()

                # ********** Categorise trials **********#
                # plot one item at a time
                for item in sorted(data.df_exp['Picture Number'].unique()):
                    dict_trial_categories[item] = {}

                    # plot all trials for item, but iterate because one trial should be categorised at a time
                    n_trials = np.arange(
                        0, len(data.df_exp[data.df_exp['Picture Number'] == item]), step=1)
                    x_tick_labels = np.arange(
                        1, len(data.df_exp[data.df_exp['Picture Number'] == item]) + 1, step=1)
                    for n_trial in n_trials:

                        # in case user input is empty or invalid
                        continue_categorising = True
                        while continue_categorising:
                            try:
                                fig, ax = plt.subplots(
                                    nrows=1, ncols=1, figsize=(5, 5))
                                ax2 = ax.twinx()

                                outlier_arr = data.df_exp[data.df_exp['Picture Number'] == item]['outlier'].to_numpy(
                                )
                                outlier_indices = np.where(
                                    outlier_arr == True)[0]
                                # ********** Has outliers **********#
                                if len(outlier_indices) > 0:
                                    outlier_indices = outlier_indices.astype(
                                        int)
                                    if args.verbose:
                                        print(
                                            f"Item {item} has outliers, beware of this when categorising item!")

                                    # ********** Plot trial error **********#
                                    # plot error outlier dot and lines in red
                                    ax.plot(
                                        n_trials, dict_eucl_distances[item], color=plot_colors['outlier'], linestyle='--', linewidth=1)
                                    ax.plot(n_trials[outlier_indices], np.array(dict_eucl_distances[item])[
                                            outlier_indices], marker='o', markersize=5, linestyle='None', color=plot_colors['outlier'], label='outlier')
                                    ax.legend(
                                        loc='best', fontsize=8, shadow=True)
                                    # ********** Plot reaction time (outlier) **********#
                                    # plot outlier reaction time dot and line in red
                                    ax2.plot(data.df_exp[data.df_exp['Picture Number'] == item]['Reaction Time (computed)'].to_numpy(
                                    ), color=plot_colors['outlier'], linestyle=':', linewidth=0.75)
                                    ax2.plot(n_trials[outlier_indices], data.df_exp[data.df_exp['Picture Number'] == item]['Reaction Time (computed)'].to_numpy()[
                                             outlier_indices], marker='o', markersize=5, linestyle='None', color=plot_colors['outlier'])

                                    # ********** Plot trial error (no outlier) **********#
                                    # plot non-outlier error in blue
                                    x = np.delete(n_trials, outlier_indices)
                                    y1 = np.delete(
                                        dict_eucl_distances[item], outlier_indices)
                                    # plot a dot if only one point else lines with no dots
                                    if len(y1) == 1:
                                        ax.plot(x, y1, marker='o', markersize=5,
                                                linestyle='None', color=plot_colors['error'])
                                    else:
                                        ax.plot(x, y1, color=plot_colors['error'],
                                                linestyle='-', linewidth=1.5)

                                    # ********** Plot reaction time **********#
                                    # plot outlier reaction time dot and line in red
                                    ax2.plot(data.df_exp[data.df_exp['Picture Number'] == item]['Reaction Time (computed)'].to_numpy(
                                    ), color=plot_colors['outlier'], linestyle=':', linewidth=0.75)
                                    ax2.plot(n_trials[outlier_indices], data.df_exp[data.df_exp['Picture Number'] == item]['Reaction Time (computed)'].to_numpy()[
                                             outlier_indices], marker='o', markersize=5, linestyle='None', color=plot_colors['outlier'])
                                    # plot non-outlier reaction times line in orange
                                    y2 = np.delete(
                                        data.df_exp[data.df_exp['Picture Number'] == item]['Reaction Time (computed)'].to_numpy(), outlier_indices)
                                    # plot a dot if only one point else lines with no dots
                                    if len(y2) == 1:
                                        ax2.plot(
                                            x, y2, marker='o', markersize=5, linestyle='None', color=plot_colors['reaction_time'])
                                    else:
                                        ax2.plot(x, y2, color=plot_colors['reaction_time'],
                                                 linestyle='--', linewidth=1)
                                # ********** No outliers **********#
                                else:
                                    # ********** Plot trial error **********#
                                    ax.plot(
                                        dict_eucl_distances[item], color=plot_colors['error'], linestyle='-', linewidth=1.5)
                                    # ********** Plot reaction time **********#
                                    ax2.plot(data.df_exp[data.df_exp['Picture Number'] == item]['Reaction Time (computed)'].to_numpy(
                                    ), color=plot_colors['reaction_time'], linestyle='--', linewidth=1)

                                # ********** Plot dot to indicate current trial **********#
                                ax.plot(
                                    n_trial, dict_eucl_distances[item][n_trial], marker='o', markersize=8, linestyle='None', markerfacecolor='none', markeredgecolor=plot_colors['current_trial'])

                                ax.set_title(
                                    f'Trial {n_trial+1}', fontsize=12, color=plot_colors['current_trial'])
                                ax.set_xlabel('Trial', fontsize=10)
                                ax.set_ylabel(
                                    'Error', fontsize=10, color=plot_colors['error'])
                                ax2.set_ylabel(
                                    "Reaction Time (s)", fontsize=10, color=plot_colors['reaction_time'])
                                ax.axhline(
                                    y=150, color=plot_colors['threshold'], linestyle='--', linewidth=0.75)
                                ax.set_xticks(n_trials, x_tick_labels)
                                ax.tick_params(
                                    axis='y', labelcolor=plot_colors['error'])
                                ax2.tick_params(
                                    axis='y', labelcolor=plot_colors['reaction_time'])
                                fig.suptitle(f"Item {item}", fontsize=14)
                                plt.tight_layout()
                                plt.show(block=False)

                                # ********** Get user input, trial category **********#
                                print(f"ID : Category\n{category_names_str}")
                                inp_trial_category = input(
                                    f"Enter category ID for trial {n_trial+1}: ").strip()
                                # correct user input
                                if inp_trial_category in category_dict.keys():
                                    if args.verbose:
                                        print(
                                            f"Trial {n_trial+1} labelled \"{category_dict[inp_trial_category]}\"")
                                    dict_trial_categories[item].update(
                                        {n_trial+1: category_dict[inp_trial_category]})
                                    continue_categorising = False
                                    plt.close()
                                # invalid user input
                                elif inp_trial_category == '':
                                    raise ValueError(
                                        "Category ID cannot be empty!")
                                elif inp_trial_category not in category_dict.keys():
                                    raise ValueError(
                                        f"Invalid category ID {inp_trial_category}, please provide one of the valid category identifiers as shown above!")
                                else:
                                    raise Exception("Unknown error occurred!")
                            except KeyboardInterrupt:
                                print(
                                    f"Received keyboard interrupt, skipping categorising trial {n_trial+1}...")
                                continue_categorising = False
                                plt.close('all')
                                dict_trial_categories[item].update(
                                    {n_trial+1: None})
                                break
                            except ValueError as e:
                                print(e)
                                plt.close('all')
                                continue

                # ********** Plot items and trial category labels **********#
                arg = input(
                    "Do you wish to display items and corresponding category labels? ([y]/n): ").strip().lower()
                if arg == 'n' or arg == 'no':
                    print("skipping...")
                else:
                    # ********** Get user to assign a color to each category (one time) **********#
                    if dict_category_colors is None and stop_asking_cat_colors == False:
                        dict_category_colors = {}
                        print(f"ID : Category\n{category_names_str}")
                        continue_setting_colors = True
                        while continue_setting_colors:
                            try:
                                for cat in category_dict.values():
                                    inp_category_colors = input(
                                        f"Plot color for \"{cat}\": ").strip().lower()
                                    # valid matplotlib CSS colors
                                    if inp_category_colors in list(mcolors.CSS4_COLORS.keys()):
                                        dict_category_colors[cat] = inp_category_colors
                                    elif inp_category_colors == '':
                                        raise ValueError(
                                            "Color cannot be empty, please provide a valid matplotlib CSS color!")
                                    else:
                                        raise ValueError(
                                            f"Invalid color {repr(inp_category_colors)}, please provide a valid matplotlib CSS color!")

                                # check if all categories have been assigned a color
                                if len(dict_category_colors) == len(category_dict):
                                    category_colors_str = '\n'.join(
                                        [f"{k} : {v}" for k, v in dict_category_colors.items()])
                                    print(
                                        f"Category : Color\n{category_colors_str}")
                                    continue_setting_colors = False
                                    stop_asking_cat_colors = True
                            except ValueError as e:
                                print(e)
                                continue

                    arg = input(
                        "Do you wish to save the figures? ([y]/n): ").strip().lower()
                    if arg == 'n' or arg == 'no':
                        print("skipping...")
                        bool_save_fig = False
                    else:
                        setting_save_path = True
                        while setting_save_path:
                            try:
                                bool_save_fig = True
                                save_path = input(
                                    f"Enter path (default is {args.path}): ").strip()
                                if save_path == '':
                                    print(
                                        "Using default path to save figures...")
                                    save_path = args.path
                                    setting_save_path = False
                                    break
                                elif os.path.isdir(save_path) == False:
                                    print(
                                        f"Directory {repr(save_path)} does not exist, creating...")
                                    Path(save_path).mkdir(
                                        parents=True, exist_ok=True)
                                    setting_save_path = False
                                    break
                                else:
                                    print(
                                        f"Saving figures to {repr(save_path)}...")
                                    setting_save_path = False
                                    break
                            except KeyboardInterrupt:
                                print(
                                    "received keyboard interrupt, skipping saving figures...")
                                bool_save_fig = False
                                setting_save_path = False
                                break
                            except Exception as e:
                                print(e)
                                break

                    # plot 5 items at a time
                    n_items_to_plot = 5
                    for n in range(0, len(dict_trial_categories.keys()), n_items_to_plot):
                        item_numbers = []

                        fig, axs = plt.subplots(
                            nrows=1, ncols=n_items_to_plot, figsize=(15, 3))
                        axs = axs.flatten()
                        for i, pic in zip(range(0, n_items_to_plot), list(dict_trial_categories.keys())[n:n+n_items_to_plot+1]):
                            item_numbers.append(pic)

                            # create twin y axis
                            ax2 = axs[i].twinx()

                            # define x axis
                            x_ticks = np.arange(
                                0, len(dict_eucl_distances[pic]), step=1)
                            x_ticks_labels = np.arange(
                                1, len(dict_eucl_distances[pic]) + 1, step=1)

                            # start with plotting big markers indicating trial categories
                            seen_categories = []
                            for trial, category in dict_trial_categories[pic].items():
                                if category is not None:
                                    # only set label one time for each category
                                    if category not in seen_categories:
                                        seen_categories.append(category)
                                        axs[i].plot(trial-1, dict_eucl_distances[pic][trial-1], marker='o', markersize=8, linestyle='None',
                                                    markerfacecolor='none', markeredgecolor=dict_category_colors[category], label=category)
                                    else:
                                        axs[i].plot(trial-1, dict_eucl_distances[pic][trial-1], marker='o', markersize=8, linestyle='None',
                                                    markerfacecolor='none', markeredgecolor=dict_category_colors[category])

                            # check if item has outliers
                            outlier_arr = data.df_exp[data.df_exp['Picture Number'] == pic]['outlier'].to_numpy(
                            )
                            outlier_indices = np.where(
                                outlier_arr == True)[0]
                            # ********** Has outliers **********#
                            if len(outlier_indices) > 0:
                                outlier_indices = outlier_indices.astype(int)

                                # ********** Plot trial error **********#
                                # plot error outlier dot and lines in red
                                axs[i].plot(
                                    x_ticks, dict_eucl_distances[pic], color=plot_colors['outlier'], linestyle='--', linewidth=1)
                                axs[i].plot(x_ticks[outlier_indices],
                                            np.array(dict_eucl_distances[pic])[
                                    outlier_indices],
                                    marker='o',
                                    markersize=5,
                                    linestyle='None',
                                    color=plot_colors['outlier'],
                                    label='outlier')
                                # plot non-outlier error in blue
                                x = np.delete(x_ticks, outlier_indices)
                                y1 = np.delete(
                                    dict_eucl_distances[pic], outlier_indices)
                                # plot a dot if only one point else lines with no dots
                                if len(y1) == 1:
                                    axs[i].plot(
                                        x, y1, marker='o', markersize=5, linestyle='None', color=plot_colors['error'])
                                else:
                                    axs[i].plot(
                                        x, y1, color=plot_colors['error'], linestyle='-', linewidth=1.5)

                                # ********** Plot reaction time **********#
                                # plot outlier reaction time dot and line in red
                                ax2.plot(data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(),
                                         color=plot_colors['outlier'], linestyle=':', linewidth=0.75)
                                ax2.plot(x_ticks[outlier_indices], data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy()[
                                         outlier_indices], marker='o', markersize=5, linestyle='None', color=plot_colors['outlier'])
                                # plot non-outlier reaction times line in orange
                                y2 = np.delete(
                                    data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(), outlier_indices)
                                # plot a dot if only one point else lines with no dots
                                if len(y2) == 1:
                                    ax2.plot(
                                        x, y2, marker='o', markersize=5, linestyle='None', color=plot_colors['reaction_time'])
                                else:
                                    ax2.plot(x, y2, color=plot_colors['reaction_time'],
                                             linestyle='--', linewidth=1)

                            # ********** No outliers **********#
                            else:
                                # ********** Plot trial error **********#
                                axs[i].plot(
                                    dict_eucl_distances[pic], color=plot_colors['error'], linestyle='-', linewidth=1.5)
                                # ********** Plot reaction time **********#
                                ax2.plot(data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(),
                                         color=plot_colors['reaction_time'], linestyle='--', linewidth=1)

                            axs[i].set_title(
                                f'Item {pic}', fontsize=14)
                            axs[i].set_xlabel('Trial', fontsize=10)
                            # only set y1 ylabel for first plot
                            axs[0].set_ylabel(
                                'Error', fontsize=10, color=plot_colors['error'])
                            # only set y2 label for last plot
                            if i == n_items_to_plot-1:
                                ax2.set_ylabel(
                                    "Reaction Time (s)", fontsize=10, color=plot_colors['reaction_time'])
                            axs[i].axhline(y=150, color=plot_colors['threshold'],
                                           linestyle='--', linewidth=0.75)
                            axs[i].set_xticks(x_ticks, x_ticks_labels)
                            axs[i].tick_params(
                                axis='y', labelcolor=plot_colors['error'])
                            ax2.tick_params(
                                axis='y', labelcolor=plot_colors['reaction_time'])
                            axs[i].legend(
                                loc='best', fontsize=6, shadow=True)
                        plt.tight_layout()

                        # save figure
                        if bool_save_fig:
                            full_save_path = os.path.join(
                                save_path, f"subject_{subject_id}_items_{item_numbers[0]}-{item_numbers[-1]}_trial_labels.png")
                            print(f"Saving figure at {full_save_path}...")
                            fig.savefig(full_save_path, dpi=300,
                                        bbox_inches='tight')
                        else:
                            pass
                        plt.show(block=False)
                    arg = input(
                        "Press any key to continue: ")
                    plt.close('all')

                # ********** Create a Pandas series with labels **********#
                label_series = pd.Series(
                    [None] * len(data.df_exp), name='Trial Category')

                # iterate over items
                for pic in data.df_exp['Picture Number'].unique():
                    if len(np.where(np.array(list(dict_trial_categories[pic].values())) == None)[0]) > 0:
                        print(
                            f"Item {pic} has trials that have not been labelled and will appear as NaN in 'Trial Category'!")

                    # iterate over trials in item to set the labels
                    for trial in data.df_exp[data.df_exp['Picture Number'] == pic]['Trial Identifier']:
                        # get trial number and index in dataframe
                        num_trial = trial.split('-')[1]
                        idx_trial = data.df_exp[data.df_exp['Trial Identifier'] == trial].index.to_list(
                        )
                        # save label
                        if dict_trial_categories[pic][int(num_trial)] is not None:
                            label_series[idx_trial] = dict_trial_categories[pic][int(
                                num_trial)]
                        else:
                            label_series[idx_trial] = None

                # ********** Save trial categories as CSV file **********#
                inp_save_result = input(
                    "Save trial categories as a CSV file? ([y]/n): ").strip().lower()
                if inp_save_result == 'y' or inp_save_result == 'yes' or inp_save_result == '':
                    inp_save_path = input(
                        f"Enter path (default is {args.path}): ").strip()
                    if inp_save_path == '':
                        inp_save_path = args.path

                    filename = ''.join(
                        (inp_save_path, "/sub", f"{'0' + subject_id if len(subject_id) == 1 else subject_id}", "_trial_labels.csv"))
                    try:
                        label_series.to_csv(filename, index=False)
                        if args.verbose:
                            print(
                                f"Saved trial categories as CSV file at {filename}")
                    except Exception as e:
                        print(e)
                        print(traceback.format_exc())
                        raise ValueError(
                            f"Failed to save trial categories as CSV file at {filename}") from e

                elif inp_save_result == 'n' or inp_save_result == 'no':
                    print("Skipping...")
                else:
                    raise ValueError(
                        f"Invalid input {repr(inp_save_result)}!")

                # ********** Edit and save experiemnt dataframe **********#
                inp_edit_df = input(
                    "Edit subject's experiment dataframe? ([y]/n): ").strip().lower()
                if inp_edit_df == 'y' or inp_edit_df == 'yes' or inp_edit_df == '':

                    # edit subject dataframe
                    data.df_exp['Trial Category'] = label_series

                    # ********** Save edited dataframe **********#
                    # save edited subject dataframe
                    data.save(path=args.path,
                              postfix=postfix,
                              save_ieeg=False,
                              save_chan=False,
                              save_exp=True,
                              save_targets=False,
                              verbose=args.verbose)

                elif inp_edit_df == 'n' or inp_edit_df == 'no':
                    print("Skipping...")
                else:
                    raise ValueError(
                        f"Invalid input {repr(inp_edit_df)}!")

                # ********** Continue to new subject? **********#
                # prompt user to continue processing another subject
                inp_continue = input(
                    f"Do you want to continue onto next subject? ([y]/n): ").strip().lower()
                if inp_continue == 'n' or inp_continue == 'no':
                    continue_selecting = False
                elif inp_continue == 'y' or inp_continue == 'yes' or inp_continue == '':
                    # Reset variables
                    subject_id = None
                    continue
                else:
                    raise ValueError(
                        f"Invalid input {repr(inp_continue)}!")

            except ValueError as e:
                print(e)
                print(traceback.format_exc())
                # Reset variables
                subject_id = None
                items_to_keep_list = []
                trial_numbers_list = []
                continue
    except KeyboardInterrupt:
        print("Interrupted by user!")
        continue_selecting = False
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        plt.close('all')
    finally:
        print("Exiting...")
        exit(0)
