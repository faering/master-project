import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import traceback
import numpy as np
from pathlib import Path

import neuropsy as nps


# ******** FUNCTIONS ********
def parse_args():
    parser = ArgumentParser(prog='categorise_items.py',
                            description='Categorise items from experiment for further analysis.',
                            epilog="Done per subject individually, the user will define categories and put items into said categories.")
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
    # dict of items and corresponding category for subject being currently processed
    dict_item_categories = {}
    # (optional) list of corresponding trial indices for items to keep, for dividing trials into uncertain (before index) and certain (after index) conditions
    trial_numbers_list = []

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
                        "Enter category names and identifiers (example, good:1,average:2,bad:3): ").strip().split(',')
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

                # ********** Categorise items **********#
                # plot 5 items at a time
                n_items_to_plot = 5
                for n in range(0, len(dict_eucl_distances), n_items_to_plot):

                    # in case user input is empty or invalid
                    continue_categorising = True
                    while continue_categorising:

                        try:
                            fig, axs = plt.subplots(
                                nrows=1, ncols=n_items_to_plot, figsize=(15, 3))
                            axs = axs.flatten()
                            item_numbers = []
                            for i, pic in zip(range(0, n_items_to_plot), list(dict_eucl_distances.keys())[n:n+n_items_to_plot+1]):
                                # used below to display item and selected category
                                item_numbers.append(pic)

                                # create twin y axis
                                ax2 = axs[i].twinx()

                                # define x axis
                                x_ticks = np.arange(
                                    0, len(dict_eucl_distances[pic]), step=1)
                                x_ticks_labels = np.arange(
                                    1, len(dict_eucl_distances[pic]) + 1, step=1)

                                # check if item has outliers
                                outlier_arr = data.df_exp[data.df_exp['Picture Number'] == pic]['outlier'].to_numpy(
                                )
                                outlier_indices = np.where(
                                    outlier_arr == True)[0]

                                # ********** Has outliers **********#
                                if len(outlier_indices) > 0:
                                    outlier_indices = outlier_indices.astype(
                                        int)
                                    if args.verbose:
                                        print(
                                            f"Item {pic} has outliers, beware of this when categorising item!")

                                    # ********** Plot trial error **********#
                                    # plot error outlier dot and lines in red
                                    axs[i].plot(
                                        x_ticks, dict_eucl_distances[pic], color='red', linestyle='--', linewidth=1)
                                    axs[i].plot(x_ticks[outlier_indices],
                                                np.array(dict_eucl_distances[pic])[
                                        outlier_indices],
                                        marker='o',
                                        markersize=5,
                                        linestyle='None',
                                        color='red',
                                        # markerfacecolor='red',
                                        # markeredgecolor='red',
                                        label='outlier')

                                    # plot non-outlier error in blue
                                    x = np.delete(x_ticks, outlier_indices)
                                    y1 = np.delete(
                                        dict_eucl_distances[pic], outlier_indices)
                                    # plot a dot if only one point else lines with no dots
                                    if len(y1) == 1:
                                        axs[i].plot(
                                            x, y1, marker='o', markersize=5, linestyle='None', color='dodgerblue')
                                    else:
                                        axs[i].plot(
                                            x, y1, color='dodgerblue', linestyle='-', linewidth=1.5)
                                    axs[i].legend(
                                        loc='best', fontsize=8, shadow=True)

                                    # ********** Plot reaction time **********#
                                    # plot outlier reaction time dot and line in red
                                    ax2.plot(data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(),
                                             color='red', linestyle=':', linewidth=0.75)
                                    ax2.plot(x_ticks[outlier_indices], data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy()[
                                             outlier_indices], marker='o', markersize=5, linestyle='None', color='red')
                                    # plot non-outlier reaction times line in orange
                                    y2 = np.delete(
                                        data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(), outlier_indices)
                                    # plot a dot if only one point else lines with no dots
                                    if len(y2) == 1:
                                        ax2.plot(
                                            x, y2, marker='o', markersize=5, linestyle='None', color='darkorange')
                                    else:
                                        ax2.plot(x, y2, color='darkorange',
                                                 linestyle='--', linewidth=1)
                                # ********** No outliers **********#
                                else:
                                    # ********** Plot trial error **********#
                                    axs[i].plot(
                                        dict_eucl_distances[pic], color='dodgerblue', linestyle='-', linewidth=1.5)
                                    # ********** Plot reaction time **********#
                                    ax2.plot(data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(),
                                             color='darkorange', linestyle='--', linewidth=1)

                                axs[i].set_title(f'Item {pic}', fontsize=14)
                                axs[i].set_xlabel('Trial', fontsize=10)
                                # only set y1 ylabel for first plot
                                axs[0].set_ylabel(
                                    'Error', fontsize=10, color='dodgerblue')
                                # only set y2 label for last plot
                                if i == n_items_to_plot-1:
                                    ax2.set_ylabel(
                                        "Reaction Time (s)", fontsize=10, color='darkorange')
                                axs[i].axhline(
                                    y=150, color='forestgreen', linestyle='--', linewidth=0.75)
                                axs[i].set_xticks(x_ticks, x_ticks_labels)
                            plt.tight_layout()
                            plt.show(block=False)

                            # get user input for item categories
                            print(f"ID : Category\n{category_names_str}")
                            inp_item_categories = input(
                                "Enter category ID for each item (separated by comma):").strip().split(',')
                            if inp_item_categories == ['']:
                                raise ValueError(
                                    "Category IDs cannot be empty!")
                            elif len(inp_item_categories) != n_items_to_plot:
                                raise ValueError(
                                    "Number of category IDs must match the number of items displayed!")
                            else:
                                # check if input is valid
                                for j, cat in enumerate(inp_item_categories):
                                    if cat == '':
                                        raise ValueError(
                                            f"Identifier at position {j+1} is empty, please provide valid category identifier!")
                                    elif cat not in category_dict.keys():
                                        raise ValueError(
                                            f"Invalid category ID {cat}, please provide one of the valid category identifiers as shown above!")
                                    else:
                                        if args.verbose:
                                            print(
                                                f"Item {item_numbers[j]} labelled \"{category_dict[cat]}\"")
                                # No errors in input
                                # save selected category for item
                                dict_item_categories.update(
                                    {item_numbers[j]: category_dict[cat] for j, cat in enumerate(inp_item_categories)})
                                continue_categorising = False
                                plt.close()
                        except KeyboardInterrupt:
                            print(
                                "received keyboard interrupt, skipping categorising items...")
                            continue_categorising = False
                            plt.close()
                            dict_item_categories.update(
                                {item_numbers[j]: None for j, _ in enumerate(inp_item_categories)})
                            break
                        except ValueError as e:
                            print(e)
                            plt.close()
                            continue

                # ********** Plot items with category labels **********#
                arg = input(
                    "Do you wish to display items and corresponding category labels? ([y]/n): ").strip().lower()
                if arg == 'n' or arg == 'no':
                    print("skipping...")
                else:
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
                                    "Enter path to save figures (default is provided data path): ").strip()
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
                    for n in range(0, len(dict_item_categories), n_items_to_plot):
                        item_numbers = []

                        fig, axs = plt.subplots(
                            nrows=1, ncols=n_items_to_plot, figsize=(15, 3))
                        axs = axs.flatten()
                        for i, pic in zip(range(0, n_items_to_plot), list(dict_item_categories.keys())[n:n+n_items_to_plot+1]):
                            item_numbers.append(pic)

                            # create twin y axis
                            ax2 = axs[i].twinx()

                            # define x axis
                            x_ticks = np.arange(
                                0, len(dict_eucl_distances[pic]), step=1)
                            x_ticks_labels = np.arange(
                                1, len(dict_eucl_distances[pic]) + 1, step=1)

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
                                    x_ticks, dict_eucl_distances[pic], color='red', linestyle='--', linewidth=1)
                                axs[i].plot(x_ticks[outlier_indices],
                                            np.array(dict_eucl_distances[pic])[
                                    outlier_indices],
                                    marker='o',
                                    markersize=5,
                                    linestyle='None',
                                    color='red',
                                    label='outlier')
                                # plot non-outlier error in blue
                                x = np.delete(x_ticks, outlier_indices)
                                y1 = np.delete(
                                    dict_eucl_distances[pic], outlier_indices)
                                # plot a dot if only one point else lines with no dots
                                if len(y1) == 1:
                                    axs[i].plot(
                                        x, y1, marker='o', markersize=5, linestyle='None', color='dodgerblue')
                                else:
                                    axs[i].plot(
                                        x, y1, color='dodgerblue', linestyle='-', linewidth=1.5)
                                axs[i].legend(
                                    loc='best', fontsize=8, shadow=True)

                                # ********** Plot reaction time **********#
                                # plot outlier reaction time dot and line in red
                                ax2.plot(data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(),
                                         color='red', linestyle=':', linewidth=0.75)
                                ax2.plot(x_ticks[outlier_indices], data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy()[
                                         outlier_indices], marker='o', markersize=5, linestyle='None', color='red')
                                # plot non-outlier reaction times line in orange
                                y2 = np.delete(
                                    data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(), outlier_indices)
                                # plot a dot if only one point else lines with no dots
                                if len(y2) == 1:
                                    ax2.plot(
                                        x, y2, marker='o', markersize=5, linestyle='None', color='darkorange')
                                else:
                                    ax2.plot(x, y2, color='darkorange',
                                             linestyle='--', linewidth=1)

                            # ********** No outliers **********#
                            else:
                                # ********** Plot trial error **********#
                                axs[i].plot(
                                    dict_eucl_distances[pic], color='dodgerblue', linestyle='-', linewidth=1.5)
                                # ********** Plot reaction time **********#
                                ax2.plot(data.df_exp[data.df_exp['Picture Number'] == pic]['Reaction Time (computed)'].to_numpy(),
                                         color='darkorange', linestyle='--', linewidth=1)

                            axs[i].set_title(
                                f'Item {pic} - "{dict_item_categories[pic]}"', fontsize=14)
                            axs[i].set_xlabel('Trial', fontsize=10)
                            # only set y1 ylabel for first plot
                            axs[0].set_ylabel(
                                'Error', fontsize=10, color='dodgerblue')
                            # only set y2 label for last plot
                            if i == n_items_to_plot-1:
                                ax2.set_ylabel(
                                    "Reaction Time (s)", fontsize=10, color='darkorange')
                            axs[i].axhline(y=150, color='forestgreen',
                                           linestyle='--', linewidth=0.75)
                            axs[i].set_xticks(x_ticks, x_ticks_labels)
                        plt.tight_layout()

                        # save figure
                        if bool_save_fig:
                            full_save_path = os.path.join(
                                save_path, f"subject_{subject_id}_items_{item_numbers[0]}-{item_numbers[-1]}_item_labels.png")
                            print(f"Saving figure at {full_save_path}...")
                            fig.savefig(full_save_path, dpi=300,
                                        bbox_inches='tight')
                        else:
                            pass
                        plt.show(block=False)
                    arg = input(
                        "Press any key to continue: ")
                    plt.close('all')

                # ********** Edit and save experiemnt dataframe **********#
                if len(np.where(np.array(list(dict_item_categories.values())) == None)[0]) == len(dict_item_categories):
                    print(
                        "No items have been labelled, skipping editing experiment dataframe...")
                else:
                    # prompt user if subject dataframe should be edited and saved
                    inp_edit_df = input(
                        "Edit and save experiment dataframe? ([y]/n): ").strip().lower()
                    if inp_edit_df == 'n' or inp_edit_df == 'no':
                        print("skipping...")
                    elif inp_edit_df == 'y' or inp_edit_df == 'yes' or inp_edit_df == '':

                        # ********** Add column 'Item Category' **********#
                        print(
                            "Appending new column 'Item Category' with True/False values to experiment dataframe...")
                        # initiate column as pd.Series
                        label_series = pd.Series(
                            [None] * len(data.df_exp), name='Item Category')
                        # set the labels for the items
                        for item, label in dict_item_categories.items():
                            idx = data.df_exp[data.df_exp['Picture Number']
                                              == item].index.to_list()
                            label_series[idx] = label
                        # edit subject dataframe
                        data.df_exp['Item Category'] = label_series

                        # ********** Save edited dataframe **********#
                        # save edited subject dataframe
                        data.save(path=args.path,
                                  postfix=postfix,
                                  save_ieeg=False,
                                  save_chan=False,
                                  save_exp=True,
                                  save_targets=False,
                                  verbose=args.verbose)
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
                continue
    except KeyboardInterrupt:
        print("Interrupted by user!")
        continue_selecting = False
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    finally:
        print("Exiting...")
        exit(0)
