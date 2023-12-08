import numpy as np
import matplotlib.pyplot as plt
import mne


def plot_raw(data: np.ndarray, events: (list, tuple, np.ndarray), event_ids: dict, ch_names: (list, tuple, np.ndarray), fs: int, shading: bool = False, shading_onset: (list, tuple, np.ndarray) = None, shading_duration: (list, tuple, np.ndarray) = None, shading_desc: (list, tuple, np.ndarray) = None, use_qt: bool = False, title: str = None, start: int = None, duration: int = None):
    """plot Plot data with optional annotations.

    Args:
        data (np.ndarray): Data to plot, must be 2D array with shape (channels, data).
        events (list, tuple, np.ndarray): Events to annotate on the plot, can be 1D or 2D.
                                          If 2D, must be (events, samples).
        event_ids (dict): Mapping between event_id and event description. Must be same order as events.
        ch_names (list, tuple, np.ndarray): Names of channels in the data, must be same as data.shape[0].
        fs (int): Sampling frequency of the data.
        shading (bool, optional): If coloured shading is desired across trial duration. Defaults to False.
        shading_onset (list, tuple, np.ndarray, optional): The onset of the shading. Defaults to None.
        shading_duration (list, tuple, np.ndarray, optional): The time duration where the area should be shaded in a colour. Defaults to None.
        shading_desc (list, tuple, np.ndarray, optional): Text to annotate above the shaded area (eg. Trial number). Defaults to None.
        use_qt (bool, optional): Interactive plotting using the Qt5 backend. Defaults to False.

    Raises:
        ValueError: Error in provided value for data.
        ValueError: Error in provided value for events.
        UserWarning: Warning if events is (samples, events) instead of (events, samples).
        ValueError: Error in provided value for event_ids.
        ValueError: Error in provided value for ch_names.
        ValueError: Error in provided value for fs.
        ValueError: Error in provided value for shading.
        ValueError: Error in provided value for shading_onset.
        ValueError: Error in provided value for shading_duration.
        ValueError: Error in provided value for shading_desc.
        ValueError: Error in provided value for use_qt.
    """
    # check arguments
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(
                f"data must be 2D array with shape (channels, data), got {data.ndim}D array")
        elif data.shape[0] > data.shape[1]:
            data = data.T
    else:
        raise ValueError(
            f"data must be a numpy array, got {type(data)}")
    if isinstance(events, (list, tuple, np.ndarray)):
        if np.array(events).dtype != int:
            raise ValueError(
                f"events must be a list, tuple or numpy array of integers, got {np.array(events).dtype}")
        if np.array(events).ndim == 1:
            events = np.array(events).reshape(1, -1)
        if np.array(events).ndim == 2:
            if np.array(events).shape[0] > np.array(events).shape[1]:
                raise UserWarning(
                    f"events should be (events, samples), got shape {np.array(events).shape}, you sure it's not (samples, events)?")
    else:
        raise ValueError(
            f"events must be a list, tuple or numpy array, got {type(events)}")
    if isinstance(event_ids, dict):
        if np.array(events).ndim == 1:
            if len(event_ids) != 1:
                raise ValueError(
                    f"event_ids must have same length as unique events, got {len(event_ids)} but expected 1")
        elif np.array(events).ndim == 2:
            if len(event_ids) != np.array(events).shape[0]:
                raise ValueError(
                    f"event_ids must have same length as unique events, got {len(event_ids)} but expected {np.array(events).shape[0]}")
    else:
        raise ValueError(
            f"event_ids must be a dictionary, got {type(event_ids)}")
    if isinstance(ch_names, (list, tuple, np.ndarray)):
        if np.array(ch_names).ndim != 1:
            raise ValueError(
                f"ch_names must be 1D, got {np.array(ch_names).ndim}D")
    else:
        raise ValueError(
            f"ch_names must be a list, tuple or numpy array, got {type(ch_names)}")
    if not isinstance(fs, int):
        raise ValueError(
            f"fs must be an integer, got {type(fs)}")
    if not isinstance(shading, bool):
        raise ValueError(
            f"shading must be a boolean, got {type(shading)}")
    if isinstance(shading, bool):
        if shading:
            if isinstance(shading_onset, (list, tuple, np.ndarray)):
                if np.array(shading_onset).ndim != 1:
                    raise ValueError(
                        f"shading_onset must be 1D, got {np.array(shading_onset).ndim}D")
            else:
                raise ValueError(
                    f"shading_onset must be a list, tuple or numpy array, got {type(shading_onset)}")
            if isinstance(shading_duration, (list, tuple, np.ndarray)):
                if np.array(shading_duration).ndim != 1:
                    raise ValueError(
                        f"shading_duration must be 1D, got {np.array(shading_duration).ndim}D")
            else:
                raise ValueError(
                    f"shading_duration must be a list, tuple or numpy array, got {type(shading_duration)}")
            if isinstance(shading_desc, (list, tuple, np.ndarray)):
                if np.array(shading_desc).ndim != 1:
                    raise ValueError(
                        f"shading_desc must be 1D, got {np.array(shading_desc).ndim}D")
            else:
                raise ValueError(
                    f"shading_desc must be a list, tuple or numpy array, got {type(shading_desc)}")
            if len(shading_onset) != len(shading_duration) or len(shading_onset) != len(shading_desc):
                raise ValueError(
                    f"shading_onset, shading_duration and shading_desc must have same length, got {len(shading_onset)}, {len(shading_duration)}, {len(shading_desc)}")
    else:
        raise ValueError(
            f"shading must be a boolean, got {type(shading)}")
    if not isinstance(use_qt, bool):
        raise ValueError(
            f"use_qt must be a boolean, got {type(use_qt)}")
    if not (isinstance(title, str) or title is None):
        raise ValueError(
            f"title must be a string or None, got {type(title)}")
    if start is not None and duration is not None:
        if isinstance(start, int) and isinstance(duration, int):
            if start < 0 or start > (len(data[0])//fs) - (fs//duration):
                raise ValueError(
                    f"start must be greater than or equal to 0 and not more than {(len(data[0])//fs) - (fs//duration)}, got {start}")
        else:
            raise ValueError(
                f"start and duration must be integers, got {type(start)} and {type(duration)}")

    # use Qt5Agg backend for matplotlib (needed for interactive plotting)
    if use_qt:
        import matplotlib
        matplotlib.use('Qt5Agg')

    # MNE info object will be used to create MNE Raw object
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=['eeg'] * len(ch_names),
        sfreq=fs
    )

    # create events array (n_events, 3) with (sample, signal_value_preceding_sample, event_id)
    # single event ID
    if np.array(events).ndim == 1:
        events_final = np.array(
            [events,
             np.zeros(len(events), dtype=int),
             np.array(list(event_ids.keys()) * len(events), dtype=int)]
        ).T
    # multiple event IDs
    else:
        events_final = np.array(
            [np.ravel(np.array([arr for arr in events]).T),
             np.zeros(np.array(events).shape[1] *
                      len(list(event_ids.keys())), dtype=int),
             np.array(list(event_ids.keys()) * np.array(events).shape[1], dtype=int)]
        ).T

    # create MNE Raw object
    raw = mne.io.RawArray(data, info)

    # create annotations from events
    annotations_from_events = mne.annotations_from_events(
        events=events_final,
        event_desc=event_ids,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )

    if shading:
        # create annotations for trial shading
        annotations_trial_shading = mne.Annotations(
            onset=shading_onset,
            duration=shading_duration,
            description=shading_desc
        )
        annotations = mne.Annotations.__add__(
            annotations_from_events, annotations_trial_shading)
        # Add annotations to raw object
        raw.set_annotations(annotations)

    # plot continues data with annotations
    if data.shape[0] > 20:
        n_channels = 20
    else:
        n_channels = data.shape[0]
    raw.plot(n_channels=n_channels,
             scalings='auto',
             show=True,
             title=title,
             start=start,
             duration=duration
             )
