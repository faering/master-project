import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def digital_filter(in_signal: np.ndarray, order: int, cutoff: (int, float, list, tuple, np.ndarray), fs: int, filter_type: str = 'low', apply_filter: bool = True, show_bodeplot: bool = False, show_impulse_response: bool = False, len_impulse: int = 1000, filter_mode: str = 'sos', use_qt: bool = False):
    """digital_filter Create a digital filter using scipy's butter function.

    Create a digital filter using scipy's butter function and compute the
    filter frequency response. The filter frequency response is always plotted
    unless explicitely specified to False, this is to ensure the desired filter design
    is obtained.

    Args:
        in_signal (np.array, optional): If specified and apply_filter is set to True, the filter is applied to the signal 
                                    and the filtered signal is returned. Defaults to None.
        order (int, optional): The order of the filter. Defaults to 4.
        cutoff (int or vector, optional): The cutoff frequency of the filter (in Hz). If filter_type is 'bandpass' or
                                        'bandstop' cutoff is a length-2 vector. Defaults to 500.
        fs (int, optional): The sampling frequency used to sample the signal. Defaults to 1000.
        filter_type (str, optional): The type of filter ('lowpass', 'highpass', 'bandstop', 'bandpass'). Defaults to 'high'.
        apply_filter (bool, optional): If True then the filter is applied to the signal in_signal. Filter will be applied forward
                                    and backward on the signal to eliminate phase shift. Defaults to False.
        show_bodeplot (bool, optional): If True then the bode plot of the filter frequency response is plotted. Defaults to True.
        show_impulse_response (bool, optional): If True then the impulse response of the filter is plotted. Defaults to True.
        len_impulse (int, optional): The length of the impulse response. If not specified defaults to 1000.
        filter_mode (str, optional): Can be 'sos' (default) or 'ba'. Defaults to 'sos'.
        use_qt (bool, optional): If True then the matplotlib backend is set to 'Qt5Agg'. Defaults to False. [REMOVE]

    Returns:
        b, a (ndarray): Default setting for function is to return the 
                        filter coefficients.
        out_signal (ndarray): If apply_filter is set to True and in_signal 
                            is given, the function will apply the digital 
                            filter and return the filtered signal.
    """
    if isinstance(in_signal, np.ndarray):
        if in_signal.ndim != 1:
            raise ValueError(
                f"in_signal must be 1D array, got {in_signal.ndim}D array")
    else:
        raise ValueError(
            f"in_signal must be numpy array, got {type(in_signal)}")
    if not isinstance(order, int):
        raise ValueError(f"order must be integer, got {type(order)}")
    if not isinstance(fs, int):
        raise ValueError(f"fs must be integer, got {type(fs)}")
    if isinstance(filter_type, str):
        if filter_type == 'bandstop' or filter_type == 'bs' or filter_type == 'bandpass' or filter_type == 'bp':
            cutoff_need_2d = True
        else:
            cutoff_need_2d = False
    else:
        raise ValueError(
            f"filter_type must be string, got {type(filter_type)}")
    if isinstance(cutoff, (int, float, list, tuple, np.ndarray)):
        if cutoff_need_2d:
            if isinstance(cutoff, (list, tuple, np.ndarray)):
                if len(np.ravel(cutoff)) != 2:
                    raise ValueError(
                        f"filter_type is {repr(filter_type)}, cutoff must be a list, tuple, or array of length 2, got length {len(np.unravel(cutoff))}")
            else:
                raise ValueError(
                    f"filter_type is {repr(filter_type)}, cutoff must be a list, tuple, or array")
        else:
            if not isinstance(cutoff, (int, float)):
                raise ValueError(
                    f"filter_type is {repr(filter_type)}, cutoff must be int or float, got {type(filter_type)}")
    else:
        raise ValueError(
            f"cutoff must be int, float, list, tuple, or array, got {type(cutoff)}")
    if not isinstance(apply_filter, bool):
        raise ValueError(
            f"apply_filter must be True or False, got {type(apply_filter)}")
    if not isinstance(show_bodeplot, bool):
        raise ValueError(
            f"show_bodeplot must be True or False, got {type(show_bodeplot)}")
    if not isinstance(show_impulse_response, bool):
        raise ValueError(
            f"show_impulse_response must be True or False, got {type(show_impulse_response)}")
    if not isinstance(len_impulse, int):
        raise ValueError(
            f"len_impulse must be integer, got {type(len_impulse)}")
    if isinstance(filter_mode, str):
        if filter_mode not in ('sos', 'ba'):
            raise ValueError(
                f"filter_mode must be 'sos' or 'ba', got {repr(filter_mode)}")
    else:
        raise ValueError(
            f"filter_mode must be string, got {type(filter_mode)}")
    # if isinstance(use_qt, bool):
    #     if use_qt:
    #         import matplotlib
    #         matplotlib.use('Qt5Agg')
    #         print("Using Qt5Agg backend for matplotlib")
    #     else:
    #         pass
    # else:
    #     raise ValueError(
    #         f"use_qt must be True or False, got {type(use_qt)}")

    try:
        # Create digital filter and compute filter frequency response
        if filter_mode == 'ba':
            b, a = signal.butter(N=order, Wn=cutoff,
                                 btype=filter_type, fs=fs, analog=False)
            w, h = signal.freqz(b, a, fs=fs)
        elif filter_mode == 'sos':
            sos = signal.butter(N=order, Wn=cutoff, btype=filter_type,
                                fs=fs, analog=False, output='sos')
            w, h = signal.sosfreqz(sos, fs=fs)

        # Compute impulse response
        if show_impulse_response:
            # Generate and compute impulse response
            # create impulse
            impulse = signal.unit_impulse(len_impulse)
            # create time vector for impulse
            t_impulse = np.arange(-(len_impulse//10),
                                  len_impulse-(len_impulse//10))
            if filter_mode == 'ba':
                # compute filter response to impulse
                response = signal.lfilter(b, a, impulse)
            elif filter_mode == 'sos':
                response = signal.sosfilt(sos, impulse)

        # Plotting
        if show_bodeplot and show_impulse_response:
            fig, ax = plt.subplots(1, 2, figsize=(12, 7))
            # Bode plot
            if h[0] == 0:
                # [FIXME] This is a hack to fix the divide by zero error
                ax[0].semilogx(w, np.insert(
                    20*np.log10(abs(h[1:])), 1, 0), linewidth=3)
            else:
                ax[0].semilogx(w, 20*np.log10(abs(h)), linewidth=3)
            ax[0].set_title("Digital Filter Frequency Response", fontsize=20)
            ax[0].set_xlabel("Frequency [Hz]", fontsize=16)
            ax[0].set_ylabel("Amplitude [dB]", fontsize=16)
            ax[0].grid(which='both', axis='both')
            if filter_type in ('low', 'lowpass'):
                ax[0].margins(0, 0.15)
            if filter_type in ('high', 'highpass'):
                ax[0].margins(0.15, 0.1)
            if filter_type in ('bs', 'bandstop', 'bp', 'bandpass'):
                ax[0].axvline(cutoff[0], color='orange',
                              label=f"Cutoff frequency start ({cutoff[0]} Hz)", linewidth=2, linestyle='--')
                ax[0].axvline(cutoff[1], color='orange',
                              label=f"Cutoff frequency stop ({cutoff[1]} Hz)", linewidth=2, linestyle='--')
            if filter_type not in ('bs', 'bandstop', 'bp', 'bandpass'):
                # plot a star symbol at -3 dB attenuation
                ax[0].semilogx(cutoff, 20*np.log10(0.5*np.sqrt(2)), '*')
                ax[0].axvline(
                    cutoff, color='orange', label=f"Cutoff frequency ({cutoff} Hz)", linewidth=2, linestyle='--')
            ax[0].legend(loc='best', fontsize=12, shadow=True)
            # Impulse response plot
            ax[1].plot(t_impulse, impulse, 'b-', label='Impulse')
            ax[1].plot(t_impulse, response, 'g-',
                       linewidth=2, label='Response')
            ax[1].set_title(
                'Impulse and Response of Digital Filter', fontsize=20)
            ax[1].set_xlabel('Time [sec]', fontsize=16)
            ax[1].set_ylabel('Amplitude', fontsize=16)
            ax[1].grid(which='both', axis='both')
            ax[1].legend(loc='best', fontsize=12, shadow=True)
            plt.tight_layout()
            plt.show(block=False)
        elif show_bodeplot:
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            if h[0] == 0:
                # [FIXME] This is a hack to fix the divide by zero error
                ax[0].semilogx(w, np.insert(
                    20*np.log10(abs(h[1:])), 1, 0), linewidth=3)
            else:
                ax[0].semilogx(w, 20*np.log10(abs(h)), linewidth=3)
            ax[0].set_title("Digital Filter Frequency Response", fontsize=20)
            ax[0].set_xlabel("Frequency [Hz]", fontsize=16)
            ax[0].set_ylabel("Amplitude [dB]", fontsize=16)
            ax[0].grid(which='both', axis='both')
            if filter_type in ('low', 'lowpass'):
                ax[0].margins(0, 0.15)
            if filter_type in ('high', 'highpass'):
                ax[0].margins(0.15, 0.1)
            if filter_type in ('bs', 'bandstop', 'bp', 'bandpass'):
                ax[0].axvline(cutoff[0], color='orange',
                              label=f"Cutoff frequency start ({cutoff[0]} Hz)", linewidth=2, linestyle='--')
                ax[0].axvline(cutoff[1], color='orange',
                              label=f"Cutoff frequency stop ({cutoff[1]} Hz)", linewidth=2, linestyle='--')
            if filter_type not in ('bs', 'bandstop', 'bp', 'bandpass'):
                # plot a star symbol at -3 dB attenuation
                ax[0].semilogx(cutoff, 20*np.log10(0.5*np.sqrt(2)), '*')
                ax[0].axvline(
                    cutoff, color='orange', label=f"Cutoff frequency ({cutoff} Hz)", linewidth=2, linestyle='--')
            ax[0].legend(loc='best', fontsize=12, shadow=True)
            plt.tight_layout()
            plt.show(block=False)
        elif show_impulse_response:
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            ax[0].plot(t_impulse, impulse, 'b-', label='Impulse')
            ax[0].plot(t_impulse, response, 'g-',
                       linewidth=2, label='Response')
            ax[0].set_title(
                'Impulse and Response of Digital Filter', fontsize=20)
            ax[0].set_xlabel('Time [sec]', fontsize=16)
            ax[0].set_ylabel('Amplitude', fontsize=16)
            ax[0].grid(which='both', axis='both')
            ax[0].legend(loc='best', fontsize=12, shadow=True)
            plt.tight_layout()
            plt.show(block=False)
        else:
            pass

        # if use_qt and (show_bodeplot or show_impulse_response):
        #     plt.show(block=True)
        #     matplotlib.use('Agg')  # reset to default backend
        # else:
        #     plt.show(block=False)  # do not block the code execution

        if apply_filter == True and in_signal is not None:
            # (Defualt) Apply filter forward and backward to eliminate phase shift and return the filtered signal
            if filter_mode == 'ba':
                out_signal = signal.filtfilt(b, a, in_signal)
            elif filter_mode == 'sos':
                out_signal = signal.sosfiltfilt(sos, in_signal)
            return out_signal
        # If apply_filter is set to False, return the filter coefficients
        else:
            if filter_mode == 'ba':
                return b, a
            else:
                return sos

    except Exception as e:
        err = ''.join("Error in digital_filter: ", e)
        raise Exception(err)
