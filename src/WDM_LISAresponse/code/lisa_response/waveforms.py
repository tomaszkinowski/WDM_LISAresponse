import numpy as np
from scipy.signal.windows import tukey

def waveform_polarisations(A_plus, A_cross, phase):
    """
    This is for a sine wave signal.

    Parameters
    ----------
    A_plus : float
        plus polarisation amplitude [strain]
    A_cross : float
        cross polarisation amplitude [strain]
    phase : ndarray [rad]
        phase of the gravitational wave phi(t)
    

    Returns
    -------
    h_plus : ndarray
        plus polarised signal, shaped like time.
    h_cross : ndarray
        cross polarised signal, shaped like time.
    """
    h_plus = A_plus * np.cos(phase)
    h_cross = A_cross * np.sin(phase)
    return h_plus, h_cross

def generate_windowed_padded_waveform_monochromatic(A_plus, A_cross, f, phi_0, dt, t0, N, alpha, tukey_factor=1.0):
    """
    Generates an monochromatic input waveform with tukey window and padding.

    Parameters
    ----------
    A_plus : float
        plus polarisation amplitude [strain]
    A_cross : float
        cross polarisation amplitude [strain]
    f : float
        GW frequency [Hz]
    phi_0 : float
        phase angle [rad]
    dt : float
        separation of time variables
    t0 : float
        initial start time for data stream 
    N : int
        number of data points of the waveform
    alpha : float
        tukey window parameter 
    tukey_factor : float, optional
        Multiplier for the padding length relative to the Tukey-affected
        half-segment. Padding on each side is
        int(tukey_factor * alpha * 0.5 * N). Default is 1.0.

    Returns
    -------
    times_pad : ndarray
        padded time series.
    w_h_plus_pad : ndarray
        padded and windowed plus polarised signal, shaped like times_pad.
    w_h_cross_pad : ndarray
        padded and windowed cross polarised signal, shaped like times_pad.
    """
    
    times = np.arange(t0, t0+N*dt, dt) 

    phase = 2.0*np.pi * f * times + phi_0

    h_plus_td, h_cross_td = waveform_polarisations(A_plus, A_cross, phase)

    N_tukey = len(times)
    w = tukey(N_tukey, alpha=alpha)
    
    w_h_plus_td = w* h_plus_td
    w_h_cross_td = w* h_cross_td

    pad_half = max(0, int(tukey_factor * alpha * 0.5 * N))
    P_left = pad_half
    P_right = pad_half

    w_h_plus_td_pad = np.pad(w_h_plus_td, (P_left, P_right), mode="constant", constant_values=0.0)
    w_h_cross_td_pad = np.pad(w_h_cross_td, (P_left, P_right), mode="constant", constant_values=0.0)

    times_left = times[0] - dt*np.arange(P_left, 0, -1)
    times_right = times[-1] + dt*np.arange(1, P_right+1)
    times_pad = np.concatenate([times_left, times, times_right])
    N_pad = len(times_pad)
    print("The padded length is:", N_pad)

    return times_pad, w_h_plus_td_pad, w_h_cross_td_pad

def generate_windowed_padded_waveform_linear_freq(A_plus, A_cross, f_start, f_end, phi_0, dt, t0, N, alpha, tukey_factor=1.0):
    """
    Generates an input sine waveform with changing frequency from f_start to f_end and adds tukey window and padding.

    Parameters
    ----------
    A_plus : float
        plus polarisation amplitude [strain]
    A_cross : float
        cross polarisation amplitude [strain]
    f_start : float
        starting GW frequency [Hz]
    f_end : float
        ending GW frequency [Hz]
    phi_0 : float
        phase angle [rad]
    dt : float
        separation of time variables
    t0 : float
        initial start time for data stream 
    N : int
        number of data points of the waveform
    alpha : float
        tukey window parameter 
    tukey_factor : float, optional
        Multiplier for the padding length relative to the Tukey-affected
        half-segment. Padding on each side is
        int(tukey_factor * alpha * 0.5 * N). Default is 1.0.

    Returns
    -------
    times_pad : ndarray
        padded time series.
    w_h_plus_pad : ndarray
        padded and windowed plus polarised signal, shaped like times_pad.
    w_h_cross_pad : ndarray
        padded and windowed cross polarised signal, shaped like times_pad.
    """
    
    times = np.arange(t0, t0+N*dt, dt) 

    times = t0 + dt*np.arange(N)
    t1 = times[-1]
    k = (f_end - f_start) / (t1 - t0)          # Hz/s

    tau = times - t0                           # time since start
    phase = phi_0 + 2*np.pi*(f_start*tau + 0.5*k*tau**2)


    h_plus_td, h_cross_td = waveform_polarisations(A_plus, A_cross, phase)

    N_tukey = len(times)
    w = tukey(N_tukey, alpha=alpha)
    
    w_h_plus_td = w* h_plus_td
    w_h_cross_td = w* h_cross_td

    pad_half = max(0, int(tukey_factor * alpha * 0.5 * N))
    P_left = pad_half
    P_right = pad_half

    w_h_plus_td_pad = np.pad(w_h_plus_td, (P_left, P_right), mode="constant", constant_values=0.0)
    w_h_cross_td_pad = np.pad(w_h_cross_td, (P_left, P_right), mode="constant", constant_values=0.0)

    times_left = times[0] - dt*np.arange(P_left, 0, -1)
    times_right = times[-1] + dt*np.arange(1, P_right+1)
    times_pad = np.concatenate([times_left, times, times_right])
    N_pad = len(times_pad)
    print("The padded length is:", N_pad)

    return times_pad, w_h_plus_td_pad, w_h_cross_td_pad