import numpy as np
from scipy.signal.windows import tukey

def waveform_polarisations(time, A_plus, A_cross, f, phi_0):
    """
    This is for a sine wave signal.

    Parameters
    ----------
    time : ndarray
        time in the solar system barycenter [s]
    A_plus : float
        plus polarisation amplitude [strain]
    A_cross : float
        cross polarisation amplitude [strain]
    f : float
        GW frequency [Hz]
    phi_0 : float
        phase angle [rad]

    Returns
    -------
    h_plus : ndarray
        plus polarised signal, shaped like time.
    h_cross : ndarray
        cross polarised signal, shaped like time.
    """
    h_plus = A_plus * np.cos(2*np.pi*f*time+phi_0)
    h_cross = A_cross * np.sin(2*np.pi*f*time+phi_0)
    return h_plus, h_cross

def generate_windowed_padded_waveform_monochromatic(A_plus, A_cross, f, phi_0, dt, t0, N, alpha):
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
    h_plus_td, h_cross_td = waveform_polarisations(times, A_plus, A_cross, f, phi_0)

    N_tukey = len(times)
    w = tukey(N_tukey, alpha=alpha)
    
    w_h_plus_td = w* h_plus_td
    w_h_cross_td = w* h_cross_td

    P_left =  int(alpha*0.5 * N)
    P_right =  int(alpha*0.5 * N)   

    w_h_plus_td_pad = np.pad(w_h_plus_td, (P_left, P_right), mode="constant", constant_values=0.0)
    w_h_cross_td_pad = np.pad(w_h_cross_td, (P_left, P_right), mode="constant", constant_values=0.0)

    times_left = times[0] - dt*np.arange(P_left, 0, -1)
    times_right = times[-1] + dt*np.arange(1, P_right+1)
    times_pad = np.concatenate([times_left, times, times_right])
    N_pad = len(times_pad)
    print("The padded length is:", N_pad)

    return times_pad, w_h_plus_td_pad, w_h_cross_td_pad