"""Waveform generation, windowing and padding helpers."""

import numpy as np
from scipy.signal.windows import tukey
from few.utils.constants import YRSID_SI
from typing import Dict, Optional, Any
from ..utils.utils import wdm_tail_padding_info


def _get_waveform_models():
    """Return the registry mapping waveform model names to generator functions.

    Returns
    -------
    dict[str, callable]
        Mapping used by ``generate_windowed_padded_waveform`` to dispatch to
        the selected waveform generator.
    """
    return {
        "monochromatic": generate_waveform_monochromatic,
        "linear_freq": generate_waveform_linear_freq,
        "emri": generate_waveform_emri,
    }


def _get_window_functions():
    """Return the registry mapping window names to windowing functions.

    Returns
    -------
    dict[str, callable]
        Mapping used to apply the selected window function to plus/cross
        polarizations.
    """
    return {
        "tukey": apply_tukey_window,
    }


def generate_windowed_padded_waveform(
    waveform_model: str,
    t0: float,
    dt: float,
    T_obs: Optional[float] = None,
    N: Optional[int] = None,
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    window_function: Optional[str] = None,
    window_kwargs: Optional[Dict[str, Any]] = None,
    wdm_Nf: Optional[int] = None,
    return_padding_info: bool = False,
):
    """
    Master function for generating various waveform models. 
    The function also pads each signal by a multiple of the data affected by the window.

    Parameters
    ----------
    waveform_model : string
        Name of the waveform model to generate 
    t0 : float [seconds]
        starting time for the signal 
    dt : float [sec]
        time step for the generated waveform
    T_obs : float [years]
        observational time in years 
    N : int
        number of data samples 
    model_kwargs : dict
        parameters of the particular model
    window function : string 
        parameter determining the type of window function
    window_kwargs : dict
        parameters of the window function application 
    wdm_Nf : int or None
        If provided, add right-tail zero padding so total length becomes a
        multiple of ``2*Nf`` so ``Nt`` is always even for WDM.
    return_padding_info : bool
        If true, returns an extra dictionary containing padding metadata.
    
    Returns
    -------
    times : ndarray
        time series generated using dt and either N, or T_obs in seconds
    h_plus_window_pad : ndarray
        windowed and padded plus polarised signal, shaped like time.
    h_cross_window_pad : ndarray
        windowed and padded cross polarised signal, shaped like time.
    """

    #generate the time series for intial input 

    if dt <= 0:
        raise ValueError("dt must be positive.")
    
    if T_obs is not None and T_obs <= 0:
        raise ValueError("T_obs must be positive.")

    if T_obs is None and N is None:
        raise ValueError("Either T_obs or N must be provided.")

    if T_obs is not None and N is not None:
        raise ValueError("Provide only one of T_obs or N, not both.")

    if T_obs is not None:
        T_sec = T_obs * YRSID_SI
        N = int(np.floor(T_sec / dt))

    if N is None or N < 2:
        raise ValueError("At least two samples are required, so N must be >= 2.")

    times = t0 + np.arange(N) * dt
    #generate the waveform
    
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    waveform_models = _get_waveform_models()
    try:
        model_fn = waveform_models[waveform_model]
    except KeyError:
        raise ValueError(f"Waveform model '{waveform_model}' is not available.")
    
    h_plus, h_cross = model_fn(times=times, **model_kwargs)

    #apply windowing and zero pad

    window_kwargs = {} if window_kwargs is None else dict(window_kwargs)

    if window_function is not None:
        window_functions = _get_window_functions()
        try:
            window_fn = window_functions[window_function]
        except KeyError:
            raise ValueError(f"Window function '{window_function}' is not available.")

        h_plus_window, h_cross_window = window_fn(h_plus, h_cross, **window_kwargs)
    else:
        h_plus_window, h_cross_window = h_plus, h_cross
    
    pad_left, pad_right = get_window_padding(window_function, len(times), window_kwargs)

    times_pad, h_plus_window_pad, h_cross_window_pad = apply_zero_padding(
        times,
        h_plus_window,
        h_cross_window,
        pad_left,
        pad_right,
    )

    pad_info = None
    if return_padding_info:
        pad_info = {
            "original_length": int(len(times)),
            "window_pad_left": int(pad_left),
            "window_pad_right": int(pad_right),
            "window_function": window_function,
        }

    if wdm_Nf is not None:
        tail_info = wdm_tail_padding_info(
            len(times_pad),
            int(wdm_Nf),
        )
        tail_right = int(tail_info["pad_right"])

        if tail_right > 0:
            times_pad, h_plus_window_pad, h_cross_window_pad = apply_zero_padding(
                times_pad,
                h_plus_window_pad,
                h_cross_window_pad,
                0,
                tail_right,
            )

        if return_padding_info:
            pad_info.update(
                {
                    "wdm_Nf": int(wdm_Nf),
                    "wdm_tail_pad_right": tail_right,
                    "wdm_target_length": int(tail_info["target_length"]),
                    "wdm_Nt": int(tail_info["Nt"]),
                }
            )
    else:
        if return_padding_info:
            pad_info.update(
                {
                    "wdm_Nf": None,
                    "wdm_tail_pad_right": 0,
                    "wdm_target_length": int(len(times_pad)),
                    "wdm_Nt": None,
                }
            )

    if return_padding_info:
        return times_pad, h_plus_window_pad, h_cross_window_pad, pad_info

    return times_pad, h_plus_window_pad, h_cross_window_pad

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

def generate_waveform_monochromatic(times, A_plus, A_cross, f, phi_0):
    """
    Generates an monochromatic input waveform with tukey window and padding.

    Parameters
    ----------
    times : ndarray 
        numpy array of times for sampling
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
        plus polarised signal, shaped like times
    h_cross : ndarray
        cross polarised signal, shaped like times
    """
    phase = 2.0*np.pi * f * times + phi_0

    h_plus, h_cross = waveform_polarisations(A_plus, A_cross, phase)

    return h_plus, h_cross

def generate_waveform_linear_freq(times, A_plus, A_cross, f_start, f_end, phi_0):
    """
    Generates an input sine waveform with changing frequency from f_start to f_end and adds tukey window and padding.

    Parameters
    ----------
    times : ndarray 
        numpy array of times for sampling
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

    Returns
    -------
    h_plus : ndarray
        plus polarised signal, shaped like times
    h_cross : ndarray
        cross polarised signal, shaped like times
    """

    if len(times) < 2 or times[-1] == times[0]:
        raise ValueError("linear_freq waveform requires at least two distinct time samples.")
    
    t1 = times[-1]
    t0 = times[0]
    k = (f_end - f_start) / (t1 - t0)          # Hz/s

    tau = times - times[0]                           # time since start
    phase = phi_0 + 2*np.pi*(f_start*tau + 0.5*k*tau**2)

    h_plus, h_cross = waveform_polarisations(A_plus, A_cross, phase)

    return h_plus, h_cross

def generate_waveform_emri(
    times,
    td_gen,
    m1,
    m2,
    a,
    p0,
    e0,
    x0,
    dist,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    eps,
    qS=None,
    phiS=None,
    theta=None,
    phi=None,
):
    """
    Generate a FEW EMRI waveform on the sampling implied by `times`.

    Parameters
    ----------
    times : ndarray
        Uniformly sampled time array.
    td_gen : FEW waveform generator
        Example: GenerateEMRIWaveform(...)
    ...
    Returns
    -------
    h_plus : ndarray
    h_cross : ndarray
    """
    times = np.asarray(times)

    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a 1D array with at least two samples.")

    dt = times[1] - times[0]
    if not np.allclose(np.diff(times), dt):
        raise ValueError("EMRI waveform generation requires uniformly sampled times.")

    qS_eff = theta if theta is not None else qS
    phiS_eff = phi if phi is not None else phiS

    if qS_eff is None or phiS_eff is None:
        raise ValueError("Provide sky location as either (qS, phiS) or (theta, phi).")

    T_obs = len(times) * dt / YRSID_SI

    waveform_kwargs = {
        "T": T_obs,
        "dt": dt,
        "mode_selection_threshold": eps,
    }

    parameters = [
        m1, m2, a, p0, e0, x0,
        dist, qS_eff, phiS_eff, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
    ]

    waveform_td = td_gen(*parameters, **waveform_kwargs)

    h_plus = np.asarray(waveform_td[0])
    h_cross = np.asarray(waveform_td[1])

    N = len(times)
    h_plus = h_plus[:N]
    h_cross = h_cross[:N]

    if len(h_plus) < N:
        pad = N - len(h_plus)
        h_plus = np.pad(h_plus, (0, pad), mode="constant")
        h_cross = np.pad(h_cross, (0, pad), mode="constant")

    return h_plus, h_cross


def apply_tukey_window(h_plus, h_cross, alpha, **_):
    """Apply a Tukey window to plus/cross waveform polarizations.

    Parameters
    ----------
    h_plus : ndarray
        Plus polarization time series.
    h_cross : ndarray
        Cross polarization time series.
    alpha : float
        Tukey window shape parameter in ``[0, 1]``.
    **_ : dict
        Additional unused keyword arguments accepted for compatibility with
        generic window-calling paths.

    Returns
    -------
    tuple[ndarray, ndarray]
        Windowed ``(h_plus, h_cross)`` arrays.
    """
    w = tukey(len(h_plus), alpha=alpha)
    return w * h_plus, w * h_cross

def get_window_padding(window_function: Optional[str], n_samples: int, window_kwargs: Dict[str, Any]) -> tuple[int, int]:
    """
    Return left/right padding widths implied by the chosen window.

    For Tukey, the physical taper affects alpha/2 of the data on each side,
    and tukey_factor scales how much zero padding to add relative to that.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")

    if window_function is None:
        return 0, 0

    if window_function == "tukey":
        alpha = float(window_kwargs.get("alpha", 0.0))
        tukey_factor = float(window_kwargs.get("tukey_factor", 1.0))

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Tukey alpha must lie in [0, 1], got {alpha}.")

        if tukey_factor < 0:
            raise ValueError(f"tukey_factor must be non-negative, got {tukey_factor}.")

        # Number of samples in one taper half, scaled by tukey_factor
        pad_half = int(np.ceil(tukey_factor * alpha * 0.5 * n_samples))
        return pad_half, pad_half

    return 0, 0


def apply_zero_padding(times, h_plus, h_cross, pad_left: int, pad_right: int):
    """
    Apply zero padding to waveform polarizations and extend the time array consistently.
    """
    times = np.asarray(times)
    h_plus = np.asarray(h_plus)
    h_cross = np.asarray(h_cross)

    if times.ndim != 1 or h_plus.ndim != 1 or h_cross.ndim != 1:
        raise ValueError("times, h_plus, and h_cross must all be 1D arrays.")

    if not (len(times) == len(h_plus) == len(h_cross)):
        raise ValueError("times, h_plus, and h_cross must have the same length.")

    if len(times) < 2:
        raise ValueError("Need at least two time samples to infer dt for padding.")

    if pad_left < 0 or pad_right < 0:
        raise ValueError("pad_left and pad_right must be non-negative.")

    dt = times[1] - times[0]
    if not np.allclose(np.diff(times), dt):
        raise ValueError("times must be uniformly sampled to apply zero padding.")

    h_plus_pad = np.pad(h_plus, (pad_left, pad_right), mode="constant", constant_values=0.0)
    h_cross_pad = np.pad(h_cross, (pad_left, pad_right), mode="constant", constant_values=0.0)

    if pad_left == 0 and pad_right == 0:
        return times.copy(), h_plus_pad, h_cross_pad

    times_left = times[0] - dt * np.arange(pad_left, 0, -1)
    times_right = times[-1] + dt * np.arange(1, pad_right + 1)
    times_pad = np.concatenate((times_left, times, times_right))

    return times_pad, h_plus_pad, h_cross_pad


def generate_windowed_padded_waveform_monochromatic(
    A_plus,
    A_cross,
    f,
    phi_0,
    dt,
    t0,
    N,
    alpha,
    tukey_factor=1.0,
    wdm_Nf=None,
    return_padding_info=False,
):
    """Compatibility wrapper for legacy monochromatic notebook workflows."""
    return generate_windowed_padded_waveform(
        waveform_model="monochromatic",
        t0=t0,
        dt=dt,
        N=N,
        model_kwargs={
            "A_plus": A_plus,
            "A_cross": A_cross,
            "f": f,
            "phi_0": phi_0,
        },
        window_function="tukey",
        window_kwargs={
            "alpha": alpha,
            "tukey_factor": tukey_factor,
        },
        wdm_Nf=wdm_Nf,
        return_padding_info=return_padding_info,
    )


def generate_windowed_padded_waveform_linear_freq(
    A_plus,
    A_cross,
    f_start,
    f_end,
    phi_0,
    dt,
    t0,
    N,
    alpha,
    tukey_factor=1.0,
    wdm_Nf=None,
    return_padding_info=False,
):
    """Compatibility wrapper for legacy linear-frequency notebook workflows."""
    return generate_windowed_padded_waveform(
        waveform_model="linear_freq",
        t0=t0,
        dt=dt,
        N=N,
        model_kwargs={
            "A_plus": A_plus,
            "A_cross": A_cross,
            "f_start": f_start,
            "f_end": f_end,
            "phi_0": phi_0,
        },
        window_function="tukey",
        window_kwargs={
            "alpha": alpha,
            "tukey_factor": tukey_factor,
        },
        wdm_Nf=wdm_Nf,
        return_padding_info=return_padding_info,
    )


def generate_windowed_padded_waveform_emri(
    td_gen,
    m1,
    m2,
    a,
    p0,
    e0,
    x0,
    dist,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    eps,
    dt,
    t0,
    N,
    alpha,
    tukey_factor=1.0,
    qS=None,
    phiS=None,
    theta=None,
    phi=None,
    wdm_Nf=None,
    return_padding_info=False,
):
    """Compatibility wrapper for legacy EMRI notebook workflows."""
    return generate_windowed_padded_waveform(
        waveform_model="emri",
        t0=t0,
        dt=dt,
        N=N,
        model_kwargs={
            "td_gen": td_gen,
            "m1": m1,
            "m2": m2,
            "a": a,
            "p0": p0,
            "e0": e0,
            "x0": x0,
            "dist": dist,
            "qK": qK,
            "phiK": phiK,
            "Phi_phi0": Phi_phi0,
            "Phi_theta0": Phi_theta0,
            "Phi_r0": Phi_r0,
            "eps": eps,
            "qS": qS,
            "phiS": phiS,
            "theta": theta,
            "phi": phi,
        },
        window_function="tukey",
        window_kwargs={
            "alpha": alpha,
            "tukey_factor": tukey_factor,
        },
        wdm_Nf=wdm_Nf,
        return_padding_info=return_padding_info,
    )