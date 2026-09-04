"""Waveform generation and windowing.

Padding lives in ``utils.utils`` and is applied by the caller: it is a
property of the grid the record will be analysed on, not of the waveform, and
the two response paths need different amounts of it. Use ``utils.pad_for_wdm``
for the wavelet path and ``utils.apply_zero_padding`` for the time-domain one.
"""

import numpy as np
from scipy.signal.windows import tukey
from lisaconstants import SIDEREALYEAR_J2000DAY
from typing import Dict, Optional, Any

# Sidereal year in seconds. Matches ``few.utils.constants.YRSID_SI`` to within
# one ULP, so FEW stays an optional (dev-only) dependency: EMRI generators are
# passed in by the caller rather than constructed here.
YRSID_SI = SIDEREALYEAR_J2000DAY * 86400.0


def _get_waveform_models():
    """Return the registry mapping waveform model names to generator functions.

    Returns
    -------
    dict[str, callable]
        Mapping used by ``generate_windowed_waveform`` to dispatch to the
        selected waveform generator.
    """
    return {
        "monochromatic": generate_waveform_monochromatic,
        "linear_freq": generate_waveform_linear_freq,
        "emri": generate_waveform_emri,
    }


def _get_window_functions():
    """Return the registry mapping window names to windowing functions.

    Windows taper; they do not pad. Padding is a separate, explicit step in
    ``utils.utils``, so nothing here implies any.

    Returns
    -------
    dict[str, callable]
        Mapping used to apply the selected window function to plus/cross
        polarizations.
    """
    return {
        "tukey": apply_tukey_window,
    }


def generate_windowed_waveform(
    waveform_model: str,
    t0: float,
    dt: float,
    T_obs: Optional[float] = None,
    N: Optional[int] = None,
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    window_function: Optional[str] = None,
    window_kwargs: Optional[Dict[str, Any]] = None,
):
    r"""
    Generate a waveform on a uniform grid and optionally taper it.

    Dispatches to one of the registered models, evaluates it on
    :math:`t_k = t_0 + k \delta t`, and applies a window if asked. Nothing
    here pads: padding belongs to the grid the record will be analysed on
    rather than to the waveform, and the two response paths need different
    amounts of it. Follow this with ``utils.pad_for_wdm`` for the wavelet
    path, or ``utils.apply_zero_padding`` for the time-domain one.

    Parameters
    ----------
    waveform_model : string
        Name of the waveform model to generate: ``"monochromatic"``,
        ``"linear_freq"`` or ``"emri"``.
    t0 : float [seconds]
        starting time for the signal
    dt : float [sec]
        time step for the generated waveform
    T_obs : float [years]
        observational time in years. Give this or ``N``, not both.
    N : int
        number of data samples. Give this or ``T_obs``, not both.
    model_kwargs : dict
        parameters of the particular model
    window_function : string
        parameter determining the type of window function, or ``None``
    window_kwargs : dict
        parameters of the window function application, for example ``alpha``
        for the Tukey taper

    Returns
    -------
    times : ndarray
        time series generated using dt and either N, or T_obs in seconds
    h_plus : ndarray
        windowed plus polarised signal, shaped like times.
    h_cross : ndarray
        windowed cross polarised signal, shaped like times.
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

        h_plus, h_cross = window_fn(h_plus, h_cross, **window_kwargs)

    return times, h_plus, h_cross

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
    Generates a monochromatic input waveform.

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
    Generates an input sine waveform sweeping linearly from f_start to f_end.

    The chirp rate is fixed so the instantaneous frequency reaches ``f_end`` at
    the *last sample*, not at any nominal observation time.

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

    The result is forced to ``len(times)``, and both directions of that are
    expected rather than exceptional. FEW returns a *longer* series when its
    ``pad_output``/``odd_len`` settings overshoot the requested duration, and a
    *shorter* one when the inspiral plunges before ``T_obs`` is up - a physical
    outcome, not an error. Long output is truncated, short output is zero
    filled, which is the correct continuation in both cases.

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
    # scipy clips alpha silently, turning a typo into a rectangular or Hann
    # window rather than an error.
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"Tukey alpha must lie in [0, 1], got {alpha}.")

    w = tukey(len(h_plus), alpha=alpha)
    return w * h_plus, w * h_cross
