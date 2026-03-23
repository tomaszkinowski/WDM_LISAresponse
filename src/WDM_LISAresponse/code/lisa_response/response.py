"""
LISA response calculations for WDM_LISAresponse
"""

import numpy as np 
from lisaconstants import c
from scipy.interpolate import interp1d
import WDM
from WDM.code.time_delay_filters.time_shift_fast import wdm_time_shift_variable, wdm_time_shift_variable_batch
from numpy.fft import fft, fftfreq
from .response_sub import (
    antenna_pattern_factors,
    build_sky_basis,
    _compute_ltt_iterative,
    rhat, 
    gnm_at_times_grid_np_interp,
    generate_time_shifts,
    wdm_pad_inputs_right,
)
# Add your LISA response functions here

def one_arm_response_td(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, num_int_points=10):
    """
    Evaluate the time-domain one-arm response for a single directed link.

    Parameters
    ----------
    times : ndarray
        Sample times of the LISA response. Each entry is the emission time at
        the sending spacecraft.
    s : int
        Sending spacecraft index.
    r : int
        Receiving spacecraft index.
    orbits : object
        Orbit object used to evaluate spacecraft positions.
    theta : float
        Source polar angle in radians.
    phi : float
        Source azimuthal angle in radians.
    td_waveform_plus : ndarray
        Plus-polarization waveform sampled on ``times``.
    td_waveform_cross : ndarray
        Cross-polarization waveform sampled on ``times``.
    num_int_points : int
        Number of quadrature points used in the trapezoidal integration.

    Returns
    -------
    ndarray
        Time-domain one-arm response sampled on ``times``.
    """
    sky_basis = build_sky_basis(theta, phi)
    x_s = orbits.compute_position(times, sc=[s])[:,0,:]

    l_sr, x_r_rec = _compute_ltt_iterative(
        times,
        s,
        r,
        orbits,
        x_s=x_s,
        return_receiver_position=True,
    )
    rhat_sr = rhat(times, s, r, orbits, x_s=x_s, x_r_rec=x_r_rec)

    t_rec = times + l_sr

    Omega = sky_basis.Omega

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)

    xi_s = times - np.einsum('x,tx->t', Omega, x_s/c)
    xi_r = t_rec - np.einsum('x,tx->t', Omega, x_r_rec/c)

    xi = np.linspace(xi_s, xi_r, num_int_points) # (num_int_points, num_times)

    h_plus_interp = interp1d(times, td_waveform_plus, kind="cubic", bounds_error=False, fill_value=0.0) #change back to extrapolate
    h_cross_interp = interp1d(times, td_waveform_cross, kind="cubic", bounds_error=False, fill_value=0.0)
    h_plus = h_plus_interp(xi)
    h_cross = h_cross_interp(xi)

    H_plus = np.trapezoid(h_plus, x=xi, axis=0)
    H_cross = np.trapezoid(h_cross, x=xi, axis=0)

    F_plus, F_cross = antenna_pattern_factors(rhat_sr, sky_basis)
    dl_sr = 0.5 * (1-Om_dot_rhat)**(-1) * (F_plus * H_plus + F_cross * H_cross)

    return c* dl_sr

def one_arm_response_freq_domain(
    times, s, r, orbits, theta, phi,
    td_waveform_plus, td_waveform_cross,
    chunk_size=512
):
    """
    Evaluate the frequency-domain one-arm response with chunked frequency work.

    This implementation uses direct scalar antenna factors rather than explicit
    polarization-tensor contractions and processes the frequency axis in
    chunks to limit peak memory use.
    """
    times = np.asarray(times)
    Nt = times.size
    sky_basis = build_sky_basis(theta, phi)

    # Assume uniform sampling
    dt = times[1] - times[0]

    # Geometry (still computed here; biggest speedup across many calls is caching these externally)
    x_s = orbits.compute_position(times, sc=[s])[:, 0, :]          # (Nt,3)
    l_sr = _compute_ltt_iterative(times, s, r, orbits)              # (Nt,)
    rhat_sr = rhat(times, s, r, orbits)                             # (Nt,3)

    Omega = sky_basis.Omega

    Om_dot_rhat = np.einsum('i,ti->t', Omega, rhat_sr)              # (Nt,)
    xi_s = times - np.einsum('i,ti->t', Omega, x_s / c)             # (Nt,)

    # FFT of waveform polarisations
    H_plus = fft(td_waveform_plus)
    H_cross = fft(td_waveform_cross)

    h_freq = fftfreq(Nt, dt)
    omega = 2.0 * np.pi * h_freq                                    # (Nt,)

    # Antenna scalars: F_+(t) = rhat^T e_+ rhat, same for cross
    F_plus, F_cross = antenna_pattern_factors(rhat_sr, sky_basis)   # (Nt,)

    # Combine constant phase shift into one term
    phase_t = (xi_s - times[0])                                     # (Nt,)

    # Accumulate inverse-sum in chunks to reduce memory and often improve speed
    acc = np.zeros(Nt, dtype=complex)
    one_minus = (1.0 - Om_dot_rhat)                                 # (Nt,)

    for j0 in range(0, Nt, chunk_size):
        j1 = min(j0 + chunk_size, Nt)

        w = omega[j0:j1]                                            # (M,)
        Hp = H_plus[j0:j1]
        Hc = H_cross[j0:j1]

        # transfer_func_arg = 0.5 * omega * l_sr * (1 - Om·rhat)
        arg = 0.5 * (l_sr[:, None] * one_minus[:, None]) * w[None, :]  # (Nt,M)

        # transfer function: sinc(arg/pi) * exp(i*arg)
        tf = np.sinc(arg / np.pi) * np.exp(1j * arg)                # (Nt,M)

        # scalar contraction: 0.5 * tf * (Hp*F_plus + Hc*F_cross)
        resp = 0.5 * tf * (F_plus[:, None] * Hp[None, :] + F_cross[:, None] * Hc[None, :])  # (Nt,M)

        # exp(i w * (xi_s - t0))
        ph = np.exp(1j * phase_t[:, None] * w[None, :])             # (Nt,M)

        acc += np.sum(ph * resp, axis=1)                            # (Nt,)

    dl_sr = c * l_sr * (acc / Nt)
    return dl_sr


def _resolve_wdm_q_value(N, Nf, q=None):
    """Resolve WDM truncation parameter q with an auto-max default.

    When ``q`` is ``None``, use the largest value supported by the current
    WDM tiling, i.e. ``floor(Nt/2)`` where ``Nt = floor(N / Nf)``.
    """
    Nt = int(N // Nf)
    if Nt < 2:
        raise ValueError(f"Invalid WDM tiling for q resolution: Nt={Nt} (from N={N}, Nf={Nf}).")

    q_max = max(1, Nt // 2)
    if q is None:
        return q_max

    q_val = int(q)
    if q_val < 1:
        raise ValueError(f"q must be >= 1, got {q_val}.")
    if q_val > q_max:
        raise ValueError(f"q={q_val} exceeds max allowed q={q_max} for Nt={Nt}.")
    return q_val



def one_arm_response_wdm(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, Nf=500, gnm_method="cubic", gnm_wrap=True, d = 4, q=None, calc_m0=True):
    """
    Evaluate the one-arm WDM response on the native sample grid.

    Parameters
    ----------
    times : ndarray
        Sample times of the LISA response. Each entry is the emission time at
        the sending spacecraft.
    s : int
        Sending spacecraft index.
    r : int
        Receiving spacecraft index.
    orbits : object
        Orbit object used to evaluate spacecraft positions.
    theta : float
        Source polar angle in radians.
    phi : float
        Source azimuthal angle in radians.
    td_waveform_plus : ndarray
        Plus-polarization waveform sampled on ``times``.
    td_waveform_cross : ndarray
        Cross-polarization waveform sampled on ``times``.
    Nf : int
        Number of basis elements in the frequency dimension.
    gnm_method : str, optional
        Interpolation method used for the WDM basis evaluation.
    gnm_wrap : bool, optional
        Boundary handling flag passed to the WDM basis interpolator.
    d : int, optional
        WDM oversampling/filter parameter.
    q : int or None, optional
        WDM truncation parameter. When ``None``, it is set to the maximum
        allowed value for the current tiling.
    calc_m0 : bool, optional
        Whether to include the ``m=0`` WDM component when constructing the
        transform.
    Returns
    -------
    ndarray
        One-arm WDM response evaluated on ``times``. The sky basis is built
        once and reused for both the phase geometry and antenna factors.
    """
    times = np.asarray(times, dtype=float)
    N_in = len(times)
    sky_basis = build_sky_basis(theta, phi)

    times_wdm, td_plus_wdm, td_cross_wdm, _ = wdm_pad_inputs_right(
        times,
        td_waveform_plus,
        td_waveform_cross,
        Nf,
    )

    N = len(times_wdm)
    dt = times_wdm[1] - times_wdm[0]

    q_val = _resolve_wdm_q_value(N, Nf, q=q)

    wdm = WDM.WDM.WDM_transform(dt=dt, Nf=Nf, N=N, q=q_val, calc_m0=calc_m0, d = d)
    #print(wdm)


    #print(f"How much padding was added: {(N-current_length)*wdm.dt}")

    w_t_plus = wdm.dwt(td_plus_wdm)
    w_t_cross = wdm.dwt(td_cross_wdm)

    #basic implementation of wdm transform 

    x_s = orbits.compute_position(times, sc=[s])[:,0,:]

    l_sr, x_r_rec = _compute_ltt_iterative(
        times,
        s,
        r,
        orbits,
        x_s=x_s,
        return_receiver_position=True,
    )
    rhat_sr = rhat(times, s, r, orbits, x_s=x_s, x_r_rec=x_r_rec)

    Omega = sky_basis.Omega

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)
        
    xi_s = times - np.einsum('x,tx->t', Omega, x_s/c)

    freq_vals = np.arange(0, wdm.Nf) * wdm.dF

    tau_s = l_sr * (1 - Om_dot_rhat) * 0.5
    sinc_arg = 2.0 * np.pi * freq_vals[np.newaxis, :] * tau_s[:, np.newaxis]

    F_plus, F_cross = antenna_pattern_factors(rhat_sr, sky_basis)

    n_grid, m_grid = np.meshgrid(np.arange(wdm.Nt), np.arange(wdm.Nf), indexing="ij")

    vals = gnm_at_times_grid_np_interp(wdm, n_grid, m_grid, xi_s + tau_s - times[0], method=gnm_method, wrap=gnm_wrap)

    sum_nm_plus = np.sinc(sinc_arg / np.pi)[:, np.newaxis, :] * w_t_plus[np.newaxis, :, :] * vals
    sum_nm_cross = np.sinc(sinc_arg / np.pi)[:, np.newaxis, :] * w_t_cross[np.newaxis, :, :] * vals

    dl_sr_wdm = 0.5 * c * l_sr * (
        F_plus * np.einsum("tnm->t", sum_nm_plus)
        + F_cross * np.einsum("tnm->t", sum_nm_cross)
    )

    return dl_sr_wdm[:N_in]


def _build_wdm_dT_context(times, td_waveform_plus, td_waveform_cross, Nf=500, d=4, m_offset=0.0, q=None, calc_m0=True):
    """Build reusable WDM objects and waveform projections for dT responses.

    Parameters
    ----------
    times : ndarray
        Uniformly sampled time grid.
    td_waveform_plus : ndarray
        Plus-polarization time series.
    td_waveform_cross : ndarray
        Cross-polarization time series.
    Nf : int, optional
        Number of WDM frequency bins.
    d : int, optional
        WDM oversampling/filter parameter.
    m_offset : float, optional
        Optional frequency-index offset used in some WDM conventions.
    q : int or None, optional
        WDM truncation parameter. When ``None``, it is set to the maximum
        allowed value for the current tiling.
    calc_m0 : bool, optional
        Whether to include the ``m=0`` WDM component when constructing the
        transform.
    Returns
    -------
    dict
        Dictionary containing precomputed WDM object, transformed waveforms,
        and grids used repeatedly by one-arm response calls.
    """
    times = np.asarray(times, dtype=float)
    times_wdm, td_plus_wdm, td_cross_wdm, _ = wdm_pad_inputs_right(
        times,
        td_waveform_plus,
        td_waveform_cross,
        Nf,
    )
    N = len(times_wdm)
    dt = times_wdm[1] - times_wdm[0]

    q_val = _resolve_wdm_q_value(N, Nf, q=q)

    wdm = WDM.WDM.WDM_transform(dt=dt, Nf=Nf, N=N, q=q_val, calc_m0=calc_m0, d=d, A_frac=0.25)
    w_t_plus = wdm.dwt(td_plus_wdm)
    w_t_cross = wdm.dwt(td_cross_wdm)

    freq_vals = (np.arange(wdm.Nf) + m_offset) * wdm.dF
    times_vals = np.arange(wdm.Nt) * wdm.dT
    int_times_vals = times[0] + times_vals

    return {
        "times": times,
        "wdm": wdm,
        "w_t_plus": w_t_plus,
        "w_t_cross": w_t_cross,
        "freq_vals": freq_vals,
        "int_times_vals": int_times_vals,
    }


def _one_arm_response_wdm_dT_from_context(
    context,
    s,
    r,
    orbits,
    theta,
    phi,
    return_td=False,
    gnm_method="cubic",
    gnm_wrap=True,
    L_trunc=None,
    variable_shift_kwargs=None,
    sky_basis=None,
):
    """Evaluate one-arm dT WDM response from precomputed context data.

    The coefficient-space time shift uses ``wdm_time_shift_variable`` and can
    be configured through ``variable_shift_kwargs`` to use exact or
    interpolated ``Tl/Tp`` construction. A precomputed ``sky_basis`` can be
    supplied to avoid rebuilding the source-direction basis.
    """
    times = context["times"]
    wdm = context["wdm"]
    shift_kwargs = {} if variable_shift_kwargs is None else dict(variable_shift_kwargs)

    xi_s, tau_s, w_xi, time_shift = _one_arm_wdm_dT_components_from_context(
        context,
        s,
        r,
        orbits,
        theta,
        phi,
        include_time_domain_geometry=bool(return_td),
        sky_basis=sky_basis,
    )

    w_sr = wdm_time_shift_variable(
        wdm,
        w_xi,
        -time_shift,
        Nf=wdm.Nf,
        delta_mode="target",
        L_trunc=L_trunc,
        **shift_kwargs,
    )

    if not return_td:
        return w_sr

    n_grid, m_grid = np.meshgrid(np.arange(wdm.Nt), np.arange(wdm.Nf), indexing="ij")
    tq = xi_s + tau_s - times[0]
    vals = gnm_at_times_grid_np_interp(wdm, n_grid, m_grid, tq, method=gnm_method, wrap=gnm_wrap)
    dl_sr = np.einsum("tnm,nm->t", vals, w_xi)
    return dl_sr, w_sr


def _one_arm_wdm_dT_components_from_context(
    context,
    s,
    r,
    orbits,
    theta,
    phi,
    include_time_domain_geometry=False,
    sky_basis=None,
):
    """Build unshifted one-arm WDM coefficients and time-shift vector for a link.

    When ``include_time_domain_geometry`` is false, the full-sample ``times``
    geometry needed only for ``return_td=True`` is skipped. In that case the
    returned ``xi_s`` and ``tau_s`` entries are ``None``. When ``sky_basis`` is
    provided, the same source-direction basis is reused for all projections.
    """
    times = context["times"]
    wdm = context["wdm"]
    w_t_plus = context["w_t_plus"]
    w_t_cross = context["w_t_cross"]
    freq_vals = context["freq_vals"]
    int_times_vals = context["int_times_vals"]

    if sky_basis is None:
        sky_basis = build_sky_basis(theta, phi)

    Omega = sky_basis.Omega
    x_s_dT = orbits.compute_position(int_times_vals, sc=[s])[:, 0, :]
    l_sr_dT, x_r_rec_dT = _compute_ltt_iterative(
        int_times_vals,
        s,
        r,
        orbits,
        x_s=x_s_dT,
        return_receiver_position=True,
    )
    rhat_sr_dT = rhat(int_times_vals, s, r, orbits, x_s=x_s_dT, x_r_rec=x_r_rec_dT)
    Om_dot_rhat_dT = rhat_sr_dT @ Omega
    tau_n = 0.5 * l_sr_dT * (1 - Om_dot_rhat_dT)

    if include_time_domain_geometry:
        x_s = orbits.compute_position(times, sc=[s])[:, 0, :]
        xi_s = times - (x_s @ (Omega / c))
        l_sr, x_r_rec = _compute_ltt_iterative(
            times,
            s,
            r,
            orbits,
            x_s=x_s,
            return_receiver_position=True,
        )
        rhat_sr = rhat(times, s, r, orbits, x_s=x_s, x_r_rec=x_r_rec)
        Om_dot_rhat = rhat_sr @ Omega
        tau_s = 0.5 * l_sr * (1 - Om_dot_rhat)
    else:
        xi_s = None
        tau_s = None

    proj_dT_plus, proj_dT_cross = antenna_pattern_factors(rhat_sr_dT, sky_basis)

    sinc_mat = np.sinc(2.0 * freq_vals[None, :] * tau_n[:, None])

    A_nm_plus = 0.5 * c * l_sr_dT[:, None] * proj_dT_plus[:, None] * sinc_mat
    A_nm_cross = 0.5 * c * l_sr_dT[:, None] * proj_dT_cross[:, None] * sinc_mat

    w_xi = A_nm_plus * w_t_plus + A_nm_cross * w_t_cross

    time_shift = generate_time_shifts(
        wdm.Nt,
        times[0],
        wdm.dT,
        theta,
        phi,
        s,
        r,
        orbits,
        sky_basis=sky_basis,
    )
    return xi_s, tau_s, w_xi, time_shift


def precompute_one_arm_wdm_dT_links(
    times,
    orbits,
    theta,
    phi,
    td_waveform_plus,
    td_waveform_cross,
    Nf=500,
    d=4,
    m_offset=0.0,
    L_trunc=None,
    links=None,
    cache=None,
    variable_shift_kwargs=None,
    q=None,
    calc_m0=True,
):
    """Precompute directed one-arm dT WDM responses and return them as a map.

    Parameters
    ----------
    times : ndarray
        Uniformly sampled time grid.
    orbits : object
        Orbit object used to evaluate spacecraft positions.
    theta : float
        Source polar angle in radians.
    phi : float
        Source azimuthal angle in radians.
    td_waveform_plus : ndarray
        Plus-polarization time-domain waveform.
    td_waveform_cross : ndarray
        Cross-polarization time-domain waveform.
    Nf : int, optional
        Number of WDM frequency bins.
    d : int, optional
        WDM oversampling/filter parameter.
    m_offset : float, optional
        Optional frequency-index offset.
    L_trunc : int or None, optional
        Time-shift truncation parameter forwarded to fast shifting.
    links : iterable of tuple[int, int] or None
        Directed links to compute. Defaults to the six LISA arms.
    cache : dict or None
        Optional cache dictionary for reuse across multiple TDI calls.
    variable_shift_kwargs : dict or None
        Optional keyword arguments forwarded to
        ``wdm_time_shift_variable_batch``. This can enable interpolated Tl/Tp
        construction for the one-arm coefficient-space shifts.
    q : int or None, optional
        WDM truncation parameter. When ``None``, it is set to the maximum
        allowed value for the current tiling.
    calc_m0 : bool, optional
        Whether to include the ``m=0`` WDM component when constructing the
        transform.
    Returns
    -------
    tuple
        ``(wdm, responses)`` where ``wdm`` is the WDM transform object and
        ``responses`` maps directed links ``(s, r)`` to WDM arrays.

    Notes
    -----
    ``L_trunc`` is forwarded to the shifter and limits the lag range used
    during assembly when provided. ``variable_shift_kwargs`` controls whether
    the one-arm shifts use exact or interpolated Tl/Tp construction. The sky
    basis for ``theta`` and ``phi`` is built once and reused across all links.
    """
    times = np.asarray(times, dtype=float)
    sky_basis = build_sky_basis(theta, phi)
    shift_kwargs = {} if variable_shift_kwargs is None else dict(variable_shift_kwargs)
    q_val = _resolve_wdm_q_value(int(times.size), int(Nf), q=q)
    shift_key = tuple(sorted((str(k), repr(v)) for k, v in shift_kwargs.items()))
    if links is None:
        links = ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3), (1, 2))
    links = tuple((int(s), int(r)) for s, r in links)

    cache_key = (
        id(orbits),
        float(theta),
        float(phi),
        int(Nf),
        int(d),
        int(q_val),
        bool(calc_m0),
        float(m_offset),
        None if L_trunc is None else int(L_trunc),
        int(times.size),
        float(times[0]),
        float(times[1] - times[0]),
        id(td_waveform_plus),
        id(td_waveform_cross),
        links,
        shift_key,
    )

    if cache is not None and cache_key in cache:
        return cache[cache_key]

    context = _build_wdm_dT_context(
        times,
        td_waveform_plus,
        td_waveform_cross,
        Nf=Nf,
        d=d,
        m_offset=m_offset,
        q=q_val,
        calc_m0=calc_m0,
    )

    shift_jobs = []
    for s, r in links:
        _, _, w_xi, time_shift = _one_arm_wdm_dT_components_from_context(
            context,
            s,
            r,
            orbits,
            theta,
            phi,
            include_time_domain_geometry=False,
            sky_basis=sky_basis,
        )
        shift_jobs.append((w_xi, -time_shift))

    shifted_links = wdm_time_shift_variable_batch(
        context["wdm"],
        shift_jobs,
        Nf=context["wdm"].Nf,
        L_trunc=L_trunc,
        batch_chunk=shift_kwargs.pop("batch_chunk", max(1, len(shift_jobs))),
        **shift_kwargs,
    )
    responses = {link: shifted for link, shifted in zip(links, shifted_links)}

    payload = (context["wdm"], responses)
    if cache is not None:
        cache[cache_key] = payload
    return payload

def one_arm_response_wdm_dT(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, Nf=500, return_td = False, gnm_method="cubic", gnm_wrap=True, d = 4, m_offset = 0.0, L_trunc=None, variable_shift_kwargs=None, q=None, calc_m0=True):
    """
    Evaluate the one-arm dT WDM response for a single directed link.

    This path builds the unshifted WDM coefficients on the ``dT`` grid,
    applies the fast coefficient-space time shift, and optionally reconstructs
    the time-domain response.

    Parameters
    ----------
    times : ndarray
        Sample times of the LISA response. Each entry is the emission time at
        the sending spacecraft.
    s : int
        Sending spacecraft index.
    r : int
        Receiving spacecraft index.
    orbits : object
        Orbit object used to evaluate spacecraft positions.
    theta : float
        Source polar angle in radians.
    phi : float
        Source azimuthal angle in radians.
    td_waveform_plus : ndarray
        Plus-polarization waveform sampled on ``times``.
    td_waveform_cross : ndarray
        Cross-polarization waveform sampled on ``times``.
    Nf : int
        Number of basis elements in the frequency dimension.
    return_td : bool
        Whether to also reconstruct and return the time-domain response.
    gnm_method : str
        Interpolation method used only when ``return_td=True``.
    gnm_wrap : bool
        Boundary handling used only when ``return_td=True``.
    d : int
        WDM oversampling/filter parameter.
    m_offset : float
        Optional frequency-index offset.
    L_trunc : int or None
        Optional lag truncation forwarded to the coefficient-space time shift.
    variable_shift_kwargs : dict or None
        Optional keyword arguments forwarded to ``wdm_time_shift_variable``.
        This can enable interpolated Tl/Tp construction for the one-arm shift.
    q : int or None, optional
        WDM truncation parameter. When ``None``, it is set to the maximum
        allowed value for the current tiling.
    calc_m0 : bool, optional
        Whether to include the ``m=0`` WDM component when constructing the
        transform.
    Returns
    -------
    ndarray or tuple
        Returns ``w_sr`` when ``return_td`` is false, otherwise
        ``(dl_sr, w_sr)``. The source-direction basis is built once and reused
        across the coefficient and time-shift calculations.
    """
    sky_basis = build_sky_basis(theta, phi)
    context = _build_wdm_dT_context(
        times,
        td_waveform_plus,
        td_waveform_cross,
        Nf=Nf,
        d=d,
        m_offset=m_offset,
        q=q,
        calc_m0=calc_m0,
    )
    return _one_arm_response_wdm_dT_from_context(
        context,
        s,
        r,
        orbits,
        theta,
        phi,
        return_td=return_td,
        gnm_method=gnm_method,
        gnm_wrap=gnm_wrap,
        L_trunc=L_trunc,
        variable_shift_kwargs=variable_shift_kwargs,
        sky_basis=sky_basis,
    )


