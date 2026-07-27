"""LISA one-arm response calculations in time, frequency and WDM form.

The production WDM path is built around :class:`OneArmWDMPlan`.  The direct
time-domain and basis-interpolation functions are retained as scientific
reference implementations; they are not alternative production shift
backends.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from lisaconstants import c
from numpy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import WDM

from WDM.code.time_delay_filters.time_shift_fast import wdm_time_shift_variable

from .one_arm_plan import (
    OneArmWDMPlan,
    OneArmWDMStaticPlan,
    _normalise_one_arm_execution_mode,
    _normalise_shift_backend,
    _validate_jax_request,
)
from .response_sub import (
    _compute_ltt_iterative,
    antenna_pattern_factors,
    build_sky_basis,
    gnm_at_times_grid_np_interp,
    rhat,
    wdm_pad_inputs_right,
)
from ..utils.utils import wdm_tail_padding_info


def _transform_wdm_polarizations(
    wdm,
    td_waveform_plus,
    td_waveform_cross,
    *,
    parallel=False,
):
    """Transform plus/cross polarizations sequentially or concurrently.

    The two DWT calls are numerically independent. The threaded route is a
    CPU scheduling option and does not change the WDM definition.
    """

    if not parallel:
        return (
            wdm.dwt(td_waveform_plus),
            wdm.dwt(td_waveform_cross),
        )

    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="wdm-polarization",
    ) as executor:
        plus_future = executor.submit(wdm.dwt, td_waveform_plus)
        cross_future = executor.submit(wdm.dwt, td_waveform_cross)
        return plus_future.result(), cross_future.result()


def one_arm_response_td(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, num_int_points=10):
    """
    Evaluate the time-domain one-arm response for a single directed link.

    Parameters
    ----------
    times : ndarray
        Sample reception/readout times of the LISA response.
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
    l_sr, x_r_rec = _compute_ltt_iterative(
        times,
        s,
        r,
        orbits,
        return_receiver_position=True,
    )
    t_emit = times - l_sr
    x_s = orbits.compute_position(t_emit, sc=[s])[:,0,:]
    rhat_sr = rhat(times, s, r, orbits, tt=l_sr, x_r_rec=x_r_rec)

    t_rec = times

    n = sky_basis.n
    Omega = sky_basis.Omega

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)

    xi_s = t_emit - np.einsum('x,tx->t', Omega, x_s/c)
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


def one_arm_response_td_chunked(
    times,
    s,
    r,
    orbits,
    theta,
    phi,
    td_waveform_plus,
    td_waveform_cross,
    num_int_points=10,
    chunk_size=100_000,
):
    """Evaluate the time-domain one-arm response in chunks over the time axis.

    This reproduces ``one_arm_response_td`` while limiting peak memory to the
    chunked time window instead of the full time series.
    """
    times = np.asarray(times, dtype=float)
    td_waveform_plus = np.asarray(td_waveform_plus)
    td_waveform_cross = np.asarray(td_waveform_cross)

    sky_basis = build_sky_basis(theta, phi)
    Omega = sky_basis.Omega

    h_plus_interp = interp1d(times, td_waveform_plus, kind="cubic", bounds_error=False, fill_value=0.0)
    h_cross_interp = interp1d(times, td_waveform_cross, kind="cubic", bounds_error=False, fill_value=0.0)

    out = np.empty_like(times, dtype=np.result_type(td_waveform_plus, td_waveform_cross, float))

    total_times = times.size
    for start in range(0, total_times, int(chunk_size)):
        stop = min(start + int(chunk_size), total_times)
        t_chunk = times[start:stop]

        l_sr, x_r_rec = _compute_ltt_iterative(
            t_chunk,
            s,
            r,
            orbits,
            return_receiver_position=True,
        )
        t_emit = t_chunk - l_sr
        x_s = orbits.compute_position(t_emit, sc=[s])[:, 0, :]
        rhat_sr = rhat(t_chunk, s, r, orbits, tt=l_sr, x_r_rec=x_r_rec)
        t_rec = t_chunk

        Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)

        xi_s = t_emit - np.einsum('x,tx->t', Omega, x_s / c)
        xi_r = t_rec - np.einsum('x,tx->t', Omega, x_r_rec / c)

        xi = np.linspace(xi_s, xi_r, num_int_points)
        h_plus = h_plus_interp(xi)
        h_cross = h_cross_interp(xi)

        H_plus = np.trapezoid(h_plus, x=xi, axis=0)
        H_cross = np.trapezoid(h_cross, x=xi, axis=0)

        F_plus, F_cross = antenna_pattern_factors(rhat_sr, sky_basis)
        dl_chunk = 0.5 * (1 - Om_dot_rhat)**(-1) * (F_plus * H_plus + F_cross * H_cross)
        out[start:stop] = c * dl_chunk

    return out

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
    l_sr, x_r_rec = _compute_ltt_iterative(times, s, r, orbits, return_receiver_position=True)  # (Nt,)
    rhat_sr = rhat(times, s, r, orbits, tt=l_sr, x_r_rec=x_r_rec)                             # (Nt,3)

    Omega = sky_basis.Omega

    Om_dot_rhat = np.einsum('i,ti->t', Omega, rhat_sr)              # (Nt,)
    xi_r = times - np.einsum('i,ti->t', Omega, x_r_rec / c)         # (Nt,)

    # FFT of waveform polarisations
    H_plus = fft(td_waveform_plus)
    H_cross = fft(td_waveform_cross)

    h_freq = fftfreq(Nt, dt)
    omega = 2.0 * np.pi * h_freq                                    # (Nt,)

    # Antenna scalars: F_+(t) = rhat^T e_+ rhat, same for cross
    F_plus, F_cross = antenna_pattern_factors(rhat_sr, sky_basis)   # (Nt,)

    # Combine constant phase shift into one term
    phase_t = (xi_r - times[0])                                     # (Nt,)

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

        # transfer function: sinc(arg/pi) * exp(-i*arg)
        tf = np.sinc(arg / np.pi) * np.exp(-1j * arg)               # (Nt,M)

        # scalar contraction: 0.5 * tf * (Hp*F_plus + Hc*F_cross)
        resp = 0.5 * tf * (F_plus[:, None] * Hp[None, :] + F_cross[:, None] * Hc[None, :])  # (Nt,M)

        # exp(i w * (xi_r - t0)); reception-time endpoint convention
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



def one_arm_response_wdm(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, Nf=500, gnm_method="cubic", gnm_wrap=True, d = 4, q=None, calc_m0=True, A_frac=0.25, parallel_wdm_transforms=False):
    """
    Evaluate the one-arm WDM response on the native sample grid.

    Parameters
    ----------
    times : ndarray
        Sample reception/readout times of the LISA response.
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
    A_frac : float, optional
        WDM basis/window shape parameter.
    q : int or None, optional
        WDM truncation parameter. When ``None``, it is set to the maximum
        allowed value for the current tiling.
    calc_m0 : bool, optional
        Whether to include the ``m=0`` WDM component when constructing the
        transform.
    parallel_wdm_transforms : bool, optional
        Evaluate the independent plus/cross WDM transforms concurrently on
        two Python threads.
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

    wdm = WDM.WDM.WDM_transform(dt=dt, Nf=Nf, N=N, q=q_val, calc_m0=calc_m0, d=d, A_frac=A_frac)
    #print(wdm)


    #print(f"How much padding was added: {(N-current_length)*wdm.dt}")

    w_t_plus, w_t_cross = _transform_wdm_polarizations(
        wdm,
        td_plus_wdm,
        td_cross_wdm,
        parallel=parallel_wdm_transforms,
    )

    #basic implementation of wdm transform 

    l_sr, x_r_rec = _compute_ltt_iterative(
        times,
        s,
        r,
        orbits,
        return_receiver_position=True,
    )
    rhat_sr = rhat(times, s, r, orbits, tt=l_sr, x_r_rec=x_r_rec)

    Omega = sky_basis.Omega

    Om_dot_rhat = np.einsum('x,tx->t', Omega, rhat_sr)
        
    xi_r = times - np.einsum('x,tx->t', Omega, x_r_rec/c)

    freq_vals = np.arange(0, wdm.Nf) * wdm.dF

    tau_s = l_sr * (1 - Om_dot_rhat) * 0.5
    sinc_arg = 2.0 * np.pi * freq_vals[np.newaxis, :] * tau_s[:, np.newaxis]

    F_plus, F_cross = antenna_pattern_factors(rhat_sr, sky_basis)

    n_grid, m_grid = np.meshgrid(np.arange(wdm.Nt), np.arange(wdm.Nf), indexing="ij")

    vals = gnm_at_times_grid_np_interp(wdm, n_grid, m_grid, xi_r - tau_s - times[0], method=gnm_method, wrap=gnm_wrap)

    sum_nm_plus = np.sinc(sinc_arg / np.pi)[:, np.newaxis, :] * w_t_plus[np.newaxis, :, :] * vals
    sum_nm_cross = np.sinc(sinc_arg / np.pi)[:, np.newaxis, :] * w_t_cross[np.newaxis, :, :] * vals

    dl_sr_wdm = 0.5 * c * l_sr * (
        F_plus * np.einsum("tnm->t", sum_nm_plus)
        + F_cross * np.einsum("tnm->t", sum_nm_cross)
    )

    return dl_sr_wdm[:N_in]


def _build_wdm_dT_grid(
    times,
    *,
    Nf=500,
    d=4,
    A_frac=0.25,
    m_offset=0.0,
    q=None,
    calc_m0=True,
):
    """Build WDM grid metadata without projecting a waveform.

    This is the waveform-independent half of ``_build_wdm_dT_context``.  It
    constructs the final padded WDM tiling, transform object, frequency grid
    and WDM centre-time grid, but deliberately performs no ``dwt`` calls.

    Parameters
    ----------
    times : ndarray
        Uniformly sampled input time grid.  The returned WDM transform uses
        the right-tail padded length required by ``2 * Nf`` tiling.
    Nf, d, A_frac, m_offset, q, calc_m0
        WDM construction settings, with the same meaning as in
        ``_build_wdm_dT_context``.

    Returns
    -------
    dict
        Waveform-independent WDM metadata.  In particular, this dictionary
        does not contain ``w_t_plus`` or ``w_t_cross``.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a 1D array with at least two samples.")

    dt = float(times[1] - times[0])
    if dt <= 0.0:
        raise ValueError("times must be strictly increasing.")
    if not np.allclose(np.diff(times), dt):
        raise ValueError("times must be uniformly sampled for WDM transforms.")

    Nf = int(Nf)
    padding_info = wdm_tail_padding_info(times.size, Nf)
    N = int(padding_info["target_length"])
    q_val = _resolve_wdm_q_value(N, Nf, q=q)

    wdm = WDM.WDM.WDM_transform(
        dt=dt,
        Nf=Nf,
        N=N,
        q=q_val,
        calc_m0=calc_m0,
        d=d,
        A_frac=A_frac,
    )

    freq_vals = (np.arange(wdm.Nf) + m_offset) * wdm.dF
    int_times_vals = times[0] + np.arange(wdm.Nt) * wdm.dT

    return {
        "times": times,
        "wdm": wdm,
        "A_frac": float(A_frac),
        "freq_vals": freq_vals,
        "int_times_vals": int_times_vals,
        "padding_info": padding_info,
    }


def _build_wdm_dT_context_from_wdm(
    times,
    td_waveform_plus,
    td_waveform_cross,
    wdm,
    m_offset=0.0,
    parallel_wdm_transforms=False,
):
    """Build reusable dT context using an existing WDM transform object.

    This avoids rebuilding ``WDM.WDM.WDM_transform`` and is intended for
    staged pipelines where geometry already precomputed the WDM tiling.
    """
    times = np.asarray(times, dtype=float)
    Nf = int(wdm.Nf)

    times_wdm, td_plus_wdm, td_cross_wdm, _ = wdm_pad_inputs_right(
        times,
        td_waveform_plus,
        td_waveform_cross,
        Nf,
    )

    if int(times_wdm.size) != int(wdm.N):
        raise ValueError(
            f"Padded waveform length {times_wdm.size} does not match precomputed WDM N={wdm.N}."
        )

    if times_wdm.size > 1:
        dt_wdm = float(times_wdm[1] - times_wdm[0])
        if not np.isclose(dt_wdm, float(wdm.dt), rtol=0.0, atol=1e-15):
            raise ValueError(
                f"Input dt={dt_wdm} does not match precomputed WDM dt={float(wdm.dt)}."
            )

    w_t_plus, w_t_cross = _transform_wdm_polarizations(
        wdm,
        td_plus_wdm,
        td_cross_wdm,
        parallel=parallel_wdm_transforms,
    )

    freq_vals = (np.arange(wdm.Nf) + m_offset) * wdm.dF
    times_vals = np.arange(wdm.Nt) * wdm.dT
    int_times_vals = times[0] + times_vals

    return {
        "times": times,
        "wdm": wdm,
        "A_frac": float(getattr(wdm, "A_frac", 0.25)),
        "w_t_plus": w_t_plus,
        "w_t_cross": w_t_cross,
        "freq_vals": freq_vals,
        "int_times_vals": int_times_vals,
        "parallel_wdm_transforms": bool(parallel_wdm_transforms),
    }


def _build_wdm_dT_context(
    times,
    td_waveform_plus,
    td_waveform_cross,
    Nf=500,
    d=4,
    A_frac=0.25,
    m_offset=0.0,
    q=None,
    calc_m0=True,
    wdm=None,
    parallel_wdm_transforms=False,
):
    """Build waveform-specific WDM projections on a reusable grid.

    Grid construction is delegated to ``_build_wdm_dT_grid``.  Consequently,
    callers that only need WDM metadata can use that function without running
    two dummy DWTs.
    """
    if wdm is None:
        grid_context = _build_wdm_dT_grid(
            times,
            Nf=Nf,
            d=d,
            A_frac=A_frac,
            m_offset=m_offset,
            q=q,
            calc_m0=calc_m0,
        )
        wdm = grid_context["wdm"]
    else:
        grid_context = None

    context = _build_wdm_dT_context_from_wdm(
        times,
        td_waveform_plus,
        td_waveform_cross,
        wdm,
        m_offset=m_offset,
        parallel_wdm_transforms=parallel_wdm_transforms,
    )

    if grid_context is not None:
        context["padding_info"] = grid_context["padding_info"]

    return context




def apply_one_arm_wdm_runtime_from_context(
    context,
    one_arm_static_plan,
    theta,
    phi,
    *,
    return_device=False,
    return_profile=False,
):
    """Apply a sky-independent one-arm plan at a runtime sky location."""

    if not isinstance(one_arm_static_plan, OneArmWDMStaticPlan):
        raise TypeError(
            "one_arm_static_plan must be a OneArmWDMStaticPlan instance."
        )
    candidate = context.get("wdm")
    if candidate is not one_arm_static_plan.wdm and (
        candidate is None
        or int(candidate.Nt) != int(one_arm_static_plan.wdm.Nt)
        or int(candidate.Nf) != int(one_arm_static_plan.wdm.Nf)
        or not np.isclose(float(candidate.dF), float(one_arm_static_plan.wdm.dF))
    ):
        raise ValueError(
            "The waveform context WDM grid does not match the static one-arm plan."
        )

    if return_device:
        result = one_arm_static_plan.apply_device(
            context["w_t_plus"],
            context["w_t_cross"],
            theta,
            phi,
            return_profile=return_profile,
        )
    else:
        if return_profile:
            shifted, profile = one_arm_static_plan.apply_batch(
                context["w_t_plus"],
                context["w_t_cross"],
                theta,
                phi,
                return_profile=True,
            )
            result = (
                {
                    link: shifted[index]
                    for index, link in enumerate(one_arm_static_plan.links)
                },
                profile,
            )
        else:
            result = one_arm_static_plan.apply(
                context["w_t_plus"],
                context["w_t_cross"],
                theta,
                phi,
            )
    return context["wdm"], result


def precompute_one_arm_wdm_dT_links_from_context(
    context,
    orbits,
    theta,
    phi,
    *,
    L_trunc=None,
    links=None,
    cache=None,
    variable_shift_kwargs=None,
    use_jax=None,
    one_arm_plan=None,
    return_device=False,
    one_arm_execution_mode="materialized",
    one_arm_parallel_links_per_group=1,
    one_arm_parallel_max_workers=None,
):
    """Evaluate directed one-arm WDM responses from a prebuilt context.

    Parameters
    ----------
    one_arm_plan : OneArmWDMPlan or None, optional
        Reusable waveform-independent response/shift plan. When omitted, a
        plan is built from the context, orbit and sky location.
    return_device : bool, optional
        When true, return link responses as JAX arrays without copying the
        shifted output batch back to NumPy.
    one_arm_execution_mode : {"materialized",
                              "materialized_parallel_links"}, optional
        Device execution strategy used when ``return_device=True``.
    one_arm_parallel_links_per_group : int, optional
        Number of contiguous directed links evaluated by each production
        shift call in ``materialized_parallel_links`` mode.
    one_arm_parallel_max_workers : int or None, optional
        Maximum number of concurrent link-group workers. The default uses one
        worker per group.

    Returns
    -------
    tuple
        ``(wdm, responses)`` using the directed-link dictionary API.
    """
    _validate_jax_request(use_jax)
    shift_kwargs, _ = _split_variable_shift_kwargs(variable_shift_kwargs)
    execution_mode = _normalise_one_arm_execution_mode(
        one_arm_execution_mode
    )

    parallel_links_per_group = int(
        one_arm_parallel_links_per_group
    )
    if parallel_links_per_group < 1:
        raise ValueError(
            "one_arm_parallel_links_per_group must be >= 1."
        )
    parallel_max_workers = (
        None
        if one_arm_parallel_max_workers is None
        else int(one_arm_parallel_max_workers)
    )
    if parallel_max_workers is not None and parallel_max_workers < 1:
        raise ValueError(
            "one_arm_parallel_max_workers must be >= 1 or None."
        )

    if links is None:
        links = ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3), (1, 2))
    links = tuple((int(s), int(r)) for s, r in links)

    context_times = np.asarray(context["times"], dtype=float)
    wdm = context["wdm"]

    if one_arm_plan is not None:
        if not isinstance(one_arm_plan, OneArmWDMPlan):
            raise TypeError(
                "one_arm_plan must be a OneArmWDMPlan instance."
            )
        if tuple(one_arm_plan.links) != links:
            raise ValueError(
                "The supplied one_arm_plan uses a different "
                "directed-link order."
            )

        if return_device:
            responses = one_arm_plan.apply_context_device(
                context,
                execution_mode=execution_mode,
                parallel_links_per_group=parallel_links_per_group,
                parallel_max_workers=parallel_max_workers,
            )
        else:
            responses = one_arm_plan.apply_context(context)

        return wdm, responses

    shift_key = tuple(
        sorted((str(k), repr(v)) for k, v in shift_kwargs.items())
    )
    cache_key = (
        id(wdm),
        id(context.get("w_t_plus")),
        id(context.get("w_t_cross")),
        id(orbits),
        float(theta),
        float(phi),
        int(wdm.Nf),
        int(wdm.Nt),
        float(context_times[0]),
        float(wdm.dT),
        None if L_trunc is None else int(L_trunc),
        links,
        shift_key,
        bool(return_device),
        execution_mode if return_device else None,
        parallel_links_per_group if return_device else None,
        parallel_max_workers if return_device else None,
    )

    if cache is not None and cache_key in cache:
        return cache[cache_key]

    plan = OneArmWDMPlan.build_from_context(
        context,
        orbits,
        theta,
        phi,
        links=links,
        L_trunc=L_trunc,
        variable_shift_kwargs=shift_kwargs,
        use_jax=use_jax,
    )

    if return_device:
        responses = plan.apply_context_device(
            context,
            execution_mode=execution_mode,
            parallel_links_per_group=parallel_links_per_group,
            parallel_max_workers=parallel_max_workers,
        )
    else:
        responses = plan.apply_context(context)

    payload = (wdm, responses)
    if cache is not None:
        cache[cache_key] = payload
    return payload



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
    use_jax=None,
    sky_basis=None,
):
    """Evaluate one-arm dT WDM response from precomputed context data.

    The coefficient-space shift uses the maintained production or reference
    WDM backend, with exact or interpolated ``Tl/Tp`` construction. A
    precomputed ``sky_basis`` may be supplied to avoid rebuilding the
    source-direction basis.
    """
    times = context["times"]
    wdm = context["wdm"]
    _validate_jax_request(use_jax)
    shift_kwargs, _ = _split_variable_shift_kwargs(variable_shift_kwargs)

    phase_center, tau_s, w_xi, time_shift = _one_arm_wdm_dT_components_from_context(
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
    tq = phase_center - times[0]
    vals = gnm_at_times_grid_np_interp(wdm, n_grid, m_grid, tq, method=gnm_method, wrap=gnm_wrap)
    dl_sr = np.einsum("tnm,nm->t", vals, w_xi)
    return dl_sr, w_sr


def _split_variable_shift_kwargs(variable_shift_kwargs):
    """Return cleaned one-arm and downstream TDI shift options.

    The one-arm plan builds all six links together and therefore uses its own
    default batch size.  ``batch_chunk`` and optional profiling flags remain
    available to downstream direct TDI batch calls.  Historical backend names
    are normalized onto the maintained ``production`` or ``reference`` paths.
    """

    shift_kwargs = (
        {} if variable_shift_kwargs is None else dict(variable_shift_kwargs)
    )
    _validate_jax_request(shift_kwargs.pop("use_jax", None))

    removed_options = {
        "assembly_vmap",
        "job_block_size",
        "jax_pad_last_chunk",
        "tl_tp_interp_backend",
    }
    present_removed = sorted(removed_options.intersection(shift_kwargs))
    if present_removed:
        raise TypeError(
            "Removed variable-shift options were supplied: "
            + ", ".join(present_removed)
        )

    shift_kwargs["assembly_backend"] = _normalise_shift_backend(
        shift_kwargs.get("assembly_backend", "production")
    )
    shift_kwargs.setdefault("assembly_precision", "complex64")
    shift_kwargs.setdefault("row_chunk_size", 128)
    shift_kwargs.setdefault("lag_block_size", 1)
    shift_kwargs.setdefault("tl_tp_mode", "exact")

    one_arm_shift_kwargs = dict(shift_kwargs)
    for key in (
        "batch_chunk",
        "profile_shift_stages",
        "profile_label",
        "return_shift_profile",
    ):
        one_arm_shift_kwargs.pop(key, None)

    tdi_batch_shift_kwargs = dict(shift_kwargs)
    return one_arm_shift_kwargs, tdi_batch_shift_kwargs


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

    n = sky_basis.n
    Omega = sky_basis.Omega

    l_sr_dT, x_r_rec_dT = _compute_ltt_iterative(
        int_times_vals,
        s,
        r,
        orbits,
        return_receiver_position=True,
    )
    rhat_sr_dT = rhat(int_times_vals, s, r, orbits, tt=l_sr_dT, x_r_rec=x_r_rec_dT)
    Om_dot_rhat_dT = rhat_sr_dT @ Omega
    tau_n = 0.5 * l_sr_dT * (1 - Om_dot_rhat_dT)
    proj_dT_plus, proj_dT_cross = antenna_pattern_factors(rhat_sr_dT, sky_basis)
    time_shift = x_r_rec_dT @ (n / c) - tau_n

    if include_time_domain_geometry:
        l_sr, x_r_rec = _compute_ltt_iterative(
        times,
        s,
        r,
        orbits,
        return_receiver_position=True,
    )
        rhat_sr = rhat(times, s, r, orbits, tt=l_sr, x_r_rec=x_r_rec)
        Om_dot_rhat = rhat_sr @ Omega
        tau_s = 0.5 * l_sr * (1 - Om_dot_rhat)

        # Reception-time phase centre:
        # xi_r - tau = t_rec - Omega·x_r(t_rec)/c - tau
        phase_center = times - (x_r_rec @ (Omega / c)) - tau_s
    else:
        phase_center = None
        tau_s = None

    sinc_mat = np.sinc(2.0 * freq_vals[None, :] * tau_n[:, None])

    A_nm_plus = 0.5 * c * l_sr_dT[:, None] * proj_dT_plus[:, None] * sinc_mat
    A_nm_cross = 0.5 * c * l_sr_dT[:, None] * proj_dT_cross[:, None] * sinc_mat

    w_xi = A_nm_plus * w_t_plus + A_nm_cross * w_t_cross

    return phase_center, tau_s, w_xi, time_shift



def precompute_one_arm_wdm_dT_links(
    times,
    orbits,
    theta,
    phi,
    td_waveform_plus,
    td_waveform_cross,
    Nf=500,
    d=4,
    A_frac=0.25,
    m_offset=0.0,
    L_trunc=None,
    links=None,
    cache=None,
    variable_shift_kwargs=None,
    use_jax=None,
    q=None,
    calc_m0=True,
    return_device=False,
    one_arm_execution_mode="materialized",
    one_arm_parallel_links_per_group=1,
    one_arm_parallel_max_workers=None,
    parallel_wdm_transforms=False,
):
    """Precompute directed one-arm dT WDM responses and return them as a map.

    The independent plus/cross WDM transforms may be evaluated concurrently
    with ``parallel_wdm_transforms=True``. When returning device arrays,
    ``one_arm_execution_mode='materialized_parallel_links'`` evaluates
    contiguous link groups concurrently with the unchanged production shift
    kernel.

    Parameters
    ----------
    one_arm_parallel_links_per_group : int, optional
        Number of directed links per concurrent production shift call.
    one_arm_parallel_max_workers : int or None, optional
        Maximum number of concurrent link-group workers.
    parallel_wdm_transforms : bool, optional
        Evaluate the plus/cross DWTs concurrently on two Python threads.

    Returns
    -------
    tuple
        ``(wdm, responses)`` where ``responses`` maps directed links to WDM
        coefficient arrays.
    """
    times = np.asarray(times, dtype=float)
    _validate_jax_request(use_jax)
    shift_kwargs, _ = _split_variable_shift_kwargs(variable_shift_kwargs)
    execution_mode = _normalise_one_arm_execution_mode(
        one_arm_execution_mode
    )
    q_val = _resolve_wdm_q_value(int(times.size), int(Nf), q=q)

    parallel_links_per_group = int(
        one_arm_parallel_links_per_group
    )
    if parallel_links_per_group < 1:
        raise ValueError(
            "one_arm_parallel_links_per_group must be >= 1."
        )
    parallel_max_workers = (
        None
        if one_arm_parallel_max_workers is None
        else int(one_arm_parallel_max_workers)
    )
    if parallel_max_workers is not None and parallel_max_workers < 1:
        raise ValueError(
            "one_arm_parallel_max_workers must be >= 1 or None."
        )

    shift_key = tuple(
        sorted((str(k), repr(v)) for k, v in shift_kwargs.items())
    )
    if links is None:
        links = ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3), (1, 2))
    links = tuple((int(s), int(r)) for s, r in links)

    cache_key = (
        id(orbits),
        float(theta),
        float(phi),
        int(Nf),
        int(d),
        float(A_frac),
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
        bool(return_device),
        execution_mode if return_device else None,
        parallel_links_per_group if return_device else None,
        parallel_max_workers if return_device else None,
        bool(parallel_wdm_transforms),
    )

    if cache is not None and cache_key in cache:
        return cache[cache_key]

    context = _build_wdm_dT_context(
        times,
        td_waveform_plus,
        td_waveform_cross,
        Nf=Nf,
        d=d,
        A_frac=A_frac,
        m_offset=m_offset,
        q=q_val,
        calc_m0=calc_m0,
        parallel_wdm_transforms=parallel_wdm_transforms,
    )

    payload = precompute_one_arm_wdm_dT_links_from_context(
        context,
        orbits,
        theta,
        phi,
        L_trunc=L_trunc,
        links=links,
        cache=None,
        variable_shift_kwargs=shift_kwargs,
        use_jax=use_jax,
        return_device=return_device,
        one_arm_execution_mode=execution_mode,
        one_arm_parallel_links_per_group=parallel_links_per_group,
        one_arm_parallel_max_workers=parallel_max_workers,
    )
    if cache is not None:
        cache[cache_key] = payload
    return payload


def one_arm_response_wdm_dT(times, s, r, orbits, theta, phi, td_waveform_plus, td_waveform_cross, Nf=500, return_td = False, gnm_method="cubic", gnm_wrap=True, d = 4, A_frac=0.25, m_offset = 0.0, L_trunc=None, variable_shift_kwargs=None, use_jax=None, q=None, calc_m0=True, parallel_wdm_transforms=False):
    """
    Evaluate the one-arm dT WDM response for a single directed link.

    This path builds the unshifted WDM coefficients on the ``dT`` grid,
    applies the fast coefficient-space time shift, and optionally reconstructs
    the time-domain response.

    Parameters
    ----------
    times : ndarray
        Sample reception/readout times of the LISA response.
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
    A_frac : float
        WDM basis/window shape parameter.
    m_offset : float
        Optional frequency-index offset.
    L_trunc : int or None
        Optional lag truncation forwarded to the coefficient-space time shift.
    variable_shift_kwargs : dict or None
        Optional numerical settings forwarded to the maintained production or
        reference WDM shift backend.
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
        A_frac=A_frac,
        m_offset=m_offset,
        q=q,
        calc_m0=calc_m0,
        parallel_wdm_transforms=parallel_wdm_transforms,
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
        use_jax=use_jax,
        sky_basis=sky_basis,
    )

