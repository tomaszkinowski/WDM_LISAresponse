"""Geometry, interpolation and padding helpers for LISA responses."""

from typing import NamedTuple, Tuple

import numpy as np
from lisaconstants import c
import WDM

from ..utils.utils import wdm_tail_padding_info


def wdm_pad_inputs_right(times, td_waveform_plus, td_waveform_cross, Nf):
    """Pad plus/cross waveforms on the right using ``WDM.utils.pad_signal``."""
    times = np.asarray(times, dtype=float)
    td_waveform_plus = np.asarray(td_waveform_plus)
    td_waveform_cross = np.asarray(td_waveform_cross)

    if times.ndim != 1 or td_waveform_plus.ndim != 1 or td_waveform_cross.ndim != 1:
        raise ValueError("times, td_waveform_plus, and td_waveform_cross must be 1D arrays.")
    if not (len(times) == len(td_waveform_plus) == len(td_waveform_cross)):
        raise ValueError("times, td_waveform_plus, and td_waveform_cross must have the same length.")
    if len(times) < 2:
        raise ValueError("Need at least two time samples for WDM input padding.")

    dt = times[1] - times[0]
    if not np.allclose(np.diff(times), dt):
        raise ValueError("times must be uniformly sampled for WDM transforms.")

    info = wdm_tail_padding_info(len(times), int(Nf))
    target_length = int(info["target_length"])

    td_plus_pad, mask = WDM.utils.pad_signal(td_waveform_plus, target_length)
    td_cross_pad, _ = WDM.utils.pad_signal(td_waveform_cross, target_length)

    pad_right = int(target_length - len(times))
    if pad_right == 0:
        times_pad = times.copy()
    else:
        times_right = times[-1] + dt * np.arange(1, pad_right + 1)
        times_pad = np.concatenate((times, times_right))

    info["mask"] = mask
    return times_pad, np.asarray(td_plus_pad), np.asarray(td_cross_pad), info


class SkyBasis(NamedTuple):
    """Reusable sky-frame quantities for a fixed source direction."""

    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    Omega: np.ndarray
    e_plus: np.ndarray
    e_cross: np.ndarray


def build_sky_basis(theta: float, phi: float) -> SkyBasis:
    """Build reusable sky-frame vectors, propagation direction, and GW tensors.

    Parameters
    ----------
    theta : float
        Polar angle of the source direction in radians.
    phi : float
        Azimuthal angle of the source direction in radians.

    Returns
    -------
    SkyBasis
        Precomputed sky-frame basis vectors ``(u, v, n)``, propagation
        direction ``Omega = -n``, and GW polarization tensors ``e_plus`` and
        ``e_cross``.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    u = np.array([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], dtype=float)
    v = np.array([sin_phi, -cos_phi, 0.0], dtype=float)
    n = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], dtype=float)
    Omega = -n
    e_plus = np.outer(u, u) - np.outer(v, v)
    e_cross = np.outer(u, v) + np.outer(v, u)
    return SkyBasis(u=u, v=v, n=n, Omega=Omega, e_plus=e_plus, e_cross=e_cross)


def basis_vectors(theta : float, 
                  phi : float) -> Tuple[np.ndarray, 
                                        np.ndarray, 
                                        np.ndarray]:
    """
    Return the orthonormal sky-frame basis vectors ``(u, v, n)``.

    Parameters
    ----------
    theta : float
        Polar angle of the source direction in radians.
    phi : float
        Azimuthal angle of the source direction in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The ``u``, ``v``, and ``n`` basis vectors, each with shape ``(3,)``.
    """
    sky_basis = build_sky_basis(theta, phi)
    return sky_basis.u, sky_basis.v, sky_basis.n

def basis_tensors(theta : float, 
                  phi : float) -> Tuple[np.ndarray, 
                                        np.ndarray]:
    """
    Return the GW polarization tensors for a fixed sky direction.

    Parameters
    ----------
    theta : float
        Polar angle of the source direction in radians.
    phi : float
        Azimuthal angle of the source direction in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The plus and cross polarization tensors, each with shape ``(3, 3)``.
    """
    sky_basis = build_sky_basis(theta, phi)
    return sky_basis.e_plus, sky_basis.e_cross


def antenna_pattern_factors(rhat_vec: np.ndarray, sky_basis: SkyBasis) -> Tuple[np.ndarray, np.ndarray]:
    """Project link directions onto the GW polarization basis without tensors.

    Parameters
    ----------
    rhat_vec : ndarray
        Link unit vector array with shape ``(num_times, 3)`` or ``(3,)``.
    sky_basis : SkyBasis
        Precomputed sky-basis quantities for the source direction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The plus and cross antenna-pattern factors evaluated on ``rhat_vec``.
    """
    rhat_vec = np.asarray(rhat_vec, dtype=float)
    ru = rhat_vec @ sky_basis.u
    rv = rhat_vec @ sky_basis.v
    return ru**2 - rv**2, 2.0 * ru * rv


def _solve_link_geometry(times, s, r, orbits, x_s=None):
    """Solve sender/receiver geometry for a directed link on a reception grid."""
    times = np.asarray(times, dtype=float)
    tt_niter = 4

    x_r = orbits.compute_position(times, sc=[r])[:, 0, :]
    if x_s is None:
        x_s = orbits.compute_position(times, sc=[s])[:, 0, :]
    else:
        x_s = np.asarray(x_s, dtype=float)

    tt = np.linalg.norm(x_r - x_s, axis=1) / c

    for _ in range(tt_niter):
        t_emit = times - tt
        x_s = orbits.compute_position(t_emit, sc=[s])[:, 0, :]
        tt = np.linalg.norm(x_r - x_s, axis=1) / c

    return x_s, x_r, tt


def _compute_ltt_iterative(times, s, r, orbits, x_s=None, return_receiver_position=False):
    """
    Iterate the light-travel-time solve for a directed LISA link.

    Parameters
    ----------
    times : ndarray
        Sample reception/readout times for the directed link, in seconds.
    s : int
        Sending spacecraft index.
    r : int
        Receiving spacecraft index.
    orbits : object
        Orbit object used to evaluate spacecraft positions.
    x_s : ndarray or None, optional
        Optional precomputed sender positions with shape ``(num_times, 3)``.
    return_receiver_position : bool, optional
        When true, also return the receiver position evaluated at reception
        time.

    Returns
    -------
    ndarray or tuple
        The light-travel time array. When ``return_receiver_position`` is true,
        returns ``(tt, x_r_rec)``.
    """
    _, x_r, tt = _solve_link_geometry(times, s, r, orbits, x_s=x_s)

    if return_receiver_position:
        return tt, x_r
    return tt 


def rhat(times, s, r, orbits, tt=None, x_s=None, x_r_rec=None):
    """
    Compute the directed link unit vector from sender emission to receiver readout.

    Parameters
    ----------
    times : ndarray
        Sample reception/readout times for the directed link, in seconds.
    s : int
        Sending spacecraft index.
    r : int
        Receiving spacecraft index.
    orbits : object
        Orbit object used to evaluate spacecraft positions.
    tt : ndarray or None, optional
        Optional light-travel time array corresponding to ``times``.
    x_s : ndarray or None, optional
        Optional precomputed sender positions at emission time with shape
        ``(num_times, 3)``.
    x_r_rec : ndarray or None, optional
        Optional precomputed receiver positions at reception time with shape
        ``(num_times, 3)``.

    Returns
    -------
    ndarray
        Directed link unit vectors with shape ``(num_times, 3)``.
    """
    if x_s is None and x_r_rec is None and tt is None:
        x_s, x_r_rec, _ = _solve_link_geometry(times, s, r, orbits)
    else:
        if tt is not None:
            tt = np.asarray(tt, dtype=float)

        if x_r_rec is None:
            x_r_rec = orbits.compute_position(times, [r])[:, 0, :]
        else:
            x_r_rec = np.asarray(x_r_rec, dtype=float)

        if x_s is None:
            if tt is None:
                x_s, x_r_rec, _ = _solve_link_geometry(times, s, r, orbits)
            else:
                t_emit = np.asarray(times, dtype=float) - tt
                x_s = orbits.compute_position(t_emit, [s])[:, 0, :]
        else:
            x_s = np.asarray(x_s, dtype=float)

    r_hat = (x_r_rec - x_s)
    r_hat = r_hat / np.linalg.norm(r_hat, axis=1)[:, np.newaxis]
    return r_hat


def _cubic_kernel_keys(t, a=-0.5):
    """Keys cubic convolution kernel (Catmull–Rom when a=-0.5)."""
    t = np.abs(t)
    w = np.zeros_like(t, dtype=np.float64)

    m1 = (t < 1)
    m2 = (t >= 1) & (t < 2)

    w[m1] = (a + 2)*t[m1]**3 - (a + 3)*t[m1]**2 + 1
    w[m2] = a*t[m2]**3 - 5*a*t[m2]**2 + 8*a*t[m2] - 4*a
    return w

def _lagrange_weight_matrix(frac, x_nodes):
    """
    Build weights W of shape (P, Nt) for evaluation points frac (Nt,)
    and node locations x_nodes (P,).
    """
    frac = frac.astype(np.float64)
    x_nodes = x_nodes.astype(np.float64)
    P = len(x_nodes)
    Nt = len(frac)
    W = np.ones((P, Nt), dtype=np.float64)
    for j in range(P):
        for k in range(P):
            if k == j:
                continue
            W[j, :] *= (frac - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
    return W

def _lanczos_weights(frac, A):
    """
    frac: (Nt,) fractional part in [0,1)
    A: half-width of kernel (integer), total taps = 2A
    Returns:
      offsets: (2A,) integer offsets
      W: (2A, Nt) weights for each offset
    """
    # offsets centered around 0: [-A+1, ..., A]
    offsets = np.arange(-A + 1, A + 1, dtype=int)  # length 2A

    # x = frac - offset for each tap; shape (2A, Nt)
    x = (frac[None, :] - offsets[:, None]).astype(np.float64)

    # sinc(pi x)/(pi x) using numpy's normalized sinc: np.sinc(x) = sin(pi x)/(pi x)
    # Lanczos window: sinc(x/A)
    W = np.sinc(x) * np.sinc(x / A)

    # Normalize so weights sum to 1 at each frac (helps DC consistency)
    s = np.sum(W, axis=0, keepdims=True)
    W = np.divide(W, s, out=W, where=(s != 0))

    return offsets, W


def gnm_at_times_grid_np_interp(
    wdm, n_grid, m_grid, t_query,
    method="cubic",
    time_first=True,
    a=-0.5,
    wrap=False,
):
    """
    Evaluate g_nm(t) for a grid of (n,m) at times t_query with selectable interpolation.

    Parameters
    ----------
    wdm : WDM_transform (needs dt, N, and gnm(n,m)->(N,))
    n_grid, m_grid : arrays of same shape (grid of indices)
    t_query : scalar or array of query times (seconds)
    method : {"nearest","linear","cubic"}
    time_first : if True -> output shape (t_shape + grid_shape), else (grid_shape + t_shape)
    a : Keys parameter for cubic (a=-0.5 Catmull–Rom)
    wrap : if True, periodic wrap indices mod N. If False, zero-extend out-of-range.

    Returns
    -------
    values : np.ndarray
        If time_first: shape = t_shape + grid_shape
        else:         shape = grid_shape + t_shape
    """
    dt = wdm.dt
    N  = wdm.N

    n_grid = np.asarray(n_grid)
    m_grid = np.asarray(m_grid)
    if n_grid.shape != m_grid.shape:
        raise ValueError("n_grid and m_grid must have the same shape")

    t_query = np.asarray(t_query)
    t_shape = t_query.shape if t_query.shape != () else (1,)
    t_flat = t_query.ravel()

    # fractional sample coordinate
    u = t_flat / dt

    # ----- build neighbor indices + weights depending on method -----
    if method == "nearest":
        k0 = np.rint(u).astype(int)

        if wrap:
            k0i = k0 % N
            w0  = np.ones_like(u, dtype=np.float64)
            idxs = [k0i]
            ws   = [w0]
        else:
            v0  = (k0 >= 0) & (k0 < N)
            k0c = np.clip(k0, 0, N-1)
            w0  = v0.astype(np.float64)
            idxs = [k0c]
            ws   = [w0]

    elif method == "linear":
        k = np.floor(u).astype(int)
        frac = u - k

        k0 = k
        k1 = k + 1

        if wrap:
            k0i = k0 % N
            k1i = k1 % N
            w0  = (1.0 - frac).astype(np.float64)
            w1  = frac.astype(np.float64)
            idxs = [k0i, k1i]
            ws   = [w0,  w1]
        else:
            v0  = (k0 >= 0) & (k0 < N)
            v1  = (k1 >= 0) & (k1 < N)
            k0c = np.clip(k0, 0, N-1)
            k1c = np.clip(k1, 0, N-1)
            w0  = (1.0 - frac) * v0
            w1  = frac * v1
            idxs = [k0c, k1c]
            ws   = [w0.astype(np.float64), w1.astype(np.float64)]

    elif method == "cubic":
        k = np.floor(u).astype(int)
        frac = u - k

        k_m1 = k - 1
        k_0  = k
        k_p1 = k + 1
        k_p2 = k + 2

        w_m1 = _cubic_kernel_keys(frac + 1.0, a=a)
        w_0  = _cubic_kernel_keys(frac,       a=a)
        w_p1 = _cubic_kernel_keys(frac - 1.0, a=a)
        w_p2 = _cubic_kernel_keys(frac - 2.0, a=a)

        if wrap:
            idxs = [k_m1 % N, k_0 % N, k_p1 % N, k_p2 % N]
            ws   = [w_m1,     w_0,     w_p1,     w_p2]
        else:
            v_m1 = (k_m1 >= 0) & (k_m1 < N)
            v_0  = (k_0  >= 0) & (k_0  < N)
            v_p1 = (k_p1 >= 0) & (k_p1 < N)
            v_p2 = (k_p2 >= 0) & (k_p2 < N)

            # zero-extend
            w_m1 = w_m1 * v_m1
            w_0  = w_0  * v_0
            w_p1 = w_p1 * v_p1
            w_p2 = w_p2 * v_p2

            idxs = [
                np.clip(k_m1, 0, N-1),
                np.clip(k_0,  0, N-1),
                np.clip(k_p1, 0, N-1),
                np.clip(k_p2, 0, N-1),
            ]
            ws = [w_m1, w_0, w_p1, w_p2]

        ws = [w.astype(np.float64) for w in ws]

    elif method.startswith("lagrange"):
        P = int(method.replace("lagrange", ""))
        if P < 2:
            raise ValueError("Lagrange needs >=2 points")
        # choose a symmetric stencil; for P=8 -> offsets [-3,-2,-1,0,1,2,3,4]
        half = P // 2
        offsets = np.arange(-half + 1, half + 1, dtype=int)  # length P for even P

        k = np.floor(u).astype(int)
        frac = (u - k).astype(np.float64)  # in [0,1)

        x_nodes = offsets.astype(np.float64)
        W = _lagrange_weight_matrix(frac, x_nodes)  # (P, Nt)

        idxs = [k + off for off in offsets]

        if wrap:
            idxs = [idx % N for idx in idxs]
            ws = [W[j, :] for j in range(P)]
        else:
            valid = [(idx >= 0) & (idx < N) for idx in idxs]
            idxs = [np.clip(idx, 0, N-1) for idx in idxs]
            ws = [(W[j, :] * valid[j]) for j in range(P)]

    elif method.startswith("lanczos"):
    # e.g. method="lanczos16" or "lanczos32"
        A = int(method.replace("lanczos", ""))
        if A < 2:
            raise ValueError("Lanczos needs A>=2")

        k = np.floor(u).astype(int)
        frac = (u - k).astype(np.float64)

        offsets, W = _lanczos_weights(frac, A)   # offsets (2A,), W (2A,Nt)

        idxs = [k + off for off in offsets]
        if wrap:
            idxs = [idx % N for idx in idxs]
            ws = [W[j, :] for j in range(len(offsets))]
        else:
            valid = [(idx >= 0) & (idx < N) for idx in idxs]
            idxs = [np.clip(idx, 0, N-1) for idx in idxs]
            ws = [(W[j, :] * valid[j]) for j in range(len(offsets))]

    else:
        raise ValueError(f"Unknown method={method}")


    # ----- evaluate for each (n,m) mode -----
    n_flat = n_grid.ravel()
    m_flat = m_grid.ravel()

    values_list = []
    for n_i, m_i in zip(n_flat, m_flat):
        g = np.asarray(wdm.gnm(n=int(n_i), m=int(m_i)))  # (N,)

        out = np.zeros_like(u, dtype=np.float64)
        for k_idx, w in zip(idxs, ws):
            out += w * g[k_idx]
        values_list.append(out.reshape(t_shape))

    values_modes_t = np.stack(values_list, axis=0)  # (Nmodes, *t_shape)

    grid_shape = n_grid.shape
    values = values_modes_t.reshape(grid_shape + t_shape)

    if time_first:
        g_dim = len(grid_shape)
        t_dim = len(t_shape)
        perm = list(range(g_dim, g_dim + t_dim)) + list(range(0, g_dim))
        values = np.transpose(values, perm)

    return values

def generate_time_shifts(Nt, times_0, dT, theta, phi, s, r, orbits, sky_basis=None):
    """
    Generate coefficient-space time shifts for the dT-grid WDM response.

    Parameters
    ----------
    Nt : int
        Number of samples on the WDM time grid.
    times_0 : float
        Start time of the grid in seconds.
    dT : float
        WDM time resolution in seconds.
    theta : float
        Polar angle of the incoming waveform in radians.
    phi : float
        Azimuthal angle of the incoming waveform in radians.
    s : int
        Sending spacecraft index.
    r : int
        Receiving spacecraft index.
    orbits : object
        Orbit object used to evaluate spacecraft positions.
    sky_basis : SkyBasis or None, optional
        Optional precomputed sky basis for ``theta`` and ``phi``. When omitted,
        it is built internally.

    Returns
    -------
    ndarray
        Time-shift array with length ``Nt``.
    """

    t_p = times_0 + np.arange(Nt) * dT

    if sky_basis is None:
        sky_basis = build_sky_basis(theta, phi)

    n = sky_basis.n
    Omega = sky_basis.Omega
    l_sr_tp, x_r_rec_tp = _compute_ltt_iterative(
        t_p,
        s,
        r,
        orbits,
        return_receiver_position=True,
    )
    rhat_sr_tp = rhat(t_p, s, r, orbits, tt=l_sr_tp, x_r_rec=x_r_rec_tp)
    Om_dot_rhat_tp = rhat_sr_tp @ Omega
    tau_p = 0.5 * l_sr_tp * (1 - Om_dot_rhat_tp)

    t_shifts = x_r_rec_tp @ (n / c) - tau_p


    return t_shifts