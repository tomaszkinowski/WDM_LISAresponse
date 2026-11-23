r"""
The first-order response (``split_amp_sinc_first``) for every link of one source
as a single jitted call.

The waveform arrives already evaluated at :math:`t+\delta_R(t)`, with
:math:`\delta_R=\hat n\cdot x_c/c` the Roemer delay of the constellation
centre, and each link adds the amplitude-rate and sinc terms to its unshifted
grid and shifts the rest with the first-order delay-rate kernel. The rest is
the delay that composes with the warp to :math:`\delta_n`,
:math:`(\delta_n-\delta_R(t_n))/(1+\dot\delta_R(t_n))`. The input is the
transforms of that warped waveform, as production's input is the transforms
of the plain one.

Structure:

- everything source-independent is built once (`setup`): the filter, rate,
  amplitude and sinc tables, the link geometry, :math:`x_c(t_n)`;
- the amplitude-rate and sinc lag sums have link-independent weights, so they
  are taken once per polarisation (`shared_sums`, one pass giving the
  amplitude sum and the three sinc sums) and every link multiplies them by its
  own :math:`\dot A` and sinc slope;
- the six links are one graph, unrolled as in production.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from WDM_LISAresponse.code.lisa_response import response_sub as rs


def setup(response, tables, centre):
    """The source-independent arrays of the call, as a dict of jnp arrays."""
    return dict(filters=response.filter_tables, rate=tables["first"],
                amp=tables["amp"], sinc=tables["sinc"],
                ltt=response.light_travel_time,
                link_direction=response.link_direction,
                receiver_position=response.receiver_position,
                centre=jnp.asarray(centre), freqs=response.wavelet_freqs)


def roemer_rows(theta, phi, centre):
    r""":math:`\delta_R(t_n)=\hat n\cdot x_c(t_n)/c`, shape (Nt,)."""
    n_hat = rs.build_sky_basis(theta, phi)[2]
    return centre @ n_hat / rs.c


def shared_sums(wdm, amp_weights, sinc_weights, wdm_coeff):
    r"""
    The amplitude-rate lag sum and the three sinc lag sums of one grid, in one
    pass, before any slope. Terms whose weight vanishes by the window's symmetry
    are left out: every amplitude weight at sigma = 0, and the sinc weights at
    sigma = 0 for even l.

    Returns
    -------
    amp : (Nt, Nf)
    sinc : (3, Nt, Nf), indexed sigma+1
    """
    max_lag_L = (amp_weights.shape[1] - 1)//2
    sign_n = jnp.where(jnp.arange(wdm.Nt) % 2 == 0, 1.0, -1.0)[:, None]
    sign_m = jnp.where(jnp.arange(wdm.Nf) % 2 == 0, 1.0, -1.0)[None, :]

    ext = jnp.concatenate([wdm_coeff[-max_lag_L:], wdm_coeff,
                           wdm_coeff[:max_lag_L]], axis=0)
    ext = jnp.concatenate([ext[:, -1:], ext, ext[:, :1]], axis=1)

    amp_same, amp_flip = 0.0, 0.0
    sinc = [0.0, 0.0, 0.0]
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(ext, max_lag_L - l, wdm.Nt, axis=0)
        for k, sigma in enumerate((-1, 0, 1)):
            coeff = jax.lax.slice_in_dim(rows, sigma + 1, sigma + 1 + wdm.Nf,
                                         axis=1)
            odd = (l + sigma) % 2 != 0
            if odd:
                coeff = coeff*sign_n
            flip = odd != (l % 2 != 0)
            if sigma != 0:                 # amp weights vanish at sigma = 0
                a = amp_weights[k, i]*coeff
                if flip:
                    amp_flip = amp_flip + a
                else:
                    amp_same = amp_same + a
            if sigma != 0 or l % 2 != 0:   # sinc: and at sigma = 0, l even
                s = sinc_weights[k, i]*coeff
                sinc[k] = sinc[k] + (s*sign_m if flip else s)
    return amp_same + amp_flip*sign_m, tuple(sinc)


def _grid_core(freqs, dF, ap, ac, tau, d_ap, d_ac, d_tau, wp, wc, zp, sp, zc, sc):
    """`link_grid` on arguments already shaped to broadcast against each other."""
    x = tau*freqs
    sin_x, cos_x = jnp.sin(jnp.pi*x), jnp.cos(jnp.pi*x)
    half = 0.5*jnp.pi*dF*tau
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)

    def sinc_and_slope(sigma):
        xs = x + sigma*0.5*dF*tau
        sn = sin_x if sigma == 0 else sin_x*cos_h + sigma*cos_x*sin_h
        cs = cos_x if sigma == 0 else cos_x*cos_h - sigma*sin_x*sin_h
        zero = xs == 0
        safe = jnp.where(zero, 1.0, xs)
        s = jnp.where(zero, 1.0, sn/(jnp.pi*safe))
        return s, jnp.where(zero, 0.0, (cs - s)/safe)

    s0, d0 = sinc_and_slope(0)
    grid = (s0*(ap*wp + ac*wc + d_ap*zp + d_ac*zc)
            + freqs*d_tau*d0*(ap*zp + ac*zc))
    sinc_sum = d0*(ap*sp[1] + ac*sc[1])
    for k, sigma in ((0, -1), (2, 1)):
        sinc_sum = sinc_sum + sinc_and_slope(sigma)[1]*(ap*sp[k] + ac*sc[k])
    return grid + tau*sinc_sum


def link_grids_minor(freqs, dF, amp_row_p, amp_row_c, tau, d_amp_p, d_amp_c, d_tau,
                     wp, wc, zp, sp, zc, sc):
    r"""
    Every link's `link_grid` in one pass, the link axis innermost, so that
    each element of the shared grids is read once for all links. Per-row
    arguments (num_links, Nt); returns (num_links, Nt, Nf).
    """
    row = lambda q: q.T[:, None, :]
    grid = lambda g: g[:, :, None]
    out = _grid_core(freqs[None, :, None], dF, row(amp_row_p), row(amp_row_c), row(tau),
                     row(d_amp_p), row(d_amp_c), row(d_tau), grid(wp), grid(wc),
                     grid(zp), tuple(map(grid, sp)), grid(zc), tuple(map(grid, sc)))
    return jnp.transpose(out, (2, 0, 1))


def _geometry(wdm, consts, theta, phi):
    theta, phi = jnp.atleast_1d(theta), jnp.atleast_1d(phi)
    proj_p, proj_c, tau, n_hat = rs.compute_angular_geometry_wdm(
        theta, phi, consts["ltt"], consts["link_direction"])
    delta = rs.compute_time_shifts(n_hat, consts["receiver_position"], tau)
    # Shifting h(t + delta_R(t)) by the rest gives h(t + rest + delta_R(t + rest)), so the
    # rest solves rest + delta_R(t_n + rest) = delta_n; to first order in the rest (the
    # next order is ~1e-9 s) that is rest = (delta_n - delta_R(t_n))/(1 + delta_R'(t_n)).
    roemer = roemer_rows(theta[0], phi[0], consts["centre"])
    rest = ((delta - roemer)/(1 + jnp.gradient(roemer, wdm.dT)))[0]
    ltt, proj_p, proj_c, tau = landing_sliced(
        jnp.stack(jnp.broadcast_arrays(consts["ltt"], proj_p, proj_c, tau)),
        rest, wdm.dT)[:, 0]
    return rest, 0.5*ltt*proj_p, 0.5*ltt*proj_c, tau


@partial(jax.jit, static_argnums=0)
def stage_common(wdm, consts, theta, phi, wp, wc):
    r"""
    The link-independent part: per-row rest delay and amplitudes of every
    link, and the shared lag sums of both polarisations. Its own jit, so the
    sums are stored once: in one graph with their consumers XLA:CPU fuses them
    into every link's grid instead (optimization_barrier does not stop it).
    """
    rows = _geometry(wdm, consts, theta, phi)
    sums = (shared_sums(wdm, consts["amp"], consts["sinc"], wp)
            + shared_sums(wdm, consts["amp"], consts["sinc"], wc))
    return rows, sums


@partial(jax.jit, static_argnums=0)
def stage_grids(wdm, consts, rows, wp, wc, sums):
    """Every link's grid, all links in one pass (`link_grids_minor`)."""
    rest, amp_row_p, amp_row_c, tau = rows
    grad = lambda q: jnp.gradient(q, wdm.dT, axis=-1)
    return link_grids_minor(consts["freqs"], wdm.dF, amp_row_p, amp_row_c, tau,
                            grad(amp_row_p), grad(amp_row_c), grad(tau), wp, wc, *sums)


def weighted_shift(wdm, weights, wdm_coeff, delta):
    """`rate_shift` from the weights of `shift_weights`."""
    max_lag_L = (weights.shape[0] - 1)//2
    m = jnp.arange(wdm.Nf)
    sign_m = jnp.where(m % 2 == 0, 1.0, -1.0)[None, :]
    m_dF = (m*wdm.dF)[None, :]
    dtype = wdm_coeff.dtype
    sign_m, m_dF = sign_m.astype(dtype), m_dF.astype(dtype)
    theta = 2*jnp.pi*wdm.dF*jnp.outer(delta, m)
    cos_t, sin_t = jnp.cos(theta).astype(dtype), jnp.sin(theta).astype(dtype)
    weights = weights.astype(dtype)
    ext = jnp.concatenate([wdm_coeff[-max_lag_L:], wdm_coeff,
                           wdm_coeff[:max_lag_L]], axis=0)
    ext = jnp.concatenate([ext[:, -1:], ext, ext[:, :1]], axis=1)
    acc = 0.0
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(ext, max_lag_L - l, wdm.Nt, axis=0)
        total = 0.0
        for k, sigma in enumerate((-1, 0, 1)):
            coeff = jax.lax.slice_in_dim(rows, sigma + 1, sigma + 1 + wdm.Nf, axis=1)
            p_c, p_s, q_c, q_s = (weights[i, k, c][:, None] for c in range(4))
            term = coeff*(cos_t*(p_c + m_dF*q_c) + sin_t*(p_s + m_dF*q_s))
            alternates = ((l + sigma) % 2 != 0) != (l % 2 != 0)
            total = total + (term*sign_m if alternates else term)
        acc = acc + total
    return acc


@partial(jax.jit, static_argnums=0)
def stage_wshifts(wdm, rows, grids, weights):
    return jnp.stack([weighted_shift(wdm, weights[j], grids[j], rows[0][j])
                      for j in range(rows[0].shape[0])])


#: bound on the per-link rest |delta_n - delta_R(t_n)|, seconds: |x_r - x_c|/c
#: (4.8 s) plus half the effective light travel time (<= l_sr, 8.4 s), with margin
REST_BOUND = 16.0


def table_nodes(wdm, num_interp_points, bound=REST_BOUND):
    """The table nodes a delay within +-bound can touch, as a static tuple."""
    spacing = wdm.dT/(num_interp_points - 1)
    centre = 0.5*(num_interp_points - 1)
    lo = int(np.floor(centre - bound/spacing)) - 1
    hi = int(np.ceil(centre + bound/spacing)) + 1
    return tuple(range(max(lo, 0), min(hi, num_interp_points - 1) + 1))


def row_filters_local(wdm, filter_tables, rate_tables, delta, rate, nodes):
    r"""
    `row_filters` for a delay that stays within the table nodes ``nodes``
    (`table_nodes`): the same linear interpolation, as hat weights of those
    nodes times the restricted tables, with no gather.
    """
    num_points = filter_tables.shape[-1]
    u = (-delta + 0.5*wdm.dT)/wdm.dT*(num_points - 1)
    i0 = jnp.clip(jnp.floor(u), 0, num_points - 2)
    frac = u - i0
    nodes_arr = jnp.asarray(nodes, dtype=u.dtype)[:, None]
    hat = (jnp.where(nodes_arr == i0, 1.0 - frac, 0.0)
           + jnp.where(nodes_arr == i0 + 1, frac, 0.0))          # (J, Nt)
    idx = np.asarray(nodes)
    tables = jnp.concatenate([filter_tables[::-1][:, :, idx],    # T, T'
                              rate_tables[:, :, :, idx].reshape(12, -1, len(idx))])
    # an explicit sum over the few nodes, not a contraction: XLA:GPU runs
    # float32 matmuls in TF32 (1e-3 relative), which cost 1.7e-4 in the response
    out = sum(tables[:, :, k, None]*hat[k] for k in range(len(nodes)))
    return out.at[2:].multiply(rate)


def shift_weights_vec(wdm, filters, delta):
    r"""
    `shift_weights` as whole-array operations over the lag axis: the rank-1
    mixing factors depend on the lag only through l mod 4, so they are built
    for the four classes and tiled along the lags (no gather), and every
    weight is formed in one elementwise pass. Same shape and values.
    """
    L = (filters.shape[1] - 1)//2
    sign_n = jnp.where(jnp.arange(wdm.Nt) % 2 == 0, 1.0, -1.0)
    half = jnp.pi*wdm.dF*delta
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)
    i_pow = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))
    lags = np.arange(-L, L + 1)
    # (re_a, re_b) per (l mod 4, sigma), shape (4, 3, 2, Nt); im_a = re_b, im_b = -re_a
    classes = []
    for r in range(4):
        per_sigma = []
        for sigma in (-1, 0, 1):
            i_re, i_im = i_pow[r]
            lag_re, lag_im = ((1.0, 0.0) if sigma == 0 else
                              (i_re, i_im if sigma == -1 else -i_im))
            cos_s = 1.0 if sigma == 0 else cos_h
            sin_s = 0.0 if sigma == 0 else sigma*sin_h
            a = lag_re*cos_s + lag_im*sin_s
            b = lag_im*cos_s - lag_re*sin_s
            if (r + sigma) % 2 != 0:
                a, b = -sign_n*b, sign_n*a
            per_sigma.append(jnp.stack(jnp.broadcast_arrays(a, b, sign_n)[:2]))
        classes.append(jnp.stack(per_sigma))
    classes = jnp.stack(classes)
    first = lags[0] % 4
    reps = -(-(len(lags) + first)//4)
    mix = jnp.tile(classes, (reps, 1, 1, 1))[first:first + len(lags)]   # (2L+1, 3, 2, Nt)
    a, b = mix[:, :, 0], mix[:, :, 1]
    T = jnp.stack([filters[1], filters[0], filters[1]], axis=1)          # (2L+1, 3, Nt)
    rate = filters[2:].reshape(3, 4, 2*L + 1, -1).transpose(2, 0, 1, 3)  # (2L+1, 3, 4, Nt)
    Ur, Ui, Vr, Vi = (rate[:, :, c] for c in range(4))
    shift = (0.5*wdm.dF*jnp.arange(-1, 2))[None, :, None]
    q_c, q_s = a*Ur + b*Ui, b*Ur - a*Ui
    p_c = a*(T + Vr) + b*Vi + shift*q_c
    p_s = b*(T + Vr) - a*Vi + shift*q_s
    return jnp.stack([p_c, p_s, q_c, q_s], axis=2)


@partial(jax.jit, static_argnums=(0, 3))
def stage_weights_vec(wdm, consts, rows, nodes):
    return jnp.stack([shift_weights_vec(
        wdm, row_filters_local(wdm, consts["filters"], consts["rate"], rows[0][j],
                               jnp.gradient(rows[0][j], wdm.dT), nodes), rows[0][j])
        for j in range(rows[0].shape[0])])


#: polynomial degree of the filters in the rest delay: 1e-8 (dT = 256 s) to
#: 1e-15 (dT >= 4096 s) on |x| <= REST_BOUND, against 6.6e-6 (T, T') and
#: 1.9e-5 (U, V) for the P = 128 linear tables


def landing_sliced(row_values, time_shift, row_spacing):
    r"""
    `response_sub.evaluate_at_landing_times` with the centred stencil as
    static slices of the interior rows and the end rows' off-centre stencils
    evaluated on the first and last ``num_taps`` rows only, joined once: the
    same Taylor step, without a concatenation per tap.
    """
    num_taps, order = rs.LANDING_NUM_TAPS, rs.LANDING_TAYLOR_ORDER
    weights = rs.landing_derivative_weights(num_taps, order)   # (taps, order+1, taps)
    centre = num_taps//2 - 1
    num_rows = row_values.shape[-1]
    num_middle = num_rows - num_taps + 1

    def derivs(position, block):
        """Row derivatives at stencil position(s) from taps ``block`` (..., taps, rows)."""
        return [sum(weights[position, k, j]*block[j] for j in range(num_taps))
                for k in range(order + 1)]

    middle = [sum(weights[centre, k, j]*row_values[..., j:j + num_middle]
                  for j in range(num_taps)) for k in range(order + 1)]
    head = [jnp.stack([sum(weights[p, k, j]*row_values[..., j] for j in range(num_taps))
                       for p in range(centre)], axis=-1) for k in range(order + 1)]
    tail = [jnp.stack([sum(weights[p, k, j]*row_values[..., num_middle - 1 + j]
                           for j in range(num_taps))
                       for p in range(centre + 1, num_taps)], axis=-1)
            for k in range(order + 1)]
    derivatives = [jnp.concatenate([h, m, t], axis=-1) for h, m, t in zip(head, middle, tail)]
    step = -jnp.broadcast_to(time_shift, row_values.shape)/row_spacing
    landed = derivatives[order]
    for k in range(order - 1, -1, -1):
        landed = derivatives[k] + step/(k + 1)*landed
    return landed
