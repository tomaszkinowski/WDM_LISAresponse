r"""
``split_amp_sinc_first`` for every link of one source as a single jitted call.

The model is `response_corrections.split_responses`' ``split_amp_sinc_first``:
the waveform arrives already evaluated at :math:`t+\delta_R(t)`, with
:math:`\delta_R=\hat n\cdot x_c/c` the Roemer delay of the constellation
centre, and each link adds the amplitude-rate and sinc terms to its unshifted
grid and shifts the rest with the first-order delay-rate kernel. The rest is
the delay that composes with the warp to :math:`\delta_n`,
:math:`(\delta_n-\delta_R(t_n))/(1+\dot\delta_R(t_n))`. The input is the
transforms of that warped waveform, as production's input is the transforms
of the plain one.

What changes is only the arrangement:

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
    pass: `amp_sinc_kernel.apply_amplitude_rate` and the per-sigma
    accumulators of `apply_sinc_rate`, before any slope. Terms whose weight
    vanishes by the window's symmetry are left out: every amplitude weight at
    sigma = 0, and the sinc weights at sigma = 0 for even l.

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


def link_grid(freqs, dF, amp_row_p, amp_row_c, tau, d_amp_p, d_amp_c, d_tau,
              wp, wc, zp, sp, zc, sc):
    r"""
    One link's unshifted grid with the amplitude-rate and sinc terms,

    .. math::

        S_0(a^+\omega^+ + a^\times\omega^\times + \dot a^+ Z^+ + \dot a^\times Z^\times)
        + f\dot\tau\, D_0\,(a^+ Z^+ + a^\times Z^\times)
        + \tau\sum_\sigma D_\sigma\,(a^+ S^+_\sigma + a^\times S^\times_\sigma),

    with :math:`a=\tfrac12\ell F` per row, :math:`S_0=\mathrm{sinc}(f_m\tau)`,
    :math:`D_\sigma=\mathrm{sinc}'(x_\sigma)` at
    :math:`x_\sigma=(f_m+\sigma\Delta F/2)\tau`: :math:`\dot A` by the product
    rule on the per-row rates, and every sinc and slope from one sincos grid,
    the half-bin offsets by angle addition.
    """
    return _grid_core(freqs[None, :], dF, amp_row_p[:, None], amp_row_c[:, None],
                      tau[:, None], d_amp_p[:, None], d_amp_c[:, None], d_tau[:, None],
                      wp, wc, zp, sp, zc, sc)


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


def _link_grid(wdm, consts, rows, wp, wc, sums, j):
    rest, amp_row_p, amp_row_c, tau = rows
    grad = lambda q: jnp.gradient(q, wdm.dT, axis=-1)
    return link_grid(consts["freqs"], wdm.dF, amp_row_p[j], amp_row_c[j], tau[j],
                     grad(amp_row_p[j]), grad(amp_row_c[j]), grad(tau[j]),
                     wp, wc, *sums)


def _link_shift(wdm, consts, rows, grid, j):
    rest = rows[0][j]
    filters = row_filters(wdm, consts["filters"], consts["rate"], rest,
                          jnp.gradient(rest, wdm.dT))
    return rate_shift(wdm, filters, grid, rest)


@partial(jax.jit, static_argnums=0)
def stage_links(wdm, consts, rows, wp, wc, sums):
    """Every link's grid and its rate-corrected shift, in one graph."""
    return jnp.stack([_link_shift(wdm, consts, rows,
                                  _link_grid(wdm, consts, rows, wp, wc, sums, j), j)
                      for j in range(rows[0].shape[0])])


@partial(jax.jit, static_argnums=0)
def stage_grids(wdm, consts, rows, wp, wc, sums):
    """Every link's grid, all links in one pass (`link_grids_minor`)."""
    rest, amp_row_p, amp_row_c, tau = rows
    grad = lambda q: jnp.gradient(q, wdm.dT, axis=-1)
    return link_grids_minor(consts["freqs"], wdm.dF, amp_row_p, amp_row_c, tau,
                            grad(amp_row_p), grad(amp_row_c), grad(tau), wp, wc, *sums)


@partial(jax.jit, static_argnums=0)
def stage_shifts(wdm, consts, rows, grids):
    return jnp.stack([_link_shift(wdm, consts, rows, grids[j], j)
                      for j in range(rows[0].shape[0])])


def response_split(wdm, consts, theta, phi, wp, wc, separate_grids=False):
    r"""
    The ``split_amp_sinc_first`` response of every link, shape
    (num_links, Nt, Nf), from the transforms ``wp``, ``wc`` of the waveform
    evaluated at :math:`t+\delta_R(t)`: two or three jitted stages.
    """
    rows, sums = stage_common(wdm, consts, theta, phi, wp, wc)
    if separate_grids:
        return stage_shifts(wdm, consts, rows,
                            stage_grids(wdm, consts, rows, wp, wc, sums))
    return stage_links(wdm, consts, rows, wp, wc, sums)


def row_filters(wdm, filter_tables, rate_tables, delta, rate):
    r"""
    Every filter the rate-corrected shift needs at this delay, for all lags:
    :math:`T_l`, :math:`T'_l` and, with the rate folded in,
    :math:`\dot\delta(U_r, U_i, V_r, V_i)^\sigma_l`. Shape (14, 2L+1, Nt):
    [T, T', then sigma = -1, 0, 1 times (U_r, U_i, V_r, V_i)].
    """
    max_lag_L = (filter_tables.shape[1] - 1)//2
    lags = jnp.arange(-max_lag_L, max_lag_L + 1)
    look = lambda table: wdm._interp_filter(table, lags, -delta)
    rows = [look(filter_tables[1]), look(filter_tables[0])]
    rows += [look(rate_tables[s, k])*rate for s in range(3) for k in range(4)]
    return jnp.stack(rows)


def rate_shift(wdm, filters, wdm_coeff, delta):
    r"""
    `rate_kernel.apply_rate_corrected_shift` with its filters looked up
    beforehand (`row_filters`); the same lag sum, term for term.
    """
    max_lag_L = (filters.shape[1] - 1)//2
    n = jnp.arange(wdm.Nt)
    m = jnp.arange(wdm.Nf)
    sign_n = jnp.where(n % 2 == 0, 1.0, -1.0)[:, None]
    sign_m = jnp.where(m % 2 == 0, 1.0, -1.0)[None, :]

    theta = 2*jnp.pi*wdm.dF*jnp.outer(delta, m)
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    half = jnp.pi*wdm.dF*delta[:, None]
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)
    delay_filter = {-1: filters[1], 0: filters[0], 1: filters[1]}
    rate_filter = {sigma: filters[2 + 4*(sigma + 1):6 + 4*(sigma + 1)]
                   for sigma in (-1, 0, 1)}
    m_sigma = {sigma: ((m + 0.5*sigma)*wdm.dF)[None, :] for sigma in (-1, 0, 1)}

    ext = jnp.concatenate([wdm_coeff[-max_lag_L:], wdm_coeff,
                           wdm_coeff[:max_lag_L]], axis=0)
    ext = jnp.concatenate([ext[:, -1:], ext, ext[:, :1]], axis=1)
    i_pow = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))

    acc = 0.0
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(ext, max_lag_L - l, wdm.Nt, axis=0)
        total = 0.0
        for sigma in (-1, 0, 1):
            coeff = jax.lax.slice_in_dim(rows, sigma + 1, sigma + 1 + wdm.Nf,
                                         axis=1)
            i_re, i_im = i_pow[l % 4]
            lag_re, lag_im = ((1.0, 0.0) if sigma == 0 else
                              (i_re, i_im if sigma == -1 else -i_im))
            cos_s = 1.0 if sigma == 0 else cos_h
            sin_s = 0.0 if sigma == 0 else sigma*sin_h
            re_a = lag_re*cos_s + lag_im*sin_s
            re_b = lag_im*cos_s - lag_re*sin_s
            im_a = lag_im*cos_s - lag_re*sin_s
            im_b = -(lag_im*sin_s + lag_re*cos_s)
            alternates = (l + sigma) % 2 != 0
            if alternates:
                re_a, re_b, im_a, im_b = (-sign_n*im_a, -sign_n*im_b,
                                          sign_n*re_a, sign_n*re_b)
            if l % 2 != 0:
                alternates = not alternates
            T = delay_filter[sigma][i][:, None]
            Ur, Ui, Vr, Vi = (rate_filter[sigma][k, i][:, None] for k in range(4))
            p_c = re_a*(T + Vr) + im_a*Vi
            p_s = re_b*(T + Vr) + im_b*Vi
            q_c = re_a*Ur + im_a*Ui
            q_s = re_b*Ur + im_b*Ui
            ms = m_sigma[sigma]
            term = coeff*(cos_t*(p_c + ms*q_c) + sin_t*(p_s + ms*q_s))
            total = total + (term*sign_m if alternates else term)
        acc = acc + total
    return acc


def shift_weights(wdm, filters, delta):
    r"""
    The per-row weights of every term of `rate_shift`, precomputed:
    :math:`p_c, p_s, q_c, q_s` for each (l, sigma), with the lag and half-bin
    phases, the parity swap and the rate already folded in, and the
    :math:`\sigma\Delta F/2` part of :math:`m_\sigma` moved into :math:`p`,
    so the lag loop multiplies only by :math:`m\Delta F`. Shape
    (2L+1, 3, 4, Nt); whether a term carries :math:`(-1)^m` is static.
    """
    max_lag_L = (filters.shape[1] - 1)//2
    sign_n = jnp.where(jnp.arange(wdm.Nt) % 2 == 0, 1.0, -1.0)
    half = jnp.pi*wdm.dF*delta
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)
    i_pow = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))
    out = []
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        per_sigma = []
        for sigma in (-1, 0, 1):
            i_re, i_im = i_pow[l % 4]
            lag_re, lag_im = ((1.0, 0.0) if sigma == 0 else
                              (i_re, i_im if sigma == -1 else -i_im))
            cos_s = 1.0 if sigma == 0 else cos_h
            sin_s = 0.0 if sigma == 0 else sigma*sin_h
            re_a = lag_re*cos_s + lag_im*sin_s
            re_b = lag_im*cos_s - lag_re*sin_s
            im_a = lag_im*cos_s - lag_re*sin_s
            im_b = -(lag_im*sin_s + lag_re*cos_s)
            if (l + sigma) % 2 != 0:
                re_a, re_b, im_a, im_b = (-sign_n*im_a, -sign_n*im_b,
                                          sign_n*re_a, sign_n*re_b)
            T = filters[0 if sigma == 0 else 1][i]
            Ur, Ui, Vr, Vi = filters[2 + 4*(sigma + 1):6 + 4*(sigma + 1), i]
            q_c = re_a*Ur + im_a*Ui
            q_s = re_b*Ur + im_b*Ui
            shift = 0.5*sigma*wdm.dF
            p_c = re_a*(T + Vr) + im_a*Vi + shift*q_c
            p_s = re_b*(T + Vr) + im_b*Vi + shift*q_s
            per_sigma.append(jnp.stack(jnp.broadcast_arrays(p_c, p_s, q_c, q_s)))
        out.append(jnp.stack(per_sigma))
    return jnp.stack(out)


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
def stage_weights(wdm, consts, rows):
    out = []
    for j in range(rows[0].shape[0]):
        rest = rows[0][j]
        filters = row_filters(wdm, consts["filters"], consts["rate"], rest,
                              jnp.gradient(rest, wdm.dT))
        out.append(shift_weights(wdm, filters, rest))
    return jnp.stack(out)


@partial(jax.jit, static_argnums=0)
def stage_wshifts(wdm, rows, grids, weights):
    return jnp.stack([weighted_shift(wdm, weights[j], grids[j], rows[0][j])
                      for j in range(rows[0].shape[0])])


@partial(jax.jit, static_argnums=(0, 6))
def stage_grids_typed(wdm, consts, rows, wp, wc, sums, dtype):
    """`stage_grids` in ``dtype``, from per-row quantities formed in float64."""
    rest, amp_row_p, amp_row_c, tau = rows
    grad = lambda q: jnp.gradient(q, wdm.dT, axis=-1)
    f = lambda x: x.astype(dtype)
    sums = jax.tree.map(f, sums)
    return link_grids_minor(f(consts["freqs"]), wdm.dF, f(amp_row_p), f(amp_row_c), f(tau),
                            f(grad(amp_row_p)), f(grad(amp_row_c)), f(grad(tau)),
                            f(wp), f(wc), *sums)


@partial(jax.jit, static_argnums=(0, 4))
def stage_common_typed(wdm, consts, theta, phi, dtype, wp, wc):
    """`stage_common` with the shared sums in ``dtype``."""
    rows = _geometry(wdm, consts, theta, phi)
    sums = (shared_sums(wdm, consts["amp"].astype(dtype),
                        consts["sinc"].astype(dtype), wp.astype(dtype))
            + shared_sums(wdm, consts["amp"].astype(dtype),
                          consts["sinc"].astype(dtype), wc.astype(dtype)))
    return rows, sums


def response_split_w(wdm, consts, theta, phi, wp, wc, dtype=jnp.float64,
                     filters="table"):
    r"""
    `response_split` with the shift's per-row weights as their own stage.
    ``dtype`` is the working precision of the shared sums, grids, weights and
    lag loop; geometry, delays and the carrier phase stay float64, and the
    result is float64. ``filters`` is ``"table"`` (the linearly interpolated
    tables) or ``"poly"`` (the polynomials in ``consts["poly"]``, `add_poly`).
    """
    nodes = table_nodes(wdm, consts["filters"].shape[-1])
    if dtype == jnp.float64:
        rows, sums = stage_common(wdm, consts, theta, phi, wp, wc)
        grids = stage_grids(wdm, consts, rows, wp, wc, sums)
    else:
        rows, sums = stage_common_typed(wdm, consts, theta, phi, dtype, wp, wc)
        grids = stage_grids_typed(wdm, consts, rows, wp, wc, sums, dtype)
    if filters == "poly":
        weights = stage_weights_poly(wdm, consts, rows)
    else:
        weights = stage_weights_vec(wdm, consts, rows, nodes)
    return stage_wshifts(wdm, rows, grids, weights).astype(jnp.float64)


def shared_sums_grouped(wdm, amp_weights, sinc_weights, wdm_coeff):
    r"""
    `shared_sums` with one read of each lag's rows: one accumulator per
    (sum, sigma, lag parity) over the unshifted columns, then the
    :math:`(-1)^n` of odd :math:`l+\sigma`, the column :math:`m+\sigma` and
    the :math:`(-1)^m` of sigma = +-1 applied once, after the sum.
    """
    max_lag_L = (amp_weights.shape[1] - 1)//2
    sign_n = jnp.where(jnp.arange(wdm.Nt) % 2 == 0, 1.0, -1.0)[:, None]
    sign_m = jnp.where(jnp.arange(wdm.Nf) % 2 == 0, 1.0, -1.0)[None, :]
    ext = jnp.concatenate([wdm_coeff[-max_lag_L:], wdm_coeff,
                           wdm_coeff[:max_lag_L]], axis=0)
    acc = {}
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(ext, max_lag_L - l, wdm.Nt, axis=0)
        for k, sigma in enumerate((-1, 0, 1)):
            for name, w in (("amp", amp_weights), ("sinc", sinc_weights)):
                if sigma == 0 and (name == "amp" or l % 2 == 0):
                    continue
                key = (name, sigma, l % 2)
                acc[key] = acc.get(key, 0.0) + w[k, i]*rows
    move = lambda x, sigma: x if sigma == 0 else jnp.roll(x, -sigma, axis=1)*sign_m
    parts = {}
    for (name, sigma, parity), x in acc.items():
        x = x*sign_n if (parity + sigma) % 2 else x
        parts[(name, sigma)] = parts.get((name, sigma), 0.0) + x
    amp = move(parts[("amp", -1)], -1) + move(parts[("amp", 1)], 1)
    return amp, tuple(move(parts[("sinc", s)], s) for s in (-1, 0, 1))


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


@partial(jax.jit, static_argnums=(0, 3))
def stage_weights_local(wdm, consts, rows, nodes):
    out = []
    for j in range(rows[0].shape[0]):
        rest = rows[0][j]
        filters = row_filters_local(wdm, consts["filters"], consts["rate"], rest,
                                    jnp.gradient(rest, wdm.dT), nodes)
        out.append(shift_weights(wdm, filters, rest))
    return jnp.stack(out)


def poly_filters(wdm, max_lag_L, degree, bound=REST_BOUND, num_nodes=17):
    r"""
    Every filter of the rate-corrected shift as a polynomial in the scaled
    delay :math:`s=x/B` on :math:`|x|\le B` (`REST_BOUND`): least squares on
    Chebyshev nodes, the filters evaluated there from their definitions
    (`build_filter_tables` for :math:`T, T'`, the window integrals of
    `rate_kernel.build_rate_tables` for :math:`U, V`). Coefficients by
    descending power, shape (14, 2L+1, degree+1), ordered as `row_filters`
    (T, T', then sigma = -1, 0, 1 times U_r, U_i, V_r, V_i) before the rate.
    """
    from WDM.code.time_delay_filters.filters import build_filter_tables
    from elements_delay_rate import u_nodes, window_and_derivative

    s = np.cos(np.pi*(np.arange(num_nodes) + 0.5)/num_nodes)
    x = bound*s
    lags = np.arange(-max_lag_L, max_lag_L + 1)
    TT = np.asarray(build_filter_tables(jnp.asarray(lags), jnp.asarray(x/wdm.dT),
                                        wdm.d, wdm.A_frac))
    u, du = u_nodes(wdm)
    lag_phase = np.exp(2j*np.pi*np.outer(lags*wdm.dT, u))
    x_phase = np.exp(-2j*np.pi*np.outer(u, x))
    rate = []
    for sigma in (-1, 0, 1):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        _, dhi = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        U = (lag_phase*(lo*dhi*du)) @ x_phase
        V = (lag_phase*(u*lo*dhi*du)) @ x_phase
        rate += [U.real, U.imag, V.real, V.imag]
    values = np.concatenate([TT[::-1], np.stack(rate)])         # (14, 2L+1, nodes)
    vander = np.vander(s, degree + 1)
    coef, *_ = np.linalg.lstsq(vander, values.reshape(-1, num_nodes).T, rcond=None)
    return coef.T.reshape(14, 2*max_lag_L + 1, degree + 1)


def row_filters_poly(coef, delta, rate, bound=REST_BOUND):
    """`row_filters` from the polynomials of `poly_filters` (Horner per row)."""
    s = -delta/bound
    out = coef[..., 0][..., None]*jnp.ones_like(s)
    for k in range(1, coef.shape[-1]):
        out = out*s + coef[..., k][..., None]
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
POLY_DEGREE = 4


def add_poly(consts, wdm, degree=POLY_DEGREE):
    """``consts`` with the polynomial filters of `poly_filters` added."""
    max_lag_L = (consts["filters"].shape[1] - 1)//2
    return dict(consts, poly=jnp.asarray(poly_filters(wdm, max_lag_L, degree)))


@partial(jax.jit, static_argnums=0)
def stage_weights_poly(wdm, consts, rows):
    return jnp.stack([shift_weights_vec(
        wdm, row_filters_poly(consts["poly"], rows[0][j],
                              jnp.gradient(rows[0][j], wdm.dT)), rows[0][j])
        for j in range(rows[0].shape[0])])


def add_offset(consts):
    r"""
    ``consts`` with ``offset`` = :math:`x_r(t_n) - x_c(t_n)` per link
    (num_links, Nt, 3), metres: the rest delay is then
    :math:`\hat n\cdot(x_r - x_c)/c - \tau/2` directly, a few seconds formed
    without the cancellation of two ~500 s Roemer delays.
    """
    return dict(consts, offset=consts["receiver_position"] - consts["centre"][None])


def geometry_offset(wdm, consts, theta, phi, dtype=jnp.float64):
    r"""
    `_geometry` from ``consts["offset"]`` (`add_offset`), in ``dtype``: no
    term larger than the rest delay itself is formed, so float32 keeps it to
    about a microsecond.
    """
    f = lambda x: jnp.asarray(x).astype(dtype)
    theta, phi = jnp.atleast_1d(f(theta)), jnp.atleast_1d(f(phi))
    ltt = f(consts["ltt"])
    proj_p, proj_c, tau, n_hat = rs.compute_angular_geometry_wdm(
        theta, phi, ltt, f(consts["link_direction"]))
    rest = (jnp.tensordot(n_hat/rs.c, f(consts["offset"]), axes=([-1], [-1]))
            - 0.5*tau)[0]
    ltt, proj_p, proj_c, tau = rs.evaluate_at_landing_times(
        jnp.stack(jnp.broadcast_arrays(ltt, proj_p, proj_c, tau)), rest, wdm.dT)[:, 0]
    return rest, 0.5*ltt*proj_p, 0.5*ltt*proj_c, tau


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
