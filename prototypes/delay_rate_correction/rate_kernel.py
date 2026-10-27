r"""
The delay-rate-corrected variable time shift as one jitted kernel.

.. math::

    \tilde\omega_{nm} = \sum_{l,\sigma}\omega_{(n-l)(m+\sigma)}\left[
        X_{n(n-l);m(m+\sigma)}(-\delta_n)
        + \dot\delta_n\, Y_{n(n-l);m(m+\sigma)}(\delta_n)\right]

`apply_variable_time_shift` with the first-order delay-rate term of
`reference_delay_rate_shift.py` fused into the same lag sum. A prototype: it mirrors the
production kernel line for line and adds the :math:`Y` weights, so the two
can be timed against each other.

The :math:`Y` element is :math:`\mathrm{Re}[f\,\overline{W}]` with :math:`f`
the bookkeeping factor of :math:`X` (filter set to one) and
:math:`W=(m+\sigma/2)\Delta F\,U^\sigma_l+V^\sigma_l`, so

.. math::

    Y = \mathrm{Re}f\,(m_\sigma U_r + V_r) + \mathrm{Im}f\,(m_\sigma U_i + V_i),
    \qquad m_\sigma = (m+\sigma/2)\Delta F .

:math:`\mathrm{Re}f` is what the production kernel already writes as
:math:`a\cos\theta+b\sin\theta`; :math:`\mathrm{Im}f` has the same form with
its own rank-1 weights and the same :math:`(-1)^m` alternation. So each
:math:`(l,\sigma)` term stays

.. math::

    \omega\left[\cos\theta\,(p_c + m_\sigma q_c)
              + \sin\theta\,(p_s + m_\sigma q_s)\right]

with :math:`p, q` rank-1 in :math:`n` and :math:`\dot\delta_n` folded into
them: one fused pass per lag, as before, with two more multiply-adds per
element.

:math:`U` and :math:`V` depend on :math:`l\Delta T - x` only, as :math:`T`
does, so they are tabulated on the same delay grid and looked up with the same
`_interp_filter`, lag folding included.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from elements_delay_rate import u_grid, window_and_derivative


def build_rate_tables(wdm, max_lag_L, num_interp_points):
    r"""
    :math:`U^\sigma_l`, :math:`V^\sigma_l` on `wdm`'s delay grid,
    :math:`x\in[-\Delta T/2,\Delta T/2]` with `num_interp_points` samples.

    Returns
    -------
    tables : jnp.ndarray
        shape=(3, 4, 2L+1, P), indexed [sigma+1, (U_r, U_i, V_r, V_i)].
    """
    u, du = u_grid(wdm)
    lags_dT = np.arange(-max_lag_L, max_lag_L + 1)*wdm.dT
    x = np.linspace(-0.5*wdm.dT, 0.5*wdm.dT, num_interp_points)
    lag_phase = np.exp(2j*np.pi*np.outer(lags_dT, u))
    x_phase = np.exp(-2j*np.pi*np.outer(u, x))

    tables = np.empty((3, 4, len(lags_dT), num_interp_points))
    for i, sigma in enumerate((-1, 0, 1)):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        _, dhi = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        U = (lag_phase*(lo*dhi)) @ x_phase*du
        V = (lag_phase*(u*lo*dhi)) @ x_phase*du
        tables[i] = np.stack([U.real, U.imag, V.real, V.imag])
    return jnp.asarray(tables)


@partial(jax.jit, static_argnums=0)
def apply_rate_corrected_shift(wdm, filter_tables, rate_tables, wdm_coeff,
                               delta, delta_dot):
    r"""
    The variable time shift with the first-order delay-rate term.

    Parameters
    ----------
    wdm : WDM_transform
        Static.
    filter_tables : jnp.ndarray
        From `build_time_delay_filter_interpolants`, shape=(2, 2L+1, P).
    rate_tables : jnp.ndarray
        From `build_rate_tables`, shape=(3, 4, 2L+1, P), same L and P.
    wdm_coeff : jnp.ndarray
        shape=(Nt, Nf).
    delta, delta_dot : jnp.ndarray
        :math:`\delta_n` and :math:`\dot\delta_n`, shape=(Nt,).

    Returns
    -------
    shifted : jnp.ndarray
        shape=(Nt, Nf).
    """
    max_lag_L = (filter_tables.shape[1] - 1)//2

    n = jnp.arange(wdm.Nt)
    m = jnp.arange(wdm.Nf)

    sign_n = jnp.where(n % 2 == 0, 1.0, -1.0)[:, jnp.newaxis]
    sign_m = jnp.where(m % 2 == 0, 1.0, -1.0)[jnp.newaxis, :]

    theta = 2*jnp.pi*wdm.dF*jnp.outer(delta, m)
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    half = jnp.pi*wdm.dF*delta[:, jnp.newaxis]
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)

    lags = jnp.arange(-max_lag_L, max_lag_L + 1)
    Tl = wdm._interp_filter(filter_tables[1], lags, -delta)
    Tprimel = wdm._interp_filter(filter_tables[0], lags, -delta)
    delay_filter = {-1: Tprimel, 0: Tl, +1: Tprimel}

    # U_r, U_i, V_r, V_i for each sigma, every lag at once: (4, 2L+1, Nt),
    # with delta_dot folded in so the rate never touches a grid
    rate_filter = {sigma: jnp.stack([wdm._interp_filter(rate_tables[sigma+1, k],
                                                        lags, -delta)
                                     for k in range(4)])*delta_dot
                   for sigma in (-1, 0, 1)}

    # (m + sigma/2) dF, the only frequency dependence of the rate term
    m_sigma = {sigma: ((m + 0.5*sigma)*wdm.dF)[jnp.newaxis, :]
               for sigma in (-1, 0, 1)}

    cyclic_ext = jnp.concatenate([wdm_coeff[-max_lag_L:],
                                  wdm_coeff,
                                  wdm_coeff[:max_lag_L]], axis=0)
    cyclic_ext = jnp.concatenate([cyclic_ext[:, -1:],
                                  cyclic_ext,
                                  cyclic_ext[:, :1]], axis=1)

    i_pow = jnp.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])

    def weights(l, i, sigma):
        r"""
        Rank-1 weights of `cos_t` and `sin_t`: :math:`p` (lag-independent of
        m) and :math:`q` (multiplying :math:`m_\sigma`), shape (Nt, 1) each.
        """
        i_pow_re, i_pow_im = i_pow[l % 4]
        if sigma == 0:
            lag_phase_re, lag_phase_im = 1.0, 0.0
        else:
            lag_phase_re = i_pow_re
            lag_phase_im = i_pow_im if sigma == -1 else -i_pow_im

        cos_s = 1.0 if sigma == 0 else cos_h
        sin_s = 0.0 if sigma == 0 else sigma*sin_h

        # Re and Im of C exp(-i phi), as weights of cos_t and sin_t
        re_a = lag_phase_re*cos_s + lag_phase_im*sin_s
        re_b = lag_phase_im*cos_s - lag_phase_re*sin_s
        im_a = lag_phase_im*cos_s - lag_phase_re*sin_s
        im_b = -(lag_phase_im*sin_s + lag_phase_re*cos_s)

        # parity: (l+sigma) odd multiplies by i (-1)^(n+m), which swaps Re/Im
        if (l + sigma) % 2 == 0:
            alternates = False
        else:
            re_a, re_b, im_a, im_b = (-sign_n*im_a, -sign_n*im_b,
                                      sign_n*re_a, sign_n*re_b)
            alternates = True

        if l % 2 != 0:
            alternates = not alternates

        T = delay_filter[sigma][i][:, jnp.newaxis]
        Ur, Ui, Vr, Vi = (rate_filter[sigma][k, i][:, jnp.newaxis]
                          for k in range(4))

        p_c = re_a*(T + Vr) + im_a*Vi
        p_s = re_b*(T + Vr) + im_b*Vi
        q_c = re_a*Ur + im_a*Ui
        q_s = re_b*Ur + im_b*Ui

        return p_c, p_s, q_c, q_s, alternates

    def body(i, acc):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(cyclic_ext, max_lag_L - l,
                                            wdm.Nt, axis=0)
        total = 0.0
        for sigma in (-1, 0, 1):
            coeff = jax.lax.slice_in_dim(rows, sigma + 1,
                                         sigma + 1 + wdm.Nf, axis=1)
            p_c, p_s, q_c, q_s, alternates = weights(l, i, sigma)
            ms = m_sigma[sigma]
            term = coeff*(cos_t*(p_c + ms*q_c) + sin_t*(p_s + ms*q_s))
            total = total + (term*sign_m if alternates else term)
        return acc + total

    acc = jnp.zeros((wdm.Nt, wdm.Nf), dtype=wdm_coeff.dtype)
    for i in range(2*max_lag_L + 1):
        acc = body(i, acc)
    return acc


def build_rate_tables_2(wdm, max_lag_L, num_interp_points):
    r"""
    :math:`U, V, U_2, V_2, W_2` (see `elements_second_order.py`) on `wdm`'s delay
    grid, shape (3, 10, 2L+1, P), indexed [sigma+1, (U_r, U_i, V_r, V_i,
    U2_r, U2_i, V2_r, V2_i, W2_r, W2_i)].
    """
    from elements_second_order import window_second_derivative

    u, du = u_grid(wdm)
    lags_dT = np.arange(-max_lag_L, max_lag_L + 1)*wdm.dT
    x = np.linspace(-0.5*wdm.dT, 0.5*wdm.dT, num_interp_points)
    lag_phase = np.exp(2j*np.pi*np.outer(lags_dT, u))
    x_phase = np.exp(-2j*np.pi*np.outer(u, x))

    tables = np.empty((3, 10, len(lags_dT), num_interp_points))
    for i, sigma in enumerate((-1, 0, 1)):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        _, dhi = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        d2hi = window_second_derivative(wdm, u + 0.5*sigma*wdm.dF)
        U = (lag_phase*(lo*dhi)) @ x_phase*du
        V = (lag_phase*(u*lo*dhi)) @ x_phase*du
        U2 = (lag_phase*(lo*d2hi)) @ x_phase*du
        V2 = (lag_phase*(u*lo*d2hi)) @ x_phase*du
        W2 = (lag_phase*(u*u*lo*d2hi)) @ x_phase*du
        tables[i] = np.stack([U.real, U.imag, V.real, V.imag, U2.real, U2.imag,
                              V2.real, V2.imag, W2.real, W2.imag])
    return jnp.asarray(tables)


@partial(jax.jit, static_argnums=0)
def apply_corrected_shift_2(wdm, filter_tables, tables, wdm_coeff,
                            delta, rate, curvature):
    r"""
    The variable time shift to second order in the delay's rate and first in
    its curvature:

    .. math::

        \sum\omega\left[X + \dot\delta_n Y + \tfrac12\dot\delta_n^2 Y^{(2)}
                        + \tfrac12\ddot\delta_n Y^{(c)}\right] ,

    every element :math:`\mathrm{Re}[f\,\overline{W}]` with, in powers of
    :math:`c=(m+\sigma/2)\Delta F`,

    .. math::

        W = \underbrace{\dot\delta V + \tfrac12\dot\delta^2 W_2
                        + \tfrac12\ddot\delta\kappa_c V_2}_{c^0}
          + c\,\underbrace{\left(\dot\delta U + \dot\delta^2 V_2
                        + \tfrac12\ddot\delta\kappa_c U_2\right)}_{c^1}
          + c^2\,\underbrace{\tfrac12\dot\delta^2 U_2}_{c^2},
        \qquad \kappa_c = -\frac{i}{2\pi} .

    The correction filters are looked up for every lag at once and folded
    into the three coefficients before the sum: :math:`6(2L+1)N_t` per
    :math:`\sigma`. Looking them up lag by lag inside the sum saved that
    memory but cost 2.5x (Nf = 250) to 7x (Nf = 100) in gathers.
    """
    max_lag_L = (filter_tables.shape[1] - 1)//2

    n = jnp.arange(wdm.Nt)
    m = jnp.arange(wdm.Nf)
    sign_n = jnp.where(n % 2 == 0, 1.0, -1.0)[:, jnp.newaxis]
    sign_m = jnp.where(m % 2 == 0, 1.0, -1.0)[jnp.newaxis, :]

    theta = 2*jnp.pi*wdm.dF*jnp.outer(delta, m)
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    half = jnp.pi*wdm.dF*delta[:, jnp.newaxis]
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)

    lags = jnp.arange(-max_lag_L, max_lag_L + 1)
    Tl = wdm._interp_filter(filter_tables[1], lags, -delta)
    Tprimel = wdm._interp_filter(filter_tables[0], lags, -delta)
    delay_filter = {-1: Tprimel, 0: Tl, +1: Tprimel}

    c_sigma = {sigma: ((m + 0.5*sigma)*wdm.dF)[jnp.newaxis, :]
               for sigma in (-1, 0, 1)}

    # Every lag at once, folded straight into the three polynomial
    # coefficients so only six (2L+1, Nt) arrays per sigma stay live.
    # kappa_c = -i/(2 pi): Re(kappa_c z) = Im z/(2 pi), Im(kappa_c z) = -Re z/(2 pi)
    def coefficients(sigma):
        Ur, Ui, Vr, Vi, U2r, U2i, V2r, V2i, W2r, W2i = (
            wdm._interp_filter(tables[sigma + 1, k], lags, -delta) for k in range(10))
        a1, a2, ac = rate[jnp.newaxis], 0.5*rate[jnp.newaxis]**2, \
            0.5*curvature[jnp.newaxis]/(2*jnp.pi)
        return jnp.stack([a1*Vr + a2*W2r + ac*V2i,
                          a1*Vi + a2*W2i - ac*V2r,
                          a1*Ur + 2*a2*V2r + ac*U2i,
                          a1*Ui + 2*a2*V2i - ac*U2r,
                          a2*U2r, a2*U2i])
    poly = {sigma: coefficients(sigma) for sigma in (-1, 0, 1)}

    cyclic_ext = jnp.concatenate([wdm_coeff[-max_lag_L:], wdm_coeff,
                                  wdm_coeff[:max_lag_L]], axis=0)
    cyclic_ext = jnp.concatenate([cyclic_ext[:, -1:], cyclic_ext,
                                  cyclic_ext[:, :1]], axis=1)

    i_pow = jnp.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])

    def weights(l, i, sigma):
        i_pow_re, i_pow_im = i_pow[l % 4]
        if sigma == 0:
            lag_phase_re, lag_phase_im = 1.0, 0.0
        else:
            lag_phase_re = i_pow_re
            lag_phase_im = i_pow_im if sigma == -1 else -i_pow_im
        cos_s = 1.0 if sigma == 0 else cos_h
        sin_s = 0.0 if sigma == 0 else sigma*sin_h

        re_a = lag_phase_re*cos_s + lag_phase_im*sin_s
        re_b = lag_phase_im*cos_s - lag_phase_re*sin_s
        im_a = lag_phase_im*cos_s - lag_phase_re*sin_s
        im_b = -(lag_phase_im*sin_s + lag_phase_re*cos_s)
        if (l + sigma) % 2 == 0:
            alternates = False
        else:
            re_a, re_b, im_a, im_b = (-sign_n*im_a, -sign_n*im_b,
                                      sign_n*re_a, sign_n*re_b)
            alternates = True
        if l % 2 != 0:
            alternates = not alternates

        T = delay_filter[sigma][i][:, jnp.newaxis]
        c0r, c0i, c1r, c1i, c2r, c2i = (poly[sigma][k, i][:, jnp.newaxis]
                                        for k in range(6))

        p_c = re_a*(T + c0r) + im_a*c0i
        p_s = re_b*(T + c0r) + im_b*c0i
        q_c, q_s = re_a*c1r + im_a*c1i, re_b*c1r + im_b*c1i
        r_c, r_s = re_a*c2r + im_a*c2i, re_b*c2r + im_b*c2i
        return p_c, p_s, q_c, q_s, r_c, r_s, alternates

    acc = jnp.zeros((wdm.Nt, wdm.Nf), dtype=wdm_coeff.dtype)
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(cyclic_ext, max_lag_L - l,
                                            wdm.Nt, axis=0)
        total = 0.0
        for sigma in (-1, 0, 1):
            coeff = jax.lax.slice_in_dim(rows, sigma + 1, sigma + 1 + wdm.Nf,
                                         axis=1)
            p_c, p_s, q_c, q_s, r_c, r_s, alternates = weights(l, i, sigma)
            c = c_sigma[sigma]
            term = coeff*(cos_t*(p_c + c*(q_c + c*r_c))
                          + sin_t*(p_s + c*(q_s + c*r_s)))
            total = total + (term*sign_m if alternates else term)
        acc = acc + total
    return acc


def build_fit_tables(wdm, max_lag_L, num_interp_points, eps_max, degree,
                     num_nodes=None):
    r"""
    The exact linear-warp shift element, fitted in the window offset.

    For a delay :math:`\delta+\dot\delta(t-t_n)` the element's filter is, up to
    the stretch term,

    .. math::

        G^\sigma_l(x,\varepsilon) = \int\mathrm{d}u\,e^{2\pi i u(l\Delta T-x)}
            \tilde\Phi(u-\tfrac{\sigma}{2}\Delta F)\,
            \tilde\Phi(u+\tfrac{\sigma}{2}\Delta F+\varepsilon),
        \qquad \varepsilon=\dot\delta\,c .

    :math:`G-T` is fitted by least squares on Chebyshev nodes over
    :math:`|\varepsilon|\le\varepsilon_{\max}` with the powers
    :math:`(\varepsilon/\varepsilon_{\max})^k`, :math:`k=1..K` (`fit_window_offset.py`
    measures the error), rather than Taylor-expanded at zero as
    `build_rate_tables_2` does.

    Returns
    -------
    tables : jnp.ndarray
        shape (3, 2K, 2L+1, P): [sigma+1, (a_1 re, a_1 im, ..., a_K re, a_K im)],
        the coefficients of :math:`(\varepsilon/\varepsilon_{\max})^k`.
    """
    num_nodes = num_nodes or 4*degree
    u, du = u_grid(wdm, num=20001)
    lags_dT = np.arange(-max_lag_L, max_lag_L + 1)*wdm.dT
    x = np.linspace(-0.5*wdm.dT, 0.5*wdm.dT, num_interp_points)
    lag_phase = np.exp(2j*np.pi*np.outer(lags_dT, u))
    x_phase = np.exp(-2j*np.pi*np.outer(u, x))
    z = np.cos(np.pi*(np.arange(num_nodes) + 0.5)/num_nodes)
    V = np.vander(z, degree + 1, increasing=True)[:, 1:]

    tables = np.empty((3, 2*degree, len(lags_dT), num_interp_points))
    for i, sigma in enumerate((-1, 0, 1)):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        hi0, _ = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        diffs = np.empty((num_nodes, len(lags_dT)*num_interp_points), complex)
        for k, zk in enumerate(z):
            hi, _ = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF + zk*eps_max)
            diffs[k] = ((lag_phase*(lo*(hi - hi0))) @ x_phase*du).ravel()
        coef, *_ = np.linalg.lstsq(V, diffs, rcond=None)
        coef = coef.reshape(degree, len(lags_dT), num_interp_points)
        tables[i, 0::2], tables[i, 1::2] = coef.real, coef.imag
    return jnp.asarray(tables)


@partial(jax.jit, static_argnums=0)
def apply_corrected_shift_fit(wdm, filter_tables, fit_tables, tables_2, eps_max,
                              wdm_coeff, delta, rate, curvature):
    r"""
    The variable time shift with the delay's rate handled through the fitted
    linear-warp element (`build_fit_tables`) instead of a Taylor series:

    .. math::

        \overline{W} \text{ with }
        W = \sum_{k=1}^K a_k\left(\frac{\dot\delta_n c}{\varepsilon_{\max}}\right)^k
          + \dot\delta_n V + \tfrac12\ddot\delta_n\kappa_c(c\,U_2+V_2),

    the stretch (:math:`V`) and curvature terms kept to first order, from
    `build_rate_tables_2`. Each power of :math:`c` has a rank-1 coefficient,
    so the lag sum keeps the structure of `apply_corrected_shift_2`, evaluated
    by Horner in the scaled frequency.
    """
    max_lag_L = (filter_tables.shape[1] - 1)//2
    degree = fit_tables.shape[1]//2

    n = jnp.arange(wdm.Nt)
    m = jnp.arange(wdm.Nf)
    sign_n = jnp.where(n % 2 == 0, 1.0, -1.0)[:, jnp.newaxis]
    sign_m = jnp.where(m % 2 == 0, 1.0, -1.0)[jnp.newaxis, :]

    theta = 2*jnp.pi*wdm.dF*jnp.outer(delta, m)
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    half = jnp.pi*wdm.dF*delta[:, jnp.newaxis]
    cos_h, sin_h = jnp.cos(half), jnp.sin(half)

    lags = jnp.arange(-max_lag_L, max_lag_L + 1)
    Tl = wdm._interp_filter(filter_tables[1], lags, -delta)
    Tprimel = wdm._interp_filter(filter_tables[0], lags, -delta)
    delay_filter = {-1: Tprimel, 0: Tl, +1: Tprimel}

    # The fit variable is z = rate c / eps_max. It is split as
    # (rate / rate_scale) (c rate_scale / eps_max) so both factors stay O(1):
    # powers of the scaled frequency carry rank-1 coefficients a_k (rate /
    # rate_scale)^k, plus the stretch and curvature terms in powers 0 and 1.
    rate_scale = jnp.max(jnp.abs(rate))
    c_scale = rate_scale/eps_max
    scaled_rate = (rate/rate_scale)[jnp.newaxis]
    def coefficients(sigma):
        look = lambda tab: wdm._interp_filter(tab, lags, -delta)
        out = [None]*(degree + 1)
        for k in range(1, degree + 1):
            w = scaled_rate**k
            out[k] = (w*look(fit_tables[sigma + 1, 2*k - 2]),
                      w*look(fit_tables[sigma + 1, 2*k - 1]))
        Vr, Vi = look(tables_2[sigma + 1, 2]), look(tables_2[sigma + 1, 3])
        U2r, U2i = look(tables_2[sigma + 1, 4]), look(tables_2[sigma + 1, 5])
        V2r, V2i = look(tables_2[sigma + 1, 6]), look(tables_2[sigma + 1, 7])
        r1 = rate[jnp.newaxis]
        rc = 0.5*curvature[jnp.newaxis]/(2*jnp.pi)
        out[0] = (r1*Vr + rc*V2i, r1*Vi - rc*V2r)
        # kappa_c c U2 term, with c = (scaled c) / c_scale
        out[1] = (out[1][0] + rc*U2i/c_scale, out[1][1] - rc*U2r/c_scale)
        return jnp.stack([part for pair in out for part in pair])
    poly = {sigma: coefficients(sigma) for sigma in (-1, 0, 1)}

    c_sigma = {sigma: ((m + 0.5*sigma)*wdm.dF*c_scale)[jnp.newaxis, :]
               for sigma in (-1, 0, 1)}

    cyclic_ext = jnp.concatenate([wdm_coeff[-max_lag_L:], wdm_coeff,
                                  wdm_coeff[:max_lag_L]], axis=0)
    cyclic_ext = jnp.concatenate([cyclic_ext[:, -1:], cyclic_ext,
                                  cyclic_ext[:, :1]], axis=1)
    i_pow = jnp.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])

    acc = jnp.zeros((wdm.Nt, wdm.Nf), dtype=wdm_coeff.dtype)
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(cyclic_ext, max_lag_L - l,
                                            wdm.Nt, axis=0)
        total = 0.0
        for sigma in (-1, 0, 1):
            coeff = jax.lax.slice_in_dim(rows, sigma + 1, sigma + 1 + wdm.Nf,
                                         axis=1)
            i_pow_re, i_pow_im = i_pow[l % 4]
            if sigma == 0:
                lag_phase_re, lag_phase_im = 1.0, 0.0
            else:
                lag_phase_re = i_pow_re
                lag_phase_im = i_pow_im if sigma == -1 else -i_pow_im
            cos_s = 1.0 if sigma == 0 else cos_h
            sin_s = 0.0 if sigma == 0 else sigma*sin_h
            re_a = lag_phase_re*cos_s + lag_phase_im*sin_s
            re_b = lag_phase_im*cos_s - lag_phase_re*sin_s
            im_a = lag_phase_im*cos_s - lag_phase_re*sin_s
            im_b = -(lag_phase_im*sin_s + lag_phase_re*cos_s)
            if (l + sigma) % 2 == 0:
                alternates = False
            else:
                re_a, re_b, im_a, im_b = (-sign_n*im_a, -sign_n*im_b,
                                          sign_n*re_a, sign_n*re_b)
                alternates = True
            if l % 2 != 0:
                alternates = not alternates

            T = delay_filter[sigma][i][:, jnp.newaxis]
            P = poly[sigma][:, i][:, :, jnp.newaxis]     # (2(K+1), Nt, 1)
            c = c_sigma[sigma]
            # Horner in c, the cos and sin weights together
            acc_c = re_a*P[2*degree] + im_a*P[2*degree + 1]
            acc_s = re_b*P[2*degree] + im_b*P[2*degree + 1]
            for k in range(degree - 1, -1, -1):
                acc_c = re_a*P[2*k] + im_a*P[2*k + 1] + c*acc_c
                acc_s = re_b*P[2*k] + im_b*P[2*k + 1] + c*acc_s
            term = coeff*(cos_t*(re_a*T + acc_c) + sin_t*(re_b*T + acc_s))
            total = total + (term*sign_m if alternates else term)
        acc = acc + total
    return acc
