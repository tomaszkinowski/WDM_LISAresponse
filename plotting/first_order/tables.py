r"""
The correction tables of the first-order response, from the Meyer window
:math:`\tilde\Phi` and its derivative by Gauss-Legendre quadrature over
:math:`u = f - (m+\sigma/2)\Delta F`.

Delay rate. With :math:`\delta(t)\approx\delta_n+\dot\delta_n(t-t_n)` the shift
gains, to first order in :math:`\dot\delta_n`, a lag sum with the element
:math:`Y_{nn';mm'}(\delta)=\int(t-t_n)\,\dot g_{n'm'}(t+\delta)\,g_{nm}(t)\,dt`,
built from

.. math::

    W^\sigma_{l,m}(x) = (m+\tfrac{\sigma}{2})\Delta F\,U^\sigma_l(x) + V^\sigma_l(x),
    \quad
    U^\sigma_l(x) = \int du\, e^{2\pi i u(l\Delta T - x)}
        \tilde\Phi(u-\tfrac{\sigma}{2}\Delta F)\,\tilde\Phi'(u+\tfrac{\sigma}{2}\Delta F),

with :math:`V` the same integral with an extra factor :math:`u`
(`build_rate_tables`, tabulated on the filter tables' delay grid).

Amplitude rate. At zero delay the element is
:math:`Z=\mathrm{Re}[f\,\overline{\kappa U^\sigma_l(0)}]`, :math:`\kappa=-i/2\pi`,
a constant per :math:`(l,\sigma)` (`build_amp_table`).

Sinc. The window-average sinc to first order has the element
:math:`\overline{T^{(1)\sigma}_l}`, the zero-delay integral of
:math:`u\,\tilde\Phi\tilde\Phi` (`build_sinc_table`).

`amp_weights` turns either zero-delay table into real per-:math:`(l,\sigma)` weights.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax.scipy.special import betaln

from WDM.code.utils.Meyer import Meyer, nu_d


def window_and_derivative(wdm, f):
    r"""
    :math:`\tilde\Phi(f)` as `wdm.window_FD` defines it,
    :math:`\sqrt{2\pi}\,\mathrm{Meyer}(2\pi f)`, and its derivative in
    :math:`f`, both evaluated analytically at arbitrary frequencies.
    """
    f = np.asarray(f, dtype=float)
    omega = 2*np.pi*f
    absw = np.abs(omega)
    norm = np.sqrt(2*np.pi)/np.sqrt(float(wdm.dOmega))

    Phi = np.sqrt(2*np.pi)*np.asarray(Meyer(jnp.asarray(omega), wdm.d,
                                            wdm.A, wdm.B))

    x = np.clip((absw - float(wdm.A))/float(wdm.B), 0.0, 1.0)
    nu = np.asarray(nu_d(jnp.asarray(x), wdm.d))
    dnu = np.exp((wdm.d - 1)*np.log(np.maximum(x*(1 - x), 1e-300))
                 - float(betaln(wdm.d, wdm.d)))
    in_ramp = (absw > wdm.A) & (absw < wdm.A + wdm.B)
    dPhi_domega = np.where(in_ramp,
                           -norm*np.sin(0.5*np.pi*nu)*0.5*np.pi*dnu
                           * np.sign(omega)/float(wdm.B),
                           0.0)

    return Phi, 2*np.pi*dPhi_domega

def u_nodes(wdm, max_abs_lambda=100, per_piece=64, offset=0.0):
    r"""
    Gauss-Legendre nodes and weights in :math:`f` for integrals of
    :math:`\tilde\Phi(f-\sigma\Delta F/2)` times :math:`\tilde\Phi` or its
    derivatives at :math:`f+\sigma\Delta F/2+` `offset`,
    :math:`\sigma\in\{-1,0,1\}`.

    The pieces are split at every kink of those windows, with and without the
    offset, so each integrand is smooth on each piece, with enough nodes to
    resolve :math:`e^{2\pi i f(\ell\Delta T-x)}` for
    :math:`|\ell-x/\Delta T|\le` `max_abs_lambda`.
    """
    A, edge = float(wdm.A_frac), 1. - float(wdm.A_frac)
    # Phi(u + s + offset) has its kinks at u = k - s - offset; s runs over both signs
    shift = float(offset)/float(wdm.dF)
    kinks = sorted({s + k - o for s in (-0.5, 0., 0.5) for k in (A, -A, edge, -edge)
                    for o in (0., shift)})
    nodes, weights = [], []
    for a, b in zip(kinks[:-1], kinks[1:]):
        num = per_piece + int(np.ceil(0.5*np.pi*max_abs_lambda*(b - a)))
        g, w = np.polynomial.legendre.leggauss(num)
        nodes.append(0.5*(b - a)*g + 0.5*(b + a))
        weights.append(0.5*(b - a)*w)
    dF = float(wdm.dF)
    return np.concatenate(nodes)*dF, np.concatenate(weights)*dF

def build_rate_tables(wdm, max_lag_L, num_interp_points):
    r"""
    :math:`U^\sigma_l`, :math:`V^\sigma_l` on `wdm`'s delay grid,
    :math:`x\in[-\Delta T/2,\Delta T/2]` with `num_interp_points` samples.

    Returns
    -------
    tables : jnp.ndarray
        shape=(3, 4, 2L+1, P), indexed [sigma+1, (U_r, U_i, V_r, V_i)].
    """
    u, du = u_nodes(wdm)
    lags_dT = np.arange(-max_lag_L, max_lag_L + 1)*wdm.dT
    x = np.linspace(-0.5*wdm.dT, 0.5*wdm.dT, num_interp_points)
    lag_phase = np.exp(2j*np.pi*np.outer(lags_dT, u))
    x_phase = np.exp(-2j*np.pi*np.outer(u, x))

    tables = np.empty((3, 4, len(lags_dT), num_interp_points))
    for i, sigma in enumerate((-1, 0, 1)):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        _, dhi = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        U = (lag_phase*(lo*dhi*du)) @ x_phase
        V = (lag_phase*(u*lo*dhi*du)) @ x_phase
        tables[i] = np.stack([U.real, U.imag, V.real, V.imag])
    return jnp.asarray(tables)

def build_amp_table(wdm, max_lag_L):
    r"""
    :math:`\overline{\kappa U^\sigma_l(0)}` for :math:`\sigma=-1,0,1` and
    :math:`|l|\le L`, shape (3, 2L+1) complex.
    """
    u, du = u_nodes(wdm)
    lags_dT = np.arange(-max_lag_L, max_lag_L + 1)*wdm.dT
    out = np.empty((3, len(lags_dT)), complex)
    for i, sigma in enumerate((-1, 0, 1)):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        _, dhi = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        U = np.exp(2j*np.pi*np.outer(lags_dT, u)) @ (lo*dhi*du)
        out[i] = np.conj(-1j/(2*np.pi)*U)
    return out

def amp_weights(amp_table):
    r"""
    The real per-(l, sigma) weights: ``(weight, alternates_nm)`` as nested
    python data, so the jitted sum unrolls with static structure. Returned as
    a (3, 2L+1) array of weights; whether :math:`(-1)^{n+m}` applies is
    :math:`l+\sigma` odd.
    """
    max_lag_L = (amp_table.shape[1] - 1)//2
    w = np.empty(amp_table.shape)
    for i, sigma in enumerate((-1, 0, 1)):
        for j, l in enumerate(range(-max_lag_L, max_lag_L + 1)):
            scalar = {0: 1.0, -1: 1j**l, 1: (-1j)**l}[sigma]
            z = scalar*amp_table[i, j]
            w[i, j] = np.real(z) if (l + sigma) % 2 == 0 else np.real(1j*z)
    return jnp.asarray(w)

def build_sinc_table(wdm, max_lag_L):
    r"""
    :math:`\overline{T^{(1)\sigma}_l}`, the first-order sinc elements, shape
    (3, 2L+1) complex; pass to `amp_weights`.
    """
    u, du = u_nodes(wdm)
    lags_dT = np.arange(-max_lag_L, max_lag_L + 1)*wdm.dT
    out = np.empty((3, len(lags_dT)), complex)
    for i, sigma in enumerate((-1, 0, 1)):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        hi, _ = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        out[i] = np.conj(np.exp(2j*np.pi*np.outer(lags_dT, u)) @ (u*lo*hi*du))
    return out
