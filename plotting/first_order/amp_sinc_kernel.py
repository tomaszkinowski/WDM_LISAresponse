r"""
The amplitude-rate term of the unshifted grid, as one jitted lag sum.

.. math::

    \sum_{l,\sigma}\omega_{(n-l)(m+\sigma)} Z_{n(n-l);m(m+\sigma)},
    \qquad
    Z = \mathrm{Re}\!\left[f\,\overline{\kappa\,U^\sigma_l(0)}\right],
    \quad \kappa = -\frac{i}{2\pi},

with :math:`f` the zero-delay bookkeeping factor of :math:`X`
(`elements_amplitude.py` fixes :math:`\kappa`). Multiplied by
:math:`\dot A_{nm}` and added to :math:`A_{nm}\omega_{nm}`, it gives the
unshifted grid to first order in the amplitude's rate.

At zero delay the carrier is one, so every element is a constant per
:math:`(l,\sigma)` times :math:`(-1)^{lm}` and, when :math:`l+\sigma` is odd,
:math:`(-1)^{n+m}`: the sum needs no tables in :math:`\delta`.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from .elements_delay_rate import u_nodes, window_and_derivative


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


@partial(jax.jit, static_argnums=0)
def apply_amplitude_rate(wdm, weights, wdm_coeff):
    r"""
    :math:`\sum_{l,\sigma}\omega_{(n-l)(m+\sigma)}Z_{n(n-l);m(m+\sigma)}`,
    wrapped periodically like the shift.

    Parameters
    ----------
    wdm : WDM_transform
        Static.
    weights : jnp.ndarray
        From `amp_weights`, shape (3, 2L+1).
    wdm_coeff : jnp.ndarray
        shape (Nt, Nf).
    """
    max_lag_L = (weights.shape[1] - 1)//2
    n = jnp.arange(wdm.Nt)
    m = jnp.arange(wdm.Nf)
    sign_n = jnp.where(n % 2 == 0, 1.0, -1.0)[:, jnp.newaxis]
    sign_m = jnp.where(m % 2 == 0, 1.0, -1.0)[jnp.newaxis, :]

    cyclic_ext = jnp.concatenate([wdm_coeff[-max_lag_L:], wdm_coeff,
                                  wdm_coeff[:max_lag_L]], axis=0)
    cyclic_ext = jnp.concatenate([cyclic_ext[:, -1:], cyclic_ext,
                                  cyclic_ext[:, :1]], axis=1)

    acc = jnp.zeros((wdm.Nt, wdm.Nf), dtype=wdm_coeff.dtype)
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(cyclic_ext, max_lag_L - l,
                                            wdm.Nt, axis=0)
        same, flipped = 0.0, 0.0           # without / with (-1)^m
        for k, sigma in enumerate((-1, 0, 1)):
            coeff = jax.lax.slice_in_dim(rows, sigma + 1, sigma + 1 + wdm.Nf,
                                         axis=1)
            term = weights[k, i]*coeff
            odd = (l + sigma) % 2 != 0     # carries (-1)^(n+m)
            if odd:
                term = term*sign_n
            if odd != (l % 2 != 0):        # (-1)^m from parity xor (-1)^(lm)
                flipped = flipped + term
            else:
                same = same + term
        acc = acc + same + flipped*sign_m
    return acc


def build_sinc_table(wdm, max_lag_L):
    r"""
    :math:`\overline{T^{(1)\sigma}_l}`, the first-order sinc elements of
    `elements_sinc.py`, shape (3, 2L+1) complex; pass to `amp_weights`.
    """
    u, du = u_nodes(wdm)
    lags_dT = np.arange(-max_lag_L, max_lag_L + 1)*wdm.dT
    out = np.empty((3, len(lags_dT)), complex)
    for i, sigma in enumerate((-1, 0, 1)):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        hi, _ = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        out[i] = np.conj(np.exp(2j*np.pi*np.outer(lags_dT, u)) @ (u*lo*hi*du))
    return out


@partial(jax.jit, static_argnums=0)
def apply_sinc_rate(wdm, weights, wdm_coeff, slope):
    r"""
    :math:`\sum_{l,\sigma}\omega_{(n-l)(m+\sigma)}\,S'_\sigma(n,m)\,
    E^{(1)}_{n(n-l);m(m+\sigma)}`: the first-order sinc term, with
    :math:`S'_\sigma(n,m)` the slope of :math:`\mathrm{sinc}(f\tau_n)` at
    :math:`f=(m+\sigma/2)\Delta F`, shape (3, Nt, Nf) indexed by sigma+1.

    As `apply_amplitude_rate`, except that the slope depends on sigma, so
    each sigma keeps its own lag sum and the slope multiplies each one.
    """
    max_lag_L = (weights.shape[1] - 1)//2
    n = jnp.arange(wdm.Nt)
    m = jnp.arange(wdm.Nf)
    sign_n = jnp.where(n % 2 == 0, 1.0, -1.0)[:, jnp.newaxis]
    sign_m = jnp.where(m % 2 == 0, 1.0, -1.0)[jnp.newaxis, :]

    cyclic_ext = jnp.concatenate([wdm_coeff[-max_lag_L:], wdm_coeff,
                                  wdm_coeff[:max_lag_L]], axis=0)
    cyclic_ext = jnp.concatenate([cyclic_ext[:, -1:], cyclic_ext,
                                  cyclic_ext[:, :1]], axis=1)

    acc = {sigma: jnp.zeros((wdm.Nt, wdm.Nf), dtype=wdm_coeff.dtype)
           for sigma in (-1, 0, 1)}
    for i in range(2*max_lag_L + 1):
        l = i - max_lag_L
        rows = jax.lax.dynamic_slice_in_dim(cyclic_ext, max_lag_L - l,
                                            wdm.Nt, axis=0)
        for k, sigma in enumerate((-1, 0, 1)):
            coeff = jax.lax.slice_in_dim(rows, sigma + 1, sigma + 1 + wdm.Nf,
                                         axis=1)
            term = weights[k, i]*coeff
            odd = (l + sigma) % 2 != 0
            if odd:
                term = term*sign_n
            if odd != (l % 2 != 0):
                term = term*sign_m
            acc[sigma] = acc[sigma] + term
    # the slope depends on (n, m, sigma) only, so it multiplies each sigma's
    # lag sum once, after the sum
    return sum(slope[k]*acc[sigma] for k, sigma in enumerate((-1, 0, 1)))


def sinc_slope(tau, wavelet_freqs, dF):
    r"""
    :math:`\partial_f\,\mathrm{sinc}(f\tau_n)` at :math:`f=f_m+\sigma\Delta F/2`
    for sigma = -1, 0, 1, shape (3, Nt, Nf). numpy's normalised sinc, as
    `compute_detector_amplitudes` uses.
    """
    out = []
    for sigma in (-1, 0, 1):
        x = tau[:, jnp.newaxis]*(wavelet_freqs + 0.5*sigma*dF)[jnp.newaxis, :]
        safe = jnp.where(x == 0, 1.0, x)
        d = jnp.where(x == 0, 0.0, (jnp.cos(jnp.pi*safe) - jnp.sinc(safe))/safe)
        out.append(d*tau[:, jnp.newaxis])
    return jnp.stack(out)
