r"""
Matrix elements for the first-order sinc term.

The one-arm amplitude carries :math:`S(f)=\mathrm{sinc}(f\tau_n)`, a
time-invariant filter (the window average). In the wavelet basis, at zero
delay, its elements are :math:`X(0)`'s bookkeeping with the filter

.. math::

    \int\mathrm{d}u\,e^{2\pi i u l\Delta T}
        \tilde\Phi(u-\tfrac{\sigma}{2}\Delta F)\tilde\Phi(u+\tfrac{\sigma}{2}\Delta F)
        \,S(u+c), \qquad c=(m+\tfrac{\sigma}{2})\Delta F .

The zeroth order, :math:`S(c)X(0)`, is the production per-pixel product
(:math:`X(0)` is the identity). The first order is :math:`S'(c)` times

.. math::

    T^{(1)\sigma}_l = \int\mathrm{d}u\,u\,e^{2\pi i u l\Delta T}
        \tilde\Phi(u-\tfrac{\sigma}{2}\Delta F)\tilde\Phi(u+\tfrac{\sigma}{2}\Delta F) .

Its phase convention is fixed here against the direct integral of the
defining element with the filter :math:`R(f) = |f| - c` (even in f), the
exact first-order coefficient.

    python prototypes/delay_rate_correction/elements_sinc.py
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from WDM.code.discrete_wavelet_transform import WDM
from elements_delay_rate import bookkeeping, u_grid, window_and_derivative


def T1(wdm, l, sigma):
    u, du = u_grid(wdm)
    lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
    hi, _ = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
    return np.sum(u*np.exp(2j*np.pi*u*l*wdm.dT)*lo*hi)*du


def direct(wdm, n, m, l, sigma):
    c = (m + 0.5*sigma)*wdm.dF
    freqs = np.fft.fftfreq(wdm.N, d=wdm.dt)
    g_in = np.asarray(wdm.gnm(n - l, m + sigma))
    filtered = np.real(np.fft.ifft(np.fft.fft(g_in)*(np.abs(freqs) - c)))
    return wdm.dt*np.sum(filtered*np.asarray(wdm.gnm(n, m)))


def main():
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32)
    variants = {f"{name}{'*' if conj else ''}": (ph, conj)
                for name, ph in (("1", 1), ("-1", -1), ("i", 1j), ("-i", -1j))
                for conj in (False, True)}
    worst = dict.fromkeys(variants, 0.0)
    scale = 0.0
    for l in (-2, -1, 0, 1, 2):
        for sigma in (-1, 0, 1):
            t1 = T1(wdm, l, sigma)
            for n in (14, 15, 16, 17):
                for m in range(1, wdm.Nf - 1):
                    if not 1 <= m + sigma <= wdm.Nf - 2:
                        continue
                    d = direct(wdm, n, m, l, sigma)
                    scale = max(scale, abs(d))
                    factor, _ = bookkeeping(wdm, n, m, l, sigma, 0.0)
                    for key, (ph, conj) in variants.items():
                        z = ph*(np.conj(t1) if conj else t1)
                        worst[key] = max(worst[key], abs(np.real(factor*z) - d))
    print(f"max |element| {scale:.3e} Hz")
    for key, val in worst.items():
        print(f"{key:>4}: {val:.2e}")


if __name__ == "__main__":
    main()
