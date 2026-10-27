r"""
Closed-form matrix elements for the amplitude-rate term.

With :math:`A(t)\approx A_n+\dot A_n(t-t_n)` the unshifted grid
:math:`W(A h)` picks up

.. math::

    W(Ah)_{nm} \approx A_{n}\omega_{nm}
        + \dot A_{n}\sum_{l,\sigma}\omega_{(n-l)(m+\sigma)}
          Z_{n(n-l);m(m+\sigma)},
    \qquad
    Z_{nn';mm'} = \int\mathrm{d}t\,(t-t_n)\,g_{n'm'}(t)\,g_{nm}(t) .

This is :math:`Y` at zero delay without the derivative, so in the frequency
domain only :math:`U^\sigma_l(0)` survives, with a constant in place of the
:math:`2\pi i f` the derivative brought. Its value and phase are fixed here
against direct integrals, trying the four unit phases over :math:`2\pi`.

    python prototypes/delay_rate_correction/elements_amplitude.py
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from WDM.code.discrete_wavelet_transform import WDM
from elements_delay_rate import bookkeeping, filter_UV


def direct_Z(wdm, n, m, l, sigma):
    t = np.asarray(wdm.times)
    s = (t - n*wdm.dT + 0.5*wdm.T) % wdm.T - 0.5*wdm.T
    return wdm.dt*np.sum(s*np.asarray(wdm.gnm(n - l, m + sigma))
                         * np.asarray(wdm.gnm(n, m)))


def closed_Z(wdm, n, m, l, sigma, kappa):
    factor, _ = bookkeeping(wdm, n, m, l, sigma, 0.0)
    U, _ = filter_UV(wdm, l, 0.0, sigma)
    return np.real(factor*np.conj(kappa*U))


def main():
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32)
    kappas = {"1/2pi": 1/(2*np.pi), "-1/2pi": -1/(2*np.pi),
              "i/2pi": 1j/(2*np.pi), "-i/2pi": -1j/(2*np.pi)}
    worst = dict.fromkeys(kappas, 0.0)
    scale = 0.0
    for l in (-2, -1, 0, 1, 2):
        for sigma in (-1, 0, 1):
            for n in (14, 15, 16, 17):
                for m in range(1, wdm.Nf - 1):
                    if not 1 <= m + sigma <= wdm.Nf - 2:
                        continue
                    Zd = direct_Z(wdm, n, m, l, sigma)
                    scale = max(scale, abs(Zd))
                    for key, kappa in kappas.items():
                        worst[key] = max(worst[key],
                                         abs(closed_Z(wdm, n, m, l, sigma, kappa) - Zd))
    print(f"max |Z| {scale:.3f} s")
    for key, val in worst.items():
        print(f"kappa {key:>7}: max abs error {val:.2e} s")


if __name__ == "__main__":
    main()
