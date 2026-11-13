r"""
How well a polynomial in the window offset represents the
exact linear-warp element.

For a linear warp the shift element's filter is

.. math::

    G^\sigma_l(x,\varepsilon) = \int\mathrm{d}u\,e^{2\pi i u(l\Delta T-x)}
        \tilde\Phi(u-\tfrac{\sigma}{2}\Delta F)\,
        \tilde\Phi(u+\tfrac{\sigma}{2}\Delta F+\varepsilon),
    \qquad \varepsilon=\dot\delta\,(m+\tfrac{\sigma}{2})\Delta F,

(the stretch term :math:`\dot\delta u` neglected). Taylor-expanding it at
:math:`\varepsilon=0` is what T1, T2 do; here a least-squares Chebyshev fit of
degree K over :math:`|\varepsilon|\le\varepsilon_{\max}` (powers 1..K, so it
vanishes at zero as :math:`G-T` does, on Chebyshev nodes) is tried instead, and
its worst error over the range reported relative to :math:`\max|G|`.

    python prototypes/delay_rate_correction/fit_window_offset.py [Nf]
"""
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from WDM.code.discrete_wavelet_transform import WDM
from elements_delay_rate import u_nodes, window_and_derivative

RATE_MAX = 9.9e-5


def G_table(wdm, sigma, lags, x, eps):
    out = np.empty((len(eps), len(lags), len(x)), complex)
    for k, e in enumerate(eps):
        # the offset window moves its kinks, so each offset gets its own nodes
        u, du = u_nodes(wdm, offset=e)
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        hi, _ = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF + e)
        lag_phase = np.exp(2j*np.pi*np.outer(lags*wdm.dT, u))
        x_phase = np.exp(-2j*np.pi*np.outer(u, x))
        out[k] = (lag_phase*(lo*hi*du)) @ x_phase
    return out


def main(Nf=7906):
    wdm = WDM.WDM_transform(dt=4.0, Nf=Nf, N=Nf*200)
    lags = np.arange(-25, 26)
    x = np.linspace(-0.5*wdm.dT, 0.5*wdm.dT, 33)
    for label, f_top in (("30 mHz", 3.0e-2), ("Nyquist", 0.125)):
        e_max = RATE_MAX*f_top
        eps = e_max*np.cos(np.pi*(np.arange(121) + 0.5)/121)   # Chebyshev nodes
        print(f"Nf={Nf} dT={wdm.dT:.4g} s, {label}: eps_max = {e_max:.2e} Hz = "
              f"{e_max/wdm.dF:.2f} dF", flush=True)
        for sigma in (0, 1):
            G = G_table(wdm, sigma, lags, x, eps)
            G0 = G_table(wdm, sigma, lags, x, np.array([0.0]))[0]
            scale = np.max(np.abs(G))
            y = (G - G0[None]).reshape(len(eps), -1)        # the correction part
            z = eps/e_max
            line = []
            for K in (2, 3, 4, 6, 8, 10):
                # powers z^1..z^K vanish at 0, as G - T does (Chebyshev T_2k do not)
                V = np.vander(z, K + 1, increasing=True)[:, 1:]
                coef, *_ = np.linalg.lstsq(V, y, rcond=None)
                err = np.max(np.abs(V @ coef - y))/scale
                line.append(f"K={K}: {err:.1e}")
            print(f"  sigma={sigma:+d} | max|G - T|/max|G| {np.max(np.abs(y))/scale:.1e} | "
                  + "  ".join(line), flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
