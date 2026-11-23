r"""
Closed-form matrix elements for the second-order shift terms.

Taking :math:`f(t+\delta_n+\dot\delta_n s+\tfrac12\ddot\delta_n s^2)` to
second order in the rate, and to first in the curvature, :math:`s=t-t_n`,
adds

.. math::

    \tfrac12\dot\delta_n^2 \sum\omega\, Y^{(2)}
    + \tfrac12\ddot\delta_n \sum\omega\, Y^{(c)},
    \qquad
    Y^{(2)} = \int s^2\,\ddot g_{n'm'}(t+\delta)\,g_{nm}(t)\,\mathrm{d}t,
    \quad
    Y^{(c)} = \int s^2\,\dot g_{n'm'}(t+\delta)\,g_{nm}(t)\,\mathrm{d}t .

In the frequency domain :math:`s^2 g_{nm}` brings :math:`\tilde\Phi''` and the
derivatives powers of :math:`f`; after :math:`f=u+c`,
:math:`c=(m+\sigma/2)\Delta F`, the building blocks are

.. math::

    W^{(2)} = \kappa_2\,(c^2 U_2 + 2c V_2 + W_2), \qquad
    W^{(c)} = \kappa_c\,(c\,U_2 + V_2),

with :math:`U_2, V_2, W_2 = \int u^{0,1,2} e^{2\pi i u(l\Delta T - x)}
\tilde\Phi(u-\sigma\Delta F/2)\tilde\Phi''(u+\sigma\Delta F/2)\,\mathrm{d}u`
and each element :math:`\mathrm{Re}[f\,\overline{W}]` as for :math:`Y`. The
constants are fixed here against direct integrals.

    python prototypes/delay_rate_correction/elements_second_order.py
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from jax.scipy.special import betaln

from WDM.code.discrete_wavelet_transform import WDM
from .elements_delay_rate import bookkeeping, spectral, u_nodes, window_and_derivative
from WDM.code.utils.Meyer import nu_d


def window_second_derivative(wdm, f):
    r""":math:`\tilde\Phi''(f)`, analytically, for `wdm`'s window."""
    f = np.asarray(f, dtype=float)
    absw = np.abs(2*np.pi*f)
    norm = np.sqrt(2*np.pi)/np.sqrt(float(wdm.dOmega))
    x = np.clip((absw - float(wdm.A))/float(wdm.B), 1e-300, 1 - 1e-16)
    nu = np.asarray(nu_d(np.asarray(x), wdm.d))
    dnu = np.exp((wdm.d - 1)*np.log(x*(1 - x)) - float(betaln(wdm.d, wdm.d)))
    ddnu = dnu*(wdm.d - 1)*(1/x - 1/(1 - x))
    in_ramp = (absw > wdm.A) & (absw < wdm.A + wdm.B)
    d2_domega2 = -norm*((0.5*np.pi)**2*np.cos(0.5*np.pi*nu)*dnu**2
                        + 0.5*np.pi*np.sin(0.5*np.pi*nu)*ddnu)/float(wdm.B)**2
    return np.where(in_ramp, d2_domega2, 0.0)*(2*np.pi)**2


def filter_second(wdm, l, x, sigma):
    """U_2, V_2, W_2 at delay x."""
    u, du = u_nodes(wdm)
    lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
    d2hi = window_second_derivative(wdm, u + 0.5*sigma*wdm.dF)
    phase = np.exp(2j*np.pi*u*(l*wdm.dT - x))*lo*d2hi
    return np.sum(phase*du), np.sum(u*phase*du), np.sum(u*u*phase*du)


def closed(wdm, n, m, l, sigma, delta, kappa2, kappac):
    factor, offset = bookkeeping(wdm, n, m, l, sigma, delta)
    c = (m + offset)*wdm.dF
    U2, V2, W2 = filter_second(wdm, l, -delta, sigma)
    return (np.real(factor*np.conj(kappa2*(c*c*U2 + 2*c*V2 + W2))),
            np.real(factor*np.conj(kappac*(c*U2 + V2))))


def direct(wdm, n, m, l, sigma, delta):
    t = np.asarray(wdm.times)
    g_out = np.asarray(wdm.gnm(n, m))
    g_in = np.asarray(wdm.gnm(n - l, m + sigma))
    s = (t - n*wdm.dT + 0.5*wdm.T) % wdm.T - 0.5*wdm.T
    freqs = np.fft.fftfreq(wdm.N, d=wdm.dt)
    G = np.fft.fft(g_in)*np.exp(2j*np.pi*freqs*delta)
    d1 = np.real(np.fft.ifft(G*2j*np.pi*freqs))
    d2 = np.real(np.fft.ifft(G*(2j*np.pi*freqs)**2))
    return (wdm.dt*np.sum(s*s*d2*g_out), wdm.dt*np.sum(s*s*d1*g_out))


def main():
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32)

    f = np.asarray(wdm.freqs)
    _, d1 = window_and_derivative(wdm, f)
    fd = np.gradient(d1, float(wdm.df))
    d2 = window_second_derivative(wdm, f)
    fine = np.linspace(-1.2*wdm.dF, 1.2*wdm.dF, 200001)
    _, d1f = window_and_derivative(wdm, fine)
    fdf = np.gradient(d1f, fine[1] - fine[0])
    print(f"Phi'' vs finite difference of Phi' on a fine grid: "
          f"{np.max(np.abs(window_second_derivative(wdm, fine) - fdf))/np.max(np.abs(fdf)):.1e} rel")

    phases = {"1": 1, "-1": -1, "i": 1j, "-i": -1j}
    delta = 0.31*wdm.dT
    worst2 = {k: 0.0 for k in phases}
    worstc = {k: 0.0 for k in phases}
    scale2 = scalec = 0.0
    for l in (-2, -1, 0, 1, 2):
        for sigma in (-1, 0, 1):
            for n in (14, 15, 16, 17):
                for m in range(1, wdm.Nf - 1):
                    if not 1 <= m + sigma <= wdm.Nf - 2:
                        continue
                    Y2d, Ycd = direct(wdm, n, m, l, sigma, delta)
                    scale2, scalec = max(scale2, abs(Y2d)), max(scalec, abs(Ycd))
                    for k, ph in phases.items():
                        Y2, _ = closed(wdm, n, m, l, sigma, delta, ph, 0)
                        _, Yc = closed(wdm, n, m, l, sigma, delta, 0, ph/(2*np.pi))
                        worst2[k] = max(worst2[k], abs(Y2 - Y2d))
                        worstc[k] = max(worstc[k], abs(Yc - Ycd))
    print(f"max |Y2| {scale2:.3f}, |Yc| {scalec:.3f}")
    for k in phases:
        print(f"kappa2 = {k:>2}: {worst2[k]:.2e}    kappac = {k:>2}/2pi: {worstc[k]:.2e}")


if __name__ == "__main__":
    main()
