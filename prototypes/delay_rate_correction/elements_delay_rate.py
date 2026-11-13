r"""
Closed-form matrix elements for the delay-rate term.

With :math:`\delta(t)\approx\delta_n+\dot\delta_n(t-t_n)` the shift picks up,
to first order in :math:`\dot\delta_n`,

.. math::

    \tilde\omega_{nm} \mathrel{+}= \dot\delta_n \sum_{l,\sigma}
        \omega_{(n-l)(m+\sigma)}\, Y_{n(n-l);m(m+\sigma)}(\delta_n),
    \qquad
    Y_{nn';mm'}(\delta) = \int\mathrm{d}t\,(t-t_n)\,
        \dot g_{n'm'}(t+\delta)\, g_{nm}(t) .

In the frequency domain :math:`(t-t_n)g_{nm}` turns the output window
:math:`\tilde\Phi(f-m\Delta F)` into :math:`\tfrac{i}{2\pi}\tilde\Phi'(f-m
\Delta F)` (the :math:`t_n` terms cancel) and the derivative brings
:math:`2\pi i f`. After :math:`f=u+(m+\sigma/2)\Delta F`, as for `X`, the
building block is

.. math::

    W^\sigma_{l,m}(x) = \int\mathrm{d}u\, e^{2\pi i u(l\Delta T - x)}
        \left[u+(m+\tfrac{\sigma}{2})\Delta F\right]
        \tilde\Phi(u-\tfrac{\sigma}{2}\Delta F)\,
        \tilde\Phi'(u+\tfrac{\sigma}{2}\Delta F)
    = (m+\tfrac{\sigma}{2})\Delta F\,U^\sigma_l(x) + V^\sigma_l(x) ,

which takes the place of :math:`T_l` / :math:`T'_l` in the `X` bookkeeping.
Unlike :math:`T_l`, :math:`W` is complex (:math:`U^0` is imaginary) and
:math:`\sigma=\pm1` need separate tables.

The phase bookkeeping is taken from `time_delay_matrix_X` and the result is
checked here against direct time-domain integrals of the definition, on the
small grid `test_time_delay_matrix_X_matches_direct_integral` uses. `X` is
rebuilt the same way first, as a control on the machinery.

    python prototypes/delay_rate_correction/elements_delay_rate.py
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax.scipy.special import betaln

from WDM.code.discrete_wavelet_transform import WDM
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


def u_grid(wdm, num=40001):
    edge = float(wdm.A + wdm.B)/(2*np.pi) + float(wdm.dF)
    u = np.linspace(-edge, edge, num)
    return u, u[1] - u[0]


def u_nodes(wdm, max_abs_lambda=100, per_piece=64):
    r"""
    Gauss-Legendre nodes and weights in :math:`f` for integrals of
    :math:`\tilde\Phi(f-\sigma\Delta F/2)` times :math:`\tilde\Phi` or its
    derivatives at :math:`f+\sigma\Delta F/2`, :math:`\sigma\in\{-1,0,1\}`.

    The pieces are split at every kink of those windows, so each integrand is
    smooth on each piece, with enough nodes to resolve
    :math:`e^{2\pi i f(\ell\Delta T-x)}` for :math:`|\ell-x/\Delta T|\le`
    `max_abs_lambda`.
    """
    A, edge = float(wdm.A_frac), 1. - float(wdm.A_frac)
    kinks = sorted({s + k for s in (-0.5, 0., 0.5) for k in (A, -A, edge, -edge)})
    nodes, weights = [], []
    for a, b in zip(kinks[:-1], kinks[1:]):
        num = per_piece + int(np.ceil(0.5*np.pi*max_abs_lambda*(b - a)))
        g, w = np.polynomial.legendre.leggauss(num)
        nodes.append(0.5*(b - a)*g + 0.5*(b + a))
        weights.append(0.5*(b - a)*w)
    dF = float(wdm.dF)
    return np.concatenate(nodes)*dF, np.concatenate(weights)*dF


def filter_T(wdm, l, x, sigma):
    """T_l (sigma=0) or T'_l (sigma=+-1) at delay x, evaluated analytically."""
    u, du = u_nodes(wdm)
    lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
    hi, _ = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
    return np.sum(np.exp(2j*np.pi*u*(l*wdm.dT - x))*lo*hi*du)


def filter_UV(wdm, l, x, sigma):
    """U^sigma_l(x) and V^sigma_l(x)."""
    u, du = u_nodes(wdm)
    lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
    _, dhi = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
    phase = np.exp(2j*np.pi*u*(l*wdm.dT - x))*lo*dhi*du
    return np.sum(phase), np.sum(u*phase)


def bookkeeping(wdm, n, m, l, sigma, delta):
    """The factors of `time_delay_matrix_X` other than the filter."""
    n_, m_ = n - l, m + sigma
    alternating = (-1.0)**((n - n_)*m)
    parity = np.conj(1j**((n + m) % 2))*1j**((n_ + m_) % 2)
    scalar, offset = {0: (1.0, 0.0),
                      -1: (1j**l, -0.5),
                      1: ((-1j)**l, 0.5)}[sigma]
    carrier = np.exp(-2j*np.pi*(m + offset)*wdm.dF*delta)
    return alternating*scalar*carrier*parity, offset


def X_closed(wdm, n, m, l, sigma, delta):
    factor, _ = bookkeeping(wdm, n, m, l, sigma, delta)
    return np.real(factor*filter_T(wdm, l, -delta, sigma))


def Y_closed(wdm, n, m, l, sigma, delta, variant):
    factor, offset = bookkeeping(wdm, n, m, l, sigma, delta)
    U, V = filter_UV(wdm, l, -delta, sigma)
    W = (m + offset)*wdm.dF*U + V
    if variant == "conj":
        W = np.conj(W)
    return np.real(factor*W)


def spectral(wdm, g, shift, derivative=False):
    freqs = np.fft.fftfreq(wdm.N, d=wdm.dt)
    kernel = np.exp(2j*np.pi*freqs*shift)
    if derivative:
        kernel = kernel*2j*np.pi*freqs
    return np.real(np.fft.ifft(np.fft.fft(g)*kernel))


def direct(wdm, n, m, l, sigma, delta):
    """X and Y from their defining integrals."""
    t = np.asarray(wdm.times)
    g_out = np.asarray(wdm.gnm(n, m))
    g_in = np.asarray(wdm.gnm(n - l, m + sigma))
    s = (t - n*wdm.dT + 0.5*wdm.T) % wdm.T - 0.5*wdm.T

    X = wdm.dt*np.sum(spectral(wdm, g_in, delta)*g_out)
    Y = wdm.dt*np.sum(s*spectral(wdm, g_in, delta, derivative=True)*g_out)
    return X, Y


def main():
    wdm = WDM.WDM_transform(dt=0.5, Nf=8, N=8*32)
    delta = 0.31*wdm.dT

    Phi, dPhi = window_and_derivative(wdm, np.asarray(wdm.freqs))
    fd = np.gradient(Phi, float(wdm.df))
    print(f"window vs window_FD: {np.max(np.abs(Phi - np.asarray(wdm.window_FD))):.1e}; "
          f"dPhi vs finite difference: "
          f"{np.max(np.abs(dPhi - fd))/np.max(np.abs(dPhi)):.1e} (rel)")

    worst = {"X": 0.0, "Y": 0.0, "Yconj": 0.0}
    scale = {"X": 0.0, "Y": 0.0}
    for l in (-2, -1, 0, 1, 2):
        for sigma in (-1, 0, 1):
            for n in (14, 15, 16, 17):
                for m in range(1, wdm.Nf - 1):
                    if not 1 <= m + sigma <= wdm.Nf - 2:
                        continue
                    Xd, Yd = direct(wdm, n, m, l, sigma, delta)
                    scale["X"] = max(scale["X"], abs(Xd))
                    scale["Y"] = max(scale["Y"], abs(Yd))
                    worst["X"] = max(worst["X"],
                                     abs(X_closed(wdm, n, m, l, sigma, delta) - Xd))
                    worst["Y"] = max(worst["Y"],
                                     abs(Y_closed(wdm, n, m, l, sigma, delta, "") - Yd))
                    worst["Yconj"] = max(worst["Yconj"],
                                         abs(Y_closed(wdm, n, m, l, sigma, delta, "conj") - Yd))

    print(f"max |X| {scale['X']:.3f}, |Y| {scale['Y']:.3f}")
    for key, val in worst.items():
        print(f"{key:6s} max abs error {val:.2e}")


if __name__ == "__main__":
    main()
