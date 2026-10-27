r"""
The delay-rate-corrected shift on the bare-shift test.

.. math::

    \tilde\omega_{nm} = \underbrace{\sum_{l,\sigma}\omega_{(n-l)(m+\sigma)}
        X_{n(n-l);m(m+\sigma)}(-\delta_n)}_{\texttt{apply\_variable\_time\_shift}}
      + \dot\delta_n \sum_{l,\sigma}\omega_{(n-l)(m+\sigma)}
        Y_{n(n-l);m(m+\sigma)}(\delta_n)

with :math:`Y` from `elements_delay_rate.py`: the `X` bookkeeping with the
filter replaced by :math:`\overline{W}`,
:math:`W=(m+\sigma/2)\Delta F\,U^\sigma_l+V^\sigma_l`. :math:`U^\sigma_l` and
:math:`V^\sigma_l` are tabulated at :math:`x=-\delta` on a uniform grid of
`NUM_INTERP_POINTS` delays over :math:`\pm\Delta T/2` and interpolated
linearly, as the `X` tables are. The frequency integrals use the analytic
window, so no half-bin issue arises.

Same tones, delay and record as `oracle_local_delay.py`. Reports the corrected
shift against the true warp, beside the uncorrected operator and the ``T1``
oracle it should reproduce.

    python prototypes/delay_rate_correction/reference_delay_rate_shift.py [log2_samples]
"""
import sys
import time

import numpy as np
import jax.numpy as jnp

import oracle_local_delay as s
from elements_delay_rate import u_grid, window_and_derivative


CASES = [(6, 0.0), (6, 0.49), (20, 0.0), (20, 0.4), (20, 0.49),
         (60, 0.0), (60, 0.25), (60, 0.4), (60, 0.49)]


def build_UV_tables(wdm, max_lag_L, num_interp_points):
    r"""
    :math:`U^\sigma_l(x)`, :math:`V^\sigma_l(x)` for :math:`\sigma=-1,0,1`,
    shape (3, 2L+1, P) complex each, on :math:`x\in[-\Delta T/2,\Delta T/2]`.
    """
    u, du = u_grid(wdm)
    lags_dT = np.arange(-max_lag_L, max_lag_L + 1)*wdm.dT
    x = np.linspace(-0.5*wdm.dT, 0.5*wdm.dT, num_interp_points)

    lag_phase = np.exp(2j*np.pi*np.outer(lags_dT, u))
    x_phase = np.exp(-2j*np.pi*np.outer(u, x))

    U = np.empty((3, len(lags_dT), num_interp_points), complex)
    V = np.empty_like(U)
    for i, sigma in enumerate((-1, 0, 1)):
        lo, _ = window_and_derivative(wdm, u - 0.5*sigma*wdm.dF)
        _, dhi = window_and_derivative(wdm, u + 0.5*sigma*wdm.dF)
        U[i] = (lag_phase*(lo*dhi)) @ x_phase*du
        V[i] = (lag_phase*(u*lo*dhi)) @ x_phase*du
    return U, V, x


def interp(table_l, x_grid, x):
    """Linear interpolation of one lag's table at delays x (inside the grid)."""
    return np.interp(x, x_grid, table_l.real) + 1j*np.interp(x, x_grid,
                                                             table_l.imag)


def rate_term(wdm, U, V, x_grid, w, delta_n):
    r"""
    :math:`\sum_{l,\sigma}\omega_{(n-l)(m+\sigma)}Y_{n(n-l);m(m+\sigma)}
    (\delta_n)`, wrapped periodically like the shift itself.
    """
    max_lag_L = (U.shape[1] - 1)//2
    n = np.arange(wdm.Nt)[:, None]
    m = np.arange(wdm.Nf)[None, :]
    out = np.zeros((wdm.Nt, wdm.Nf))

    for i, sigma in enumerate((-1, 0, 1)):
        offset = 0.5*sigma
        carrier = np.exp(-2j*np.pi*(m + offset)*wdm.dF*delta_n[:, None])
        for l in range(-max_lag_L, max_lag_L + 1):
            scalar = {0: 1.0, -1: 1j**l, 1: (-1j)**l}[sigma]
            alternating = (-1.0)**((l*m) % 2)
            parity = np.conj(1j**((n + m) % 2))*1j**((n - l + m + sigma) % 2)

            Ul = interp(U[i, l + max_lag_L], x_grid, -delta_n)[:, None]
            Vl = interp(V[i, l + max_lag_L], x_grid, -delta_n)[:, None]
            W = (m + offset)*wdm.dF*Ul + Vl

            Y = np.real(alternating*scalar*parity*carrier*np.conj(W))
            coeff = np.roll(np.roll(w, l, axis=0), -sigma, axis=1)
            out += coeff*Y
    return out


def main(log2_samples=23):
    n_samples = 2**log2_samples
    pad = s.wdm_padding_info(n_samples, s.NF, max_lag_L=s.MAX_LAG_L,
                             taper_samples=s.tukey_taper_samples(n_samples,
                                                                 s.ALPHA))
    cut = s.wdm_cut_info(pad)
    wdm = s.WDM.WDM_transform(dt=s.DT_SAMPLE, Nf=s.NF, N=pad["target_length"])
    t = np.asarray(wdm.times)
    t_start = pad["pad_left"]*s.DT_SAMPLE
    w_env = s.tukey(t, t_start, n_samples, s.ALPHA)

    def dwt(x):
        return np.asarray(wdm.dwt(jnp.asarray(x)))

    def mm(a, b):
        return s.wdm_mismatch(s.apply_wdm_cut(a, cut),
                              s.apply_wdm_cut(b, cut))["mismatch"]

    t0 = time.perf_counter()
    tables = wdm.build_time_delay_filter_interpolants(s.MAX_LAG_L,
                                                      s.NUM_INTERP_POINTS)
    U, V, x_grid = build_UV_tables(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    print(f"Nt={wdm.Nt}; X and U/V tables {time.perf_counter()-t0:.1f} s",
          flush=True)

    t_n = np.arange(wdm.Nt)*wdm.dT
    delta_n = s.delay(t_n)
    rate_n = s.delay_dot(t_n)
    t_ref = 0.5*t[-1]
    tau, tau_n = t - t_ref, (t_n - t_ref)[:, None]

    for m, r in CASES:
        f = (m + r)*wdm.dF
        t0 = time.perf_counter()
        a = 2*np.pi*f*t + s.PHI_0

        w_in = dwt(w_env*np.cos(a))
        op = np.asarray(wdm.apply_variable_time_shift(tables, jnp.asarray(w_in),
                                                      jnp.asarray(delta_n)))
        corrected = op + rate_n[:, None]*rate_term(wdm, U, V, x_grid,
                                                   w_in, delta_n)

        # T1 oracle (phase-warped tone), as in oracle_local_delay
        b = (2*np.pi*f*delta_n)[:, None]
        k = (2*np.pi*f*rate_n)[:, None]
        cb, sb = np.cos(b), np.sin(b)
        Dc0, Ds0 = w_in, dwt(w_env*np.sin(a))
        Dc1, Ds1 = dwt(tau*w_env*np.cos(a)), dwt(tau*w_env*np.sin(a))
        C = cb*Dc0 - sb*Ds0
        S0 = sb*Dc0 + cb*Ds0
        S1 = sb*Dc1 + cb*Ds1
        T1 = C - k*(S1 - tau_n*S0)
        del Dc1, Ds1, S0, S1

        warped = t + s.delay(t)
        E_true = dwt(s.tukey(warped, t_start, n_samples, s.ALPHA)
                     * np.cos(2*np.pi*f*warped + s.PHI_0))
        E_phase = dwt(w_env*np.cos(2*np.pi*f*warped + s.PHI_0))

        print(f"f={f*1e3:7.3f} mHz m={m:2d} r={r:.2f} | "
              f"op {mm(op, E_true):.2e}  corrected {mm(corrected, E_true):.2e}"
              f" | T1 oracle {mm(T1, E_phase):.2e}  "
              f"corrected-vs-T1 {mm(corrected, T1):.2e} "
              f"({time.perf_counter()-t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
