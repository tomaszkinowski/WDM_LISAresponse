r"""
A local-delay oracle for the bare shift.

`apply_variable_time_shift` treats the delay as constant across each output
wavelet, :math:`\delta(t)\approx\delta_n`. This script asks how much of its
off-bin error a better *local* delay model removes, before any new filter
tables are built.

Each local model gives every row :math:`n` its own warped signal, so no single
`dwt` call produces it. For a tone the row dependence factors out. With
:math:`x(t) = w(t)\cos(a(t))`, :math:`a = 2\pi f t + \phi_0`,
:math:`b_n = 2\pi f\delta_n`, :math:`k_n = 2\pi f\dot\delta_n` and
:math:`s = t - t_n`, the phase-warped tone
:math:`w(t)\cos(a + b_n + k_n s)` expanded in :math:`k_n s` is

.. math::

    \cos(a+b_n) - k_n s\,\sin(a+b_n) - \tfrac12 k_n^2 s^2\cos(a+b_n) + \ldots

and :math:`\cos(a+b_n)`, :math:`\sin(a+b_n)` are fixed combinations of
:math:`w\cos a` and :math:`w\sin a`. So row :math:`n` of every model is a
row-dependent combination of the transforms of six fixed signals,
:math:`\tau^p w\cos a` and :math:`\tau^p w\sin a` for :math:`p = 0, 1, 2`,
with :math:`\tau = t - t_{\rm ref}` centred on the record to keep the
:math:`s = \tau - \tau_n` cancellations small. This is exact: no truncation
and no per-row sum.

Models, all for the phase-warped tone:

- ``C``  :math:`\delta_n` - what the production operator approximates;
- ``T1`` first order in :math:`\dot\delta_n` - what new derivative filter
  tables would implement;
- ``T2`` second order in :math:`\dot\delta_n`; ``T1``-``T2`` is the cost of
  stopping at first order;
- ``T2Q`` ``T2`` plus the curvature :math:`\tfrac12\ddot\delta_n s^2` to
  first order.

References:

- ``E_phase`` `dwt(w(t) cos(2 pi f (t + delta(t)) + phi_0))`, the exact
  phase-warped tone, which the models are measured against;
- ``E_true`` `dwt(h(t + delta(t)))`, the envelope warped too, which the
  operator is measured against. ``E_true`` vs ``E_phase`` shows what the
  envelope alone contributes inside the cut.

The record follows the paper set-up (WDM_LISAresponse plotting/parameters.py:
dt = 4 s, 2**23 samples, Tukey alpha = 0.05, L = 25, P = 128), padded with
`wdm_padding_info` and compared over `wdm_cut_info` with `wdm_mismatch`.

Run from the WDM_LISAresponse root with the lisatools env:

    python prototypes/delay_rate_correction/oracle_local_delay.py [log2_samples]
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from WDM.code.discrete_wavelet_transform import WDM
from WDM_LISAresponse.code.utils.utils import (apply_wdm_cut,
                                               wdm_cut_info,
                                               wdm_mismatch,
                                               wdm_padding_info)
from WDM_LISAresponse.code.waveforms.waveforms import tukey_taper_samples


DT_SAMPLE = 4.0
DT_ROW = 1000.0
NF = int(DT_ROW/DT_SAMPLE)
ALPHA = 0.05
MAX_LAG_L = 25
NUM_INTERP_POINTS = 128
PHI_0 = 0.0

# Orbit-like delay: annual sinusoid, amplitude of n.x/c at 1 AU. Max rate
# 8.7e-5 s/s, matching the orbital delay used in the earlier shift tests.
YEAR = 365.25*86400.
DELAY_AMP = 436.0
DELAY_PHASE = 0.3

CASES = [(m, r) for m in (6, 20, 60) for r in (0.0, 0.25, 0.4, 0.49)]

# rows on which the quadrature construction is checked against a direct dwt
CHECK_ROWS = 3


def delay(t):
    return DELAY_AMP*np.sin(2*np.pi*t/YEAR + DELAY_PHASE)


def delay_dot(t):
    return DELAY_AMP*(2*np.pi/YEAR)*np.cos(2*np.pi*t/YEAR + DELAY_PHASE)


def delay_ddot(t):
    return -DELAY_AMP*(2*np.pi/YEAR)**2*np.sin(2*np.pi*t/YEAR + DELAY_PHASE)


def tukey(t, t_start, n_samples, alpha):
    r"""
    Continuous Tukey window matching `scipy.signal.windows.tukey` on the
    samples :math:`t_{\rm start} + k\,\delta t`; zero outside the record.
    """
    u = (t - t_start)/((n_samples - 1)*DT_SAMPLE)
    ramp = 0.5*alpha
    w = np.ones_like(u)
    lo = u < ramp
    hi = u > 1 - ramp
    w[lo] = 0.5*(1 + np.cos(np.pi*(u[lo]/ramp - 1)))
    w[hi] = 0.5*(1 + np.cos(np.pi*((u[hi] - 1)/ramp + 1)))
    w[(u < 0) | (u > 1)] = 0.0
    return w


def main(log2_samples=23):
    n_samples = 2**log2_samples
    pad = wdm_padding_info(n_samples, NF, max_lag_L=MAX_LAG_L,
                           taper_samples=tukey_taper_samples(n_samples, ALPHA))
    cut = wdm_cut_info(pad)

    wdm = WDM.WDM_transform(dt=DT_SAMPLE, Nf=NF, N=pad["target_length"])
    t = np.asarray(wdm.times)
    t_start = pad["pad_left"]*DT_SAMPLE
    w_env = tukey(t, t_start, n_samples, ALPHA)

    def dwt(x):
        return np.asarray(wdm.dwt(jnp.asarray(x)))

    t0 = time.perf_counter()
    tables = wdm.build_time_delay_filter_interpolants(MAX_LAG_L,
                                                      NUM_INTERP_POINTS)
    print(f"N={wdm.N} Nt={wdm.Nt} Nf={wdm.Nf} dT={wdm.dT} dF={wdm.dF:.3e}; "
          f"rows kept {cut['time_slice'].start}..{cut['time_slice'].stop}; "
          f"tables {time.perf_counter()-t0:.1f} s", flush=True)

    t_n = np.arange(wdm.Nt)*wdm.dT
    t_ref = 0.5*t[-1]
    tau = t - t_ref
    tau_n = (t_n - t_ref)[:, None]

    delta_n = delay(t_n)
    check_rows = np.linspace(cut["time_slice"].start,
                             cut["time_slice"].stop - 1,
                             CHECK_ROWS).astype(int)

    def mm(a, b):
        return wdm_mismatch(apply_wdm_cut(a, cut), apply_wdm_cut(b, cut))["mismatch"]

    results = []
    for m, r in CASES:
        f = (m + r)*wdm.dF
        t0 = time.perf_counter()

        a = 2*np.pi*f*t + PHI_0
        b = (2*np.pi*f*delta_n)[:, None]
        k = (2*np.pi*f*delay_dot(t_n))[:, None]
        kq = (np.pi*f*delay_ddot(t_n))[:, None]   # 2 pi f * delta_ddot / 2
        cb, sb = np.cos(b), np.sin(b)

        # row-n transforms of cos(a+b_n) tau^p w and sin(a+b_n) tau^p w
        Cp, Sp = [], []
        for p in range(3):
            Dc = dwt(tau**p*w_env*np.cos(a))
            Ds = dwt(tau**p*w_env*np.sin(a))
            Cp.append(cb*Dc - sb*Ds)
            Sp.append(sb*Dc + cb*Ds)
            del Dc, Ds

        # s^p in terms of tau^p: s = tau - tau_n
        sC = [Cp[0], Cp[1] - tau_n*Cp[0],
              Cp[2] - 2*tau_n*Cp[1] + tau_n**2*Cp[0]]
        sS = [Sp[0], Sp[1] - tau_n*Sp[0],
              Sp[2] - 2*tau_n*Sp[1] + tau_n**2*Sp[0]]

        C = sC[0]
        T1 = C - k*sS[1]
        T2 = T1 - 0.5*k**2*sC[2]
        T2Q = T2 - kq*sS[2]

        # direct check of the construction on a few rows
        check = 0.0
        for n in check_rows:
            x = w_env*(np.cos(a + b[n, 0])
                       - k[n, 0]*(t - t_n[n])*np.sin(a + b[n, 0]))
            row = dwt(x)[n, 1:]
            check = max(check, np.max(np.abs(row - T1[n, 1:]))
                        / np.max(np.abs(row)))

        h = w_env*np.cos(a)
        warped = t + delay(t)
        E_phase = dwt(w_env*np.cos(2*np.pi*f*warped + PHI_0))
        E_true = dwt(tukey(warped, t_start, n_samples, ALPHA)
                     * np.cos(2*np.pi*f*warped + PHI_0))
        op = np.asarray(wdm.apply_variable_time_shift(
                            tables, jnp.asarray(dwt(h)), jnp.asarray(delta_n)))

        row = dict(f=f, m=m, r=r, check=check,
                   op=mm(op, E_true), op_vs_C=mm(op, C),
                   env=mm(E_true, E_phase),
                   C=mm(C, E_phase), T1=mm(T1, E_phase),
                   T2=mm(T2, E_phase), T2Q=mm(T2Q, E_phase))
        results.append(row)
        print(f"f={f*1e3:7.3f} mHz m={m:2d} r={r:.2f} | check {check:.0e} "
              f"op-vs-C {row['op_vs_C']:.1e} env {row['env']:.1e} | "
              f"op {row['op']:.2e} C {row['C']:.2e} T1 {row['T1']:.2e} "
              f"T2 {row['T2']:.2e} T2Q {row['T2Q']:.2e} "
              f"({time.perf_counter()-t0:.0f} s)", flush=True)

        del Cp, Sp, sC, sS, C, T1, T2, T2Q, E_phase, E_true, op

    np.save(f"prototypes/delay_rate_correction/oracle_local_delay_results_2p{log2_samples}.npy",
            np.array(results, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
