r"""
What a local delay model can reach at coarse Delta T.

The bare-shift oracle of `oracle_local_delay.py` at
coarse tilings, for an off-bin tone: ``C``, ``T1``, ``T2`` from the factorised
transforms on every row, and on a sample of rows the exact local warps

- ``Lin``   :math:`\delta_n+\dot\delta_n(t-t_n)`,
- ``LinQ``  :math:`\delta_n+\dot\delta_n(t-t_n)+\tfrac12\ddot\delta_n(t-t_n)^2`,

each by one full `dwt` per sampled row. All against the exact phase-warped
tone. Same record, delay and cut as `oracle_local_delay.py`.

    python prototypes/delay_rate_correction/oracle_coarse.py [Nf ...]
"""
import sys
import time

import numpy as np
import jax.numpy as jnp

import oracle_local_delay as s

NUM_ROWS = 40
F_TONE = 3e-2
R = 0.5


def run(Nf, log2_samples=23):
    n_samples = 2**log2_samples
    pad = s.wdm_padding_info(n_samples, Nf, max_lag_L=s.MAX_LAG_L,
                             taper_samples=s.tukey_taper_samples(n_samples, s.ALPHA))
    cut = s.wdm_cut_info(pad)
    wdm = s.WDM.WDM_transform(dt=s.DT_SAMPLE, Nf=Nf, N=pad["target_length"])
    t = np.asarray(wdm.times)
    w_env = s.tukey(t, pad["pad_left"]*s.DT_SAMPLE, n_samples, s.ALPHA)
    dwt = lambda x: np.asarray(wdm.dwt(jnp.asarray(x)))

    t_n = np.arange(wdm.Nt)*wdm.dT
    tau = t - 0.5*t[-1]
    tau_n = (t_n - 0.5*t[-1])[:, None]
    rows_all = np.arange(cut["time_slice"].start, cut["time_slice"].stop)
    rows = np.linspace(rows_all[0], rows_all[-1], min(NUM_ROWS, len(rows_all))).astype(int)

    f = (round(F_TONE/wdm.dF) + R)*wdm.dF
    a = 2*np.pi*f*t + s.PHI_0
    b = (2*np.pi*f*s.delay(t_n))[:, None]
    k = (2*np.pi*f*s.delay_dot(t_n))[:, None]
    cb, sb = np.cos(b), np.sin(b)

    Cp, Sp = [], []
    for p in range(3):
        Dc, Ds = dwt(tau**p*w_env*np.cos(a)), dwt(tau**p*w_env*np.sin(a))
        Cp.append(cb*Dc - sb*Ds)
        Sp.append(sb*Dc + cb*Ds)
    C = Cp[0]
    S1 = Sp[1] - tau_n*Sp[0]
    C2 = Cp[2] - 2*tau_n*Cp[1] + tau_n**2*Cp[0]
    T1 = C - k*S1
    T2 = T1 - 0.5*k**2*C2
    E = dwt(w_env*np.cos(2*np.pi*f*(t + s.delay(t)) + s.PHI_0))

    lin, linq = np.empty((len(rows), wdm.Nf)), np.empty((len(rows), wdm.Nf))
    for i, n in enumerate(rows):
        sn = t - t_n[n]
        base = s.delay(t_n[n]) + s.delay_dot(t_n[n])*sn
        lin[i] = dwt(w_env*np.cos(2*np.pi*f*(t + base) + s.PHI_0))[n]
        linq[i] = dwt(w_env*np.cos(2*np.pi*f*(t + base + 0.5*s.delay_ddot(t_n[n])*sn**2)
                                  + s.PHI_0))[n]

    full = lambda X: s.wdm_mismatch(X[rows_all, 1:], E[rows_all, 1:])["mismatch"]
    some = lambda X: s.wdm_mismatch(X[:, 1:], E[rows, 1:])["mismatch"]
    phase = 2*np.pi*f*np.max(np.abs(s.delay_dot(t_n)))*wdm.dT
    print(f"dT={wdm.dT:8.4g} s f={f*1e3:.4f} mHz (2 pi f rate dT = {phase:.2f}) | all rows: "
          f"C {full(C):.1e} T1 {full(T1):.1e} T2 {full(T2):.1e} | {len(rows)} rows: "
          f"C {some(C[rows]):.1e} T1 {some(T1[rows]):.1e} T2 {some(T2[rows]):.1e} "
          f"Lin {some(lin):.1e} LinQ {some(linq):.1e}", flush=True)


if __name__ == "__main__":
    for Nf in [int(x) for x in sys.argv[1:]] or [1406, 2500, 7906, 25000]:
        t0 = time.perf_counter()
        run(Nf)
        print(f"    ({time.perf_counter() - t0:.0f} s)", flush=True)
