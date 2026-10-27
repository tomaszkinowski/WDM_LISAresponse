r"""
Check `rate_kernel.apply_corrected_shift_fit` on the bare shift at coarse
tilings: a 30 mHz tone halfway between bands, full year, against the exact
warp, beside the production shift and the second-order Taylor kernel.

    python prototypes/delay_rate_correction/check_fit_kernel.py [Nf ...]
"""
import sys
import time

import numpy as np
import jax.numpy as jnp

import oracle_local_delay as s
from rate_kernel import (apply_corrected_shift_2, apply_corrected_shift_fit,
                         build_fit_tables, build_rate_tables_2)

F_FIT = 0.035
DEGREES = tuple(int(k) for k in __import__('os').environ.get('FIT_DEGREES', '6,8,10').split(','))


def run(Nf, log2_samples=23):
    n_samples = 2**log2_samples
    pad = s.wdm_padding_info(n_samples, Nf, max_lag_L=s.MAX_LAG_L,
                             taper_samples=s.tukey_taper_samples(n_samples, s.ALPHA))
    cut = s.wdm_cut_info(pad)
    wdm = s.WDM.WDM_transform(dt=s.DT_SAMPLE, Nf=Nf, N=pad["target_length"])
    tables = wdm.build_time_delay_filter_interpolants(s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    t2 = build_rate_tables_2(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)

    t_n = np.arange(wdm.Nt)*wdm.dT
    d = jnp.asarray(s.delay(t_n))
    r = jnp.asarray(s.delay_dot(t_n))
    c = jnp.asarray(s.delay_ddot(t_n))
    eps_max = float(np.max(np.abs(r)))*F_FIT

    t = np.asarray(wdm.times)
    t0 = pad["pad_left"]*s.DT_SAMPLE
    w_env = s.tukey(t, t0, n_samples, s.ALPHA)
    mm = lambda a, b: s.wdm_mismatch(s.apply_wdm_cut(np.asarray(a), cut),
                                     s.apply_wdm_cut(np.asarray(b), cut))["mismatch"]
    f = (round(3e-2/wdm.dF) + 0.5)*wdm.dF
    win = wdm.dwt(jnp.asarray(w_env*np.cos(2*np.pi*f*t)))
    warped = t + s.delay(t)
    exact = np.asarray(wdm.dwt(jnp.asarray(
        s.tukey(warped, t0, n_samples, s.ALPHA)*np.cos(2*np.pi*f*warped))))

    line = [f"shift {mm(wdm.apply_variable_time_shift(tables, win, d), exact):.1e}",
            f"taylor2 {mm(apply_corrected_shift_2(wdm, tables, t2, win, d, r, c), exact):.1e}"]
    for K in DEGREES:
        tick = time.perf_counter()
        fit = build_fit_tables(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS, eps_max, K)
        build = time.perf_counter() - tick
        out = apply_corrected_shift_fit(wdm, tables, fit, t2, eps_max, win, d, r, c)
        flat = apply_corrected_shift_fit(wdm, tables, fit, t2, eps_max, win, d, r,
                                         jnp.zeros_like(c))
        line.append(f"fit K={K} {mm(out, exact):.1e} [no curv {mm(flat, exact):.1e}] "
                    f"(tables {build:.0f} s)")
    print(f"dT={wdm.dT:8.4g} s eps_max/dF={eps_max/wdm.dF:.2f} | " + "  ".join(line),
          flush=True)


if __name__ == "__main__":
    for Nf in [int(x) for x in sys.argv[1:]] or [1406, 2500, 7906, 25000]:
        run(Nf)
