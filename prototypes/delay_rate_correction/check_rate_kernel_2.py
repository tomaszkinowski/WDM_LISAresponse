r"""
Check `rate_kernel.apply_corrected_shift_2`: equal to the production shift at
zero rate and curvature, then on the bare-shift tones of `oracle_local_delay.py`
(analytic delay, rate and curvature) against the exact warp, beside the
first-order kernel.

    python prototypes/delay_rate_correction/check_rate_kernel_2.py [log2_samples]
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp

import oracle_local_delay as s
from rate_kernel import (apply_corrected_shift_2, apply_rate_corrected_shift,
                         build_rate_tables, build_rate_tables_2)


def main(log2_samples=19):
    n_samples = 2**log2_samples
    pad = s.wdm_padding_info(n_samples, s.NF, max_lag_L=s.MAX_LAG_L,
                             taper_samples=s.tukey_taper_samples(n_samples, s.ALPHA))
    cut = s.wdm_cut_info(pad)
    wdm = s.WDM.WDM_transform(dt=s.DT_SAMPLE, Nf=s.NF, N=pad["target_length"])
    tables = wdm.build_time_delay_filter_interpolants(s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    t1 = build_rate_tables(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    t2 = build_rate_tables_2(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)

    t_n = np.arange(wdm.Nt)*wdm.dT
    d = jnp.asarray(s.delay(t_n))
    r = jnp.asarray(s.delay_dot(t_n))
    c = jnp.asarray(s.delay_ddot(t_n))
    zero = jnp.zeros_like(r)

    w = jax.random.normal(jax.random.PRNGKey(1), (wdm.Nt, wdm.Nf))
    op = np.asarray(wdm.apply_variable_time_shift(tables, w, d))
    k0 = np.asarray(apply_corrected_shift_2(wdm, tables, t2, w, d, zero, zero))
    print(f"zero rate: kernel 2 vs production, max rel "
          f"{np.max(np.abs(k0 - op))/np.max(np.abs(op)):.1e}", flush=True)

    t = np.asarray(wdm.times)
    t0 = pad["pad_left"]*s.DT_SAMPLE
    w_env = s.tukey(t, t0, n_samples, s.ALPHA)
    def mm(a, b):
        return s.wdm_mismatch(s.apply_wdm_cut(a, cut), s.apply_wdm_cut(b, cut))["mismatch"]
    for m, rr in ((6, 0.49), (20, 0.49), (60, 0.0), (60, 0.25), (60, 0.4), (60, 0.49)):
        f = (m + rr)*wdm.dF
        win = wdm.dwt(jnp.asarray(w_env*np.cos(2*np.pi*f*t)))
        warped = t + s.delay(t)
        exact = np.asarray(wdm.dwt(jnp.asarray(
            s.tukey(warped, t0, n_samples, s.ALPHA)*np.cos(2*np.pi*f*warped))))
        op = np.asarray(wdm.apply_variable_time_shift(tables, win, d))
        k1 = np.asarray(apply_rate_corrected_shift(wdm, tables, t1, win, d, r))
        k2 = np.asarray(apply_corrected_shift_2(wdm, tables, t2, win, d, r, zero))
        k2c = np.asarray(apply_corrected_shift_2(wdm, tables, t2, win, d, r, c))
        print(f"tone {f*1e3:7.3f} mHz r={rr:.2f}: shift {mm(op, exact):.2e}  "
              f"1st {mm(k1, exact):.2e}  2nd {mm(k2, exact):.2e}  "
              f"2nd+curv {mm(k2c, exact):.2e}", flush=True)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
