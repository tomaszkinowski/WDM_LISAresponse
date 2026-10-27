r"""
Check `rate_kernel.apply_rate_corrected_shift` against the production shift
(rate zero) and against the numpy prototype `reference_delay_rate_shift.rate_term` (rate
non-zero), on random coefficients and the `oracle_local_delay.py` delay, then repeat the
bare-shift tone test of `reference_delay_rate_shift.py` with the kernel.

    python prototypes/delay_rate_correction/check_rate_kernel.py [log2_samples]
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp

import oracle_local_delay as s
import reference_delay_rate_shift
from rate_kernel import apply_rate_corrected_shift, build_rate_tables


def main(log2_samples=19):
    n_samples = 2**log2_samples
    pad = s.wdm_padding_info(n_samples, s.NF, max_lag_L=s.MAX_LAG_L,
                             taper_samples=s.tukey_taper_samples(n_samples, s.ALPHA))
    cut = s.wdm_cut_info(pad)
    wdm = s.WDM.WDM_transform(dt=s.DT_SAMPLE, Nf=s.NF, N=pad["target_length"])
    tables = wdm.build_time_delay_filter_interpolants(s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    rate_tables = build_rate_tables(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    U, V, x_grid = reference_delay_rate_shift.build_UV_tables(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)

    t_n = np.arange(wdm.Nt)*wdm.dT
    delta_n, rate_n = s.delay(t_n), s.delay_dot(t_n)
    d, r = jnp.asarray(delta_n), jnp.asarray(rate_n)

    w = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (wdm.Nt, wdm.Nf)))
    rel = lambda a, b: float(np.max(np.abs(a - b))/np.max(np.abs(b)))

    op = np.asarray(wdm.apply_variable_time_shift(tables, jnp.asarray(w), d))
    k0 = np.asarray(apply_rate_corrected_shift(wdm, tables, rate_tables,
                                               jnp.asarray(w), d, jnp.zeros_like(r)))
    k1 = np.asarray(apply_rate_corrected_shift(wdm, tables, rate_tables,
                                               jnp.asarray(w), d, r))
    numpy_ref = op + rate_n[:, None]*reference_delay_rate_shift.rate_term(wdm, U, V, x_grid,
                                                             w, delta_n)
    print(f"rate 0: kernel vs production shift, max rel {rel(k0, op):.1e}")
    print(f"rate on: kernel vs numpy prototype, max rel {rel(k1, numpy_ref):.1e}; "
          f"rate term itself max rel {rel(k1 - op, numpy_ref - op):.1e}")

    t = np.asarray(wdm.times)
    w_env = s.tukey(t, pad["pad_left"]*s.DT_SAMPLE, n_samples, s.ALPHA)
    def mm(a, b):
        return s.wdm_mismatch(s.apply_wdm_cut(a, cut), s.apply_wdm_cut(b, cut))["mismatch"]
    for m, rr in ((20, 0.49), (60, 0.0), (60, 0.25), (60, 0.4), (60, 0.49)):
        f = (m + rr)*wdm.dF
        win = wdm.dwt(jnp.asarray(w_env*np.cos(2*np.pi*f*t)))
        warped = t + s.delay(t)
        exact = np.asarray(wdm.dwt(jnp.asarray(
            s.tukey(warped, pad["pad_left"]*s.DT_SAMPLE, n_samples, s.ALPHA)
            * np.cos(2*np.pi*f*warped))))
        op = np.asarray(wdm.apply_variable_time_shift(tables, win, d))
        k1 = np.asarray(apply_rate_corrected_shift(wdm, tables, rate_tables, win, d, r))
        print(f"tone {f*1e3:7.3f} mHz r={rr:.2f}: shift {mm(op, exact):.2e}  "
              f"kernel {mm(k1, exact):.2e}")


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:]))
