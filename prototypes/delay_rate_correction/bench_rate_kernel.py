r"""
Time `rate_kernel.apply_rate_corrected_shift` against the production
`apply_variable_time_shift` at paper parameters (2**23 samples, dt = 4 s,
L = 25, P = 128), for several Nf. The two are interleaved and the minimum of
`ROUNDS` runs is reported, since thermal drift on this machine makes
back-to-back blocks unreliable.

    python prototypes/delay_rate_correction/bench_rate_kernel.py [Nf ...]
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import oracle_local_delay as s
from rate_kernel import (apply_corrected_shift_2, apply_rate_corrected_shift,
                         build_rate_tables, build_rate_tables_2)
from amp_sinc_kernel import amp_weights, apply_amplitude_rate, build_amp_table

ROUNDS = 7


def bench(Nf):
    n_samples = 2**23
    pad = s.wdm_padding_info(n_samples, Nf, max_lag_L=s.MAX_LAG_L,
                             taper_samples=s.tukey_taper_samples(n_samples, s.ALPHA))
    wdm = s.WDM.WDM_transform(dt=s.DT_SAMPLE, Nf=Nf, N=pad["target_length"])
    tables = wdm.build_time_delay_filter_interpolants(s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    rate_tables = build_rate_tables(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    tables_2 = build_rate_tables_2(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    weights = amp_weights(build_amp_table(wdm, 10))

    t_n = np.arange(wdm.Nt)*wdm.dT
    d = jnp.asarray(s.delay(t_n) % wdm.dT - 0.5*wdm.dT)   # keep within +-dT/2
    r = jnp.asarray(s.delay_dot(t_n))
    w = jax.random.normal(jax.random.PRNGKey(0), (wdm.Nt, wdm.Nf))

    runs = {
        "shift": lambda: wdm.apply_variable_time_shift(tables, w, d),
        "corrected": lambda: apply_rate_corrected_shift(wdm, tables, rate_tables,
                                                        w, d, r),
        "second": lambda: apply_corrected_shift_2(wdm, tables, tables_2, w, d, r,
                                                  0.01*r),
        "amp": lambda: apply_amplitude_rate(wdm, weights, w),
    }
    compile_s = {}
    for name, fn in runs.items():
        t0 = time.perf_counter()
        fn().block_until_ready()
        compile_s[name] = time.perf_counter() - t0

    times = {name: [] for name in runs}
    for _ in range(ROUNDS):
        for name, fn in runs.items():
            t0 = time.perf_counter()
            fn().block_until_ready()
            times[name].append(time.perf_counter() - t0)

    best = {name: min(v) for name, v in times.items()}
    print(f"Nf={Nf:5d} Nt={wdm.Nt:6d} dT={wdm.dT:7.0f} s | shift "
          f"{best['shift']*1e3:6.1f} ms | 1st {best['corrected']/best['shift']:.2f}x  "
          f"2nd {best['second']/best['shift']:.2f}x  Z-sum {best['amp']/best['shift']:.2f}x | "
          f"first call " + " / ".join(f"{compile_s[k]:.1f}" for k in runs) + " s",
          flush=True)
    jax.clear_caches()


if __name__ == "__main__":
    for Nf in (int(x) for x in sys.argv[1:] or (100, 250, 900, 2500)):
        bench(Nf)
