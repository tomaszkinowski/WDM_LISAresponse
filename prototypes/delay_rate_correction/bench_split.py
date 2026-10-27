r"""
Time the pieces of the split-delay response at paper parameters (2**23
samples, dt = 4 s, L = 25, P = 128): per link the plain shift, the first-order
rate kernel and the sinc sum; per waveform the amplitude-rate (Z) sum and the
Roemer warp of a sampled waveform, by quintic spline (as the tests use), by
jitted 8-tap Lagrange interpolation, and analytically for a tone. Minimum of
`ROUNDS` interleaved runs; `JAX_PLATFORMS=cpu`.

    python prototypes/delay_rate_correction/bench_split.py [Nf ...]
"""
import sys
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import make_interp_spline

jax.config.update("jax_enable_x64", True)

import oracle_local_delay as s
from amp_sinc_kernel import (amp_weights, apply_amplitude_rate, apply_sinc_rate,
                        build_amp_table, build_sinc_table)
from rate_kernel import apply_rate_corrected_shift, build_rate_tables

ROUNDS = 5
TAPS = 8


@partial(jax.jit, static_argnums=(3,))
def lagrange_warp(h, t0, dt, taps, t_new):
    """h sampled at t0 + k dt, evaluated at t_new by a centred `taps`-point
    Lagrange stencil; zero outside the record."""
    u = (t_new - t0)/dt
    i0 = jnp.floor(u).astype(int) - taps//2 + 1
    frac = u - jnp.floor(u)
    nodes = jnp.arange(taps) - taps//2 + 1          # stencil offsets
    out = jnp.zeros_like(t_new)
    for j in range(taps):
        w = jnp.ones_like(t_new)
        for k in range(taps):
            if k != j:
                w = w*(frac - nodes[k])/(nodes[j] - nodes[k])
        idx = i0 + j
        ok = (idx >= 0) & (idx < h.shape[0])
        out = out + jnp.where(ok, w*h[jnp.clip(idx, 0, h.shape[0] - 1)], 0.0)
    return out


def best(fn):
    fn()
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
        times.append(time.perf_counter() - t0)
    return min(times)


def bench(Nf):
    n_samples = 2**23
    pad = s.wdm_padding_info(n_samples, Nf, max_lag_L=s.MAX_LAG_L,
                             taper_samples=s.tukey_taper_samples(n_samples, s.ALPHA))
    wdm = s.WDM.WDM_transform(dt=s.DT_SAMPLE, Nf=Nf, N=pad["target_length"])
    tables = wdm.build_time_delay_filter_interpolants(s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    t1 = build_rate_tables(wdm, s.MAX_LAG_L, s.NUM_INTERP_POINTS)
    zw = amp_weights(build_amp_table(wdm, 10))
    sw = amp_weights(build_sinc_table(wdm, 10))

    t_n = np.arange(wdm.Nt)*wdm.dT
    d = jnp.asarray(s.delay(t_n) % wdm.dT - 0.5*wdm.dT)
    r = jnp.asarray(s.delay_dot(t_n))
    w = jax.random.normal(jax.random.PRNGKey(0), (wdm.Nt, wdm.Nf))
    slope = jax.random.normal(jax.random.PRNGKey(1), (3, wdm.Nt, wdm.Nf))*1e-3

    t = np.asarray(wdm.times)
    h = np.cos(2*np.pi*3e-3*t)*np.exp(-((t - t.mean())/t.std())**2)
    warped = t + 400*np.sin(2*np.pi*t/s.YEAR)

    costs = {
        "shift": best(lambda: wdm.apply_variable_time_shift(tables, w, d)),
        "first-order kernel": best(lambda: apply_rate_corrected_shift(wdm, tables, t1, w, d, r)),
        "Z sum": best(lambda: apply_amplitude_rate(wdm, zw, w)),
        "sinc sum": best(lambda: apply_sinc_rate(wdm, sw, w, slope)),
        "dwt": best(lambda: wdm.dwt(jnp.asarray(h))),
        "warp: quintic spline": best(lambda: make_interp_spline(t, h, k=5)(warped)),
        f"warp: {TAPS}-tap Lagrange": best(lambda: lagrange_warp(
            jnp.asarray(h), 0.0, float(wdm.dt), TAPS, jnp.asarray(warped))),
        "warp: analytic tone": best(lambda: np.cos(2*np.pi*3e-3*warped)),
    }
    shift = costs["shift"]
    print(f"Nf={Nf} (dT={wdm.dT:.0f} s): " + "  ".join(
        f"{k} {v*1e3:.0f} ms ({v/shift:.2f}x)" for k, v in costs.items()), flush=True)

    # the 8-tap stencil's accuracy on the test waveform (3 mHz)
    exact = np.cos(2*np.pi*3e-3*warped)*np.exp(-((warped - t.mean())/t.std())**2)
    lag = np.asarray(lagrange_warp(jnp.asarray(h), 0.0, float(wdm.dt), TAPS,
                                   jnp.asarray(warped)))
    inner = slice(1000, -1000)
    print(f"    {TAPS}-tap Lagrange max rel error at 3 mHz: "
          f"{np.max(np.abs(lag - exact)[inner])/np.max(np.abs(exact)):.1e}", flush=True)
    jax.clear_caches()


if __name__ == "__main__":
    for Nf in [int(x) for x in sys.argv[1:]] or [250, 900, 2500]:
        bench(Nf)
