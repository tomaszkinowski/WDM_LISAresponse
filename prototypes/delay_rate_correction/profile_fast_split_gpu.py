r"""
Pieces of one link of `fast_split_gpu` on the GPU, synthetic grids at the
paper's size: the per-row weights (float64 and float32 builds), the rate
kernel, its recombination, and the shared-sums kernel; production's shift
kernel for scale.

    python prototypes/delay_rate_correction/profile_fast_split_gpu.py [Nf ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plotting"))

import WDM
import fig14_mismatch_vs_dT_zeroth_order as fig14
from parameters import DT, GC_PHI, GC_THETA, MAX_LAG_L, N_SIGNAL, NUM_INTERP_POINTS
from WDM_LISAresponse import LISAresponse
from WDM.code.time_delay_filters import shift_gpu

import fast_split as fs
import fast_split_gpu as fg
from bench_fast_split_gpu import build

REPS = 7
block = jax.block_until_ready


def main(nfs):
    orbits, start, _, _ = fig14.setup()
    for Nf in nfs:
        _, wdm, response, consts = build(Nf, orbits, start)
        nodes = fs.table_nodes(wdm, NUM_INTERP_POINTS)
        rng = np.random.default_rng(0)
        grid = jnp.asarray(rng.standard_normal((wdm.Nt, wdm.Nf)), jnp.float32)
        w2 = jnp.asarray(rng.standard_normal((2, wdm.Nt, wdm.Nf)), jnp.float32)
        rows = jax.jit(lambda c: fs._geometry(wdm, c, GC_THETA, GC_PHI))(consts)
        rest = rows[0][0]

        def weights(dtype):
            @jax.jit
            def f(c, rest):
                filt = fs.row_filters_local(wdm, c["filters"].astype(dtype),
                                            c["rate"].astype(dtype), rest.astype(dtype),
                                            jnp.gradient(rest, wdm.dT).astype(dtype), nodes)
                return fs.shift_weights(wdm, filt, rest.astype(dtype)).astype(jnp.float32)
            return f
        w64, w32 = weights(jnp.float64), weights(jnp.float32)
        W = block(w64(consts, rest))
        kern = jax.jit(lambda w, g: fg.rate_shift_kernel(w[None], g[None], wdm.dF)[0])
        K = block(kern(W, grid))
        coef = np.asarray(fs.poly_filters(wdm, MAX_LAG_L, fs.POLY_DEGREE))
        rv = fg.poly_rowvals(rest, wdm.dT, wdm.dF).astype(jnp.float32)
        def pvar(**kw):
            return jax.jit(lambda r, g: fg.rate_shift_kernel_poly(r[None], g[None], wdm.dF, coef, **kw)[0])
        polyw = jax.jit(lambda c, rest: fs.shift_weights_vec(
            wdm, fs.row_filters_poly(c.astype(jnp.float32), rest.astype(jnp.float32),
                                     jnp.gradient(rest, wdm.dT).astype(jnp.float32)),
            rest.astype(jnp.float32)))
        cpoly = jnp.asarray(coef)
        variants = {
            "table s 4x256": jax.jit(lambda w, g: fg.rate_shift_kernel(w[None], g[None], wdm.dF, by_sigma=True, tile=(4, 256))[0]),
        }
        pvariants = {f"poly {t[0]}x{t[1]} w{w}": pvar(tile=t, warps=w)
                     for t, w in (((4, 256), 4), ((4, 512), 4), ((4, 1024), 4), ((8, 512), 4), ((2, 1024), 4), ((4, 1024), 8))}
        rec = jax.jit(lambda K, r: fg._recombine_shift(K, r, wdm.dF))
        amp_w, sinc_w = np.asarray(consts["amp"]), np.asarray(consts["sinc"])
        sums = jax.jit(lambda w: fg.shared_sums_kernel(amp_w, sinc_w, w)[1])
        prod = jax.jit(lambda g, d: shift_gpu.variable_time_shift_gpu(
            response.filter_tables, g[None], d[None], wdm.dF, wdm.dT)[0])
        pieces = {
            "weights f64": lambda: w64(consts, rest),
            "weights f32": lambda: w32(consts, rest),
            "rate kernel": lambda: kern(W, grid),
            **{f"rate {name}": (lambda f=f: f(W, grid)) for name, f in variants.items()},
            **{name: (lambda f=f: f(rv, grid)) for name, f in pvariants.items()},
            "weights polyvec f32": lambda: polyw(cpoly, rest),
            "recombine": lambda: rec(K, rest),
            "shared sums x2": lambda: sums(w2),
            "production shift x1": lambda: prod(grid, rest),
        }
        for f in pieces.values():
            block(f())
        best = {k: np.inf for k in pieces}
        for _ in range(REPS):
            for k, f in pieces.items():
                tick = time.perf_counter()
                block(f())
                best[k] = min(best[k], time.perf_counter() - tick)
        print(f"Nf={Nf} Nt={wdm.Nt}: " + "  ".join(f"{k} {v*1e3:.2f}" for k, v in best.items())
              + " ms", flush=True)
        del response, wdm, consts
        jax.clear_caches()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [256, 1024, 2500])
