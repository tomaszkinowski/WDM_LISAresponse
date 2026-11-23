r"""
One link's shift on the GPU, the candidate arrangements side by side:
the rate kernel with its recombination in XLA (two grids per sigma) or with
the carrier applied in the kernel (one grid per sigma), and the split into
production's shift plus a delay-rate correction kernel in float32 or
float16. Synthetic grids at the paper's sizes.

    python prototypes/delay_rate_correction/profile_kernels_gpu.py [Nf ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plotting"))

import fig14_mismatch_vs_dT_zeroth_order as fig14
from parameters import GC_PHI, GC_THETA
from WDM.code.time_delay_filters import shift_gpu

import fast_split as fs
import fast_split_gpu as fg
from bench_fast_split_gpu import build

REPS = 9
block = jax.block_until_ready


def main(nfs):
    orbits, start, _, _ = fig14.setup()
    for Nf in nfs:
        _, wdm, response, consts = build(Nf, orbits, start)
        rng = np.random.default_rng(0)
        grid = jnp.asarray(rng.standard_normal((wdm.Nt, wdm.Nf)), jnp.float32)
        rest = jax.jit(lambda c: fs._geometry(wdm, c, GC_THETA, GC_PHI))(consts)[0][0]
        tile, warps = fg.RATE_TILE, fg.RATE_WARPS

        @jax.jit
        def weights(c, rest):
            r = rest.astype(jnp.float32)
            return fs.shift_weights_vec(wdm, fs.row_filters_poly(
                c["poly"].astype(jnp.float32), r,
                jnp.gradient(rest, wdm.dT).astype(jnp.float32)), r)

        W = block(weights(consts, rest))
        two = jax.jit(lambda w, g, r: fg._recombine_shift(fg.rate_shift_kernel(
            w[None], g[None], wdm.dF, by_sigma=True, tile=tile, warps=warps)[0], r, wdm.dF))
        one = jax.jit(lambda w, g, r: fg._recombine_carried(fg.rate_shift_kernel(
            w[None], g[None], wdm.dF, by_sigma=True, tile=tile, warps=warps,
            cycles=(wdm.dF*r)[None])[0]))
        main_shift = jax.jit(lambda g, r: shift_gpu.variable_time_shift_gpu(
            consts["filters"], g[None], r[None], wdm.dF, wdm.dT)[0])
        corr = {d: jax.jit(lambda c, r, g, d=d: fg.correction_shift(wdm, c, r, g, d))
                for d in (jnp.float32, jnp.float16)}
        a = np.asarray(two(W, grid, rest))
        c32 = np.asarray(corr[jnp.float32](consts, rest, grid), float)
        c16 = np.asarray(corr[jnp.float16](consts, rest, grid), float)
        print(f"   f16 vs f32 correction, rel to shift {np.linalg.norm(c16 - c32)/np.linalg.norm(a):.1e} "
              f"(correction/shift {np.linalg.norm(c32)/np.linalg.norm(a):.1e})", flush=True)
        del a, c32, c16
        pieces = {
            "weights f32": lambda: weights(consts, rest),
            "rate kernel + XLA recombine": lambda: two(W, grid, rest),
            "production shift": lambda: main_shift(grid, rest),
            "correction f32": lambda: corr[jnp.float32](consts, rest, grid),
            "correction f16": lambda: corr[jnp.float16](consts, rest, grid),
            "production + correction f16": lambda: (main_shift(grid, rest),
                                                    corr[jnp.float16](consts, rest, grid)),
        }
        for g in pieces.values():
            block(g())
        best = {k: np.inf for k in pieces}
        for _ in range(REPS):
            for k, g in pieces.items():
                tick = time.perf_counter()
                block(g())
                best[k] = min(best[k], time.perf_counter() - tick)
        print(f"Nf={Nf} Nt={wdm.Nt}: " + "  ".join(f"{k} {v*1e3:.2f}" for k, v in best.items())
              + " ms", flush=True)
        del response, wdm, consts, W
        jax.clear_caches()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [256, 2500])
