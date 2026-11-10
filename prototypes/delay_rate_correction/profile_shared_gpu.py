r"""
The link-independent stage of `fast_split_gpu` on the GPU, piece by piece:
the per-row geometry (with and without the landing step), the shared-sums
kernel and its recombination, and the grid assembly. Synthetic coefficient
grids, the paper's sizes.

    python prototypes/delay_rate_correction/profile_shared_gpu.py [Nf ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plotting"))

import fig8_mismatch_vs_dT as fig8
from parameters import GC_PHI, GC_THETA
from WDM_LISAresponse.code.lisa_response import response_sub as rs

import fast_split as fs
import fast_split_gpu as fg
from bench_fast_split_gpu import build

REPS = 7
block = jax.block_until_ready


def main(nfs):
    orbits, start, _, _ = fig8.setup()
    for Nf in nfs:
        _, wdm, response, consts = build(Nf, orbits, start)
        rng = np.random.default_rng(0)
        wp, wc = (jnp.asarray(rng.standard_normal((wdm.Nt, wdm.Nf))) for _ in range(2))
        f, st = fg.make_response(wdm, consts)
        amp_w, sinc_w = np.asarray(consts["amp"]), np.asarray(consts["sinc"])

        @jax.jit
        def geometry_no_landing(c):
            theta, phi = jnp.atleast_1d(GC_THETA), jnp.atleast_1d(GC_PHI)
            proj_p, proj_c, tau, n_hat = rs.compute_angular_geometry_wdm(
                theta, phi, c["ltt"], c["link_direction"])
            delta = rs.compute_time_shifts(n_hat, c["receiver_position"], tau)
            return delta - fs.roemer_rows(theta[0], phi[0], c["centre"]), proj_p, proj_c, tau

        @jax.jit
        def landing(c, rest, proj_p, proj_c, tau):
            return rs.evaluate_at_landing_times(
                jnp.stack(jnp.broadcast_arrays(c["ltt"], proj_p, proj_c, tau)), rest, wdm.dT)

        geo = jax.jit(lambda c: fs._geometry(wdm, c, GC_THETA, GC_PHI))
        landing_s = jax.jit(lambda c, rest, proj_p, proj_c, tau: fs.landing_sliced(
            jnp.stack(jnp.broadcast_arrays(c["ltt"], proj_p, proj_c, tau)), rest, wdm.dT))

        @jax.jit
        def landing32(c, rest, proj_p, proj_c, tau):
            f = lambda x: x.astype(jnp.float32)
            return rs.evaluate_at_landing_times(
                jnp.stack(jnp.broadcast_arrays(f(c["ltt"]), f(proj_p), f(proj_c), f(tau))),
                f(rest), wdm.dT)
        g0 = block(geometry_no_landing(consts))
        sums_kernel = jax.jit(lambda w: fg.shared_sums_kernel5(amp_w, sinc_w, w))
        w2 = jnp.stack([wp, wc]).astype(jnp.float32)
        acc = block(sums_kernel(w2))
        recomb = jax.jit(lambda a: (fg._recombine_sums5(a[0]), fg._recombine_sums5(a[1])))
        l64 = np.asarray(landing(consts, *g0)); l32 = np.asarray(landing32(consts, *g0))
        print("   landing f32 vs f64: max rel per quantity",
              [f"{np.abs(l32[k] - l64[k]).max()/np.abs(l64[k]).max():.1e}" for k in range(4)], flush=True)
        del l64, l32
        rows, sums = block(st["common"](consts, GC_THETA, GC_PHI, wp.astype(jnp.float32),
                                        wc.astype(jnp.float32)))
        wp32, wc32 = wp.astype(jnp.float32), wc.astype(jnp.float32)
        del wp, wc

        pieces = {
            "common (all)": lambda: st["common"](consts, GC_THETA, GC_PHI, wp32, wc32),
            "geometry f64": lambda: geo(consts),
            "geometry w/o landing": lambda: geometry_no_landing(consts),
            "landing step": lambda: landing(consts, *g0),
            "landing step f32": lambda: landing32(consts, *g0),
            "landing sliced f64": lambda: landing_s(consts, *g0),
            "sums kernel": lambda: sums_kernel(w2),
            "sums recombine": lambda: recomb(acc),
            "grids (all 6, link-innermost)": lambda: st["grids"](consts, rows, wp32, wc32, sums),
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
        del response, wdm, consts, rows, sums, acc
        jax.clear_caches()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [256, 2500])
