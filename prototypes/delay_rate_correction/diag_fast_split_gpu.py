r"""
Where `fast_split_gpu` departs from the float64 path, stage by stage, on the
saved inputs of `bench_fast_split_gpu.py --save`: shared sums, grids, and each
link's shift, the float32 GPU stages against the float64 `fast_split` stages
evaluated on the same device.

    python prototypes/delay_rate_correction/diag_fast_split_gpu.py DIR Nf
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plotting"))

import fig6_mismatch_vs_dT as fig6
from parameters import GC_PHI, GC_THETA

import fast_split as fs
import fast_split_gpu as fg
from bench_fast_split_gpu import build

rel = lambda a, b: float(jnp.linalg.norm(jnp.asarray(a, jnp.float64) - b)/jnp.linalg.norm(b))


def main(src, Nf):
    orbits, start, _, _ = fig6.setup()
    _, wdm, response, consts = build(Nf, orbits, start)
    wp, wc = (jnp.asarray(np.load(src/f"{k}_{Nf}.npy")) for k in ("wp", "wc"))
    ref = np.load(src/f"ref_{Nf}.npy")
    f, st = fg.make_response(wdm, consts)
    rows32, sums32 = st["common"](consts, GC_THETA, GC_PHI, wp, wc)
    rows, sums = fs.stage_common(wdm, consts, GC_THETA, GC_PHI, wp, wc)
    print("rows   ", [f"{rel(a, b):.1e}" for a, b in zip(rows32, rows)], flush=True)
    print("sums   ", [f"{rel(a, b):.1e}" for a, b in zip(jax.tree.leaves(sums32), jax.tree.leaves(sums))], flush=True)
    g32 = st["grids"](consts, rows32, wp, wc, sums32)
    g = fs.stage_grids(wdm, consts, rows, wp, wc, sums)
    print("grids  ", [f"{rel(g32[j], g[j]):.1e}" for j in range(6)], flush=True)
    del sums32, sums
    nodes = fs.table_nodes(wdm, consts["filters"].shape[-1])
    out = []
    for j in range(6):
        rest = rows[0][j]
        filt = fs.row_filters_local(wdm, consts["filters"], consts["rate"], rest,
                                    jnp.gradient(rest, wdm.dT), nodes)
        w64 = fs.shift_weights(wdm, filt, rest)
        exact = fs.weighted_shift(wdm, w64, g[j], rest)              # f64 on device
        K = fg.rate_shift_kernel(w64.astype(jnp.float32)[None], g[j].astype(jnp.float32)[None], wdm.dF)[0]
        ours = fg._recombine_shift(K, rest, wdm.dF)
        out.append(f"link {j}: kernel(f64 grid, f64 weights)->f32 {rel(ours, exact):.1e}, "
                   f"f64 path vs saved ref {rel(exact, ref[j]):.1e}")
        print(out[-1], flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
