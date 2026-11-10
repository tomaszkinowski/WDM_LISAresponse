r"""
Runtime of `fast_split.response_split` (six links, ``split_amp_sinc_first``)
against production's six-link call, stage by stage, at the paper's parameters.
Interleaved, min of REPS after a warm-up. ``--check`` also compares with
`response_corrections.split_responses` at the first Nf (slow at a full year).

    python prototypes/delay_rate_correction/bench_fast_split.py [--check] [--log2n=K] [Nf ...]
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
import fig8_mismatch_vs_dT as fig8
from parameters import (DT, GC_PHI, GC_THETA, MAX_LAG_L, MONOCHROMATIC_PARAMS,
                        N_SIGNAL, NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)
from WDM_LISAresponse import LISAresponse
from WDM_LISAresponse.code.utils import utils
from WDM_LISAresponse.code.waveforms import waveforms

import fast_split as fs
from response_corrections import build_tables, split_responses

REPS = 5
block = jax.block_until_ready


def main(nfs, check, n_signal=N_SIGNAL):
    orbits, start, _, _ = fig8.setup()
    signal_times = start + DT*np.arange(n_signal)
    print(f"jax {jax.__version__} on {jax.devices()[0]}", flush=True)
    for Nf in nfs:
        info = utils.wdm_padding_info(
            n_signal, Nf, max_lag_L=MAX_LAG_L,
            taper_samples=waveforms.tukey_taper_samples(n_signal, WINDOW_KWARGS["alpha"]))
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"]*DT,
                                max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS)
        tables = build_tables(wdm, MAX_LAG_L, NUM_INTERP_POINTS)
        centre = np.mean(orbits.compute_position(np.asarray(response.wavelet_times),
                                                 sc=[1, 2, 3]), axis=1)
        params = dict(MONOCHROMATIC_PARAMS[1], f=(round(3e-3/wdm.dF) + 0.4)*wdm.dF)
        _, hp, hc = waveforms.generate_windowed_waveform(
            "monochromatic", t0=start, dt=DT, N=n_signal, model_kwargs=params,
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                             info["pad_left"], info["pad_right"])
        wp, wc = block(wdm.dwt(hp)), block(wdm.dwt(hc))
        consts = fs.setup(response, tables, centre)
        theta, phi = GC_THETA, GC_PHI

        rows, sums = block(fs.stage_common(wdm, consts, theta, phi, wp, wc))
        grids = block(fs.stage_grids(wdm, consts, rows, wp, wc, sums))
        nodes = fs.table_nodes(wdm, NUM_INTERP_POINTS)
        weights = block(fs.stage_weights_vec(wdm, consts, rows, nodes))
        a = np.asarray(fs.response_split(wdm, consts, theta, phi, wp, wc, True))
        b = np.asarray(fs.response_split_w(wdm, consts, theta, phi, wp, wc))
        c = np.asarray(fs.response_split_w(wdm, consts, theta, phi, wp, wc, jnp.float32))
        mm = 1 - np.vdot(b, c)/np.sqrt(np.vdot(b, b)*np.vdot(c, c))
        print(f"   weighted vs direct shift: rel L2 {np.linalg.norm(a - b)/np.linalg.norm(a):.1e}; "
              f"float32 vs float64: rel L2 {np.linalg.norm(c - b)/np.linalg.norm(b):.1e}, "
              f"mismatch {mm:.1e}", flush=True)
        del a, b, c
        pieces = {
            "production": lambda: response(theta, phi, wp, wc),
            "corrected": lambda: fs.response_split(wdm, consts, theta, phi, wp, wc, True),
            "common": lambda: fs.stage_common(wdm, consts, theta, phi, wp, wc),
            "grids": lambda: fs.stage_grids(wdm, consts, rows, wp, wc, sums),
            "shifts": lambda: fs.stage_shifts(wdm, consts, rows, grids),
            "corrected_w": lambda: fs.response_split_w(wdm, consts, theta, phi, wp, wc),
            "corrected_w32": lambda: fs.response_split_w(wdm, consts, theta, phi, wp, wc,
                                                         jnp.float32),
            "weights": lambda: fs.stage_weights_vec(wdm, consts, rows, nodes),
            "wshifts": lambda: fs.stage_wshifts(wdm, rows, grids, weights),
        }
        compile_s = {}
        for k, f in pieces.items():
            tick = time.perf_counter()
            block(f())
            compile_s[k] = time.perf_counter() - tick
        best = {k: np.inf for k in pieces}
        for _ in range(REPS):
            for k, f in pieces.items():
                tick = time.perf_counter()
                block(f())
                best[k] = min(best[k], time.perf_counter() - tick)
        ratio = best["corrected"]/best["production"]
        print(f"Nf={Nf} Nt={wdm.Nt} dT={wdm.dT:.0f} s: "
              + "  ".join(f"{k} {v*1e3:.1f}" for k, v in best.items())
              + f" ms | corrected/production {ratio:.2f}x | first calls "
              + " ".join(f"{k} {v:.1f}" for k, v in compile_s.items()) + " s", flush=True)

        if check:
            ours = np.asarray(fs.response_split(wdm, consts, theta, phi, wp, wc, True))
            roemer = np.asarray(fs.roemer_rows(theta, phi, consts["centre"]))
            for j in range(response.num_links):
                ref = np.asarray(split_responses(response, wdm, tables, theta, phi, wp, wc,
                                                 roemer, j)["split_amp_sinc_first"])
                print(f"   link {j}: rel L2 vs split_responses "
                      f"{np.linalg.norm(ours[j] - ref)/np.linalg.norm(ref):.1e}", flush=True)
            check = False
        del response, wdm, consts, rows, sums, grids, weights
        jax.clear_caches()


if __name__ == "__main__":
    nfs = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    log2n = [int(a.split("=")[1]) for a in sys.argv if a.startswith("--log2n=")]
    main(nfs or [256, 1024, 2500], "--check" in sys.argv,
         2**log2n[0] if log2n else N_SIGNAL)
