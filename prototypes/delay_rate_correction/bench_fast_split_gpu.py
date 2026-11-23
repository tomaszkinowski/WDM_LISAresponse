r"""
GPU runtime of `fast_split_gpu` against production's GPU call, six links,
paper parameters, with the result checked against the CPU float64 corrected
response. Two steps, because the full-year dwt does not fit beside the
response on a 4 GB card:

    JAX_PLATFORMS=cpu python .../bench_fast_split_gpu.py --save DIR [Nf ...]
    python .../bench_fast_split_gpu.py --load DIR [Nf ...]

``--save`` writes, per Nf, the coefficients and the CPU float64 response
(`fast_split.response_split_w`); ``--load`` times on the default device.
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
from parameters import (DT, GC_PHI, GC_THETA, MAX_LAG_L, MONOCHROMATIC_PARAMS,
                        N_SIGNAL, NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)
from WDM_LISAresponse import LISAresponse
from WDM_LISAresponse.code.utils import utils
from WDM_LISAresponse.code.waveforms import waveforms

import fast_split as fs
from response_corrections import build_tables

REPS = 7
#: links per launch to time besides one at a time (memory permitting)
GROUPS = (2, 3, 6)
block = jax.block_until_ready


def build(Nf, orbits, start):
    info = utils.wdm_padding_info(
        N_SIGNAL, Nf, max_lag_L=MAX_LAG_L,
        taper_samples=waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"]))
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
    response = LISAresponse(wdm, orbits, t0=start - info["pad_left"]*DT,
                            max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS)
    # the correction tables from a host transform: their window integrals
    # run full-length FFTs, which do not fit beside the response on 4 GB
    with jax.default_device(jax.devices("cpu")[0]):
        host = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        tables = jax.tree.map(np.asarray, build_tables(host, MAX_LAG_L, NUM_INTERP_POINTS))
        poly = np.asarray(fs.poly_filters(host, MAX_LAG_L, fs.POLY_DEGREE))
        del host
    tables = jax.tree.map(jnp.asarray, tables)
    centre = np.mean(orbits.compute_position(np.asarray(response.wavelet_times),
                                             sc=[1, 2, 3]), axis=1)
    return info, wdm, response, dict(fs.setup(response, tables, centre), poly=jnp.asarray(poly))


def mismatch(a, b):
    a, b = a.ravel(), b.ravel()
    return 1 - a @ b/np.sqrt((a @ a)*(b @ b))


def save(out, nfs):
    orbits, start, signal_times, _ = fig14.setup()
    for Nf in nfs:
        info, wdm, response, consts = build(Nf, orbits, start)
        params = dict(MONOCHROMATIC_PARAMS[1], f=(round(3e-3/wdm.dF) + 0.4)*wdm.dF)
        _, hp, hc = waveforms.generate_windowed_waveform(
            "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                             info["pad_left"], info["pad_right"])
        wp, wc = block(wdm.dwt(hp)), block(wdm.dwt(hc))
        ref = fs.response_split_w(wdm, consts, GC_THETA, GC_PHI, wp, wc, filters="poly")
        np.save(out/f"wp_{Nf}.npy", np.asarray(wp))
        np.save(out/f"wc_{Nf}.npy", np.asarray(wc))
        np.save(out/f"ref_{Nf}.npy", np.asarray(ref))
        print(f"saved Nf={Nf}", flush=True)
        del response, wdm, consts
        jax.clear_caches()


def load(src, nfs):
    import fast_split_gpu as fg

    orbits, start, _, _ = fig14.setup()
    print(f"jax {jax.__version__} on {jax.devices()[0]}", flush=True)
    for Nf in nfs:
        _, wdm, response, consts = build(Nf, orbits, start)
        wp, wc = (jnp.asarray(np.load(src/f"{k}_{Nf}.npy")) for k in ("wp", "wc"))
        f, stages = fg.make_response(wdm, consts)                 # polyw
        f_table, _ = fg.make_response(wdm, consts, filters="table")
        groups = {g: fg.make_response(wdm, consts, links_per_launch=g)[0] for g in GROUPS}
        theta, phi = GC_THETA, GC_PHI

        tick = time.perf_counter()
        ours = np.asarray(block(f(theta, phi, wp, wc)))
        first = time.perf_counter() - tick
        ref = np.load(src/f"ref_{Nf}.npy")
        rel = np.linalg.norm(ours - ref)/np.linalg.norm(ref)
        mm = mismatch(ours, ref)
        del ours
        for g, h in list(groups.items()):
            try:
                alt = np.asarray(block(h(theta, phi, wp, wc)))
                print(f"   x{g} vs CPU f64 poly: mismatch {mismatch(alt, ref):.1e}", flush=True)
                del alt
            except jax.errors.JaxRuntimeError as err:
                print(f"   x{g}: {str(err).splitlines()[0][:90]}", flush=True)
                del groups[g]
        del ref

        def timed(pieces):
            compile_s = {}
            for k, g in pieces.items():
                tick = time.perf_counter()
                block(g())
                compile_s[k] = time.perf_counter() - tick
            best = {k: np.inf for k in pieces}
            for _ in range(REPS):
                for k, g in pieces.items():
                    tick = time.perf_counter()
                    block(g())
                    best[k] = min(best[k], time.perf_counter() - tick)
            return best, compile_s

        # whole calls first, with nothing else held on the card
        best, compile_s = timed({
            "production": lambda: response(theta, phi, wp, wc),
            "corrected": lambda: f(theta, phi, wp, wc),
            "corrected_table": lambda: f_table(theta, phi, wp, wc),
            **{f"corrected_x{g}": (lambda h=h: h(theta, phi, wp, wc)) for g, h in groups.items()},
        })
        # then the stages of the corrected call, from float32 inputs as it uses them
        wp32, wc32 = stages["cast"](wp), stages["cast"](wc)
        rows, sums = block(stages["common"](consts, theta, phi, wp32, wc32))
        grids = block(stages["grids"](consts, rows, wp32, wc32, sums))
        stage_best, _ = timed({
            "common": lambda: stages["common"](consts, theta, phi, wp32, wc32),
            "grids": lambda: stages["grids"](consts, rows, wp32, wc32, sums),
            "shifts": lambda: stages["shifts"](rows, grids),
        })
        best.update(stage_best)
        print(f"Nf={Nf} Nt={wdm.Nt}: " + "  ".join(f"{k} {v*1e3:.1f}" for k, v in best.items())
              + f" ms | corrected/production {best['corrected']/best['production']:.2f}x | "
              f"vs CPU f64: rel L2 {rel:.1e}, mismatch {mm:.1e} | first call {first:.1f} s, "
              f"production {compile_s['production']:.1f} s", flush=True)
        del rows, sums, grids, wp32, wc32
        del response, wdm, consts, wp, wc
        jax.clear_caches()


if __name__ == "__main__":
    groups = [a.split("=")[1] for a in sys.argv if a.startswith("--groups=")]
    if groups:
        GROUPS = tuple(int(g) for g in groups[0].split(","))
    nfs = [int(a) for a in sys.argv[1:] if not a.startswith("--") and a.isdigit()]
    nfs = nfs or [256, 1024, 2500]
    if "--save" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--save") + 1]); out.mkdir(parents=True, exist_ok=True)
        save(out, nfs)
    else:
        load(Path(sys.argv[sys.argv.index("--load") + 1]), nfs)
