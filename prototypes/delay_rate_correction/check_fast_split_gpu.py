r"""
`fast_split_gpu.make_response` against `fast_split.response_split_w`, in the
dtype given (float64 checks the algebra exactly; float32 is the GPU setting).
``--interpret`` runs the kernels in the Pallas interpreter (any device, small
records only).

    python prototypes/delay_rate_correction/check_fast_split_gpu.py [--interpret] [--log2n=K] [Nf ...]
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
import fast_split_gpu as fg
from response_corrections import build_tables

block = jax.block_until_ready


def mismatch(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return 1 - a @ b/np.sqrt((a @ a)*(b @ b))


def main(nfs, n_signal, dtypes):
    orbits, start, _, _ = fig8.setup()
    signal_times = start + DT*np.arange(n_signal)
    print(f"jax {jax.__version__} on {jax.devices()[0]}, interpret={fg.INTERPRET}", flush=True)
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
        consts = fs.add_poly(fs.setup(response, tables, centre), wdm)
        refs = {k: np.asarray(fs.response_split_w(wdm, consts, GC_THETA, GC_PHI, wp, wc,
                                                  filters=k)) for k in ("table", "poly")}
        d = np.linalg.norm(refs["poly"] - refs["table"])/np.linalg.norm(refs["table"])
        print(f"Nf={Nf}: CPU f64 poly vs table filters: rel L2 {d:.1e}, "
              f"mismatch {mismatch(refs['poly'], refs['table']):.1e}", flush=True)
        for dtype, kind in ((d_, k_) for d_ in dtypes for k_ in ("table", "poly")):
            ref = refs[kind]
            f, _ = fg.make_response(wdm, consts, dtype, kind)
            tick = time.perf_counter()
            ours = np.asarray(block(f(GC_THETA, GC_PHI, wp, wc)))
            first = time.perf_counter() - tick
            rel = np.linalg.norm(ours - ref)/np.linalg.norm(ref)
            print(f"Nf={Nf} Nt={wdm.Nt} {jnp.dtype(dtype).name} {kind}: rel L2 {rel:.1e}, "
                  f"mismatch {mismatch(ours, ref):.1e} (first call {first:.1f} s)", flush=True)
        del response, wdm, consts
        jax.clear_caches()


if __name__ == "__main__":
    fg.INTERPRET = "--interpret" in sys.argv
    log2n = [int(a.split("=")[1]) for a in sys.argv if a.startswith("--log2n=")]
    nfs = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    main(nfs or [256], 2**log2n[0] if log2n else N_SIGNAL, (jnp.float64, jnp.float32))
