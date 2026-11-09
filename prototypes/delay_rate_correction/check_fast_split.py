r"""
`fast_split.response_split` against `response_corrections.split_responses`
link by link, and the time of each (min of REPS, interleaved).

    python prototypes/delay_rate_correction/check_fast_split.py [log2N] [Nf ...]
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
import fig6_mismatch_vs_dT as fig6
from parameters import (DT, GC_PHI, GC_THETA, MAX_LAG_L, MONOCHROMATIC_PARAMS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)
from WDM_LISAresponse import LISAresponse
from WDM_LISAresponse.code.utils import utils
from WDM_LISAresponse.code.waveforms import waveforms

from fast_split import response_split, roemer_rows, setup
from response_corrections import build_tables, split_responses

REPS = 3
block = jax.block_until_ready


def main(n_signal, nfs):
    orbits, start, _, _ = fig6.setup()
    signal_times = start + DT*np.arange(n_signal)
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

        consts = setup(response, tables, centre)
        roemer = np.asarray(roemer_rows(GC_THETA, GC_PHI, consts["centre"]))

        def old():
            return [split_responses(response, wdm, tables, GC_THETA, GC_PHI, wp, wc,
                                    roemer, j)["split_amp_sinc_first"]
                    for j in range(response.num_links)]

        def new():
            return block(response_split(wdm, consts, GC_THETA, GC_PHI, wp, wc))
        def three():
            return block(response_split(wdm, consts, GC_THETA, GC_PHI, wp, wc, True))
        variants = dict(new=new, three_stages=three)

        def production():
            return block(response(GC_THETA, GC_PHI, wp, wc))

        ref, ours = np.stack(old()), np.asarray(new())
        rel = np.linalg.norm(ours - ref)/np.linalg.norm(ref)
        worst = max(np.linalg.norm(ours[j] - ref[j])/np.linalg.norm(ref[j])
                    for j in range(len(ref)))
        production()
        funcs = dict(old=old, production=production, **variants)
        for f in variants.values():
            f()
        best = {k: np.inf for k in funcs}
        for _ in range(REPS):
            for name, f in funcs.items():
                tick = time.perf_counter()
                f()
                best[name] = min(best[name], time.perf_counter() - tick)
        print(f"Nf={Nf} Nt={wdm.Nt}: rel L2 {rel:.1e} (worst link {worst:.1e}) | "
              + "  ".join(f"{k} {v*1e3:.0f} ms" for k, v in best.items()), flush=True)
        del response, wdm
        jax.clear_caches()


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(2**(args[0] if args else 19), args[1:] or [256])
