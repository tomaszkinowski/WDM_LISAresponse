r"""
``split_amp_sinc_first`` with the linearly interpolated filter tables against
the same model with the polynomial filters (`fast_split.add_poly`), both
against the time-domain response, on figure 8's link and tilings: tones at
3 and 30 mHz with in-band offsets r, towards the Galactic centre. Run on the
desktop (full-year time-domain references).

    python prototypes/delay_rate_correction/validate_poly_filters.py [Nf ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plotting"))

from scipy.interpolate import make_interp_spline

import fig14_mismatch_vs_dT_zeroth_order as fig14
import investigation_fig8_delay_rate as inv
from parameters import (DT, GC_PHI, GC_THETA, MONOCHROMATIC_PARAMS, N_SIGNAL,
                        NUM_INTEGRAL_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)
from WDM_LISAresponse.code.lisa_response import response_sub as rs
from WDM_LISAresponse.code.utils import utils
from WDM_LISAresponse.code.waveforms import waveforms

import fast_split as fs

TONES = (3e-3, 3e-2)
OFFSETS = (0.0, 0.25, 0.45)


def main(nfs):
    orbits, start, signal_times, link = fig14.setup()
    for Nf in nfs:
        tick = time.perf_counter()
        info, wdm, response, tables, centre, cut = inv.tiling(orbits, start, link, Nf)
        consts = fs.add_poly(fs.setup(response, tables, centre), wdm)
        times = np.asarray(response.t0 + wdm.times)
        t_rows = np.asarray(response.wavelet_times)
        n_hat = np.asarray(rs.build_sky_basis(GC_THETA, GC_PHI)[2])
        roemer = make_interp_spline(t_rows, centre @ n_hat/rs.c, k=5)(times)
        cutm = lambda a: utils.apply_wdm_cut(np.asarray(a)[0], cut)
        for f0 in TONES:
            for r in OFFSETS:
                f = (round(f0/wdm.dF) + r)*wdm.dF
                params = dict(MONOCHROMATIC_PARAMS[0], f=f)
                _, hp, hc = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
                    window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
                _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                     info["pad_left"], info["pad_right"])
                hp_w, hc_w = inv.tone_warp(params, start)(times + roemer)
                wp, wc = wdm.dwt(hp_w), wdm.dwt(hc_w)
                ref = utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(
                    response.one_arm_response_time_domain(
                        GC_THETA, GC_PHI, hp, hc, link=link,
                        num_integral_points=NUM_INTEGRAL_POINTS,
                        num_taps=12 if f < 0.03 else 20)))), cut)
                out = {k: cutm(fs.response_split_w(wdm, consts, GC_THETA, GC_PHI, wp, wc,
                                                   filters=k)) for k in ("table", "poly")}
                mm = {k: utils.wdm_mismatch(v, ref)["mismatch"] for k, v in out.items()}
                between = utils.wdm_mismatch(out["poly"], out["table"])["mismatch"]
                print(f"Nf={Nf} dT={wdm.dT:.0f} s f={f*1e3:.4f} mHz r={r:.2f}: "
                      f"vs TD  table {mm['table']:.2e}  poly {mm['poly']:.2e}  "
                      f"| poly vs table {between:.1e}", flush=True)
        print(f"   ({time.perf_counter() - tick:.0f} s)", flush=True)
        del response, wdm, consts
        jax.clear_caches()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [256, 1024, 2500])
