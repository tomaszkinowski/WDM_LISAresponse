"""The cost of linearly interpolating the time-delay filters at P = NUM_INTERP_POINTS
(appendix B).

(a) The filters T_l and T'_l, |l| <= L, evaluated exactly on 16 points per table
    interval and interpolated linearly from the P-point table the response uses
    (delta / Delta T in [-1/2, 1/2]): the largest error relative to each filter's
    peak, and the worst lag relative to its own peak.
(b) Each example Galactic binary moved to its nearest off-bin frequency
    (m + 1/2) Delta F at Delta T = 1e4 s, link 1 -> 2: the mismatch between the
    response with the P-point table and with one REFINE times finer (its nodes
    include the P-point ones), for the first- and zeroth-order responses.

Printed and written to numbers/interpolation.txt. ``--short`` runs (b) on 2**19
samples (24 d) and writes nothing - a check that fits a laptop.
"""

import sys
from pathlib import Path

import numpy as np

import fig14_mismatch_vs_dT_zeroth_order as fig14
from parameters import (DT, MAX_LAG_L, MONOCHROMATIC_PARAMS, N_SIGNAL,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
NF = 2500                                       # Delta T = 1e4 s
SUBSAMPLE = 16                                  # exact evaluations per table interval
REFINE = 8                                      # the reference table, REFINE x finer


def filter_errors(wdm, points=NUM_INTERP_POINTS):
    """(a): per filter, the largest interpolation error over its peak, and the
    worst lag's error over that lag's own peak, with the lag."""
    from WDM.code.time_delay_filters.filters import build_filter_tables

    lags = np.arange(-MAX_LAG_L, MAX_LAG_L + 1)
    nodes = np.linspace(-0.5, 0.5, points)
    fine = np.linspace(-0.5, 0.5, (points - 1) * SUBSAMPLE + 1)
    table = np.asarray(build_filter_tables(lags, nodes, wdm.d, wdm.A_frac))
    exact = np.asarray(build_filter_tables(lags, fine, wdm.d, wdm.A_frac))
    out = {}
    for index, name in ((1, "T"), (0, "T'")):
        interp = np.array([np.interp(fine, nodes, row) for row in table[index]])
        error = np.abs(interp - exact[index]).max(axis=1)        # per lag
        peak = np.abs(exact[index]).max(axis=1)
        worst = np.argmax(error / peak)
        out[name] = (error.max() / peak.max(), error[worst] / peak[worst], lags[worst])
    return out


def off_bin_mismatch(n_signal, points):
    """(b): per tone, the mismatch between the P-point and the refined table, for
    each response order."""
    import jax

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms
    from first_order.model import one_link

    orbits, start, _, link = fig14.setup()
    signal_times = start + DT * np.arange(n_signal)
    info = utils.wdm_padding_info(
        n_signal, NF, max_lag_L=MAX_LAG_L,
        taper_samples=waveforms.tukey_taper_samples(n_signal, WINDOW_KWARGS["alpha"]))
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=info["target_length"])
    cut = utils.wdm_cut_info(info)
    off = fig14.bound_frequencies([NF])[1][:, 0]
    models = dict(first_order=one_link, zeroth_order=fig14.zeroth_order)

    grids = {}
    for p in (points, (points - 1) * REFINE + 1):
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                                max_lag_L=MAX_LAG_L, num_interp_points=p,
                                link_convention=(link,))
        for order, model in models.items():
            respond = model(response, wdm, link, start, n_signal)
            for i, params in enumerate(MONOCHROMATIC_PARAMS):
                tone = dict(params, f=off[i])
                _, hp, hc = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=n_signal, model_kwargs=tone,
                    window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
                _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                     info["pad_left"], info["pad_right"])
                grids[p, order, i] = utils.apply_wdm_cut(
                    np.asarray(respond(fig14.GC_THETA, fig14.GC_PHI, hp, hc, tone=tone)),
                    cut)
        del response
        jax.clear_caches()
    fine = (points - 1) * REFINE + 1
    return off, {(order, i): utils.wdm_mismatch(grids[points, order, i],
                                                 grids[fine, order, i])["mismatch"]
                 for order in models for i in range(len(MONOCHROMATIC_PARAMS))}


if __name__ == "__main__":
    import WDM

    short = "--short" in sys.argv
    n_signal = 2 ** 19 if short else N_SIGNAL
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=2 * NF * 64)
    lines = [f"P = {NUM_INTERP_POINTS}, L = {MAX_LAG_L}, Meyer d = {wdm.d}, "
             f"A_frac = {wdm.A_frac}", "",
             f"(a) linear interpolation of the filters, against exact values on "
             f"{SUBSAMPLE} points per interval"]
    for name, (overall, own, lag) in filter_errors(wdm).items():
        lines.append(f"  {name:<3} largest error / filter peak {overall:.2e}; worst lag "
                     f"l = {lag:+d}: error / its own peak {own:.2e}")
    off, mismatch = off_bin_mismatch(n_signal, NUM_INTERP_POINTS)
    lines += ["", f"(b) off-bin tones at Delta T = {NF * DT:g} s, link 1 -> 2, "
                  f"N = 2**{int(np.log2(n_signal))} samples: mismatch of P = "
                  f"{NUM_INTERP_POINTS} against P = {(NUM_INTERP_POINTS - 1) * REFINE + 1}"]
    for i, f in enumerate(off):
        lines.append(f"  f = {f * 1e3:.4f} mHz:  first order {mismatch['first_order', i]:.2e}"
                     f",  zeroth order {mismatch['zeroth_order', i]:.2e}")
    text = "\n".join(lines)
    print(text)
    if not short:
        (HERE / "numbers").mkdir(exist_ok=True)
        (HERE / "numbers" / "interpolation.txt").write_text(text + "\n")
