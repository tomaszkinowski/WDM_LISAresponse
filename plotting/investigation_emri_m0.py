"""Investigation (not a paper figure): why the EMRI's mismatch in figure 6 climbs at
fine tilings. Figure 6 samples Delta T four times a decade; this resamples
80-3000 s densely and records, at each tiling, besides the mismatch:
  m0        the fraction of the EMRI's WDM power (h_plus and h_cross) in the
            dropped m = 0 band
  compared  the fraction of it in bands 2 to Nf - 2, those figure 6's
            --drop-edge-bands variant compares
  below     the fraction of its Fourier power below (1 - A_frac) Delta F, the
            tones' leak criterion in figure 6
  off_bin   <sin^2(pi f / Delta F)>, weighted by its Fourier power: 0 if all the
            power sat on band centres, 1 if all of it sat halfway between them
and f_quantile, the frequency with POWER_QUANTILE of the Fourier power below it,
which figure 6 uses for the EMRI's shading. The first two test the m = 0 explanation, the third whether it is instead an
on-/off-bin effect like the tones'. Everything else is as in figure 6, whose
setup and reference grid this reuses.

Run with no arguments to compute (or reuse the cache) and plot. ``--only NF`` runs
one tiling, prints it and caches nothing.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _colours import DARK, HAIRLINE, INK, PALETTE
from fig6_mismatch_vs_dT import (A_FRAC, CACHE as FIG6_CACHE, EDGE_FREQ_SLICE,
                                 NF as FIG6_NF, REF_NF, REF_ROWS, setup)
from parameters import (DT, EMRI_PARAMS, MAX_LAG_L, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
CACHE = HERE / "cache" / "investigation_emri_m0.npz"

# 80-3000 s, densely, plus figure 6's tilings in that range so it can look up its
# own points here; Nf must be even.
NF = np.unique(np.concatenate([
    2 * np.round(np.logspace(np.log10(80), np.log10(3000), 40) / (2 * DT)).astype(int),
    FIG6_NF[FIG6_NF * DT <= 3000]]))
# Figure 6 shades the EMRI below (1 - A_frac) Delta F of the frequency with this
# fraction of its Fourier power below it, as a tone is shaded below its own f.
POWER_QUANTILE = 0.01


def compute(nfs):
    import jax

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    orbits, start, signal_times, link = setup()
    theta, phi = EMRI_PARAMS["theta"], EMRI_PARAMS["phi"]

    tick = time.perf_counter()
    td_gen = waveforms.emri_generator()
    _, h_plus, h_cross = waveforms.generate_windowed_waveform(
        "emri", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    del td_gen
    h_plus, h_cross = np.asarray(h_plus), np.asarray(h_cross)
    print(f"EMRI generated ({time.perf_counter() - tick:.0f} s)", flush=True)

    freqs = np.fft.rfftfreq(N_SIGNAL, DT)
    power = np.abs(np.fft.rfft(h_plus)) ** 2 + np.abs(np.fft.rfft(h_cross)) ** 2
    power /= power.sum()
    f_quantile = np.interp(POWER_QUANTILE, np.cumsum(power), freqs)

    def padded(info):
        _, hp, hc = utils.apply_zero_padding(signal_times, h_plus, h_cross,
                                             info["pad_left"], info["pad_right"])
        return hp, hc

    # One reference on figure 6's grid, moved onto each tiling.
    tick = time.perf_counter()
    ref_info = utils.wdm_padding_info(N_SIGNAL, REF_NF, extra_rows_left=REF_ROWS,
                                      extra_rows_right=REF_ROWS)
    ref_response = LISAresponse(
        WDM.WDM.WDM_transform(dt=DT, Nf=REF_NF, N=ref_info["target_length"]), orbits,
        t0=start - ref_info["pad_left"] * DT, max_lag_L=MAX_LAG_L,
        num_interp_points=NUM_INTERP_POINTS, link_convention=(link,))
    reference = np.asarray(ref_response.one_arm_response_time_domain(
        theta, phi, *padded(ref_info), link=link, num_integral_points=NUM_INTEGRAL_POINTS))
    del ref_response
    jax.clear_caches()
    print(f"reference ({time.perf_counter() - tick:.0f} s)", flush=True)

    taper = waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"])
    out = dict(Nf=np.asarray(nfs), f_quantile=f_quantile, **{key: np.full(len(nfs), np.nan)
                                      for key in ("mismatch", "m0", "compared", "below",
                                                  "off_bin")})
    for j, Nf in enumerate(nfs):
        tick = time.perf_counter()
        delta_f = 1 / (2 * Nf * DT)
        out["below"][j] = power[freqs < (1 - A_FRAC) * delta_f].sum()
        out["off_bin"][j] = (power * np.sin(np.pi * freqs / delta_f) ** 2).sum()

        info = utils.wdm_padding_info(N_SIGNAL, int(Nf), max_lag_L=MAX_LAG_L,
                                      taper_samples=taper)
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=int(Nf), N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                                max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS,
                                link_convention=(link,))
        cut = utils.wdm_cut_info(info)
        hp_wdm, hc_wdm = (np.asarray(wdm.dwt(h)) for h in padded(info))
        wdm_power = hp_wdm ** 2 + hc_wdm ** 2                 # (Nt, Nf)
        out["m0"][j] = wdm_power[:, 0].sum() / wdm_power.sum()
        out["compared"][j] = (wdm_power[:, EDGE_FREQ_SLICE].sum()
                              / wdm_power.sum())

        ours = response.one_arm_response(theta, phi, hp_wdm, hc_wdm, link=link)
        moved = utils.apply_repadding(reference, ref_info["pad_left"],
                                      ref_info["pad_right"], info["pad_left"],
                                      info["pad_right"])
        out["mismatch"][j] = utils.wdm_mismatch(
            utils.apply_wdm_cut(np.asarray(ours), cut),
            utils.apply_wdm_cut(np.asarray(wdm.dwt(moved)), cut))["mismatch"]

        del response, wdm
        jax.clear_caches()
        print(f"Nf = {Nf:>4}, dT = {Nf * DT:6.0f} s:  mismatch {out['mismatch'][j]:.2e}  "
              f"m0 {out['m0'][j]:.2e}  below {out['below'][j]:.2e}  "
              f"off-bin {out['off_bin'][j]:.3f}   ({time.perf_counter() - tick:.0f} s)",
              flush=True)

    return out


def plot(res):
    plt.style.use(HERE / "paper.mplstyle")
    delta_t = res["Nf"] * DT
    fig6 = dict(np.load(FIG6_CACHE))
    keep = fig6["Nf"] * DT <= delta_t[-1] * 1.05

    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.0), sharex=True,
                             layout="constrained")
    ax = axes[0]
    ax.plot(delta_t, res["mismatch"], color=DARK, lw=0.8, marker="o", ms=2.2,
            mec="white", mew=0.3, label="dense sweep")
    ax.plot(fig6["Nf"][keep] * DT, fig6["EMRI"][keep], ls="none", marker="D", ms=4,
            mfc="none", mec=PALETTE[4], mew=0.8, label="figure 6")
    ax.set(yscale="log", ylabel="mismatch")
    ax.legend(loc="upper right", fontsize="small", frameon=False)

    ax = axes[1]
    ax.plot(delta_t, res["m0"], color=PALETTE[0], lw=0.8, marker="o", ms=2.2,
            mec="white", mew=0.3, label=r"WDM power in $m = 0$")
    ax.plot(delta_t, res["below"], color=PALETTE[1], lw=0.8, marker="s", ms=2.2,
            mec="white", mew=0.3, label=r"power below $(1 - A_\mathrm{frac})\,\Delta F$")
    ax.set(yscale="log", ylabel="fraction", ylim=(1e-12, 2))
    ax.legend(loc="upper right", fontsize="small", frameon=False)

    ax = axes[2]
    ax.plot(delta_t, res["off_bin"], color=PALETTE[2], lw=0.8, marker="o", ms=2.2,
            mec="white", mew=0.3)
    ax.set(xscale="log", ylim=(0, 1), xlabel=r"$\Delta T = N_f\,\delta t$ [s]",
           ylabel=r"$\langle \sin^2(\pi f / \Delta F) \rangle$")
    ax.axhline(0.5, color=INK, lw=0.6, ls=":")

    for ax in axes:
        ax.grid(color=HAIRLINE, lw=0.4)
        ax.set_axisbelow(True)

    out = HERE / "figures" / "investigations"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "emri_m0.pdf")
    print(f"wrote {out / 'emri_m0.pdf'}")


if __name__ == "__main__":
    if "--only" in sys.argv:
        compute([int(sys.argv[sys.argv.index("--only") + 1])])
        sys.exit()
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute(NF)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plot(results)
