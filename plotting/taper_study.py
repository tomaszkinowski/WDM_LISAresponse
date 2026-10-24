"""Taper study: which window, and does the cut need a reach past it?

Wavelet one-arm response against the time-domain reference, one link, for the
3 mHz Galactic binary at the Galactic centre, over 2^23 samples at dt = 4 s
(388 days), scanning dT from 1e2 to 1e6 s. Only the window changes:

  alpha   fixed Tukey fraction, 0.05 (about 9.7 d per ramp)
  10 dT   each ramp ten wavelet rows long, so it scales with the tiling
  1e4 s   each ramp 1e4 s, whatever the tiling

Grey shading marks dT where the source leaks into the dropped m = 0 band
(f < (1 - A_frac) dF, see DT_WINDOW_EDGE); the dotted line is f = dF/2.

Every grid is compared on the rows that hold neither padding nor a tapered
sample ("reach no"), and on those minus a further L rows at each end ("reach
yes", L being the shift kernel's maximum lag). The full mismatch-against-cut-
depth curve is cached too, so other reaches can be read off without rerunning.

Run with no arguments to compute (or reuse the cache) and plot. ``--only NF``
runs one tiling, prints it and caches nothing - for checking time and memory.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _colours import HAIRLINE, MUTED, PALETTE
from parameters import NUM_INTEGRAL_POINTS

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "taper_study.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

DT = 4.0                                        # sample cadence [s]
N_SIGNAL = 2 ** 23                              # 388 d
F_GW = 3e-3                                     # [Hz], PALETTE[1] in every figure
AMP = 1e-21
# The Galactic centre, in ecliptic J2000 - the orbit file's frame (LISA's
# orbital plane is within 0.006 deg of its xy-plane). Latitude -5.54 deg,
# longitude 266.84 deg.
THETA, PHI = 1.66742348, 4.65722819
MAX_LAG = 25
REACHES = np.array([0, 1, 2, 3, 5, 10, 25])     # rows past the taper; 25 = L
REACH_YES = int(np.flatnonzero(REACHES == MAX_LAG)[0])

# 1e2 to 1e6 s, three tilings per decade; Nf must be even.
NF = np.unique(2 * np.round(np.logspace(2, 6, 13) / (2 * DT)).astype(int))

# Below these dT the source reaches the m = 0 band, which is dropped. The band
# edge is dF/2, but the Meyer window extends to (1 - A_frac) dF, so a tone
# between the two still leaks into it: measured 1e-5 of its power at 0.70 dF,
# 1e-1 at 0.55 dF, and nothing above 0.75 dF. WDM_transform's default A_frac.
A_FRAC = 0.25
DT_BAND_EDGE = 0.5 / (2 * F_GW)                  # f = dF/2, 83 s
DT_WINDOW_EDGE = (1 - A_FRAC) / (2 * F_GW)       # f = (1 - A_frac) dF, 125 s

TAPERS = ("alpha", "10 dT", "1e4 s")
ALPHA_FIXED = 0.05
TAPER_ROWS = 10
TAPER_SECONDS = 1.0e4


def alpha_for(taper, Nf):
    """The Tukey fraction each window type asks for on this tiling."""
    ramp = {"alpha": None, "10 dT": TAPER_ROWS * Nf * DT, "1e4 s": TAPER_SECONDS}[taper]
    return ALPHA_FIXED if ramp is None else 2 * ramp / (DT * (N_SIGNAL - 1))


def taper_samples(alpha):
    """Samples scipy's Tukey ramp covers at each end (one over, at most)."""
    return 0 if alpha == 0 else int(np.floor(alpha * (N_SIGNAL - 1) / 2)) + 1


def compute(nfs):
    import jax
    from lisaorbits import InterpolatedOrbits
    from lisaorbits.file import read_orbit_file

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    data = read_orbit_file(str(ORBIT_FILE))
    orbit_times = data.t0 + data.dt * np.arange(data.size)
    orbits = InterpolatedOrbits(orbit_times, data.positions, data.velocities)
    # 200 d in: the widest tiling pads about 160 d ahead of the signal.
    start = orbit_times[0] + 200 * 86400.0
    signal_times = start + DT * np.arange(N_SIGNAL)
    link = utils.Link(1, 2)

    shape = (len(TAPERS), len(nfs))
    out = dict(
        Nf=np.asarray(nfs),
        mismatch=np.full(shape + (len(REACHES),), np.nan),   # pad + taper + reach
        pad_only=np.full(shape, np.nan),                     # no taper cut at all
        kept_rows=np.zeros(shape + (len(REACHES),), dtype=int),
        taper_rows=np.zeros(shape, dtype=int),
        signal_rows=np.zeros(len(nfs), dtype=int),
    )

    for j, Nf in enumerate(nfs):
        info = utils.wdm_padding_info(N_SIGNAL, int(Nf), max_lag_L=MAX_LAG)
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=int(Nf), N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                                max_lag_L=MAX_LAG, link_convention=(link,))
        first, end = info["pad_left"], info["pad_left"] + N_SIGNAL
        pad_rows = slice(info["rows_left"], info["Nt"] - info["rows_right"])
        out["signal_rows"][j] = pad_rows.stop - pad_rows.start

        for i, taper in enumerate(TAPERS):
            tick = time.perf_counter()
            alpha = alpha_for(taper, Nf)
            if alpha > 1.0:
                print(f"Nf={Nf:6d} {taper:>6}: ramps longer than the record, skipped")
                continue
            k = taper_samples(alpha)
            # Rows holding no padding and no tapered sample.
            clean = slice(-(-(first + k) // Nf),
                          (end - k) // Nf if k else -(-end // Nf))
            out["taper_rows"][i, j] = clean.start - pad_rows.start

            _, h_plus, h_cross = waveforms.generate_windowed_waveform(
                "monochromatic", t0=start, dt=DT, N=N_SIGNAL,
                model_kwargs=dict(A_plus=AMP, A_cross=AMP, f=F_GW, phi_0=0.0),
                window_function="tukey", window_kwargs=dict(alpha=alpha))
            _, h_plus, h_cross = utils.apply_zero_padding(
                signal_times, h_plus, h_cross, info["pad_left"], info["pad_right"])

            ours = np.asarray(response.one_arm_response(
                THETA, PHI, wdm.dwt(h_plus), wdm.dwt(h_cross), link=link))
            reference = np.asarray(wdm.dwt(np.asarray(
                response.one_arm_response_time_domain(
                    THETA, PHI, h_plus, h_cross, link=link,
                    num_integral_points=NUM_INTEGRAL_POINTS))))
            ours = ours.reshape(reference.shape)[:, 1:]          # m = 0 dropped
            reference = reference[:, 1:]

            def mismatch(rows):
                return utils.wdm_mismatch(ours[rows], reference[rows])["mismatch"]

            out["pad_only"][i, j] = mismatch(pad_rows)
            for r, reach in enumerate(REACHES):
                rows = slice(clean.start + reach, clean.stop - reach)
                out["kept_rows"][i, j, r] = max(rows.stop - rows.start, 0)
                if rows.stop > rows.start:
                    out["mismatch"][i, j, r] = mismatch(rows)

            print(f"Nf={Nf:6d} dT={Nf * DT:9.3g} s {taper:>6}: ramp {k * DT / 3600:7.2f} h "
                  f"= {out['taper_rows'][i, j]:4d} rows | pad {out['pad_only'][i, j]:.2e}  "
                  + "  ".join(f"+{reach} {m:.2e}" for reach, m in
                              zip(REACHES, out["mismatch"][i, j]))
                  + f"  ({time.perf_counter() - tick:.0f} s)", flush=True)
        jax.clear_caches()

    return out


def plot(res):
    plt.style.use(HERE / "paper.mplstyle")
    colour = PALETTE[1]
    styles = {"alpha": "-", "10 dT": "--", "1e4 s": ":"}
    labels = {"alpha": rf"$\alpha = {ALPHA_FIXED:g}$",
              "10 dT": rf"${TAPER_ROWS}\,\Delta T$",
              "1e4 s": r"$10^4\,\mathrm{s}$"}
    delta_t = res["Nf"] * DT

    fig, axes = plt.subplots(3, 1, figsize=(3.375, 5.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 0.7]))
    for ax in axes:
        # The source leaks into the dropped m = 0 band left of the window edge.
        ax.axvspan(50.0, DT_WINDOW_EDGE, color=MUTED, alpha=0.25, lw=0)
        ax.axvline(DT_BAND_EDGE, color=MUTED, ls=":", lw=0.8)
        ax.grid(color=HAIRLINE, lw=0.5)
        ax.set_xscale("log")
        ax.set_xlim(70.0, 1.4e6)

    for ax, r, title in ((axes[0], 0, "reach no"), (axes[1], REACH_YES, f"reach yes ($L = {MAX_LAG}$ rows)")):
        for i, taper in enumerate(TAPERS):
            ax.plot(delta_t, res["mismatch"][i, :, r], styles[taper], color=colour,
                    marker="o", ms=2.5, lw=1, label=labels[taper])
        ax.set_yscale("log")
        ax.set_ylabel("mismatch")
        ax.set_title(title, fontsize="medium")
    axes[0].legend(title="ramp per end", fontsize="small", title_fontsize="small")
    axes[0].text(DT_BAND_EDGE * 1.08, 0.97, r"$f < \Delta F/2$", rotation=90,
                 va="top", ha="left", transform=axes[0].get_xaxis_transform(),
                 fontsize="x-small", color=MUTED)

    for i, taper in enumerate(TAPERS):
        for r, width, tone in ((0, 1.0, colour), (REACH_YES, 0.7, MUTED)):
            axes[2].plot(delta_t, res["kept_rows"][i, :, r] / res["signal_rows"],
                         styles[taper], color=tone, lw=width)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("fraction kept")
    axes[2].set_xlabel(r"$\Delta T$ [s]")
    axes[2].text(0.02, 0.08, "orange: reach no, grey: reach yes",
                 transform=axes[2].transAxes, fontsize="small", color=MUTED)

    (HERE / "figures" / "investigations").mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / "figures" / "investigations" / "taper_study.pdf")
    print(f"wrote {HERE / 'figures' / 'investigations' / 'taper_study.pdf'}")


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
