"""Taper study above dT = 8e4 s: where alpha = 0.05 drops below ~10 rows.

Same setup as taper_study.py (2^23 samples at dt = 4 s, 3 mHz binary at the
Galactic centre, link 1->2, ESA orbits, L = 25), on the on-bin tilings from
8e4 to 5e5 s (Nf a multiple of 250, so f/dF is an integer). Over one year
alpha = 0.05 gives a ramp of alpha N_rows / 2 rows: 10.5 at 8e4 s down to 1.7
at 5e5 s. 10 dT and 20 dT ramps are the controls whose edges should not matter.

Besides the mismatch at each cut depth, every row's error and signal energy
(summed over m >= 1) is cached, so any cut - or a control that removes the
same number of rows from the middle - can be evaluated afterwards.

  (a)  mismatch with no reach against dT, per taper, and alpha = 0.05 with a
       further 10 rows cut; the grey line is the SNR = 100 threshold,
       1 / (2 SNR^2). Top axis: alpha's ramp in rows.
  (b)  signal kept by the taper cut: fraction of signal rows (lines) and of
       the response's energy (open markers).
  (c)  bad region or less data: error over signal energy for alpha = 0.05
       after removing 10 rows per side next to the taper, against removing
       the same 20 rows from the middle of the record.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from _colours import HAIRLINE, MUTED, PALETTE

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "taper_high_dT.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

DT = 4.0
N_SIGNAL = 2 ** 23
F_GW = 3e-3
AMP = 1e-21
THETA, PHI = 1.66742348, 4.65722819             # Galactic centre, ecliptic J2000
MAX_LAG = 25
NUM_INTEGRAL_POINTS = 40
# On-bin (f/dF = 0.024 Nf integer, Nf even): multiples of 250.
NF = np.array([20000, 25000, 30000, 37500, 50000, 62500, 75000, 100000, 125000])
TAPERS = ("alpha", "10 dT", "20 dT")
ALPHA_FIXED = 0.05
REACHES = np.array([0, 1, 2, 3, 5, 10, 25])


def alpha_for(taper, Nf):
    rows = {"alpha": None, "10 dT": 10, "20 dT": 20}[taper]
    return ALPHA_FIXED if rows is None else 2 * rows * Nf / (N_SIGNAL - 1)


def compute():
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
    start = orbit_times[0] + 200 * 86400.0
    signal_times = start + DT * np.arange(N_SIGNAL)
    link = utils.Link(1, 2)

    shape = (len(TAPERS), len(NF))
    out = dict(Nf=NF, reaches=REACHES,
               mismatch=np.full(shape + (len(REACHES),), np.nan),
               pad_only=np.full(shape, np.nan),
               ramp_rows=np.zeros(shape),                 # nominal, alpha N / (2 Nf)
               clean_start=np.zeros(shape, dtype=int),    # first row past the ramp
               clean_stop=np.zeros(shape, dtype=int),
               signal_start=np.zeros(len(NF), dtype=int),
               signal_stop=np.zeros(len(NF), dtype=int))
    row_error, row_signal = {}, {}

    for j, Nf in enumerate(NF):
        info = utils.wdm_padding_info(N_SIGNAL, int(Nf), max_lag_L=MAX_LAG)
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=int(Nf), N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                                max_lag_L=MAX_LAG, link_convention=(link,))
        first, end = info["pad_left"], info["pad_left"] + N_SIGNAL
        pad_rows = slice(info["rows_left"], info["Nt"] - info["rows_right"])
        out["signal_start"][j], out["signal_stop"][j] = pad_rows.start, pad_rows.stop

        for i, taper in enumerate(TAPERS):
            tick = time.perf_counter()
            alpha = alpha_for(taper, Nf)
            k = int(np.floor(alpha * (N_SIGNAL - 1) / 2)) + 1
            clean = slice(-(-(first + k) // Nf), (end - k) // Nf)
            out["ramp_rows"][i, j] = alpha * (N_SIGNAL - 1) / (2 * Nf)
            out["clean_start"][i, j], out["clean_stop"][i, j] = clean.start, clean.stop

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

            row_error[f"row_error_{i}_{j}"] = ((ours - reference) ** 2).sum(-1)
            row_signal[f"row_signal_{i}_{j}"] = (reference ** 2).sum(-1)

            def mismatch(rows):
                if rows.stop - rows.start < 3:
                    return np.nan
                return utils.wdm_mismatch(ours[rows], reference[rows])["mismatch"]

            out["pad_only"][i, j] = mismatch(pad_rows)
            for r, reach in enumerate(REACHES):
                out["mismatch"][i, j, r] = mismatch(slice(clean.start + reach,
                                                          clean.stop - reach))

            print(f"dT={Nf * DT:8.3g} s {taper:>6}: ramp {out['ramp_rows'][i, j]:5.1f} rows, "
                  f"{clean.stop - clean.start:3d} clean of {pad_rows.stop - pad_rows.start} | "
                  f"pad {out['pad_only'][i, j]:.2e}  "
                  + "  ".join(f"+{reach} {m:.2e}" for reach, m in
                              zip(REACHES, out["mismatch"][i, j]))
                  + f"  ({time.perf_counter() - tick:.0f} s)", flush=True)
        jax.clear_caches()

    return out | row_error | row_signal


SNR = 100
THRESHOLD = 1 / (2 * SNR ** 2)                  # two signals distinguishable above this
CONTROL_ROWS = 10                               # rows per side for the region test


def kept_fractions(res, i, j):
    """Fraction of signal rows, and of response energy, inside the taper cut."""
    s0, s1 = res["signal_start"][j], res["signal_stop"][j]
    c0, c1 = res["clean_start"][i, j], res["clean_stop"][i, j]
    energy = res[f"row_signal_{i}_{j}"]
    return (c1 - c0) / (s1 - s0), energy[c0:c1].sum() / energy[s0:s1].sum()


def region_test(res, i, j):
    """Error over signal energy with CONTROL_ROWS per side removed next to the
    taper, and with the same number removed from the middle instead."""
    c0, c1 = res["clean_start"][i, j], res["clean_stop"][i, j]
    error, signal = res[f"row_error_{i}_{j}"], res[f"row_signal_{i}_{j}"]
    rows = np.arange(c0, c1)
    mid = len(rows) // 2

    def ratio(keep):
        return error[keep].sum() / signal[keep].sum()

    edges = rows[CONTROL_ROWS:-CONTROL_ROWS]
    middle = np.concatenate([rows[:mid - CONTROL_ROWS], rows[mid + CONTROL_ROWS:]])
    return ratio(edges), ratio(middle)


def plot(res):
    plt.style.use(HERE / "paper.mplstyle")
    colour = PALETTE[1]
    styles = {"alpha": "-", "10 dT": "--", "20 dT": ":"}
    markers = {"alpha": "o", "10 dT": "s", "20 dT": "^"}
    labels = {"alpha": rf"$\alpha = {ALPHA_FIXED:g}$",
              "10 dT": r"$10\,\Delta T$", "20 dT": r"$20\,\Delta T$"}
    delta_t = res["Nf"] * DT
    reach_10 = int(np.flatnonzero(res["reaches"] == 10)[0])

    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.3, 0.9, 1]))

    # (a) accuracy
    ax = axes[0]
    for i, taper in enumerate(TAPERS):
        ax.plot(delta_t, res["mismatch"][i, :, 0], styles[taper], color=colour,
                marker="o", ms=2.5, lw=1, label=labels[taper])
    ax.plot(delta_t, res["mismatch"][0, :, reach_10], "-", color=colour, alpha=0.4,
            marker="o", ms=2, lw=0.8, label=rf"$\alpha = {ALPHA_FIXED:g}$, $+10$ rows")
    ax.axhline(THRESHOLD, color=MUTED, ls="--", lw=0.8)
    ax.text(delta_t[0], THRESHOLD * 1.4, rf"SNR $= {SNR}$", color=MUTED, fontsize="x-small")
    ax.set_yscale("log")
    ax.set_ylabel("mismatch")
    # Below the threshold line, where the curves have not yet risen.
    ax.legend(fontsize="x-small", loc="upper left", bbox_to_anchor=(0.0, 0.76))

    # (b) signal kept
    ax = axes[1]
    for i, taper in enumerate(TAPERS):
        rows, energy = np.array([kept_fractions(res, i, j) for j in range(len(res["Nf"]))]).T
        ax.plot(delta_t, rows, styles[taper], color=colour, lw=1)
        ax.plot(delta_t, energy, markers[taper], ms=3, mfc="none", mec=colour, mew=0.7)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("fraction kept")
    ax.legend([Line2D([], [], color=colour, ls=styles[t], marker=markers[t], ms=3,
                      mfc="none", mew=0.7, lw=1) for t in TAPERS],
              [labels[t] for t in TAPERS], fontsize="x-small", loc="lower left",
              title="rows: line, energy: marker", title_fontsize="x-small")

    # (c) bad region or less data
    ax = axes[2]
    edges, middle = np.array([region_test(res, 0, j) for j in range(len(res["Nf"]))]).T
    ax.plot(delta_t, edges, "-", color=colour, marker="o", ms=2.5, lw=1,
            label=f"{CONTROL_ROWS} rows/side cut next to the taper")
    ax.plot(delta_t, middle, "--", color=colour, marker="o", ms=2.5, lw=1,
            label=f"{2 * CONTROL_ROWS} rows cut from the middle")
    ax.set_yscale("log")
    ax.set_ylabel(r"error / signal, $\alpha = 0.05$")
    ax.set_xlabel(r"$\Delta T$ [s]")
    ax.legend(fontsize="x-small", loc="upper left")

    ticks = [1e5, 2e5, 3e5, 5e5]
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([r"$10^5$", r"$2\times10^5$",
                                                     r"$3\times10^5$", r"$5\times10^5$"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(color=HAIRLINE, lw=0.5)

    # alpha's ramp in rows, along the top - after the log scale, which resets it.
    top = axes[0].secondary_xaxis("top")
    top.xaxis.set_major_locator(FixedLocator(delta_t[::2]))
    top.xaxis.set_major_formatter(FixedFormatter([f"{r:.1f}" for r in res["ramp_rows"][0, ::2]]))
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel(r"$\alpha = 0.05$ ramp [rows]", fontsize="small")

    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "taper_high_dT.pdf")
    print(f"wrote {HERE / 'figures' / 'taper_high_dT.pdf'}")

if __name__ == "__main__":
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute()
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plot(results)
