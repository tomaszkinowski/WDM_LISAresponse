"""Taper study, continued: does the ramp-length prescription depend on L?

Same setup as taper_study.py (2^23 samples at dt = 4 s, 3 mHz binary at the
Galactic centre, link 1->2, ESA orbits), at the two on-bin tilings where the
edges matter, dT = 1e4 and 1e5 s. Tukey ramps of 2-20 rows per end, and the
fixed alpha = 0.05, each at maximum lag L = 15, 25 and 51.

  (a), (b)  mismatch with no reach against ramp length, one line per L; the
            thin dotted line is each L's floor (the deepest cut).
  (c)       at 1e5 s, mismatch against the rows cut past the taper, one line
            per ramp length; solid L = 25, dashed L = 51.

Ramp lengths are nominal, R dT of taper per end (scipy adds one sample, so a
ramp touches R + 1 rows).

Run with no arguments to compute (or reuse the cache) and plot: 36 runs of
about 20 s each, 2 GB peak. The cache in use was parsed from the first run's
printed output, so it carries two significant figures; delete it to recompute
at full precision.
"""

import itertools
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from _colours import HAIRLINE, MUTED, PALETTE
from parameters import NUM_INTEGRAL_POINTS

HERE = Path(__file__).parent
ROOT = HERE.parent
CACHE = HERE / "cache" / "taper_L_study.npz"
ORBIT_FILE = ROOT / "examples" / "esa-trailing-orbits.h5"

DT = 4.0                                        # sample cadence [s]
N_SIGNAL = 2 ** 23                              # 388 d
F_GW = 3e-3                                     # [Hz], PALETTE[1] in every figure
AMP = 1e-21
THETA, PHI = 1.66742348, 4.65722819             # Galactic centre, ecliptic J2000
NFS = [2500, 25000]                             # dT = 1e4, 1e5 s; f/dF = 60, 600
LS = [15, 25, 51]
RAMPS = [2, 5, 10, 15, 20]                      # nominal rows per end
ALPHA_FIXED = 0.05
REACHES = np.array([0, 1, 2, 3, 5, 10, 25, 51])  # rows cut past the taper


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

    out = {key: [] for key in ("delta_t", "L", "is_alpha", "ramp_rows",
                               "rows_touched", "mismatch")}
    for Nf, L in itertools.product(NFS, LS):
        info = utils.wdm_padding_info(N_SIGNAL, Nf, max_lag_L=L)
        wdm = WDM.WDM.WDM_transform(dt=DT, Nf=Nf, N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                                max_lag_L=L, link_convention=(link,))
        first, end = info["pad_left"], info["pad_left"] + N_SIGNAL

        for is_alpha, value in [(False, R) for R in RAMPS] + [(True, ALPHA_FIXED)]:
            tick = time.perf_counter()
            alpha = value if is_alpha else 2 * value * Nf / (N_SIGNAL - 1)
            k = int(np.floor(alpha * (N_SIGNAL - 1) / 2)) + 1
            # Rows holding no padding and no tapered sample.
            clean = slice(-(-(first + k) // Nf), (end - k) // Nf)

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

            mismatch = []
            for reach in REACHES:
                rows = slice(clean.start + reach, clean.stop - reach)
                mismatch.append(
                    utils.wdm_mismatch(ours[rows], reference[rows])["mismatch"]
                    if rows.stop - rows.start > 2 else np.nan)

            out["delta_t"].append(Nf * DT)
            out["L"].append(L)
            out["is_alpha"].append(is_alpha)
            out["ramp_rows"].append(alpha * (N_SIGNAL - 1) / (2 * Nf))
            out["rows_touched"].append(clean.start - info["rows_left"])
            out["mismatch"].append(mismatch)
            print(f"dT={Nf * DT:7.0e} L={L:2d} ramp={out['ramp_rows'][-1]:6.2f} rows | "
                  + " ".join(f"+{r} {m:.1e}" for r, m in zip(REACHES, mismatch))
                  + f"  ({time.perf_counter() - tick:.0f} s)", flush=True)
        jax.clear_caches()

    return {key: np.asarray(value) for key, value in out.items()} | {"reaches": REACHES}


def sci_label(value, _):
    """1e-8 as 10^-8, 2e-8 as 2 x 10^-8."""
    exponent = int(np.floor(np.log10(value) + 1e-9))
    mantissa = round(value / 10.0 ** exponent)
    lead = "" if mantissa == 1 else rf"{mantissa}\times"
    return rf"${lead}10^{{{exponent}}}$"


def plot(res):
    plt.style.use(HERE / "paper.mplstyle")
    reaches = res["reaches"]
    colour = PALETTE[1]                                  # 3 mHz in every figure
    l_style = {15: ":", 25: "-", 51: "--"}
    # Lighter for shorter ramps, from the 3 mHz orange towards white.
    shades = {R: tuple(np.array(plt.matplotlib.colors.to_rgb(colour)) * (0.45 + 0.55 * i / 4)
                       + (1 - (0.45 + 0.55 * i / 4)))
              for i, R in enumerate(RAMPS)}

    def select(dT, L, alpha=False):
        keep = (res["delta_t"] == dT) & (res["L"] == L) & (res["is_alpha"] == alpha)
        order = np.argsort(res["ramp_rows"][keep])
        return res["ramp_rows"][keep][order], res["mismatch"][keep][order]

    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.0),
                             gridspec_kw=dict(height_ratios=[1, 1, 1.15]))

    for ax, dT, tag in ((axes[0], 1e4, "a"), (axes[1], 1e5, "b")):
        for L, style in l_style.items():
            ramp, mm = select(dT, L)
            ax.plot(ramp, mm[:, 0], style, color=colour, marker="o", ms=2.5, lw=1)
            ax.axhline(np.nanmin(res["mismatch"][(res["delta_t"] == dT) & (res["L"] == L)]),
                       color=MUTED, ls=":", lw=0.6)
            a_ramp, a_mm = select(dT, L, alpha=True)
            ax.plot(a_ramp, a_mm[:, 0], marker="*", ms=6, color=colour, ls="none")
        ax.set_yscale("log")
        # Labels at 1, 2 and 5 per decade: panel (a) spans less than one.
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(sci_label))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0, 22) if dT == 1e5 else ax.set_xlim(0, 88)
        ax.set_xlabel("ramp per end [rows]")
        ax.set_ylabel("mismatch, no reach")
        ax.set_title(rf"({tag}) $\Delta T = 10^{{{int(np.log10(dT))}}}$ s", fontsize="medium")
        ax.grid(color=HAIRLINE, lw=0.5)

    axes[0].legend(
        [Line2D([], [], color=colour, ls=s, marker="o", ms=2.5, lw=1) for s in l_style.values()]
        + [Line2D([], [], color=colour, marker="*", ms=6, ls="none")],
        [rf"$L = {L}$" for L in l_style] + [rf"$\alpha = {ALPHA_FIXED:g}$"],
        fontsize="small", ncol=2, loc="center right")

    ax = axes[2]
    x = np.arange(len(reaches))                          # reaches are unevenly spaced
    for R in RAMPS:
        for L, style in ((25, "-"), (51, "--")):
            ramp, mm = select(1e5, L)
            ax.plot(x, mm[list(np.round(ramp)).index(R)], style, color=shades[R],
                    marker="o", ms=2, lw=1)
    ax.set_xticks(x, [str(r) for r in reaches])
    ax.set_yscale("log")
    ax.set_xlabel("rows cut past the taper")
    ax.set_ylabel("mismatch")
    ax.set_title(r"(c) $\Delta T = 10^5$ s, reach", fontsize="medium")
    ax.grid(color=HAIRLINE, lw=0.5)
    ax.legend([Line2D([], [], color=shades[R], lw=1.5) for R in RAMPS]
              + [Line2D([], [], color=MUTED, ls="-", lw=1),
                 Line2D([], [], color=MUTED, ls="--", lw=1)],
              [f"{R} rows" for R in RAMPS] + [r"$L = 25$", r"$L = 51$"],
              fontsize="x-small", ncol=2, loc="upper right", handlelength=1.4,
              columnspacing=0.8)

    fig.tight_layout()
    (HERE / "figures" / "investigations").mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / "figures" / "investigations" / "taper_L_study.pdf")
    print(f"wrote {HERE / 'figures' / 'investigations' / 'taper_L_study.pdf'}")


if __name__ == "__main__":
    if CACHE.exists():
        results = dict(np.load(CACHE))
    else:
        results = compute()
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, **results)
    plot(results)
