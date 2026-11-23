"""Figure 2: where in (f, dT) the approximations behind the WDM response hold.

Grey regions are where an approximation fails (overlaps darken):
  narrowband            dF << 1/tau, with dF dT = 1/2 and tau <= 2l  =>  dT >> l/c
  short-atom            dT << 1 yr
  slowly-varying delay  dT << 1 / (2 pi f |dDelta/dt|), |dDelta/dt| <= v_orb/c
The yellow band is the dT in [1, 10] h we recommend. Vertical lines are the three
example Galactic binaries, in the PALETTE slots their curves use in later figures. The x-axis f
stands in for f_eff in the slowly-varying-delay condition. The right axis is dF = 1/(2 dT).

The same regions for split_amp_sinc_first go to figures/investigations/: the waveform
carries the constellation centre's Roemer delay, so the shift carries only each link's
rest delay, n.(x_mid - x_c)/c - l/2 with x_mid the link midpoint. Its rate is at most
pi l / (sqrt(3) c yr) ~ 4.8e-7 (the midpoint sits L/(2 sqrt 3) from the centre and turns
once a year; 4.9e-7 measured over six links, the year and the sky), which moves the
diagonal up; the dashed line is the production diagonal. The amplitude-rate and sinc
terms are corrected to first order, but where their expansion parameters reach one
does not move.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import lisaconstants as lc

from _colours import MUTED, PALETTE

HERE = Path(__file__).parent
plt.style.use(HERE / "paper.mplstyle")

ARM_LENGTH = 2.5e9                                      # nominal LISA arm [m]
YEAR = lc.SIDEREALYEAR_J2000DAY * 86400                 # [s]
V_ORB = 2 * np.pi * lc.ASTRONOMICAL_UNIT / YEAR         # LISA's orbital speed [m/s]

DT_NARROWBAND = ARM_LENGTH / lc.SPEED_OF_LIGHT          # ~8.3 s
DT_SHORT_ATOM = YEAR
DELAY_RATE = V_ORB / lc.SPEED_OF_LIGHT                  # max |dDelta/dt| ~ 1e-4
REST_DELAY_RATE = np.pi * ARM_LENGTH / (np.sqrt(3) * lc.SPEED_OF_LIGHT * YEAR)  # ~4.8e-7


def dt_slow_delay(f, rate=DELAY_RATE):
    return 1 / (2 * np.pi * f * rate)


def delta_f(delta_t):
    """dF = 1 / (2 dT), its own inverse."""
    with np.errstate(divide="ignore"):
        return 0.5 / np.asarray(delta_t, dtype=float)


F_LIM = (1e-5, 1.0)
DT_LIM = (1.0, 3e8)
DT_BAND = (3600.0, 36000.0)                             # 3600-36000 s
F_GB = [3e-4, 3e-3, 3e-2]                               # example Galactic binaries [Hz]

GREY = dict(color="k", alpha=0.25, lw=0)
EDGE = dict(color="k", ls=":", lw=0.8)
FONT = 8


def diagonal_angle(ax, rate):
    """The slow-delay diagonal's angle on screen [deg]."""
    f0, f1 = 1e-2, 1e-1
    (x0, y0), (x1, y1) = ax.transData.transform(
        [(f0, dt_slow_delay(f0, rate)), (f1, dt_slow_delay(f1, rate))])
    return np.degrees(np.arctan2(y1 - y0, x1 - x0))


def label_diagonal(fig, ax, rate, text, split_after):
    """`text` parallel to the diagonal, just above it, placed so that the space after
    word `split_after` falls on the 3e-2 Hz line (the label is wider than any gap
    between the Galactic-binary lines)."""
    angle = diagonal_angle(ax, rate)
    renderer = fig.canvas.get_renderer()

    def text_width(txt):
        t = ax.text(0, 0, txt, fontsize=FONT)
        w = t.get_window_extent(renderer).width
        t.remove()
        return w
    words = text.split()
    head, tail = " ".join(words[:split_after]), " ".join(words[split_after:])
    w_space = text_width(head + " " + tail) - text_width(head) - text_width(tail)
    along = text_width(head) + 0.5 * w_space                 # anchor -> gap centre, on screen
    x_gap = ax.transData.transform([(F_GB[-1], 1)])[0][0]
    # Aim at the gap at mid-letter height, not at the baseline: rotated text leans, so the
    # gap shifts sideways by (height * sin(angle)) as you go up the letters.
    mid_height = 0.5 * renderer.points_to_pixels(FONT) * 0.45   # half the x-height, roughly
    shift_up_line = 6                                           # [pt] slide up-left along the line
    x_start = (x_gap - (along + renderer.points_to_pixels(shift_up_line))
               * np.cos(np.radians(angle)) + mid_height * np.sin(np.radians(angle)))
    f_label = ax.transData.inverted().transform([(x_start, 0)])[0][0]
    ax.text(f_label, dt_slow_delay(f_label, rate) * 2.0, text, fontsize=FONT,
            ha="left", va="bottom", rotation=angle, rotation_mode="anchor", zorder=5)


def label_right(ax, rate, text):
    """`text` parallel to the diagonal, just below it, ending at the right edge: the
    grey wedge above is too narrow to hold it."""
    f_label = 0.92 * F_LIM[1]
    ax.text(f_label, dt_slow_delay(f_label, rate) / 1.6, text, fontsize=FONT,
            ha="right", va="top", rotation=diagonal_angle(ax, rate),
            rotation_mode="anchor", zorder=5)


def draw(path, rate=DELAY_RATE, delay_label="Slowly-varying delay", split_after=1,
         reference_rate=None):
    """The regions with the slow-delay diagonal at `rate`; `reference_rate` adds
    another diagonal, dashed, for comparison. With `split_after` None the label
    ends at the right edge instead of straddling the last Galactic binary."""
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*F_LIM)
    ax.set_ylim(*DT_LIM)

    f = np.logspace(np.log10(F_LIM[0]), np.log10(F_LIM[1]), 400)

    # Regions where an approximation fails
    ax.axhspan(DT_LIM[0], DT_NARROWBAND, **GREY)
    ax.axhspan(DT_SHORT_ATOM, DT_LIM[1], **GREY)
    ax.fill_between(f, np.minimum(dt_slow_delay(f, rate), DT_LIM[1]), DT_LIM[1], **GREY)

    ax.axhline(DT_NARROWBAND, **EDGE)
    ax.axhline(DT_SHORT_ATOM, **EDGE)
    ax.plot(f, dt_slow_delay(f, rate), **EDGE)
    if reference_rate is not None:
        ax.plot(f, dt_slow_delay(f, reference_rate), color=MUTED, ls="--", lw=0.6)

    # Recommended resolutions
    ax.axhspan(*DT_BAND, color=PALETTE[3], alpha=0.3, lw=0)

    # Example Galactic binaries
    for f_gb, c in zip(F_GB, PALETTE):
        ax.axvline(f_gb, color=c, lw=1)

    # Region labels sit right of the last GB line, the only stretch no vertical line crosses.
    ax.text(3.5e-2, np.sqrt(DT_LIM[0] * DT_NARROWBAND), "Narrowband", fontsize=FONT,
            va="center")
    ax.text(3.5e-2, np.sqrt(DT_SHORT_ATOM * DT_LIM[1]), "Short-atom", fontsize=FONT,
            va="center")
    if split_after is None:
        label_right(ax, rate, delay_label)
    else:
        label_diagonal(fig, ax, rate, delay_label, split_after)

    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$\Delta T$ [s]")
    ax.secondary_yaxis("right", functions=(delta_f, delta_f)).set_ylabel(
        r"$\Delta F$ [Hz]")

    fig.savefig(path)
    print("wrote", path)


draw(HERE / "figures" / "fig2_validity_regions.pdf")
draw(HERE / "figures" / "investigations" / "fig2_validity_regions_split_amp_sinc_first.pdf",
     rate=REST_DELAY_RATE, delay_label="Slowly-varying rest delay", split_after=None,
     reference_rate=DELAY_RATE)
