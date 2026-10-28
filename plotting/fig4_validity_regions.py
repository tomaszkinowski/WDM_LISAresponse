"""Figure 4: where in (f, dT) the approximations behind the WDM response hold.

Grey regions are where an approximation fails (overlaps darken):
  narrowband            dF << 1/tau, with dF dT = 1/2 and tau <= 2l  =>  dT >> l/c
  short-atom            dT << 1 yr
  slowly varying delay  dT << 1 / (2 pi f |dDelta/dt|), |dDelta/dt| <= v_orb/c
The yellow band is the dT in [1, 10] h we recommend. Vertical lines are the three
example Galactic binaries, in the PALETTE slots their curves use in later figures. The x-axis f
stands in for f_eff in the slowly-varying-delay condition.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import lisaconstants as lc

from _colours import PALETTE

HERE = Path(__file__).parent
plt.style.use(HERE / "paper.mplstyle")

ARM_LENGTH = 2.5e9                                      # nominal LISA arm [m]
YEAR = lc.SIDEREALYEAR_J2000DAY * 86400                 # [s]
V_ORB = 2 * np.pi * lc.ASTRONOMICAL_UNIT / YEAR         # LISA's orbital speed [m/s]

DT_NARROWBAND = ARM_LENGTH / lc.SPEED_OF_LIGHT          # ~8.3 s
DT_SHORT_ATOM = YEAR
DELAY_RATE = V_ORB / lc.SPEED_OF_LIGHT                  # max |dDelta/dt| ~ 1e-4


def dt_slow_delay(f):
    return 1 / (2 * np.pi * f * DELAY_RATE)


F_LIM = (1e-5, 1.0)
DT_LIM = (1.0, 3e8)
DT_BAND = (3600.0, 36000.0)                             # 3600-36000 s
F_GB = [3e-4, 3e-3, 3e-2]                               # example Galactic binaries [Hz]

GREY = dict(color="k", alpha=0.25, lw=0)
EDGE = dict(color="k", ls=":", lw=0.8)

fig, ax = plt.subplots(figsize=(3.375, 2.2))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*F_LIM)
ax.set_ylim(*DT_LIM)

f = np.logspace(np.log10(F_LIM[0]), np.log10(F_LIM[1]), 400)

# Regions where an approximation fails
ax.axhspan(DT_LIM[0], DT_NARROWBAND, **GREY)
ax.axhspan(DT_SHORT_ATOM, DT_LIM[1], **GREY)
ax.fill_between(f, np.minimum(dt_slow_delay(f), DT_LIM[1]), DT_LIM[1], **GREY)

ax.axhline(DT_NARROWBAND, **EDGE)
ax.axhline(DT_SHORT_ATOM, **EDGE)
ax.plot(f, dt_slow_delay(f), **EDGE)

# Recommended resolutions
ax.axhspan(*DT_BAND, color=PALETTE[3], alpha=0.3, lw=0)

# Example Galactic binaries
for f_gb, c in zip(F_GB, PALETTE):
    ax.axvline(f_gb, color=c, lw=1)

# Region labels sit right of the last GB line, the only stretch no vertical line crosses.
ax.text(3.5e-2, np.sqrt(DT_LIM[0] * DT_NARROWBAND), "Narrowband", fontsize=8, va="center")
ax.text(3.5e-2, np.sqrt(DT_SHORT_ATOM * DT_LIM[1]), "Short-atom", fontsize=8, va="center")

# Rotate the diagonal's label to run parallel to the line on screen, sitting above it
f0, f1 = 1e-2, 1e-1
(x0, y0), (x1, y1) = ax.transData.transform([(f0, dt_slow_delay(f0)), (f1, dt_slow_delay(f1))])
angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
# The label is wider than any gap between the GB lines, so it straddles the 3e-2 Hz line:
# start it so the space between "Slowly" and "varying" falls exactly on that line.
renderer = fig.canvas.get_renderer()
def text_width(txt):
    t = ax.text(0, 0, txt, fontsize=8)
    w = t.get_window_extent(renderer).width
    t.remove()
    return w
w_space = text_width("Slowly varying") - text_width("Slowly") - text_width("varying")
along = text_width("Slowly") + 0.5 * w_space                 # anchor -> gap centre, on screen
x_gap = ax.transData.transform([(F_GB[-1], 1)])[0][0]
# Aim at the gap at mid-letter height, not at the baseline: rotated text leans, so the
# gap shifts sideways by (height * sin(angle)) as you go up the letters.
mid_height = 0.5 * renderer.points_to_pixels(8) * 0.45       # half the x-height, roughly
SHIFT_UP_LINE = 6                                             # [pt] slide up-left along the line
x_start = (x_gap - (along + renderer.points_to_pixels(SHIFT_UP_LINE)) * np.cos(np.radians(angle))
           + mid_height * np.sin(np.radians(angle)))
F_LABEL = ax.transData.inverted().transform([(x_start, 0)])[0][0]
ax.text(F_LABEL, dt_slow_delay(F_LABEL) * 2.0, "Slowly varying delay", fontsize=8,
        ha="left", va="bottom", rotation=angle, rotation_mode="anchor", zorder=5)

ax.set_xlabel(r"$f$ [Hz]")
ax.set_ylabel(r"$\Delta T$ [s]")

fig.savefig(HERE / "figures" / "fig4_validity_regions.pdf")
