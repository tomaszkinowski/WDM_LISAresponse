"""Figure 4: where in (f, dT) the approximations behind the WDM response hold.

Grey regions are where an approximation fails (overlaps darken):
  narrowband            dF << 1/tau, with dF dT = 1/2 and tau <= 2l  =>  dT >> l/c
  short-atom            dT << 1 yr
  slowly varying delay  dT << 1 / (2 pi f |dDelta/dt|), |dDelta/dt| <= v_orb/c
The yellow band is the dT in [1, 10] h we recommend, nested inside a paler band for
the dT range studied in the results. Vertical lines are the three example Galactic
binaries, in the PALETTE slots their curves use in later figures. The x-axis f
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
DT_BAND = (3600.0, 36000.0)                             # 1-10 h
DT_STUDIED = (1e2, 1e6)                                 # range scanned in the results [s]
F_GB = [3e-4, 1e-3, 1e-2]                               # example Galactic binaries [Hz]

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

# Studied resolutions (pale) with the recommended ones nested inside (stronger)
ax.axhspan(*DT_STUDIED, color=PALETTE[3], alpha=0.2, lw=0)
ax.axhspan(*DT_BAND, color=PALETTE[3], alpha=0.3, lw=0)

# Example Galactic binaries
for f_gb, c in zip(F_GB, PALETTE):
    ax.axvline(f_gb, color=c, lw=1)

# Region labels sit right of the last GB line, the only stretch no vertical line crosses.
ax.text(2e-2, np.sqrt(DT_LIM[0] * DT_NARROWBAND), "Narrowband", fontsize=8, va="center")
ax.text(2e-2, np.sqrt(DT_SHORT_ATOM * DT_LIM[1]), "Short-atom", fontsize=8, va="center")

# Rotate the diagonal's label to follow the line on screen
f0, f1 = 1e-3, 1e-2
(x0, y0), (x1, y1) = ax.transData.transform([(f0, dt_slow_delay(f0)), (f1, dt_slow_delay(f1))])
angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
ax.text(1.2e-2, dt_slow_delay(1.2e-2) * 2.2, "Slowly varying delay", fontsize=7,
        rotation=angle, rotation_mode="anchor")

ax.set_xlabel(r"$f$ [Hz]")
ax.set_ylabel(r"$\Delta T$ [s]")

fig.savefig(HERE / "figures" / "fig4_validity_regions.pdf")
