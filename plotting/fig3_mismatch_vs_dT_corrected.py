"""Figure 3: figure 8 for the split_amp_sinc_first response (the constellation
centre's delay applied to the waveform, with the amplitude-rate, sinc and
first-order delay-rate corrections on the per-link rest), the shading raised to
the source where it lies above the off-bin bound.

Plots the caches investigation_fig8_delay_rate.py writes for that variant; run
that script to make them.
"""

from pathlib import Path

import numpy as np

import fig8_mismatch_vs_dT as fig8

HERE = Path(__file__).parent
KIND = "split_amp_sinc_first"
CACHES = (HERE / "cache" / f"investigation_fig8_{KIND}.npz",
          HERE / "cache" / f"investigation_fig8_{KIND}_bounds.npz")

if __name__ == "__main__":
    fig8.plot(*(dict(np.load(path)) for path in CACHES),
              HERE / "figures" / "fig3_mismatch_vs_dT_corrected.pdf", source_envelope=True)
