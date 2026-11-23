"""Figure 9: figure 14 for the first-order response (first_order/model.py: the
constellation centre's delay applied to the waveform, with the amplitude-rate, sinc
and first-order delay-rate corrections on the per-link rest), the shading raised to
the source where it lies above the off-bin bound.

Caches cache/fig9_mismatch_vs_dT.npz and cache/fig9_bounds.npz. Options as figure
14's: no arguments computes what is missing and plots; ``--only NF``,
``--only-bounds NF``, ``--part K/COUNT`` and ``--merge COUNT``.
"""

from pathlib import Path

import fig14_mismatch_vs_dT_zeroth_order as fig14

HERE = Path(__file__).parent
CACHE = HERE / "cache" / "fig9_mismatch_vs_dT.npz"
BOUNDS_CACHE = HERE / "cache" / "fig9_bounds.npz"

if __name__ == "__main__":
    from first_order.model import one_link

    fig14.main(CACHE, BOUNDS_CACHE, HERE / "figures" / "fig9_mismatch_vs_dT.pdf",
               model=one_link, source_envelope=True)
