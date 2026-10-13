"""Colours shared by every paper figure.

PALETTE is a colour-blind-safe categorical palette: take colours from the front,
in order, so the same slot means the same thing from one figure to the next.
The neutrals are for anything that is structure rather than data (grids,
reference lines, shading).
"""

# blue, orange, aqua, yellow, magenta. Consecutive slots stay distinguishable
# under colour blindness; slots 2 and 4 (orange, yellow) are the closest pair,
# so avoid putting those two right next to each other.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

DARK = "#2b2b2b"        # a fourth series that should read as "the main one"
INK = "#52514e"         # text and annotations
MUTED = "#898781"       # secondary lines
HAIRLINE = "#e1e0d9"    # grids
SHADE = "#f3f2ee"       # shaded regions
