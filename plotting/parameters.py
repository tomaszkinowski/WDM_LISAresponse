"""Shared source and sampling parameters for the paper plots.

Sky positions use solar-system barycentric J2000 ecliptic coordinates:
theta is colatitude from the north ecliptic pole, and phi is longitude
from the vernal equinox. All angles passed to waveform/response code are
in radians. Positions below use the rounded values reported in the paper.
Importing this module does not generate waveforms or load orbit data.
"""

from math import pi, radians


# Sampling before tapering and padding: about 388 days.
DT = 4.0  # seconds
N_SIGNAL = 2**23
T_SIGNAL = N_SIGNAL * DT  # seconds; last sample is at (N_SIGNAL - 1) * DT

# The taper of every signal. Padding depends on the grid, so each script computes
# it once Nf and the maximum lag are chosen.
WINDOW_FUNCTION = "tukey"
WINDOW_KWARGS = dict(alpha=0.05)

# Time-shift settings passed to LISAresponse. With linear interpolation of the
# time-delay filters, the worst case (an off-bin tone) has relative L2 error
# ~0.19/(P-1)^2, so P=128 gives ~1e-5, well below the ~1e-4 local-delay term.
# P sets only the one-off table build, not the per-waveform cost.
MAX_LAG_L = 25
NUM_INTERP_POINTS = 128

# Gauss-Legendre nodes per link integral in the time-domain reference,
# one_arm_response_time_domain (unrelated to NUM_INTERP_POINTS above). Twelve
# leave the quadrature below 1e-13 up to 0.1 Hz on the worst link.
NUM_INTEGRAL_POINTS = 12

# Circularly polarised monochromatic test sources towards the Galactic centre.
GC_THETA = radians(95.6)
GC_PHI = radians(266.9)
FREQUENCIES = (3e-4, 3e-3, 3e-2)  # GW frequencies in Hz
A_PLUS = 1e-21  # dimensionless strain, before windowing
A_CROSS = 1e-21
PHI_0 = 0.0  # phase at absolute t=0 for the monochromatic generator
MONOCHROMATIC_PARAMS = tuple(
    dict(A_plus=A_PLUS, A_cross=A_CROSS, f=f, phi_0=PHI_0)
    for f in FREQUENCIES
)

# Illustrative EMRI at the Virgo Cluster distance, using M87's direction.
# This is not an inspiral into M87's own central black hole.
VIRGO_THETA = radians(75.58)
VIRGO_PHI = radians(182.06)
EMRI_PARAMS = dict(
    m1=1e6,  # primary mass in solar masses
    m2=10.0,  # secondary mass in solar masses
    a=0.9,  # dimensionless Kerr spin
    p0=8.00,  # initial semilatus rectum in units of GM/c^2; plunges ~0.5 yr after the window
    e0=0.3,
    x0=1.0,  # prograde equatorial orbit relative to the primary spin
    dist=0.0165,  # luminosity distance in Gpc: 16.5 Mpc
    theta=VIRGO_THETA,
    phi=VIRGO_PHI,
    qK=pi / 3,  # spin colatitude
    phiK=pi / 3,  # spin longitude
    Phi_phi0=pi / 3,
    Phi_theta0=0.0,  # no independent polar motion for an equatorial orbit
    Phi_r0=pi / 3,
    eps=1e-2,  # FEW mode-selection threshold
)

# For generate_windowed_waveform, pass a separately constructed FEW generator
# with model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS). Use the same theta and
# phi from EMRI_PARAMS when evaluating the detector response.
