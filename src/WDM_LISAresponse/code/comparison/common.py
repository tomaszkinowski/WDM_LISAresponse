r"""What every external comparison shares.

The external codes return the fractional frequency shift :math:`y` of the
received light, in their polarisation basis; this package returns the extra
light travel time :math:`\delta\ell`, in seconds, in its own. Two signs relate
them:

- Light received at :math:`t` left at :math:`t - \ell(t)`, so a growing delay
  spreads the wavefronts out and lowers the received frequency:
  :math:`y = -\mathrm{d}\ell/\mathrm{d}t`. Their formula has
  :math:`h_\mathrm{em} - h_\mathrm{rec}` where the derivative of our integral
  has :math:`h_\mathrm{rec} - h_\mathrm{em}`.
- Their :math:`(\hat{u}, \hat{v})` is ours turned by 90 degrees -
  :math:`\hat{u}_\mathrm{theirs} = \hat{v}`,
  :math:`\hat{v}_\mathrm{theirs} = -\hat{u}` - which flips both antenna
  patterns, ``BASIS_SIGN``.

The two cancel:

.. math::

    \mathrm{theirs} = +\frac{\mathrm{d}}{\mathrm{d}t}\,\mathrm{ours} .

Their formula is the derivative of our delay with the constellation frozen.
The spacecraft moving together multiply it by
:math:`1 - \hat{\Omega}\cdot\mathbf{v}/c`, with :math:`\hat{\Omega} = -\hat{n}`
the propagation direction, up to :math:`10^{-4}`; ``velocity_factor`` applies
that to their output. Each arm's ends moving apart and its direction turning
add terms of relative size :math:`\sim 1/(f \times 1\,\mathrm{yr})`,
:math:`10^{-5}` at 5 mHz and growing as :math:`1/f`. They need our geometry
and window integral, so they are not applied, and they set the floor of any
comparison against these codes. Cornish & Rubbo's Eq. 9 drops further terms of
the same order, which none of the codes has.
"""

import re

import numpy as np
from lisaconstants import c

from ..lisa_response.response import DEFAULT_DIRECTED_LINKS
from ..lisa_response.response_sub import build_sky_basis
from ..utils.utils import wdm_mismatch


#: Their antenna patterns are minus ours: the bases differ by 90 degrees.
BASIS_SIGN = -1.0

#: Our default link order, labelled receiver-first like ``lisaconstants.LINKS``.
LINK_LABELS = tuple(10 * r + s for s, r in DEFAULT_DIRECTED_LINKS)


def check_version(name : str, version : str, minimum : tuple) -> None:
    """Refuse a release older than ``minimum``, e.g. ``(1, 2, 1)``.

    Pre-release tags are ignored, so ``1.2.1a0`` counts as ``(1, 2, 1)``.
    """
    release = re.match(r"\d+(\.\d+)*", version).group()
    release = tuple(int(part) for part in release.split("."))

    assert release >= tuple(minimum), \
                f"{name} {version} is older than " \
                f"{'.'.join(map(str, minimum))}, the oldest release this " \
                f"comparison was checked against."


def sky_angles(theta, phi) -> tuple:
    r"""Our polar angle and azimuth as the longitude and latitude the external
    codes take.

    ``fastlisaresponse`` names them ecliptic :math:`(\lambda, \beta)`,
    ``lisagwresponse`` ICRS :math:`(\alpha, \delta)`. Neither converts the
    orbit file on load, so both are in the file's frame, as ours is.
    """
    return float(phi), float(np.pi / 2.0 - theta)


def velocity_factor(orbits, times, theta, phi) -> np.ndarray:
    r"""The factor :math:`1 - \hat{\Omega}\cdot\bar{\mathbf{v}}/c` that turns
    their link response into ours, to first order in :math:`v/c`.

    :math:`\bar{\mathbf{v}}` is the mean spacecraft velocity. Using it rather
    than each link's own two spacecraft leaves ~1e-7, part of the terms this
    factor does not cover. It varies over the year, so it is evaluated every
    1000 s and interpolated.

    Parameters
    ----------
    orbits : lisaorbits.Orbits
        Provider with ``compute_velocity``, e.g. ``response.orbits_class``.
    times : np.ndarray
        Sample times, in seconds. Array shape=(N,).
    theta, phi : float
        Source direction, in our convention.

    Returns
    -------
    factor : np.ndarray
        Multiply their output by this. Array shape=(N,).
    """
    _, _, n = build_sky_basis(theta, phi)
    omega = -np.asarray(n)

    times = np.asarray(times, dtype=float)
    knots = np.linspace(times[0], times[-1],
                        int((times[-1] - times[0]) // 1000.0) + 2)
    velocity = np.asarray(orbits.compute_velocity(knots)).mean(axis=1)

    return 1.0 - np.interp(times, knots, velocity @ omega) / c


def to_wdm_units(series, cfg) -> np.ndarray:
    r"""Their fractional frequency as our delay, in our basis.

    Integrated spectrally, so exact for a band-limited record; the constant is
    unobservable and dropped. The sign is ``BASIS_SIGN`` times the :math:`-1`
    of :math:`y = -\mathrm{d}\ell/\mathrm{d}t`, so :math:`+1`.

    Parameters
    ----------
    series : np.ndarray
        Their output on our grid. Array shape=(..., cfg.N).
    cfg : ComparisonConfig
        Supplies ``N`` and ``dt``.

    Returns
    -------
    converted : np.ndarray
        Delay in seconds. Array shape=(..., cfg.N).
    """
    series = np.asarray(series, dtype=float)

    assert series.shape[-1] == cfg.N, \
                f"Expected {cfg.N} samples on our grid, got {series.shape[-1]}."

    frequencies = np.fft.rfftfreq(cfg.N, d=cfg.dt)
    integral = np.zeros_like(frequencies, dtype=complex)
    integral[1:] = 1.0 / (2.0j * np.pi * frequencies[1:])

    sign = BASIS_SIGN * -1.0

    return sign * np.fft.irfft(np.fft.rfft(series, axis=-1) * integral,
                               n=cfg.N, axis=-1)


def convert(series, cfg, wdm, factor=None) -> np.ndarray:
    """Their output to wavelet grids in our units.

    Applied to *their* output only, so the quantity under test is never
    touched by what makes it comparable.

    Parameters
    ----------
    series : np.ndarray
        Their channels or links on our grid. Array shape=(num_rows, cfg.N).
    cfg : ComparisonConfig
        Supplies ``N`` and ``dt``.
    wdm : WDM.WDM_transform
        The transform, from ``cfg.build()``.
    factor : np.ndarray or None
        ``velocity_factor``, applied first when given. Optional.

    Returns
    -------
    grids : np.ndarray
        Array shape=(num_rows, Nt, Nf), in seconds.
    """
    series = np.asarray(series, dtype=float)

    if factor is not None:
        series = series * np.asarray(factor)

    return np.stack([np.asarray(wdm.dwt(row))
                     for row in to_wdm_units(series, cfg)])


def interior(cfg, dead_samples : int = 0) -> slice:
    """Wavelet rows every code has valid output on.

    Ours is limited by the guard band. ``dead_samples`` is the widest unusable
    stretch at either end of the other codes' records, and must fit inside it.
    """
    guard = min(cfg.padding["rows_left"], cfg.padding["rows_right"]) * cfg.Nf

    assert dead_samples <= guard, \
                f"{dead_samples} dead samples do not fit in the " \
                f"{guard}-sample guard band."

    return slice(cfg.padding["rows_left"], cfg.Nt - cfg.padding["rows_right"])


def report(grids : dict,
           rows : slice,
           reference : str,
           measure : str = "rel_L2",
           channels = ("X", "Y", "Z")) -> str:
    """Tabulate every method in ``grids`` against ``reference``.

    Parameters
    ----------
    grids : dict
        Method name to its channels on the wavelet grid.
        Array shape=(num_channels, Nt, Nf).
    rows : slice
        Rows to compare on, from ``interior``.
    reference : str
        The key the others are compared against.
    measure : str
        Any entry of ``utils.wdm_mismatch``'s result. Optional.
    channels : sequence of str
        Channel names, for the table. Optional.

    Returns
    -------
    text : str
        One line per channel, one column per method.
    """
    others = [name for name in grids if name != reference]
    width = max(11, *(len(name) + 2 for name in others))

    lines = [f"{measure} against {reference}",
             "ch " + "".join(f"{name:>{width}}" for name in others)]

    for i, channel in enumerate(channels):
        ours = np.asarray(grids[reference][i])[rows]
        values = [wdm_mismatch(ours, np.asarray(grids[name][i])[rows])[measure]
                  for name in others]
        lines.append(f"{channel:<3}"
                     + "".join(f"{value:>{width}.2e}" for value in values))

    return "\n".join(lines)
