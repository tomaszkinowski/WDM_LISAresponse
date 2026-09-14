r"""What the external comparisons share.

The mapping between their output and ours: they return the fractional
frequency shift :math:`y` of the received light, we return the extra light
travel time :math:`\delta\ell` in seconds, and after two signs that cancel
theirs is :math:`+\mathrm{d}/\mathrm{d}t` of ours with the constellation
frozen (``to_wdm_units``, ``BASIS_SIGN``, ``velocity_factor``, ``convert``).
Plus the bookkeeping every comparison needs: sky angles, link labels, the
shared interior and the results table.

The derivation, and the :math:`\sim 10^{-5}` residual it leaves, is in
``examples/response_comparison.ipynb``.
"""

import numpy as np
from lisaconstants import c

from ..lisa_response.response import DEFAULT_DIRECTED_LINKS
from ..lisa_response.response_sub import build_sky_basis
from ..utils.utils import apply_wdm_cut, wdm_cut_info, wdm_mismatch


#: Their antenna patterns are minus ours: the bases differ by 90 degrees.
BASIS_SIGN = -1.0

#: Our link order as the two-digit labels the external codes use: receiver in
#: the tens digit, sender in the units, as ``lisaorbits.LINKS``. So
#: ``Link(2, 1)``, from 2 to 1, is 12.
LINK_LABELS = tuple(10 * r + s for s, r in DEFAULT_DIRECTED_LINKS)


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

    :math:`\bar{\mathbf{v}}` is the mean velocity of the three spacecraft at
    each time, so one factor serves every link and the TDI channels, which mix
    all six. It varies over the year, so it is evaluated every 1000 s and
    interpolated, to 2e-13. Why the mean is enough is in
    ``examples/response_comparison.ipynb``.

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

    Their :math:`y` integrated over time (spectrally, which is exact here),
    since :math:`y = -\mathrm{d}\ell/\mathrm{d}t`. The sign is ``BASIS_SIGN``
    times that :math:`-1`, so :math:`+1`. The result has zero mean: the
    integration constant is not in their output, and it would only touch the
    :math:`m = 0` band, which ``interior`` cuts.

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


def interior(cfg, dead_samples : int = 0) -> dict:
    """The part of the wavelet grid every code has valid output on.

    ``utils.wdm_cut_info`` on this configuration's padding: the guard rows go,
    and so does the :math:`m = 0` band, which ``WDM_transform`` does not
    compute correctly by default. Apply it with ``utils.apply_wdm_cut``.
    ``dead_samples`` is the widest unusable stretch at either end of the other
    codes' records, and must fit inside our guard band.
    """
    pad = cfg.padding
    guard = min(pad["rows_left"], pad["rows_right"]) * cfg.Nf

    assert dead_samples <= guard, \
                f"{dead_samples} dead samples do not fit in the " \
                f"{guard}-sample guard band."

    return wdm_cut_info(pad)


def report(grids : dict,
           cut : dict,
           reference : str,
           measures = ("rel_L2", "mismatch"),
           channels = ("X", "Y", "Z")) -> str:
    """Tabulate every method in ``grids`` against ``reference``.

    Parameters
    ----------
    grids : dict
        Method name to its channels on the wavelet grid.
        Array shape=(num_channels, Nt, Nf).
    cut : dict
        The part of the grid to compare on, from ``interior``.
    reference : str
        The key the others are compared against.
    measures : sequence of str
        Entries of ``utils.wdm_mismatch``'s result, one row each per
        channel. Optional.
    channels : sequence of str
        Channel names, for the table. Optional.

    Returns
    -------
    text : str
        One row per channel and measure, one column per method.
    """
    others = [name for name in grids if name != reference]
    width = max(11, *(len(name) + 2 for name in others))
    label = max(len(measure) for measure in measures) + 2

    lines = [f"against {reference}",
             f"{'ch':<4}{'measure':<{label}}"
             + "".join(f"{name:>{width}}" for name in others)]

    def cut_grid(name, i):
        return apply_wdm_cut(np.asarray(grids[name][i]), cut)

    for i, channel in enumerate(channels):
        ours = cut_grid(reference, i)
        results = [wdm_mismatch(ours, cut_grid(name, i)) for name in others]
        for j, measure in enumerate(measures):
            lead = channel if j == 0 else ""
            lines.append(f"{lead:<4}{measure:<{label}}"
                         + "".join(f"{result[measure]:>{width}.2e}"
                                   for result in results))

    return "\n".join(lines)
