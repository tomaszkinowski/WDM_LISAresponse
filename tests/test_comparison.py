"""The external-code comparisons: the conversion first, then each code.

The first tests need no external code: the basis sign written out from both
definitions, the conversion against a closed form and the velocity factor
against its definition. The rest run ``lisagwresponse``,
``pytdi`` and ``fastlisaresponse`` when they are installed and skip when not.
"""

import importlib.util

import numpy as np
import pytest
from lisaconstants import c

from WDM_LISAresponse.code.comparison import (
    common, config, conventions_flr, conventions_gwr,
)
from WDM_LISAresponse.code.lisa_response.response_sub import (
    antenna_pattern_factors, build_sky_basis,
)
from WDM_LISAresponse.code.utils.utils import (
    apply_wdm_cut, apply_zero_padding, wdm_mismatch,
)


# -- no external code ----------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_their_antenna_patterns_are_minus_ours(seed):
    """Their basis, written out as both external codes build it, is ours
    turned 90 degrees, so both antenna patterns change sign."""
    rng = np.random.default_rng(seed)
    theta, phi = rng.uniform(0.1, np.pi - 0.1), rng.uniform(0.0, 2.0 * np.pi)
    lon, lat = common.sky_angles(theta, phi)

    u = np.array([np.sin(lon), -np.cos(lon), 0.0])
    v = np.array([-np.sin(lat) * np.cos(lon),
                  -np.sin(lat) * np.sin(lon),
                  np.cos(lat)])

    r = rng.normal(size=(16, 3))
    r /= np.linalg.norm(r, axis=-1, keepdims=True)

    theirs = ((r @ u) ** 2 - (r @ v) ** 2, 2.0 * (r @ u) * (r @ v))
    ours = antenna_pattern_factors(r, *build_sky_basis(theta, phi)[:2])

    for their, our in zip(theirs, ours):
        np.testing.assert_allclose(their, common.BASIS_SIGN * np.asarray(our),
                                   atol=1e-14)


def test_the_conversion_is_a_plus_integral():
    """theirs = +d/dt ours, so a cosine of theirs comes back as sin / omega.

    The tone sits on an FFT bin, so the spectral integral is exact.
    """
    cfg = config.ANALYTIC
    omega = 2.0 * np.pi * cfg.m_bin * cfg.dF
    t = np.arange(cfg.N) * cfg.dt

    got = common.to_wdm_units(np.cos(omega * t), cfg)

    np.testing.assert_allclose(omega * got, np.sin(omega * t), atol=1e-10)


def test_the_velocity_factor_is_one_minus_omega_dot_v_over_c():
    """Against the definition, with the velocities taken directly rather than
    interpolated."""
    cfg = config.SHORT
    orbits = cfg.build_orbits()
    times = cfg.padded_times[::1009]

    _, _, n = build_sky_basis(cfg.theta, cfg.phi)
    velocity = np.asarray(orbits.compute_velocity(times)).mean(axis=1)
    want = 1.0 + velocity @ np.asarray(n) / c    # Omega = -n

    got = common.velocity_factor(orbits, times, cfg.theta, cfg.phi)

    np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)


# -- the external codes --------------------------------------------------


#: Every arm keeps 0.26 <= |Omega.r| <= 0.58, so all six links are strong and
#: each arm's two directions differ. At the study's sky the 1-3 arm is nearly
#: along the wave and 15 times weaker.
THETA, PHI = np.radians(90.0), np.radians(170.0)


def _needs(*names):
    """Skip unless every package is installed, without importing any."""
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"not installed: {', '.join(missing)}")


@pytest.fixture(scope="module")
def six_days():
    """Six days on the ESA orbits: an interior to compare on, quick to build."""
    cfg = config.ComparisonConfig.for_esa(N=2 ** 16, theta=THETA, phi=PHI)
    wdm, response = cfg.build()

    h_plus, h_cross = cfg.waveform()
    _, h_plus, h_cross = apply_zero_padding(
        cfg.signal_times, h_plus, h_cross,
        cfg.padding["pad_left"], cfg.padding["pad_right"],
    )

    return cfg, wdm, response, h_plus, h_cross


@pytest.mark.slow
def test_lisagwresponse_links_match_the_time_domain_reference(six_days):
    """With the velocity factor every link agrees at the ~1e-5 floor. A sign,
    unit or link-order error would leave O(1)."""
    _needs("lisagwresponse")
    from lisagwresponse import ReadStrain

    cfg, wdm, response, h_plus, h_cross = six_days

    factor = common.velocity_factor(response.orbits_class, cfg.padded_times,
                                    cfg.theta, cfg.phi)
    theirs = common.convert(
        conventions_gwr.links(ReadStrain, cfg, h_plus, h_cross,
                              cfg.theta, cfg.phi),
        cfg, wdm, factor)

    ours = np.asarray(response.one_arm_response_time_domain(
        cfg.theta, cfg.phi, h_plus, h_cross, out_domain="wdm",
        num_integral_points=cfg.num_integral_points))

    cut = common.interior(cfg, conventions_gwr.dead_samples(cfg))
    ours, theirs = apply_wdm_cut(ours, cut), apply_wdm_cut(theirs, cut)
    for k in range(6):
        assert wdm_mismatch(ours[k], theirs[k])["rel_L2"] < 4e-5


@pytest.mark.slow
def test_blocking_does_not_change_the_answer(six_days):
    """Both codes run in blocks to bound memory; the blocks must not show."""
    _needs("lisagwresponse", "pytdi")
    from lisagwresponse import ReadStrain
    from pytdi import Data
    from pytdi.michelson import X2

    cfg, _, _, h_plus, h_cross = six_days
    args = (ReadStrain, cfg, h_plus, h_cross, cfg.theta, cfg.phi)

    whole = conventions_gwr.links(*args, chunk_size=cfg.N)
    blocked = conventions_gwr.links(*args, chunk_size=5000)

    np.testing.assert_array_equal(blocked, whole)

    whole = conventions_gwr.channels(Data, (X2,), cfg, whole,
                                     block_size=cfg.N)
    blocked = conventions_gwr.channels(Data, (X2,), cfg, blocked,
                                       block_size=10000)

    np.testing.assert_allclose(blocked, whole, rtol=0,
                               atol=1e-12 * np.abs(whole).max())


@pytest.mark.slow
def test_fastlisaresponse_matches_lisagwresponse_and_pytdi(six_days):
    """The same link formula in independent code: links and XYZ agree to a few
    1e-6. Before 1.2.1a0 they differed in sign."""
    _needs("fastlisaresponse", "lisatools", "lisagwresponse", "pytdi")

    # Checked before any import: the clash aborts the interpreter.
    report = conventions_flr.check_environment(raise_on_problem=False)
    if "vendor separate copies" in report:
        pytest.skip(report)

    from fastlisaresponse import pyResponseTDI
    from lisagwresponse import ReadStrain
    from lisatools.detector import Orbits
    from pytdi import Data
    from pytdi.michelson import X2, Y2, Z2

    cfg, _, _, h_plus, h_cross = six_days

    orbits = Orbits(cfg.orbit_file)
    orbits.configure(linear_interp_setup=True)

    flr = conventions_flr.run(pyResponseTDI, orbits, cfg, h_plus, h_cross,
                              cfg.theta, cfg.phi)
    gwr_links = conventions_gwr.links(ReadStrain, cfg, h_plus, h_cross,
                                      cfg.theta, cfg.phi)
    gwr = (gwr_links,
           conventions_gwr.channels(Data, (X2, Y2, Z2), cfg, gwr_links))

    cut = common.interior(cfg, conventions_flr.dead_samples(cfg))
    rows = cut["time_slice"]
    samples = slice(rows.start * cfg.Nf, rows.stop * cfg.Nf)

    for theirs, other in zip(flr, gwr):
        for a, b in zip(theirs, other):
            a, b = a[samples], b[samples]
            assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-5
