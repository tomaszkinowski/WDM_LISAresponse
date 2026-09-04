"""What ``LISAresponse`` refuses to be built from.

Construction runs both setup stages eagerly, so an invalid argument should
surface here rather than as a shape error deep inside a jitted kernel much
later. Every branch of ``validate_parameters`` gets a case.

The derived grids are checked here too: they are fixed at construction and
everything downstream indexes them, so they belong with the constructor
rather than with the response calculation.
"""

import numpy as np
import pytest

from WDM_LISAresponse.code.lisa_response.response import LISAresponse
from WDM_LISAresponse.code.utils.utils import Link

from conftest import DT, MAX_LAG_L, NF, NT, T0


def _build(wdm, orbits, **overrides):
    """Construct a response, with one argument replaced."""
    arguments = dict(wdm_class=wdm, orbits_class=orbits, t0=T0,
                     max_lag_L=MAX_LAG_L)
    arguments.update(overrides)
    return LISAresponse(**arguments)


# ------------------------------------------------------------ the transform

@pytest.mark.parametrize("missing", [
    "dt", "N", "Nf", "Nt", "dT", "dF",
    "build_time_delay_filter_interpolants",
])
def test_a_transform_missing_an_attribute_is_refused(missing, wdm, orbits):
    """The transform contract is spelled out, attribute by attribute.

    Duck typing is deliberate - any object with these members works - so the
    members themselves are the contract and each is worth naming.
    """
    class Incomplete:
        pass

    stand_in = Incomplete()
    for name in ("dt", "N", "Nf", "Nt", "dT", "dF",
                 "build_time_delay_filter_interpolants"):
        if name != missing:
            setattr(stand_in, name, getattr(wdm, name))

    with pytest.raises(AssertionError, match=missing):
        _build(stand_in, orbits)


def test_a_provider_without_compute_position_is_refused(wdm):
    """The orbit contract is one method, and it is checked."""
    class NotAProvider:
        pass

    with pytest.raises(AssertionError, match="compute_position"):
        _build(wdm, NotAProvider())


def test_a_provider_whose_compute_position_is_not_callable_is_refused(wdm):
    """An attribute of the right name but the wrong kind is still wrong."""
    class Impostor:
        compute_position = "not a method"

    with pytest.raises(AssertionError, match="compute_position"):
        _build(wdm, Impostor())


# --------------------------------------------------------------- the numbers

@pytest.mark.parametrize("t0", [np.nan, np.inf, -np.inf])
def test_a_non_finite_epoch_is_refused(t0, wdm, orbits):
    """t0 places the grid on the provider's timeline; it must be a number."""
    with pytest.raises(AssertionError, match="t0 must be finite"):
        _build(wdm, orbits, t0=t0)


@pytest.mark.parametrize("max_lag_L", [0, -1, NT, NT + 5])
def test_a_lag_outside_the_record_is_refused(max_lag_L, wdm, orbits):
    """L must satisfy 0 < L < Nt.

    At L >= Nt the shift's lag sum reaches right around the record, so no
    amount of padding would help and the result would be meaningless.
    """
    with pytest.raises(AssertionError, match="max_lag_L must satisfy"):
        _build(wdm, orbits, max_lag_L=max_lag_L)


@pytest.mark.parametrize("num_interp_points", [1, 0, -3])
def test_too_few_interpolation_points_is_refused(
        num_interp_points, wdm, orbits):
    """One point cannot interpolate the fractional part of a delay."""
    with pytest.raises(AssertionError, match="num_interp_points"):
        _build(wdm, orbits, num_interp_points=num_interp_points)


# ----------------------------------------------------------------- the links

def test_an_empty_link_convention_is_refused(wdm, orbits):
    """A response over no links has nothing to compute."""
    with pytest.raises(AssertionError, match="At least one directed link"):
        _build(wdm, orbits, link_convention=())


@pytest.mark.parametrize("bad", [
    Link(0, 1), Link(4, 1), Link(1, 0), Link(1, 9),
])
def test_a_spacecraft_index_outside_one_to_three_is_refused(
        bad, wdm, orbits):
    """Spacecraft are 1-based and there are three of them.

    A zero here is the likely symptom of someone passing array indices, so
    it is worth catching at construction rather than as an orbit lookup
    failure.
    """
    with pytest.raises(AssertionError, match="1-based"):
        _build(wdm, orbits, link_convention=(bad,))


def test_a_link_from_a_spacecraft_to_itself_is_refused(wdm, orbits):
    """A link joins two distinct spacecraft."""
    with pytest.raises(AssertionError, match="two distinct spacecraft"):
        _build(wdm, orbits, link_convention=(Link(2, 2),))


def test_a_repeated_link_is_refused(wdm, orbits):
    """Duplicates would give two rows of identical geometry.

    Harmless numerically, but it means an index no longer names a unique
    arm, which is the property the whole convention rests on.
    """
    with pytest.raises(AssertionError, match="must be distinct"):
        _build(wdm, orbits, link_convention=(Link(2, 1), Link(2, 1)))


# ------------------------------------------------------------- what gets built

def test_the_tiling_is_copied_from_the_transform(response, wdm):
    """The grid comes from the transform, not from its own arithmetic."""
    assert response.dt == wdm.dt
    assert response.N == wdm.N
    assert response.Nf == wdm.Nf
    assert response.Nt == wdm.Nt
    assert response.dT == wdm.dT
    assert response.dF == wdm.dF


def test_the_resolutions_satisfy_the_uncertainty_relation(response):
    """dF * dT = 1/2, the defining property of the tiling."""
    assert response.dF * response.dT == pytest.approx(0.5, rel=1e-14)


def test_the_sample_grid_starts_at_the_epoch(response):
    """Sample times run t0 + k*dt across the whole record."""
    assert float(response.times[0]) == pytest.approx(T0)
    assert len(response.times) == response.N
    assert float(response.times[-1]) == pytest.approx(
        T0 + (response.N - 1) * DT)


def test_the_wavelet_grid_is_the_sample_grid_at_one_row_spacing(response):
    """Pixel centre times run t0 + n*dT, one per row.

    The stored geometry lives on this grid rather than on the sample grid,
    which is what makes the geometry Nf times cheaper than the time-domain
    path's.
    """
    expected = T0 + np.arange(response.Nt) * response.dT

    assert len(response.wavelet_times) == response.Nt
    assert np.allclose(response.wavelet_times, expected, rtol=1e-14)


def test_the_frequency_grid_starts_at_zero(response):
    """Pixel centre frequencies run 0 to (Nf-1) dF."""
    expected = np.arange(response.Nf) * response.dF

    assert np.allclose(response.wavelet_freqs, expected, rtol=1e-14)


def test_the_geometry_is_stored_per_link_on_the_wavelet_grid(response):
    """Geometry arrays are indexed by link first, then wavelet row."""
    links = len(response.link_convention)

    assert response.light_travel_time.shape == (links, response.Nt)
    assert response.link_direction.shape == (links, response.Nt, 3)
    assert response.receiver_position.shape == (links, response.Nt, 3)


def test_the_filter_tables_are_sized_by_the_lag_and_the_stencil(response):
    """Two tables, 2L+1 lags, and one column per interpolation point."""
    assert response.filter_tables.shape == (
        2, 2 * MAX_LAG_L + 1, response.num_interp_points)


def test_the_link_convention_is_stored_as_links(response):
    """Whatever was passed, the stored convention is Link-typed.

    Downstream code reads ``link_convention[i]`` and reports it to users, so
    a consistent type keeps error messages and reprs legible.
    """
    assert all(isinstance(link, Link) for link in response.link_convention)
    assert response.num_links == len(response.link_convention)


def test_plain_pairs_are_accepted_and_converted(wdm, orbits):
    """Callers are not forced to import Link to build a response."""
    built = _build(wdm, orbits, link_convention=((2, 1), (3, 2)))

    assert built.link_convention == (Link(2, 1), Link(3, 2))
    assert all(isinstance(link, Link) for link in built.link_convention)


def test_a_subset_of_links_builds_only_that_geometry(wdm, orbits):
    """Asking for one arm builds one arm, for debugging."""
    built = _build(wdm, orbits, link_convention=(Link(2, 1),))

    assert built.num_links == 1
    assert built.light_travel_time.shape == (1, NT)
