"""Comparing two coefficient grids.

``wdm_mismatch`` answers two questions at once: how similar are the grids in
shape, and how far apart are they outright. The first is scale-blind and
sign-maximised, the second is neither, and keeping both is what makes a small
mismatch beside a large residual readable as "same shape, wrong amplitude".

The mismatch is evaluated through a normalised residual rather than by
subtracting an overlap from one. That is the property most worth pinning: the
direct form loses everything below about 1e-15, which is exactly the regime
two near-identical implementations land in.
"""

import numpy as np
import pytest

from WDM_LISAresponse.code.utils.utils import wdm_mismatch

SHAPE = (64, 16)


@pytest.fixture
def grid(rng):
    """A reference grid to compare things against."""
    return rng.normal(size=SHAPE)


# ------------------------------------------------------------- the easy cases

def test_a_grid_matches_itself_exactly(grid):
    """Identical grids have zero mismatch and unit correlation."""
    result = wdm_mismatch(grid, grid)

    assert result["mismatch"] == pytest.approx(0.0, abs=1e-15)
    assert result["correlation"] == pytest.approx(1.0, abs=1e-12)
    assert result["rel_L2"] == pytest.approx(0.0, abs=1e-15)


def test_uncorrelated_grids_are_almost_completely_mismatched(grid, rng):
    """Unrelated noise gives a mismatch close to one."""
    result = wdm_mismatch(grid, rng.normal(size=SHAPE))

    assert 0.9 < result["mismatch"] <= 1.0


# ------------------------------------------------- the numerical contract

@pytest.mark.parametrize("epsilon", [1e-3, 1e-6, 1e-9, 1e-11])
def test_it_stays_accurate_far_below_the_naive_floor(epsilon, grid, rng):
    """Mismatch tracks the analytic value down to 1e-22 and beyond.

    Perturbing a grid by a small random amount gives a mismatch of about
    ``epsilon^2 / 2``. The direct form ``1 - |c|`` subtracts two numbers that
    are both nearly one, so it is already wrong by 1e-15 and returns exact
    zero below 1e-17. The residual form computes the small number directly.
    """
    perturbed = grid + epsilon * rng.normal(size=SHAPE)

    got = wdm_mismatch(grid, perturbed)["mismatch"]

    assert got == pytest.approx(0.5 * epsilon ** 2, rel=0.2)


def test_the_naive_form_would_have_failed_here(grid, rng):
    """The comparison that justifies the implementation's extra work.

    At a perturbation of 1e-9 the direct overlap difference has lost most of
    its significant digits. If this ever stops being true the residual form
    is no longer buying anything and could be simplified away.
    """
    perturbed = grid + 1e-9 * rng.normal(size=SHAPE)

    stable = wdm_mismatch(grid, perturbed)["mismatch"]

    unit_a = grid / np.linalg.norm(grid)
    unit_b = perturbed / np.linalg.norm(perturbed)
    naive = 1.0 - abs(float(np.dot(unit_a.ravel(), unit_b.ravel())))

    assert stable == pytest.approx(0.5e-18, rel=0.2)
    assert abs(naive - stable) > 0.1 * stable


# --------------------------------------------------------------- what it sees

def test_a_sign_flip_is_maximised_over_but_still_reported(grid):
    """Mismatch ignores an overall sign; ``correlation`` does not.

    Real coefficients have exactly one global freedom, and it is maximised
    over so a flipped grid does not read as a total disagreement. The signed
    correlation is kept so the flip is still visible.
    """
    result = wdm_mismatch(grid, -grid)

    assert result["mismatch"] == pytest.approx(0.0, abs=1e-15)
    assert result["correlation"] == pytest.approx(-1.0, abs=1e-12)


def test_mismatch_ignores_scale_and_the_residual_does_not(grid):
    """The two diagnostics answer different questions, deliberately.

    A grid scaled by three has the same shape and three times the amplitude.
    Reporting only the mismatch would hide that completely.
    """
    result = wdm_mismatch(grid, 3.0 * grid)

    assert result["mismatch"] == pytest.approx(0.0, abs=1e-15)
    assert result["rel_L2"] == pytest.approx(2.0, rel=1e-12)


def test_the_norms_are_reported_unnormalised(grid):
    """``norm1`` and ``norm2`` are the raw squared norms."""
    result = wdm_mismatch(grid, 2.0 * grid)

    assert result["norm1"] == pytest.approx(float(np.sum(grid ** 2)))
    assert result["norm2"] == pytest.approx(4.0 * result["norm1"])


def test_it_is_symmetric_in_its_arguments(grid, rng):
    """Swapping the grids leaves mismatch and correlation unchanged."""
    other = grid + 0.3 * rng.normal(size=SHAPE)

    forward = wdm_mismatch(grid, other)
    backward = wdm_mismatch(other, grid)

    assert forward["mismatch"] == pytest.approx(backward["mismatch"])
    assert forward["correlation"] == pytest.approx(backward["correlation"])


def test_any_shape_is_accepted(rng):
    """The inner product runs over every element, whatever the rank.

    Cut grids arrive with link and batch axes still attached, and comparing
    them should not require flattening at the call site.
    """
    batched = rng.normal(size=(3, 6, 32, 8))

    result = wdm_mismatch(batched, batched)

    assert result["mismatch"] == pytest.approx(0.0, abs=1e-15)


# ------------------------------------------------------------------- weights

def test_a_per_band_weight_broadcasts_across_the_grid(grid, rng):
    """Weights may be given per frequency band and broadcast over time.

    That is the shape a noise power spectrum arrives in, and requiring the
    caller to tile it would be friction for no reason.
    """
    weights = 1.0 / (1.0 + np.arange(SHAPE[1])) ** 2
    perturbed = grid + 1e-4 * rng.normal(size=SHAPE)

    weighted = wdm_mismatch(grid, perturbed, weights=weights)["mismatch"]
    flat = wdm_mismatch(grid, perturbed)["mismatch"]

    assert np.isfinite(weighted)
    assert weighted != pytest.approx(flat, rel=1e-6)


def test_uniform_weights_agree_with_no_weights(grid, rng):
    """A constant weight rescales both norms and cancels out."""
    perturbed = grid + 1e-3 * rng.normal(size=SHAPE)

    weighted = wdm_mismatch(grid, perturbed, weights=np.full(SHAPE, 7.0))
    flat = wdm_mismatch(grid, perturbed)

    assert weighted["mismatch"] == pytest.approx(flat["mismatch"], rel=1e-10)


@pytest.mark.parametrize("weights", [
    np.full(7, 1.0),
    np.ones((3, 3)),
])
def test_weights_that_cannot_broadcast_raise(weights, grid):
    """A weight array of the wrong shape is a mistake, not a broadcast."""
    with pytest.raises(ValueError, match="cannot broadcast"):
        wdm_mismatch(grid, grid, weights=weights)


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_negative_or_non_finite_weights_raise(bad, grid):
    """Weights are a measure; negative or infinite entries are invalid."""
    weights = np.ones(SHAPE[1])
    weights[3] = bad

    with pytest.raises(ValueError, match="finite and non-negative"):
        wdm_mismatch(grid, grid, weights=weights)


# ------------------------------------------------------- degenerate input

@pytest.mark.parametrize("spoil", [np.nan, np.inf, -np.inf])
def test_non_finite_input_returns_nan_rather_than_raising(spoil, grid):
    """A spoiled grid reports nan across the board.

    Mismatches are computed in sweeps where one bad point should not stop
    the run, so this returns rather than raises - but every field goes nan
    so the result cannot be mistaken for a good one.
    """
    other = grid.copy()
    other[2, 3] = spoil

    result = wdm_mismatch(grid, other)

    assert all(np.isnan(value) for value in result.values())


def test_a_zero_grid_returns_nan(grid):
    """A grid with no power has no direction, so nothing to compare."""
    result = wdm_mismatch(grid, np.zeros(SHAPE))

    assert np.isnan(result["mismatch"])


def test_mismatched_shapes_raise(grid):
    """Two grids of different shape cannot be compared at all."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        wdm_mismatch(grid, np.zeros((32, 16)))


def test_the_mismatch_never_goes_negative(grid, rng):
    """Rounding must not push a norm below zero.

    The mismatch is half a squared norm, so a negative value would be
    impossible in exact arithmetic and meaningless if it escaped.
    """
    for epsilon in [0.0, 1e-16, 1e-14, 1e-12]:
        perturbed = grid + epsilon * rng.normal(size=SHAPE)
        assert wdm_mismatch(grid, perturbed)["mismatch"] >= 0.0
