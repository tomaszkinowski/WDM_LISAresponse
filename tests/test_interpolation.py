"""Lagrange interpolation on a uniform grid.

The contract is exactness: an ``n``-tap Lagrange stencil reproduces any
polynomial of degree below ``n`` exactly. Everything else here is edge
behaviour - what happens at the ends of the record, and outside it - which is
where the implementation makes choices a reader would not guess.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from WDM_LISAresponse.code.utils.utils import interpolate_uniform_grid

GRID_START = 0.0
GRID_STEP = 0.5
NUM_SAMPLES = 64


def _grid():
    """Sample coordinates of the reference grid."""
    return GRID_START + np.arange(NUM_SAMPLES) * GRID_STEP


def _interpolate(values, queries, num_taps=12):
    """Call the interpolant with the batch axis this module always uses."""
    return np.asarray(interpolate_uniform_grid(
        jnp.asarray(values)[None, :],
        GRID_START,
        GRID_STEP,
        jnp.asarray(queries)[None, :],
        num_taps,
    ))[0]


def _unit_interval(times):
    """Map the grid onto [-1, 1].

    Polynomials are built in this variable so their values stay of order one.
    Evaluating a degree-11 polynomial directly in ``t`` over a span of 30
    reaches 1e16, where float64 loses the very precision under test - the
    conditioning of the polynomial, not the interpolant, would set the
    result. The map is affine, so degree is preserved and exactness still
    applies.
    """
    grid = _grid()
    midpoint = 0.5 * (grid[0] + grid[-1])
    half_span = 0.5 * (grid[-1] - grid[0])
    return (times - midpoint) / half_span


# ------------------------------------------------------------- the contract

@pytest.mark.parametrize("num_taps", [4, 8, 12])
def test_exact_on_polynomials_below_the_stencil_degree(num_taps, rng):
    """An n-tap stencil reproduces every polynomial of degree < n exactly.

    This is the mathematical contract of Lagrange interpolation and the one
    property worth pinning: it holds for any implementation that is correct,
    and fails for any that is not.
    """
    times = _grid()
    # Query well inside, so edge clamping plays no part.
    queries = rng.uniform(times[16], times[-16], size=40)

    for degree in range(num_taps):
        polynomial = np.polynomial.Polynomial(rng.normal(size=degree + 1))
        sampled = polynomial(_unit_interval(times))

        got = _interpolate(sampled, queries, num_taps)

        expected = polynomial(_unit_interval(queries))
        assert np.allclose(got, expected, rtol=0, atol=1e-11)


def test_a_polynomial_at_the_stencil_degree_is_not_exact(rng):
    """Degree n is beyond the stencil, and must visibly fail.

    Without this the exactness test above could pass vacuously - a function
    returning the right answer for the wrong reason, or a tolerance so loose
    it accepts anything.
    """
    num_taps = 4
    times = _grid()
    queries = rng.uniform(times[16], times[-16], size=40)
    polynomial = np.polynomial.Polynomial(rng.normal(size=num_taps + 1))
    sampled = polynomial(_unit_interval(times))

    got = _interpolate(sampled, queries, num_taps)

    expected = polynomial(_unit_interval(queries))
    assert not np.allclose(got, expected, rtol=0, atol=1e-11)


def test_exact_sample_points_return_the_samples(rng):
    """Querying a grid point returns that sample, to machine precision."""
    times = _grid()
    values = rng.normal(size=NUM_SAMPLES)
    interior = times[16:-16]

    got = _interpolate(values, interior)

    assert np.allclose(got, values[16:-16], rtol=0, atol=1e-13)


@pytest.mark.parametrize("num_taps", [2, 4, 12])
def test_a_constant_series_stays_constant_everywhere(num_taps):
    """Clamping at the edges must not leak amplitude.

    Taps that fall off either end are clamped to the end sample, and the
    weights stay normalised. Dropping those taps instead would leave the
    weights short of one and ramp a constant series towards zero near the
    edges, which this catches and nothing else would.
    """
    times = _grid()
    queries = np.linspace(times[0], times[-1], 501)

    got = _interpolate(np.full(NUM_SAMPLES, 2.5), queries, num_taps)

    assert np.allclose(got, 2.5, rtol=0, atol=1e-12)


def test_two_taps_is_linear_interpolation(rng):
    """The narrowest stencil reduces to the familiar case."""
    times = _grid()
    values = rng.normal(size=NUM_SAMPLES)
    queries = rng.uniform(times[0], times[-1], size=50)

    got = _interpolate(values, queries, num_taps=2)

    assert np.allclose(got, np.interp(queries, times, values), atol=1e-12)


# --------------------------------------------------------------- outside it

@pytest.mark.parametrize("query", [-1.0, -1e-6, 31.5 + 1e-6, 100.0])
def test_queries_outside_the_sampled_span_return_zero(query, rng):
    """Out of range is exactly zero, not an extrapolation.

    Deliberately separate from clamping: a delayed waveform stops at the end
    of the record rather than ramping down across a cell.
    """
    got = _interpolate(rng.normal(size=NUM_SAMPLES), np.array([query]))

    assert got[0] == 0.0


@pytest.mark.parametrize("query", [0.0, 31.5])
def test_the_span_is_inclusive_at_both_ends(query, rng):
    """The first and last samples are inside the range, not outside it."""
    values = rng.normal(size=NUM_SAMPLES)

    got = _interpolate(values, np.array([query]))

    assert got[0] != 0.0


# ------------------------------------------------------------------ refusal

@pytest.mark.parametrize("num_taps", [1, 0, -2, 3, 7])
def test_an_odd_or_degenerate_stencil_raises(num_taps):
    """The stencil must be even and at least two.

    It is centred on the query, so an odd width has no centre, and fewer
    than two taps cannot interpolate at all.
    """
    with pytest.raises(AssertionError, match="num_taps must be even"):
        _interpolate(np.zeros(NUM_SAMPLES), np.array([1.0]), num_taps)


# -------------------------------------------------------------------- shape

def test_the_batch_axis_is_preserved(rng):
    """Output shape follows the query, sharing the leading batch axis."""
    values = jnp.asarray(rng.normal(size=(3, NUM_SAMPLES)))
    queries = jnp.asarray(rng.uniform(2.0, 28.0, size=(3, 17)))

    got = interpolate_uniform_grid(values, GRID_START, GRID_STEP, queries, 12)

    assert got.shape == (3, 17)


def test_each_batch_row_is_independent(rng):
    """Row i of the result depends only on row i of the input."""
    values = rng.normal(size=(3, NUM_SAMPLES))
    queries = rng.uniform(2.0, 28.0, size=(3, 11))

    together = np.asarray(interpolate_uniform_grid(
        jnp.asarray(values), GRID_START, GRID_STEP,
        jnp.asarray(queries), 12))

    for row in range(3):
        alone = _interpolate(values[row], queries[row])
        assert np.allclose(together[row], alone, rtol=0, atol=1e-14)
