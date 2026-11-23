"""Applying a per-link variable time shift to a batch of grids.

``apply_batch_shift`` is a thin wrapper: it flattens the source and link axes
together, sweeps them through ``WDM_transform.apply_variable_time_shift`` in a
``lax.scan``, and unflattens. So the tests aim at the wrapper's own job -
batching, shapes and argument passthrough - rather than at the operator's
accuracy, which belongs to the WDM package.

Comparing every grid against a direct operator call is what does most of the
work here: it pins the wrapper exactly, for real delays, so a mispairing, a
transpose or a negated delay all fail there. The sign test that follows is
belt-and-braces on top of that, kept because the convention is confusing
enough to be worth stating in runnable form.

Deliberately absent: any check of the operator's own accuracy. A zero delay
is only approximately the identity because ``L`` truncates the lag sum, which
is a property of the WDM package at the chosen ``L`` and not of anything here.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from WDM_LISAresponse.code.lisa_response.response_sub import apply_batch_shift


@pytest.fixture
def coefficients(wdm, rng):
    """A batch of two sources across three links."""
    return jnp.asarray(rng.normal(size=(2, 3, wdm.Nt, wdm.Nf)))


@pytest.fixture
def delays(wdm, rng):
    """Per-row delays of a few wavelet rows, varying along the record."""
    return jnp.asarray(rng.uniform(-2.0, 2.0, size=(2, 3, wdm.Nt)) * wdm.dT)


def _packet(wdm, width_rows=20):
    """A narrow wave packet centred in the middle of the record."""
    times = np.arange(wdm.N) * wdm.dt
    centre = 0.5 * wdm.N * wdm.dt
    envelope = np.exp(-0.5 * ((times - centre) / (width_rows * wdm.dt)) ** 2)
    carrier = np.sin(2.0 * np.pi * (5.0 / (2.0 * wdm.Nf * wdm.dt)) * times)
    return envelope * carrier


def _centroid_row(grid):
    """Energy-weighted mean row of a coefficient grid."""
    per_row = np.linalg.norm(np.asarray(grid), axis=-1)
    return float(np.sum(per_row * np.arange(len(per_row))) / per_row.sum())


# ------------------------------------------------------------ the sign of it

@pytest.mark.parametrize("rows", [2.0, -2.0, 4.0])
def test_a_positive_delay_moves_the_signal_earlier(rows, wdm, filter_tables):
    """Passing delta > 0 samples the wave at t + delta.

    Shifting the coefficients is the active transformation and carries the
    opposite sign to shifting the basis functions, so a packet fed through
    with a positive constant delay comes back centred *earlier*.

    Strictly this is redundant: ``test_each_grid_is_shifted_independently``
    compares against the operator directly, so a wrapper that negated the
    delay would fail there too. It is kept because the wrapper makes an
    explicit promise to pass the delay through unnegated, the convention is
    genuinely confusing, and a later refactor could helpfully add a flip.
    Documentation that runs.
    """
    grid = jnp.asarray(wdm.dwt(jnp.asarray(_packet(wdm))))
    before = _centroid_row(grid)
    delay = jnp.full((1, 1, wdm.Nt), rows * wdm.dT)

    shifted = apply_batch_shift(wdm, filter_tables, grid[None, None], delay)

    assert _centroid_row(shifted[0, 0]) == pytest.approx(
        before - rows, abs=0.1)


# ----------------------------------------------------------------- batching

def test_each_grid_is_shifted_independently(
        wdm, filter_tables, coefficients, delays):
    """Every (source, link) pair matches a standalone single-grid call.

    This is the wrapper's entire job. If the flatten, scan or unflatten
    mispairs a grid with another link's delay the result still has the right
    shape and plausible values, so only this catches it.
    """
    got = apply_batch_shift(wdm, filter_tables, coefficients, delays)

    for source in range(coefficients.shape[0]):
        for link in range(coefficients.shape[1]):
            alone = wdm.apply_variable_time_shift(
                filter_tables,
                coefficients[source, link],
                delays[source, link],
            )
            assert np.allclose(got[source, link], alone, rtol=0, atol=1e-14)


def test_permuting_the_inputs_permutes_the_output(
        wdm, filter_tables, coefficients, delays):
    """Reordering the sources reorders the results, pairing intact.

    Each grid must travel with its own delay. Permuting both operands and
    checking the output follows catches a mispairing that the per-grid test
    above could miss if a transpose happened to land on a matching axis.
    """
    straight = apply_batch_shift(wdm, filter_tables, coefficients, delays)
    swapped = apply_batch_shift(
        wdm, filter_tables, coefficients[::-1], delays[::-1])

    assert np.allclose(swapped[0], straight[1], rtol=0, atol=1e-14)
    assert np.allclose(swapped[1], straight[0], rtol=0, atol=1e-14)


def test_a_grid_shifted_by_another_grids_delay_differs(
        wdm, filter_tables, coefficients, delays):
    """Pairing is load-bearing: the wrong delay gives a different answer.

    Without this the permutation test above could pass on a wrapper that
    ignored the delays entirely.
    """
    straight = apply_batch_shift(wdm, filter_tables, coefficients, delays)
    crossed = apply_batch_shift(
        wdm, filter_tables, coefficients, delays[::-1])

    assert not np.allclose(crossed[0], straight[0], rtol=0, atol=1e-8)


# ------------------------------------------------------------------- shapes

def test_the_shape_survives_the_round_trip(wdm, coefficients, delays,
                                           filter_tables):
    """Output shape equals input shape, with Nt and Nf distinct.

    Nt = 64 against Nf = 16 is deliberate: a transpose in the reshape would
    be invisible on a square grid.
    """
    got = apply_batch_shift(wdm, filter_tables, coefficients, delays)

    assert wdm.Nt != wdm.Nf
    assert got.shape == coefficients.shape


@pytest.mark.parametrize("batch, links", [(1, 1), (1, 6), (4, 1), (3, 2)])
def test_any_batch_and_link_count_is_accepted(
        batch, links, wdm, filter_tables, rng):
    """The flattening works for any pair of leading axis lengths."""
    grids = jnp.asarray(rng.normal(size=(batch, links, wdm.Nt, wdm.Nf)))
    shifts = jnp.asarray(rng.normal(size=(batch, links, wdm.Nt)) * wdm.dT)

    got = apply_batch_shift(wdm, filter_tables, grids, shifts)

    assert got.shape == (batch, links, wdm.Nt, wdm.Nf)


def test_the_result_is_real(wdm, filter_tables, coefficients, delays):
    """WDM coefficients are real, and the shift keeps them so.

    The operator evaluates its matrix elements in real arithmetic precisely
    to avoid a complex intermediate; a complex result would mean that
    rearrangement had been undone.
    """
    got = apply_batch_shift(wdm, filter_tables, coefficients, delays)

    assert not np.iscomplexobj(np.asarray(got))
