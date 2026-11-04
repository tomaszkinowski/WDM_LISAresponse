"""The detector quantities taken to the time each row lands at.

The shift carries row n from t_n to t_n - delta_n, so the amplitudes fed to it
are built from the geometry at that landing time. These pin the helper to its
definition, q(t_n - delta_n), on series whose value at any time is known in
closed form, and check that the response is built from it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from WDM_LISAresponse.code.lisa_response import response_sub
from WDM_LISAresponse.code.lisa_response.response_sub import (
    evaluate_at_landing_times,
)

NUM_ROWS = 40
ROW_SPACING = 1000.0

#: One cycle per 50 rows: faster, relative to the rows, than the orbital
#: quantities at any tiling the paper uses (thousands of rows per cycle at
#: Delta T = 1e4 s; ~31 at 1e6 s, where the delay is 1e-3 of a row), so a pass
#: here covers the real ones.
PERIOD = 50.0 * ROW_SPACING


def _series(times):
    """A smooth quantity known at every time."""
    return 1.3 + 0.4 * np.sin(2.0 * np.pi * times / PERIOD + 0.2)


def _rows():
    return np.arange(NUM_ROWS) * ROW_SPACING


def test_a_zero_shift_returns_the_rows():
    """With no delay every row lands where it started."""
    values = jnp.asarray(_series(_rows()))[None]

    got = evaluate_at_landing_times(values, jnp.zeros((1, NUM_ROWS)), ROW_SPACING)

    assert np.allclose(got, values, rtol=0, atol=1e-14)


def test_each_row_is_taken_to_its_own_landing_time(rng):
    """Row n comes back as q(t_n - delta_n), with delta varying row to row.

    Every row, the end rows included: the stencil stays on the rows there
    rather than flattening them, so nothing relies on the record's padding.
    """
    rows = _rows()
    shift = rng.uniform(-450.0, 450.0, size=NUM_ROWS)

    got = evaluate_at_landing_times(
        jnp.asarray(_series(rows))[None], jnp.asarray(shift)[None], ROW_SPACING)

    assert np.allclose(np.asarray(got)[0], _series(rows - shift),
                       rtol=0, atol=1e-11)


def test_a_positive_delay_reads_an_earlier_time():
    """delta > 0 lands a row earlier, so it must read the quantity earlier.

    A ramp makes the direction unmistakable: the landed value is smaller than
    the row's own by exactly slope * delta.
    """
    slope = 1e-3
    rows = _rows()
    values = jnp.asarray(slope * rows)[None]

    got = evaluate_at_landing_times(
        values, jnp.full((1, NUM_ROWS), 300.0), ROW_SPACING)

    assert np.allclose(np.asarray(got)[0], slope * (rows - 300.0),
                       rtol=0, atol=1e-12)


@pytest.mark.parametrize("shift", [4500.0, -4500.0])
def test_landing_times_off_the_record_are_extrapolated(shift):
    """Rows landing beyond either end read the quantity there, not a frozen copy.

    A delay of 4.5 rows - the size it reaches at the finest tiling - puts the
    first (or last) rows' landing times off the record. The series is slow, as
    the geometry is at such a tiling, so extrapolating it is accurate.
    """
    rows = _rows()
    period = 300.0 * ROW_SPACING
    series = 1.3 + 0.4 * np.sin(2.0 * np.pi * rows / period + 0.2)

    got = evaluate_at_landing_times(
        jnp.asarray(series)[None], jnp.full((1, NUM_ROWS), shift), ROW_SPACING)

    expected = 1.3 + 0.4 * np.sin(2.0 * np.pi * (rows - shift) / period + 0.2)
    assert np.allclose(np.asarray(got)[0], expected, rtol=0, atol=1e-9)


def test_every_source_and_link_lands_by_its_own_delay(rng):
    """Leading axes are independent: each (source, link) uses its own delta."""
    rows = _rows()
    shift = rng.uniform(-450.0, 450.0, size=(3, 2, NUM_ROWS))
    values = np.broadcast_to(_series(rows), shift.shape)

    got = evaluate_at_landing_times(
        jnp.asarray(values), jnp.asarray(shift), ROW_SPACING)

    assert got.shape == shift.shape
    for i in range(3):
        for j in range(2):
            single = evaluate_at_landing_times(
                jnp.asarray(values[i, j])[None], jnp.asarray(shift[i, j])[None],
                ROW_SPACING)
            assert np.array_equal(np.asarray(got[i, j]), np.asarray(single[0]))


def test_the_delay_may_broadcast_against_the_quantities(rng):
    """One delay per (source, link, row) serves a stack of quantities."""
    rows = _rows()
    shift = rng.uniform(-450.0, 450.0, size=(2, NUM_ROWS))
    stacked = np.stack([_series(rows), 2.0 * _series(rows)])[:, None].repeat(2, 1)

    got = evaluate_at_landing_times(
        jnp.asarray(stacked), jnp.asarray(shift), ROW_SPACING)

    assert np.allclose(got[1], 2.0 * got[0], rtol=1e-14)


def test_the_response_builds_its_amplitudes_at_the_landing_times(
        response, wavelet_waveform, monkeypatch):
    """The wavelet response feeds the amplitudes the landed geometry.

    Replacing the helper with the identity must change the result, and
    replacing it with itself must not: the step is used, and used once.
    """
    theta, phi = 1.1, 2.3
    landed = np.asarray(response.one_arm_response(theta, phi, *wavelet_waveform))

    monkeypatch.setattr(response_sub, "evaluate_at_landing_times",
                        lambda values, shift, spacing: values)
    response_sub.one_arm_response_wdm.clear_cache()
    unlanded = np.asarray(response.one_arm_response(theta, phi, *wavelet_waveform))
    monkeypatch.undo()
    response_sub.one_arm_response_wdm.clear_cache()
    again = np.asarray(response.one_arm_response(theta, phi, *wavelet_waveform))

    assert not np.array_equal(landed, unlanded)
    assert np.array_equal(landed, again)
