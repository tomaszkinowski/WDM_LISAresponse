"""Does the guard band actually do its job?

An integration test, and honest about it: each case generates a waveform, pads
it, builds a transform, builds a response, runs it and cuts the padding back
off. That is slow and touches most of the package, which is exactly why it is
here rather than filed among the unit tests.

It is also the only test that justifies the padding rule at all. The
arithmetic in ``test_padding_arithmetic`` checks that the helper computes what
it says; nothing there says the amount is right. This does, in both
directions - enough padding changes nothing, and less than enough visibly
breaks.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import WDM
from WDM_LISAresponse.code.lisa_response.response import LISAresponse
from WDM_LISAresponse.code.utils.utils import (
    apply_wdm_cut,
    apply_zero_padding,
    wdm_cut_info,
    wdm_mismatch,
    wdm_padding_info,
)
from WDM_LISAresponse.code.waveforms import waveforms

from conftest import DT, MAX_LAG_L, NF, Q, T0

pytestmark = pytest.mark.slow

SIGNAL_SAMPLES = 1024
THETA, PHI = 0.7, 1.3


def _run(orbits, max_lag_L, extra_rows):
    """Run the whole pipeline and return the response over the signal span.

    The padding varies between calls, so the grids differ in size; cutting
    each back to the rows the signal occupies is what makes them comparable.
    """
    times, h_plus, h_cross = waveforms.generate_windowed_waveform(
        "monochromatic", t0=T0, dt=DT, N=SIGNAL_SAMPLES,
        model_kwargs=dict(A_plus=1.0, A_cross=0.6, f=0.031, phi_0=0.3),
        window_function="tukey", window_kwargs=dict(alpha=0.1))

    plan = wdm_padding_info(len(times), NF, max_lag_L=max_lag_L,
                            extra_rows_left=extra_rows,
                            extra_rows_right=extra_rows)
    times, h_plus, h_cross = apply_zero_padding(
        times, h_plus, h_cross, plan["pad_left"], plan["pad_right"])

    transform = WDM.WDM.WDM_transform(
        dt=DT, Nf=NF, N=plan["target_length"], q=Q)
    response = LISAresponse(transform, orbits, t0=float(times[0]),
                            max_lag_L=MAX_LAG_L)
    grid = response(THETA, PHI,
                    transform.dwt(jnp.asarray(h_plus)),
                    transform.dwt(jnp.asarray(h_cross)))

    return np.asarray(apply_wdm_cut(grid, wdm_cut_info(plan))), plan


@pytest.fixture(scope="session")
def interior(orbits):
    """Run one padding configuration, and remember the result.

    Every test here compares configurations against each other, so the same
    handful of pipelines are wanted repeatedly - the default one by almost
    all of them. Each run builds a waveform, a transform and a response and
    evaluates it, so recomputing them is most of this file's runtime and
    none of its value.

    The cached arrays are read-only by convention: nothing here mutates a
    result, and a test that did would be reaching past what it is checking.
    """
    computed = {}

    def run(*, max_lag_L=MAX_LAG_L, extra_rows=0):
        key = (max_lag_L, extra_rows)
        if key not in computed:
            computed[key] = _run(orbits, max_lag_L, extra_rows)
        return computed[key]

    return run


# --------------------------------------------------------------- sufficient

@pytest.mark.parametrize("extra_rows", [4, 16, 40])
def test_padding_beyond_the_guard_band_changes_nothing(
        extra_rows, interior):
    """More padding than ceil(L/2) leaves the interior alone.

    If the guard band is doing its job every wrapped read already lands on
    zeros, so adding more zeros cannot change the answer. The grids differ
    in size and the cut brings them back to the same rows.
    """
    minimal, _ = interior()
    generous, _ = interior(extra_rows=extra_rows)

    assert minimal.shape == generous.shape
    assert wdm_mismatch(minimal, generous)["mismatch"] < 1e-9


def test_the_grids_really_were_different_sizes(interior):
    """The comparison above is between genuinely different records.

    Without this the acceptance test could pass on padding that silently
    ignored ``extra_rows``, which is the one way it could be vacuous.
    """
    _, minimal = interior()
    _, generous = interior(extra_rows=40)

    assert generous["Nt"] > minimal["Nt"] + 40
    assert generous["target_length"] > minimal["target_length"]


# ---------------------------------------------------------------- necessary

def test_too_little_padding_visibly_breaks_the_interior(interior):
    """Below ceil(L/2) rows the wrap reaches into the signal.

    Run without a guard band at all, the shift pulls material round from the
    far end of the record and the interior no longer matches. This is the
    half that makes the rule a requirement rather than a habit.

    Asserted against the floor rather than an absolute threshold. How much
    contamination a missing guard band actually costs depends on how much
    power the signal has near the ends - a heavier taper leaves less to wrap
    - and on the epoch, through the delays. What does not depend on either
    is that removing the guard band is orders of magnitude worse than
    adding more of it.
    """
    correct, _ = interior()
    generous, _ = interior(extra_rows=40)
    unguarded, plan = interior(max_lag_L=None)

    floor = wdm_mismatch(correct, generous)["mismatch"]
    rows = min(correct.shape[-2], unguarded.shape[-2])
    unguarded_mismatch = wdm_mismatch(correct[..., :rows, :],
                                      unguarded[..., :rows, :])["mismatch"]

    assert plan["guard_rows"] == 0
    assert unguarded_mismatch > 1e4 * max(floor, 1e-15)


def test_the_error_falls_as_the_guard_band_grows(interior):
    """Contamination decreases monotonically towards the full guard band.

    A single under-padded case could fail for an unrelated reason. Watching
    the error fall as rows are added is what identifies the wrap as the
    cause.
    """
    correct, _ = interior()

    mismatches = []
    for rows in (0, 4, 8):
        # Ask for no guard band, then add rows back by hand.
        trial, _ = interior(max_lag_L=None, extra_rows=rows)
        span = min(correct.shape[-2], trial.shape[-2])
        mismatches.append(wdm_mismatch(correct[..., :span, :],
                                       trial[..., :span, :])["mismatch"])

    assert mismatches[0] > mismatches[1] > mismatches[2]


def test_the_full_guard_band_reaches_the_floor(interior):
    """Padding to exactly ceil(L/2) rows is already enough.

    Asking for the guard band by hand, without ``max_lag_L``, must land on
    the same answer as asking for it by name - so the rule is about the
    number of rows, not about which argument supplied them.
    """
    correct, plan = interior()
    by_hand, _ = interior(max_lag_L=None,
                          extra_rows=plan["rows_left"])

    span = min(correct.shape[-2], by_hand.shape[-2])
    mismatch = wdm_mismatch(correct[..., :span, :],
                            by_hand[..., :span, :])["mismatch"]

    assert mismatch < 1e-9


# ------------------------------------------------------------------ the cut

def test_the_cut_returns_the_rows_the_signal_occupies(interior):
    """Whatever the padding, the interior is the same span of the record."""
    minimal, minimal_plan = interior()
    generous, _ = interior(extra_rows=40)

    assert minimal.shape[-2] == SIGNAL_SAMPLES // NF
    assert generous.shape[-2] == SIGNAL_SAMPLES // NF
    assert minimal_plan["rows_left"] >= minimal_plan["guard_rows"]
