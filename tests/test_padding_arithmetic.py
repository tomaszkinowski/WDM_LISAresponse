"""Padding arithmetic: how much to add, and how to take it back off.

Pure integer bookkeeping - these two functions never touch an array. The
end-to-end check that the padding actually does its job lives in
``test_padding_acceptance.py``, which needs orbits and a response.

Three constraints have to hold at once: the WDM tiling, the guard band that
stops the variable time shift wrapping signal round the record, and an even
leading offset so differently padded grids stay comparable.
"""

import numpy as np
import pytest
from scipy.signal.windows import tukey

from WDM_LISAresponse.code.waveforms.waveforms import tukey_taper_samples
from WDM_LISAresponse.code.utils.utils import (
    apply_repadding,
    apply_zero_padding,
    apply_wdm_cut,
    wdm_cut_info,
    wdm_padding_info,
)

NF = 16
LENGTHS = [1024, 1000, 1537, 2, 4096]
LAGS = [1, 2, 25, 30, 63]


# ------------------------------------------------------------- the constraints

@pytest.mark.parametrize("n_samples", LENGTHS)
@pytest.mark.parametrize("max_lag_L", LAGS)
def test_the_tiling_constraint_holds(n_samples, max_lag_L):
    """Total length is a multiple of 2*Nf, so Nt comes out even.

    ``WDM_transform`` rejects anything else, so this is the constraint that
    makes the padded record usable at all.
    """
    info = wdm_padding_info(n_samples, NF, max_lag_L=max_lag_L)

    assert info["target_length"] % (2 * NF) == 0
    assert info["Nt"] % 2 == 0
    assert info["Nt"] == info["target_length"] // NF


@pytest.mark.parametrize("max_lag_L", LAGS)
def test_the_guard_band_is_ceil_half_of_L(max_lag_L):
    """Both ends get ceil(L/2) rows.

    Half of L rather than all of it, because padding *both* ends is what
    lets a wrapped read land in the other end's zeros.
    """
    info = wdm_padding_info(1024, NF, max_lag_L=max_lag_L)

    assert info["guard_rows"] == -(-max_lag_L // 2)
    assert info["rows_left"] >= info["guard_rows"]
    assert info["rows_right"] >= info["guard_rows"]


@pytest.mark.parametrize("n_samples", LENGTHS)
@pytest.mark.parametrize("extra", [0, 1, 2, 3, 7, 40])
def test_the_leading_pad_is_always_an_even_number_of_rows(n_samples, extra):
    """Whatever is asked for, the left offset comes out even.

    Bookkeeping rather than correctness - it keeps differently padded grids
    comparable coefficient by coefficient - and it costs at most one row.
    """
    info = wdm_padding_info(n_samples, NF, max_lag_L=25,
                            extra_rows_left=extra)

    assert info["rows_left"] % 2 == 0


def test_no_guard_band_is_requested_when_no_lag_is_given():
    """Without ``max_lag_L`` only the tiling constraint applies.

    The time-domain path has no wrap to guard against, so it must be
    possible to ask for the tiling alone.
    """
    info = wdm_padding_info(1024, NF)

    assert info["guard_rows"] == 0
    assert info["rows_left"] == 0
    assert info["pad_left"] == 0


# ------------------------------------------------------------------- ordering

@pytest.mark.parametrize("n_samples", LENGTHS)
def test_the_left_pad_is_exactly_what_was_asked_for(n_samples):
    """The tiling remainder lands on the right, never the left.

    That ordering is the whole reason the cut can be integer slicing: the
    signal starts on a row the caller chose. Split the remainder, or put it
    on the left, and the start row moves by an amount nobody picked.
    """
    info = wdm_padding_info(n_samples, NF, max_lag_L=25, extra_rows_left=3)

    expected_rows = 13 + 3
    expected_rows += expected_rows % 2

    assert info["rows_left"] == expected_rows
    assert info["pad_left"] == expected_rows * NF


@pytest.mark.parametrize("n_samples", LENGTHS)
def test_the_three_pieces_add_up(n_samples):
    """Left pad, signal and right pad account for the whole record."""
    info = wdm_padding_info(n_samples, NF, max_lag_L=25)

    assert (info["pad_left"] + info["input_length"] + info["pad_right"]
            == info["target_length"])


def test_padding_is_quantised_to_whole_rows():
    """Both pads are whole numbers of rows, so the signal starts on one."""
    info = wdm_padding_info(1000, NF, max_lag_L=25)

    assert info["pad_left"] % NF == 0
    assert info["rows_right"] == info["pad_right"] // NF


# ------------------------------------------------------------------- the cut

@pytest.mark.parametrize("n_samples", LENGTHS)
@pytest.mark.parametrize("max_lag_L", [1, 25, 63])
def test_the_cut_recovers_the_original_span(n_samples, max_lag_L):
    """Cutting removes exactly the rows that were added.

    Because the padding went on in whole rows, the rows to drop are known
    integers - no sample-to-row conversion, no ceiling.
    """
    info = wdm_padding_info(n_samples, NF, max_lag_L=max_lag_L)
    cut = wdm_cut_info(info)

    kept = cut["time_slice"].stop - cut["time_slice"].start

    assert cut["time_slice"].start == info["rows_left"]
    assert kept == info["Nt"] - info["rows_left"] - info["rows_right"]


def test_extra_rows_trim_further_at_both_ends():
    """``extra_rows`` cuts beyond the padding and taper, one row per row."""
    info = wdm_padding_info(1024, NF, max_lag_L=25)

    plain = wdm_cut_info(info)
    trimmed = wdm_cut_info(info, extra_rows=5)

    assert trimmed["time_slice"].start == plain["time_slice"].start + 5
    assert trimmed["time_slice"].stop == plain["time_slice"].stop - 5


@pytest.mark.parametrize("kwargs, expected", [
    ({}, slice(1, None)),
    ({"drop_m0": True}, slice(1, None)),
    ({"drop_m0": False}, slice(None)),
    ({"drop_freq_edges": True}, slice(1, -1)),
    ({"drop_m0": False, "drop_freq_edges": True}, slice(1, -1)),
])
def test_the_frequency_slice_follows_the_flags(kwargs, expected):
    """m=0 is dropped by default; ``drop_freq_edges`` takes precedence.

    The lowest band is wrong unless the transform was built with
    ``calc_m0=True``, so dropping it is the safe default.
    """
    info = wdm_padding_info(1024, NF, max_lag_L=25)

    assert wdm_cut_info(info, **kwargs)["freq_slice"] == expected


def test_apply_wdm_cut_leaves_leading_axes_alone():
    """The cut applies to the trailing (Nt, Nf) pair, whatever precedes it."""
    info = wdm_padding_info(1024, NF, max_lag_L=25)
    cut = wdm_cut_info(info)
    grid = np.zeros((2, 6, info["Nt"], NF))

    got = apply_wdm_cut(grid, cut)

    assert got.shape[:2] == (2, 6)
    assert got.shape[2] == info["Nt"] - 2 * info["rows_left"]
    assert got.shape[3] == NF - 1


def test_padding_then_cutting_is_a_round_trip():
    """The rows the cut keeps are the rows the signal occupies.

    Padding and cutting are checked together because that is how they are
    used, and separately neither says anything about the other.
    """
    times = np.arange(1024) * 5.0
    signal = np.sin(times)
    info = wdm_padding_info(len(times), NF, max_lag_L=25)

    padded_times, padded, _ = apply_zero_padding(
        times, signal, signal, info["pad_left"], info["pad_right"])
    rows = padded.reshape(info["Nt"], NF)
    cut = wdm_cut_info(info, drop_m0=False)

    assert len(padded_times) == info["target_length"]
    assert np.array_equal(rows[cut["time_slice"]].reshape(-1), signal)


# ------------------------------------------------------------------ the taper

@pytest.mark.parametrize("n_samples", [1024, 1000, 1537, 4096])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_the_kept_rows_hold_no_tapered_sample(n_samples, alpha):
    """Every kept row is window exactly one; the rows either side are not.

    Checked on the window itself, padded as the signal would be, so the test
    does not repeat the arithmetic it is checking - and 1000 and 1537 leave
    a partial last row for it to get wrong.
    """
    info = wdm_padding_info(n_samples, NF, max_lag_L=25,
                            taper_samples=tukey_taper_samples(n_samples, alpha))
    rows = np.pad(tukey(n_samples, alpha),
                  (info["pad_left"], info["pad_right"])).reshape(-1, NF)

    kept = wdm_cut_info(info)["time_slice"]

    assert np.all(rows[kept] == 1.0)
    assert np.any(rows[kept.start - 1] < 1.0)
    assert np.any(rows[kept.stop] < 1.0)


@pytest.mark.parametrize("n_samples", LENGTHS)
def test_without_a_taper_the_cut_is_unchanged(n_samples):
    """No recorded taper, or a zero one, cuts the padding and nothing else."""
    plain = wdm_padding_info(n_samples, NF, max_lag_L=25)
    zero = wdm_padding_info(n_samples, NF, max_lag_L=25, taper_samples=0)

    for info in (plain, zero):
        assert wdm_cut_info(info)["time_slice"] == slice(
            info["rows_left"], info["Nt"] - info["rows_right"])


def test_a_taper_changes_no_padding():
    """The taper is recorded for the cut; the padding is decided without it."""
    plain = wdm_padding_info(1000, NF, max_lag_L=25)
    tapered = wdm_padding_info(1000, NF, max_lag_L=25, taper_samples=100)

    assert {k: v for k, v in tapered.items() if k != "taper_samples"} == \
           {k: v for k, v in plain.items() if k != "taper_samples"}


def test_extra_rows_count_from_the_end_of_the_taper():
    """``extra_rows`` steps in from the first full-amplitude row, not the pad."""
    info = wdm_padding_info(1024, NF, max_lag_L=25, taper_samples=40)

    plain = wdm_cut_info(info)["time_slice"]
    trimmed = wdm_cut_info(info, extra_rows=3)["time_slice"]

    assert trimmed == slice(plain.start + 3, plain.stop - 3)


@pytest.mark.parametrize("taper", [-1, 513])
def test_a_taper_longer_than_half_the_record_raises(taper):
    """Two ramps cannot together outrun the record."""
    with pytest.raises(ValueError, match="taper_samples must lie"):
        wdm_padding_info(1024, NF, taper_samples=taper)


# ------------------------------------------------ moving between two paddings

@pytest.mark.parametrize("pads_from, pads_to", [
    ((40, 70), (40, 70)),      # unchanged
    ((10, 5), (40, 70)),       # both grow
    ((40, 70), (10, 5)),       # both shrink
    ((10, 70), (40, 5)),       # one of each
    ((0, 0), (16, 48)),        # from no padding at all
])
def test_repadding_matches_padding_directly(pads_from, pads_to):
    """Moving a record to new padding is the same as padding it that way.

    The signal ends in zeros, as a tapered one does, so shrinking a pad only
    ever drops zeros and is allowed.
    """
    times = np.arange(512) * 5.0
    signal = np.sin(times) * np.hanning(len(times))
    _, source, _ = apply_zero_padding(times, signal, signal, *pads_from)
    _, direct, _ = apply_zero_padding(times, signal, signal, *pads_to)

    assert np.array_equal(apply_repadding(source, *pads_from, *pads_to), direct)


def test_repadding_keeps_response_that_sits_in_the_padding():
    """Samples outside the signal survive the move when the new pad reaches.

    The case it exists for: a response leaks a few hundred seconds past the
    signal, and that leakage belongs to the record on every grid.
    """
    source = np.zeros(100)
    source[18:83] = 1.0                                    # 2 into each pad

    moved = apply_repadding(source, 20, 20, 5, 35)

    assert moved.shape == (100,)
    assert np.array_equal(np.nonzero(moved)[0], np.arange(3, 68))


def test_repadding_leaves_leading_axes_alone():
    """Links and batch axes pass through; only the last axis moves."""
    source = np.random.default_rng(0).normal(size=(3, 6, 64))
    source[..., :8] = source[..., -8:] = 0.0

    moved = apply_repadding(source, 8, 8, 16, 4)

    assert moved.shape == (3, 6, 68)
    assert np.array_equal(moved[..., 16:64], source[..., 8:56])


def test_repadding_refuses_to_drop_response():
    """A crop that discards non-zero samples would change the record."""
    source = np.ones(64)

    with pytest.raises(ValueError, match="discard non-zero samples"):
        apply_repadding(source, 8, 8, 4, 8)


@pytest.mark.parametrize("pads", [(-1, 0, 0, 0), (0, 0, 0, -1), (40, 30, 0, 0)])
def test_repadding_with_impossible_pads_raises(pads):
    """Negative pads, or pads that leave no signal, are mistakes."""
    with pytest.raises(ValueError, match="non-negative|leaves nothing"):
        apply_repadding(np.zeros(64), *pads)


# ------------------------------------------------------------------- refusal

@pytest.mark.parametrize("kwargs", [
    {"n_samples": 0}, {"n_samples": -1},
])
def test_a_degenerate_length_raises(kwargs):
    """A record of no samples has no padding, it is a mistake."""
    with pytest.raises(ValueError, match="n_samples must be"):
        wdm_padding_info(Nf=NF, **kwargs)


@pytest.mark.parametrize("bad_nf", [0, -4])
def test_a_degenerate_band_count_raises(bad_nf):
    """Nf sets the row size, so it cannot be zero or negative."""
    with pytest.raises(ValueError, match="Nf must be"):
        wdm_padding_info(1024, bad_nf)


def test_a_negative_lag_raises():
    """A negative guard band is meaningless."""
    with pytest.raises(ValueError, match="max_lag_L must be"):
        wdm_padding_info(1024, NF, max_lag_L=-1)


@pytest.mark.parametrize("side", ["extra_rows_left", "extra_rows_right"])
def test_negative_extra_rows_raise(side):
    """Padding cannot be negative on either side."""
    with pytest.raises(ValueError, match="must be non-negative"):
        wdm_padding_info(1024, NF, **{side: -1})


def test_cutting_everything_away_raises():
    """A cut that leaves nothing is a mistake, not an empty result."""
    info = wdm_padding_info(1024, NF, max_lag_L=25)

    with pytest.raises(ValueError, match="Nothing left after cutting"):
        wdm_cut_info(info, extra_rows=info["Nt"])
