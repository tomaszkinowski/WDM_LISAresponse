import numpy as np
import pytest

pytest.importorskip("WDM")
pytest.importorskip("lisaconstants")

from WDM_LISAresponse.code.lisa_response import TDI_with_one_arm_plan as tdi_mod
from WDM_LISAresponse.code.lisa_response.TDI_with_one_arm_plan import (
    gen2_michelson_td,
    gen2_michelson_td_from_precomputed,
    precompute_gen2_michelson_td_geometry,
    precompute_gen2_michelson_td_waveform,
)


class SimpleOrbits:
    def __init__(self, arm_length=2.5e9):
        self.arm_length = float(arm_length)

    def compute_position(self, times, sc):
        times = np.atleast_1d(times)
        spacecraft = int(sc[0])
        position = np.zeros((times.size, 1, 3), dtype=float)
        position[:, 0, 0] = (spacecraft - 1) * self.arm_length
        return position


def _build_short_inputs(n_samples=64):
    times = np.arange(n_samples, dtype=float) * 0.5
    waveform_plus = np.sin(0.012 * times) + 0.1 * np.cos(0.027 * times)
    waveform_cross = 0.8 * np.cos(0.015 * times + 0.2)
    orbits = SimpleOrbits()
    theta = 0.8
    phi = 1.2
    return times, waveform_plus, waveform_cross, theta, phi, orbits


def test_staged_td_wrapper_matches_manual_precompute():
    times, waveform_plus, waveform_cross, theta, phi, orbits = _build_short_inputs()
    shifts_cache = {}
    one_arm_cache = {}

    direct = gen2_michelson_td(
        times,
        waveform_plus,
        waveform_cross,
        theta,
        phi,
        orbits,
        order=7,
        mode="zero",
        num_int_points=10,
        shifts_cache=shifts_cache,
        one_arm_cache=one_arm_cache,
        chunk_size=8,
    )

    geom = precompute_gen2_michelson_td_geometry(
        times,
        theta,
        phi,
        orbits,
        shifts_cache=shifts_cache,
    )
    wave = precompute_gen2_michelson_td_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        num_int_points=10,
        one_arm_cache=one_arm_cache,
        chunk_size=8,
    )
    staged = gen2_michelson_td_from_precomputed(wave, order=7, mode="zero")

    for direct_arr, staged_arr in zip(direct, staged):
        np.testing.assert_allclose(direct_arr, staged_arr, rtol=1e-12, atol=1e-12)


def test_from_precomputed_does_not_recompute_one_arm_links(monkeypatch):
    times, waveform_plus, waveform_cross, theta, phi, orbits = _build_short_inputs()
    geom = precompute_gen2_michelson_td_geometry(times, theta, phi, orbits)
    wave = precompute_gen2_michelson_td_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        num_int_points=10,
        chunk_size=8,
    )

    call_count = {"count": 0}

    def _counting_one_arm(*args, **kwargs):
        call_count["count"] += 1
        raise AssertionError("one_arm_response_td should not be called during assembly")

    monkeypatch.setattr(tdi_mod, "one_arm_response_td", _counting_one_arm)
    monkeypatch.setattr(tdi_mod, "one_arm_response_td_chunked", _counting_one_arm)

    first = gen2_michelson_td_from_precomputed(wave, order=7, mode="zero")
    second = gen2_michelson_td_from_precomputed(wave, order=7, mode="zero")

    assert call_count["count"] == 0
    for first_arr, second_arr in zip(first, second):
        np.testing.assert_allclose(first_arr, second_arr, rtol=0.0, atol=0.0)
