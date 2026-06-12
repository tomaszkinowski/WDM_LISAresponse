import numpy as np
import pytest

pytest.importorskip("WDM")
pytest.importorskip("lisaconstants")

from WDM_LISAresponse.code.lisa_response.response import (
    one_arm_response_td,
    one_arm_response_td_chunked,
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


def test_one_arm_response_td_chunked_matches_full():
    times = np.linspace(0.0, 9_990.0, 64)
    td_waveform_plus = np.sin(2.0 * np.pi * times / times[-1])
    td_waveform_cross = np.cos(2.0 * np.pi * times / times[-1])

    orbits = SimpleOrbits()

    y_full = one_arm_response_td(
        times,
        2,
        1,
        orbits,
        theta=0.7,
        phi=1.2,
        td_waveform_plus=td_waveform_plus,
        td_waveform_cross=td_waveform_cross,
        num_int_points=10,
    )
    y_chunked = one_arm_response_td_chunked(
        times,
        2,
        1,
        orbits,
        theta=0.7,
        phi=1.2,
        td_waveform_plus=td_waveform_plus,
        td_waveform_cross=td_waveform_cross,
        num_int_points=10,
        chunk_size=7,
    )

    abs_diff = np.max(np.abs(y_full - y_chunked))
    rel_diff = np.linalg.norm(y_full - y_chunked) / np.linalg.norm(y_full)

    print(abs_diff)
    print(rel_diff)

    assert y_full.shape == y_chunked.shape
    assert abs_diff <= 1e-12
    assert rel_diff <= 1e-12
