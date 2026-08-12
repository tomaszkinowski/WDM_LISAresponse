
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("lisaconstants")

from WDM_LISAresponse.code.lisa_response.response_sub_old import (
    all_sky_time_shift_bounds,
    build_directed_link_geometry,
    evaluate_sky_link_geometry,
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


def test_static_link_geometry_is_reused_across_sky_locations():
    times = np.arange(32, dtype=float) * 10.0
    links = ((1, 2), (2, 1))
    geometry = build_directed_link_geometry(times, SimpleOrbits(), links)

    first = evaluate_sky_link_geometry(geometry, 0.7, 1.2)
    second = evaluate_sky_link_geometry(geometry, 1.1, 0.2)

    assert first.time_shift.shape == second.time_shift.shape == (2, 32)
    assert geometry.light_time.shape == (2, 32)
    assert geometry.receiver_position.shape == (2, 32, 3)
    assert geometry.direction.shape == (2, 32, 3)
    assert not np.allclose(first.time_shift, second.time_shift)


def test_global_delay_bounds_cover_sampled_sky_positions():
    times = np.arange(16, dtype=float) * 10.0
    geometry = build_directed_link_geometry(
        times,
        SimpleOrbits(),
        ((1, 2), (2, 1)),
    )
    lower, upper = all_sky_time_shift_bounds(geometry)

    for theta in np.linspace(0.0, np.pi, 7):
        for phi in np.linspace(0.0, 2.0 * np.pi, 9):
            sky = evaluate_sky_link_geometry(geometry, theta, phi)
            assert np.min(sky.time_shift) >= lower - 1e-10
            assert np.max(sky.time_shift) <= upper + 1e-10
