"""Focused tests for grouped Gen-2 Michelson plan execution."""

from types import SimpleNamespace

import numpy as np

from WDM_LISAresponse.code.lisa_response.gen2_michelson_plan import (
    Gen2MichelsonWDMPlan,
)


class _FakeGroupedShiftPlan:
    kernel_memory_bytes = 1234

    def __init__(self, shifted):
        self.shifted = np.asarray(shifted)
        self.last_source_shape = None

    def apply_indexed_and_accumulate(
        self,
        source_coefficients,
        source_indices,
        output_indices,
        weights,
        *,
        num_outputs,
        base,
        return_profile=False,
        cache_device_plan=True,
    ):
        source_coefficients = np.asarray(source_coefficients)
        self.last_source_shape = source_coefficients.shape

        outputs = np.asarray(base).copy()
        np.add.at(
            outputs,
            np.asarray(output_indices),
            np.asarray(weights)[:, None, None] * self.shifted,
        )

        if return_profile:
            return outputs, {
                "total_s": 0.0,
                "device_plan_cached": bool(cache_device_plan),
            }
        return outputs


def test_gen2_plan_uses_six_source_grouped_path():
    Nt, Nf = 4, 3
    wdm = SimpleNamespace(Nt=Nt, Nf=Nf)

    links = (
        (2, 1),
        (3, 2),
        (1, 3),
        (3, 1),
        (2, 3),
        (1, 2),
    )
    link_w = {
        link: np.full((Nt, Nf), index + 1.0)
        for index, link in enumerate(links)
    }

    shifted = np.stack(
        [np.full((Nt, Nf), job + 1.0) for job in range(42)],
        axis=0,
    )
    grouped_shift = _FakeGroupedShiftPlan(shifted)

    job_links = tuple(links[job % len(links)] for job in range(42))
    channels = np.repeat(np.arange(3, dtype=np.int8), 14)
    signs = np.tile(
        np.concatenate((np.ones(7), -np.ones(7))),
        3,
    ).astype(np.float32)

    plan = Gen2MichelsonWDMPlan(
        wdm=wdm,
        job_names=tuple(f"job-{job}" for job in range(42)),
        job_links=job_links,
        output_channels=channels,
        signs=signs,
        base_positive_links=((2, 1), (3, 2), (1, 3)),
        base_negative_links=((3, 1), (1, 2), (2, 3)),
        shift_plan=grouped_shift,
        build_seconds=0.25,
    )

    (X, Y, Z), profile = plan.apply(link_w, return_profile=True)

    assert grouped_shift.last_source_shape == (6, Nt, Nf)
    assert profile["tdi_execution_mode"] == "indexed_accumulate"
    assert profile["tdi_plan_build_s"] == 0.25
    assert profile["tdi_kernel_memory_bytes"] == 1234

    expected_base = (
        link_w[(2, 1)] - link_w[(3, 1)],
        link_w[(3, 2)] - link_w[(1, 2)],
        link_w[(1, 3)] - link_w[(2, 3)],
    )
    expected = []
    for channel in range(3):
        start = 14 * channel
        signed_sum = sum(
            signs[start + offset] * shifted[start + offset]
            for offset in range(14)
        )
        expected.append(expected_base[channel] + signed_sum)

    np.testing.assert_allclose(X, expected[0])
    np.testing.assert_allclose(Y, expected[1])
    np.testing.assert_allclose(Z, expected[2])