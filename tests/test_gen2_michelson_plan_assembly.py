"""Focused tests for the LISA-specific prepared Gen-2 Michelson wrapper."""

from types import SimpleNamespace

import numpy as np

from WDM_LISAresponse.code.lisa_response.gen2_michelson_plan import (
    Gen2MichelsonWDMPlan,
)


class _FakeShiftPlan:
    def __init__(self, shifted):
        self._shifted = np.asarray(shifted)
        self.kernel_memory_bytes = 1234

    def apply(self, coefficients, *, return_profile=False):
        coefficients = np.asarray(coefficients)
        assert coefficients.shape == self._shifted.shape
        if return_profile:
            return self._shifted.copy(), {"total_s": 0.0, "assembly_s": 0.0}
        return self._shifted.copy()


def test_prepared_tdi_plan_assembles_channel_signs_and_base_terms():
    Nt, Nf = 4, 3
    wdm = SimpleNamespace(Nt=Nt, Nf=Nf)

    links = (
        (2, 1), (3, 2), (1, 3), (3, 1), (2, 3), (1, 2)
    )
    link_w = {
        link: np.full((Nt, Nf), index + 1.0)
        for index, link in enumerate(links)
    }

    # Give every shifted job a unique constant value, so signs and channel
    # grouping can be checked independently of the numerical shifter.
    shifted = np.stack(
        [np.full((Nt, Nf), job + 1.0) for job in range(42)],
        axis=0,
    )

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
        shift_plan=_FakeShiftPlan(shifted),
        build_seconds=0.0,
    )

    X, Y, Z = plan.apply(link_w)

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


def test_prepared_tdi_plan_profile_is_forwarded():
    Nt, Nf = 2, 2
    wdm = SimpleNamespace(Nt=Nt, Nf=Nf)
    link = (2, 1)
    link_w = {link: np.ones((Nt, Nf))}
    shifted = np.zeros((42, Nt, Nf))

    plan = Gen2MichelsonWDMPlan(
        wdm=wdm,
        job_names=tuple(str(i) for i in range(42)),
        job_links=(link,) * 42,
        output_channels=np.repeat(np.arange(3, dtype=np.int8), 14),
        signs=np.ones(42, dtype=np.float32),
        base_positive_links=(link, link, link),
        base_negative_links=(link, link, link),
        shift_plan=_FakeShiftPlan(shifted),
        build_seconds=0.25,
    )

    (_, _, _), profile = plan.apply(link_w, return_profile=True)
    assert profile["tdi_plan_build_s"] == 0.25
    assert profile["tdi_kernel_memory_bytes"] == 1234