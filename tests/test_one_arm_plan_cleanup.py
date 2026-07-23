"""Focused tests for the cleaned one-arm execution interface."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from WDM_LISAresponse.code.lisa_response.one_arm_plan import (
    OneArmWDMPlan,
    _normalise_one_arm_execution_mode,
    _shift_options_to_plan_config,
)


class _FakeShiftPlan:
    resolved_assembly_precision = "complex64"

    def __init__(self):
        self.last_dtype = None

    def apply_device(
        self,
        coefficients,
        *,
        return_profile=False,
        cache_device_plan=True,
    ):
        self.last_dtype = coefficients.dtype
        if return_profile:
            return coefficients, {"total_s": 0.0}
        return coefficients

    def apply_device_parallel_groups(
        self,
        coefficients,
        *,
        group_size,
        max_workers,
        return_profile=False,
        cache_device_plan=True,
    ):
        self.last_dtype = coefficients.dtype
        if return_profile:
            return coefficients, {"total_s": 0.0}
        return coefficients

    def clear_device_cache(self):
        pass


def _fake_plan():
    Nt, Nf = 4, 3
    shift_plan = _FakeShiftPlan()
    plan = OneArmWDMPlan(
        wdm=SimpleNamespace(Nt=Nt, Nf=Nf, dF=1.0),
        links=((2, 1), (3, 2)),
        theta=0.5,
        phi=1.0,
        amplitude_plus=np.ones((2, Nt, Nf), dtype=np.float32),
        amplitude_cross=np.full((2, Nt, Nf), 0.5, dtype=np.float32),
        time_shift=np.zeros((2, Nt), dtype=np.float32),
        shift_plan=shift_plan,
    )
    return plan, shift_plan


def test_real_waveform_pair_remains_real_on_device():
    plan, shift_plan = _fake_plan()
    plus = np.ones((4, 3), dtype=np.float64)
    cross = np.ones((4, 3), dtype=np.float64)

    result = plan.apply_device(plus, cross)

    assert shift_plan.last_dtype == jnp.float32
    assert set(result) == set(plan.links)
    assert all(value.dtype == jnp.float32 for value in result.values())


def test_removed_fused_mode_is_rejected_explicitly():
    with pytest.raises(ValueError, match="fused_low_memory"):
        _normalise_one_arm_execution_mode("fused_low_memory")


def test_legacy_production_alias_maps_to_clean_backend():
    config, _ = _shift_options_to_plan_config(
        L_trunc=3,
        variable_shift_kwargs={
            "assembly_backend": "lagfirst_chunked_lagblock",
            "tl_tp_mode": "exact",
        },
        use_jax=True,
    )
    assert config.assembly_backend == "production"
    assert config.lag_truncation == 3


def test_removed_jobblock_option_is_rejected():
    with pytest.raises(TypeError, match="job_block_size"):
        _shift_options_to_plan_config(
            L_trunc=3,
            variable_shift_kwargs={"job_block_size": 2},
            use_jax=True,
        )
