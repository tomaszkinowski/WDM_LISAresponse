
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("lisaconstants")

from WDM_LISAresponse.code.lisa_response.TDI_with_one_arm_plan import (
    gen2_michelson_wdm_from_static_waveform,
    precompute_gen2_michelson_wdm_static,
    precompute_gen2_michelson_wdm_waveform_runtime,
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


def _inputs():
    times = np.arange(128, dtype=float) * 0.5
    plus = np.sin(0.012 * times) + 0.1 * np.cos(0.027 * times)
    cross = 0.8 * np.cos(0.015 * times + 0.2)
    return times, plus, cross


def _kwargs():
    return {
        "assembly_backend": "production",
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "lag_block_size": 2,
        "tl_tp_mode": "interp",
        "tl_tp_interp_kind": "linear",
        "tl_tp_table_points": 2048,
        "tl_tp_table_pad": 0.02,
        "batch_chunk": 2,
    }


def test_static_pipeline_reuses_orbits_wdm_table_and_tdi_across_sky():
    times, plus, cross = _inputs()
    static = precompute_gen2_michelson_wdm_static(
        times,
        SimpleOrbits(),
        Nf=8,
        L_trunc=2,
        q=4,
        variable_shift_kwargs=_kwargs(),
        use_jax=True,
    )
    waveform = precompute_gen2_michelson_wdm_waveform_runtime(
        static,
        plus,
        cross,
    )

    table_id = id(
        static["one_arm_static_plan"].shift_context.interpolation_table
    )
    tdi_id = id(static["tdi_plan"])
    first = gen2_michelson_wdm_from_static_waveform(
        static,
        waveform,
        0.8,
        1.2,
    )
    second = gen2_michelson_wdm_from_static_waveform(
        static,
        waveform,
        1.0,
        0.4,
    )

    assert id(static["one_arm_static_plan"].shift_context.interpolation_table) == table_id
    assert id(static["tdi_plan"]) == tdi_id
    assert all(array.shape == (static["wdm"].Nt, static["wdm"].Nf) for array in first)
    assert not np.allclose(first[0], second[0])


def test_waveform_stage_contains_no_sky_specific_links():
    times, plus, cross = _inputs()
    static = precompute_gen2_michelson_wdm_static(
        times,
        SimpleOrbits(),
        Nf=8,
        L_trunc=2,
        q=4,
        variable_shift_kwargs=_kwargs(),
        use_jax=True,
    )
    waveform = precompute_gen2_michelson_wdm_waveform_runtime(static, plus, cross)

    assert "link_w" not in waveform
    assert "theta" not in waveform
    assert "phi" not in waveform
    assert waveform["wave_context"]["w_t_plus"].shape == (
        static["wdm"].Nt,
        static["wdm"].Nf,
    )
