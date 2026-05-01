import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("lisaconstants")

from WDM_LISAresponse.code.lisa_response.TDI import (
    gen2_michelson_wdm,
    gen2_michelson_wdm_from_precomputed,
    gen2_michelson_wdm_from_waveform_precomputed,
    precompute_gen2_michelson_wdm_inputs,
    precompute_gen2_michelson_wdm_geometry,
    precompute_gen2_michelson_wdm_waveform,
)
from WDM_LISAresponse.code.lisa_response.response import precompute_one_arm_wdm_dT_links


class SimpleOrbits:
    def __init__(self, arm_length=2.5e9):
        self.arm_length = float(arm_length)

    def compute_position(self, times, sc):
        times = np.atleast_1d(times)
        spacecraft = int(sc[0])
        x = (spacecraft - 1) * self.arm_length
        position = np.zeros((times.size, 1, 3), dtype=float)
        position[:, 0, 0] = x
        return position


def _build_inputs(n_samples=128, n_f=8, seed=1234):
    times = np.arange(n_samples, dtype=float) * 0.5
    waveform_plus = np.sin(0.012 * times) + 0.1 * np.cos(0.027 * times)
    waveform_cross = 0.8 * np.cos(0.015 * times + 0.2)
    rng = np.random.default_rng(seed)
    waveform_plus = waveform_plus + 0.01 * rng.normal(size=times.size)
    waveform_cross = waveform_cross + 0.01 * rng.normal(size=times.size)
    return times, waveform_plus, waveform_cross


def _compare_triplet(test_triplet, ref_triplet, atol, rtol):
    for test_arr, ref_arr in zip(test_triplet, ref_triplet):
        assert np.allclose(test_arr, ref_arr, atol=atol, rtol=rtol)


def _exact_shift_kwargs(backend="lagfirst_chunked", batch_chunk=None):
    out = {
        "assembly_backend": backend,
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "tl_tp_mode": "exact",
    }
    if backend == "lagfirst_chunked_lagblock":
        out["lag_block_size"] = 2
    if batch_chunk is not None:
        out["batch_chunk"] = int(batch_chunk)
    return out


def _build_staged_wave_precomp(seed=1234, backend="lagfirst_chunked_lagblock", batch_chunk=2):
    times, waveform_plus, waveform_cross = _build_inputs(seed=seed)
    orbits = SimpleOrbits()
    theta = 0.8
    phi = 1.2
    n_f = 8
    kwargs = _exact_shift_kwargs(backend, batch_chunk=batch_chunk)

    geom = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=n_f,
        L_trunc=2,
        q=4,
        use_jax=True,
        variable_shift_kwargs=kwargs,
    )
    wave = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=kwargs,
    )
    return wave, kwargs, n_f


def test_backward_compatible_wrapper_matches_direct_call():
    times, waveform_plus, waveform_cross = _build_inputs()
    orbits = SimpleOrbits()
    theta = 1.0
    phi = 0.4
    n_f = 8
    variable_shift_kwargs = _exact_shift_kwargs("lagfirst_chunked")

    direct = gen2_michelson_wdm(
        times,
        waveform_plus,
        waveform_cross,
        theta,
        phi,
        orbits,
        n_f,
        L_trunc=2,
        variable_shift_kwargs=variable_shift_kwargs,
        use_jax=True,
        q=4,
    )

    precomp = precompute_gen2_michelson_wdm_inputs(
        times,
        waveform_plus,
        waveform_cross,
        theta,
        phi,
        orbits,
        n_f,
        L_trunc=2,
        variable_shift_kwargs=variable_shift_kwargs,
        use_jax=True,
        q=4,
    )
    staged = gen2_michelson_wdm_from_precomputed(
        precomp,
        variable_shift_kwargs=variable_shift_kwargs,
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )

    _compare_triplet(staged, direct, atol=1e-12, rtol=1e-12)


def test_new_split_matches_old_staged_and_direct():
    times, waveform_plus, waveform_cross = _build_inputs(seed=2222)
    orbits = SimpleOrbits()
    theta = 0.6
    phi = 1.1
    n_f = 8
    kwargs = _exact_shift_kwargs("lagfirst_chunked_lagblock", batch_chunk=2)

    direct = gen2_michelson_wdm(
        times,
        waveform_plus,
        waveform_cross,
        theta,
        phi,
        orbits,
        n_f,
        L_trunc=2,
        variable_shift_kwargs=kwargs,
        use_jax=True,
        q=4,
    )

    old_precomp = precompute_gen2_michelson_wdm_inputs(
        times,
        waveform_plus,
        waveform_cross,
        theta,
        phi,
        orbits,
        n_f,
        L_trunc=2,
        variable_shift_kwargs=kwargs,
        use_jax=True,
        q=4,
    )
    old_staged = gen2_michelson_wdm_from_precomputed(
        old_precomp,
        variable_shift_kwargs=kwargs,
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )

    geom = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=n_f,
        L_trunc=2,
        q=4,
        use_jax=True,
        variable_shift_kwargs=kwargs,
    )
    wave = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=kwargs,
    )
    new_staged = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs=kwargs,
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )

    _compare_triplet(old_staged, direct, atol=1e-12, rtol=1e-12)
    _compare_triplet(new_staged, direct, atol=1e-12, rtol=1e-12)
    _compare_triplet(new_staged, old_staged, atol=1e-12, rtol=1e-12)


def test_geometry_reuse_across_waveforms_and_no_waveform_payload_in_geom():
    times, waveform_plus_a, waveform_cross_a = _build_inputs(seed=3333)
    _, waveform_plus_b, waveform_cross_b = _build_inputs(seed=4444)
    orbits = SimpleOrbits()
    theta = 0.8
    phi = 1.2
    n_f = 8
    kwargs = _exact_shift_kwargs("lagfirst_chunked")

    geom = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=n_f,
        L_trunc=2,
        q=4,
        use_jax=True,
        variable_shift_kwargs=kwargs,
    )

    assert "link_w" not in geom
    assert "all_jobs" not in geom
    assert "x_jobs" not in geom
    assert "y_jobs" not in geom
    assert "z_jobs" not in geom

    wave_a = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus_a,
        waveform_cross_a,
        variable_shift_kwargs=kwargs,
    )
    wave_b = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus_b,
        waveform_cross_b,
        variable_shift_kwargs=kwargs,
    )

    # Geometry dictionaries are reused by reference.
    assert wave_a["shifts_X"] is geom["shifts_X"]
    assert wave_a["shifts_Y"] is geom["shifts_Y"]
    assert wave_a["shifts_Z"] is geom["shifts_Z"]
    assert wave_b["shifts_X"] is geom["shifts_X"]
    assert wave_b["shifts_Y"] is geom["shifts_Y"]
    assert wave_b["shifts_Z"] is geom["shifts_Z"]

    out_a = gen2_michelson_wdm_from_waveform_precomputed(
        wave_a,
        variable_shift_kwargs={"batch_chunk": 2},
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )
    out_b = gen2_michelson_wdm_from_waveform_precomputed(
        wave_b,
        variable_shift_kwargs={"batch_chunk": 2},
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )

    # Different waveforms should produce different outputs while running correctly.
    assert not np.allclose(out_a[0], out_b[0], atol=1e-14, rtol=1e-14)
    assert not np.allclose(out_a[1], out_b[1], atol=1e-14, rtol=1e-14)
    assert not np.allclose(out_a[2], out_b[2], atol=1e-14, rtol=1e-14)


def test_batch_chunk_independence_from_same_wave_precompute():
    times, waveform_plus, waveform_cross = _build_inputs(seed=5678)
    orbits = SimpleOrbits()
    theta = 0.8
    phi = 1.2
    n_f = 8

    geom = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=n_f,
        L_trunc=2,
        q=4,
        use_jax=True,
        variable_shift_kwargs=_exact_shift_kwargs("lagfirst_chunked_lagblock", batch_chunk=1),
    )
    wave = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=_exact_shift_kwargs("lagfirst_chunked_lagblock", batch_chunk=1),
    )

    link_w_id = id(wave["link_w"])

    output_batch_1 = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs={"batch_chunk": 1},
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )
    output_batch_2 = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs={"batch_chunk": 2},
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )
    output_batch_4 = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs={"batch_chunk": 4},
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )

    assert id(wave["link_w"]) == link_w_id
    _compare_triplet(output_batch_1, output_batch_2, atol=1e-12, rtol=1e-12)
    _compare_triplet(output_batch_1, output_batch_4, atol=1e-12, rtol=1e-12)
    assert "batch_chunk" not in wave["one_arm_shift_kwargs"]
    assert wave["tdi_batch_shift_kwargs"]["batch_chunk"] == 1


def test_tdi_assembly_accumulate_matches_batch_map_and_default():
    wave, kwargs, n_f = _build_staged_wave_precomp(seed=7777)

    default_out = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs=kwargs,
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )
    batch_map_out = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs=kwargs,
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
        tdi_assembly_mode="batch_map",
    )
    accumulate_1 = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs=kwargs,
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
        tdi_assembly_mode="accumulate",
        accumulator_chunk=1,
    )
    accumulate_4 = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs=kwargs,
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
        tdi_assembly_mode="accumulate",
        accumulator_chunk=4,
    )

    _compare_triplet(default_out, batch_map_out, atol=1e-12, rtol=1e-12)
    _compare_triplet(accumulate_1, batch_map_out, atol=1e-12, rtol=1e-12)
    _compare_triplet(accumulate_4, batch_map_out, atol=1e-12, rtol=1e-12)
    _compare_triplet(accumulate_1, accumulate_4, atol=1e-12, rtol=1e-12)


def test_invalid_tdi_assembly_mode_raises():
    wave, kwargs, n_f = _build_staged_wave_precomp(seed=8888)

    with pytest.raises(ValueError, match="Unknown tdi_assembly_mode"):
        gen2_michelson_wdm_from_waveform_precomputed(
            wave,
            variable_shift_kwargs=kwargs,
            L_trunc=2,
            use_jax=True,
            Nf=n_f,
            tdi_assembly_mode="not-a-mode",
        )


def test_one_arm_precompute_cache_ignores_batch_chunk():
    times, waveform_plus, waveform_cross = _build_inputs(seed=9012)
    orbits = SimpleOrbits()
    theta = 1.1
    phi = 0.2
    n_f = 8
    cache = {}

    first = precompute_one_arm_wdm_dT_links(
        times,
        orbits,
        theta,
        phi,
        waveform_plus,
        waveform_cross,
        Nf=n_f,
        L_trunc=2,
        cache=cache,
        variable_shift_kwargs={
            "assembly_backend": "lagfirst_chunked_lagblock",
            "assembly_precision": "complex128",
            "row_chunk_size": 8,
            "lag_block_size": 2,
            "batch_chunk": 2,
            "tl_tp_mode": "exact",
        },
        use_jax=True,
        q=4,
    )
    second = precompute_one_arm_wdm_dT_links(
        times,
        orbits,
        theta,
        phi,
        waveform_plus,
        waveform_cross,
        Nf=n_f,
        L_trunc=2,
        cache=cache,
        variable_shift_kwargs={
            "assembly_backend": "lagfirst_chunked_lagblock",
            "assembly_precision": "complex128",
            "row_chunk_size": 8,
            "lag_block_size": 2,
            "batch_chunk": 9,
            "tl_tp_mode": "exact",
        },
        use_jax=True,
        q=4,
    )

    assert len(cache) == 1
    assert first[0] is second[0]
    assert set(first[1]) == set(second[1])
    for key in first[1]:
        assert np.allclose(first[1][key], second[1][key], atol=1e-12, rtol=1e-12)


def test_lagblock_backend_works_through_new_split_api():
    times, waveform_plus, waveform_cross = _build_inputs(seed=2468)
    orbits = SimpleOrbits()
    theta = 0.9
    phi = 0.7
    n_f = 8

    lagblock_kwargs = {
        "assembly_backend": "lagfirst_chunked_lagblock",
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "lag_block_size": 2,
        "tl_tp_mode": "exact",
    }
    baseline_kwargs = {
        "assembly_backend": "lagfirst_chunked",
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "tl_tp_mode": "exact",
    }

    geom_lagblock = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=n_f,
        L_trunc=2,
        q=4,
        use_jax=True,
        variable_shift_kwargs=lagblock_kwargs,
    )
    wave_lagblock = precompute_gen2_michelson_wdm_waveform(
        geom_lagblock,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=lagblock_kwargs,
    )
    staged_lagblock = gen2_michelson_wdm_from_waveform_precomputed(
        wave_lagblock,
        variable_shift_kwargs={"batch_chunk": 2},
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )
    geom_baseline = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=n_f,
        L_trunc=2,
        q=4,
        use_jax=True,
        variable_shift_kwargs=baseline_kwargs,
    )
    wave_baseline = precompute_gen2_michelson_wdm_waveform(
        geom_baseline,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=baseline_kwargs,
    )
    staged_baseline = gen2_michelson_wdm_from_waveform_precomputed(
        wave_baseline,
        variable_shift_kwargs={"batch_chunk": 2},
        L_trunc=2,
        use_jax=True,
        Nf=n_f,
    )

    _compare_triplet(staged_lagblock, staged_baseline, atol=1e-12, rtol=1e-12)


def test_geometry_stage_calls_shift_precompute_but_waveform_stage_does_not(monkeypatch):
    times, waveform_plus, waveform_cross = _build_inputs(seed=7777)
    orbits = SimpleOrbits()
    theta = 1.0
    phi = 0.5
    n_f = 8

    import WDM_LISAresponse.code.lisa_response.TDI as tdi_module

    call_counter = {"count": 0}
    real_fn = tdi_module.precompute_gen2_michelson_shifts

    def wrapped(*args, **kwargs):
        call_counter["count"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(tdi_module, "precompute_gen2_michelson_shifts", wrapped)

    geom = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=n_f,
        L_trunc=2,
        q=4,
        use_jax=True,
        variable_shift_kwargs=_exact_shift_kwargs("lagfirst_chunked"),
    )
    assert call_counter["count"] == 1

    _ = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=_exact_shift_kwargs("lagfirst_chunked"),
    )
    assert call_counter["count"] == 1


def test_waveform_stage_reuses_geometry_wdm_and_skips_wdm_context_rebuild(monkeypatch):
    times, waveform_plus, waveform_cross = _build_inputs(seed=8888)
    orbits = SimpleOrbits()
    theta = 0.9
    phi = 0.1
    n_f = 8

    geom = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=n_f,
        L_trunc=2,
        q=4,
        use_jax=True,
        variable_shift_kwargs=_exact_shift_kwargs("lagfirst_chunked_lagblock", batch_chunk=2),
    )

    import WDM_LISAresponse.code.lisa_response.response as response_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_build_wdm_dT_context should not be called during A1 waveform stage")

    monkeypatch.setattr(response_module, "_build_wdm_dT_context", fail_if_called)

    wave = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=_exact_shift_kwargs("lagfirst_chunked_lagblock", batch_chunk=2),
    )

    assert wave["wdm"] is geom["wdm"]
    assert wave["waveform_context_build_s"] >= 0.0
    assert wave["six_link_compute_s"] >= 0.0
    assert wave["job_build_s"] >= 0.0
