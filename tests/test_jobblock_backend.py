"""Tests for the job-block vectorized WDM time-shift assembly backend."""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("lisaconstants")

from WDM_LISAresponse.code.lisa_response.TDI import (
    gen2_michelson_wdm_from_waveform_precomputed,
    precompute_gen2_michelson_wdm_geometry,
    precompute_gen2_michelson_wdm_waveform,
)


class SimpleOrbits:
    """Minimal orbit provider for testing."""

    def __init__(self, arm_length=2.5e9):
        self.arm_length = float(arm_length)

    def compute_position(self, times, sc):
        times = np.atleast_1d(times)
        spacecraft = int(sc[0])
        x = (spacecraft - 1) * self.arm_length
        position = np.zeros((times.size, 1, 3), dtype=float)
        position[:, 0, 0] = x
        return position


def _build_small_wave_setup(n_samples=64, n_f=8, seed=999):
    """Build a small test waveform and orbit setup."""
    times = np.arange(n_samples, dtype=float) * 0.5
    waveform_plus = np.sin(0.012 * times) + 0.1 * np.cos(0.027 * times)
    waveform_cross = 0.8 * np.cos(0.015 * times + 0.2)
    rng = np.random.default_rng(seed)
    waveform_plus = waveform_plus + 0.01 * rng.normal(size=times.size)
    waveform_cross = waveform_cross + 0.01 * rng.normal(size=times.size)

    orbits = SimpleOrbits()
    return times, waveform_plus, waveform_cross, orbits



def test_jobblock_backend_tdi_matches_lagblock():
    """Test that jobblock backend produces same results as lagblock in TDI pipeline."""
    times, waveform_plus, waveform_cross, orbits = _build_small_wave_setup(
        n_samples=48, n_f=6, seed=1001
    )
    theta = np.pi / 3
    phi = np.pi / 4

    # Test configuration for lagblock backend
    lagblock_kwargs = {
        "assembly_backend": "lagfirst_chunked_lagblock",
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "lag_block_size": 2,
        "tl_tp_mode": "exact",
    }

    # Precompute geometry and waveform with lagblock
    geom_lagblock = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=6,
        L_trunc=3,
        q=2,
        use_jax=True,
        variable_shift_kwargs=lagblock_kwargs,
    )

    wave_lagblock = precompute_gen2_michelson_wdm_waveform(
        geom_lagblock,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=lagblock_kwargs,
    )

    X_lagblock, Y_lagblock, Z_lagblock = gen2_michelson_wdm_from_waveform_precomputed(
        wave_lagblock,
        variable_shift_kwargs={"batch_chunk": 1},
        L_trunc=3,
        use_jax=True,
        Nf=6,
    )

    # Test configuration for jobblock backend
    jobblock_kwargs = {
        "assembly_backend": "lagfirst_chunked_lagblock_jobblock",
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "lag_block_size": 2,
        "tl_tp_mode": "exact",
    }

    # Precompute geometry and waveform with jobblock
    geom_jobblock = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=6,
        L_trunc=3,
        q=2,
        use_jax=True,
        variable_shift_kwargs=jobblock_kwargs,
    )

    wave_jobblock = precompute_gen2_michelson_wdm_waveform(
        geom_jobblock,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=jobblock_kwargs,
    )

    X_jobblock, Y_jobblock, Z_jobblock = gen2_michelson_wdm_from_waveform_precomputed(
        wave_jobblock,
        variable_shift_kwargs={"batch_chunk": 1, "job_block_size": 2},
        L_trunc=3,
        use_jax=True,
        Nf=6,
    )

    # Compare results
    np.testing.assert_allclose(X_jobblock, X_lagblock, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(Y_jobblock, Y_lagblock, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(Z_jobblock, Z_lagblock, rtol=1e-4, atol=1e-7)


def test_jobblock_backend_with_different_sizes():
    """Test jobblock backend with different job_block_size values."""
    times, waveform_plus, waveform_cross, orbits = _build_small_wave_setup(
        n_samples=56, n_f=6, seed=1002
    )
    theta = np.pi / 2
    phi = 0.0

    jobblock_kwargs = {
        "assembly_backend": "lagfirst_chunked_lagblock_jobblock",
        "assembly_precision": "complex64",
        "row_chunk_size": 8,
        "lag_block_size": 2,
        "tl_tp_mode": "exact",
    }

    geom = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=6,
        L_trunc=2,
        q=2,
        use_jax=True,
        variable_shift_kwargs=jobblock_kwargs,
    )

    wave = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=jobblock_kwargs,
    )

    # Compute with different job_block_size values
    results = {}
    for jb_size in [1, 2]:
        X, Y, Z = gen2_michelson_wdm_from_waveform_precomputed(
            wave,
            variable_shift_kwargs={"batch_chunk": 1, "job_block_size": jb_size},
            L_trunc=2,
            use_jax=True,
            Nf=6,
        )
        results[jb_size] = (X, Y, Z)

    # Compare results
    X_1, Y_1, Z_1 = results[1]
    X_2, Y_2, Z_2 = results[2]
    np.testing.assert_allclose(X_2, X_1, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(Y_2, Y_1, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(Z_2, Z_1, rtol=1e-4, atol=1e-7)


def test_jobblock_backend_with_accumulate_mode():
    """Test jobblock backend in accumulate TDI assembly mode."""
    times, waveform_plus, waveform_cross, orbits = _build_small_wave_setup(
        n_samples=40, n_f=6, seed=1003
    )
    theta = np.pi / 4
    phi = np.pi / 6

    jobblock_kwargs = {
        "assembly_backend": "lagfirst_chunked_lagblock_jobblock",
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "lag_block_size": 2,
        "tl_tp_mode": "exact",
    }

    geom = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=6,
        L_trunc=2,
        q=2,
        use_jax=True,
        variable_shift_kwargs=jobblock_kwargs,
    )

    wave = precompute_gen2_michelson_wdm_waveform(
        geom,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=jobblock_kwargs,
    )

    # Test both batch_map and accumulate modes
    X_batch, Y_batch, Z_batch = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs={"batch_chunk": 2, "job_block_size": 2},
        L_trunc=2,
        use_jax=True,
        Nf=6,
        tdi_assembly_mode="batch_map",
    )

    X_accum, Y_accum, Z_accum = gen2_michelson_wdm_from_waveform_precomputed(
        wave,
        variable_shift_kwargs={"batch_chunk": 2, "job_block_size": 2},
        L_trunc=2,
        use_jax=True,
        Nf=6,
        tdi_assembly_mode="accumulate",
    )

    # Both modes should give same results
    np.testing.assert_allclose(X_accum, X_batch, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Y_accum, Y_batch, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Z_accum, Z_batch, rtol=1e-5, atol=1e-8)


def test_jobblock_default_compatibility():
    """Test that default job_block_size=1 is backward compatible."""
    times, waveform_plus, waveform_cross, orbits = _build_small_wave_setup(
        n_samples=32, n_f=6, seed=1004
    )
    theta = np.pi / 5
    phi = 0.5

    # Lagblock backend with default settings
    lagblock_kwargs = {
        "assembly_backend": "lagfirst_chunked_lagblock",
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "lag_block_size": 2,
        "tl_tp_mode": "exact",
    }

    geom_lag = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=6,
        L_trunc=2,
        q=2,
        variable_shift_kwargs=lagblock_kwargs,
    )

    wave_lag = precompute_gen2_michelson_wdm_waveform(
        geom_lag,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=lagblock_kwargs,
    )

    X_lag, Y_lag, Z_lag = gen2_michelson_wdm_from_waveform_precomputed(
        wave_lag,
        variable_shift_kwargs={"batch_chunk": 1},
        L_trunc=2,
        Nf=6,
    )

    # Jobblock backend with job_block_size=1 (default)
    jobblock_kwargs = {
        "assembly_backend": "lagfirst_chunked_lagblock_jobblock",
        "assembly_precision": "complex128",
        "row_chunk_size": 8,
        "lag_block_size": 2,
        "tl_tp_mode": "exact",
    }

    geom_jb = precompute_gen2_michelson_wdm_geometry(
        times,
        theta,
        phi,
        orbits,
        Nf=6,
        L_trunc=2,
        q=2,
        variable_shift_kwargs=jobblock_kwargs,
    )

    wave_jb = precompute_gen2_michelson_wdm_waveform(
        geom_jb,
        waveform_plus,
        waveform_cross,
        variable_shift_kwargs=jobblock_kwargs,
    )

    # Note: job_block_size=1 means one job per vmap call, same as lagblock
    X_jb, Y_jb, Z_jb = gen2_michelson_wdm_from_waveform_precomputed(
        wave_jb,
        variable_shift_kwargs={"batch_chunk": 1, "job_block_size": 1},
        L_trunc=2,
        Nf=6,
    )

    # Results should match
    np.testing.assert_allclose(X_jb, X_lag, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(Y_jb, Y_lag, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(Z_jb, Z_lag, rtol=1e-4, atol=1e-7)



if __name__ == "__main__":
    # Quick smoke tests
    test_jobblock_backend_tdi_matches_lagblock()
    print("✓ test_jobblock_backend_tdi_matches_lagblock")

    test_jobblock_backend_with_different_sizes()
    print("✓ test_jobblock_backend_with_different_sizes")

    test_jobblock_backend_with_accumulate_mode()
    print("✓ test_jobblock_backend_with_accumulate_mode")

    test_jobblock_default_compatibility()
    print("✓ test_jobblock_default_compatibility")

    print("\nAll jobblock tests passed!")

