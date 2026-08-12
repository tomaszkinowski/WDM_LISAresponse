"""Small, backend-agnostic helpers for dense WDM runtime notebooks.

The helpers deliberately contain no response construction.  A benchmark
notebook supplies a zero-argument callable, while this module handles JAX
synchronisation, robust timing summaries, numerical comparisons, memory
snapshots and provenance.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - optional on benchmark hosts
    psutil = None


def block_result(value: Any) -> Any:
    """Synchronise all JAX leaves before a timer is stopped."""
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def timed_call(function: Callable[[], Any]) -> tuple[Any, float]:
    """Run and synchronise one call, returning ``(value, elapsed_seconds)``."""
    started = time.perf_counter()
    value = function()
    block_result(value)
    return value, float(time.perf_counter() - started)


def timing_summary(samples: list[float] | np.ndarray) -> dict[str, Any]:
    """Return robust summary statistics for execution-time samples."""
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("At least one timing sample is required.")
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return {
        "median_s": float(np.median(values)),
        "q25_s": float(q25),
        "q75_s": float(q75),
        "iqr_s": float(q75 - q25),
        "minimum_s": float(np.min(values)),
        "maximum_s": float(np.max(values)),
        "all_s": [float(value) for value in values],
    }


def compare_arrays(
    candidate: Any,
    reference: Any,
    *,
    m_start: int = 0,
) -> dict[str, float]:
    """Return max-absolute and relative-L2 differences on the JAX device."""
    candidate_array = jnp.asarray(candidate)[..., m_start:]
    reference_array = jnp.asarray(reference)[..., m_start:]
    difference = candidate_array - reference_array

    difference_power = jnp.sum(jnp.abs(difference) ** 2)
    reference_power = jnp.sum(jnp.abs(reference_array) ** 2)
    max_abs = jnp.max(jnp.abs(difference))
    rel_l2 = jnp.where(
        reference_power > 0,
        jnp.sqrt(difference_power / reference_power),
        jnp.sqrt(difference_power),
    )
    block_result((max_abs, rel_l2))
    return {"max_abs": float(max_abs), "rel_l2": float(rel_l2)}


def compare_link_maps(
    candidate: Mapping[Any, Any],
    reference: Mapping[Any, Any],
    *,
    m_start: int = 0,
) -> dict[str, Any]:
    """Compare two directed-link dictionaries."""
    if set(candidate) != set(reference):
        raise ValueError("Directed-link dictionaries contain different keys.")

    per_link = {
        str(link): compare_arrays(
            candidate[link],
            reference[link],
            m_start=m_start,
        )
        for link in reference
    }
    return {
        "per_link": per_link,
        "max_abs": max(item["max_abs"] for item in per_link.values()),
        "rel_l2": max(item["rel_l2"] for item in per_link.values()),
    }


def current_rss_bytes() -> int | None:
    """Return current resident memory, including native/JAX allocations."""
    if psutil is None:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def peak_rss_bytes() -> int:
    """Return the process high-water RSS in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def memory_snapshot() -> dict[str, float | int | None]:
    """Return current and cumulative peak memory."""
    rss = current_rss_bytes()
    peak = peak_rss_bytes()
    return {
        "rss_bytes": rss,
        "peak_rss_bytes": peak,
        "rss_gib": rss / 1024**3 if rss is not None else None,
        "peak_rss_gib": peak / 1024**3,
    }


def benchmark_candidate(
    name: str,
    function: Callable[[], Any],
    *,
    reference: Mapping[Any, Any] | None = None,
    warmups: int = 1,
    repeats: int = 5,
    m_start: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Benchmark one zero-argument callable and optionally compare link maps.

    ``first_call_s`` is recorded separately.  It includes compilation only when
    the required executable is not already present in JAX's process cache.
    Steady-state statistics use subsequent calls only.
    """
    if warmups < 0:
        raise ValueError("warmups must be non-negative.")
    if repeats < 1:
        raise ValueError("repeats must be at least one.")

    gc.collect()
    memory_before = memory_snapshot()
    latest, first_call_s = timed_call(function)

    for _ in range(int(warmups)):
        latest = function()
        block_result(latest)

    samples = []
    for _ in range(int(repeats)):
        latest, elapsed = timed_call(function)
        samples.append(elapsed)

    comparison = (
        compare_link_maps(latest, reference, m_start=m_start)
        if reference is not None
        else {"per_link": {}, "max_abs": 0.0, "rel_l2": 0.0}
    )
    memory_after = memory_snapshot()

    row = {
        "name": str(name),
        "first_call_s": float(first_call_s),
        **timing_summary(samples),
        "max_abs": float(comparison["max_abs"]),
        "rel_l2": float(comparison["rel_l2"]),
        "per_link": comparison["per_link"],
        "rss_before_gib": memory_before["rss_gib"],
        "rss_after_gib": memory_after["rss_gib"],
        "process_peak_rss_gib": memory_after["peak_rss_gib"],
    }
    if metadata:
        row.update(dict(metadata))
    return latest, row


def numerical_gate(
    row: Mapping[str, Any],
    *,
    rel_l2_tolerance: float,
    max_abs_tolerance: float | None = None,
) -> bool:
    """Return whether one candidate passes the configured numerical gate."""
    rel_l2 = float(row["rel_l2"])
    if not np.isfinite(rel_l2) or rel_l2 > rel_l2_tolerance:
        return False
    if max_abs_tolerance is None:
        return True
    max_abs = float(row["max_abs"])
    return bool(np.isfinite(max_abs) and max_abs <= max_abs_tolerance)


def fastest_passing(
    rows: list[Mapping[str, Any]],
    *,
    rel_l2_tolerance: float,
    max_abs_tolerance: float | None = None,
) -> Mapping[str, Any]:
    """Return the fastest candidate satisfying the numerical gate."""
    passing = [
        row
        for row in rows
        if numerical_gate(
            row,
            rel_l2_tolerance=rel_l2_tolerance,
            max_abs_tolerance=max_abs_tolerance,
        )
    ]
    if not passing:
        raise RuntimeError("No benchmark candidate passed the numerical gate.")
    return min(passing, key=lambda row: float(row["median_s"]))


def git_state(package_file: str | Path) -> dict[str, Any]:
    """Return Git root, commit and dirty state for an imported package."""
    package_path = Path(package_file).resolve()
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(package_path.parent),
            "rev-parse",
            "--show-toplevel",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"root": None, "commit": None, "dirty": None}

    root = Path(completed.stdout.strip()).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"root": str(root), "commit": commit, "dirty": dirty}


def system_metadata() -> dict[str, Any]:
    """Return enough host metadata to interpret a timing result."""
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "physical_cores": (
            psutil.cpu_count(logical=False) if psutil is not None else None
        ),
        "logical_cores": (
            psutil.cpu_count(logical=True) if psutil is not None else None
        ),
    }


def json_ready(value: Any) -> Any:
    """Recursively convert NumPy/JAX values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def save_json(path: str | Path, payload: Any) -> Path:
    """Write a readable JSON result."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_ready(payload), indent=2),
        encoding="utf-8",
    )
    return destination