"""Every paper figure in one run, each script in its own process, with its wall time
and peak memory.

    python make_figures.py                  # every figure
    python make_figures.py --only 4,9,10    # some of them, by figure number
    python make_figures.py --parts 4        # the sweeps of figures 9, 10 and 14 in
                                            # 4 parallel processes each, where missing
    python make_figures.py --numbers        # also the number scripts (numbers_*.py)
    python make_figures.py --log FILE       # also write the table, and the calibration
                                            # estimate_runtime.py compares against

Each script computes what is not cached and draws from the cache otherwise, so a
second run only redraws. Any script can also be run on its own. With --parts, each
part holds its own copy of the full-year signals: allow ~5 GiB per part. JAX runs on
the CPU, as for the paper, unless JAX_PLATFORMS is set.
"""

import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable

# (figure number, script, extra arguments), in an order that computes each cache
# once: figure 4 draws from the caches of figures 9 and 14.
FIGURES = [
    (2, "fig2_emri_tf_map.py", []),
    (3, "fig3_validity_regions.py", []),
    (5, "fig5_meyer_window.py", []),
    (6, "fig6_wdm_basis.py", []),
    (7, "fig7_time_delay_filters.py", []),
    (8, "fig8_emri_polarisations.py", []),
    (14, "fig14_mismatch_vs_dT_zeroth_order.py", []),
    (9, "fig9_mismatch_vs_dT.py", []),
    (4, "fig4_emri_mismatch_vs_dT.py", []),
    (10, "fig10_mismatch_map.py", ["--kind", "zeroth_order"]),
    (10, "fig10_mismatch_map.py", ["--kind", "first_order"]),
    (11, "fig11_lag_truncation.py", []),
    (12, "fig12_emri_link_comparison.py", []),
    (13, "fig13_emri_tdi_comparison.py", []),
    (15, "fig15_off_bin.py", []),
    (16, "fig16_lag_truncation_zeroth_order.py", []),
]
# The CPU runtimes; the GPU ones (README) need a GPU and its builds.
NUMBERS = [
    ("numbers_emri_power.py", []),
    ("numbers_interpolation.py", []),
    ("numbers_runtime.py", ["zeroth"]),
    ("numbers_runtime.py", ["first"]),
    ("numbers_runtime.py", ["first", "--float32"]),
    ("numbers_runtime.py", ["flr"]),
    ("numbers_runtime.py", ["gwr"]),
]

# Runs a script as __main__ and leaves its peak memory (bytes) in MAKE_FIGURES_PEAK.
WRAPPER = r"""
import os, resource, runpy, sys
script = sys.argv[1]
sys.argv = sys.argv[1:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    with open(os.environ["MAKE_FIGURES_PEAK"], "w") as f:
        f.write(str(peak))
"""


def sweep_caches(script, args):
    """The caches a split run would compute, for the scripts that take --part."""
    if script.startswith("fig14"):
        import fig14_mismatch_vs_dT_zeroth_order as fig14
        return [fig14.CACHE, fig14.BOUNDS_CACHE]
    if script.startswith("fig9_"):
        import fig9_mismatch_vs_dT as fig9
        return [fig9.CACHE, fig9.BOUNDS_CACHE]
    if script.startswith("fig10"):
        import fig10_mismatch_map as fig10
        return [fig10.cache_path(args[args.index("--kind") + 1])]
    return None


def start(script, args, peak_file):
    # an absolute path, so the script's own __file__ (and every path built on it) is
    env = dict(os.environ, MAKE_FIGURES_PEAK=peak_file)
    env.setdefault("JAX_PLATFORMS", "cpu")
    return subprocess.Popen([PY, "-c", WRAPPER, str((HERE / script).resolve()), *args],
                            cwd=HERE, env=env)


def run(script, args, parts):
    """Run one script (as `parts` parallel parts and a merge, where it can split and
    its caches are missing); its wall time, peak memory in GiB and exit status."""
    caches = sweep_caches(script, args)
    split = parts > 1 and caches is not None and not all(p.exists() for p in caches)
    steps = ([[[*args, "--part", f"{k}/{parts}"] for k in range(parts)],
              [[*args, "--merge", str(parts)]]] if split else [[args]])
    tick, peak, status = time.perf_counter(), 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for step in steps:
            files = [str(Path(tmp) / f"peak{i}") for i in range(len(step))]
            procs = [start(script, a, f) for a, f in zip(step, files)]
            status = max(p.wait() for p in procs)
            for f in files:
                if Path(f).exists():
                    peak = max(peak, int(Path(f).read_text()))
            if status:
                break
    return time.perf_counter() - tick, peak / 2**30, status


def machine():
    """The hardware lines of the log: CPU, cores and memory."""
    from numbers_runtime import code_version, cpu_model

    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return [f"date        {datetime.now().isoformat(timespec='seconds')}",
            f"cpu         {cpu_model()}, {os.cpu_count()} logical cores",
            f"memory      {memory:.1f} GiB",
            f"code        WDM_LISAresponse {code_version()}"]


if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = {int(n) for n in sys.argv[sys.argv.index("--only") + 1].split(",")}
    parts = int(sys.argv[sys.argv.index("--parts") + 1]) if "--parts" in sys.argv else 1
    jobs = [(script, args) for number, script, args in FIGURES
            if only is None or number in only]
    if "--numbers" in sys.argv:
        jobs += NUMBERS

    rows = []
    for script, args in jobs:
        name = " ".join([script, *args])
        print(f"=== {name}", flush=True)
        seconds, peak, status = run(script, args, parts)
        rows.append((name, seconds, peak, status))
        print(f"=== {name}: {seconds:.1f} s, peak {peak:.2f} GiB"
              + (f", FAILED (exit {status})" if status else ""), flush=True)

    width = max(len(name) for name, *_ in rows)
    table = [f"{'script':<{width}} | seconds | peak GiB"]
    table += [f"{name:<{width}} | {seconds:7.1f} | {peak:8.2f}" + (" | FAILED" if status else "")
              for name, seconds, peak, status in rows]
    print("\n" + "\n".join(table))
    if "--log" in sys.argv:
        from estimate_runtime import calibrate

        path = Path(sys.argv[sys.argv.index("--log") + 1])
        lines = ["make_figures.py run", *machine(), f"parts       {parts}", "", *table, "",
                 f"calibration | {calibrate():7.1f}"]
        path.write_text("\n".join(lines) + "\n")
        print(f"wrote {path}")
    sys.exit(max((status for *_, status in rows), default=0))
