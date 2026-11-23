"""Runtime of the six one-arm links at the paper's parameters, on the CPU.

The three example Galactic binaries on the record of figure 14 (2**23 samples at
dt = 4 s, about 388 d, Tukey 0.05, 200 d into the ESA orbit file), padded for
Delta T = 1e4 s (Nf = 2500) with L = 25, P = 128. Every call takes a new sky
position and a new waveform.

    python numbers_runtime.py zeroth             # LISAresponse.one_arm_response
    python numbers_runtime.py first [--float32]  # first_order.model.links

Each times the response on wavelet coefficients, the call a likelihood makes; the
forward transform of h+ and hx is timed apart. ``first`` takes the transforms of
the waveform evaluated at t + delta_R(t) (first_order/model.py), formed for each
call before its timer starts, as the plain transforms are for ``zeroth``.
``--float32`` runs its working precision in float32 and records the difference
from float64 on the same input. Writes numbers/runtime_<label>.txt. Peak memory
is about 6 GiB.
"""

import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import fig14_mismatch_vs_dT_zeroth_order as fig14
from parameters import (DT, MAX_LAG_L, MONOCHROMATIC_PARAMS, N_SIGNAL,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

HERE = Path(__file__).parent
NF = 2500
REPEATS = 11
# New sky position every call, away from the poles.
SKIES = [(0.4 + 0.21 * i, 0.3 + 0.57 * i) for i in range(REPEATS + 1)]
TITLES = dict(
    zeroth_order="zeroth-order response, LISAresponse.one_arm_response",
    first_order_f64="first-order response, first_order.model.links, float64",
    first_order_f32="first-order response, first_order.model.links, float32")


def setup():
    """Orbits, the padded grid and the three windowed tones on it."""
    from types import SimpleNamespace

    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    orbits, start, signal_times, _ = fig14.setup()
    info = utils.wdm_padding_info(
        N_SIGNAL, NF, max_lag_L=MAX_LAG_L,
        taper_samples=waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"]))
    waves = []
    for params in MONOCHROMATIC_PARAMS:
        _, hp, hc = waveforms.generate_windowed_waveform(
            "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                             info["pad_left"], info["pad_right"])
        waves.append((np.asarray(hp), np.asarray(hc)))
    grid = SimpleNamespace(N=info["target_length"], start=start,
                           t0=start - info["pad_left"] * DT)
    return orbits, grid, waves


def timed(call):
    tick = time.perf_counter()
    out = call()
    return time.perf_counter() - tick, out


def compare(out, ref):
    """Relative L2 difference and mismatch of `out` against `ref`."""
    l2 = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    mismatch = 1 - np.sum(out * ref) / np.sqrt(np.sum(out**2) * np.sum(ref**2))
    return f"rel L2 {l2:.2e}, mismatch {mismatch:.2e}"


def run(orbits, grid, waves, label):
    """Timings of the zeroth- or first-order response, by `label`: rows of
    (description, seconds) and notes."""
    import jax
    import jax.numpy as jnp

    import WDM
    from WDM_LISAresponse import LISAresponse

    first_order = label.startswith("first_order")
    dtype = jnp.float32 if label == "first_order_f32" else jnp.float64
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=grid.N)

    def build():
        response = LISAresponse(wdm, orbits, t0=grid.t0, max_lag_L=MAX_LAG_L,
                                num_interp_points=NUM_INTERP_POINTS)
        if not first_order:
            return response, None
        from first_order import model
        return response, model.constants(response, wdm)
    setup_time, (response, consts) = timed(build)

    dwt_times = [timed(lambda: [jax.block_until_ready(wdm.dwt(x)) for x in pair])[0]
                 for pair in waves]

    def inputs(i):
        """The call's wavelet coefficients; not timed."""
        k = i % len(waves)
        if not first_order:
            return tuple(wdm.dwt(x) for x in waves[k])
        from first_order import model
        warp = model.tone_warp(MONOCHROMATIC_PARAMS[k], grid.start, N_SIGNAL)
        return model.warped_transforms(response, wdm, consts, *SKIES[i], warp)

    if first_order:
        from first_order import model

        def call(i, coeffs, precision=dtype):
            return model.links(wdm, consts, *SKIES[i], *coeffs, precision)
    else:
        def call(i, coeffs):
            return jax.block_until_ready(response(*SKIES[i], *coeffs))

    coeffs = inputs(0)
    first, out = timed(lambda: call(0, coeffs))
    later = []
    for i in range(1, REPEATS + 1):
        coeffs_i = inputs(i)
        later.append(timed(lambda: call(i, coeffs_i))[0])

    notes = [f"output: shape {np.shape(out)}"]
    if label == "first_order_f32":
        notes.append("float32 vs float64, first call's input: "
                     + compare(np.asarray(out), np.asarray(call(0, coeffs, jnp.float64))))
    built = ("LISAresponse construction, filter tables"
             + (", first-order tables" if first_order else ""))
    rows = [(f"setup ({built})", [setup_time]),
            ("first call (compilation + one evaluation)", [first]),
            ("subsequent calls (new sky + waveform each)", later),
            ("dwt of h+ and hx, first waveform (incl. compilation)", dwt_times[:1]),
            ("dwt of h+ and hx, later waveforms", dwt_times[1:])]
    return rows, notes


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()


def code_version():
    """This repository's commit, with -dirty when the working tree has changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True,
                              text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return "unrecorded"
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no")
                     else "")


def write(label, rows, notes, grid):
    from importlib.metadata import version

    path = HERE / "numbers" / f"runtime_{label}.txt"
    lines = [TITLES[label] + " (six links, wavelet domain)", "",
             f"date        {datetime.now().isoformat(timespec='seconds')}",
             f"cpu         {cpu_model()}, {os.cpu_count()} logical cores",
             f"packages    jax {version('jax')}, numpy {version('numpy')}, "
             f"WDM_GW_wavelets {version('WDM_GW_wavelets')}",
             f"code        WDM_LISAresponse {code_version()}",
             "",
             f"signal      N = 2**{int(np.log2(N_SIGNAL))} samples, dt = {DT:g} s, "
             f"T = {N_SIGNAL * DT / 86400:.1f} d, Tukey alpha = {WINDOW_KWARGS['alpha']}",
             f"padded      N = {grid.N} samples ({grid.N * DT / 86400:.1f} d)",
             f"WDM grid    Nf = {NF}, Delta T = {NF * DT:g} s, Nt = {grid.N // NF}, "
             f"L = {MAX_LAG_L}, P = {NUM_INTERP_POINTS}",
             "links       12, 23, 31, 13, 32, 21", ""]
    for name, values in rows:
        values = np.asarray(values)
        if len(values) == 1:
            lines.append(f"{name}: {values[0]:.4g} s")
        elif len(values):
            lines.append(f"{name}: min {values.min():.4g} s, median "
                         f"{np.median(values):.4g} s over {len(values)} "
                         f"(all: {', '.join(f'{v:.4g}' for v in values)})")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak /= 2**30 if sys.platform == "darwin" else 2**20  # bytes / KiB
    lines += [f"peak memory (max RSS): {peak:.2f} GiB"] + notes
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    assert sys.argv[1:2] in (["zeroth"], ["first"]), __doc__
    os.environ["JAX_PLATFORMS"] = "cpu"           # before JAX is first imported
    label = ("zeroth_order" if sys.argv[1] == "zeroth" else
             "first_order_f32" if "--float32" in sys.argv else "first_order_f64")
    orbits, grid, waves = setup()
    write(label, *run(orbits, grid, waves, label), grid)
