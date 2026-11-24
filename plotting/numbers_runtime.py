"""Runtime of the six one-arm links at the paper's parameters.

The three example Galactic binaries on the record of figure 14 (2**23 samples at
dt = 4 s, about 388 d, Tukey 0.05, 200 d into the ESA orbit file), padded for
Delta T = 1e4 s (Nf = 2500) with L = 25, P = 128. Every call takes a new sky
position and a new waveform.

    python numbers_runtime.py zeroth [--gpu]            # LISAresponse.one_arm_response
    python numbers_runtime.py first [--float32] [--gpu] # first_order.model.links
    python numbers_runtime.py gwr                       # lisagwresponse
    python numbers_runtime.py flr [--gpu]               # fastlisaresponse, cpu or cuda12x

``zeroth`` and ``first`` run on the CPU, or with ``--gpu`` on JAX's default device
(unless JAX_PLATFORMS is set), and time the response on
wavelet coefficients, the call a likelihood makes; the forward transform of h+ and
hx is timed apart, on the host CPU. ``first`` takes the transforms of the waveform
evaluated at t + delta_R(t) (first_order/model.py), formed for each call before
its timer starts, as the plain transforms are for ``zeroth``. ``--float32`` runs
its working precision in float32 and records the difference from float64 on the
same input. The external codes take the padded time series and return the six
links in time. Each run writes numbers/runtime_<label>_<device>.txt; a CPU run of
``zeroth`` or ``first`` also caches its first output, which a GPU run is compared
against. Peak host memory is about 6 GiB.
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
REPEATS = dict(zeroth=11, first=11, flr=5, gwr=2)
# New sky position every call, away from the poles.
SKIES = [(0.4 + 0.21 * i, 0.3 + 0.57 * i) for i in range(12)]
TITLES = dict(
    zeroth_order="zeroth-order response, LISAresponse.one_arm_response (six links, "
                 "wavelet domain)",
    first_order_f64="first-order response, first_order.model.links, float64 (six "
                    "links, wavelet domain)",
    first_order_f32="first-order response, first_order.model.links, float32 (six "
                    "links, wavelet domain)",
    gwr="lisagwresponse (six links, time domain)",
    flr="fastlisaresponse (six links, time domain)")


def setup():
    """Orbits, the padded grid and the three windowed tones on it."""
    from types import SimpleNamespace

    from lisaorbits.file import read_orbit_file
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
    t0 = start - info["pad_left"] * DT
    n = info["target_length"]
    grid = SimpleNamespace(N=n, dt=DT, t0=t0, start=start,
                           orbit_file=str(fig14.ORBIT_FILE),
                           file_t0=float(read_orbit_file(str(fig14.ORBIT_FILE)).t0),
                           padded_times=t0 + DT * np.arange(n))
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


def run_wavelet(orbits, grid, waves, label):
    """The zeroth- or first-order response, by `label`."""
    import jax
    import jax.numpy as jnp

    import WDM
    from WDM_LISAresponse import LISAresponse

    device = jax.devices()[0]
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

    # The waveform transform on the host, as a pipeline would; a transform built
    # on the GPU runs its dwt there, and 2**23 samples overfill 4 GB.
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        host = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=grid.N)
        dwt_times = []
        for hp, hc in waves:
            dwt_times.append(timed(lambda: [jax.block_until_ready(host.dwt(x))
                                            for x in (hp, hc)])[0])

    def inputs(i):
        """The call's wavelet coefficients, on the device; not timed."""
        k = i % len(waves)
        if first_order:
            from first_order import model
            warp = model.tone_warp(MONOCHROMATIC_PARAMS[k], grid.start, N_SIGNAL)
            with jax.default_device(cpu):
                pair = model.warped_transforms(response, host, consts, *SKIES[i], warp)
        else:
            with jax.default_device(cpu):
                pair = tuple(host.dwt(x) for x in waves[k])
        return tuple(jax.device_put(np.asarray(x), device) for x in pair)

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
    for i in range(1, REPEATS["first" if first_order else "zeroth"] + 1):
        coeffs_i = inputs(i)
        later.append(timed(lambda: call(i, coeffs_i))[0])

    notes = [f"output: shape {np.shape(out)}"]
    out = np.asarray(out, dtype=np.float64)
    if label == "first_order_f32":
        notes.append("float32 vs float64, first call's input: "
                     + compare(out, np.asarray(call(0, coeffs, jnp.float64))))
    reference = HERE / "cache" / f"runtime_{label}_cpu_first_output.npy"
    if device.platform == "cpu":
        reference.parent.mkdir(exist_ok=True)
        np.save(reference, out)
    elif reference.exists():
        notes.append("first output vs the CPU run's: " + compare(out, np.load(reference)))
    if device.platform == "gpu":
        stats = device.memory_stats()
        notes.append(f"peak GPU memory: {stats['peak_bytes_in_use'] / 2**30:.2f} GiB")
    built = ("LISAresponse construction, filter tables"
             + (", first-order tables" if first_order else ""))
    rows = [(f"setup ({built})", [setup_time]),
            ("first call (compilation + one evaluation)", [first]),
            ("subsequent calls (new sky + waveform each)", later),
            ("dwt of h+ and hx, host CPU, first waveform (incl. compilation)",
             dwt_times[:1]),
            ("dwt of h+ and hx, host CPU, later waveforms", dwt_times[1:])]
    return device.platform, f"{device.device_kind}", rows, notes


def run_gwr(orbits, grid, waves):
    from lisagwresponse import ReadStrain

    from WDM_LISAresponse.code.comparison import conventions_gwr

    times = [timed(lambda: conventions_gwr.links(
        ReadStrain, grid, *waves[i % len(waves)], *SKIES[i]))[0]
        for i in range(REPEATS["gwr"] + 1)]
    rows = [("first call", times[:1]), ("subsequent calls (new sky + waveform each)",
                                        times[1:])]
    notes = ["each call builds ReadStrain (the strain splines) and evaluates the six "
             f"links in blocks of 2**18 samples (conventions_gwr.links)"]
    return "cpu", None, rows, notes


def run_flr(orbits, grid, waves, gpu):
    from fastlisaresponse import pyResponseTDI
    from lisatools.detector import Orbits

    from WDM_LISAresponse.code.comparison.common import sky_angles
    from WDM_LISAresponse.code.comparison.conventions_flr import T_BUFFER

    backend = "cuda12x" if gpu else "cpu"

    def build():
        flr_orbits = Orbits(str(fig14.ORBIT_FILE), force_backend=backend)
        flr_orbits.configure(linear_interp_setup=True)
        return pyResponseTDI(1.0 / DT, grid.N, orbits=flr_orbits,
                             tdi_orbits=flr_orbits, force_backend=backend)

    setup_time, model = timed(build)
    assert model.num_pts == grid.N
    xp = model.xp
    sync = (lambda: xp.cuda.Stream.null.synchronize()) if gpu else (lambda: None)
    records = [xp.asarray(hp + 1j * hc) for hp, hc in waves]
    sync()

    def call(i):
        lam, beta = sky_angles(*SKIES[i])
        model.get_projections(records[i % len(records)], lam, beta,
                              t0=grid.t0 - grid.file_t0,
                              t_buffer=T_BUFFER)
        sync()

    first = timed(lambda: call(0))[0]
    later = [timed(lambda: call(i))[0] for i in range(1, REPEATS["flr"] + 1)]
    rows = [("setup (orbits + pyResponseTDI construction)", [setup_time]),
            ("first call", [first]),
            ("subsequent calls (new sky + waveform each)", later)]
    notes = [f"backend {model.backend.name}, Lagrange order {model.order}; "
             "get_projections only (the six links, no TDI); input already on the "
             "device, output left there"]
    kind = None
    if gpu:
        kind = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return backend, kind, rows, notes


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()


def versions(label):
    from importlib.metadata import PackageNotFoundError, version

    names = dict(gwr=("lisagwresponse",), flr=("fastlisaresponse",)).get(
        label, ("jax", "jaxlib"))
    out = []
    for name in names + ("numpy",):
        try:
            out.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            module = __import__(name)
            out.append(f"{name} {getattr(module, '__version__', '?')}")
    return ", ".join(out)


def code_version():
    """This repository's commit, with -dirty when a tracked Python file has changes
    (regenerated figures and numbers do not count)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True,
                              text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return "unrecorded"
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no",
                                     "--", ":(top)*.py") else "")


def write(label, platform_name, device_kind, rows, notes, grid):
    path = HERE / "numbers" / f"runtime_{label}_{platform_name}.txt"
    lines = [TITLES[label], "",
             f"date        {datetime.now().isoformat(timespec='seconds')}",
             f"cpu         {cpu_model()}, {os.cpu_count()} logical cores",
             f"device      {platform_name}" + (f" ({device_kind})" if device_kind else ""),
             f"packages    {versions(label)}",
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
    lines += [f"peak host memory (max RSS): {peak:.2f} GiB"] + notes
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    code = sys.argv[1]
    assert code in ("zeroth", "first", "gwr", "flr"), __doc__
    if code == "flr" and "--gpu" in sys.argv:
        # cupy (CUDA 12) must look up its runtime before JAX loads CUDA 13's
        import cupy
        cupy.cuda.get_local_runtime_version()
    if code in ("zeroth", "first") and "--gpu" not in sys.argv:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")   # before JAX is first imported
    orbits, grid, waves = setup()
    if code in ("zeroth", "first"):
        label = ("zeroth_order" if code == "zeroth" else
                 "first_order_f32" if "--float32" in sys.argv else "first_order_f64")
        result = run_wavelet(orbits, grid, waves, label)
    elif code == "gwr":
        label, result = code, run_gwr(orbits, grid, waves)
    else:
        label, result = code, run_flr(orbits, grid, waves, "--gpu" in sys.argv)
    write(label, *result, grid)
