"""Runtime of the six one-arm links at the paper's parameters.

The signal of figure 14 (2**23 samples at dt = 4 s, about 388 d, Tukey 0.05,
200 d into the ESA orbit file), padded for Delta T = 1e4 s (Nf = 2500) with
L = 25, P = 128. Every call takes a new sky position and a new waveform.

    python runtime_benchmark.py wdm            # LISAresponse on JAX's default device
    python runtime_benchmark.py gwr            # lisagwresponse
    python runtime_benchmark.py flr [--gpu]    # fastlisaresponse, cpu or cuda12x

``wdm`` times the response on wavelet coefficients (the call a likelihood
makes) and, apart, the forward transform of h+ and hx on the host CPU. The
external codes take the padded time series and return the six links in time.
Each run writes figures/runtime_<code>_<platform>.txt; ``wdm`` on the CPU also
saves its first output to the cache, which the GPU run is compared against.
BENCH_CODE (environment) is recorded as the code version.
"""

import os
import platform
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
REPEATS = dict(wdm=11, flr=5, gwr=2)
# New sky position every call, away from the poles.
SKIES = [(0.4 + 0.21 * i, 0.3 + 0.57 * i) for i in range(12)]
CPU_OUTPUT = HERE / "cache" / "runtime_wdm_cpu_first_output.npy"


def setup():
    """Orbits, the padded grid and three windowed tones on it."""
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
    from lisaorbits.file import read_orbit_file

    t0 = start - info["pad_left"] * DT
    n = info["target_length"]
    grid = SimpleNamespace(N=n, dt=DT, t0=t0, orbit_file=str(fig14.ORBIT_FILE),
                           file_t0=float(read_orbit_file(str(fig14.ORBIT_FILE)).t0),
                           padded_times=t0 + DT * np.arange(n))
    return orbits, grid, waves


def timed(call):
    tick = time.perf_counter()
    out = call()
    return time.perf_counter() - tick, out


def run_wdm(orbits, grid, waves):
    import jax

    import WDM
    from WDM_LISAresponse import LISAresponse

    device = jax.devices()[0]
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=grid.N)
    setup_time, response = timed(lambda: LISAresponse(
        wdm, orbits, t0=grid.t0, max_lag_L=MAX_LAG_L,
        num_interp_points=NUM_INTERP_POINTS))

    # The waveform transform on the host, as a pipeline would; a transform
    # built on the GPU runs its dwt there, and 2**23 samples overfill 4 GB.
    with jax.default_device(jax.devices("cpu")[0]):
        host = WDM.WDM.WDM_transform(dt=DT, Nf=NF, N=grid.N)
        dwt_times, coeffs = [], []
        for hp, hc in waves:
            t, pair = timed(lambda: tuple(
                np.asarray(jax.block_until_ready(host.dwt(x))) for x in (hp, hc)))
            dwt_times.append(t)
            coeffs.append(pair)
        del host
    coeffs = [tuple(jax.device_put(x, device) for x in pair) for pair in coeffs]

    first, out = timed(lambda: jax.block_until_ready(response(*SKIES[0], *coeffs[0])))
    later = [timed(lambda: jax.block_until_ready(
        response(*SKIES[i], *coeffs[i % len(coeffs)])))[0]
        for i in range(1, REPEATS["wdm"] + 1)]

    notes = [f"output: shape {out.shape}, dtype {out.dtype}"]
    out = np.asarray(out, dtype=np.float64)
    if device.platform == "cpu":
        CPU_OUTPUT.parent.mkdir(exist_ok=True)
        np.save(CPU_OUTPUT, out)
    elif CPU_OUTPUT.exists():
        ref = np.load(CPU_OUTPUT)
        l2 = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        mismatch = 1 - np.sum(out * ref) / np.sqrt(np.sum(out**2) * np.sum(ref**2))
        notes.append(f"first output vs the CPU run's: rel L2 {l2:.2e}, "
                     f"mismatch {mismatch:.2e}")
    if device.platform == "gpu":
        stats = device.memory_stats()
        notes.append(f"peak GPU memory: {stats['peak_bytes_in_use'] / 2**30:.2f} GiB")
    rows = [("setup (LISAresponse construction, filter tables)", [setup_time]),
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


def versions(code):
    from importlib.metadata import PackageNotFoundError, version

    names = dict(wdm=("jax", "jaxlib"), gwr=("lisagwresponse",),
                 flr=("fastlisaresponse",))[code]
    out = []
    for name in names + ("numpy",):
        try:
            out.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            module = __import__(name)
            out.append(f"{name} {getattr(module, '__version__', '?')}")
    return ", ".join(out)


def write(code, platform_name, device_kind, rows, notes, grid):
    path = HERE / "figures" / f"runtime_{code}_{platform_name}.txt"
    title = dict(wdm="WDM_LISAresponse one_arm_response (six links, wavelet domain)",
                 gwr="lisagwresponse (six links, time domain)",
                 flr="fastlisaresponse (six links, time domain)")[code]
    lines = [title, "",
             f"date        {datetime.now().isoformat(timespec='seconds')}",
             f"host        {platform.node()}",
             f"cpu         {cpu_model()}, {os.cpu_count()} logical cores",
             f"device      {platform_name}" + (f" ({device_kind})" if device_kind else ""),
             f"packages    {versions(code)}",
             f"code        {os.environ.get('BENCH_CODE', 'unrecorded')}",
             "",
             f"signal      N = 2**{int(np.log2(N_SIGNAL))} samples, dt = {DT:g} s, "
             f"T = {N_SIGNAL * DT / 86400:.1f} d, Tukey alpha = {WINDOW_KWARGS['alpha']}",
             f"padded      N = {grid.N} samples ({grid.N * DT / 86400:.1f} d)",
             f"WDM grid    Nf = {NF}, Delta T = {NF * DT:g} s, Nt = {grid.N // NF}, "
             f"L = {MAX_LAG_L}, P = {NUM_INTERP_POINTS}",
             "links       12, 23, 31, 13, 32, 21", ""]
    for label, values in rows:
        values = np.asarray(values)
        if len(values) == 1:
            lines.append(f"{label}: {values[0]:.4g} s")
        elif len(values):
            lines.append(f"{label}: min {values.min():.4g} s, median "
                         f"{np.median(values):.4g} s over {len(values)} "
                         f"(all: {', '.join(f'{v:.4g}' for v in values)})")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak /= 2**30 if sys.platform == "darwin" else 2**20  # bytes / KiB
    lines += [f"peak host memory (max RSS): {peak:.2f} GiB"] + notes
    path.write_text("\n".join(lines) + "\n")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    code = sys.argv[1]
    if code == "flr" and "--gpu" in sys.argv:
        # cupy (CUDA 12) must look up its runtime before JAX loads CUDA 13's
        import cupy
        cupy.cuda.get_local_runtime_version()
    orbits, grid, waves = setup()
    if code == "wdm":
        result = run_wdm(orbits, grid, waves)
    elif code == "gwr":
        result = run_gwr(orbits, grid, waves)
    elif code == "flr":
        result = run_flr(orbits, grid, waves, "--gpu" in sys.argv)
    write(code, *result, grid)
