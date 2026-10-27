"""Figure 5 with the prototype response corrections.

Figure 5 (fig5_mismatch_vs_dT.py) recomputed several ways from the same
responses, via prototypes/delay_rate_correction/response_corrections.py in this
repository:

- production: the wavelet response as figure 5 computes it, on the current
  WDM_GW_wavelets (which fixes the T' filter half-bin offset when Nt = 2 mod 4,
  so it differs from figure 5's caches at coarse Delta T);
- first: the shift with the first-order delay-rate term;
- full: the amplitude-rate term on the unshifted grid, then the shift to
  second order in the delay's rate and first in its curvature;
- full_sinc: full plus the window-average sinc to first order;
- split, split_amp, split_amp_sinc, split_amp_sinc_first: the common Roemer
  delay of the constellation centre applied to the waveform itself (tones
  analytically, the EMRI by spline), the plain production shift for the
  per-link rest, with no further term, with the amplitude-rate term, with the
  sinc term too, and with the first-order delay-rate kernel for the rest.

Tilings, signals, references, on-/off-bin frequencies, the cut (padding, taper
and m = 0) and the plot are figure 5's own. ../prototypes/delay_rate_correction
must be on PYTHONPATH. Run with no arguments to compute (or reuse the caches) and plot;
``--only NF`` runs one tiling and the bounds at it, prints, caches nothing.
``--part K/COUNT`` computes tilings K, K + COUNT, ... into cache/parts/ and
stops, so COUNT processes can split the work; ``--merge COUNT`` assembles
their caches and plots.
"""

import sys
import time
from pathlib import Path

import numpy as np

import fig5_mismatch_vs_dT as fig5
from parameters import (DT, EMRI_PARAMS, GC_PHI, GC_THETA, MAX_LAG_L,
                        MONOCHROMATIC_PARAMS, N_SIGNAL, NUM_INTEGRAL_POINTS,
                        NUM_INTERP_POINTS, WINDOW_FUNCTION, WINDOW_KWARGS)

from response_corrections import (SPLIT_VARIANTS, VARIANTS as CORRECTED_VARIANTS,
                                  build_tables, responses as corrected,
                                  split_responses)

HERE = Path(__file__).parent
VARIANTS = CORRECTED_VARIANTS + SPLIT_VARIANTS

CACHES = {kind: (HERE / "cache" / f"investigation_fig5_{kind}.npz",
                 HERE / "cache" / f"investigation_fig5_{kind}_bounds.npz")
          for kind in VARIANTS}


def tukey_continuous(t, t_start, n_samples, alpha):
    """scipy's Tukey window on t_start + k dt, at arbitrary times; zero outside."""
    u = (t - t_start) / ((n_samples - 1) * DT)
    ramp = 0.5 * alpha
    w = np.ones_like(u)
    lo, hi = u < ramp, u > 1 - ramp
    w[lo] = 0.5 * (1 + np.cos(np.pi * (u[lo] / ramp - 1)))
    w[hi] = 0.5 * (1 + np.cos(np.pi * ((u[hi] - 1) / ramp + 1)))
    w[(u < 0) | (u > 1)] = 0.0
    return w


def tone_warp(params, start):
    """The windowed tone at arbitrary (warped) times, analytically."""
    from WDM_LISAresponse.code.waveforms import waveforms

    def warp(times):
        window = tukey_continuous(times, start, N_SIGNAL, WINDOW_KWARGS["alpha"])
        hp, hc = waveforms.generate_waveform_monochromatic(
            times, params["A_plus"], params["A_cross"], params["f"], params["phi_0"])
        return window * hp, window * hc
    return warp


def spline_warp(times, h_plus, h_cross):
    """A sampled waveform at arbitrary times, by quintic spline (EMRI, <= mHz)."""
    from scipy.interpolate import make_interp_spline

    sp = make_interp_spline(times, h_plus, k=5)
    sc = make_interp_spline(times, h_cross, k=5)
    return lambda t: (sp(t), sc(t))


def responses(response, wdm, tables, centre, theta, phi, h_plus, h_cross, link, warp):
    """
    Every variant of the wavelet response on one link: the corrected ones of
    `response_corrections.responses`, and the split-delay ones, for which the
    waveform is evaluated (`warp`) at t + delta_R(t), delta_R = n.x_c/c the
    Roemer delay of the constellation centre `centre` (rows, metres).
    """
    import jax.numpy as jnp
    from scipy.interpolate import make_interp_spline
    from WDM_LISAresponse.code.lisa_response import response_sub as rs

    index = response.link_convention.index(link)
    out = corrected(response, wdm, tables, theta, phi, wdm.dwt(h_plus),
                    wdm.dwt(h_cross), index)
    n_hat = np.asarray(rs.compute_angular_geometry_wdm(
        jnp.atleast_1d(theta), jnp.atleast_1d(phi),
        response.light_travel_time[jnp.asarray([index])],
        response.link_direction[jnp.asarray([index])])[3])[0]
    roemer_rows = centre @ n_hat / rs.c
    t_rows = np.asarray(response.wavelet_times)
    times = np.asarray(response.t0 + wdm.times)
    hp_w, hc_w = warp(times + make_interp_spline(t_rows, roemer_rows, k=5)(times))
    out.update(split_responses(response, wdm, tables, theta, phi, wdm.dwt(hp_w),
                               wdm.dwt(hc_w), roemer_rows, index))
    return out


def tiling(orbits, start, link, Nf):
    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    info = utils.wdm_padding_info(
        N_SIGNAL, int(Nf), max_lag_L=MAX_LAG_L,
        taper_samples=waveforms.tukey_taper_samples(N_SIGNAL, WINDOW_KWARGS["alpha"]))
    wdm = WDM.WDM.WDM_transform(dt=DT, Nf=int(Nf), N=info["target_length"])
    response = LISAresponse(wdm, orbits, t0=start - info["pad_left"] * DT,
                            max_lag_L=MAX_LAG_L, num_interp_points=NUM_INTERP_POINTS,
                            link_convention=(link,))
    tables = build_tables(wdm, MAX_LAG_L, NUM_INTERP_POINTS)
    centre = np.mean(orbits.compute_position(np.asarray(response.wavelet_times),
                                             sc=[1, 2, 3]), axis=1)
    return info, wdm, response, tables, centre, utils.wdm_cut_info(info)


def compute(nfs):
    """fig5.compute for every variant, sharing references and responses."""
    import jax

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    orbits, start, signal_times, link = fig5.setup()
    sky = {name: (GC_THETA, GC_PHI) for name in fig5.TONES}
    sky["EMRI"] = (EMRI_PARAMS["theta"], EMRI_PARAMS["phi"])

    signals, tone_params = {}, {}
    for name, params in zip(fig5.TONES, MONOCHROMATIC_PARAMS):
        _, hp, hc = waveforms.generate_windowed_waveform(
            "monochromatic", t0=start, dt=DT, N=N_SIGNAL, model_kwargs=params,
            window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
        signals[name] = (hp, hc)
        tone_params[name] = params
    td_gen = waveforms.emri_generator()
    _, hp, hc = waveforms.generate_windowed_waveform(
        "emri", t0=start, dt=DT, N=N_SIGNAL,
        model_kwargs=dict(td_gen=td_gen, **EMRI_PARAMS),
        window_function=WINDOW_FUNCTION, window_kwargs=WINDOW_KWARGS)
    signals["EMRI"] = (hp, hc)
    del td_gen

    def padded(info, hp, hc):
        _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                             info["pad_left"], info["pad_right"])
        return hp, hc

    ref_info = utils.wdm_padding_info(N_SIGNAL, fig5.REF_NF,
                                      extra_rows_left=fig5.REF_ROWS,
                                      extra_rows_right=fig5.REF_ROWS)
    ref_response = LISAresponse(
        WDM.WDM.WDM_transform(dt=DT, Nf=fig5.REF_NF, N=ref_info["target_length"]),
        orbits, t0=start - ref_info["pad_left"] * DT, max_lag_L=MAX_LAG_L,
        num_interp_points=NUM_INTERP_POINTS, link_convention=(link,))
    references = {}
    for name, (hp, hc) in signals.items():
        references[name] = np.asarray(ref_response.one_arm_response_time_domain(
            *sky[name], *padded(ref_info, hp, hc), link=link,
            num_integral_points=NUM_INTEGRAL_POINTS))
    del ref_response
    jax.clear_caches()
    print("references done", flush=True)

    out = {kind: dict(Nf=np.asarray(nfs), kept_rows=np.zeros(len(nfs), dtype=int),
                      **{name: np.full(len(nfs), np.nan) for name in fig5.NAMES})
           for kind in VARIANTS}
    for j, Nf in enumerate(nfs):
        tick = time.perf_counter()
        info, wdm, response, tables, centre, cut = tiling(orbits, start, link, Nf)
        for kind in VARIANTS:
            out[kind]["kept_rows"][j] = cut["time_slice"].stop - cut["time_slice"].start
        for name, (hp, hc) in signals.items():
            hp, hc = padded(info, hp, hc)
            warp = (tone_warp(tone_params[name], start) if name in tone_params
                    else spline_warp(np.asarray(response.t0 + wdm.times),
                                     np.asarray(hp), np.asarray(hc)))
            ours = responses(response, wdm, tables, centre, *sky[name], hp, hc,
                             link, warp)
            reference = utils.apply_wdm_cut(np.asarray(wdm.dwt(utils.apply_repadding(
                references[name], ref_info["pad_left"], ref_info["pad_right"],
                info["pad_left"], info["pad_right"]))), cut)
            for kind in VARIANTS:
                out[kind][name][j] = utils.wdm_mismatch(
                    utils.apply_wdm_cut(ours[kind], cut), reference)["mismatch"]
        del response, wdm
        jax.clear_caches()
        print(f"Nf = {Nf:>6}, dT = {Nf * DT:8.3g} s:  "
              + "  ".join(f"{name} " + "/".join(f"{out[kind][name][j]:.1e}"
                                                 for kind in VARIANTS)
                          for name in fig5.NAMES)
              + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)
    return out


def compute_bounds(nfs):
    """fig5.compute_bounds for every variant."""
    import jax

    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    orbits, start, signal_times, link = fig5.setup()
    freqs = dict(zip(("on", "off"), fig5.bound_frequencies(nfs)))
    out = {kind: dict(Nf=np.asarray(nfs), f_on=freqs["on"], f_off=freqs["off"],
                      on=np.full(freqs["on"].shape, np.nan),
                      off=np.full(freqs["on"].shape, np.nan))
           for kind in VARIANTS}
    for j, Nf in enumerate(nfs):
        if np.all(np.isnan(freqs["on"][:, j])):
            continue
        tick = time.perf_counter()
        info, wdm, response, tables, centre, cut = tiling(orbits, start, link, Nf)
        for i, params in enumerate(MONOCHROMATIC_PARAMS):
            for which in ("on", "off"):
                f = freqs[which][i, j]
                if np.isnan(f):
                    continue
                _, hp, hc = waveforms.generate_windowed_waveform(
                    "monochromatic", t0=start, dt=DT, N=N_SIGNAL,
                    model_kwargs=dict(params, f=f), window_function=WINDOW_FUNCTION,
                    window_kwargs=WINDOW_KWARGS)
                _, hp, hc = utils.apply_zero_padding(signal_times, hp, hc,
                                                     info["pad_left"], info["pad_right"])
                ours = responses(response, wdm, tables, centre, GC_THETA, GC_PHI,
                                 hp, hc, link, tone_warp(dict(params, f=f), start))
                reference = utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(
                    response.one_arm_response_time_domain(
                        GC_THETA, GC_PHI, hp, hc, link=link,
                        num_integral_points=NUM_INTEGRAL_POINTS)))), cut)
                for kind in VARIANTS:
                    out[kind][which][i, j] = utils.wdm_mismatch(
                        utils.apply_wdm_cut(ours[kind], cut), reference)["mismatch"]
        del response, wdm
        jax.clear_caches()
        print(f"Nf = {Nf:>6}, dT = {Nf * DT:8.3g} s bounds:  "
              + "  ".join(f"{name} off " + "/".join(f"{out[kind]['off'][i, j]:.1e}"
                                                     for kind in VARIANTS)
                          for i, name in enumerate(fig5.TONES))
              + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)
    return out


def part_paths(index, count):
    return {kind: tuple(HERE / "cache" / "parts"
                        / f"investigation_fig5_{kind}{suffix}_part{index}of{count}.npz"
                        for suffix in ("", "_bounds"))
            for kind in VARIANTS}


def merge(count):
    """Assemble the parts' caches, in tiling order, into the full caches."""
    for kind in VARIANTS:
        for which in (0, 1):
            parts = [dict(np.load(part_paths(i, count)[kind][which]))
                     for i in range(count)]
            order = np.argsort(np.concatenate([p["Nf"] for p in parts]))
            merged = {key: np.concatenate([p[key] for p in parts], axis=-1)[..., order]
                      for key in parts[0]}
            assert np.array_equal(merged["Nf"], fig5.NF), merged["Nf"]
            np.savez(CACHES[kind][which], **merged)


if __name__ == "__main__":
    if "--only" in sys.argv:
        Nf = int(sys.argv[sys.argv.index("--only") + 1])
        compute([Nf])
        compute_bounds([Nf])
        sys.exit()
    if "--part" in sys.argv:
        # --part K/COUNT: tilings K, K + COUNT, ..., caches of its own, no plot
        index, count = map(int, sys.argv[sys.argv.index("--part") + 1].split("/"))
        nfs = fig5.NF[index::count]
        paths = part_paths(index, count)
        paths[VARIANTS[0]][0].parent.mkdir(parents=True, exist_ok=True)
        for which, run in ((0, compute), (1, compute_bounds)):
            out = run(nfs)
            for kind in VARIANTS:
                np.savez(paths[kind][which], **out[kind])
        sys.exit()
    if "--merge" in sys.argv:
        merge(int(sys.argv[sys.argv.index("--merge") + 1]))
    results = {}
    for index, run in ((0, compute), (1, compute_bounds)):
        paths = [CACHES[kind][index] for kind in VARIANTS]
        if all(path.exists() for path in paths):
            loaded = {kind: dict(np.load(CACHES[kind][index])) for kind in VARIANTS}
        else:
            loaded = run(fig5.NF)
            for kind in VARIANTS:
                np.savez(CACHES[kind][index], **loaded[kind])
        for kind in VARIANTS:
            results.setdefault(kind, []).append(loaded[kind])
    for kind in VARIANTS:
        fig5.plot(*results[kind], HERE / "figures" / "investigations"
                  / f"fig5_{kind}.pdf")
