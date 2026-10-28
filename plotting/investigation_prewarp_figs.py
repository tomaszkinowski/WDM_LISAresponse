"""Figures 5, 6 and 7 with the amplitude pre-warp (provisional_prewarp.py).

Each figure's own compute and plot, unchanged, with LISAresponse.one_arm_response
swapped for the pre-warped stand-in; the time-domain references are untouched.
Caches go to cache/provisional_prewarp_*.npz and the PDFs to
figures/investigations/prewarp/figures/ under the figures' own names, so the
paper's figures and caches are never overwritten.

    python investigation_prewarp_figs.py fig5 --part K/COUNT   # tilings K, K + COUNT, ...
    python investigation_prewarp_figs.py fig5 --merge COUNT     # assemble the parts
    python investigation_prewarp_figs.py fig6                   # compute (or reuse) and plot
    python investigation_prewarp_figs.py fig6 --lag=40          # the same at L = 40, tagged _L40
    python investigation_prewarp_figs.py fig7
    python investigation_prewarp_figs.py fig5 --plot            # plot from the merged caches

fig5 also takes ``--lag=L`` (the module's MAX_LAG_L) and ``--envelope`` (the band's
lower edge as the least over offsets around the band centre and the source itself,
see envelope_bounds); either tags its caches and PDF, e.g. _L40_envelope.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import provisional_prewarp

HERE = Path(__file__).parent
OUT = HERE / "figures" / "investigations" / "prewarp" / "figures"
CACHE = HERE / "cache"


def plot_into_investigations(module, plot, suffix=""):
    """Run a figure module's plot, which reads its style from and writes into
    module.HERE / "figures", in a staging directory, then move each PDF to
    OUT under its own name."""
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        shutil.copy(HERE / "paper.mplstyle", staging / "paper.mplstyle")
        (staging / "figures").mkdir()
        module.HERE = staging
        plot(staging / "figures")
        for pdf in (staging / "figures").glob("*.pdf"):
            name = f"{pdf.stem}{suffix}{pdf.suffix}"
            shutil.move(pdf, OUT / name)
            print(f"wrote {OUT / name}")


# --envelope: offsets from the nearest band centre, in units of Delta F, at which
# each tone is also evaluated; the band's lower edge is then the least of these, the
# on-bin value and the source's own, so it brackets the source by construction.
DELTAS = (-0.25, -0.1, -0.05, 0.05, 0.1, 0.25)


def fig5_paths(tag="", index=None, count=None):
    part = "" if index is None else f"_part{index}of{count}"
    folder = CACHE / "parts" if index is not None else CACHE
    return (folder / f"provisional_prewarp{tag}_fig5{part}.npz",
            folder / f"provisional_prewarp{tag}_fig5_bounds{part}.npz")


def envelope_bounds(fig5, nfs):
    """fig5.compute_bounds plus each tone at m Delta F + DELTAS Delta F, m its
    nearest band; NaN where a frequency would leak into m = 0. ``offsets`` has
    shape (tones, len(DELTAS), tilings)."""
    import time

    import jax

    import WDM
    from WDM_LISAresponse import LISAresponse
    from WDM_LISAresponse.code.utils import utils
    from WDM_LISAresponse.code.waveforms import waveforms

    orbits, start, signal_times, link = fig5.setup()
    on, off = fig5.bound_frequencies(nfs)
    delta_f = 1/(2*np.asarray(nfs)*fig5.DT)
    shifted = on[:, None, :] + np.asarray(DELTAS)[None, :, None]*delta_f
    shifted[shifted < fig5.leak_edge()*delta_f] = np.nan
    freqs = dict(on=on, off=off)
    out = dict(Nf=np.asarray(nfs), f_on=on, f_off=off, f_offsets=shifted,
               on=np.full(on.shape, np.nan), off=np.full(on.shape, np.nan),
               offsets=np.full(shifted.shape, np.nan))
    taper = waveforms.tukey_taper_samples(fig5.N_SIGNAL, fig5.WINDOW_KWARGS["alpha"])
    for j, Nf in enumerate(nfs):
        if np.all(np.isnan(on[:, j])):
            continue
        tick = time.perf_counter()
        info = utils.wdm_padding_info(fig5.N_SIGNAL, int(Nf), max_lag_L=fig5.MAX_LAG_L,
                                      taper_samples=taper)
        wdm = WDM.WDM.WDM_transform(dt=fig5.DT, Nf=int(Nf), N=info["target_length"])
        response = LISAresponse(wdm, orbits, t0=start - info["pad_left"]*fig5.DT,
                                max_lag_L=fig5.MAX_LAG_L,
                                num_interp_points=fig5.NUM_INTERP_POINTS,
                                link_convention=(link,))
        cut = utils.wdm_cut_info(info)

        def mismatch(params, f):
            _, h_plus, h_cross = waveforms.generate_windowed_waveform(
                "monochromatic", t0=start, dt=fig5.DT, N=fig5.N_SIGNAL,
                model_kwargs=dict(params, f=f), window_function=fig5.WINDOW_FUNCTION,
                window_kwargs=fig5.WINDOW_KWARGS)
            _, h_plus, h_cross = utils.apply_zero_padding(
                signal_times, h_plus, h_cross, info["pad_left"], info["pad_right"])
            ours = response.one_arm_response(fig5.GC_THETA, fig5.GC_PHI, wdm.dwt(h_plus),
                                             wdm.dwt(h_cross), link=link)
            reference = response.one_arm_response_time_domain(
                fig5.GC_THETA, fig5.GC_PHI, h_plus, h_cross, link=link,
                num_integral_points=fig5.NUM_INTEGRAL_POINTS)
            return utils.wdm_mismatch(
                utils.apply_wdm_cut(np.asarray(ours), cut),
                utils.apply_wdm_cut(np.asarray(wdm.dwt(np.asarray(reference))), cut)
            )["mismatch"]

        for i, params in enumerate(fig5.MONOCHROMATIC_PARAMS):
            for kind in ("on", "off"):
                if not np.isnan(freqs[kind][i, j]):
                    out[kind][i, j] = mismatch(params, freqs[kind][i, j])
            for k in range(len(DELTAS)):
                if not np.isnan(shifted[i, k, j]):
                    out["offsets"][i, k, j] = mismatch(params, shifted[i, k, j])

        del response, wdm
        jax.clear_caches()
        print(f"Nf = {Nf:>6}, dT = {Nf*fig5.DT:8.3g} s:  "
              + "  ".join(f"{name} on {out['on'][i, j]:.1e} "
                          f"min {np.nanmin(np.append(out['offsets'][i, :, j], out['on'][i, j])):.1e} "
                          f"off {out['off'][i, j]:.1e}"
                          for i, name in enumerate(fig5.TONES)
                          if not np.isnan(out["on"][i, j]))
              + f"   ({time.perf_counter() - tick:.0f} s)", flush=True)
    return out


def with_skipped(out, nfs):
    """Spread arrays over the kept tilings (last axis) onto all of ``nfs``: NaN
    for floats, 0 for integers, and Nf itself."""
    columns = np.searchsorted(nfs, out["Nf"])
    full = {}
    for key, value in out.items():
        if key == "Nf":
            full[key] = np.asarray(nfs)
            continue
        blank = 0 if np.issubdtype(value.dtype, np.integer) else np.nan
        full[key] = np.full(value.shape[:-1] + (len(nfs),), blank, dtype=value.dtype)
        full[key][..., columns] = value
    return full


def envelope_for_plot(fig5, res, bounds):
    """The bounds with "on" replaced by the least of on-bin, the offsets and the
    source's own mismatch, where the band exists."""
    bounds = dict(bounds)
    lower = np.nanmin(np.concatenate(
        [bounds["on"][:, None], bounds["offsets"],
         np.stack([res[name] for name in fig5.TONES])[:, None]], axis=1), axis=1)
    bounds["on"] = np.where(np.isnan(bounds["on"]), np.nan, lower)
    return bounds


def fig5_main(args):
    import fig5_mismatch_vs_dT as fig5

    lag = [int(a.split("=")[1]) for a in args if a.startswith("--lag=")]
    envelope = "--envelope" in args
    tag = (f"_L{lag[0]}" if lag else "") + ("_envelope" if envelope else "")
    if lag:
        fig5.MAX_LAG_L = lag[0]

    if "--part" in args:
        index, count = map(int, args[args.index("--part") + 1].split("/"))
        nfs = fig5.NF[index::count]
        # --skip=NF,...: tilings left out (NaN), e.g. where the padding for a large L
        # runs past the orbit file
        skip = [int(n) for a in args if a.startswith("--skip=")
                for n in a.split("=")[1].split(",")]
        kept = np.array([n for n in nfs if n not in skip])
        paths = fig5_paths(tag, index, count)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        np.savez(paths[0], **with_skipped(fig5.compute(kept), nfs))
        bounds = envelope_bounds(fig5, kept) if envelope else fig5.compute_bounds(kept)
        np.savez(paths[1], **with_skipped(bounds, nfs))
        return
    if "--merge" in args:
        count = int(args[args.index("--merge") + 1])
        for which in (0, 1):
            parts = [dict(np.load(fig5_paths(tag, i, count)[which])) for i in range(count)]
            order = np.argsort(np.concatenate([p["Nf"] for p in parts]))
            merged = {key: np.concatenate([p[key] for p in parts], axis=-1)[..., order]
                      for key in parts[0]}
            assert np.array_equal(merged["Nf"], fig5.NF), merged["Nf"]
            np.savez(fig5_paths(tag)[which], **merged)
    res, bounds = [dict(np.load(path)) for path in fig5_paths(tag)]
    if envelope:
        bounds = envelope_for_plot(fig5, res, bounds)
    plot_into_investigations(
        fig5, lambda figures: fig5.plot(res, bounds, figures / "fig5_mismatch_vs_dT.pdf"),
        tag)


def single_main(module, compute, lag=None):
    """``lag`` overrides the module's MAX_LAG_L (padding and kernel both) and tags
    the cache and PDF with _L<lag>."""
    suffix = ""
    if lag is not None:
        module.MAX_LAG_L = lag
        suffix = f"_L{lag}"
    module.CACHE = CACHE / f"provisional_prewarp{suffix}_{module.CACHE.name}"
    if module.CACHE.exists():
        results = dict(np.load(module.CACHE))
    else:
        results = compute()
        np.savez(module.CACHE, **results)
    if "--no-plot" not in sys.argv:
        plot_into_investigations(module, lambda figures: module.plot(results), suffix)


if __name__ == "__main__":
    provisional_prewarp.patch()
    which, args = sys.argv[1], sys.argv[2:]
    if which == "fig5":
        fig5_main(args)
    elif which == "fig6":
        import fig6_off_bin as fig6
        lag = [int(a.split("=")[1]) for a in args if a.startswith("--lag=")]
        single_main(fig6, lambda: fig6.compute(fig6.R), *lag)
    elif which == "fig7":
        import fig7_lag_truncation as fig7
        single_main(fig7, lambda: fig7.compute(fig7.NF, fig7.LAGS))
    else:
        raise SystemExit(f"unknown figure {which!r}")
