# Figures for *Modeling the LISA response in the time-frequency domain*

T. Kinowski and C. J. Moore. This folder regenerates every figure in the paper
(Fig. 1 is a TikZ diagram drawn in the paper source) and the numbers its text quotes.
Each figure has one script; each script runs on its own, computing whatever it
needs that is not already cached. `make_figures.py` runs them all.

All the paper's parameters are in [`parameters.py`](parameters.py): dt = 4 s,
N = 2²³ samples (388 days), Tukey taper α = 0.05, maximum lag L = 25, P = 128
interpolation points, the three example Galactic binaries (0.3, 3 and 30 mHz) and
the EMRI.

## Setup

1. **The environment.** From the repository root:

   ```bash
   conda env create -f environment.yml
   conda activate wdm-lisa
   ```

   On macOS, follow the note at the top of `environment.yml` about
   `fastlisaresponse`'s bundled `libstdc++` before running figures 12 and 13.

2. **WDM_GW_wavelets 0.0.6 or later.** Earlier versions place the T′ filter half
   a sample off when Nt = 2 mod 4, which changes the coarse-ΔT results.
   `environment.yml` installs a checkout next to this repository
   (`../WDM_GW_wavelets`); check it out at `v0.0.6` or later, or install the tag
   directly:

   ```bash
   pip install "WDM_GW_wavelets @ git+https://github.com/cjm96/WDM_GW_wavelets@v0.0.6"
   ```

3. **The EMRI waveform (FEW 2.0).**

   ```bash
   pip install -e ".[emri]"
   ```

   The first EMRI run downloads FEW's Kerr amplitude data (about 5 GB). FEW keeps
   it in its own storage directory; set `FEW_FILE_STORAGE_DIR` to choose where.

4. **LaTeX.** The figure style (`paper.mplstyle`) typesets with LaTeX, so a TeX
   installation with `amsmath`, `amssymb` and `siunitx` must be on the `PATH`
   (matplotlib's `text.usetex` requirements).

Tested with Python 3.13, JAX 0.7 (macOS) and JAX 0.9.2 (Linux).

## Running

Run everything from this folder.

```bash
python fig9_mismatch_vs_dT.py      # one figure
python make_figures.py             # every figure
python estimate_runtime.py         # how long make_figures.py would take here
```

- Figures go to `figures/`, as PDFs. Figures 12 and 13 also write the mismatches
  quoted in their captions, `figures/fig12_emri_link_mismatches.txt` and
  `figures/fig13_emri_tdi_mismatches.txt`.
- Intermediate results are cached in `cache/` (not tracked by git; about 1.4 GB
  in all). A script whose caches exist only redraws, in seconds; it computes
  only what is missing, and says so. Delete a cache to recompute it.
- `make_figures.py --only 4,9,10` runs some figures; `--parts N` splits the
  three sweeps (figures 9, 10 and 14) into N parallel processes where their
  caches are missing - allow about 5 GiB of memory per part; `--log FILE` writes
  each script's wall time and peak memory.
- `estimate_runtime.py` runs a calibration of about a minute and scales the
  reference run (`numbers/make_figures_reference.txt`) by this machine's speed.
  One speed ratio stands in for everything: read its estimates as good to a
  factor of about two.
- The full-length computations hold the whole year in memory: redrawing from the
  caches fits a laptop with 8 GB, recomputing them needs more. Each script's peak
  memory is in the reference run, and `estimate_runtime.py` flags any that would
  not fit this machine.
- Test modes, which cache nothing: `--short` (figures 12 and 13: a 24-day
  record, mismatches printed), `--only NF` / `--only-bounds NF` (figures 9 and
  14: one tiling), `--only L` (figures 11 and 16: one lag), `--only r`
  (figure 15: one offset).
- With `SOURCE_DATE_EPOCH` set, the PDFs are byte-for-byte reproducible.

## Figures

Times and memory are those of the reference run, on an Intel Core Ultra 7 265K
(20 logical cores, 30 GB), one process per script.

| Fig. | Script | What it computes | Time | Peak memory |
|---|---|---|---|---|
| 2 | `fig2_emri_tf_map.py` | the EMRI's h₊ on the WDM grid at ΔT = 10⁴ s | — | — |
| 3 | `fig3_validity_regions.py` | where the approximations hold (analytic) | — | — |
| 4 | `fig4_emri_mismatch_vs_dT.py` | the EMRI panels of figures 9 and 14 together (draws from their caches) | — | — |
| 5 | `fig5_meyer_window.py` | the Meyer window (analytic) | — | — |
| 6 | `fig6_wdm_basis.py` | WDM basis functions and their tiles | — | — |
| 7 | `fig7_time_delay_filters.py` | the time-delay filters T_ℓ and T′_ℓ | — | — |
| 8 | `fig8_emri_polarisations.py` | the EMRI's h₊ and h× | — | — |
| 9 | `fig9_mismatch_vs_dT.py` | first-order response against the time-domain reference, ΔT = 10²–10⁶ s, the three binaries and the EMRI, with on/off-bin bounds | — | — |
| 10 | `fig10_mismatch_map.py` | mismatch over (f, ΔT), zeroth (left) and first order (right) | — | — |
| 11 | `fig11_lag_truncation.py` | first-order mismatch against the maximum lag L | — | — |
| 12 | `fig12_emri_link_comparison.py` | the EMRI on one link from four codes (needs lisagwresponse, fastlisaresponse) | — | — |
| 13 | `fig13_emri_tdi_comparison.py` | the EMRI in second-generation X from four codes (also pytdi) | — | — |
| 14 | `fig14_mismatch_vs_dT_zeroth_order.py` | as figure 9 for the zeroth-order response | — | — |
| 15 | `fig15_off_bin.py` | mismatch as each binary moves across a band, ΔT = 10⁴ s | — | — |
| 16 | `fig16_lag_truncation_zeroth_order.py` | as figure 11 for the zeroth-order response | — | — |

The first-order response lives in [`first_order/`](first_order): `model.py` is
what the scripts call, `fast_split.py` the response and `tables.py` its
correction tables. The zeroth-order response is the package's
`LISAresponse.one_arm_response`.

## Numbers in the text

| Number | Script | Output (in `numbers/`) |
|---|---|---|
| first-order response, CPU, ~360 ms (Sec. II) | `numbers_runtime.py first --float32` (and `first` for float64) | `runtime_first_order_f32_cpu.txt` |
| zeroth-order response, CPU: 3.56 s first call, 0.539 s after (Sec. VI) | `numbers_runtime.py zeroth` | `runtime_zeroth_order_cpu.txt` |
| fastlisaresponse 5.11 s and lisagwresponse 27.9 s, CPU (Sec. VI) | `numbers_runtime.py flr`, `numbers_runtime.py gwr` | `runtime_flr_cpu.txt`, `runtime_gwr_cpu.txt` |
| fastlisaresponse on an NVIDIA RTX A400, 2.84 s (Sec. VI) | `numbers_runtime.py flr --gpu`, with fastlisaresponse's CUDA 12 build | `runtime_flr_cuda12x.txt` |
| WDM response on an NVIDIA RTX A400, 0.0567 s, first call 3.66 s (Sec. VI, preliminary) | **not computable with this branch's code**: measured with the GPU backend under development (branch `feature/gpu-shift-backend` of both repositories) | `runtime_zeroth_order_gpu.txt` (the measurement, kept for the record) |
| the EMRI's derived parameters in Table I: time to plunge, p and e at the end of the record | `numbers_emri_parameters.py` | `emri_parameters.txt` |
| the EMRI's dominant power, f = 2–5 mHz (App. C) | `numbers_emri_power.py` | `emri_power.txt` |
| filter interpolation at P = 128 (App. B) | `numbers_interpolation.py` | `interpolation.txt` |

`make_figures.py --numbers` runs the CPU ones after the figures. The runtimes depend
on the machine; each file records the hardware and the code it was measured with.
The fastlisaresponse GPU row needs a GPU and its CUDA 12 build. The WDM GPU row is a
record of a preliminary measurement, not something this branch can reproduce.
