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

Tested on macOS (Apple silicon) and Linux (x86-64), with Python 3.13 and JAX 0.7 on
the CPU.

1. **Clone both repositories side by side.** The wavelet transform must be
   WDM_GW_wavelets 0.0.6 or later: earlier versions place the T′ filter half a
   sample off when Nt = 2 mod 4, which changes the coarse-ΔT results.

   ```bash
   git clone -b paper_figures https://github.com/tomaszkinowski/WDM_LISAresponse.git
   ```
   ```bash
   git clone -b v0.0.6 https://github.com/cjm96/WDM_GW_wavelets.git
   ```

2. **Create the environment**, from `WDM_LISAresponse/`. It installs JAX's CPU
   build, the external codes of figures 12 and 13, and both checkouts as editable
   packages.

   ```bash
   cd WDM_LISAresponse
   ```
   ```bash
   conda env create -f environment.yml
   ```
   ```bash
   conda activate wdm-lisa
   ```

3. **FEW 2.0, for the EMRI.**

   ```bash
   pip install -e ".[emri]"
   ```

   The first EMRI run downloads FEW's Kerr amplitude data (about 5 GB). To keep it
   on a disk with room, set both directories before running anything:

   ```bash
   export FEW_FILE_STORAGE_DIR=/path/to/few_data
   ```
   ```bash
   export FEW_FILE_DOWNLOAD_DIR=/path/to/few_data/download
   ```

4. **Platform fixes for the external codes** (figures 12 and 13 only).
   - Linux on x86-64 without AVX-512 (`grep -c avx512f /proc/cpuinfo` prints 0):
     lisaanalysistools 1.2.x stops with "Illegal instruction". Install 1.1.13:

     ```bash
     pip install --no-deps --force-reinstall lisaanalysistools==1.1.13
     ```
   - macOS: lisaanalysistools and fastlisaresponse each bundle `libstdc++`, and
     loading both aborts Python. Follow the note at the top of `environment.yml`.

   Then check that both load together:

   ```bash
   python -c "from WDM_LISAresponse.code.comparison import conventions_flr; print(conventions_flr.check_environment())"
   ```

5. **LaTeX.** The figure style (`paper.mplstyle`) typesets with LaTeX: `latex` and
   `dvipng` on the `PATH`, with `amsmath`, `amssymb`, `siunitx`, `type1cm`,
   `cm-super` and `underscore`. Without a TeX installation, TinyTeX installs one
   without root access, in a folder of your choice:

   ```bash
   export TINYTEX_DIR=/path/to/tinytex && mkdir -p "$TINYTEX_DIR"
   ```
   ```bash
   curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
   ```
   ```bash
   export PATH="$TINYTEX_DIR/.TinyTeX/bin/x86_64-linux:$PATH"
   ```
   ```bash
   tlmgr install siunitx type1cm cm-super underscore dvipng
   ```

   (On macOS the `bin` folder is `universal-darwin`.)

6. **Smoke test**, from `plotting/`: an analytic figure, then the EMRI on all four
   codes over a 24-day record (the first EMRI run also downloads FEW's data).

   ```bash
   cd plotting
   ```
   ```bash
   python fig5_meyer_window.py
   ```
   ```bash
   python fig12_emri_link_comparison.py --short
   ```

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
  in all). A script whose caches exist only redraws, in under 10 s; it computes
  only what is missing, and says so. Delete a cache to recompute it.
- Scripts share caches. Figure 2 computes the year of EMRI waveform (about a
  minute and 4.4 GiB) that figures 8, 11, 12 and 13, `numbers_emri_power.py` and
  figure 14's m = 0 sweep reuse; figure 14 computes that sweep, which figures 4
  and 9 reuse; figure 4 draws from the caches of figures 9 and 14. `make_figures.py` runs the scripts in an order that computes
  each cache once. A script run on its own from empty caches also computes what it
  borrows, so it takes longer than its row below.
- `make_figures.py --only 4,9,10` runs some figures; `--parts N` splits the
  three sweeps (figures 9, 10 and 14) into N parallel processes where their
  caches are missing; `--log FILE` writes each script's wall time and peak memory.
- `estimate_runtime.py` runs a calibration of a minute or two and scales the
  reference run (`numbers/make_figures_reference.txt`) by this machine's speed.
  One speed ratio stands in for everything: read its estimates as good to a
  factor of about two.
- The full-length computations hold the whole year in memory: redrawing from the
  caches fits a laptop with 8 GB, recomputing needs the peak memory below (16 GB
  covers every script). `estimate_runtime.py` flags any script that would not fit
  this machine.
- Test modes, which cache nothing: `--short` (figures 12 and 13: a 24-day
  record, mismatches printed), `--only NF` / `--only-bounds NF` (figures 9 and
  14: one tiling), `--only L` (figures 11 and 16: one lag), `--only r`
  (figure 15: one offset).
- JAX runs on the CPU, as for the paper.
- With `SOURCE_DATE_EPOCH` set, the PDFs are byte-for-byte reproducible.

## Figures

Times and memory are those of the reference run (`numbers/make_figures_reference.txt`)
from empty caches, on an Intel Core Ultra 7 265K (20 logical cores, 30 GB), with
`--parts 4`. The three sweeps (†) ran as four parallel processes and take longer
in one. In all, about 4.7 h.

| Fig. | Script | What it computes | Time | Peak memory |
|---|---|---|---|---|
| 2 | `fig2_emri_tf_map.py` | the EMRI's h₊ on the WDM grid at ΔT = 10⁴ s (and the year of EMRI waveform) | 78 s | 4.4 GiB |
| 3 | `fig3_validity_regions.py` | where the approximations hold (analytic) | 2 s | 0.1 GiB |
| 4 | `fig4_emri_mismatch_vs_dT.py` | the EMRI panels of figures 9 and 14 together (draws from their caches) | 2 s | 0.1 GiB |
| 5 | `fig5_meyer_window.py` | the Meyer window (analytic) | 3 s | 0.8 GiB |
| 6 | `fig6_wdm_basis.py` | WDM basis functions and their tiles | 3 s | 0.6 GiB |
| 7 | `fig7_time_delay_filters.py` | the time-delay filters T_ℓ and T′_ℓ | 4 s | 0.6 GiB |
| 8 | `fig8_emri_polarisations.py` | the EMRI's h₊ and h× | 3 s | 0.5 GiB |
| 9 | `fig9_mismatch_vs_dT.py` | first-order response against the time-domain reference, ΔT = 10²–10⁶ s, the three binaries and the EMRI, with on/off-bin bounds | 35 min † | 9.6 GiB |
| 10 | `fig10_mismatch_map.py` | mismatch over (f, ΔT), zeroth (left) and first order (right) | 86 + 96 min † | 7.8 GiB |
| 11 | `fig11_lag_truncation.py` | first-order mismatch against the maximum lag L | 2.5 min | 5.0 GiB |
| 12 | `fig12_emri_link_comparison.py` | the EMRI on one link from four codes (needs lisagwresponse, fastlisaresponse) | 1.5 min | 6.0 GiB |
| 13 | `fig13_emri_tdi_comparison.py` | the EMRI in second-generation X from four codes (also pytdi) | 11 min | 7.8 GiB |
| 14 | `fig14_mismatch_vs_dT_zeroth_order.py` | as figure 9 for the zeroth-order response, and the EMRI's m = 0 sweep | 34 min † | 8.1 GiB |
| 15 | `fig15_off_bin.py` | mismatch as each binary moves across a band, ΔT = 10⁴ s | 14 min | 4.0 GiB |
| 16 | `fig16_lag_truncation_zeroth_order.py` | as figure 11 for the zeroth-order response | 2.8 min | 5.6 GiB |

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

`make_figures.py --numbers` runs the CPU ones after the figures: about 5 minutes
more on the reference machine, peak 6.7 GiB. The runtimes depend on the machine;
each file records the hardware and the code it was measured with. The
fastlisaresponse GPU row needs a GPU and its CUDA 12 build. The WDM GPU row is a
record of a preliminary measurement, not something this branch can reproduce.
