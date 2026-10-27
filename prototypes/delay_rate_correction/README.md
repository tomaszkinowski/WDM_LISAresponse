# Response-model corrections: prototypes

Prototype code for the approximations in the one-arm wavelet response and the
corrections to them. Nothing here is part of the package; the production path
is `response_sub.one_arm_response_wdm`. Run everything from the WDM_LISAresponse
root with the lisatools env and this folder on `PYTHONPATH`, e.g.

    PYTHONPATH=prototypes/delay_rate_correction python prototypes/delay_rate_correction/check_rate_kernel.py 19

Full-year (`2**23` sample) runs are heavy; the small-record options (`19`) are
quick checks. Results depend on WDM being installed from a WDM_GW_wavelets that
includes the T' half-bin fix (28c3714).

## Approximations and where they live

| approximation | correction | elements / tables | kernel | checks and evidence |
|---|---|---|---|---|
| one delay per row, $\delta(t)\approx\delta_n$ | first order in $\dot\delta$: $\dot\delta_n\sum\omega Y$ | `elements_delay_rate.py` ($Y$, $U$, $V$; shared helpers) | `rate_kernel.apply_rate_corrected_shift` | `oracle_local_delay.py`, `reference_delay_rate_shift.py`, `check_rate_kernel.py` |
| one amplitude per row, $A_n\omega_{nm}$ | $\dot A_n\sum\omega Z$ | `elements_amplitude.py` ($Z\propto U(0)$) | `amp_sinc_kernel.apply_amplitude_rate` | `check_amplitude.py`, `breakdown_offbin.py` |
| sinc at the band centre | $S'(c)\sum\omega T^{(1)}$ | `elements_sinc.py` ($T^{(1)}$) | `amp_sinc_kernel.apply_sinc_rate` | `response_full.py` (`full_sinc`) |
| amplitude timing (on-bin floor) | split the delay, or amplitude after the shift | - | - | `breakdown_offbin.py` (timing piece), `response_split.py` |
| beyond first order: $\dot\delta^2$, curvature $\ddot\delta$ | second order + curvature (diverges at coarse $\Delta T$) | `elements_second_order.py` ($Y^{(2)}$, $Y^{(c)}$, $\Phi''$) | `rate_kernel.apply_corrected_shift_2` | `check_rate_kernel_2.py`, `oracle_coarse.py` |
| (alternative) fitted-$K$ in the window offset $\varepsilon$ | polynomial fit instead of Taylor | `fit_window_offset.py` | `rate_kernel.apply_corrected_shift_fit` | `check_fit_kernel.py`, `oracle_coarse.py` |
| split delay | Rømer delay on the waveform, shift only the per-link rest | - | production shift (+ optional first order) | `response_split.py` |

## The response variants

`response_corrections.py` assembles them for one link from the same geometry:
`production`, `first`, `full`, `full_sinc` (`responses`) and `split`,
`split_amp`, `split_amp_sinc`, `split_amp_sinc_first` (`split_responses`).
`plotting/investigation_fig5_delay_rate.py` runs all eight through figure 5
(`--part K/COUNT` to split across processes, `--merge COUNT` to assemble) and
writes `plotting/figures/investigations/fig5_<variant>.pdf`.

## Timing

`bench_rate_kernel.py` (shift, first- and second-order kernels, $Z$ sum) and
`bench_split.py` (the split variant's pieces, including the Rømer warp).
