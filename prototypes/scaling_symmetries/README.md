# Scaling symmetries of the WDM response (2026-11-13)

Question: which dependences on ΔT, f, δ, n, m in the WDM response are genuine,
and which collapse onto a few dimensionless variables? Tags: **[proved]** from the
equations/code, **[numerical]** measured here, **[speculative]** untested.

Notation: ΔF ΔT = 1/2, r = f/ΔF − m, x = l − δ/ΔT, ε = f δ̇ / ΔF = 2 f ΔT δ̇ = (m + r) δ̇,
S(r) = Σ_k Φ′(r−k)², R(r) = Σ_k Φ(r−k)² (r−k)² (Φ in bin units, Meyer d=4, A_frac=0.25).
Errors are mismatches M = L2²/2 unless stated.

## Exact
- **[proved][numerical]** T^σ_l(δ) = τ_σ(l − δ/ΔT): a dimensionless 1-D function, independent
  of Nf, Nt, dt. τ(0) = 1. `e1_tables.py`: identical to 1e-16 (T) and 5e-13 (T′) for
  Nf = 32…1000, ΔT = 128…4000 s, both Nt parities.
- **[proved]** m enters X only through the carrier e^{−iπ(m+σ/2)δ/ΔT} and the signs (−1)^{lm},
  (−1)^{n+m}; n only through (−1)^n. This is the even-layer heterodyne
  (g_{n,m+2} = Re[e^{2πit/ΔT} g⁺_{nm}], and e^{2πi t_n/ΔT} = 1 at every centre), and
  `apply_variable_time_shift` already uses it (cos/sin of 2πΔF m δ + rank-1 weights).
- **[proved][numerical]** Tone coefficients at f + 2ΔF in layer m+2 equal those at f in layer m.
  So for constant δ the relative error is identical under m → m+k (k even, kδ/ΔT ∈ ℤ).
  `e2_const.py`: m = 1, 21, 61 agree to 4 digits at δ = 0.3ΔT.

## Collapses
- **[numerical] Constant delay:** M = F(r, δ/ΔT, ψ; L), with ψ = mδ/ΔT mod 1.
  It is NOT flat in m: at small δ, M grows with ψ ≈ 2 f_m δ.
  Below δ ≈ 0.03ΔT, M ∝ (δ/ΔT)² (`e2c.py`). At δ/ΔT = 0.004, L ≈ 16–18 gives what
  L = 25 gives at δ/ΔT ≳ 0.06.
- **[numerical] The fold is not a symmetry of the implemented operator.** The lag window
  [−L, L] is not re-centred on k = round(δ/ΔT), so L_eff = L − |k|.
  - At ΔT = 256 s, L = 25: δ = 0.3 / 10.3 / 20.3 ΔT gives 3e-11 / 8e-8 / 2e-3.
  - At ΔT = 96 s, δ = 500 s: L = 25 gives 8e-10 (vs 1e-11 at δ = 20 s); L = 15 gives 6e-5 (vs 1e-7).
- **[numerical] Delay rate:** M = ½⟨ε²⟩ S(r), first order in ε.
  - Sinusoidal δ(t): identical to 4 digits across Nf = 32/64/128 at fixed (m, δ̇), even
    though f, ΔT and δ/ΔT all change 4× (`e3_rate.py`).
  - The coefficient is 4.76 measured vs ¼S(0.406) = 4.7559 analytic (`cr.py`).
  - S = 0 for |r| ≤ 1/4 (flat top).
- **[numerical] Production fig-10 map, zero fitted parameters** (`predict_map.py`, run from `plotting/`). The model:
  - ν(t) = f δ̇ + 𝒜̇/(2πi𝒜), with 𝒜 = ℓ (F₊ − iF×) sinc(fτ) and power weight w = |𝒜|².
  - M ≈ ½⟨Σ_k [Φ(r−k+Re ν/ΔF) − Φ(r−k)]²⟩_w + ½ S(r)⟨(Im ν)²⟩_w/ΔF² + ½ R(r) ΔF² ⟨(τ d ln sinc/dx)²⟩_w + floor.
  - Fit quality: ±0.1 dex for |r| = 5/12 over ΔT = 176 s–3e5 s and f = 1e-4–0.05 Hz. Every r
    in the high-f/coarse-ΔT corner (flat-top cliffs and saturation included).
  - Power-weighted rms δ̇ = 5.23e-5 (orbits) vs 5.2e-5 implied by the map. The raw rms 7.2e-5
    is wrong: weight by the link's signal power.
  - Not captured: the low-f, coarse-ΔT corner for |r| ≤ 1/4 (+1…+4 dex; higher-order antenna
    modulation), and part of r = ±1/12 at 5.6e4–3e5 s.
  - The fig-2 slow-delay diagonal is a line of constant m (ε ≈ const).
- **[numerical] EMRI production curve, zero fitted parameters** (`emri_predict.py`).
  - Formula: M ≈ ½ ∫Φ′² dx · (2ΔT)² (⟨f²⟩_P ⟨δ̇²⟩_w + ⟨|𝒜̇/𝒜|²⟩_w/4π²), with ∫Φ′² = 8.05.
  - Within ±30% from 3.2e3 s to 3.2e5 s (M from 2.5e-6 to 2e-2).
  - Over-predicts 3× at 1–2e3 s, where the harmonics do not sweep a bin (r not uniform).
  - The m = 0 leak dominates below 560 s.

## Endpoints
- m = 1 is fine unless the tone leaks into m = 0 (m=1, r=−0.41: M = 2e-2).
- m = Nf−1 with r ≤ 0.25 is fine. The m = 0 / Nyquist packing breaks every relation above.

Run from `WDM_LISAresponse/plotting/` for the scripts that read the orbits/caches
(`predict_map.py`, `emri_predict.py`, `rate_orbit.py`). The others run anywhere; all take seconds.
