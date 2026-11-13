"""Fit (window-offset) tables: new u_nodes build against the committed u_grid
build and against a high-order Gauss-Legendre reference."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
OLD = Path(sys.argv[1]) if len(sys.argv) > 1 else None
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from WDM.code.discrete_wavelet_transform.WDM import WDM_transform
sys.path.insert(0, str(HERE.parent / "delay_rate_correction"))
import elements_delay_rate as edr, rate_kernel as rk
wdm = WDM_transform(dt=4.0, Nf=2500, N=2500*64)
L, P, K = 25, 33, 6
eps_max = 9.9e-5*0.03
t0 = time.perf_counter(); new = np.asarray(rk.build_fit_tables(wdm, L, P, eps_max, K)); tn = time.perf_counter() - t0
orig = edr.u_nodes
edr.u_nodes = lambda w, offset=0.0, **k: orig(w, offset=offset, per_piece=160)
rk.u_nodes = edr.u_nodes
ref = np.asarray(rk.build_fit_tables(wdm, L, P, eps_max, K))
scale = np.abs(ref).max()
print(f"new build {tn:.2f} s; new vs high-order reference: {np.abs(new - ref).max()/scale:.1e}")
if OLD:
    import importlib
    for m in ("elements_delay_rate", "rate_kernel"): sys.modules.pop(m, None)
    sys.path.insert(0, str(OLD))
    import rate_kernel as rk_old
    t0 = time.perf_counter(); old = np.asarray(rk_old.build_fit_tables(wdm, L, P, eps_max, K)); to = time.perf_counter() - t0
    print(f"old build {to:.2f} s; old vs reference: {np.abs(old - ref).max()/scale:.1e}")
