"""Are the delay-rate / amplitude / sinc / second-order correction tables a power of
dF times a universal function of x = l - delta/dT, and how accurate is the
prototype builders against Gauss-Legendre in bin units?"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "delay_rate_correction"))
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from WDM.code.discrete_wavelet_transform.WDM import WDM_transform
from elements_delay_rate import window_and_derivative
from elements_second_order import window_second_derivative
from amp_sinc_kernel import build_amp_table, build_sinc_table
from rate_kernel import build_rate_tables, build_rate_tables_2

L, P = 25, 33

def wdm_for(Nf):
    return WDM_transform(dt=4.0, Nf=Nf, N=Nf*64)

def bins_window(wdm, u, order):
    """Phi_b^(order)(u) from the prototype's analytic window, u in bins."""
    dF = float(wdm.dF)
    if order == 2:
        return np.asarray(window_second_derivative(wdm, u*dF))*dF**2.5
    Phi, dPhi = window_and_derivative(wdm, u*dF)
    return Phi*np.sqrt(dF) if order == 0 else dPhi*dF**1.5

def gl_nodes(sigma, A, lam_max, q):
    edge = 1 - A
    lo, hi = max(-edge + sigma/2, -edge - sigma/2), min(edge + sigma/2, edge - sigma/2)
    kinks = {lo, hi}
    for s in (sigma/2, -sigma/2):
        for k in (A, -A, edge, -edge):
            if lo < k + s < hi: kinks.add(k + s)
    kinks = sorted(kinks); us, ws = [], []
    for a, b in zip(kinks[:-1], kinks[1:]):
        n = q + int(np.ceil(0.5*np.pi*lam_max*(b - a)))
        g, w = np.polynomial.legendre.leggauss(n)
        us.append(0.5*(b-a)*g + 0.5*(b+a)); ws.append(0.5*(b-a)*w)
    return np.concatenate(us), np.concatenate(ws)

def gl_table(wdm, sigma, p, j, lam, q=64):
    """dF^(p-j) * int u^p Phi_b(u - sigma/2) Phi_b^(j)(u + sigma/2) e^{i pi u lam} du."""
    u, w = gl_nodes(sigma, wdm.A_frac, np.abs(lam).max(), q)
    integrand = w*u**p*bins_window(wdm, u - sigma/2, 0)*bins_window(wdm, u + sigma/2, j)
    return float(wdm.dF)**(p - j) * (np.exp(1j*np.pi*np.outer(lam.ravel(), u)) @ integrand).reshape(lam.shape)

lags = np.arange(-L, L+1); s = np.linspace(-0.5, 0.5, P)
lam_grid = lags[:, None] - s[None, :]          # rate tables are at x = delta, lam = l - x/dT
lam_lags = lags.astype(float)

# (name, builder index, p, j, kappa) ; amp = conj(-i/2pi U(0)), sinc = conj(int u Phi Phi)
def prototype_tables(wdm):
    t1 = np.asarray(build_rate_tables(wdm, L, P)); t2 = np.asarray(build_rate_tables_2(wdm, L, P))
    out = {}
    for i, sg in enumerate((-1, 0, 1)):
        out[("U", sg)] = t1[i, 0] + 1j*t1[i, 1]; out[("V", sg)] = t1[i, 2] + 1j*t1[i, 3]
        for k, nm in enumerate(("U2", "V2", "W2")):
            out[(nm, sg)] = t2[i, 4+2*k] + 1j*t2[i, 5+2*k]
    a = build_amp_table(wdm, L); sn = build_sinc_table(wdm, L)
    for i, sg in enumerate((-1, 0, 1)):
        out[("amp", sg)] = a[i]; out[("sinc", sg)] = sn[i]
    return out

SPEC = {"U": (0, 1), "V": (1, 1), "U2": (0, 2), "V2": (1, 2), "W2": (2, 2), "amp": (0, 1), "sinc": (1, 0)}
def gl_tables(wdm, q=64):
    out = {}
    for sg in (-1, 0, 1):
        for nm in ("U", "V", "U2", "V2", "W2"):
            p, j = SPEC[nm]; out[(nm, sg)] = gl_table(wdm, sg, p, j, lam_grid, q)
        out[("amp", sg)] = np.conj(-1j/(2*np.pi)*gl_table(wdm, sg, 0, 1, lam_lags, q))
        out[("sinc", sg)] = np.conj(gl_table(wdm, sg, 1, 0, lam_lags, q))
    return out

w1, w2 = wdm_for(250), wdm_for(2500)
t0 = time.perf_counter(); P1 = prototype_tables(w1); tp = time.perf_counter() - t0
t0 = time.perf_counter(); G1 = gl_tables(w1); tg = time.perf_counter() - t0
G1hi = gl_tables(w1, q=160); P2 = prototype_tables(w2)
ratio = float(w1.dF/w2.dF)
print(f"build: prototype {tp:.2f} s, Gauss-Legendre {tg:.2f} s (L={L}, P={P})")
print(f"{'table':<10}{'GL q64 vs q160':>16}{'proto vs GL':>16}{'scaling k':>11}{'universal resid':>17}")
for key in P1:
    scale = np.abs(G1hi[key]).max()
    conv = np.abs(G1[key] - G1hi[key]).max()/scale
    err = np.abs(P1[key] - G1hi[key]).max()/scale
    # find integer k with P2 = ratio^k P1
    ks = {k: np.abs(P2[key] - ratio**k*P1[key]).max()/np.abs(P2[key]).max() for k in (-1, 0, 1, 2)}
    k = min(ks, key=ks.get)
    print(f"{key[0]+str(key[1]):<10}{conv:>16.1e}{err:>16.1e}{k:>11d}{ks[k]:>17.1e}")
