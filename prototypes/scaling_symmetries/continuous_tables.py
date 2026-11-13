"""Continuous (tiling-independent) T, T' tables by Gauss-Legendre quadrature
of the analytic Meyer window, in bin units: x = l - delta/dT,
tau(x) = int Phi(u)^2 cos(pi u x) du, tau'(x) = int Phi(u-1/2) Phi(u+1/2) cos(pi u x) du."""
import numpy as np
from scipy.special import betainc

def phi_bins(u, d, A_frac):
    B_frac = 1 - 2*A_frac
    a = np.abs(u)
    roll = np.clip((a - A_frac)/B_frac, 0.0, 1.0)
    return np.where(a < A_frac, 1.0,
           np.where(a <= A_frac + B_frac, np.cos(0.5*np.pi*betainc(d, d, roll)), 0.0))

def _nodes(breaks, q):
    g, w = np.polynomial.legendre.leggauss(q)
    us, ws = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b > a:
            us.append(0.5*(b-a)*g + 0.5*(b+a)); ws.append(0.5*(b-a)*w)
    return np.concatenate(us), np.concatenate(ws)

def tau_tables(x, d=4, A_frac=0.25, q=64):
    """Returns (2, len(x)): index 0 tau' (sigma=+-1), index 1 tau (sigma=0)."""
    edge = 1 - A_frac                                   # A + B in bins
    kinks = sorted({0.0, A_frac, edge})
    u, w = _nodes(kinks, q)
    tau = 2*np.cos(np.pi*np.outer(x, u)) @ (w*phi_bins(u, d, A_frac)**2)
    top = edge - 0.5                                    # support of the product
    kp = sorted({0.0, top} | {k for k in (0.5 - A_frac, A_frac - 0.5, edge - 0.5) if 0 < k < top})
    up, wp = _nodes(kp, q)
    taup = 2*np.cos(np.pi*np.outer(x, up)) @ (wp*phi_bins(up-0.5, d, A_frac)*phi_bins(up+0.5, d, A_frac))
    return np.stack([taup, tau])

def build_tables_continuous(max_lag_L, num_interp_points, d=4, A_frac=0.25, q=64):
    """Same layout as WDM_transform.build_time_delay_filter_interpolants: (2, 2L+1, P)."""
    lags = np.arange(-max_lag_L, max_lag_L+1)
    s = np.linspace(-0.5, 0.5, num_interp_points)       # delta/dT
    x = (lags[:, None] - s[None, :]).ravel()
    return tau_tables(x, d, A_frac, q).reshape(2, lags.size, s.size)
