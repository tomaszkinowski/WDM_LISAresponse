import jax
import jax.numpy as jnp
import numpy as np 
from lisaconstants import c
from typing import Tuple
#functions used to compute lisa responses 

def basis_vectors(theta : float, 
                  phi : float) -> Tuple[jnp.ndarray, 
                                        jnp.ndarray, 
                                        jnp.ndarray]:
    """
    Set up an orthonormal set of basis vectors (u,v,n) at a point on the sphere
    of the sky given by the spherical polar angles (theta,phi).

    Parameters
    ----------
    theta : float
        polar angle [rad]
    phi : float
        azimuthal angle [rad]

    Returns
    -------
    u : jnp.ndarray
        North/South basis vector, shape=(3,)
    v : jnp.ndarray
        East/West basis vector, shape=(3,)
    n : jnp.ndarray
        Radial basis vector, shape=(3,)
    """
    u = jnp.array([jnp.cos(theta)*jnp.cos(phi), jnp.cos(theta)*jnp.sin(phi), - jnp.sin(theta)])
    v = jnp.array([jnp.sin(phi), -jnp.cos(phi), 0.])
    n = jnp.array([jnp.sin(theta)*jnp.cos(phi), jnp.sin(theta)*jnp.sin(phi), jnp.cos(theta)])
    return u, v, n

def basis_tensors(theta : float, 
                  phi : float) -> Tuple[jnp.ndarray, 
                                        jnp.ndarray]:
    """
    Set up an two GW polarisation basis tensors at a point on the sphere
    of the sky given by the spherical polar angles (theta,phi).

    Parameters
    ----------
    theta : float
        polar angle [rad]
    phi : float
        azimuthal angle [rad]

    Returns
    -------
    e_plus : jnp.ndarray
        plus polarised tensor, shape=(3,3)
    e_cross : jnp.ndarray
        cross polarised tensor, shape=(3,3)
    """
    u, v, n = basis_vectors(theta, phi)
    e_plus = jnp.tensordot(u, u, axes=0) - jnp.tensordot(v, v, axes=0)
    e_cross = jnp.tensordot(u, v, axes=0) + jnp.tensordot(v, u, axes=0)
    return e_plus, e_cross


def _compute_ltt_iterative(times, s, r, orbits):
    """
    Interate Newton-Raphson to find light-travel times between SC.

    Parameters
    ----------
    times : ndarray
        the sample times of the LISA response [s]
        this is the time at which the signal is sent by SC 
    s : int
        sending SC
    r : int
        sending SC
    orbits : LISA orbits class objects
        computes the positions of all the SC

    Returns
    -------
    tt : ndarray
        light travel time, shaped like times [s]
    """
    tt_niter = 3 #number of iterations of technique (need to have order greater than 2)

    x_s = orbits.compute_position(times, sc=[s])[:,0,:]
    x_r = orbits.compute_position(times, sc=[r])[:,0,:]
    
    # First the iteration to find the flat spacetime travel time
    tt = np.linalg.norm(x_s - x_r, axis=1) / c  
    
    for _ in range(tt_niter):
        t_rec = times + tt  
        x_r = orbits.compute_position(t_rec, sc=[r])[:,0,:]
        tt = np.linalg.norm(x_s - x_r, axis=1) / c  

    return tt 


def rhat(times, s, r, orbits):
    """
    compute arm unit vectors

    Parameters
    ----------
    times : ndarray
        the sample times of the LISA response [s]
        this is the time at which the signal is sent by SC 
    s : int
        sending SC
    r : int
        sending SC
    orbits : LISA orbits class objects
        computes the positions of all the SC

    Returns
    -------
    r_hat : ndarray
        light travel time, shape=(num_times, 3)
    """
    tt = _compute_ltt_iterative(times, s, r, orbits)
    x_s = orbits.compute_position(times, [s])[:,0,:] 
    t_rec = times + tt  
    x_r = orbits.compute_position(t_rec, [r])[:,0,:]
    r_hat = (x_r - x_s) 
    r_hat = r_hat / np.linalg.norm(r_hat, axis=1)[:, np.newaxis]
    return r_hat

import numpy as np

def _hann_window(u):
    """
    Hann window defined on |u| <= 1.
    u can be array.
    """
    w = np.zeros_like(u, dtype=np.float64)
    m = np.abs(u) <= 1.0
    # Hann: 0.5 * (1 + cos(pi*u))
    w[m] = 0.5 * (1.0 + np.cos(np.pi * u[m]))
    return w

def _precompute_wsinc_weights_indices(t_query, dt, N, taps=16):
    """
    Precompute (indices, weights) for wrapped windowed-sinc interpolation on a uniform grid.
    - taps must be even: e.g., 8, 16
    Returns:
      idx: shape (taps, Ntq) int, wrapped indices
      w:   shape (taps, Ntq) float64, normalized weights
    """
    if taps % 2 != 0:
        raise ValueError("taps must be even (e.g., 8 or 16).")

    t_query = np.asarray(t_query)
    t_flat = t_query.ravel()
    Ntq = t_flat.size

    # fractional index in samples
    u = t_flat / dt  # can be negative or >N
    k0 = np.floor(u).astype(int)
    frac = u - k0  # in [0,1) for typical u, but okay generally

    # We’ll use taps centered around k0 (approximately)
    # Let M = taps/2. Use neighbors: k0-(M-1), ..., k0+M
    M = taps // 2
    offsets = np.arange(-(M-1), M+1, dtype=int)  # length taps

    # sample indices: k0 + offsets
    idx = (k0[np.newaxis, :] + offsets[:, np.newaxis]) % N  # (taps, Ntq)

    # distance from each neighbor index:
    # For sample at (k0 + off), the offset in samples is (frac - off)
    x = frac[np.newaxis, :] - offsets[:, np.newaxis]  # (taps, Ntq)

    # windowed sinc weights
    # window support in samples: |x| <= M
    # map to [-1,1] for Hann: u_win = x/M
    u_win = x / M
    w = np.sinc(x) * _hann_window(u_win)

    # Normalize weights so they sum to 1 for each query time (preserve DC)
    wsum = np.sum(w, axis=0, keepdims=True)
    # Avoid division by zero (shouldn’t happen, but safe)
    w = np.where(wsum != 0.0, w / wsum, w)

    return idx, w, t_query.shape

def gnm_at_times_grid_np(wdm, n_grid, m_grid, t_query, time_first=True, taps=16):
    """
    Evaluate g_nm(t) for a grid (n_grid,m_grid) at times t_query
    using wrapped windowed-sinc interpolation (Hann window), with `taps` points.

    This treats g_nm as circular over length N (FFT-style periodic).
    """
    dt = wdm.dt
    N  = wdm.N

    n_grid = np.asarray(n_grid)
    m_grid = np.asarray(m_grid)
    if n_grid.shape != m_grid.shape:
        raise ValueError("n_grid and m_grid must have the same shape")

    # precompute indices + weights once
    idx, w, t_shape = _precompute_wsinc_weights_indices(t_query, dt, N, taps=taps)
    t_flat_len = w.shape[1]

    # flatten (n,m) grid
    n_flat = n_grid.ravel()
    m_flat = m_grid.ravel()

    values_list = []
    for n_i, m_i in zip(n_flat, m_flat):
        g = np.asarray(wdm.gnm(n=int(n_i), m=int(m_i)))  # (N,)
        # gather and weighted sum: (taps, Ntq) -> (Ntq,)
        out_flat = np.sum(w * g[idx], axis=0)
        values_list.append(out_flat.reshape(t_shape))

    values_modes_t = np.stack(values_list, axis=0)  # (Nmodes, *t_shape)

    grid_shape = n_grid.shape
    values = values_modes_t.reshape(grid_shape + t_shape)

    if time_first:
        g_dim = len(grid_shape)
        t_dim = len(t_shape)
        perm = list(range(g_dim, g_dim + t_dim)) + list(range(0, g_dim))
        values = np.transpose(values, perm)

    return values


"""

def _cubic_kernel_keys(t, a=-0.5):
    
    Keys cubic convolution kernel.
    a = -0.5 -> Catmull–Rom (good default)
    
    t = np.abs(t)
    w = np.zeros_like(t, dtype=np.float64)

    m1 = (t < 1)
    m2 = (t >= 1) & (t < 2)

    w[m1] = (a + 2)*t[m1]**3 - (a + 3)*t[m1]**2 + 1
    w[m2] = a*t[m2]**3 - 5*a*t[m2]**2 + 8*a*t[m2] - 4*a
    return w

def gnm_at_times_grid_np(wdm, n_grid, m_grid, t_query, time_first=True, a=-0.5):
    
    Evaluate g_nm(t) for a grid of (n, m) at times t_query using 4-point cubic convolution
    WITH periodic wrapping. Indices are taken mod N.

    Parameters
    ----------
    wdm : WDM_transform (needs dt, N, and gnm(n,m)->(N,))
    n_grid, m_grid : arrays of same shape (grid of indices)
    t_query : scalar or array of query times (seconds)
    time_first : output shape convention
    a : Keys parameter (a=-0.5 Catmull–Rom)

    Returns
    -------
    values : np.ndarray
        If time_first: shape = t_shape + grid_shape
        else:         shape = grid_shape + t_shape
    
    dt = wdm.dt
    N  = wdm.N

    n_grid = np.asarray(n_grid)
    m_grid = np.asarray(m_grid)
    if n_grid.shape != m_grid.shape:
        raise ValueError("n_grid and m_grid must have the same shape")

    t_query = np.asarray(t_query)
    t_shape = t_query.shape if t_query.shape != () else (1,)
    t_flat = t_query.ravel()

    # fractional index (allow negative / >N; we'll wrap indices)
    u = t_flat / dt
    k = np.floor(u).astype(int)
    frac = u - k

    # neighbor indices with wrapping
    k_m1 = (k - 1) % N
    k_0  = (k)     % N
    k_p1 = (k + 1) % N
    k_p2 = (k + 2) % N

    # weights
    w_m1 = _cubic_kernel_keys(frac + 1.0, a=a)
    w_0  = _cubic_kernel_keys(frac,       a=a)
    w_p1 = _cubic_kernel_keys(frac - 1.0, a=a)
    w_p2 = _cubic_kernel_keys(frac - 2.0, a=a)

    # flatten grid for looping
    n_flat = n_grid.ravel()
    m_flat = m_grid.ravel()

    values_list = []
    for n_i, m_i in zip(n_flat, m_flat):
        g = np.asarray(wdm.gnm(n=int(n_i), m=int(m_i)))  # (N,)

        out = (
            w_m1 * g[k_m1] +
            w_0  * g[k_0]  +
            w_p1 * g[k_p1] +
            w_p2 * g[k_p2]
        )

        values_list.append(out.reshape(t_shape))

    values_modes_t = np.stack(values_list, axis=0)  # (Nmodes, *t_shape)

    grid_shape = n_grid.shape
    values = values_modes_t.reshape(grid_shape + t_shape)

    if time_first:
        g_dim = len(grid_shape)
        t_dim = len(t_shape)
        perm = list(range(g_dim, g_dim + t_dim)) + list(range(0, g_dim))
        values = np.transpose(values, perm)

    return values

"""