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


def gnm_at_times_grid_np(wdm, n_grid, m_grid, t_query, time_first=True):
    """
    Evaluate g_nm(t) for a grid of (n, m) indices at arbitrary times t_query,
    using periodic extension on the interval [0, T), where T = N * dt.

    Parameters
    ----------
    wdm : WDM_transform
        Your initialized WDM object (Christopher Moore's WDM).
        Must have attributes: dt, N and method: gnm(n, m) -> array of length N.
    n_grid, m_grid : array-like, same shape
        Indices of time and frequency bins. For example from np.meshgrid.
    t_query : float or array-like
        Times (in seconds) where you want g_nm(t).
        Can be:
          - scalar
          - 1D array (Ntq,)
          - multi-D array (e.g. (Nranges, Ntq))
    time_first : bool, optional
        If True (default), t_query axes come first in the output.
        If False, t_query axes come last.

    Returns
    -------
    values : np.ndarray
        If t_query.shape == t_shape and n_grid.shape == grid_shape:

        - If time_first is True:
            shape = t_shape + grid_shape
            e.g. (Ntq, Nt, Nf) for 1D t_query
        - If time_first is False:
            shape = grid_shape + t_shape
            e.g. (Nt, Nf, Ntq) for 1D t_query

        In all cases:
            values[..., n, m] = g_{n m}(t_query[...])
            with periodic wrap in time.
    """
    dt = wdm.dt
    N  = wdm.N
    T  = N * dt

    # Ensure arrays
    n_grid = np.asarray(n_grid)
    m_grid = np.asarray(m_grid)

    if n_grid.shape != m_grid.shape:
        raise ValueError("n_grid and m_grid must have the same shape")

    # t_query can be scalar, 1D, or multi-D
    t_query = np.asarray(t_query)
    t_shape = t_query.shape if t_query.shape != () else (1,)

    # Flatten times for easier handling
    t_flat = t_query.ravel()

    # Wrap times into [0, T)
    t_mod_flat = np.mod(t_flat, T)

    # Map to nearest grid index
    k_flat = np.rint(t_mod_flat / dt).astype(int) % N    # shape (Nt_flat,)

    # Reshape back to original t_query shape
    k = k_flat.reshape(t_shape)

    # Flatten (n, m) grid so we can loop over modes
    n_flat = n_grid.ravel()
    m_flat = m_grid.ravel()
    nm_count = n_flat.size

    # For each (n, m), evaluate g_nm at all times (broadcasting indices)
    values_list = []
    for n_i, m_i in zip(n_flat, m_flat):
        g_nm = np.asarray(wdm.gnm(n=int(n_i), m=int(m_i)))  # shape (N,)
        values_list.append(g_nm[k])                        # shape t_shape

    # Stack over modes: (Nmodes, *t_shape)
    values_modes_t = np.stack(values_list, axis=0)

    # Reshape to grid_shape + t_shape
    grid_shape = n_grid.shape
    values = values_modes_t.reshape(grid_shape + t_shape)

    if time_first:
        # Move all time axes (currently at the end) in front
        g_dim = len(grid_shape)
        t_dim = len(t_shape)
        # axes: [0..g_dim-1] = grid, [g_dim..g_dim+t_dim-1] = time
        perm = list(range(g_dim, g_dim + t_dim)) + list(range(0, g_dim))
        values = np.transpose(values, perm)

    return values