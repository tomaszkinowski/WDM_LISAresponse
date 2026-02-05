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
    tt_niter = 4 #number of iterations of technique (need to have order greater than 2)

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

def _cubic_kernel_keys(t, a=-0.5):
    """
    Keys cubic convolution kernel.
    a = -0.5 -> Catmull–Rom (good default)
    """
    t = np.abs(t)
    w = np.zeros_like(t, dtype=np.float64)

    m1 = (t < 1)
    m2 = (t >= 1) & (t < 2)

    w[m1] = (a + 2)*t[m1]**3 - (a + 3)*t[m1]**2 + 1
    w[m2] = a*t[m2]**3 - 5*a*t[m2]**2 + 8*a*t[m2] - 4*a
    return w

def gnm_at_times_grid_np(wdm, n_grid, m_grid, t_query, time_first=True, a=-0.5):
    """
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
    """
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

def _cubic_kernel_keys(t, a=-0.5):
    """Keys cubic convolution kernel (Catmull–Rom when a=-0.5)."""
    t = np.abs(t)
    w = np.zeros_like(t, dtype=np.float64)

    m1 = (t < 1)
    m2 = (t >= 1) & (t < 2)

    w[m1] = (a + 2)*t[m1]**3 - (a + 3)*t[m1]**2 + 1
    w[m2] = a*t[m2]**3 - 5*a*t[m2]**2 + 8*a*t[m2] - 4*a
    return w

def _lagrange_weight_matrix(frac, x_nodes):
    """
    Build weights W of shape (P, Nt) for evaluation points frac (Nt,)
    and node locations x_nodes (P,).
    """
    frac = frac.astype(np.float64)
    x_nodes = x_nodes.astype(np.float64)
    P = len(x_nodes)
    Nt = len(frac)
    W = np.ones((P, Nt), dtype=np.float64)
    for j in range(P):
        for k in range(P):
            if k == j:
                continue
            W[j, :] *= (frac - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
    return W

def _lanczos_weights(frac, A):
    """
    frac: (Nt,) fractional part in [0,1)
    A: half-width of kernel (integer), total taps = 2A
    Returns:
      offsets: (2A,) integer offsets
      W: (2A, Nt) weights for each offset
    """
    # offsets centered around 0: [-A+1, ..., A]
    offsets = np.arange(-A + 1, A + 1, dtype=int)  # length 2A

    # x = frac - offset for each tap; shape (2A, Nt)
    x = (frac[None, :] - offsets[:, None]).astype(np.float64)

    # sinc(pi x)/(pi x) using numpy's normalized sinc: np.sinc(x) = sin(pi x)/(pi x)
    # Lanczos window: sinc(x/A)
    W = np.sinc(x) * np.sinc(x / A)

    # Normalize so weights sum to 1 at each frac (helps DC consistency)
    s = np.sum(W, axis=0, keepdims=True)
    W = np.divide(W, s, out=W, where=(s != 0))

    return offsets, W


def gnm_at_times_grid_np_interp(
    wdm, n_grid, m_grid, t_query,
    method="cubic",
    time_first=True,
    a=-0.5,
    wrap=False,
):
    """
    Evaluate g_nm(t) for a grid of (n,m) at times t_query with selectable interpolation.

    Parameters
    ----------
    wdm : WDM_transform (needs dt, N, and gnm(n,m)->(N,))
    n_grid, m_grid : arrays of same shape (grid of indices)
    t_query : scalar or array of query times (seconds)
    method : {"nearest","linear","cubic"}
    time_first : if True -> output shape (t_shape + grid_shape), else (grid_shape + t_shape)
    a : Keys parameter for cubic (a=-0.5 Catmull–Rom)
    wrap : if True, periodic wrap indices mod N. If False, zero-extend out-of-range.

    Returns
    -------
    values : np.ndarray
        If time_first: shape = t_shape + grid_shape
        else:         shape = grid_shape + t_shape
    """
    dt = wdm.dt
    N  = wdm.N

    n_grid = np.asarray(n_grid)
    m_grid = np.asarray(m_grid)
    if n_grid.shape != m_grid.shape:
        raise ValueError("n_grid and m_grid must have the same shape")

    t_query = np.asarray(t_query)
    t_shape = t_query.shape if t_query.shape != () else (1,)
    t_flat = t_query.ravel()

    # fractional sample coordinate
    u = t_flat / dt

    # ----- build neighbor indices + weights depending on method -----
    if method == "nearest":
        k0 = np.rint(u).astype(int)

        if wrap:
            k0i = k0 % N
            w0  = np.ones_like(u, dtype=np.float64)
            idxs = [k0i]
            ws   = [w0]
        else:
            v0  = (k0 >= 0) & (k0 < N)
            k0c = np.clip(k0, 0, N-1)
            w0  = v0.astype(np.float64)
            idxs = [k0c]
            ws   = [w0]

    elif method == "linear":
        k = np.floor(u).astype(int)
        frac = u - k

        k0 = k
        k1 = k + 1

        if wrap:
            k0i = k0 % N
            k1i = k1 % N
            w0  = (1.0 - frac).astype(np.float64)
            w1  = frac.astype(np.float64)
            idxs = [k0i, k1i]
            ws   = [w0,  w1]
        else:
            v0  = (k0 >= 0) & (k0 < N)
            v1  = (k1 >= 0) & (k1 < N)
            k0c = np.clip(k0, 0, N-1)
            k1c = np.clip(k1, 0, N-1)
            w0  = (1.0 - frac) * v0
            w1  = frac * v1
            idxs = [k0c, k1c]
            ws   = [w0.astype(np.float64), w1.astype(np.float64)]

    elif method == "cubic":
        k = np.floor(u).astype(int)
        frac = u - k

        k_m1 = k - 1
        k_0  = k
        k_p1 = k + 1
        k_p2 = k + 2

        w_m1 = _cubic_kernel_keys(frac + 1.0, a=a)
        w_0  = _cubic_kernel_keys(frac,       a=a)
        w_p1 = _cubic_kernel_keys(frac - 1.0, a=a)
        w_p2 = _cubic_kernel_keys(frac - 2.0, a=a)

        if wrap:
            idxs = [k_m1 % N, k_0 % N, k_p1 % N, k_p2 % N]
            ws   = [w_m1,     w_0,     w_p1,     w_p2]
        else:
            v_m1 = (k_m1 >= 0) & (k_m1 < N)
            v_0  = (k_0  >= 0) & (k_0  < N)
            v_p1 = (k_p1 >= 0) & (k_p1 < N)
            v_p2 = (k_p2 >= 0) & (k_p2 < N)

            # zero-extend
            w_m1 = w_m1 * v_m1
            w_0  = w_0  * v_0
            w_p1 = w_p1 * v_p1
            w_p2 = w_p2 * v_p2

            idxs = [
                np.clip(k_m1, 0, N-1),
                np.clip(k_0,  0, N-1),
                np.clip(k_p1, 0, N-1),
                np.clip(k_p2, 0, N-1),
            ]
            ws = [w_m1, w_0, w_p1, w_p2]

        ws = [w.astype(np.float64) for w in ws]

    elif method.startswith("lagrange"):
        P = int(method.replace("lagrange", ""))
        if P < 2:
            raise ValueError("Lagrange needs >=2 points")
        # choose a symmetric stencil; for P=8 -> offsets [-3,-2,-1,0,1,2,3,4]
        half = P // 2
        offsets = np.arange(-half + 1, half + 1, dtype=int)  # length P for even P

        k = np.floor(u).astype(int)
        frac = (u - k).astype(np.float64)  # in [0,1)

        x_nodes = offsets.astype(np.float64)
        W = _lagrange_weight_matrix(frac, x_nodes)  # (P, Nt)

        idxs = [k + off for off in offsets]

        if wrap:
            idxs = [idx % N for idx in idxs]
            ws = [W[j, :] for j in range(P)]
        else:
            valid = [(idx >= 0) & (idx < N) for idx in idxs]
            idxs = [np.clip(idx, 0, N-1) for idx in idxs]
            ws = [(W[j, :] * valid[j]) for j in range(P)]

    elif method.startswith("lanczos"):
    # e.g. method="lanczos16" or "lanczos32"
        A = int(method.replace("lanczos", ""))
        if A < 2:
            raise ValueError("Lanczos needs A>=2")

        k = np.floor(u).astype(int)
        frac = (u - k).astype(np.float64)

        offsets, W = _lanczos_weights(frac, A)   # offsets (2A,), W (2A,Nt)

        idxs = [k + off for off in offsets]
        if wrap:
            idxs = [idx % N for idx in idxs]
            ws = [W[j, :] for j in range(len(offsets))]
        else:
            valid = [(idx >= 0) & (idx < N) for idx in idxs]
            idxs = [np.clip(idx, 0, N-1) for idx in idxs]
            ws = [(W[j, :] * valid[j]) for j in range(len(offsets))]

    else:
        raise ValueError(f"Unknown method={method}")


    # ----- evaluate for each (n,m) mode -----
    n_flat = n_grid.ravel()
    m_flat = m_grid.ravel()

    values_list = []
    for n_i, m_i in zip(n_flat, m_flat):
        g = np.asarray(wdm.gnm(n=int(n_i), m=int(m_i)))  # (N,)

        out = np.zeros_like(u, dtype=np.float64)
        for k_idx, w in zip(idxs, ws):
            out += w * g[k_idx]
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