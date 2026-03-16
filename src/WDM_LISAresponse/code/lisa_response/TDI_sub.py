#subfunctions from the TDI module 

import numpy as np
from WDM_LISAresponse.code.lisa_response.response_sub import _compute_ltt_iterative


def gen1_time_shift(s, r, orbits, t0, npts=2000, year_days=365.25):
    """
    Compute the time shift value for Gen 1 time shifts - unequal but constant arms. 
    We approximate the time-shift value by averaging the light travel time for a year of data points.

    Parameters
    ----------
    s : int
        sending SC
    r : int
        receiving SC
    orbits : LISA orbits class objects
        computes the positions of all the SC
    t0 : float 
        reference start time 

    Returns
    -------
    shift_val : float
        light travel time, single valued 
    """

    year_SI = year_days * 24 * 3600
    times = np.linspace(t0, t0 + year_SI, npts, endpoint=False)

    travel_times = _compute_ltt_iterative(times, s, r, orbits)

    shift_val = float(np.median(travel_times))

    return shift_val

def normalize_links(links):
    """Convert a sequence of signed links to canonical ``(s, r)`` arm pairs."""
    # word can be [(s,r), ...] or np.ndarray shape (k,2)
    w = np.asarray(links, dtype=np.int16)
    if w.ndim != 2 or w.shape[1] != 2:
        raise ValueError("Links must be list of (s,r) pairs or array shape (k,2)")
    return w

def gen2_time_shift(links, orbits, times):
    """
    Compute the time shift value for Gen 2 time shifts - unequal and time varying arms. 
    The function can perform recursive shifts.
    The calculation is done for a set of input times. 

    One-shot composed Gen-2 delay:
    Δ(t) = L_1(t) + L_2(t - L_1(t)) + L_3(t - L_1(t) - L_2(t - L_1(t))) + ...

    Parameters
    ----------
    links : list of tuples
        defines the order of operations as well as the direction of the light travel times
    orbits : LISA orbits class objects
        computes the positions of all the SC
    times : ndarray 

    Returns
    -------
    shift_vals : ndarray
        light travel time, same length as the input times
    """

    links_array = normalize_links(links) #we get an array shape len(links) x 2 

    times = np.asarray(times, dtype=float)
    shift_vals = np.zeros_like(times)
    shift_time = times.copy()

    for s, r in links_array:
        term = _compute_ltt_iterative(shift_time, int(s), int(r), orbits)
        shift_vals += term
        shift_time -= term

    return shift_vals

def fft_time_shift(x, dt, shift):
    """
    FFT constant time shift mapping x(t) -> x(t-shift)

    Parameters
    ----------
    x : ndarray 
        array of times for the waveforms 
    dt : float
        resolution of the time domain signal
    shift : float 
        amount we shift the input signal by 
    

    Returns
    -------
    x(t - shift) : ndarray
        light travel time, same length as the input times

    """
    N = len(x)
    freqs = np.fft.rfftfreq(N, d=dt)
    X = np.fft.rfft(x)
    X *= np.exp(-2j*np.pi*freqs*shift)
    return np.fft.irfft(X, n=N)

def precompute_gen2_sagnac_shifts(times, orbits):
    """
    Precompute composed delay arrays for three cyclicly-related variables.

    Returns
    -------
    shifts_alpha, shifts_beta, shifts_gamma : dict
        Each dict maps:
          key: ((s1,r1), (s2,r2), ...)   # link-chain pattern
          val: Δ(t) ndarray, shape (len(times),)

    Notes
    -----
    - beta and gamma patterns are cyclic permutations (1->2->3->1) of alpha patterns.
    - No "empty / no-shift" pattern is included.
    """

    times = np.asarray(times, dtype=float)

    # --- define the base alpha pattern-set directly as link chains ---
    # Edit/extend this list to match the patterns you need.
    alpha_patterns = [
        ((2, 1),),
        ((2, 1), (3, 2)),
        ((2, 1), (3, 2), (1, 3)),

        ((2, 1), (3, 2), (1, 3), (3, 1)),
        ((2, 1), (3, 2), (1, 3), (3, 1), (2, 3)),

        ((3, 1),),
        ((3, 1), (2, 3)),
        ((3, 1), (2, 3), (1, 2)),

        ((3, 1), (2, 3), (1, 2), (2, 1)),
        ((3, 1), (2, 3), (1, 2), (2, 1), (3, 2)),
    ]

    # --- cyclic permutation helper ---
    def rot_sc(i, k):
        return ((i - 1 + k) % 3) + 1

    def rot_link(link, k):
        s, r = link
        return (rot_sc(s, k), rot_sc(r, k))

    def rot_chain(chain, k):
        return tuple(rot_link(lk, k) for lk in chain)

    beta_patterns  = [rot_chain(p, 1) for p in alpha_patterns]
    gamma_patterns = [rot_chain(p, 2) for p in alpha_patterns]

    # --- compute a delay library for a set of patterns ---
    def compute_library(patterns):
        lib = {}
        for chain in patterns:
            if chain in lib:
                continue
            lib[chain] = gen2_time_shift(chain, orbits, times)
        return lib

    shifts_alpha = compute_library(alpha_patterns)
    shifts_beta  = compute_library(beta_patterns)
    shifts_gamma = compute_library(gamma_patterns)

    return shifts_alpha, shifts_beta, shifts_gamma

def interp_time_shift(x, dt, shift, order=7, mode="zero"):
    """
    Interpolating time shift mapping x(t) -> x(t - shift).

    shift can be:
      - float (constant shift)
      - ndarray shape (N,) (time-dependent shift)

    order = polynomial order (uses order+1 points). Good accuracy defaults: 7, 9, 11.
    mode:
      - "zero": out-of-range samples treated as 0
      - "edge": clamp to nearest endpoint
    """
    x = np.asarray(x)
    N = x.size
    if N == 0:
        return x.copy()

    if np.isscalar(shift):
        shift_arr = np.full(N, float(shift), dtype=np.float64)
    else:
        shift_arr = np.asarray(shift, dtype=np.float64)
        if shift_arr.shape != (N,):
            raise ValueError(f"shift must be scalar or shape {(N,)}, got {shift_arr.shape}")

    n = np.arange(N, dtype=np.float64)
    u = n - shift_arr / dt  # sample index where we evaluate x

    half = order // 2
    i0 = np.floor(u).astype(np.int64) - half
    frac = u - i0  # fractional position relative to stencil start

    # Lagrange weights on nodes 0..order
    nodes = np.arange(order + 1, dtype=np.float64)
    w = np.ones((N, order + 1), dtype=np.float64)
    for k in range(order + 1):
        denom = 1.0
        for j in range(order + 1):
            if j == k:
                continue
            w[:, k] *= (frac - j)
            denom *= (k - j)
        w[:, k] /= denom

    y = np.zeros(N, dtype=x.dtype)
    for k in range(order + 1):
        idx = i0 + k
        if mode == "edge":
            idxc = np.clip(idx, 0, N - 1)
            y += w[:, k].astype(x.real.dtype) * x[idxc]
        elif mode == "zero":
            valid = (idx >= 0) & (idx < N)
            if np.any(valid):
                y[valid] += w[valid, k].astype(x.real.dtype) * x[idx[valid]]
        else:
            raise ValueError("mode must be 'zero' or 'edge'")
    return y

def precompute_gen2_michelson_shifts(times, orbits):
    """
    Precompute composed delay arrays for three cyclicly-related Michelson-type variables.

    Returns
    -------
    shifts_X, shifts_Y, shifts_Z : dict
        Each dict maps:
          key: ((s1,r1), (s2,r2), ...)   # link-chain pattern
          val: Δ(t) ndarray, shape (len(times),)

    Notes
    -----
    - Y and Z patterns are cyclic permutations (1->2->3->1) of X patterns.
    - No "empty / no-shift" pattern is included.
    """

    times = np.asarray(times, dtype=float)

    # ------------------------------------------------------------
    # Base X pattern-set directly as link chains in YOUR notation:
    # each element is a tuple of directed links (s,r), and the order
    # is exactly the order consumed by gen2_time_shift.
    #
    # This list must contain every distinct composed delay you need
    # to build X1 (excluding the empty/no-delay case).
    # ------------------------------------------------------------
    X_patterns = [
        # group A: chains starting with (2,1) ...  (these correspond to the "3..." family in your X1)
        ((2, 1),),
        ((2, 1), (1, 2)),
        ((2, 1), (1, 2), (3, 1)),
        ((2, 1), (1, 2), (3, 1), (1, 3)),
        ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1)),
        ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3)),
        ((2, 1), (1, 2), (3, 1), (1, 3), (3, 1), (1, 3), (2, 1)),

        # group B: chains starting with (3,1) ...  (these correspond to the "...322'..." family in your X1)
        ((3, 1),),
        ((3, 1), (1, 3)),
        ((3, 1), (1, 3), (2, 1)),
        ((3, 1), (1, 3), (2, 1), (1, 2)),
        ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1)),
        ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2)),
        ((3, 1), (1, 3), (2, 1), (1, 2), (2, 1), (1, 2), (3, 1)),
    ]

    # --- cyclic permutation helper (same as your Sagnac code) ---
    def rot_sc(i, k):
        return ((i - 1 + k) % 3) + 1

    def rot_link(link, k):
        s, r = link
        return (rot_sc(s, k), rot_sc(r, k))

    def rot_chain(chain, k):
        return tuple(rot_link(lk, k) for lk in chain)

    Y_patterns = [rot_chain(p, 1) for p in X_patterns]
    Z_patterns = [rot_chain(p, 2) for p in X_patterns]

    # --- compute a delay library for a set of patterns ---
    def compute_library(patterns):
        lib = {}
        for chain in patterns:
            if chain in lib:
                continue
            lib[chain] = gen2_time_shift(chain, orbits, times)
        return lib

    shifts_X = compute_library(X_patterns)
    shifts_Y = compute_library(Y_patterns)
    shifts_Z = compute_library(Z_patterns)

    return shifts_X, shifts_Y, shifts_Z

