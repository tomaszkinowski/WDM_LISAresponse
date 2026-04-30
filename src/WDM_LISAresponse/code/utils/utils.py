"""
Utility functions for WDM_LISAresponse
"""
import numpy as np


def wdm_tail_padding_info(n_samples, Nf):
    """Compute right-tail padding needed to satisfy WDM tiling constraints."""
    n_samples_i = int(n_samples)
    Nf_i = int(Nf)

    if n_samples_i < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples_i}.")
    if Nf_i < 1:
        raise ValueError(f"Nf must be >= 1, got {Nf_i}.")

    import WDM

    block = 2 * Nf_i
    target_length = int(WDM.utils.next_multiple(n_samples_i, block))
    pad_right = int(target_length - n_samples_i)

    return {
        "input_length": n_samples_i,
        "target_length": target_length,
        "pad_right": pad_right,
        "Nf": Nf_i,
        "Nt": int(target_length // Nf_i),
    }


def next_multiple(value, base):
    """Compatibility wrapper around ``WDM.utils.next_multiple``."""
    value_i = int(value)
    base_i = int(base)
    if value_i < 0:
        raise ValueError(f"value must be non-negative, got {value_i}.")
    if base_i <= 0:
        raise ValueError(f"base must be positive, got {base_i}.")
    import WDM
    return int(WDM.utils.next_multiple(value_i, base_i))


def compute_wdm_cut_info(
    original_length,
    Nf,
    dt,
    *,
    alpha=0.0,
    tukey_factor=0.0,
    window_pad_left=None,
    window_pad_right=None,
    include_wdm_tail_padding=True,
    include_taper_margin=False,
    wdm_dT=None,
    wdm_Nt=None,
    drop_m0=True,
    drop_freq_edges = False,
):
    """Build WDM-domain cut metadata for mismatch calculations."""
    original_length_i = int(original_length)
    Nf_i = int(Nf)
    window_pad_left_i = None if window_pad_left is None else int(window_pad_left)
    window_pad_right_i = None if window_pad_right is None else int(window_pad_right)
    dt_f = float(dt)
    alpha_f = float(alpha)
    tukey_factor_f = float(tukey_factor)

    if original_length_i < 1:
        raise ValueError(f"original_length must be >= 1, got {original_length_i}.")
    if Nf_i < 1:
        raise ValueError(f"Nf must be >= 1, got {Nf_i}.")
    if dt_f <= 0.0:
        raise ValueError(f"dt must be positive, got {dt_f}.")
    if not (0.0 <= alpha_f <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha_f}.")
    if tukey_factor_f < 0.0:
        raise ValueError(f"tukey_factor must be non-negative, got {tukey_factor_f}.")

    inferred_window_pad = int(np.ceil(tukey_factor_f * alpha_f * 0.5 * original_length_i))
    if window_pad_left_i is None:
        window_pad_left_i = inferred_window_pad
    if window_pad_right_i is None:
        window_pad_right_i = inferred_window_pad

    if window_pad_left_i < 0 or window_pad_right_i < 0:
        raise ValueError("window_pad_left/window_pad_right must be non-negative.")

    taper_margin_samples = int(np.ceil(0.5 * alpha_f * original_length_i)) if include_taper_margin else 0
    total_left_samples = int(taper_margin_samples + window_pad_left_i)
    total_right_samples_no_tail = int(taper_margin_samples + window_pad_right_i)

    post_window_length = int(original_length_i + window_pad_left_i + window_pad_right_i)
    tail_info = wdm_tail_padding_info(post_window_length, Nf_i)
    tail_pad_right = int(tail_info["pad_right"])
    total_right_samples = int(
        total_right_samples_no_tail + (tail_pad_right if include_wdm_tail_padding else 0)
    )

    wdm_dT_f = float(Nf_i * dt_f) if wdm_dT is None else float(wdm_dT)
    if wdm_dT_f <= 0.0:
        raise ValueError(f"wdm_dT must be positive, got {wdm_dT_f}.")

    left_cut = int(np.ceil(total_left_samples * dt_f / wdm_dT_f))
    right_cut = int(np.ceil(total_right_samples * dt_f / wdm_dT_f))

    Nt = int(tail_info["Nt"]) if wdm_Nt is None else int(wdm_Nt)
    if Nt < 1:
        raise ValueError(f"wdm_Nt must be >= 1, got {Nt}.")

    t_start = max(0, left_cut)
    t_stop = Nt - max(0, right_cut)
    if t_stop < t_start:
        t_stop = t_start

    if drop_freq_edges:
        freq_slice = slice(1, -1)
    elif drop_m0:
        freq_slice = slice(1, None)
    else:
        freq_slice = slice(None)

    return {
        "original_length": original_length_i,
        "Nf": Nf_i,
        "dt": dt_f,
        "wdm_dT": wdm_dT_f,
        "wdm_Nt": Nt,
        "alpha": alpha_f,
        "tukey_factor": tukey_factor_f,
        "window_pad_left": window_pad_left_i,
        "window_pad_right": window_pad_right_i,
        "window_padding_inferred_per_side": inferred_window_pad,
        "taper_margin_samples_per_side": taper_margin_samples,
        "tail_pad_right": tail_pad_right,
        "left_samples_to_cut": total_left_samples,
        "right_samples_to_cut": total_right_samples,
        "left_cut": left_cut,
        "right_cut": right_cut,
        "time_slice": slice(t_start, t_stop),
        "freq_slice": freq_slice,
        "trim_m0": bool(drop_m0),
    }


def apply_wdm_cut(coeffs, cut_info):
    """Apply cut slices returned by ``compute_wdm_cut_info`` to ``(Nt, Nf)`` arrays."""
    arr = np.asarray(coeffs)
    if arr.ndim != 2:
        raise ValueError(f"coeffs must be a 2D array, got shape {arr.shape}.")
    return arr[cut_info["time_slice"], cut_info["freq_slice"]]

# Add your utility functions here

def compare_response_signals(h_ref, h_test):
    """Compute overlap, mismatch, and residual diagnostics between responses."""
    h_ref  = np.asarray(h_ref).reshape(-1)
    h_test = np.asarray(h_test).reshape(-1)

    # L2 norms
    res = h_ref - h_test
    res_L2 = np.linalg.norm(res)
    ref_L2 = np.linalg.norm(h_ref)
    rel_L2 = res_L2 / ref_L2 if ref_L2 != 0.0 else np.nan

    # Overlap / mismatch (unweighted)
    num = np.vdot(h_ref, h_test)
    den = np.linalg.norm(h_ref) * np.linalg.norm(h_test)

    if den == 0.0:
        overlap = np.nan
        mismatch = np.nan
    else:
        overlap = np.abs(num) / den
        overlap = float(np.clip(overlap, 0.0, 1.0))
        mismatch = 1.0 - overlap

    return {
        "res_L2": res_L2,
        "ref_L2": ref_L2,
        "rel_L2": rel_L2,
        "overlap": overlap,
        "mismatch": mismatch,
    }

def wdm_mismatch(w1, w2):
    """
    Robust normalized overlap/mismatch in the WDM domain.

    Parameters
    ----------
    w1 : ndarray
        WDM coefficients 1, usually the reference array.
    w2 : ndarray
        WDM coefficients 2, usually the test array.

    Returns
    -------
    overlap : float
        Phase-maximized normalized overlap between the two WDM coefficient arrays.
        Uses abs(<w1|w2>) so it is robust to an arbitrary constant complex phase.
    mismatch : float
        mismatch = 1 - overlap, with overlap clipped to [0, 1] to remove
        floating-point excursions.
    """
    w1 = np.asarray(w1, dtype=np.complex128).ravel()
    w2 = np.asarray(w2, dtype=np.complex128).ravel()

    n1 = float(np.real(np.vdot(w1, w1)))
    n2 = float(np.real(np.vdot(w2, w2)))

    if n1 <= 0.0 or n2 <= 0.0:
        return np.nan, np.nan

    inner_prod = np.vdot(w1, w2)

    # Phase-maximized normalized overlap.
    overlap_raw = float(np.abs(inner_prod) / np.sqrt(n1 * n2))

    # Remove only mathematically impossible floating-point excursions.
    overlap = float(np.clip(overlap_raw, 0.0, 1.0))
    mismatch = float(1.0 - overlap)

    return overlap, mismatch