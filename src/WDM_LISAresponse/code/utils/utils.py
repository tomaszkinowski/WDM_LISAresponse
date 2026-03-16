"""
Utility functions for WDM_LISAresponse
"""
import numpy as np

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
    Inner product in the wdm domain.

    Parameters
    ----------
    w1 : ndarray
        wdm coefficients 1 (Nt x Nf)
    w2 : float
        wdm coefficients 2 (Nt x Nf)

    Returns
    -------
    overlap : float 
        overlap between the two wdm coefficient arrays
    mismatch : float
        mismatch between the two wdm coefficient arrays 
    """

    inner_prod = np.abs(np.einsum("nm,nm->", np.conj(w1), w2))
    w1_mag = np.sqrt(np.einsum("nm,nm->", np.conj(w1), w1))
    w2_mag = np.sqrt(np.einsum("nm,nm->", np.conj(w2), w2))
    overlap = inner_prod / (w1_mag * w2_mag)
    mismatch = 1 - overlap
    
    return overlap, mismatch