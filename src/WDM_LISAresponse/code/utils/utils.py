"""
Utility functions for WDM_LISAresponse
"""
import numpy as np

# Add your utility functions here

def compare_response_signals(h_ref, h_test):
    """
    h_ref  : reference signal (e.g. time-domain)
    h_test : signal to compare (e.g. FT or WDM)

    Returns
    -------
    result : dict with
      res_L2      : L2 norm of residual ||h_ref - h_test||
      ref_L2      : L2 norm of reference ||h_ref||
      rel_L2      : relative L2 error = res_L2 / ref_L2
      overlap     : normalized overlap (GW-style)
      mismatch    : 1 - overlap
    """
    h_ref  = np.asarray(h_ref)
    h_test = np.asarray(h_test)
  
    # L2 norms
    res = h_ref - h_test
    res_L2 = np.linalg.norm(res)
    ref_L2 = np.linalg.norm(h_ref) + 1e-30
    rel_L2 = res_L2 / ref_L2

    # GW-style overlap / mismatch (without scaling)
    num_ov = np.vdot(h_ref, h_test)
    den_ov = np.linalg.norm(h_ref) * np.linalg.norm(h_test) + 1e-30
    overlap = np.abs(num_ov) / den_ov
    mismatch = 1.0 - overlap

    return {
        "res_L2": res_L2,
        "ref_L2": ref_L2,
        "rel_L2": rel_L2,
        "overlap": overlap,
        "mismatch": mismatch,
    }