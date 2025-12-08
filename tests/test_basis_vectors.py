import numpy as np
import WDM_LISAresponse
from WDM_LISAresponse.code.lisa_response.time_domain_response import basis_vectors

def test_basis_vectors():
    """
    Check unit vectors are othronormal 
    """
    theta = 1.0
    phi = 1.0

    u, v, n = basis_vectors(theta, phi)
    assert np.isclose(np.dot(u,v), 0), "u,v are not orthogonal"
    assert np.isclose(np.dot(u,n), 0), "u,n are not orthogonal"
    assert np.isclose(np.dot(v,n), 0), "v,n are not orthogonal"
