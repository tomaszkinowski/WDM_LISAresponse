import jax
import jax.numpy as jnp
import numpy as np
import WDM_LISAresponse
from WDM_LISAresponse.code.lisa_response.time_domain_response import basis_tensors, basis_vectors


def test_basis_vectors():
    """
    Check unit vectors are othronormal 
    """
    theta, phi = 1.0, 1.0
    basis = basis_vectors(theta, phi)

    u = basis[0,:]
    v = basis[1,:]
    n = basis[2,:]

    assert np.isclose(np.dot(u,v), 0), "u,v are not orthogonal"
    assert np.isclose(np.dot(u,n), 0), "u,n are not orthogonal"
    assert np.isclose(np.dot(v,n), 0), "v,n are not orthogonal"


def test_basis_vector_functions_are_vectorised():
    """
    Ultimately, we want to be able to compute the LISA reponse for many signals
    in batches, for efficiency on a GPU. Each source will, in general, have its
    own sky location, so we need to be able to compute basis vectors for lots of 
    locations at the same time.

    This unit test checks that the basis_vector function vectorises correctly.
    """
    key = jax.random.PRNGKey(0)

    num_pts = 10

    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(key, shape=(num_pts,), minval=0.0, maxval=jnp.pi)

    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(num_pts,), minval=0.0, maxval=2*jnp.pi)

    basis = basis_vectors(theta, phi)

    assert basis.shape == (3, 3, num_pts), "Basis vectors have incorrect shape"

    # check orthonormality 
    for i in range(3):
        for j in range(i, 3):
            x = jnp.einsum('a...,a...->...', basis[i,...], basis[j,...])
            assert jnp.allclose(x, jnp.where(i==j, 1.0, 0.0)), \
                f"Basis vectors {i}, {j} are not orthonormal"
            
    # check handedness - because n is defined to be outward pointing, this 
    # basis should be left-handed.
    cross = jnp.cross(basis[0,...], basis[1,...], axis=0)
    assert jnp.allclose(cross, -basis[2,...]), \
            "Basis vectors have incorrect handedness"


def test_basis_tensors():
    """
    Check polarisation basis tensors 
    """

    # first check for scalar inputs
    theta = 0.0
    phi = 0.0

    polarisation_basis = basis_tensors(theta, phi)

    assert polarisation_basis.shape == (2, 3, 3), \
        "Polarisation basis tensors have incorrect shape"
    
    e_plus = polarisation_basis[0,...]
    e_cross = polarisation_basis[1,...]

    EPLUS = jnp.array([[1,0,0],[0,-1,0],[0,0,0]])
    ECROSS = jnp.array([[0,-1,0],[-1,0,0],[0,0,0]])

    assert jnp.all(jnp.isclose(e_plus, EPLUS)), \
                    f"{e_plus} is incorrect"
    
    assert jnp.all(jnp.isclose(e_cross, ECROSS)), \
                    f"{e_cross} is incorrect"
    

    # second check for vectorised inputs
    key = jax.random.PRNGKey(0)
    num_pts = 10

    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(key, shape=(num_pts,), minval=0.0, maxval=jnp.pi)

    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(key, shape=(num_pts,), minval=0.0, maxval=2*jnp.pi)

    basis = basis_vectors(theta, phi)
    polarisation_basis = basis_tensors(theta, phi)

    assert polarisation_basis.shape == (2, 3, 3, num_pts), \
        "Polarisation basis tensors have incorrect shape for vectorised inputs"
    
    assert jnp.allclose(jnp.einsum('ii...->', polarisation_basis[0,...]), 0.0),\
        "Plus polarisation basis tensor is not traceless"
    
    assert jnp.allclose(jnp.einsum('ii...->', polarisation_basis[1,...]), 0.0),\
        "Cross polarisation basis tensor is not traceless"
    
    assert jnp.allclose(jnp.einsum('i...,ij...,j...->...', 
                                   basis[2,...], 
                                   polarisation_basis[0,...], 
                                   basis[2,...]), 0.0), \
                "Plus polarisation basis tensor is not transverse"
    
    assert jnp.allclose(jnp.einsum('i...,ij...,j...->...', 
                                   basis[2,...], 
                                   polarisation_basis[1,...], 
                                   basis[2,...]), 0.0), \
                "Cross polarisation basis tensor is not transverse"
    
     

