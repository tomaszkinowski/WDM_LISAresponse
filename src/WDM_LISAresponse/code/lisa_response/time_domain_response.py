import jax
import jax.numpy as jnp

from typing import Tuple


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





