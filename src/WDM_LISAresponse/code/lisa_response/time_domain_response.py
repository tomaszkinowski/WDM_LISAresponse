import jax
import jax.numpy as jnp

from jaxtyping import Array


@jax.jit
def basis_vectors(theta : Array, 
                  phi : Array) -> Array:
    r"""
    Creates an orthonormal triad of spatial basis vectors (Carteisian coords) 
    :math:`\{\hat{u}, \hat{v}, \hat{n}\}` at a point on the sphere of the sky 
    specified by the spherical polar angles :math:`(\theta, \phi)`;

    .. math::
        \begin{align}
            \hat{u} &= \cos\theta\cos\phi \hat{x} + 
                        \cos\theta\sin\phi \hat{y} - 
                        \sin\theta \hat{z} \\
            \hat{v} &= \sin\phi \hat{x} - 
                        \cos\phi \hat{y} \\
            \hat{n} &= \sin\theta\cos\phi \hat{x} + 
                        \sin\theta\sin\phi \hat{y} + 
                        \cos\theta \hat{z}
        \end{align} .

    The radial unit vector :math:`\hat{n}` points towards the GW source.

    Parameters
    ----------
    theta : Array
        Polar angle [rad], either a scalar JAX array (shape=()), or an array of 
        arbitrary shape=(...,).
    phi : Array
        Azimuthal angle [rad], either a scalar JAX array (shape=()), or an array 
        of arbitrary shape=(...,).

    Returns
    -------
    basis : Array
        The three orthonormal basis vectors. If inputs are scalars, returns an 
        array of shape=(3,3) with the first axis indexing the basis vectors
        :math:`\hat{u}, \hat{v}, \hat{n}` and the second axis labelling the
        Cartesian components. If inputs are arrays of shape=(...,), returns an 
        array of shape=(3,3,...).
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)

    basis = jnp.array([
                        [jnp.cos(theta)*jnp.cos(phi), 
                         jnp.cos(theta)*jnp.sin(phi), 
                         -jnp.sin(theta)]
                         ,
                         [jnp.sin(phi), 
                          -jnp.cos(phi), 
                          jnp.zeros_like(theta)]
                         ,
                         [jnp.sin(theta)*jnp.cos(phi), 
                          jnp.sin(theta)*jnp.sin(phi), 
                          jnp.cos(theta)]
                    ])
    
    return basis


@jax.jit
def basis_tangent_vectors(theta : Array, 
                          phi : Array) -> Array:
    r"""
    Similar to `basis_vectors`, but only returns the two basis vectors in the 
    tangent plane of the sphere of the sky, :math:`\{\hat{u}, \hat{v}\}`. This
    function does not compute the radial unit vector :math:`\hat{n}`.

    Creates an orthonormal pair of spatial basis vectors (Carteisian coords) 
    :math:`\{\hat{u}, \hat{v}\}` at a point on the sphere of the sky 
    specified by the spherical polar angles :math:`(\theta, \phi)` in the 
    tangent plane of the sphere of the sky;

    .. math::
        \begin{align}
            \hat{u} &= \cos\theta\cos\phi \hat{x} + 
                        \cos\theta\sin\phi \hat{y} - 
                        \sin\theta \hat{z} \\
            \hat{v} &= \sin\phi \hat{x} - 
                        \cos\phi \hat{y} 
        \end{align} .

    Parameters
    ----------
    theta : Array
        Polar angle [rad], either a scalar JAX array (shape=()), or an array of 
        arbitrary shape=(...,).
    phi : Array
        Azimuthal angle [rad], either a scalar JAX array (shape=()), or an array 
        of arbitrary shape=(...,).

    Returns
    -------
    tangent_basis : Array
        The two orthonormal basis vectors. If inputs are scalars, returns an 
        array of shape=(2,3) with the first axis indexing the basis vectors
        :math:`\hat{u}, \hat{v}` and the second axis labelling the Cartesian 
        components. If inputs are arrays of shape=(...,), returns an array of 
        shape=(3,3,...).
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)

    tangent_basis = jnp.array([
                        [jnp.cos(theta)*jnp.cos(phi), 
                         jnp.cos(theta)*jnp.sin(phi), 
                         -jnp.sin(theta)]
                         ,
                         [jnp.sin(phi), 
                          -jnp.cos(phi), 
                          jnp.zeros_like(theta)]
                    ])
    
    return tangent_basis


@jax.jit
def basis_tensors(theta : Array, 
                  phi : Array) -> Array:
    r"""
    Creates the two transverse-traceless GW polarisation basis tensors 
    :math:`e^+` and :math:`e^\times` at a point on the sphere of the sky 
    specified by the spherical polar angles :math:`(\theta, \phi)`;

    .. math::
        \begin{align}
            e^+ &= \hat{u}\otimes\hat{u} - \hat{v}\otimes\hat{v} \\
            e^\times &= \hat{u}\otimes\hat{v} + \hat{v}\otimes\hat{u} 
        \end{align} .

    Parameters
    ----------
    theta : Array
        Polar angle [rad], either a scalar JAX array (shape=()), or an array of 
        arbitrary shape=(...,).
    phi : Array
        Azimuthal angle [rad], either a scalar JAX array (shape=()), or an array 
        of arbitrary shape=(...,).

    Returns
    -------
    polarisarion_basis : Array
        Plus and cross polarisation basis tensors. If inputs are scalars, 
        returns an array of shape=(2,3,3) with the first axis indexing the 
        basis tensors :math:`e^+` and :math:`e^\times` and the second and third
        axes labelling the Cartesian components. If inputs are arrays of 
        shape=(...,), returns an array of shape=(2,3,3,...).
    """
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)

    tangent_basis = basis_tangent_vectors(theta, phi)

    polarisarion_basis = jnp.array([
                            jnp.einsum('i...,j...->ij...', 
                                        tangent_basis[0,...], 
                                        tangent_basis[0,...]) - \
                            jnp.einsum('i...,j...->ij...', 
                                        tangent_basis[1,...], 
                                        tangent_basis[1 ,...])
                            ,
                            jnp.einsum('i...,j...->ij...', 
                                        tangent_basis[0,...], 
                                        tangent_basis[1,...]) + \
                            jnp.einsum('i...,j...->ij...', 
                                        tangent_basis[1,...], 
                                        tangent_basis[0 ,...])
                            ])

    return polarisarion_basis

