r"""
.. _implicit_diff_susceptibility:

Implicit differentiation of variational quantum algorithms
==========================================================

.. meta::
   :property="og:description": Implicitly differentiating the the solution of a VQA in PennyLane.
   :property="og:image": https://github.com/PennyLaneAI/qml/tree/master/demonstrations/implicit_diff/implicit.png

.. related::


*Author: Shahnawaz Ahmed — Posted: 21 Nov 2022. Last updated: 21 Nov 2022.*

*Email: shahnawaz.ahmed95@gmail.com*

Implicit differentiation can be used to compute gradients of a function that
cannot be written down explicitly using simple elementary operations. In this
notebook we use the idea of implicit differentiation to compute gradients
through a variational quantum algorithm (VQA) that finds groudstate solutions to
a parameterized Hamiltonian :math:`H(a)` using a variational state :math:`|\psi_{z}\rangle`
such that

.. math::

    z^{*}(a) = \arg\,\min_{z} \langle \psi_{z}|H(a)|\psi_z\rangle.

We are interested in computing the gradients :math:`\partial_a z^{*}(a)` that
can also allow us to compute other quantities that depend on this gradient such
as a generalized susceptibility,

.. math::

    \partial_a \langle A \rangle = \partial_a \langle \psi_{z^{*}(a)} | A |\psi_{z^{*}(a)} \rangle.

Since a brute-force application of automatic differentiation through the full
optimization task would require keeping track of all intermediate variables and
steps, it could be computationally expensive.

Implicit differentiation provides a way to efficiently compute gradients
through such problems. In this tutorial, we will compute implicit
gradients through a variational algorithm written in PennyLane using the
modular implicit differentiation tool JAXOpt.

Background
----------

We consider differentiating the solution of an optimization problem :math:`z^{*}(a)`
given by

.. math::

    z^{*}(a) = \arg\,\min_{z} g(z, a).

We may not have an explicit analytical solution for :math:`z^{*}(a)` to
differentiate and obtain :math:`\partial_a z^{*}(a)`. However, some iterative
algorithm could compute the solution starting from an initial set of values
for :math:`z`, e.g., using gradient-based optimization of the function 
:math:`g(z, a)`. A brute-force application of automatic differentiation through
the full optimization algorithm would require keeping track of all the steps 
in the optimization and becomes memory intensive. Implicit differentiation can
compute :math:`\partial_a z^{*}(a)` more efficiently that brute force automatic
differentiation using only the solution :math:`\partial_a z^{*}(a)` 
and partial derivatives at the solution point.

The main idea is that for some analytic function :math:`f(z, a)`, if in some
local neighbourhood around :math:`(z_0, a_0)` we have :math:`f(z_0, a_0) = 0`
(points marked by the red stars below), there exists an analytic solution
:math:`z^{*}(a)` (red line) that satisfies :math:`f(z^{*}(a), a) = 0`.
    
    
.. figure:: ../demonstrations/implicit_diff/implicit_diff.png
   :scale: 65%
   :alt: circles

Since the solution function is analytic, it can be
differentiated at :math:`(z_0, a_0)` simply by differentiating the above equation
w.r.t :math:`a` as,

.. math::
    
    \partial_a f(z_0, a_0) + \partial_{z} f(z_0, a_0) \partial_{a} z^{*}(a) &=& 0 \\

    \partial_{a} z^{*}(a) &=& - (\partial_{z} f(z_0, a_0) )^{-1}\partial_a f(z_0, a_0)


Therefore we need to find a so-called optimality condition (or fixed-point equation)
that can be solved. In case of optimization tasks, such an optimality condition
would be that at the minima, the gradient of the cost function is zero,

.. math::

    f(z, a) = \partial_z g(z, a) = 0.

Then, as long as we have the solution :math:`z^{*}(a)`, the partial derivatives 
:math:`(\partial_a f, \partial_z f)` at the solution (here the Hessian of the cost function :math:`g(z, a)`) 
we can compute implicit gradients. Note that for a multivariate function that inversion
:math:`(\partial_{z} f(z_0, a_0) )^{-1}` needs to be defined and easy to
compute. It is possible to approximate this inversion in a clever way by constructing
a linear problem that can be solved approximately, see [1] [2].


Implicit differentiation through a variational quantum algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us take a parameterized Hamiltonian $H(a)$ where $a$ is a parameter that can
be continuously varied. If $\psi_{z}$ is a variational approximation
to the ground state of $H(a)$, then we can find a $z^*(a)$ that minimizes the ground
state energy, i.e.,

.. math::
    
    z^*(a) = \argmin_{z} \langle \psi_{z}| H(a) | \psi_{z}\rangle = \argmin_{z} E(z, a)


where :math:`E(z, a)` is the energy function. We consider the following Hamiltonian

.. math::
    
    H(a) = -J \sum_{i}^{N-1} \sigma^{z}_i \sigma^{z}_{i+1} - \gamma \sum_{i}^{N} \sigma^{x}_i - a H_1 + \delta \sum_i \sigma^z_i


where :math:`J` is the interaction, :math:`\sigma_{x, z}` are the spin 1/2 operators, 
:math:`\gamma` is the magnetic field strength (which is taken to be the same for all spins).
The term :math:`H_1 = M = \frac{1}{N}\sum_i^{i=N} \sigma^{z}_i` is the magnetization and a small non-zero 
magnetization :math:`\delta \sum_i \sigma^z_i` is added for numerical stability.
We have assumed a circular chain such that in the interaction term the last spin
(:math:`i=-1`) interacts with the first (:math:`i=0`). 

Now we can find the ground state of this Hamiltonian simply by eigendecomposition and apply 
automatic differentiation (AD) through the eigendecomposition to compute gradients.
We will compare this exact computation for a small system to the gradients given
by implicit differentiation through a variational ansatz for the wave function 
:math:`\psi_{z}`.

We define the following optimality condition at the solution point

.. math::
    
    f(z, a) = \nabla_z E(z, a) = 0.


In addition, if the conditions of the implicit function theorem
are also satisfied, i.e., :math:`f` is continuously differentiable
with non-singular Jacobian at the solution we can apply chain rule
and determine the implicit gradients easily.

Now, any other complicated function that depends on the variational ground state
can be easily differentiated using automatic differentiation through the argmin
solver. If $A$ is an operator such that its expectation value for the ground state
is

.. math::
    
    \langle A\rangle = \langle \psi_{z}| A| \psi_{z}\rangle.

The quantity :math:`\partial_{a} \langle A\rangle` is called generalized
susceptibility. In a similar way, we can now take implicit gradients through
any variational quantum algorithm and compute interesting quantities that
are written as gradients w.r.t. the ground state e.g., nuclear forces in
quantum chemistry, permanent electric dipolestatic polarizability,
the static hyperpolarizabilities of various orders, generalized susceptibilities,
fidelity susceptibilities, and geometric tensors.


"Talk is cheap. Show me the code." - Linus Torvalds
---------------------------------------------------
"""
from functools import reduce

from operator import add

import jax
from jax import jit
import jax.numpy as jnp
from jax.config import config

import pennylane as qml
import numpy as np

import jaxopt

import matplotlib.pyplot as plt

# Use double precision numbers
config.update("jax_enable_x64", True)

##########################
# Defining the Hamiltonian
##########################

N = 5
J = 1.0
gamma = 1.0


def build_H0(N, J, gamma):
    """Builds the non-parametric part of the Hamiltonian of a spin system.

    Args:
        N (int): Number of qubits/spins.
        J (float): Interaction strength.
        gamma (float): Interaction strength.

    Returns:
        qml.Hamiltonian: The Hamiltonian of the system.
    """
    H = qml.Hamiltonian([], [])

    for i in range(N - 1):
        H += -J * qml.PauliZ(i) @ qml.PauliZ(i + 1)

    H += -J * qml.PauliZ(N - 1) @ qml.PauliZ(0)

    # Transverse
    for i in range(N):
        H += -gamma * qml.PauliX(i)

    # Small magnetization for numerical stability
    for i in range(N):
        H += -1e-1 * qml.PauliZ(i)

    return H


# We build H0 using PennyLane and convert it into a matrix
H0 = build_H0(N, J, gamma)
H0_matrix = qml.matrix(H0)

###################################
# Defining the measurement operator
###################################

A = reduce(add, ((1 / N) * qml.PauliZ(i) for i in range(N)))
A_matrix = qml.matrix(A)


#############################################################
# Computing the exact ground state through eigendecomposition
#############################################################


@jit
def ground_state_solution_map_exact(a: float) -> jnp.array:
    """The ground state solution map that we want to differentiate
    through computed from an eigendecomposition.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).

    Returns:
        jnp.array: The ground state solution for the H(a).
    """
    H = H0_matrix + a * A_matrix
    eval, eigenstates = jnp.linalg.eigh(H)
    z_star = eigenstates[:, 0]
    return z_star


a = jnp.array(np.random.uniform(0, 1.0))
z_star_exact = ground_state_solution_map_exact(a)

#################################################################
# Suceptibility computation through the ground state solution map
#################################################################

# Let us now compute the susceptibility function by taking gradients of the expectation value
# of our operator A w.r.t a. We can use `jax.vmap` to vectorize the computation
# over different values of `a`.


@jit
def expval_A_exact(a):
    """Expectation value of $A$ as a function of $a$ where we use the
    ground_state_solution_map_exact function to find the ground state.

    Args:
        a (float): The parameter defining the Hamiltonian, H(a).

    Returns:
        float: The expectation value of A calculated using the variational state
               that should be the ground state of H(a).
    """
    z_star = ground_state_solution_map_exact(a)
    eval = jnp.conj(z_star.T) @ A_matrix @ z_star
    return eval.real


# We vectorize the whole computation by defining the susceptibility as the
# gradient of the expectation value and then vectorizing this with jax.vmap
_susceptibility_exact = jax.grad(expval_A_exact)
susceptibility_exact = jax.vmap(_susceptibility_exact)

alist = jnp.linspace(0, 3, 300)
susvals_exact = susceptibility_exact(alist)

plt.plot(alist, susvals_exact)
plt.xlabel("a")
plt.ylabel(r"$\partial_{a}\langle A \rangle$")
plt.show()

###########################################################################
# Computing susceptibility through implicit differentiation using PennyLane
###########################################################################

# We use PennyLane to find a variational ground state for the Hamiltonian H(a)
# and compute implicit gradients through the variational optimization procedure.
# The `jaxopt` library contains an implementation of gradient descent that
# automatically comes with implicit differentiation capabilities.
# We are going to use that to obtain susceptibility
# by taking gradients through the ground state minimization.

"""
.. figure:: ../demonstrations/implicit_diff/VQA.png
   :scale: 65%
   :alt: circles
"""

# In PennyLane, we can implement a variational state in different ways by
# defining a quantum circuit. There are also template circuits available such as
# `SimplifiedTwoDesign` that implements the two-design ansatz from Cerezo et al. 2021.
# The ansatz consists of layers consisting of Pauli-Y rotations with
# controlled-Z gates. In each layer there are `N - 1` parameters for Pauli-Y gates.
# Therefore the ansatz is efficient and as long as it is expressive enough to
# represent the ground-state, it can be an efficient approach to compute quantities
# such as susceptibilities.

#####################################################
# Define the Hamiltonian and variational wavefunction
#####################################################

variational_ansatz = qml.SimplifiedTwoDesign
n_layers = 5
weights_shape = variational_ansatz.shape(n_layers, N)

# Note that `shots=None` makes the computation of gradients using reverse mode
# autodifferentiation (backpropagation). It allows us to just-in-time (JIT)
# compile the functions that compute expectation values and gradients.
# In a real device we will have finite shots and the gradients are computed
# using the parameter-shift rule. However this may be slower.

dev = qml.device("default.qubit.jax", wires=N, shots=None)  # This is good ol backprop

# We use a second device to compute the expectation and take gradients to
# compute susceptibilities.

dev2 = qml.device("default.qubit.jax", wires=N, shots=None)  # This is good ol backprop


@jax.jit
@qml.qnode(dev, interface="jax")
def energy(z, a):
    """Computes the energy for a Hamiltonian H(a) using a measurement on the
    variational state U(z)|0> with U(z) being any circuit ansatz.

    Args:
        z (jnp.array): The variational parameters for the ansatz (circuit)
        a (jnp.array): The Hamiltonian parameters.

    Returns:
        float: The expectation value (energy).
    """
    variational_ansatz(*z, wires=range(N))
    # here, we compute the Hamiltonian coefficients and operations
    # 'by hand' because the qml.Hamiltonian class does not support
    # operator arithmetic with JAX device arrays.
    coeffs = jnp.concatenate([H0.coeffs, a * A.coeffs])
    return qml.expval(qml.Hamiltonian(coeffs, H0.ops + A.ops))


z_init = [jnp.array(2 * np.pi * np.random.random(s)) for s in weights_shape]
a = jnp.array([0.5])

# Compute ground state using gradient descent with `JAXOpt`
@jax.jit
def ground_state_solution_map_variational(a, z_init):
    """The ground state solution map that we want to differentiate
    through.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).
        z_init [jnp.array(jnp.float)]: The initial guess for the variational parameters.

    Returns:
        z_star (jnp.array [jnp.float]): The parameters that define the ground state solution.
    """

    @jax.jit
    def loss(z, a):
        """Loss function for the ground-state minimization with regularization.

        Args:
            z (jnp.array): The variational parameters for the ansatz (circuit)
            a (jnp.array): The Hamiltonian parameters.

        Returns:
            float: The loss value (energy + regularization)
        """

        return (
            energy(z, a)
            + 0.001 * jnp.sum(jnp.abs(z[0]))
            + 0.001 * jnp.sum(jnp.abs(z[1]))
        )

    gd = jaxopt.GradientDescent(
        fun=loss,
        stepsize=1e-2,
        acceleration=True,
        maxiter=1000,
        implicit_diff=True,
        tol=1e-15,
    )
    z_star = gd.run(z_init, a=a).params
    return z_star


# External operator M
a = jnp.array(np.random.uniform(0, 1.0))  # A random a
z_star_variational = ground_state_solution_map_variational(a, z_init)

########################################################################
# Compute the susceptibility by differentiating through gradient descent
########################################################################

# It all works due to Pennylane's excellent Jax integration. The implicit differentiation formulas
# that `jaxopt` provides can leverage VJP calculations using the Pennylane Jax interface and provide
# us with the implicit gradients. In a seperate notebook we will implement the implicit gradient
# computation ourselves but for this notebook, we use the excellent intergration between Pennylane, Jax
# and Jaxopt to compute the susceptibility.

# We first define a second quantum node that applies learned variational parameters to a new circuit and
# then computes the expectation for our operator `<M>`. Then we put it all together within one function
# that computes a ground state and then computes the expectation `<M>` and finally take gradients.


@jax.jit
@qml.qnode(dev2, interface="jax")
def expval_A_variational(z: float) -> float:
    """Expectation value of $A$ as a function of $a$ where we use the
    a variational ground state solution map.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).

    Returns:
        float: The expectation value of M on the ground state of H(a)
    """
    variational_ansatz(*z, wires=range(N))
    return qml.expval(A)


@jax.jit
def groundstate_expval_variational(a, z_init) -> float:
    """Computes ground state and calculates the expectation value of the operator M.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).
        z_init [jnp.array(jnp.float)]: The initial guess for the variational parameters.
        H0 (qml.Hamiltonian): The static part of the Hamiltonian

    """
    z_star = ground_state_solution_map_variational(a, z_init)
    return expval_A_variational(z_star)


susceptibility_variational = jax.jit(
    jax.grad(groundstate_expval_variational, argnums=0)
)
susvals_variational = []


for i in range(len(alist)):
    z_init = [jnp.array(2 * np.pi * np.random.random(s)) for s in weights_shape]
    susvals_variational.append(susceptibility_variational(alist[i], z_init))


plt.plot(alist, susvals_variational, label="Implicit diff through VQA")
plt.plot(alist, susvals_exact, "--", c="k", label="Autodiff through eigendecomposition")
plt.xlabel("a")
plt.ylabel(r"$\partial_{a}\langle A \rangle$")
plt.legend()
plt.show()

##############################################################################
# References
# ----------
# [1] Ahmed, S., Killoran, K., Carrasquilla Álvarez J. F. "Implicit differentiation
# of variational quantum algorithms." arXiv preprint arXiv:2022.XXXX (2022).
#
# [2] Blondel, Mathieu, et al. "Efficient and modular implicit
# differentiation." arXiv preprint arXiv:2105.15183 (2021).
#
# About the author
# ----------------
# .. include:: ../_static/authors/shahnawaz_ahmed.txt
