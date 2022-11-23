r"""
.. _implicit_diff_susceptibility:

Implicit differentiation of variational quantum algorithms
==========================================================

.. meta::
   :property="og:description": Implicitly differentiating the the solution of a VQA in PennyLane.
   :property="og:image": https://github.com/PennyLaneAI/qml/tree/master/demonstrations/implicit_diff/descartes.png

.. related::


*Author: Shahnawaz Ahmed — Posted: 21 Nov 2022. Last updated: 21 Nov 2022.*

*Email: shahnawaz.ahmed95@gmail.com*

René Descartes apparently challenged Pierre de Fermat to find the tangent to
a complicated curve since he was intrigued by (the then amateur) Fermat's method
of computing tangents. The curve, now called the folium of Descartes given by

.. math:: 

    x^3 + y^3 = 3axy.

This curve with the cubic terms represents an implicit equation where it is not
easy to write it simply as :math:`y = f(x)`. Therefore computing the tangent
seemed formidable with the method Descartes had then except for the vertex. 
Fermat provided the tangents at not just the vertex but at any other point on
the curve baffling Descartes and legitimizing the intellectual superiority of
Fermat. The technique used by Fermat was implicit differentiation [1]. In the 
above equation, we can just take derivatives of both sides of the equation
and re-arrange the terms to obtain :math:`dy\dx`.

Implicit differentiation can be used to compute gradients of such functions that
cannot be written down explicitly using simple elementary operations. It is a
simple technique from calculus that has found many applications in machine
learning recently - from hyperparameter optimization to training neural ordinary
differential equations and even defining a whole class of new architectures
called Deep Equilibrium Models (DEQs) [2].

The idea of implicit differentiation can be applied in quantum physics to extend
the power of automatic differentiation to situations where we are not able to
explicitly write down the solution to a problem. As a concrete example, consider
a variational quantum algorithm (VQA) that computes the ground-state solution of
a parameterized Hamiltonian :math:`H(a)` using a variational ansatz
:math:`|\psi_{z}\rangle` where :math:`z` are the variational parameters such
that we have the solution

.. math::

    z^{*}(a) = \arg\,\min_{z} \langle \psi_{z}|H(a)|\psi_z\rangle.

The solution changes as we change :math:`H(a)` therefore defining an implicit
solution function :math:`z^{*}(a)` If we are interested in properties of the
solution state, we could use measure expectation values for some operator
:math:`A` as

.. math::

    \langle A \rangle (a) = \langle \psi_{z^{*}(a)}| A | \psi_{z^{*}(a)}\rangle.

With a VQA, we can find a solution to the optimization for a fixed :math:`H(a)`
However, just like the folium of Descartes, we do not have an explicit solution
so the gradient :math:`\partial_a \langle A \rangle (a)` is not easy to compute.

Automatic differentiation techniques that construct an explicit computational
graph and differentiate through it by applying the chain rule for gradient
computation cannot be applied here easily. A brute-force application of automatic
differentiation through the full optimization that finds :math:`z^{*}(a)`
would require keeping track of all intermediate variables and steps in the optimization
and differentiating through them. This could quickly become computationally
expensive and memory intensive for quantum algorithms. Implicit differentiation
provides an alternative way to efficiently compute such a gradient. 

:math:`\partial_{a} \langle A\rangle` is the so-called 
*generalized susceptibility* arising from condensed-matter physics. The computation
of this quantity would otherwise require tedious analytical derivations or
finite difference approximations. Similarly there exist various other
interesting quantities written as gradients of a ground-state solution, 
e.g., nuclear forces in quantum chemistry, permanent electric dipolestatic
polarizability, the static hyperpolarizabilities of various orders,
fidelity susceptibilities, and geometric tensors. All such
quantities can possibly be computed using implicit differentiation on quantum
devices.

In this demo, we will show how to compute implicit gradients through a variational
algorithm written in PennyLane using a modular implicit differentiation
implementation provided by the tool JAXOpt []. We compute the generalized
susceptibility for a spin system by using a variational ansatz to compute a
ground-state and implicitly differentiating through it. In order to compare
the implicit solution, we will find the exact ground-state through eigendecomposition
and take gradients through the eigendecomposition using automatic differentiation.
Even though for the small number of spins we consider here, eigendecompostion and
gradient computation through it suffices, for larger systems it quickly becomes
infeasible.


Implicit Differentiation
------------------------

We consider differentiating a solution of the root-finding problem defined by

.. math::

    f(z, a) = 0.

A function :math:`z^{*}(a)` that satisfies :math:`f(z^{*}(a), a) = 0` gives a
solution map for fixed values of :math:`a`. An explicit analytical solution
is however difficult to obtain in general. Therefore to differentiate and obtain
:math:`\partial_a z^{*}(a)` is not always possible directly. However, some iterative
algorithm could compute the solution starting from an initial set of values
for :math:`z`, e.g., using a fixed-point solver. The optimality condition 
:math:`f(z^{*}(a), a) = 0` tells the solver when a solution is found.
A brute-force application of automatic differentiation through
the full iterative solver would require keeping track of all the steps in solving
the fixed-point problem. This can quickly make computing gradients memory
intensive.

Implicit differentiation can
compute :math:`\partial_a z^{*}(a)` more efficiently than brute-force automatic
differentiation, using only the solution :math:`z^{*}(a)` 
and partial derivatives at the solution point. We do not have to care about how
the solution is obtained and therefore do not need to differentiate through the
solution-finding algorithm.


.. topic:: Implicit function theorem (IFT) (informal)

    If :math:`f(z, a)` is some analytic function where in a local neighbourhood
    around :math:`(z_0, a_0)` we have :math:`f(z_0, a_0) = 0`, there exists an
    analytic solution :math:`z^{*}(a)` that satisfies :math:`f(z^{*}(a), a) = 0`.


.. figure:: ../demonstrations/implicit_diff/implicit_diff.png
   :scale: 65%
   :alt: circles

In the figure above, we can see solutions to the optimality condition 
:math:`f(z, a) = 0 ` (red stars) that defines a curve :math:`z^{*}(a)`. 
The IFT says that the solution function is analytic, and therefore it can be
differentiated at the solution points simply by differentiating
the above equation with respect to :math:`a` as,

.. math::
    
    \partial_a f(z_0, a_0) + \partial_{z} f(z_0, a_0) \partial_{a} z^{*}(a) = 0

which leads to

.. math::

    \partial_{a} z^{*}(a) = - (\partial_{z} f(z_0, a_0) )^{-1}\partial_a f(z_0, a_0).


Implicit differentiation can therefore be used in situations where we can
phrase our optimization problem in terms of an optimality condition, 
or a fixed point equation, that can be solved. In case of optimization tasks,
such an optimality condition would be that at the minima, the gradient of the
cost function is zero, i.e., 

.. math::

    f(z, a) = \partial_z g(z, a) = 0.

Then, as long as we have the solution :math:`z^{*}(a)`, and the partial derivatives 
:math:`(\partial_a f, \partial_z f)` at the solution (here the Hessian of the
cost function :math:`g(z, a)`), we can compute implicit gradients. Note that
for a multivariate function that inversion
:math:`(\partial_{z} f(z_0, a_0) )^{-1}` needs to be defined and easy to
compute. It is possible to approximate this inversion in a clever way by constructing
a linear problem that can be solved approximately; [1] [2].


Implicit differentiation through a variational quantum algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us take a parameterized Hamiltonian :math:`H(a)`, where :math:`a` is a parameter that can
be continuously varied. If :math:`\psi_{z}` is a variational solution
to the ground state of :math:`H(a)`, then we can find a :math:`z^*(a)` that minimizes the ground
state energy, i.e.,

.. math::
    
    z^*(a) = \arg\, \min_{z} \langle \psi_{z}| H(a) | \psi_{z}\rangle = \arg, \min_{z} E(z, a)


where :math:`E(z, a)` is the energy function. We consider the following Hamiltonian

.. math::
    
    H(a) = -J \sum_{i}^{N-1} \sigma^{z}_i \sigma^{z}_{i+1} - \gamma \sum_{i}^{N} \sigma^{x}_i - a A + \delta \sum_i \sigma^z_i,


where :math:`J` is the interaction, :math:`\sigma_{x, z}` are the spin-:math:`\frac{1}{2}`
operators, :math:`\gamma` is the magnetic field strength (which is taken to be
the same for all spins). The term :math:`A = \frac{1}{N}\sum_i^{i=N} \sigma^{z}_i`
is the magnetization and a small non-zero magnetization
:math:`\delta \sum_i \sigma^z_i` is added for numerical stability.
We have assumed a circular chain such that in the interaction term the last spin
(:math:`i=N-1`) interacts with the first (:math:`i=0`). 

Now we could find the ground state of this Hamiltonian simply by taking the 
eigendecomposition and applying automatic differentiation through the 
eigendecomposition to compute gradients.
We will compare this exact computation for a small system to the gradients given
by implicit differentiation through a variationally obtained solution.

We define the following optimality condition at the solution point:

.. math::
    
    f(z, a) = \nabla_z E(z, a) = 0.


In addition, if the conditions of the implicit function theorem
are also satisfied, i.e., :math:`f` is continuously differentiable
with non-singular Jacobian at the solution, then we can apply chain rule
and determine the implicit gradients easily.

Now, any other complicated function that depends on the variational ground state
can be easily differentiated using automatic differentiation by plugging in the
value of :math:`partial_a z^{*}(a)` where it is required. The 
expectation value of the operator :math:`A` for the ground state
is

.. math::
    
    \langle A\rangle = \langle \psi_{z^*}| A| \psi_{z*}\rangle.



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

##############################################################################
# Defining the Hamiltonian
# ------------------------
# We define the Hamiltonian by building the non-parametric part separately and
# adding the parametric part later.
#
##############################################################################

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
    through, computed from an eigendecomposition.

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
# Computing susceptibility through implicit differentiation
###########################################################################

# We use PennyLane to find a variational ground state for the Hamiltonian H(a)
# and compute implicit gradients through the variational optimization procedure.
# The `jaxopt` library contains an implementation of gradient descent that
# automatically comes with implicit differentiation capabilities.
# We are going to use that to obtain susceptibility
# by taking gradients through the ground-state minimization.

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
# Define the Hamiltonian and variational state
#####################################################

variational_ansatz = qml.SimplifiedTwoDesign
n_layers = 5
weights_shape = variational_ansatz.shape(n_layers, N)

# Note that `shots=None` makes the computation of gradients using reverse-mode
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
# us with the implicit gradients. In a seperate demo we will implement the implicit gradient
# computation ourselves but for this demo, we use the excellent intergration between Pennylane, Jax
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
# [1] Jaume Paradís, Josep Pla & Pelegrí Viader (2004) Fermat and the Quadrature
# of the Folium of Descartes, The American Mathematical Monthly, 111:3, 
# 216-229, DOI: 10.1080/00029890.2004.11920067
# [2] http://implicit-layers-tutorial.org
# [1] Ahmed, S., Killoran, N., Carrasquilla Álvarez J. F. "Implicit differentiation
# of variational quantum algorithms." arXiv preprint arXiv:2022.XXXX (2022).
#
# [2] Blondel, Mathieu, et al. "Efficient and modular implicit
# differentiation." arXiv preprint arXiv:2105.15183 (2021).
#
# About the author
# ----------------
# .. include:: ../_static/authors/shahnawaz_ahmed.txt
