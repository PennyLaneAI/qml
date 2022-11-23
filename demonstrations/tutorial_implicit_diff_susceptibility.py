r"""
.. _implicit_diff_susceptibility:

Implicit differentiation of variational quantum algorithms
==========================================================

.. meta::
   :property="og:description": Implicitly differentiating the the solution of a VQA in PennyLane.
   :property="og:image": https://github.com/PennyLaneAI/qml/tree/master/demonstrations/implicit_diff/descartes.png

.. related::


*Authors: Shahnawaz Ahmed, Juan Felipe Carrasquilla Álvarez. 
— Posted: 21 Nov 2022. Last updated: 21 Nov 2022.*
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
Fermat. The technique used by Fermat was implicit differentiation [#Paradis2004]_.
In the above equation, we can just take derivatives of both sides of the equation
and re-arrange the terms to obtain :math:`dy\dx`.

Implicit differentiation can be used to compute gradients of such functions that
cannot be written down explicitly using simple elementary operations. It is a
simple technique from calculus that has found many applications in machine
learning recently - from hyperparameter optimization to training neural ordinary
differential equations (ODEs) and even defining a whole class of new architectures
called Deep Equilibrium Models (DEQs) [#implicitlayers]_.

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
quantities could possibly be computed using implicit differentiation on quantum
devices. In our recent work, we present a unified way to implement such
computations and other applications of implicit differentiation through
variational quantum algorithms [#Ahmed2022]_.

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

Implicit differentiation can compute :math:`\partial_a z^{*}(a)`
more efficiently than brute-force automatic differentiation, using only the
solution :math:`z^{*}(a)` and partial derivatives at the solution point. We do
not have to care about how the solution is obtained and therefore do not need
to differentiate through the solution-finding algorithm [#Blondel2021]_.


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
a linear problem that can be solved approximately [#Blondel2021]_, [#implicitlayers]_.


Implicit differentiation through a variational quantum algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../demonstrations/implicit_diff/VQA.png
   :scale: 65%
   :alt: circles

Let us take a parameterized Hamiltonian :math:`H(a)`, where :math:`a` is a
parameter that can be continuously varied. If :math:`\psi_{z}` is a variational
solution to the ground state of :math:`H(a)`, then we can find a :math:`z^*(a)`
that minimizes the ground state energy, i.e.,

.. math::
    
    z^*(a) = \arg\, \min_{z} \langle \psi_{z}| H(a) | \psi_{z}\rangle = \arg, \min_{z} E(z, a)


where :math:`E(z, a)` is the energy function. We consider the following
Hamiltonian

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
eigendecomposition to compute gradients. We will compare this exact computation
for a small system to the gradients given by implicit differentiation through a
variationally obtained solution.

We define the following optimality condition at the solution point:

.. math::
    
    f(z, a) = \partial_z E(z, a) = 0.

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

In the case where :math:`A` is just the energy, i.e., :math:`A = H(a)`, the
Hellmann–Feynman theorem allows us to easily compute the gradient. However for
any general operator, we need the gradients :math:`\partial_a z^{*}(a)` and
therefore implicit differentiation is a very elegant way to go beyond the
Hellmann–Feynman theorem for arbitrary expectation values.

Let us now dive into the code and implementation.

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
# Defining the Hamiltonian and measurement operator
# -------------------------------------------------
# We define the Hamiltonian by building the non-parametric part separately and
# adding the parametric to it as a separate term. Note that for the example of
# generalized susceptibility, we are measuring expectation values of the
# operator :math:`A` that also defines the parametric part of the Hamiltonian
# However this is not necessary, we could compute gradients for any other
# operator using implicit differentiation as we have access to the gradients
# :math:`\partial_a z^{*}(a)`.
#
##############################################################################

N = 4
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
A = reduce(add, ((1 / N) * qml.PauliZ(i) for i in range(N)))
A_matrix = qml.matrix(A)

###############################################################################
# Computing the exact ground state through eigendecomposition
# -----------------------------------------------------------
# We now define a function that computes the exact ground state using
# eigendecomposition. Ideally, we would like to take gradients of this function.
# It is possible to simply apply automatic differentiation through this exact
# ground-state computation. JAX has an implementation of differentiating
# through eigendecomposition.
# 
# Note that in the plot, we have some points which are `nan` where the gradient
# computation through the eigendecomposition does not work. We see later that
# the computation thorough the VQA is more stable.
###############################################################################

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
# ---------------------------------------------------------------
# Let us now compute the susceptibility function by taking gradients of the
# expectation value of our operator :math:`A` w.r.t `a`. We can use `jax.vmap`
# to vectorize the computation over different values of `a`.
#################################################################

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

alist = jnp.linspace(0, 3, 1000)
susvals_exact = susceptibility_exact(alist)

plt.plot(alist, susvals_exact)
plt.xlabel("a")
plt.ylabel(r"$\partial_{a}\langle A \rangle$")
plt.show()

###############################################################################
# Computing susceptibility through implicit differentiation
# ---------------------------------------------------------
# We use PennyLane to find a variational ground state for the Hamiltonian
# :math:`H(a)` and compute implicit gradients through the variational
# optimization procedure. We use the `jaxopt` library which contains an
# implementation of gradient descent that automatically comes with implicit
# differentiation capabilities. We are going to use that to obtain
# susceptibility by taking gradients through the ground-state minimization.
#
# .. figure:: ../demonstrations/implicit_diff/VQA.png
#    :scale: 65%
#    :alt: circles
#
# Defining the variational state
# ------------------------------
# In PennyLane, we can implement a variational state in different ways by
# defining a quantum circuit. There are also template circuits available such as
# `SimplifiedTwoDesign` that implements the `two-design ansatz <https://docs.pennylane.ai/en/stable/code/api/pennylane.SimplifiedTwoDesign.html>`_.
# The ansatz consists of layers consisting of Pauli-Y rotations with
# controlled-Z gates. In each layer there are `N - 1` parameters for Pauli-Y gates.
# Therefore the ansatz is efficient and as long as it is expressive enough to
# represent the ground-state.
#
# We set `n_layers = 5` but you can redo this example with fewer layers to see
# how a less expressive ansatz leads to error in the susceptibility computation.
#
# .. note::
#
#   The setting `shots=None` makes the computation of gradients using reverse-mode
#   autodifferentiation (backpropagation). It allows us to just-in-time (JIT)
#   compile the functions that compute expectation values and gradients.
#   In a real device we will have finite shots and the gradients are computed
#   using the parameter-shift rule. However this may be slower.
###############################################################################

variational_ansatz = qml.SimplifiedTwoDesign
n_layers = 5
weights_shape = variational_ansatz.shape(n_layers, N)

dev = qml.device("default.qubit.jax", wires=N, shots=None)

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
print("Energy", energy(z_init, a))

###############################################################################
# Computing ground states using a variational quantum algorithm (VQA)
# -------------------------------------------------------------------
# We construct a loss function that defines a ground-state minimization
# task. We are looking for variational parameters `z` that minimize the energy
# function. Once we find a set of parameters `z`, we wish to compute the
# gradient of any function of the ground state w.r.t. `a`.
#
# .. figure:: ../demonstrations/implicit_diff/vqa.png
#   :scale: 65%
#   :alt: circles
# 
# Computing the susceptibility by differentiating through the VQA
# ---------------------------------------------------------------
# We will use the tool `jaxopt` for implicit differentiation. `jaxopt` implements
# modular implicit differentiation for various cases, e.g., for fixed-point
# functions or optimization. We can directly use `jaxopt` to optimize our loss
# function and then compute implicit gradients through it.
# It all works due to Pennylane's excellent JAX integration.
#
# The implicit differentiation formulas can eve be implemented manually with
# JAX as shown here: https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations
# `jaxopt` implements these formulas in a modular way such that using the
# `jaxopt.GradientDescent` optimizer with `implicit_diff=True` lets us compute
# implicit gradients through the gradient descent optimization.
# We use the excellent integration between Pennylane, Jax
# and Jaxopt to compute the susceptibility.
# 
# Differentiating through the `groundstate_solution_map_variational` function
# here uses implicit differentiation through the `jaxopt.GradientDescent`
# optimization. Since everything is written in JAX, simply calling the
# `jax.grad` function works as `jaxopt` computes the implicit gradients and
# plugs it any computation used by `jax.grad`. We can also just-in-time (JIT)
# compile all functions although the compilation may take some time as the
# number of spins or variational ansatz becomes more complicated. Once compiled,
# all computes run very fast for any parameters.
########################################################################

def ground_state_solution_map_variational(a, z_init):
    """The ground state solution map that we want to differentiate
    through.

    Args:
        a (float): The parameter in the Hamiltonian, H(a).
        z_init [jnp.array(jnp.float)]: The initial guess for the variational
                                       parameters.

    Returns:
        z_star (jnp.array [jnp.float]): The parameters that define the
                                        ground-state solution.
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
            energy(z, a) + 0.001 * jnp.sum(jnp.abs(z[0])) + 0.001 * jnp.sum(jnp.abs(z[1]))
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

a = jnp.array(np.random.uniform(0, 1.0))  # A random `a``
z_star_variational = ground_state_solution_map_variational(a, z_init)
print("Variational parameters for the groundstate", z_star_variational)

@jax.jit
@qml.qnode(dev, interface="jax")
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

###############################################################################
# Computing gradients through the VQA simply by calling `jax.grad`
# ---------------------------------------------------------------
# We can compute the susceptibility values by simply using `jax.grad`. After the
# first call, the function is compiled and therefore subsequent calls become
# much faster.
###############################################################################

susceptibility_variational = jax.jit(jax.grad(groundstate_expval_variational, argnums=0))
z_init = [jnp.array(2 * np.pi * np.random.random(s)) for s in weights_shape]
print("Susceptibility", susceptibility_variational(alist[0], z_init))

susvals_variational = []

for i in range(len(alist)):
    susvals_variational.append(susceptibility_variational(alist[i], z_init))

plt.plot(alist, susvals_variational, label="Implicit diff through VQA")
plt.plot(alist, susvals_exact, "--", c="k", label="Automatic diff through eigendecomposition")
plt.xlabel("a")
plt.ylabel(r"$\partial_{a}\langle A \rangle$")
plt.legend()
plt.show()

# PennyLane version and details
print(qml.about())

##############################################################################
# Conclusion
# ----------
# We have shown how a combination of JAX, PennyLane and JAXOpt can be used to
# compute implicit gradients through a VQA. The ability to compute such
# gradients opens up new possibilities, e.g., designing a Hamiltonian such that
# its ground-state has certain properties. It is also possible to perhaps look
# at this inverse-design of the Hamiltonian as a control problem. Implicit
# differentiation in the classical setting allows defining a new type of
# neural network layer --- implicit layers such as neural ODEs. In a similar
# way, we hope this demo inspires creation of new architectures for quantum
# neural networks, perhaps a quantum version of neural ODEs or quantum implicit
# layers.
#
# In future works, it would be important to assess the cost of running implicit
# differentiation through an actual quantum computer and determine the quality
# of such gradients as a function of noise as explored in a related recent work
# [#Matteo2021]_. 
#
# References
# ----------
#
# .. [#Paradis2004]
#
#     Jaume Paradís, Josep Pla & Pelegrí Viader
#     "Fermat and the Quadrature of the Folium of Descartes"
#     The American Mathematical Monthly, 111:3, 216-229
#     `10.1080/00029890.2004.11920067 <https://doi.org/10.1080/00029890.2004.11920067>`__, 2004.
#
# .. [#Ahmed2022] 
#    
#     Shahnawaz Ahmed, Nathan Killoran, Juan Felipe Carrasquilla Álvarez
#     "Implicit differentiation of variational quantum algorithms
#    `arXiv:2211.XXXX <https://arxiv.org/abs/2111.XXXX>`__, 2022.
# 
# .. [#Blondel2021]
# 
#    Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe Llinares-López, Fabian Pedregosa, Jean-Philippe Vert   
#    "Efficient and modular implicit differentiation"
#    `arXiv:2105.15183 <https://arxiv.org/abs/2105.15183>`__, 2021.
#
# .. [#implicitlayers]
# 
#     Zico Kolter, David Duvenaud, Matt Johnson.
#     "Deep Implicit Layers - Neural ODEs, Deep Equilibirum Models, and Beyond"
#    `http://implicit-layers-tutorial.org <http://implicit-layers-tutorial.org>`__, 2021.
#     
# .. [#Matteo2021]
# 
#    Olivia Di Matteo, R. M. Woloshyn
#    "Quantum computing fidelity susceptibility using automatic differentiation"
#    `arXiv:2207.06526 <https://arxiv.org/abs/2207.06526>`__, 2022.
#
# About the author
# ----------------
# .. include:: ../_static/authors/shahnawaz_ahmed.txt
