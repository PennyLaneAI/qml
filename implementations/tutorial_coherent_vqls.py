r"""
.. _coherent_vqls:

Coherent Variational Quantum Linear Solver
==========================================
*Author: Andrea Mari*

In this tutorial we propose and implement an algorithm that we call
*coherent variational quantum linear solver* (CVQLS). 
This is inspired by the VQLS proposed in Ref. [1], with an important difference: 
the matrix :math:`A` associated to the problem is physically 
applied as a probabilistic coherent operation. This approach has some advantages and
disadvantages and its practical convenience depends on the specific linear problem 
to be solved and on experimental constraints.

.. figure:: ../implementations/coherent_vqls/cvqls_circuit.png
    :align: center
    :width: 100%
    :target: javascript:void(0)

Introduction
------------

We first define the problem and the general structure of the CVQLS. 
As a second step, we consider a particular case and we solve it explicitly with PennyLane.

The problem
^^^^^^^^^^^

We are given a :math:`2^n \times 2^n` matrix :math:`A` which can be expressed as a linear
combination of :math:`L` unitary matrices :math:`A_0, A_1, \dots A_{L-1}`, i.e.,

.. math::

    A = \sum_{l=0}^{L-1} c_l A_l,

where :math:`c_l` are arbitrary complex numbers. Importantly, we assume that each of the
unitary components :math:`A_l` can be efficiently implemented with a quantum circuit
acting on :math:`n` qubits.

We are also given a normalized complex vector in the physical form of a quantum
state :math:`|b\rangle`, which can be generated by a unitary operation :math:`U`
applied to the ground state of :math:`n` qubits. , i.e.,

.. math::

    |b\rangle = U_b |0\rangle,

where again we assume that :math:`U_b` can be efficiently implemented with a quantum circuit.

The problem that we aim to solve is that of preparing a quantum state :math:`|x\rangle`, such that
:math:`A |x\rangle` is proportional to :math:`|b\rangle` or, equivalently, such that

.. math::

    |\Psi\rangle :=  \frac{A |x\rangle}{\sqrt{\langle x |A^\dagger A |x\rangle}} \approx |b\rangle.


Coherent Variational Quantum Linear Solver (CVQLS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We approximate the solution :math:`|x\rangle` with a variational quantum
circuit, i.e., a unitary circuit :math:`V` depending on a finite number of classical real parameters
:math:`w = (w_0, w_1, \dots)`:

.. math::

    |x \rangle = V(w) |0\rangle.

The parameters should be optimized in order to maximize the overlap between the quantum states
:math:`|\Psi\rangle` and :math:`|b\rangle`. We define the following cost function,

.. math::

    C = 1- |\langle b | \Psi \rangle|^2,

such that its minimization with respect to the variational parameters should lead towards the problem solution.

The approach used in Ref. [1] is to decompose the cost function in terms of many expectation values associated to the
individual components :math:`A_l` of the problem matrix :math:`A`. For this reason, in the VQLS of Ref. [1],
the state vector proportional to :math:`A |x\rangle` is not physically prepared.
On the contrary, the idea presented in this tutorial is to physically implement the linear map :math:`A` as
a coherent probabilistic operation. This approach allows to prepare the state 
:math:`|\Psi\rangle :=  A |x\rangle/\sqrt{\langle x |A^\dagger A |x\rangle}` which can be used to estimate the
cost function of the problem in a more direct way.


Coherently applying :math:`A`
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

The problem of coherently applying a liner combination of unitary operations has been already studied in Ref. [2]
and here we follow a very similar approach.

Without loss of generality we can assume that the coefficients :math:`c=(c_1, c_2, \dots c_L)` appearing
in the definition of :math:`A` represent a positive and normalized probability distribution. i.e.,

.. math::
    
    c_l \ge 0 \quad \forall l,  \qquad \sum_{l=0}^{L-1} c_l=1.

Indeed the complex phase of each coefficient :math:`c_l` can always be absorbed into the associated unitary :math:`A_l`, obtaining
in this way a vector of positive values. Moreover, since the linear problem is 
defined up to a constant scaling factor, we can also normalize the coefficients to get a probability distribution.

For simplicity, since we can always pad :math:`c` with additional zeros, we assume that :math:`L=2^m` for some positive integer :math:`m`.

Let us consider a unitary circuit :math:`U_c`, embedding the square root of :math:`c` into the quantum state :math:`|\sqrt{c}\rangle` of :math:`m` ancillary qubits:

.. math::

    |\sqrt{c} \rangle =  U_c |0\rangle = \sum_{l=0}^{L-1} \sqrt{c_l} | l \rangle,

where :math:`\{ |l\rangle \}` is the computational basis of the ancillary system.


Now, for each component :math:`A_l` of the problem matrix :math:`A`, we can define an associated controlled unitary operation :math:`CA_l`,
acting on the system and on the ancillary basis states as follows:

.. math::

    CA_l \, |j\rangle |l' \rangle  = 
    \Bigg\{
    \begin{array}{c}
    \left(A_l \otimes \mathbb{I}\right) \; |j\rangle |l \rangle \quad \; \mathrm{for}\; l'=l \\
    \qquad \qquad |j\rangle |l' \rangle  \quad \mathrm{for}\; l'\neq l 
    \end{array},

i.e., the unitary :math:`A_l` is applied only when the ancillary system is in the corresponding basis state :math:`|l\rangle`.

A natural generalization of the `Hadamard test <https://en.wikipedia.org/wiki/Hadamard_test_(quantum_computation)>`_, to the case of multiple unitary operations, is the following
(see also the figure at the top of this tutorial):

1. Prepare all qubits in the ground state.
2. Apply :math:`U_c` to the ancillary qubits.
3. Apply the variational circuit :math:`V` to the system qubits.
4. Apply all the controlled unitaries :math:`CA_l` for all values of :math:`l`.
5. Apply :math:`U_c^\dagger` to the ancillary qubits.
6. Measure the ancillary qubits in the computational basis.
7. If the outcome of the measurement is the ground state, the system collapses to
   :math:`|\Psi\rangle :=  A |x\rangle/\sqrt{\langle x |A^\dagger A |x\rangle}`.
   If the outcome is not the ground state, the experiment should be repeated.


Estimating the cost function
>>>>>>>>>>>>>>>>>>>>>>>>>>>>

From a technical point of view, the previous steps represent the most difficult part of the algorithm. 
Once we have at our disposal the quantum system prepared in the state :math:`|\Psi\rangle`,
it is very easy to compute the cost function.
Indeed one could simply continue the previous protocol with the following two steps:

8. Apply :math:`U_b^\dagger` to the system.
9. Measure the system in the computational basis. The probability of finding it
   in the ground state (given the ancillary qubits measured in their ground state),
   is :math:`|\langle 0 | U_b^\dagger |\Psi \rangle|^2 = |\langle b | \Psi \rangle|^2`.

So, with sufficiently many shots of the previous experiment, one can directly estimate
the cost function of the problem.

Importantly, the operations of steps 7 and 8 commute. Therefore all the measurements can be
delayed until the end of the quantum circuit (as shown in the figure at the top of this tutorial),
making the structure of the experiment more straightforward.  

A simple example
^^^^^^^^^^^^^^^^

In this tutorial we apply the previous theory to the following simple example 
based on a system of 3 qubits, which was already considered in Ref. [1] and also reproduced in PennyLane (VQLS):

.. math::
        \begin{align}
        A  &=  c_0 A_0 + c_1 A_1 + c_2 A_2 = \mathbb{I} + 0.2 X_0 Z_1 + 0.2 X_0, \\
        \\
        |b\rangle &= U_b |0 \rangle = H_0  H_1  H_2 |0\rangle,
        \end{align}

where :math:`Z_j, X_j, H_j` represent the Pauli :math:`Z`, Pauli :math:`X` and Hadamard gates applied to the qubit with index :math:`j`.

This problem is computationally quite easy since a single layer of local rotations is enough to generate the
solution state, i.e., we can use the following simple ansatz:

.. math::
        |x\rangle = V(w) |0\rangle = \Big [  R_y(w_0) \otimes  R_y(w_1) \otimes  R_y(w_2) \Big ]  H_0  H_1  H_2 |0\rangle.


In the code presented below we solve this particular problem, by following the general scheme of the CVQLS previously discussed.
Eventually we will compare the quantum solution with the classical one.

General setup
-------------
This Python code requires *PennyLane* and the plotting library *matplotlib*.

"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Setting of the main hyper-parameters of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_qubits = 3                # Number of system qubits.
m = 2                       # Number of ancillary qubits
n_shots = 10 ** 6           # Number of quantum measurements.
tot_qubits = n_qubits + m   # System + ancillary qubits.
ancilla_idx = n_qubits      # Index of the first ancillary qubit.
steps = 10                  # Number of optimization steps.
eta = 0.8                   # Learning rate.
q_delta = 0.001             # Initial spread of random quantum weights.
rng_seed = 0                # Seed for random number generator.


##############################################################################
# Circuits of the quantum linear problem
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##############################################################################
# We need to define the unitary operations associated to the simple example
# presented in the introduction.
#
# The coefficients of the linear combination are three positive numbers :math:`(1, 0.2, 0.2)`.
# So we can embed them in the state of  :math:`m=2` ancillary qubits by adding a final zero element and
# normalizing their sum to :math:`1`:

c = np.array([1, 0.2, 0.2, 0])
c = c / np.sum(c)
# We also compute the square root of c
sqrt_c = np.sqrt(c)

##############################################################################
# We need to embed the square root of the probability distribution ``c`` into the amplitudes
# of the ancillary state. It is easy to check that one can always embed 3 positive
# amplitudes with just three gates:
# a local :math:`R_y` rotation, a controlled-:math:`R_y` and a controlled-NOT.


def U_c():
    """Unitary matrix rotating the ground state of the ancillary qubits 
    to |sqrt(c)> = U_c |0>."""
    # Circuit mapping |00> to sqrt_c[0] |00> + sqrt_c[1] |01> + sqrt_c[2] |10>
    qml.RY(-2 * np.arccos(sqrt_c[0]), wires=ancilla_idx)
    qml.CRY(-2 * np.arctan(sqrt_c[2] / sqrt_c[1]), wires=[ancilla_idx, ancilla_idx + 1])
    qml.CNOT(wires=[ancilla_idx + 1, ancilla_idx])


def U_c_dagger():
    """Adjoint of U_c."""
    qml.CNOT(wires=[ancilla_idx + 1, ancilla_idx])
    qml.CRY(2 * np.arctan(sqrt_c[2] / sqrt_c[1]), wires=[ancilla_idx, ancilla_idx + 1])
    qml.RY(2 * np.arccos(sqrt_c[0]), wires=ancilla_idx)


##############################################################################
# We are left to define the sequence of all controlled-unitaries :math:`CA_l`, acting
# as :math:`A_l` on the system whenever the ancillary state is :math:`|l\rangle`.
# Since in our case :math:`A_0=\mathbb{I}` and ``c[3] = 0``, we only need to apply :math:`A_1` and
# :math:`A_2` controlled by the first and second ancillary qubits respectively.


def CA_all():
    """Controlled application of all the unitary components A_l of the problem matrix A."""
    # Controlled-A_1
    qml.CNOT(wires=[ancilla_idx, 0])
    qml.CZ(wires=[ancilla_idx, 1])

    # Controlled-A2
    qml.CNOT(wires=[ancilla_idx + 1, 0])


##############################################################################
# The circuit for preparing the problem vector :math:`|b\rangle` is very simple:


def U_b():
    """Unitary matrix rotating the system ground state to the 
    problem vector |b> = U_b |0>."""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)


##############################################################################
# Variational quantum circuit
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# What follows is the variational quantum circuit that should generate the solution
# state :math:`|x\rangle= V(w)|0\rangle`.
#
# The first layer of the circuit is a product of Hadamard gates preparing a
# balanced superposition of all basis states.
#
# After that, we apply a very simple variational ansatz
# which is just a single layer of qubit rotations
# :math:`R_y(w_0) \otimes  R_y(w_1) \otimes  R_y(w_2)`.
# For solving more complex problems, we suggest to use more expressive circuits as,
# e.g., the PennyLane ``StronglyEntanglingLayers`` template.


def variational_block(weights):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    # We first prepare an equal superposition of all the states of the computational basis.
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    # A very minimal variational circuit.
    for idx, element in enumerate(weights):
        qml.RY(element, wires=idx)


##############################################################################
# Full quantum circuit
# --------------------
#
# Now, we can define the full circuit associated to the CVQLS protocol presented in the introduction and
# corresponding to the figure at the top of this tutorial.


def full_circuit(weights):
    """Full quantum circuit necessary for the CVQLS protocol, without the final measurement."""
    # U_c applied to the ancillary qubits.
    U_c()

    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights)

    # Application of all the controlled-unitaries CA_l associated to the problem matrix A.
    CA_all()

    # Adjoint of U_b, where U_b |0> = |b>.
    # For this particular problem adjoint(U_b)=U_b
    U_b()

    # Adjoint of U_c, applied to the ancillary qubits.
    U_c_dagger()


##############################################################################
# To estimate the overlap of the ground state with the post-selected state, one could
# directly make use the measurement samples. However, since we want to optimize the cost
# function, it is useful to express everything in terms of expectation values through 
# the Bayes' theorem:
#
# .. math::
#   |\langle b | \Psi \rangle|^2=
#   P( \mathrm{sys}=\mathrm{ground}\,|\, \mathrm{anc} = \mathrm{ground}) =
#   P( \mathrm{all}=\mathrm{ground})/P( \mathrm{anc}=\mathrm{ground})
#
# To evaluate the two probabilities appearing on the right hand side of the previous equation
# we initialize a ``default.qubit`` device and we define two different ``qnode`` circuits.

dev = qml.device("default.qubit", wires=tot_qubits)

@qml.qnode(dev)
def global_ground(weights):
    # Circuit gates
    full_circuit(weights)
    # Projector on the global ground state.
    P = np.zeros((2 ** tot_qubits, 2 ** tot_qubits))
    P[0, 0] = 1.0
    return qml.expval(qml.Hermitian(P, wires=range(tot_qubits)))

@qml.qnode(dev)
def ancilla_ground(weights):
    # Circuit gates
    full_circuit(weights)
    # Projector on the ground state of the ancillary system.
    P_anc = np.zeros((2 ** m, 2 ** m))
    P_anc[0, 0] = 1.0
    return qml.expval(qml.Hermitian(P_anc, wires=range(n_qubits, tot_qubits)))


##############################################################################
# Variational optimization
# -----------------------------
#
# In order to variationally solve our lineaer problem, we first define the cost function
# :math:`C = 1- |\langle b | \Psi \rangle|^2` that we are going to minimize.
# As explained above, we express it in terms of expectation values thorugh the Bayes' theorem.


def cost(weights):
    """Cost function which tends to zero when A |x> tends to |b>."""

    p_global_ground = global_ground(weights)
    p_ancilla_ground = ancilla_ground(weights)
    p_cond = p_global_ground / p_ancilla_ground

    return 1 - p_cond


##############################################################################
# To minimize the cost function we use the gradient-descent optimizer.
opt = qml.GradientDescentOptimizer(eta)

##############################################################################
# We initialize the variational weights with random parameters (with a fixed seed).

np.random.seed(rng_seed)
w = q_delta * np.random.randn(n_qubits)

##############################################################################
# We are ready to perform the optimization loop.

cost_history = []
for it in range(steps):
    w = opt.step(cost, w)
    _cost = cost(w)
    print("Step {:3d}       Cost = {:9.7f}".format(it, _cost))
    cost_history.append(_cost)


##############################################################################
# We plot the cost function with respect to the optimization steps.
# We remark that this is not an abstract mathematical quantity
# since it also represents a bound for the error between the generated state
# and the exact solution of the problem.

plt.style.use("seaborn")
plt.plot(cost_history, "g")
plt.ylabel("Cost function")
plt.xlabel("Optimization steps")
plt.show()

##############################################################################
# Comparison of quantum and classical results
# -------------------------------------------
#
# Since the specific problem considered in this tutorial has a small size, we can also
# solve it in a classical way and then compare the results with our quantum solution.
#

##############################################################################
# Classical algorithm
# ^^^^^^^^^^^^^^^^^^^
# To solve the problem in a classical way, we use the explicit matrix representation in
# terms of numerical NumPy arrays.

Id = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

A_0 = np.identity(8)
A_1 = np.kron(np.kron(X, Z), Id)
A_2 = np.kron(np.kron(X, Id), Id)

A_num = c[0] * A_0 + c[1] * A_1 + c[2] * A_2
b = np.ones(8) / np.sqrt(8)

##############################################################################
# We can print the explicit values of :math:`A` and :math:`b`:

print("A = \n", A_num)
print("b = \n", b)


##############################################################################
# The solution can be computed via a matrix inversion:

A_inv = np.linalg.inv(A_num)
x = np.dot(A_inv, b)

##############################################################################
# Finally, in order to compare x with the quantum state |x>, we normalize and square its elements.
c_probs = (x / np.linalg.norm(x)) ** 2

##############################################################################
# Preparation of the quantum solution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


##############################################################################
# Given the variational weights ``w`` that we have previously optimized,
# we can generate the quantum state :math:`|x\rangle`. By measuring :math:`|x\rangle`
# in the computational basis we can estimate the probability of each basis state.
#
# For this task, we initialize a new PennyLane device and define the associated
# QNode.

dev_x = qml.device("default.qubit", wires=n_qubits, shots=n_shots)

@qml.qnode(dev_x)
def prepare_and_sample(weights):

    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights)

    # We assume that the system is measured in the computational basis.
    # If we label each basis state with a decimal integer j = 0, 1, ... 2 ** n_qubits - 1,
    # this is equivalent to a measurement of the following diagonal observable.
    basis_obs = qml.Hermitian(np.diag(range(2 ** n_qubits)), wires=range(n_qubits))

    return qml.sample(basis_obs)


##############################################################################
# To estimate the probability distribution over the basis states we first take ``n_shots``
# samples and then compute the relative frequency of each outcome.

samples = prepare_and_sample(w).astype(int)
q_probs = np.bincount(samples, minlength=2 ** n_qubits) / n_shots

##############################################################################
# Comparison
# ^^^^^^^^^^
#
# Let us print the classical result.
print("x_n^2 =\n", c_probs)

##############################################################################
# The previous probabilities should match the following quantum state probabilities.
print("|<x|n>|^2=\n", q_probs)

##############################################################################
# Let us graphically visualize both distributions.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))

ax1.bar(np.arange(0, 2 ** n_qubits), c_probs, color="blue")
ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")

ax2.bar(np.arange(0, 2 ** n_qubits), q_probs, color="green")
ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")

plt.show()


##############################################################################
# References
# ----------
#
# 1. Carlos Bravo-Prieto, Ryan LaRose, Marco Cerezo, Yigit Subasi, Lukasz Cincio, Patrick J. Coles.
#    "Variational Quantum Linear Solver: A Hybrid Algorithm for Linear Systems."
#    `arXiv:1909.05820 <https://arxiv.org/abs/1909.05820>`__, 2019.
#
# 2. Robin Kothari.
#    "Efficient algorithms in quantum query complexity."
#    PhD thesis, University of Waterloo, 2014.
#
#
