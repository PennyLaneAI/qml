r"""
.. _coherent_vqls:

Coherent Variational Quantum Linear Solver
==========================================
*Author: Andrea Mari*

In this tutorial we implement the *coherent variational quantum linear solver* (CVQLS). 
This is an algorithm inspired by the VQLS proposed in 
Ref. [1], but it has an important difference: the matrix :math:`A` defining of the linear 
problem is physically applied as a probabilistic coherent operation.


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

    |b\rangle = U |0\rangle,

where again we assume that :math:`U` can be efficiently implemented with a quantum circuit.

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
:math:`|\Psi\rangle` and :math:`|b\rangle`. This suggests to define the following cost function:

.. math::

    C_G = 1- |\langle b | \Psi \rangle|^2,

such that its minimization with respect to the variational parameters should lead towards the problem solution.

The approach used in Ref. [1] is to decompose the cost function in terms of many expectation values associated to the
individual components :math:`A_l` of the problem matrix :math:`A`. For this reason, in the VQLS proposed in Ref. [1],
the state vector proportional to :math:`A |x\rangle` is never physically prepared.

On the contrary, the idea presented in this tutorial is to physically implement the linear map :math:`A` as
a coherent probabilistic operation. This approach allows to prepare the state 
:math:`|\Psi\rangle :=  A |x\rangle/\sqrt{\langle x |A^\dagger A |x\rangle}` which can be used to estimate the
overlap :math:`|\langle b | \Psi \rangle|` and so the cost function of the problem.


Coherently applying :math:`A`
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Without loss of generality we can assume that the complex coefficients :math:`c=(c_1, c_2, \dots c_L)` appearing
in the definition of :math:`A` are normalized and that :math:`L=2^m` for some positive integer :math:`m`.
Indeed the linear problem is defined up to a constant scaling factor and, moreover, we can
always pad  :math:`c` with additional zero elements.


Let us consider a unitary circuit :math:`U_C`, embedding the classical vector :math:`c` into the quantum state :math:`|c\rangle` of :math:`m` ancillary qubits:

.. math::

    |c \rangle =  U_c |0\rangle = \sum_{l=0}^{L-1} c_l | l \rangle,

where :math:`\{ |l\rangle \}` is the computational basis of the ancillary system.


Now, for each unitary component :math:`A_l` of the problem matrix :math:`A`, we can define the associated generalized controlled operation
acting on the system and ancillary qubits as follows:

.. math::

    CA_l \, |j\rangle |l' \rangle  = 
    \left\{
    \begin{array}{c}
    \left(A_l \otimes \mathbb{I}\right) \, |j\rangle |l \rangle \quad \mathrm{for}\; l'=l \\
    \qquad \; \; \;|j\rangle |l' \rangle  \quad \mathrm{for}\; l'\neq l 
    \end{array}
    \right\},

i.e., the unitary :math:`A_l` is applied only when the ancillary system is in state :math:`|l\rangle`.


A simple example
^^^^^^^^^^^^^^^^

In this tutorial we consider the following simple example based on a system of 3 qubits (plus an ancilla)
which is very similar to the one experimentally tested in Ref. [1]:

.. math::
        \begin{align}
        A  &=  c_0 A_0 + c_1 A_1 + c_2 A_2 = \mathbb{I} + 0.2 X_0 Z_1 + 0.2 X_0, \\
        \\
        |b\rangle &= U |0 \rangle = H_0  H_1  H_2 |0\rangle,
        \end{align}

where :math:`Z_j, X_j, H_j` represent the Pauli :math:`Z`, Pauli :math:`X` and Hadamard gates applied to the qubit with index :math:`j`.

This problem is computationally quite easy since a single layer of local rotations is enough to generate the
solution state, i.e., we can use the following simple ansatz:

.. math::
        |x\rangle = V(w) |0\rangle = \Big [  R_y(w_0) \otimes  R_y(w_1) \otimes  R_y(w_2) \Big ]  H_0  H_1  H_2 |0\rangle.


In the code presented below we solve this particular problem by minimizing the local cost function :math:`C_L`.
Eventually we will compare the quantum solution with the classical one.

"""



##############################################################################
# General setup
# ------------------------
# This Python code requires *PennyLane* and the plotting library *matplotlib*.

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Plotting
import matplotlib.pyplot as plt

##############################################################################
# Setting of the main hyper-parameters of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_qubits = 3  # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 1  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.001  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator


##############################################################################
# Circuits of the quantum linear problem
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

##############################################################################
# We now define the unitary operations associated to the simple example
# presented in the introduction.
# Since we want to implement a Hadamard test, we need the unitary operations
# :math:`A_j` to be controlled by the state of an ancillary qubit.

# Coefficients of the linear combination A = c_0 A_0 + c_1 A_1 ...
c = np.array([1.0, 0.2, 0.2])

def U_b():
    """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

def CA(idx):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if idx == 0:
        # Identity operation
        None

    elif idx == 1:
        qml.CNOT(wires=[ancilla_idx, 0])
        qml.CZ(wires=[ancilla_idx, 1])

    elif idx == 2:
        qml.CNOT(wires=[ancilla_idx, 0])


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
# e.g., the PennyLane template `pennylane.templates.layers.StronglyEntanglingLayers()`.


def variational_block(weights):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    # We first prepare an equal superposition of all the states of the computational basis.
    for idx in range(n_qubits):
     qml.Hadamard(wires=idx)

    # A very minimal variational circuit.
    for idx, element in enumerate(weights):
        qml.RY(element, wires=idx)


##############################################################################
# Hadamard test
# --------------
#
# We first initialize a PennyLane device with the ``default.qubit`` backend.
#
# As a second step, we define a PennyLane ``qnode`` object representing a model of the actual quantum computation.
#
# The circuit is based on the
# `Hadamard test <https://en.wikipedia.org/wiki/Hadamard_test_(quantum_computation)>`_
# and will be used to estimate the coefficients :math:`\mu_{l,l',j}` defined in the introduction.
# A graphical representation of this circuit is shown at the top of this tutorial.

dev_mu = qml.device("default.qubit", wires=tot_qubits)

@qml.qnode(dev_mu)
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

    # First Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i" phase gate.
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights)

    # Controlled application of the unitary component A_l of the problem matrix A.
    CA(l)

    # Adjoint of the unitary U_b associated to the problem vector |b>. 
    # In this specific example Adjoint(U_b) = U_b.
    U_b()

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    # Unitary U_b associated to the problem vector |b>.
    U_b()

    # Controlled application of Adjoint(A_lp).
    # In this specific example Adjoint(A_lp) = A_lp.
    CA(lp)

    # Second Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # Expectation value of Z for the ancillary qubit.
    return qml.expval(qml.PauliZ(wires=ancilla_idx))


##############################################################################################
# To get the real and imaginary parts of :math:`\mu_{l,l',j}`, one needs to run the previous
# quantum circuit with and without a phase-shift of the ancillary qubit. This is automatically
# done by the following function.


def mu(weights, l=None, lp=None, j=None):
    """Generates the coefficients to compute the "local" cost function C_L."""

    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag


##############################################################################
# Local cost function
# ------------------------------------
#
# Let us first define a function for estimating :math:`\langle x| A^\dagger A|x\rangle`.


def psi_norm(weights):
    """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)


##############################################################################
# We can finally define the cost function of our minimization problem.
# We use the analytical expression of :math:`C_L` in terms of the
# coefficients :math:`\mu_{l,l',j}` given in the introduction.


def cost_loc(weights):
    """Local version of the cost function, which tends to zero when A |x> is proportional to |b>."""
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    # Cost function C_L
    return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))


##############################################################################
# Variational optimization
# -----------------------------
#
# We first initialize the variational weights with random parameters (with a fixed seed).

np.random.seed(rng_seed)
w = q_delta * np.random.randn(n_qubits)

##############################################################################
# To minimize the cost function we use the gradient-descent optimizer.
opt = qml.GradientDescentOptimizer(eta)


##############################################################################
# We are ready to perform the optimization loop.

cost_history = []
for it in range(steps):
    w = opt.step(cost_loc, w)
    cost = cost_loc(w)
    print("Step {:3d}       Cost_L = {:9.7f}".format(it, cost))
    cost_history.append(cost)


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
# *qnode* object.

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
q_probs = np.bincount(samples) / n_shots

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
