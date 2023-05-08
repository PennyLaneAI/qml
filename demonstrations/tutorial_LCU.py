r"""
Block encoding via LCU decompositions
=====================================================

*Author: Oriel Kiss, 5th May 2023*

In this tutorial, we will see a practical implementation of a block encoding technique based on
linear combination of unitaries (LCU), which can be useful to simulate dynamical properties of quantum systems.
"""

######################################################################
# Linear combination of unitaries
# ----------------
#
# Quantum states evolve under unitary dynamics; this however, need not be the case for subsystems of the quantum system.
# Hence, quantum computers are still able to
# perform non unitary operations, by using higher dimensional space. In practice, any matrix :math:`H`
# can be block encoded into a unitary matrix of higher dimension as
#
# .. math::
# V=\begin{pmatrix}H&*\\*&* \end{pmatrix},
#
# where :math:`*` denote arbitrary matrices such that :math:`V` is unitary.
#
# The key ingredient is to write :math:`H` as a linear combination of :math:`K` unitaries
# (`LCU <https://arxiv.org/abs/1202.5822>`__).
#
# .. math::
# H = \sum_{k=0}^{K-1} \alpha_k U_k,
#
# with :math:`\alpha_k \in \mathbb{C}^*` and :math:`U_k` unitary. This can be achieved for any hermitian
# matrix by projecting it onto the Pauli basis. Hence, tha Pauli basis is a unitary basis for hermitian
# matrices.  We note that any matrix can be decomposed into a sum of two hermitian matrices,
# making this scheme general. However, we note that the
# performance is driven by the one-norm of the decomposition, and thus it remains a non trivial task to
# choose a suitable one.
#
######################################################################
# Let start by setting up the problem. We choose a random Hermitian matrix of size
# :math:`2^n \times 2^n`, with :math:`n=2`.
#

import numpy as np
import pennylane as qml

n = 2  # physical system size

shape = (2**n, 2**n)
H = np.random.uniform(-1, 1, shape) + 1.0j * np.random.uniform(-1, 1, shape)  # random matrix
H = H + H.conjugate().transpose()  # makes it hermitian


LCU = qml.pauli_decompose(H)  # Projecting the Hamiltonian onto the Pauli basis
alphas = LCU.terms()[0]

print("LCU decomposition: \n", LCU)

phases = np.angle(alphas)
coeffs = np.abs(alphas)

coeffs = np.sqrt(coeffs)
coeffs /= np.linalg.norm(coeffs, ord=2)  # normalise the coefficients

unitaries = [qml.matrix(op) for op in LCU.terms()[1]]


K = len(coeffs)  # number of terms in the decomposition
a = int(np.ceil(np.log2(K)))  # number of ancilla qubits

######################################################################
# Block encoding
# --------------
#
# Block encoding relies on two important subroutines:
#
# -  The PREPARE subroutine encodes the coefficients of the Hamiltonian in the amplitudes of the
#    quantum state as
#
# .. math::
# \text{PREPARE}|\bar{0}\rangle = \sum_{k=0}^{K-1} \sqrt{\frac{\alpha_k}{\|\vec{\alpha}\|_1}} |k\rangle,
#
# where :math:`|\bar{0}\rangle = |0^{\otimes \lceil \log_2{K}\rceil} \rangle` is the ancillary
# register. Note that we can always assume that :math:`\alpha_k\in \mathbb{R}^+` by assimilating the
# phase into the corresponding unitary.
#
# This can be achieved using the
# ```qml.MottonenStatePreparation`` <https://docs.pennylane.ai/en/stable/code/api/pennylane.MottonenStatePreparation.html>`__
# function with the vector :math:`\vec{\alpha} = (\alpha_1, \cdots, \alpha_n)`.
#
# -  The SELECT subroutine applies the :math:`k`-th unitary :math:`U_k` on
#    :math:`|\psi\rangle`, when given access to the state :math:`|k\rangle` as follows
#
# .. math::
# \text{SELECT} |k\rangle |\psi\rangle  = |k\rangle U_k|\psi \rangle.
#
# This can be achieved using control
# ```qml.ctrl`` <https://docs.pennylane.ai/en/stable/code/api/pennylane.ctrl.html>`__ operations on
# the ancila qubits.
#
# -  :math:`H` can then be block encoded using the following operation:
#    :math:`\|\vec{\alpha}\|_1 \cdot` PREPARE\ :math:`^\dagger` SELECT PREPARE
#    :math:`|\bar{0}\rangle`.
#

######################################################################
# Apply the LCU to a quantum state
# --------------------------------
#
# Let’s focus on the particular example where the LCU is composed of :math:`K=4` terms, and you want
# to apply :math:`H` to a quantum state :math:`|\psi\rangle`. We can show that
# .. math::
# \text{PREPARE}^\dagger \text{ SELECT PREPARE} |\bar{0}\rangle |\psi\rangle = \frac{1}{\|\vec{\alpha}\|_1}|\bar{0}\rangle \sum_{k=0}^{K-1} \alpha_k U_k|\psi \rangle + |\Phi\rangle^\perp,
#
# where :math:`|\Phi\rangle^\perp` is some orthogonal state obtained when the
# algorithm fails. The desired state, up to the normalisation factor, can then be obtained via post
# selecting on :math:`|\bar{0}\rangle`. The following circuit summaries the result.
#
# .. container::
#
#


wires_ancilla = np.arange(a)
wires_physical = np.arange(a, a + n)


def Block_encoding(coeffs, phases, unitaries):
    """
    Perform a block encoding of the LCU matrix
    H = sum_k coeffs[k]*e^{i*phases[k]} *unitaries[k]
    Args:
        coeffs: absolute values of the coefficients of the LCU
        phases: phases of the coefficients of the LCU
        unitaries: unitaries of the LCU decompositions

    Returns:
        Unitary matrix containing the LCU as the first (2**nx2**n) block,
        where n is the system size.
    """

    # Prep
    qml.MottonenStatePreparation(coeffs, wires=wires_ancilla)

    # Select
    for k in range(K):
        ctrl_values = [bool(int(v)) for v in np.binary_repr(n, width=a)]
        qml.ctrl(
            qml.QubitUnitary, control=wires_ancilla, control_values= ctrl_values)(
            np.exp(1.0j * phases[k]) * unitaries[k], wires=wires_physical)

    # Reverse Prep
    qml.adjoint(qml.MottonenStatePreparation)(coeffs, wires=wires_ancilla)


matrix = qml.matrix(Block_encoding)(coeffs, phases, unitaries)
block_matrix = np.linalg.norm(alphas, ord=1) * matrix[: 2**n, : 2**n]

print("LCU block encoding:\n", np.round(block_matrix, 3), "\n")

print("Hamiltonian: \n", np.round(H, 3), "\n")

print("Error: ", np.linalg.norm(block_matrix - H), "\n")

print(
    "V is unitary: ",
    np.allclose(np.eye(matrix.shape[0]), np.dot(matrix.conjugate().transpose(), matrix)),
)


######################################################################
# We see that :math:`H` is exactly block encoded into a larger unitary matrix, up to a normalization factor, which can be run on a
# quantum computer.
#

######################################################################
# Application to quantum simulation
# ---------------------------------
#
# The main problem in quantum chemistry is to be able to extract useful information about quantum
# state. For instance, the energy of a state can be obtained with the quantum phase estimation
# algorithm, which relies on performing time evolution. Trotter-Suzuki decompositions are popular
# techniques to evolve a quantum state, by approximating the time evolution operator using product
# formulas. For example, the first order product formula for a Hamiltonian of the form
# :math:`H=\sum_{l=1}^{L} H_l`, where the :math:`H_l` terms are only *hermitian*, is given by
#
# .. math::
# e^{-iHt} \approx \prod_{l=1}^{L}e^{-iH_lt}.
#
# While these methods already run in polynomial time, better complexity can be achieved by instead
# expanding the time evolution operator into a `Taylor serie <https://arxiv.org/abs/1412.4687>`__, and
# truncate it to some order :math:`M`. Hence, we can write
#
# .. math::
# e^{-iHt} = \sum_{k=0}^{\infty} \frac{(-iHt)^k}{k!} \approx \sum_{k=0}^{M} \frac{(-iHt)^k}{k!},
#
# which can be brought into a LCU form as above, since :math:`H^k` is hermitian.
#
# As a concrete example, let us consider a first order truncation
#
# .. math::
# e^{-iHt} \approx \mathbb{1} -iHt,
#
# whose simulation can be recast into the problem we just solved!
#

######################################################################
# Conclusions
# -----------
#
# We learned how to decompose a hermitian matrix into a linear combination of Pauli operators, which
# can be block encoded into a larger unitary matrix. This scheme is useful to perform time evolution
# via Taylor expansion, which is a recurring sub routine in quantum algorithms.
#

######################################################################
# References
# ----------
#
# `[Childs12] <https://arxiv.org/abs/1202.5822>`__ Andrew M. Childs, Nathan Wiebe, *Hamiltonian
# Simulation Using Linear Combinations of Unitary Operations* (2012)
#
# `[Berry14] <https://arxiv.org/abs/1412.4687>`__ Dominic W. Berry, Andrew M. Childs, Richard Cleve,
# Robin Kothari and Rolando D. Somma *Simulating Hamiltonian dynamics with a truncated Taylor series*
# (2014)
#
