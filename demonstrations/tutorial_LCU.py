r"""

Block encoding via LCU decompositions
=====================================================

.. meta::
    :property="og:description": Learn how to perform block-encoding via LCU decompositions
    :property="og:image": /demonstrations/LCU/LCU.png

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians


*Author: Oriel Kiss — Posted: 05 May 2023. Last updated: 05 May 2023.*


In this tutorial, we will see a practical implementation of a block-encoding technique based on a
linear combination of unitaries (LCU), which can be useful to simulate dynamical properties of quantum systems.

Quantum systems evolve under unitary dynamics. However, this need not be the case for subsystems of the quantum system.
Effectively, this allows for quantum computers to be able to
perform non-unitary operations via block-encoding in a higher dimensional 
space as follows:

.. math:: V=\begin{pmatrix}H&*\\*&* \end{pmatrix}.

Here, :math:`H` is the matrix — not necessarily unitary — being block-encoded and the :math:`*` symbols denote arbitrary matrices that ensure that :math:`V` is unitary.
The key ingredient is to write :math:`H` as a linear combination of :math:`K` unitaries
(`Childs & Wiebe) (2012) <https://arxiv.org/abs/1202.5822>`_),

.. math:: H = \sum_{k=0}^{K-1} \alpha_k U_k,

with :math:`\alpha_k \in \mathbb{C}^*` and :math:`U_k` unitary. This can be achieved if :math:`H` is a Hermitian
matrix (i.e. :math: `H=H^\dagger`) by projecting it onto the Pauli basis since the Pauli basis is a unitary basis for Hermitian
matrices. We note that any matrix can be decomposed into a sum of two Hermitian matrices,
making this scheme general.

Linear combination of unitaries
--------------------------------
"""
######################################################################
# Let start by setting up the problem. We need to define a Hermitian matrix and write it as an LCU in PennyLane.
# We will choose a random Hermitian matrix of size
# :math:`2^n \times 2^n`, with :math:`n=2`.
#

import numpy as np
import pennylane as qml

n = 2  # physical system size

shape = (2**n, 2**n)
H = np.random.uniform(-1, 1, shape) + 1.0j * np.random.uniform(-1, 1, shape)  # random matrix

######################################################################
# Now that we have a random matrix, we will make it Hermitian and write it in the Pauli basis via :func:`~pennylane.pauli_decompose`.
#
H = H + H.conjugate().transpose()  # makes it Hermitian
LCU = qml.pauli_decompose(H)  # Projecting the Hamiltonian onto the Pauli basis
unitaries = [qml.matrix(op) for op in LCU.terms()[1]]

print("LCU decomposition: \n", LCU)

######################################################################
# Block-encoding
# --------------
#
# Block-encoding relies on two important subroutines: PREPARE and SELECT.
#
# 1.  The PREPARE subroutine encodes the coefficients of the LCU decomposition in the amplitudes of the quantum state as
#
# .. math:: \text{PREPARE}|\bar{0}\rangle = \sum_{k=0}^{K-1} \sqrt{\frac{\alpha_k}{\|\vec{\alpha}\|_1}} |k\rangle,
#
# where :math:`|\bar{0}\rangle = |0^{\otimes \lceil \log_2{K}\rceil} \rangle` is the ancillary
# register.
# We remark that the coefficients of the LCU are assumed to be real positive numbers, which does not hinder generality as
# the phases can be assimilated into the unitaries.
#
alphas = LCU.terms()[0]
phases = np.angle(alphas)
coeffs = np.abs(alphas)

coeffs = np.sqrt(coeffs)
coeffs /= np.linalg.norm(coeffs, ord=2)  # normalize the coefficients

######################################################################
# The number of ancilla qubits needed can be computed from the number of terms in the LCU.
#

K = len(coeffs)  # number of terms in the decomposition
a = int(np.ceil(np.log2(K)))  # number of ancilla qubits

wires_ancilla = np.arange(a)  # qubits of the physical system
wires_physical = np.arange(a, a + n)  # ancillary qubits

############################################################
# Amplitude-encoding in this fashion can be achieved using
# :class:`~pennylane.MottonenStatePreparation`
# with the vector :math:`\vec{\alpha} = (\alpha_1, \cdots, \alpha_n)`.
#
# 2. The SELECT subroutine applies the :math:`k`-th unitary :math:`U_k` on
#    :math:`|\psi\rangle` when given access to the state :math:`|k\rangle` as follows
#
# .. math:: \text{SELECT} |k\rangle |\psi\rangle  = |k\rangle U_k|\psi \rangle.
#
# This can be achieved using :func:`~pennylane.ctrl` operations on the ancillary qubits.
#
# :math:`H` can then be block-encoded using the following operation:
# :math:`\|\vec{\alpha}\|_1 \cdot` PREPARE :math:`^\dagger` SELECT PREPARE :math:`|\bar{0}\rangle`.
#
# Let’s focus on an example where the LCU is composed of :math:`K=4` terms and we want to apply
# :math:`H` to a quantum state :math:`|\psi\rangle`. We can show that
#
# .. math:: \text{PREPARE}^\dagger \text{ SELECT} \cdot \text{ PREPARE} |\bar{0}\rangle |\psi\rangle = \frac{1}{\|\vec{\alpha}\|_1}|\bar{0}\rangle \sum_{k=0}^{K-1} \alpha_k U_k|\psi \rangle + |\Phi\rangle^\perp = |\Phi\rangle + |\Phi\rangle^\perp,
#
# where :math:`|\Phi\rangle^\perp` is some orthogonal state to :math:`H|\Phi\rangle`.
# We can see that we obtain the state we want to simulate if the ancilla register 
# is measured in the zero state: :math:`|\bar{0}\rangle`. Hence, block-encoding is a probabilistic algorithm that succeeds with some probability
# related to the one-norm of the LCU decomposition. The desired state, up to the normalization factor, can then be obtained via post
# selecting on :math:`|\bar{0}\rangle`, using following circuit:
#
# .. figure:: /demonstrations/LCU/LCU.png
#     :width: 65%
#     :align: center


######################################################################
# The following quantum function performs this block-encoding procedure.
#
def block_encoding(coeffs, phases, unitaries):
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
        ctrl_values = [bool(int(v)) for v in np.binary_repr(k, width=a)]
        qml.ctrl(
            qml.QubitUnitary, control=wires_ancilla, control_values= ctrl_values)(
            np.exp(1.0j * phases[k]) * unitaries[k], wires=wires_physical)

    # Reverse Prep
    qml.adjoint(qml.MottonenStatePreparation)(coeffs, wires=wires_ancilla)

######################################################################
# Finally, we compute the block-encoding and compare to the original matrix, while making sure that
# the larger matrix :math:`V` is unitary.
#

matrix = qml.matrix(block_encoding)(coeffs, phases, unitaries)
block_matrix = np.linalg.norm(alphas, ord=1) * matrix[: 2**n, : 2**n]

print("LCU block encoding:\n", np.round(block_matrix, 3), "\n")

print("Hamiltonian: \n", np.round(H, 3), "\n")

print("Error: ", np.linalg.norm(block_matrix - H), "\n")

print(
    "V is unitary: ",
    np.allclose(np.eye(matrix.shape[0]), np.dot(matrix.conjugate().transpose(), matrix)),
)


######################################################################
# We observe that :math:`H` is exactly block-encoded into a larger unitary matrix, up to a normalization factor, and can thus be implemented on a quantum
# computer.
#

######################################################################
# Congrats! We will now see how this block-encoding technique can be used
# for quantum simulation tasks.
#

######################################################################
# Application to quantum simulation
# ---------------------------------
#
# One of the main problems in quantum chemistry is to be able to extract useful information from a quantum
# state. For instance, the energy of a state can be obtained with the quantum phase estimation
# algorithm, which relies on performing time evolution. Trotter-Suzuki decompositions are popular
# techniques to evolve a quantum state by approximating the time-evolution operator using product
# formulas. For example, the first-order product formula for a Hamiltonian of the form
# :math:`H=\sum_{l=1}^{L} H_l`, where the :math:`H_l` terms are only *Hermitian*, is given by
#
# .. math:: e^{-iHt} \approx \prod_{l=1}^{L}e^{-iH_lt}.
#
# While these methods already run in polynomial time, better complexity can be achieved by instead
# expanding the time-evolution operator as a Taylor series (`(Berry et al.) (2014) <https://arxiv.org/abs/1412.4687>`_) and
# truncating it to some order :math:`M`. Hence, we can write
#
# .. math:: e^{-iHt} = \sum_{k=0}^{\infty} \frac{(-iHt)^k}{k!} \approx \sum_{k=0}^{M} \frac{(-iHt)^k}{k!},
#
# which can be brought into an LCU form as above, since :math:`H^k` is Hermitian.
#
# As a concrete example, let us consider a first-order truncation
#
# .. math:: e^{-iHt} \approx \mathbb{1} -iHt,
#
# whose simulation can be recast into the problem we just solved!
#

######################################################################
# Conclusions
# -----------
#
# We learned how to decompose a Hermitian matrix into a linear combination of Pauli operators, which
# can be block-encoded into a larger unitary matrix. This scheme is useful to perform time-evolution
# via Taylor expansion (`(Berry et al. (2014) <https://arxiv.org/abs/1412.4687>`_) , which is a recurring sub routine in quantum algorithms.
#

######################################################################
# References
# -----------
#
# 1. Andrew M. Childs, Nathan Wiebe. "Hamiltonian Simulation Using Linear Combinations of Unitary Operations.",
# `arXiv:1202.5822 <https://arxiv.org/abs/1202.5822>`__ , 2012
#
# 2.  Dominic W. Berry, Andrew M. Childs, Richard Cleve, Robin Kothari and Rolando D. Somma. "Simulating Hamiltonian dynamics with a truncated Taylor series."
# `arXiv:1412.4687 <https://arxiv.org/abs/1412.4687>`__, 2014
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/oriel_kiss.txt

