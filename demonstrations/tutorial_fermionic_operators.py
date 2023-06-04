r"""

Fermionic Operators
===================

.. meta::
    :property="og:description": Learn how to create and work with fermionic operators
    :property="og:image": https://pennylane.ai/qml/_images/differentiable_HF.png

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe A brief overview of VQE

*Author: Soran Jahangiri — Posted: 01 June 2023. Last updated: 01 June 2023.*

The fermionic creation and annihilation operators are commonly used to describe molecules and spin
systems. The creation operator adds one particle to a given state and the annihilation operator
removes a particle from the state. Imagine a molecule with two orbitals that can each contain one
electron. The quantum state of the molecule can be described by applying creation operators to
create an electron in each orbital. Similarly, applying the annihilation operators to this state
remove the electrons and gives back the original state. These operators have interesting algebraic
properties and commutation relations that make them powerful tools for describing quantum systems
and simulating them with quantum computers. In this tutorial, you will learn how to use PennyLane's
in-built functionalities to build fermionic operators, use them to construct Hamiltonian operators
for interesting systems, and map the resulting operators to the qubit basis so that you can perform
quantum simulations of those systems.

.. figure:: /demonstrations/fermionic_operators/creation.jpg
    :width: 60%
    :align: center

    Caption.

Let's get started!

Constructing fermionic operators
--------------------------------

The fermionic creation and annihilation operators can be easily constructed in PennyLane, similar to
the Pauli operators, with the :class:`~.pennylane.FermiC` and :class:`~.pennylane.FermiA` classes
"""
import pennylane as qml
from pennylane import numpy as np

cr = qml.FermiC(0)
an = qml.FermiA(1)

##############################################################################
# Once created, this operators can be multiplied or added to each other to create new operators that
# we can call *Fermi Word* for the multiplication and *Fermi Sentence* for the summation

fermi_word = cr * an
fermi_sentence = 1.2 * cr * an + 2.4 * cr * an

##############################################################################
# In this simple example, we first created the operator :math:`a^{\dagger}_0 a_1` and then created
# the linear combination :math:`1.2 a^{\dagger}_0 a_1 + 2.4 a^{\dagger}_0 a_1` which is simplified
# to :math:`3.7 a^{\dagger}_0 a_1`. You can also perform arithmetic operations between the Fermi
# word and the Fermi sentence

fermi_sentence = fermi_sentence * fermi_word + 2.3 * fermi_word

##############################################################################
# PennyLane allows several arithmetic operations between these fermionic objects including
# multiplication, summation, subtraction and exponentiation to an integer power. For instance, we
# can create this more complicated operator
#
# .. math::
#
#     1.2 \times a_0^{\dagger} + 0.5 \times a_1 - 2.3 \times \left ( a_0^{\dagger} a_1 \right )^2,
#
# in the same way that you can write down the operator on paper

fermi_sentence = 1.2 * cr + 0.5 * an - 2.3 * (cr * an) ** 2

##############################################################################
# This Fermi sentence can be mapped to the qubit basis and reconstructed as a linear combination
# of Pauli operators with

pauli_sentence = fermi_sentence.to_qubit()

##############################################################################
# Let's learn a bit more about the details of the mapping.
#
# Mapping to Pauli operators
# --------------------------
# What makes fermionic operators particularly interesting is the similarity between them and Pauli
# operators acting on qubit states. The creation operator applied to a single orbital creates one
# electron in the orbital. This can be illustrated with
#
# .. math::
#
#     a^{\dagger} | 0 \rangle = | 1 \rangle.
#
# Similarly, the annihilation operator eliminates the electron
#
# .. math::
#
#     a | 1 \rangle = | 0 \rangle.
#
# What happens if we apply the creation operator to the :math:`| 1 \rangle` state? The Pauli
# exclusion principle tells us that two fermions cannot occupy the same quantum state. This can be
# satisfied by defining :math:`a^{\dagger} | 1 \rangle = a | 0 \rangle = 0`.
#
# The :math:`| 1 \rangle` and :math:`| 0 \rangle` states can be represented as the basis vectors
# that give the quantum states of a qubit.
#
# .. math::
#
#     | 0 \rangle = \begin{bmatrix} 1\\ 0 \end{bmatrix}, \:\:\:\:\:\:
#     \text{and} \:\:\:\:\:\: | 1 \rangle = \begin{bmatrix} 0\\ 1 \end{bmatrix}.
#
#
# Then we can obtain the matrix representation of the fermionic creation and annihilation
# operators
#
# .. math::
#
#     a^{\dagger} | 0 \rangle = \begin{bmatrix} 0 & 0\\  1 & 0 \end{bmatrix} \cdot
#     \begin{bmatrix} 1\\ 0 \end{bmatrix} = | 1 \rangle \:\:\:\:\:\: \text{and} \:\:\:\:\:\:
#      a | 1 \rangle = \begin{bmatrix} 0 & 1\\  0 & 0 \end{bmatrix} \cdot
#     \begin{bmatrix} 0\\ 1 \end{bmatrix} = | 0 \rangle.
#
# By comparing these equations with the application of Pauli operators to qubit states we can simply
# obtain
#
# .. math::
#
#     a^{\dagger} = \frac{X - iY}{2} \:\:\:\:\:\: \text{and} \:\:\:\:\:\: a = \frac{X + iY}{2}
#
# We can still use these relations if we have more than one orbital in our system but in that case
# we need to account for another important property of the fermionic operators. The creation and
# annihilation operators have specific anticommutation relations that we can account for by applying
# a string of Pauli :math:`Z` operators to get
#
# .. math::
#
#     a^{\dagger}_0 =  \left (\frac{X_0 - iY_0}{2}  \right ), \:\: \text{...,} \:\:
#     a^{\dagger}_n = Z_0 \otimes Z_1 \otimes ... \otimes Z_{n-1} \otimes \left (\frac{X_n - iY_n}{2}  \right ),
#
# and
#
# .. math::
#
#     a_0 =  \left (\frac{X_0 + iY_0}{2}  \right ), \:\: \text{...,} \:\:
#     a_n = Z_0 \otimes Z_1 \otimes ... \otimes Z_{n-1} \otimes \left (\frac{X_n + iY_n}{2}  \right ).
#
# This is the Jordan-Wigner formalism for mapping fermionic operators to Pauli operators.
#
# You can verify this with PennyLane for :math:`a^{\dagger}_0` and :math:`a^{\dagger}_10` as simple
# examples

print(qml.FermiC(0).to_qubit())
print()
print(qml.FermiC(10).to_qubit())

##############################################################################
# Remember that for more complicated combinations of fermionic operators the mapping is equally
# simple
fermi_sentence = 1.2 * cr + 0.5 * an - 2.3 * (cr * an) ** 2
fermi_sentence.to_qubit()

##############################################################################
# Now that we have all these nice tools to create and manipulate fermionic Hamiltonians, we can look
# at some interesting systems.
#
# Fermionic Hamiltonians
# ----------------------
# Our first example is a toy Hamiltonian inspired by the
# `Hückel method <https://en.wikipedia.org/wiki/H%C3%BCckel_method>`_ which is a simple method for
# describing molecules with alternating single and double bonds. The Hückel Hamiltonian has the
# general form
#
# .. math::
#
#     H = \sum_{i, j} h_{ij} a^{\dagger}_i a_j,
#
# where :math:`i, j` denote the orbitals of interest, which are typically the :math:`p_z`
# spin-orbitals of the conjugated molecule. The :math:`h_{ij}` coefficients are treated as empirical
# parameters with values of :math:`\alpha` for the diagonal terms and :math:`\beta` for the
# off-diagonal terms.
#
# Our toy model is a simplified version of the Hückel Hamiltonian and assumes only two orbitals and
# a single electron
#
# .. math::
#
#     H = \alpha \left (a^{\dagger}_0 a_0  + a^{\dagger}_1 a_1 \right ) +
#         \beta \left (a^{\dagger}_0 a_1  + a^{\dagger}_1 a_0 \right ).
#
# This Hamiltonian can be constructed with pre-defined values for :math:`\alpha` and :math:`\beta`
alpha = 0.01
beta = -0.02
h = alpha * (FermiC(0) * FermiA(0) + FermiC(1) * FermiA(1)) + \
    beta *  (FermiC(0) * FermiA(1) + FermiC(1) * FermiA(0)).

##############################################################################
# The fermionic Hamiltonian can be converted to the qubit Hamiltonian with

h = h.to_qubit()
print(h)

##############################################################################
# The matrix representation of the qubit Hamiltonian in the computational basis can be diagonalized
# to have
val, vec = np.linalg.eigh(h.sparse_matrix().toarray())
print(val)
print(np.real(vec.T))

##############################################################################
# The energy values of :math:`alpha + beta` and :math:`alpha - beta` correspond to the states
# :math:`- \frac{1}{\sqrt{2}} \left ( |10 \rangle + |01 \rangle \right )` and
# :math:`- \frac{1}{\sqrt{2}} \left ( |10 \rangle + |01 \rangle \right )`, respectively.

# The second quantized molecular electronic Hamiltonian is usually constructed as
#
# .. math::
#     H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} h_{pq} a_{p,\alpha}^{\dagger}
#     a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
#     h_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \beta}^{\dagger} a_{r, \beta} a_{s, \alpha},
#
# where :math:`\sigma` denotes the electron spin and
#
# Conclusions
# -----------
# This tutorial introduces ...
#
# References
# ----------
#
# .. [#szabo1996]
#
#     Attila Szabo, Neil S. Ostlund, "Modern Quantum Chemistry: Introduction to Advanced Electronic
#     Structure Theory". Dover Publications, 1996.
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/soran_jahangiri.txt
