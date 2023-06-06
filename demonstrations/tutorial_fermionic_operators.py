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

These creation and annihilation operators have interesting algebraic properties and commutation
relations that make them powerful tools for describing quantum systems.
The fermionic creation and annihilation operators are commonly used to construct Hamiltonians of
molecules and spin systems. The creation operator adds one particle to a given state and the
annihilation operator removes a particle from the state. Imagine a molecule with two orbitals that
can each contain one electron. The quantum state of the molecule can be described by applying
creation operators to add an electron to each orbital. Similarly, applying the annihilation
operators to this state remove the electrons and gives back the original state.
In this demo, you will learn how to use PennyLane's
in-built functionalities to create fermionic operators, use them to construct Hamiltonian operators
for interesting systems, and map the resulting operators to the qubit basis so that you can perform
quantum simulations of those systems.

.. figure:: /demonstrations/fermionic_operators/creation.jpg
    :width: 60%
    :align: center

    Caption.

Constructing fermionic operators
--------------------------------
The fermionic creation and annihilation operators can be easily constructed in PennyLane, similar to
the Pauli operators, with the :class:`~.pennylane.FermiC` and :class:`~.pennylane.FermiA` classes
"""

import pennylane as qml
from pennylane import numpy as np

c0 = qml.FermiC(0)
a1 = qml.FermiA(1)

##############################################################################
# Once created, this operators can be multiplied or added to each other to create new operators that
# we can call *Fermi Word* for the multiplication and *Fermi Sentence* for the summation

fermi_word = c0 * a1
fermi_sentence = 1.2 * c0 * a1 + 2.4 * c0 * a1
fermi_sentence

##############################################################################
# In this simple example, we first created the operator :math:`a^{\dagger}_0 a_1` and then created
# the linear combination :math:`1.2 a^{\dagger}_0 a_1 + 2.4 a^{\dagger}_0 a_1` which is simplified
# to :math:`3.7 a^{\dagger}_0 a_1`. You can also perform arithmetic operations between the Fermi
# word and the Fermi sentence

fermi_sentence = fermi_sentence * fermi_word + 2.3 * fermi_word
fermi_sentence

##############################################################################
# PennyLane allows several arithmetic operations between these fermionic objects including
# multiplication, summation, subtraction and exponentiation to an integer power. For instance, we
# can create a more complicated operator
#
# .. math::
#
#     1.2 \times a_0^{\dagger} + 0.5 \times a_1 - 2.3 \times \left ( a_0^{\dagger} a_1 \right )^2,
#
# in the same way that you can write down the operator on paper

fermi_sentence = 1.2 * c0 + 0.5 * a1 - 2.3 * (c0 * a1) ** 2
fermi_sentence

##############################################################################
# This Fermi sentence can be mapped to the qubit basis to get a linear combination
# of Pauli operators with

pauli_sentence = fermi_sentence.to_qubit()
pauli_sentence

##############################################################################
# Fermionic Hamiltonians
# ----------------------
# Now that we have all these nice tools to create and manipulate fermionic Hamiltonians, we can look
# at some interesting examples.
#
# A toy model
# ^^^^^^^^^^^
# Our first example is a toy Hamiltonian inspired by the
# `Hückel method <https://en.wikipedia.org/wiki/H%C3%BCckel_method>`_ which is a simple method for
# describing molecules with alternating single and double bonds. The Hückel Hamiltonian has the
# general form [#surjan]_
#
# .. math::
#
#     H = \sum_{i, j} h_{ij} a^{\dagger}_i a_j,
#
# where :math:`i, j` denote the orbitals of interest which are typically the :math:`p_z`
# spin-orbitals. The :math:`h_{ij}` coefficients are treated as empirical
# parameters with values :math:`\alpha` for the diagonal terms and :math:`\beta` for the
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

c1 = qml.FermiC(1)
a0 = qml.FermiA(0)

alpha = 0.01
beta = -0.02
h = alpha * (c0 * a0 + c1 * a1) + beta * (c0 * a1 + c1 * a0)

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
# The energy values of :math:`\alpha + \beta` and :math:`\alpha - \beta` correspond to the states
# :math:`- \frac{1}{\sqrt{2}} \left ( |10 \rangle + |01 \rangle \right )` and
# :math:`- \frac{1}{\sqrt{2}} \left ( |10 \rangle + |01 \rangle \right )`, respectively.
#
# Hydrogen molecule
# ^^^^^^^^^^^^^^^^^
# The second quantized molecular electronic Hamiltonian is usually constructed as
#
# .. math::
#     H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} h_{pq} a_{p,\alpha}^{\dagger}
#     a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
#     h_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \beta}^{\dagger} a_{r, \beta} a_{s, \alpha},
#
# where :math:`\sigma` denotes the electron spin and the coefficients :math:`h` are integrals over
# molecular orbitals that are obtained from Hartree-Fock calculations. These integrals can be
# computed with PennyLane using the :func:`~.pennylane.qchem.electron_integrals` function. Let's use
# the hydrogen molecule as an example. We first define the atom types and the atomic coordinates

symbols = ["H", "H"]
geometry = np.array([[-0.672943567415407, 0.0, 0.0], [0.672943567415407, 0.0, 0.0]])

mol = qml.qchem.Molecule(symbols, geometry)
coef, one, two = qml.qchem.electron_integrals(mol)()

##############################################################################
# These integrals are computed over molecular orbitals and we need to extend them to account for
# different spins

one_spin = one.repeat(2, axis=0).repeat(2, axis=1)
two_spin = two.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2).repeat(2, axis=3) * 0.5

##############################################################################
# We can now construct the fermionic Hamiltonian for the hydrogen molecule

import itertools
from pennylane import FermiC, FermiA

n = one_spin.shape[0]

h = FermiSentence({FermiWord({}): coef[0]})  # initialize with the identity term

for i, j in itertools.product(range(n), repeat=2):
    if i % 2 == j % 2:  # to account for spin-forbidden terms
        h += FermiC(i) * FermiA(j) * one_spin[i, j]
for p, q, r, s in itertools.product(range(n), repeat=4):
    if p % 2 == s % 2 and q % 2 == r % 2:  # to account for spin-forbidden terms
        h += FermiC(p) * FermiC(q) * FermiA(r) * FermiA(s) * two_spin[p, q, r, s]

##############################################################################
# We simplify the Hamiltonian to remove terms with negligible coefficients and then mapp it to the
# qubit basis

h.simplify()
h = h.to_qubit().hamiltonian()

##############################################################################
# This gives us the qubit Hamiltonian which can be used in, for example, VQE simulations. We can
# also compute the ground-state energy by diagonalizing the matrix representation of the Hamiltonian
# in the computational basis.

qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), h.wires))

##############################################################################
# Mapping to Pauli operators
# --------------------------
# Let's learn a bit more about the details of the mapping. What makes fermionic operators
# particularly interesting is the similarity between them and Pauli
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
# Conclusions
# -----------
# This demo explains how to create and manipulate fermionic operators in PennyLane. This is as
# easy as writing the operators on paper. PennyLane supports several arithmetic operations between
# fermionic operators and provides tools for mapping them to the qubit basis. This makes it easy and
# intuitive to construct complicated fermionic Hamiltonians such as molecular Hamiltonians.
#
# References
# ----------
#
# .. [#surjan]
#
#     Peter R. Surjan, "Second Quantized Approach to Quantum Chemistry". Dover Publications, 1989.
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/soran_jahangiri.txt
