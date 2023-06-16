r"""

Fermionic operators
===================

.. meta::
    :property="og:description": Learn how to work with fermionic operators
    :property="og:image": https://pennylane.ai/qml/_images/differentiable_HF.png

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe A brief overview of VQE

*Author: Soran Jahangiri — Posted: 01 June 2023. Last updated: 01 June 2023.*

Fermionic creation and annihilation operators are commonly used to construct Hamiltonians and other
observables of molecules and spin systems [#surjan]_. In this demo, you will learn how to use
PennyLane to create fermionic operators and Hamiltonians, and map the resulting operators to

a qubit representation for use in quantum algorithms.

.. figure:: /demonstrations/fermionic_operators/creation.jpg
    :width: 60%
    :align: center

    Caption.

Constructing fermionic operators
--------------------------------
The fermionic `creation and annihilation <https://en.wikipedia.org/wiki/Creation_and_annihilation_operators>`_
operators can be constructed in PennyLane similarly to Pauli operators using
:class:`~.pennylane.FermiC` (creation) and :class:`~.pennylane.FermiA` (annihilation).
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane import FermiC, FermiA

c0 = FermiC(0)
a1 = FermiA(1)

##############################################################################
# We used the compact notations :math:`c0` and :math:`a1` to denote a creation operator applied to
# the 0th orbital and an annihilation operator applied to the 1st one, respectively. Once created,
# these operators can be multiplied or added to each other to create new operators. A product of
# fermionic operators is called a *Fermi word*, and linear combination of Fermi words is called a
# *Fermi sentence*.

fermi_word = c0 * a1
fermi_sentence = 1.2 * c0 * a1 + 2.4 * c0 * a1
fermi_sentence

##############################################################################
# In this simple example, we first created the operator :math:`a^{\dagger}_0 a_1` and then created
# the linear combination :math:`1.2 a^{\dagger}_0 a_1 + 2.4 a^{\dagger}_0 a_1`, which is
# automatically simplified to :math:`3.6 a^{\dagger}_0 a_1`. You can also perform arithmetic
# operations between Fermi words and Fermi sentences.

fermi_sentence = fermi_sentence * fermi_word + 2.3 * fermi_word
fermi_sentence

##############################################################################
# Beyond multiplication, summation, and subtraction, we can exponentiate fermionic operators in
# PennyLane to an integer power. For instance, we can create a more complicated operator.
#
# .. math::
#
#     1.2 \times a_0^{\dagger} + 0.5 \times a_1 - 2.3 \times \left ( a_0^{\dagger} a_1 \right )^2,
#
# in the same way that you would write down the operator on a piece of paper.

fermi_sentence = 1.2 * c0 + 0.5 * a1 - 2.3 * (c0 * a1) ** 2
fermi_sentence

##############################################################################
# This Fermi sentence can be mapped to the qubit basis using the
# `Jordan-Wigner <https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation>`_
# transformation to get a linear combination of Pauli operators.

pauli_sentence = qml.jordan_wigner(fermi_sentence)
pauli_sentence

##############################################################################
# Fermionic Hamiltonians
# ----------------------
# Fermi words and sentences can be thought of as fermionic Hamiltonians. Now that we have nice
# tools to create and manipulate fermionic operators, we can build some interesting fermionic
# Hamiltonians.
#
# A toy model
# ^^^^^^^^^^^
# Our first example is a toy Hamiltonian inspired by the
# `Hückel method <https://en.wikipedia.org/wiki/H%C3%BCckel_method>`_, which is a simple method for
# describing molecules with alternating single and double bonds. Our toy model is a simplified
# version of the Hückel Hamiltonian and assumes only two orbitals and a single electron.
#
# .. math::
#
#     H = \alpha \left (a^{\dagger}_0 a_0  + a^{\dagger}_1 a_1 \right ) +
#         \beta \left (a^{\dagger}_0 a_1  + a^{\dagger}_1 a_0 \right ).
#
# This Hamiltonian can be constructed with pre-defined values for :math:`\alpha` and :math:`\beta`.

c1 = FermiC(1)
a0 = FermiA(0)

alpha = 0.01
beta = -0.02
h = alpha * (c0 * a0 + c1 * a1) + beta * (c0 * a1 + c1 * a0)

##############################################################################
# The fermionic Hamiltonian can be converted to the qubit Hamiltonian with

h = qml.jordan_wigner(h)
print(h)

##############################################################################
# The matrix representation of the qubit Hamiltonian in the computational basis can be diagonalized
# to get

val, vec = np.linalg.eigh(h.sparse_matrix().toarray())
print(val)
print(np.real(vec.T))

##############################################################################
# The eigenvalues of :math:`\alpha + \beta` and :math:`\alpha - \beta` correspond to the states
# :math:`- \frac{1}{\sqrt{2}} \left ( |10 \rangle + |01 \rangle \right )` and
# :math:`- \frac{1}{\sqrt{2}} \left ( |10 \rangle + |01 \rangle \right )`, respectively.
#
# Hydrogen molecule
# ^^^^^^^^^^^^^^^^^
# The `second quantized <https://en.wikipedia.org/wiki/Second_quantization>`_ molecular electronic
# Hamiltonian is usually constructed as
#
# .. math::
#     H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} h_{pq} a_{p,\alpha}^{\dagger}
#     a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
#     h_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \beta}^{\dagger} a_{r, \beta} a_{s, \alpha},
#
# where :math:`\sigma` denotes the electron spin and the coefficients :math:`h` are integrals over
# molecular orbitals that are obtained from
# `Hartree-Fock <https://pennylane.ai/qml/demos/tutorial_differentiable_HF#the-hartree-fock-method>`_
# calculations. These integrals can be
# computed with PennyLane using the :func:`~.pennylane.qchem.electron_integrals` function. Let's use
# the hydrogen molecule as an example. We first define the atom types and the atomic coordinates.

symbols = ["H", "H"]
geometry = np.array(
    [[-0.672943567415407, 0.0, 0.0], [0.672943567415407, 0.0, 0.0]], requires_grad=False
)

mol = qml.qchem.Molecule(symbols, geometry)
core, one, two = qml.qchem.electron_integrals(mol)()

##############################################################################
# These integrals are computed over molecular orbitals and we need to extend them to account for
# different spins.

one_spin = one.repeat(2, axis=0).repeat(2, axis=1)
two_spin = two.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2).repeat(2, axis=3) * 0.5

##############################################################################
# We can now construct the fermionic Hamiltonian for the hydrogen molecule. We construct the
# Hamiltonian by using fermionic arithmetic operations directly. The one-body terms can be
# added first.

import itertools

n = one_spin.shape[0]

h = 0.0

for i, j in itertools.product(range(n), repeat=2):
    if i % 2 == j % 2:  # to account for spin-forbidden terms
        h += FermiC(i) * FermiA(j) * one_spin[i, j]

##############################################################################
# The two-body terms can be added with

for p, q, r, s in itertools.product(range(n), repeat=4):
    if p % 2 == s % 2 and q % 2 == r % 2:  # to account for spin-forbidden terms
        h += FermiC(p) * FermiC(q) * FermiA(r) * FermiA(s) * two_spin[p, q, r, s]

##############################################################################
# We then simplify the Hamiltonian to remove terms with negligible coefficients and then map it to
# the qubit basis.

h.simplify()
h = qml.jordan_wigner(h)

##############################################################################
# We also need to account for the contribution of the nuclear energy.

h += np.sum(core * qml.Identity(0))

##############################################################################
# This gives us the qubit Hamiltonian which can be used as an input for quantum algorithms. We can
# also compute the ground-state energy by diagonalizing the matrix representation of the
# Hamiltonian in the computational basis.

np.linalg.eigh(h.sparse_matrix().toarray())[0].min()

##############################################################################
# Conclusions
# -----------
# This demo explains how to create and manipulate fermionic operators in PennyLane, which is as
# easy as writing the operators on paper. PennyLane supports several arithmetic operations between
# fermionic operators and provides tools for mapping them to the qubit basis. This makes it easy and
# intuitive to construct complicated fermionic Hamiltonians such as
# `molecular Hamiltonians <https://pennylane.ai/qml/demos/tutorial_quantum_chemistry>`_.
#
# References
# ----------
#
# .. [#surjan]
#
#     Peter R. Surjan, "Second Quantized Approach to Quantum Chemistry". Springer-Verlag, 1989.
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/soran_jahangiri.txt
