r"""

Fermionic operators
===================

.. meta::
    :property="og:description": Learn how to work with fermionic operators
    :property="og:image": https://pennylane.ai/qml/_images/fermionic_operators.jpg

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe A brief overview of VQE

*Author: Soran Jahangiri — Posted: 27 June 2023. Last updated: 27 June 2023.*

Fermionic creation and annihilation operators are commonly used to construct
`Hamiltonians <https://codebook.xanadu.ai/H.3>`_ and other observables of molecules and spin
systems [#surjan]_. In this demo, you will learn how to use PennyLane to create fermionic operators
and Hamiltonians, then map them to a qubit representation for use in quantum
algorithms.


.. figure:: /demonstrations/fermionic_operators/fermionic_operators
    :width: 60%
    :align: center

    Caption.

Constructing fermionic operators
--------------------------------
The fermionic `creation and annihilation <https://en.wikipedia.org/wiki/Creation_and_annihilation_operators>`_
operators can be constructed in PennyLane similarly to Pauli operators by using
:class:`~.pennylane.FermiC` (creation) and :class:`~.pennylane.FermiA` (annihilation).
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane import FermiC, FermiA

at0 = FermiC(0)
a1 = FermiA(1)

print(at0)
print(a1)

##############################################################################
# We used the compact notations :math:`at0` and :math:`a1` to denote a creation operator applied to
# the 0th orbital and an annihilation operator applied to the 1st one, respectively. Once created,
# these operators can be multiplied or added to each other to create new operators. A product of
# fermionic operators will be called a *Fermi word* and a linear combination of Fermi words will be
# called a *Fermi sentence*.

fermi_word = at0 * a1
fermi_sentence = 1.3 * at0 * a1 + 2.4 * at0 * a1
fermi_sentence

##############################################################################
# In this simple example, we first created the operator :math:`a^{\dagger}_0 a_1` and then created
# the linear combination :math:`1.3 a^{\dagger}_0 a_1 + 2.4 a^{\dagger}_0 a_1`, which is
# automatically simplified to :math:`3.7 a^{\dagger}_0 a_1`. You can also perform arithmetic
# operations between Fermi words and Fermi sentences.

fermi_sentence = fermi_sentence * fermi_word + 2.3 * fermi_word
fermi_sentence

##############################################################################
# Beyond multiplication, summation, and subtraction, we can exponentiate fermionic operators in
# PennyLane to an integer power. For instance, we can create a more complicated operator
#
# .. math::
#
#     1.2 \times a_0^{\dagger} + 0.5 \times a_1 - 2.3 \times \left ( a_0^{\dagger} a_1 \right )^2,
#
# in the same way that you would write down the operator on a piece of paper:

fermi_sentence = 1.2 * at0 + 0.5 * a1 - 2.3 * (at0 * a1) ** 2
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
# Now that we have nice tools to create and manipulate fermionic operators, we can build some
# interesting fermionic Hamiltonians.
#
# A toy model
# ^^^^^^^^^^^
# Our first example is a toy Hamiltonian inspired by the
# `Hückel method <https://en.wikipedia.org/wiki/H%C3%BCckel_method>`_, which is a method for
# describing molecules with alternating single and double bonds. Our toy model is a simplified
# version of the Hückel Hamiltonian and assumes only two orbitals and a single electron.
#
# .. math::
#
#     H = \alpha \left (a^{\dagger}_0 a_0  + a^{\dagger}_1 a_1 \right ) +
#         \beta \left (a^{\dagger}_0 a_1  + a^{\dagger}_1 a_0 \right ).
#
# This Hamiltonian can be constructed with pre-defined values for :math:`\alpha` and :math:`\beta`.

at1 = FermiC(1)
a0 = FermiA(0)

alpha = 0.01
beta = -0.02
h = alpha * (at0 * a0 + at1 * a1) + beta * (at0 * a1 + at1 * a0)

##############################################################################
# The fermionic Hamiltonian can be converted to the qubit Hamiltonian with

h = qml.jordan_wigner(h)

##############################################################################
# The matrix representation of the qubit Hamiltonian in the computational basis can be diagonalized
# to get its eigenpairs.

val, vec = np.linalg.eigh(h.sparse_matrix().toarray())
print(f"eigenvalues:\n{val}")
print()
print(f"eigenvectors:\n{np.real(vec.T)}")

##############################################################################
#
# Hydrogen molecule
# ^^^^^^^^^^^^^^^^^
# The `second quantized <https://en.wikipedia.org/wiki/Second_quantization>`_ molecular electronic
# Hamiltonian is usually constructed as
#
# .. math::
#     H = \sum_{\alpha \in \{\uparrow, \downarrow \} } \sum_{pq} c_{pq} a_{p,\alpha}^{\dagger}
#     a_{q, \alpha} + \frac{1}{2} \sum_{\alpha, \beta \in \{\uparrow, \downarrow \} } \sum_{pqrs}
#     c_{pqrs} a_{p, \alpha}^{\dagger} a_{q, \beta}^{\dagger} a_{r, \beta} a_{s, \alpha},
#
# where :math:`\alpha` and :math:`\beta` denote the electron spin, :math:`p, q, r, s` are the
# orbital indices and the coefficients :math:`c` are integrals over
# molecular orbitals that are obtained from :doc:`Hartree-Fock </demos/tutorial_differentiable_HF>`
# calculations. These integrals can be computed with PennyLane using the
# :func:`~.pennylane.qchem.electron_integrals` function. Let's build the molecular Hamiltonian for
# the hydrogen molecule as an example. We first define the atom types and the atomic coordinates.

symbols = ["H", "H"]
geometry = np.array([[-0.67294, 0.0, 0.0], [0.67294, 0.0, 0.0]], requires_grad=False)

##############################################################################
# Then we compute the one- and two-electron integrals, which are the coefficients :math:`c` in the
# second quantized molecular Hamiltonian defined above. We also obtain the core constant, which is
# later used to calculate the contribution of the nuclear energy to the Hamiltonian.

mol = qml.qchem.Molecule(symbols, geometry)
core, one, two = qml.qchem.electron_integrals(mol)()

##############################################################################
# These integrals are computed over molecular orbitals. Each molecular orbital contains a pair of
# electrons with different spins. We have assumed that the spatial distribution of these electron
# pairs is the same to simplify the calculation of the integrals. However, to properly account for
# all electrons, we need to duplicate the integrals for electrons with the same spin. For example,
# the :math:`pq` integral, which is the integral over orbital :math:`p` and orbital :math:`q`, is
# for both spin-up and spin-down electrons. Then, if we have a :math:`2 \times 2` matrix of such
# integrals, it will become a :math:`4 \times 4` matrix. The code block below simply extends the
# integrals by duplicating terms to support both spin-up and spin-down electrons.

for i in range(4):
    if i < 2:
        one = one.repeat(2, axis=i)
    two = two.repeat(2, axis=i)

##############################################################################
# We can now construct the fermionic Hamiltonian for the hydrogen molecule. We construct the
# Hamiltonian by using fermionic arithmetic operations directly. The one-body terms, which are the
# first part in the Hamiltonian above, can be added first. We will use _itertools_ to efficiently
# create all the combinations we need. Some of these combinations are not allowed because of spin
# restrictions and we need to exclude these combinations. You can find more details about
# constructing a molecular Hamiltonian in reference [#surjan]_.

import itertools

n = one.shape[0]

h = 0.0

for p, q in itertools.product(range(n), repeat=2):
    if p % 2 == q % 2:  # to account for spin-forbidden terms
        h += one[p, q] * FermiC(p) * FermiA(q)

##############################################################################
# The two-body terms can be added with

for p, q, r, s in itertools.product(range(n), repeat=4):
    if p % 2 == s % 2 and q % 2 == r % 2:  # to account for spin-forbidden terms
        h += two[p, q, r, s] / 2 * FermiC(p) * FermiC(q) * FermiA(r) * FermiA(s)

##############################################################################
# We then simplify the Hamiltonian to remove terms with negligible coefficients and then map it to
# the qubit basis.

h.simplify()
h = qml.jordan_wigner(h)

##############################################################################
# We also need to include the contribution of the nuclear energy.

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
# :doc:`molecular Hamiltonians </demos/tutorial_quantum_chemistry>`.
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
