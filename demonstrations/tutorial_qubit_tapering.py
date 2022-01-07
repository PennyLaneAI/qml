r"""

Qubit tapering
==============

.. meta::
    :property="og:description": Learn how to taper off qubits
    :property="og:image": https://pennylane.ai/qml/_images/ qubit_tapering.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE
    tutorial_givens_rotations Givens rotations for quantum chemistry
    tutorial_adaptive_circuits Adaptive circuits for quantum chemistry


*Author: PennyLane dev team. Posted:  2021. Last updated: XX January 2022*

The performance of variational quantum algorithms is considerably limited by the number of qubits
required to prepare the trial wave function ansatz. In the context of quantum chemistry, this
limitation hinders the treatment of large molecules with algorithms such as the variational quantum
eigensolver (VQE). Several approaches have been developed to reduce the qubit requirements for
electronic structure calculations. In this tutorial, we demonstrate the symmetry-based qubit
tapering approach which allows reducing the number of qubits required to perform molecular quantum
simulations based on the :math:`\mathbb{Z}_2` symmetries present in molecular Hamiltonians
[#bravyi2017]_ [#setia2019]_.

Qubit tapering with symmetries
------------------------------

A molecular qubit Hamiltonian is constructed as a linear combination of Pauli strings as

.. math:: H = \sum_{i=1}^r c_i \eta_i

where :math:`c_i` is a real coefficient and :math:`\eta_i` is a Pauli string acting on a number of
qubits.

In PennyLane, a molecular Hamiltonian can be created by specifying the atomic symbols and
coordinates and then creating a molecule object that stores all the molecular parameters needed to
construct the Hamiltonian
"""
import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "H"]
geometry = np.array([[-0.672943567415407, 0.0, 0.0],
                     [ 0.672943567415407, 0.0, 0.0]], requires_grad=True)
mol = qml.hf.Molecule(symbols, geometry)

##############################################################################
# Once we have the molecule object, the Hamiltonian is created as

H = qml.hf.generate_hamiltonian(mol)(geometry)
print(H)

##############################################################################
# This Hamiltonian contains 15 terms where each term acts on one to four qubits.
#
# The key step in the qubit tapering approach is to find a unitary operator :math:`U` that is
# applied to :math:`H` to provide a transformed Hamiltonian :math:`H'`
#
# .. math:: H' = U^{\dagger} H U = \sum_{i=1}^r c_i \mu_i,
#
# such that each :math:`\mu_i` term acts trivially, with an identity operator or at most one Pauli
# operator, on a set of :math:`k` qubits. This guarantees that each of the Pauli operators, applied
# to the :math:`k`-th qubit, commutes with each term of the transformed Hamiltonian and therefore
# they commute with the transformed Hamiltonian as well.
#
# .. math:: [H', \sigma_x^{q(i)}] = 0.
#
# Recall that two commuting operators share an eigenbasis and the molecular wavefunction is an
# eigenfunction of both the transformed Hamiltonian and each of the Pauli operators, applied to the
# :math:`k`-th qubit. Then we can factor out those Pauli operators from the transformed Hamiltonian
# and replace them with their eigenvalues which are :math:`\pm 1`. This gives us a tapered
# Hamiltonian in which the set of :math:`k` qubits are eliminated.`
#
# The unitary operator :math:`U` is constructed from the generators of the symmetry group of
# :math:`H` and a set of Pauli-X operators which act on the qubits that will be ultimately tapered
# off from the Hamiltonian [#bravyi2017]_. The symmetry group of the Hamiltonian is defined as a
# group of pauli words commuting with each term in :math:`H` and the group does not contain
# :math:`−I` [#bravyi2017]_. Recall that the
# `generators <https://en.wikipedia.org/wiki/Generating_set_of_a_group>`__ of the symmetry group are
# those elements of the group that can be linearly combined, along with their inverses, to create
# any other member of the group.
#
# Once we have the generators, :math:`\tau`, and the Pauli-X operators, the unitary operator
# :math:`U` can be constructed as
#
# .. math:: U = \Pi_i \left [\frac{1}{\sqrt{2}} \left (\sigma_x^{q_i} + \tau_i \right) \right].
#
# In PennyLane, the generators and the Pauli-X operators are constructed by the
# :func:`~.pennylane.hf.generate_symmetries` function.

generators, paulix_ops = qml.hf.generate_symmetries(H, len(H.wires))

##############################################################################
# Once the operator :math:`U` is applied, each of the Hamiltonian terms will act on the qubits
# :math:`1-3` either with Identity or with a Pauli-X operator. For each of these qubits, we can
# simply replace the Pauli-X operator with one of its eigenvalues :math:`+1` or :math:`-1`. This
# results in a total number of :math:`2^n` Hamiltonians each corresponding to one eigenvalue sector.
# The optimal sector corresponding to the ground state energy of the molecule can be obtained from
# the reference Hartree-Fock state and the generated symmetries by using the
# :func:`~.pennylane.hf.optimal_sector` function

# active_electrons = 2
# paulix_sector = qml.hf.optimal_sector(H, generators, active_electrons)
# print(paulix_sector)

##############################################################################
# The optimal eigenvalues are :math:`+1, -1, -1` for qubits :math:`1, 2, 3`, respectively. We can
# now build the tapered Hamiltonian with the :func:`~.pennylane.hf.transform_hamiltonian` function
# which constructs the operator :math:`U`, applies it to the Hamiltonian and finally tapers off the
# qubits :math:`1-3` by replacing the Pauli-X operators acting on those qubits with the optimal
# eigenvalues.

# H_tapered = qml.hf.transform_hamiltonian(H, generators, paulix_ops, paulix_sector)
# print(H_tapered)

##############################################################################
# The new Hamiltonian has only three non-zero terms acting on only 1 wire! We can verify that the
# original and the tapered Hamiltonian have similar eigenvalues by diagonalizing the matrix
# representation of the Hamiltonians in the computational basis.

print(np.linalg.eig(qml.utils.sparse_hamiltonian(H).toarray())[0])
# print(np.linalg.eig(qml.utils.sparse_hamiltonian(H_tapered).toarray())[0])

##############################################################################
# We can also compute the Hartree-Fock of the ground state by directly applying the tapered
# Hamiltonian to the reference Hartree-Fock state. This requires transforming the Hartree-Fock state
# with the same symmetries obtained for the Hamiltonian and reduce the number of qubits in the
# Hartree-Fock state to match that of the Hamiltonian. This can be done with the
# :func:`~.pennylane.hf.transform_hf`.
#
##############################################################################
# Conclusions
# -----------
#
#
# References
# ----------
#
# .. [#bravyi2017]
#
#     Sergey Bravyi, Jay M. Gambetta, Antonio Mezzacapo, Kristan Temme, "Tapering off qubits to
#     simulate fermionic Hamiltonians". `arXiv:1701.08213
#     <https://arxiv.org/abs/1701.08213>`__
#
# .. [#setia2019]
#
#     Kanav Setia, Richard Chen, Julia E. Rice, Antonio Mezzacapo, Marco Pistoia, James Whitfield,
#     "Reducing qubit requirements for quantum simulation using molecular point group symmetries".
#     `arXiv:1910.14644 <https://arxiv.org/abs/1910.14644>`__
