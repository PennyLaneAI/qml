r"""

Qubit tapering with symmetries
==============================

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

Tapering the molecular Hamiltonian
----------------------------------

A molecular qubit Hamiltonian is constructed as a linear combination of Pauli words as

.. math:: H = \sum_{i=1}^r c_i \eta_i

where :math:`c_i` is a real coefficient and :math:`\eta_i` is a tensor product of Pauli operators.

In PennyLane, a molecular Hamiltonian can be created by specifying the atomic symbols and
coordinates and then creating a molecule object that stores all the molecular parameters needed to
construct the Hamiltonian. We use the `Helium hydride cation
<https://en.wikipedia.org/wiki/Helium_hydride_ion>`__, :math:`\textrm{HeH}^+`, in this tutorial.
"""
import pennylane as qml
from pennylane import numpy as np

symbols = ["He", "H"]
geometry = np.array([[0.00000000, 0.00000000, -0.87818361],
                     [0.00000000, 0.00000000,  0.87818362]])

mol = qml.hf.Molecule(symbols, geometry, charge = 1)

##############################################################################
# Once we have the molecule object, the Hamiltonian is created as

H = qml.hf.generate_hamiltonian(mol)(geometry)
print(H)

##############################################################################
# This Hamiltonian contains 27 terms where each term acts on up to four qubits.
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
# group of Pauli words commuting with each term in :math:`H` and the group does not contain
# :math:`−I` [#bravyi2017]_. Recall that the
# `generators <https://en.wikipedia.org/wiki/Generating_set_of_a_group>`__ of the symmetry group are
# those elements of the group that can be linearly combined, along with their inverses, to create
# any other member of the group.
#
# Once we have the generators, :math:`\tau`, and the Pauli-X operators, the unitary operator
# :math:`U` can be constructed as [#bravyi2017]_
#
# .. math:: U = \Pi_i \left [\frac{1}{\sqrt{2}} \left (\sigma_x^{q_i} + \tau_i \right) \right].
#
# In PennyLane, the generators and the Pauli-X operators are constructed by the
# :func:`~.pennylane.hf.generate_symmetries` function.

generators, paulix_ops = qml.hf.generate_symmetries(H, len(H.wires))
print(f'generator: {generators[0]}, paulix_op: {paulix_ops[0]}')
print(f'generator: {generators[1]}, paulix_op: {paulix_ops[1]}')

##############################################################################
# Once the operator :math:`U` is applied, each of the Hamiltonian terms will act on the qubits
# :math:`1-3` either with Identity or with a Pauli-X operator. For each of these qubits, we can
# simply replace the Pauli-X operator with one of its eigenvalues :math:`+1` or :math:`-1`. This
# results in a total number of :math:`2^n` Hamiltonians each corresponding to one eigenvalue sector.
# The optimal sector corresponding to the ground state energy of the molecule can be obtained from
# the reference Hartree-Fock state and the generated symmetries by using the
# :func:`~.pennylane.hf.optimal_sector` function

paulix_sector = qml.hf.optimal_sector(H, generators, mol.n_electrons)
print(paulix_sector)

##############################################################################
# The optimal eigenvalues are :math:`+1, -1, -1` for qubits :math:`1, 2, 3`, respectively. We can
# now build the tapered Hamiltonian with the :func:`~.pennylane.hf.transform_hamiltonian` function
# which constructs the operator :math:`U`, applies it to the Hamiltonian and finally tapers off the
# qubits :math:`1-3` by replacing the Pauli-X operators acting on those qubits with the optimal
# eigenvalues.

H_tapered = qml.hf.transform_hamiltonian(H, generators, paulix_ops, paulix_sector)
print(H_tapered)

##############################################################################
# The new Hamiltonian has only 9 non-zero terms acting on only 2 qubits! We can verify that the
# original and the tapered Hamiltonian give the ground state energy of the :math:`\textrm{HeH}^+`
# cation, which is :math:`-2.8626948638` Ha computed with the full configuration interaction (FCI)
# method, by diagonalizing the matrix representation of the Hamiltonians in the computational basis.

print(np.linalg.eig(qml.utils.sparse_hamiltonian(H).toarray())[0])
print(np.linalg.eig(qml.utils.sparse_hamiltonian(H_tapered).toarray())[0])

##############################################################################
# Tapering the reference state
# ----------------------------
# The ground state Hartree-Fock energy of :math:`\textrm{HeH}^+` can be computed by directly
# applying the Hamiltonians to the Hartree-Fock state. For the tapered Hamiltonian, this requires
# transforming the Hartree-Fock state with the same symmetries obtained for the original
# Hamiltonian. This reduces the number of qubits in the Hartree-Fock state to match that of the
# tapered Hamiltonian. It can be done with the :func:`~.pennylane.hf.transform_hf`.

state_tapered = qml.hf.transform_hf(
                generators, paulix_ops, paulix_sector, mol.n_electrons, len(H.wires))
print(state_tapered)

##############################################################################
# Recall that the original Hartree-Fock state for the :math:`\textrm{HeH}^+` cation is
# :math:`[1 1 0 0]`. We can now generate the qubit representation of these states and compute the
# Hartree-Fock energies for each Hamiltonian

dev = qml.device('default.qubit', wires=H.wires)
@qml.qnode(dev)
def circuit():
    qml.BasisState(np.array([1, 1, 0, 0]), wires=H.wires)
    return qml.state()
qubit_state = circuit()
HF_energy = qubit_state.T @ qml.utils.sparse_hamiltonian(H).toarray() @ qubit_state
print(f'HF energy: {np.real(HF_energy):.8f} Ha')

dev = qml.device('default.qubit', wires=H_tapered.wires)
@qml.qnode(dev)
def circuit():
    qml.BasisState(np.array([0, 0]), wires=H_tapered.wires)
    return qml.state()
qubit_state = circuit()
HF_energy = qubit_state.T @ qml.utils.sparse_hamiltonian(H_tapered).toarray() @ qubit_state
print(f'HF energy (tapered): {np.real(HF_energy):.8f} Ha')

##############################################################################
# These values are identical to the reference Hartree-Fock energy :math:`-2.8543686493` Ha.
#
# VQE simulation
# --------------
# Finally, we can use the tapered Hamiltonian to perform a VQE simulation and compute the ground
# state energy of the :math:`\textrm{HeH}^+` cation. We use the tapered Hartree-Fock state and build
# a circuit that prepares a qubit coupled-cluster ansatz [#ryabinkin2018] tailored for the
# HeH:math:`^+` cation

dev = qml.device('default.qubit', wires=H_tapered.wires)
@qml.qnode(dev)
def circuit(params):
    qml.BasisState(state_tapered, wires=H_tapered.wires)
    qml.PauliRot(params[2], 'Y',  wires=[0])
    qml.PauliRot(params[1], 'Y',  wires=[1])
    qml.PauliRot(params[0], 'YX', wires=[0, 1])
    return qml.expval(H_tapered)

##############################################################################
# We define an optimizer and the initial values of the circuit parameters and optimize the circuit
# parameters with respect to the ground state energy

optimizer = qml.GradientDescentOptimizer(stepsize=0.5)
params = np.zeros(3)

for n in range(1, 21):
    params, energy = optimizer.step_and_cost(circuit, params)
    print(f'n: {n}, E: {energy:.8f} Ha')

##############################################################################
# The computed energy matches the FCI energy, :math:`-2.8626948638` Ha, and we need only two qubits
# in the simulation.
#
# Conclusions
# -----------
# Molecular Hamiltonians posses symmetries that can be leveraged to taper off qubits in quantum
# computing simulations. This tutorial introduces the PennyLane functionality that can be used for
# qubit tapering based on :math:`\mathbb{Z}_2` symmetries. The procedure of obtaining the tapered
# Hamiltonian and the tapered reference state is straightforward, however, building the wavefunction
# ansatz needs some experience. The qubit coupled-cluster method is recommended as an appropriate
# model for building the variational ansatz.
#
# References
# ----------
#
# .. [#bravyi2017]
#
#     Sergey Bravyi, Jay M. Gambetta, Antonio Mezzacapo, Kristan Temme, "Tapering off qubits to
#     simulate fermionic Hamiltonians". `arXiv:1701.08213 <https://arxiv.org/abs/1701.08213>`__
#
# .. [#setia2019]
#
#     Kanav Setia, Richard Chen, Julia E. Rice, Antonio Mezzacapo, Marco Pistoia, James Whitfield,
#     "Reducing qubit requirements for quantum simulation using molecular point group symmetries".
#     `arXiv:1910.14644 <https://arxiv.org/abs/1910.14644>`__
#
# .. [#ryabinkin2018]
#
#     Ilya G. Ryabinkin, Tzu-Ching Yen, Scott N. Genin, Artur F. Izmaylov, "Qubit coupled-cluster
#     method: A systematic approach to quantum chemistry on a quantum computer".
#     `arXiv:1809.03827 <https://arxiv.org/abs/1809.03827>`__
