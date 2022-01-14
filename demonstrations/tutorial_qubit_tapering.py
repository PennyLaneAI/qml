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

A molecular Hamiltonian in the qubit basis is constructed as a linear combination of Pauli words
as

.. math:: H = \sum_{i=1}^r c_i \eta_i

where :math:`c_i` is a real coefficient and :math:`\eta_i` is a tensor product of Pauli and
Identity operators acting on M qubits

.. math:: \eta_i \in \pm \left \{ I, \sigma_x, \sigma_y, \sigma_z \right \} ^ {\bigotimes M}.

The main idea in the symmetry-based qubit tapering approach is to find a unitary operator :math:`U`
that transforms :math:`H` to a new Hamiltonian :math:`H'` which has the same eigenvalues as
:math:`H`

.. math:: H' = U^{\dagger} H U = \sum_{i=1}^r c_i \mu_i,

such that each :math:`\mu_i` term in the new Hamiltonian acts trivially, e.g., with an Identity
operator, on a set of qubits. This allows tapering-off those qubits from the Hamiltonian. We can
also construct the unitary :math:`U` such that each :math:`\mu_i` term acts with a Pauli-X operator
on a set of qubits :math:`\left \{ q_j \right \}, j \in \left \{ l, ..., k \right \}`. This
guarantees that each term of the transformed Hamiltonian commutes with the Pauli-X operator applied
to :math:`j`-th qubit and

.. math:: [H', \sigma_x^{q_j}] = 0.

Recall that two commuting operators share an eigenbasis. This means that the eigenfunction of the
transformed Hamiltonian :math:`H'` is also an eigenfunction of each of the :math:`\sigma_x^{q_j}`
operators. As a result, we can factor out the :math:`\sigma_x^{q_j}`
operators from the transformed Hamiltonian and replace them with their eigenvalues which are
:math:`\pm 1`. This gives us a tapered Hamiltonian in which the set of
:math:`\left \{ q_j \right \}, j \in \left \{ l, ..., k \right \}` qubits are eliminated.

The unitary operator :math:`U` can be constructed as a Clifford operator [#bravyi2017]_

.. math:: U = \Pi_j \left [\frac{1}{\sqrt{2}} \left (\sigma_x^{q(j)} + \tau_j \right) \right],

where :math:`\tau` denotes the generators of the symmetry group of :math:`H` and
:math:`\sigma_x^{q}` operators which act on those qubits that will be ultimately tapered off from
the Hamiltonian.

The symmetry group of the Hamiltonian is defined as an abelian group of pauli words that commute
with each term in the Hamiltonian (excluding :math:`−I`). The
`generators <https://en.wikipedia.org/wiki/Generating_set_of_a_group>`__ of the symmetry group are
those elements of the group that can be linearly combined, along with their inverses, to create any
other member of the group. The symmetry group and the generators of the Hamiltonian can be obtained
from the binary matrix representation of the Hamiltonian [#bravyi2017]_ with the functions of the
PennyLane :py:mod:`~.pennylane.hf.tapering` module.

Let's use the qubit tapering method and obtain the ground state energy of the `Helium hydride
cation <https://en.wikipedia.org/wiki/Helium_hydride_ion>`__ :math:`\textrm{HeH}^+`.

Tapering the molecular Hamiltonian
----------------------------------

In PennyLane, a molecular Hamiltonian can be created by specifying the atomic symbols and
coordinates and then creating a molecule object that stores all the molecular parameters needed to
construct the Hamiltonian.
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
# We can now obtain the symmetry generators and the :math:`\sigma_x^{q_j}` operators that are
# applied to the qubits that we will be tapering-off from the Hamiltonian. In PennyLane, these are
# constructed by using the :func:`~.pennylane.hf.generate_symmetries` function.

generators, paulix_ops = qml.hf.generate_symmetries(H, len(H.wires))
print(f'generator: {generators[0]}, paulix_op: {paulix_ops[0]}')
print(f'generator: {generators[1]}, paulix_op: {paulix_ops[1]}')

##############################################################################
# The generators and the Pauli-X operators are used to construct the unitary :math:`U` operator that
# transforms the :math:`\textrm{HeH}^+` Hamiltonian. Once the operator :math:`U` is applied, each
# of the Hamiltonian terms will act on the qubits :math:`0, 1` either with Identity or with a
# Pauli-X operator. For each of these qubits, we can simply replace the Pauli-X operator with one
# of its eigenvalues :math:`+1` or :math:`-1`. This results in a total number of :math:`2^n`
# Hamiltonians each corresponding to one eigenvalue sector. The optimal sector corresponding to the
# ground state energy of the molecule can be obtained from the reference Hartree-Fock state and the
# generated symmetries by using the :func:`~.pennylane.hf.optimal_sector` function

paulix_sector = qml.hf.optimal_sector(H, generators, mol.n_electrons)
print(paulix_sector)

##############################################################################
# The optimal eigenvalues are :math:`-1, -1` for qubits :math:`0, 1`, respectively. We can now build
# the tapered Hamiltonian with the :func:`~.pennylane.hf.transform_hamiltonian` function which
# constructs the operator :math:`U`, applies it to the Hamiltonian and finally tapers off the
# qubits :math:`0, 1` by replacing the Pauli-X operators acting on those qubits with the optimal
# eigenvalues.

H_tapered = qml.hf.transform_hamiltonian(H, generators, paulix_ops, paulix_sector)
print(H_tapered)

##############################################################################
# The new Hamiltonian has only 9 non-zero terms acting on only 2 qubits! We can verify that the
# original and the tapered Hamiltonian both give the correct ground state energy of the
# :math:`\textrm{HeH}^+` cation, which is :math:`-2.8626948638` Ha computed with the full
# configuration interaction (FCI) method, by diagonalizing the matrix representation of the
# Hamiltonians in the computational basis.

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
# Finally, we can use the tapered Hamiltonian and the tapered references state to perform a VQE
# simulation and compute the ground state energy of the :math:`\textrm{HeH}^+` cation. We use the
# tapered Hartree-Fock state to build a circuit that prepares a variational qubit coupled-cluster
# ansatz [#ryabinkin2018] tailored for the HeH:math:`^+` cation

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
# The computed energy matches the FCI energy, :math:`-2.8626948638` Ha, while the number of qubits
# and the Hamiltonian terms is significantly reduced with respect to their original values.
#
# Conclusions
# -----------
# Molecular Hamiltonians posses symmetries that can be leveraged to taper off qubits in quantum
# computing simulations. This tutorial introduces the PennyLane functionality that can be used for
# qubit tapering based on :math:`\mathbb{Z}_2` symmetries. The procedure of obtaining the tapered
# Hamiltonian and the tapered reference state is straightforward, however, building the wavefunction
# ansatz needs some experience. The qubit coupled cluster method might be used as an appropriate
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
