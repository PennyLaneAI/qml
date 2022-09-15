r"""

Quantum Resource Estimation
===========================

.. meta::
    :property="og:description": Learn how to estimate quantum resources
    :property="og:image": https://pennylane.ai/qml/_images/differentiable_HF.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE


*Author: Soran Jahangiri. Posted: 17 August 2022. Last updated: 17 August 2022*

Quantum algorithms such s quantum phase estimation and the variational quantum eigensolver
implemented on a suitable quantum hardware are expected to tackle problems that are
intractable for conventional classical computers. In the absence of quantum devices, the
implementation of such algorithms is limited to computationally inefficient classical simulators.
This makes it difficult to properly explore the accuracy and efficiency of these algorithms for
relatively large problem sizes where the actual advantage of quantum algorithms is expected to be
seen. Despite the simulation difficulties, it is possible to estimate the amount of resources
required to implement such quantum algorithms without performing computationally expensive
simulations.

In this demo, we introduce a functionality in PennyLane that allows estimating the total number of
non-Clifford gates and logical qubits required to implement the quantum phase estimation (QPE)
algorithm for simulating molecular Hamiltonians represented in first and second quantization. We
also present the functionality for estimating the total number of measurements needed to compute
expectation values within a given error using algorithms such as the variational quantum eigensolver
(VQE). Estimating the number of gates and qubits is rather straightforward for the VQE algorithm.

Quantum Phase Estimation
------------------------
The QPE algorithm can be used to compute the phase of a unitary operator within an error
:math:`\epsilon`. The unitary operator :math:`U` can be selected to share eigenvectors
:math:`| \Psi \rangle` with a molecular Hamiltonian :math:`H` by having, for example,
:math:`U = e^{-iH}` to compute the eigenvalues of :math:`H`. A QPE conceptual circuit diagram is
shown in the following. The circuit contains a set of target wires initialized at the eigenstate
:math:`| \Psi \rangle` which encode the unitary operator and a set of estimation wires initialized
in :math:`| 0 \rangle` which are measured after applying an inverse quantum Fourier transform. The
measurement results give a binary string that can be used to estimate the phase of the unitary and
the ground state energy of the Hamiltonian. The precision in estimating the phase depends on the
number of estimation wires while the number of gates in the circuit is determined by the unitary
operator.

We are interested to estimate the number of logical qubits and the number of non-Clifford gates,
which are hard to implement, for a QPE algorithm that implements a second-quantized Hamiltonian
describing an isolated molecule and a first-quantized Hamiltonian describing a periodic material.
We assume Gaussian and plane wave basis sets for describing the molecular and periodic systems,
respectively. The PennyLane functionality in the ``resource`` module allows estimating such QPE
resources by simply defining system specifications such as atomic symbols and geometries and a
target error for estimating the ground state energy of the Hamiltonian. Let's see how!

QPE cost for simulating isolated molecules
******************************************
The cost of performing QPE simulations for isolated molecules is estimated based on the double low
rank Hamiltonian factorization algorithm. We first need to define the atomic symbols and
coordinates for the given molecule. Here we use the water molecule at its equilibrium geometry as an
example
"""
import pennylane as qml
from pennylane import numpy as np

symbols = ['O', 'H', 'H']
geometry = np.array([[0.00000000,  0.00000000,  0.28377432],
                     [0.00000000,  1.45278171, -1.00662237],
                     [0.00000000, -1.45278171, -1.00662237]], requires_grad=False)

##############################################################################
# Then we construct a molecule object by selecting a basis set and compute the one- and two-electron
# integrals in the molecular orbital basis.
mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g')
core, one, two = qml.qchem.electron_integrals(mol)()

##############################################################################
# We now initiate the ``DoubleFactorization`` class of the ``qml.resource`` module

algo = qml.resource.DoubleFactorization(one, two)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits

print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# QPE cost for simulating periodic materials
# ******************************************
# The cost of implementing the QPE algorithm for periodic materials is estimated following the
# algorithm of Su et al. for Hamiltonians in first quantization assuming constructed with a
# plane wave basis. We first need to define the number of plane waves, the number of electrons and
# the volume of the unit cell that constructs the periodic material. Let's use dilithium iron
# silicate :math:`\text{Li}_2\text{FeSiO}_4` as an example. Note that atomic units must be used
# for the cell volume.

planewaves = 100000
electrons = 156
volume = 1145

##############################################################################
# We now initiate the ``FirstQuantization`` class of the ``qml.resource`` module
algo = qml.resource.FirstQuantization(planewaves, electrons, volume)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits
print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# Computing the 1-Norm of the Hamiltonian
# ***************************************
# The resource estimation functionality assumes that the molecular and material Hamiltonian can be
# constructed as a linear combination of unitary operators.
#
# .. math:: H=\sum_{i} c_i U_i.
#
# The cost of computing the ground state energy of this Hamiltonian using the QPE algorithm depends
# on the complexity of implementing the unitary operator for encoding the Hamiltonian, which can be
# constructed as :math:`U = e^{-i \arccos (H / \lambda)}` and implemented using a quantum walk
# operator [Cao et al.]. The eigenvalues of the quantum walk operator are
# :math:`e^{-i \arccos (E / \lambda)}` which can be post-processed classically to give the
# eigenvalues of the Hamiltonian :math:`E`. The parameter :math:`\lambda` is the 1-Norm of the
# Hamiltonian and plays an important role in determining the cost of implementing the QPE algorithm.
# In PennyLane, :math:`\lambda` can be obtained with

print(f'1-Norm of the Hamiltonian: {algo.lamb}')

##############################################################################
# The 1-Norm of the Hamiltonian can also be directly computed using the ``norm`` function. For the
# first-quantized Hamiltonian it can be computed for a target error in the algorithm with

qml.resource.FirstQuantization.norm(planewaves, electrons, volume, error=0.001)

##############################################################################
# Measurement Complexity Estimation
# ---------------------------------
# In variational quantum algorithms, such as VQE, the expectation value of an observable is
# typically computed by decomposing the observable to the tensor products of Pauli and Identity
# operators apply to a limited number of qubits and then measure the expectation value for each of
# these local terms. The number of qubits required for the measurement is trivially determined by
# the number of qubits the observable acts on. The number of gates required to implement the
# variational algorithm is determined by a circuit ansatz that is also known a priori. However,
# estimating the number of circuit measurements required to achieve a certain error in computing the
# expectation value is not typically trivial. Let's now use the PennyLane functionality that
# estimates such measurement requirements for computing the expectation value of the water
# Hamiltonian as an example.
#
# First, we construct the molecular Hamiltonian

H = qml.qchem.molecular_hamiltonian(symbols, geometry)[0]

##############################################################################
# The number of measurements needed to compute :math:`\left \langle H \right \rangle` within the
# chemical accuracy, 0.0016 :math:`\text{Ha}`, can be obtained with

# m = qml.resource.estimate_shots(H.coeffs, H.ops, error=0.0016)
# print(m)

##############################################################################
# This number corresponds to the measurement process where each term in the Hamiltonian is measured
# independently. This number can be significantly reduced by partitioning the Pauli words into
# groups of commuting terms that can be measured simultaneously.

ops, coeffs = qml.grouping.group_observables(H.ops, H.coeffs)

# m = qml.resource.estimate_shots(coeffs, ops, error=0.0016)
# print(m)

##############################################################################
#.. bio:: Soran Jahangiri
#    :photo: ../_static/Soran.png
#
#    Soran Jahangiri is a quantum chemist working at Xanadu. His work is focused on developing and implementing quantum algorithms for chemistry applications.
