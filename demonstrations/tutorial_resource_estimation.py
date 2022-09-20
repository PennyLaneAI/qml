r"""

Quantum Resource Estimation
===========================

.. meta::
    :property="og:description": Learn how to estimate the number of qubits and gates needed to
     implement quantum algorithms
    :property="og:image": https://pennylane.ai/qml/_images/differentiable_HF.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE


*Author: Soran Jahangiri. Posted: 17 August 2022. Last updated: 17 August 2022*

Quantum algorithms such as quantum phase estimation and the variational quantum eigensolver
are studied as avenues to tackle problems that are intractable for conventional computers. However,
we currently do not have quantum computers or simulators capable of implementing large-scale
versions of these algorithms. This makes it difficult to properly explore their accuracy and
efficiency for problem sizes where the actual advantage of quantum algorithms can potentially occur.
Despite these difficulties, it is still possible to estimate the amount of resources required to
implement such quantum algorithms .

In this demo, we describe how to estimate the total number of
non-Clifford gates and logical qubits required to implement the quantum phase estimation (QPE)
algorithm for simulating molecular Hamiltonians represented in first and second quantization. We
also explain how to estimate the total number of measurements needed to compute
expectation values using algorithms such as the variational quantum eigensolver
(VQE). 

Quantum Phase Estimation
------------------------
The QPE algorithm can be used to compute the phase associated with an eigenstate of a unitary
operator. For the purpose of quantum simulation, the unitary operator :math:`U` can be chosen to
share eigenvectors with a molecular Hamiltonian :math:`H`, for example by setting
:math:`U = e^{-iH}`. Estimating the phase of such a unitary then permits recovering the
corresponding eigenvalue of the Hamiltonian. A conceptual QPE circuit diagram is shown below. The
circuit contains target wires, here initialized in the ground state :math:`| \psi_0 \rangle`,and a
set of estimation wires, initialized in :math:`| 0 \rangle`. The algorithm repeatedly applies powers
of `U` controlled on the state of estimation wires, which are measured after applying an inverse
quantum Fourier transform. The measurement results give a binary string that can be used to estimate
the phase of the unitary and thus also the ground state energy of the Hamiltonian. The precision in
estimating the phase depends on the number of estimation wires.

For most cases of interest, this algorithm requires more qubits and longer circuit depths than what
can be implemented on existing hardware. We are instead interested in estimating the number of
logical qubits and the number of gates which are needed to implement the algorithm. We focus on
non-Clifford gates, which are the most expensive to implement in a fault-tolerant setting. We now
explain how to perform this resource estimation for QPE algorithms based on a second-quantized
Hamiltonian describing a molecule, and a first-quantized Hamiltonian describing a periodic material.
We assume Gaussian and plane wave basis sets for describing the molecular and periodic systems,
respectively. The PennyLane functionality in the ``resource`` module allows us to estimate these
resources by simply defining system specifications and a target error for estimation. Let's see how!

QPE cost for simulating molecules
*********************************
We study the double low-rank Hamiltonian factorization algorithm of Ref. [1]. This algorithm
requires the one- and two-electron integrals as input which can be obtained in different ways. Here,
we use PennyLane to obtain these integrals. We first need to define the atomic symbols and
coordinates for the given molecule. Let's use the water molecule at its equilibrium geometry as an
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
mol = qml.qchem.Molecule(symbols, geometry, basis_name='6-31g')
core, one, two = qml.qchem.electron_integrals(mol)()

##############################################################################
# We now initiate the ``DoubleFactorization`` class of the ``qml.resource`` module

algo = qml.resource.DoubleFactorization(one, two)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits

print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# This estimation is for a target error that is set to the chemical accuracy, 0.0016
# :math:`\text{Ha}`, by default. We can change the target error and estimate the number of
# non-Clifford gates and logical qubits. For instance, setting the target error to a larger value
# leads to a smaller number of non-Clifford gates and logical qubits

algo = qml.resource.DoubleFactorization(one, two, error=0.016)
print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# QPE cost for simulating periodic materials
# ******************************************
# For periodic materials, we estimate the cost of implementing the QPE algorithm of Su et al.
# using Hamiltonians in first quantization and in a plane wave basis. We first need to define the
# number of plane waves, the number of electrons and the volume of the unit cell that constructs
# the periodic material. Let's use dilithium iron silicate :math:`\text{Li}_2\text{FeSiO}_4` as an
# example taken from Ref. [battery paper]. For this material, the unit cell contains 156 electrons
# and has dimensions :math:`9.49 \times 10.20 \times 11.83` in atomic units which amount to a volume
# of :math:`1145 a_0^3` where :math:`a_0` is the
# `Bohr radius <https://en.wikipedia.org/wiki/Bohr_radius>`_. We also use 100000 plane waves.

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
# Computing the 1-norm of the Hamiltonian
# ***************************************
# The algorithm uses a decomposition of the Hamiltonian as a linear combination of unitaries.
#
# .. math:: H=\sum_{i} c_i U_i.
#
# The parameter :math:`\lambda=\sum_i c_i` can be interpreted as a 1-norm of the
# Hamiltonian and plays an important role in determining the cost of implementing the QPE algorithm.
# In PennyLane, :math:`\lambda` can be obtained with

print(f'1-norm of the Hamiltonian: {algo.lamb}')

##############################################################################
# The 1-norm of the Hamiltonian can also be directly computed using the ``norm`` function. For the
# first-quantized Hamiltonian it can be computed for a target error in the algorithm with

qml.resource.FirstQuantization.norm(planewaves, electrons, volume, error=0.001)

##############################################################################
# Variational quantum eigensolver
# ------------------------------------------
# In variational quantum algorithms such as VQE, the expectation value of an observable is
# typically computed by decomposing the observable into a linear combination of Pauli strings,
# which are tensor products of Pauli and Identity operators. The expectation values is calculated
# through linearity by measuring the expectation value for each of these terms and combining the
# results. The number of qubits required for the measurement is trivially determined by
# the number of qubits the observable acts on. The number of gates required to implement the
# variational algorithm is determined by a circuit ansatz that is also known a priori. However,
# estimating the number of circuit measurements required to achieve a certain error in computing the
# expectation value is not as straightforward. Let's now use PennyLane to estimate
# the number of measurements needed to computing the expectation value of the water
# Hamiltonian.
#
# First, we construct the molecular Hamiltonian

H = qml.qchem.molecular_hamiltonian(symbols, geometry)[0]

##############################################################################
# The number of measurements needed to compute :math:`\left \langle H \right \rangle` can be
# obtained with the :func:`~.pennylane.resource.estimate_shots` function which requires the
# Hamiltonian coefficients and observables as input. The number of measurements required for a
# target error set to the chemical accuracy, 0.0016 :math:`\text{Ha}`, can be obtained with

# m = qml.resource.estimate_shots(H.coeffs, H.ops, error=0.0016)
# print(m)

##############################################################################
# This number corresponds to the measurement process where each term in the Hamiltonian is measured
# independently. This number can be significantly reduced by partitioning the Pauli words into
# groups of commuting terms that can be measured simultaneously.

ops, coeffs = qml.grouping.group_observables(H.ops, H.coeffs)

# m = qml.resource.estimate_shots(coeffs, ops, error=0.0016)
# print(m)

# Conclusions
# -----------
# This tutorial
#
##############################################################################
#.. bio:: Soran Jahangiri
#    :photo: ../_static/Soran.png
#
#    Soran Jahangiri is a quantum chemist working at Xanadu. His work is focused on developing and implementing quantum algorithms for chemistry applications.
