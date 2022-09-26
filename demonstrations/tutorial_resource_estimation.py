r"""

Quantum Resource Estimation
===========================

.. meta::
    :property="og:description": Learn how to estimate the number of qubits and gates needed to
     implement quantum algorithms
    :property="og:image": https://pennylane.ai/qml/_images/resource_estimation.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE


*Author: Soran Jahangiri. Posted: 17 August 2022. Last updated: 17 August 2022*

Quantum algorithms such as quantum phase estimation and the variational quantum eigensolver
are widely studied in quantum chemistry as potential avenues to tackle problems that are intractable for conventional computers. However, we currently do not have quantum computers or simulators capable of implementing large-scale
versions of these algorithms. This makes it difficult to properly explore their accuracy and
efficiency for problem sizes where the actual advantage of quantum algorithms can potentially occur.
Despite these difficulties, it is still possible to estimate the amount of resources required to
implement such quantum algorithms .

In this demo, we describe how to estimate the total number of non-Clifford gates and logical qubits
required to implement the quantum phase estimation (QPE) algorithm for simulating molecular
Hamiltonians represented in first and second quantization. We also explain how to estimate the
total number of measurements needed to compute expectation values using algorithms such as the
variational quantum eigensolver (VQE).

Quantum Phase Estimation
------------------------
The QPE algorithm can be used to compute the phase associated with an eigenstate of a unitary
operator. For the purpose of quantum simulation, the unitary operator :math:`U` can be chosen to
share eigenvectors with a molecular Hamiltonian :math:`H`, for example by setting
:math:`U = e^{-iH}`. Estimating the phase of such a unitary then permits recovering the
corresponding eigenvalue of the Hamiltonian. A conceptual QPE circuit diagram is shown below.

.. figure:: /demonstrations/resource_estimation/qpe.svg
    :width: 60%
    :align: center

    Circuit representing the quantum phase estimation algorithm.

The circuit contains target wires, here initialized in the ground state :math:`| \psi_0 \rangle`,
and a set of estimation wires, initialized in :math:`| 0 \rangle`. The algorithm repeatedly applies
powers of `U` controlled on the state of estimation wires, which are measured after applying an
inverse quantum Fourier transform. The measurement results give a binary string that can be used to
estimate the phase of the unitary and thus also the ground state energy of the Hamiltonian. The
precision in estimating the phase depends on the number of estimation wires.

For most cases of interest, this algorithm requires more qubits and longer circuit depths than what
can be implemented on existing hardware. We are instead interested in estimating the number of
logical qubits and the number of gates which are needed to implement the algorithm. We focus on
non-Clifford gates, which are the most expensive to implement in a fault-tolerant setting. We now
explain how to perform this resource estimation for QPE algorithms based on a second-quantized
Hamiltonian describing a molecule, and a first-quantized Hamiltonian describing a periodic material.
We assume Gaussian and plane wave basis sets for describing the molecular and periodic systems,
respectively. The PennyLane functionality in the :mod:`qml.resource  <pennylane.resource>`
module allows us to estimate these resources by simply defining system specifications and a target
error for estimation. Let's see how!

QPE cost for simulating molecules
*********************************
We study the double low-rank Hamiltonian factorization algorithm of [#vonburg2021]_ and [#lee2021]_.
This algorithm requires the one- and two-electron integrals as input. These integrals can be
obtained in different ways and here we use PennyLane to compute them. We first need to define the
atomic symbols and coordinates for the given molecule. Let's use the water molecule at its
equilibrium geometry as an example.
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
# We now initiate the :class:`~.pennylane.resource.DoubleFactorization` class

algo = qml.resource.DoubleFactorization(one, two)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits

print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# This estimation is for a target error that is set to the chemical accuracy, 0.0016
# :math:`\text{Ha}`, by default. We can change the target error to a larger value which leads to a
# smaller number of non-Clifford gates and logical qubits.

algo = qml.resource.DoubleFactorization(one, two, error=0.016)
print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# We can also estimate the number of non-Clifford gates and logical qubits by changing the threshold
# error values for discarding the negligible factors in the factorized Hamiltonian and plot the
# estimated numbers.

threshold = [10**-n for n in range(10)]
n_gates = []
n_qubits = []

for tol in threshold:
    algo_ = qml.resource.DoubleFactorization(one, two, tol_factor=tol, tol_eigval=tol)
    n_gates.append(algo_.gates)
    n_qubits.append(algo_.qubits)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)
ax[0].plot(threshold, n_gates, 'o', markerfacecolor='none', color='teal')
ax[1].plot(threshold, n_qubits, 'o', markerfacecolor='none', color='teal')

ax[0].set_ylabel('n gates')
ax[1].set_ylabel('n qubits')

for i in [0, 1]:
    ax[i].set_xlabel('threshold')
    ax[i].set_xscale('log')
    ax[i].tick_params(axis='x', labelrotation = 90)
    ax[i].set_xticks(threshold)
    ax[i].set_xticklabels(threshold)

fig.tight_layout()

##############################################################################
# QPE cost for simulating periodic materials
# ******************************************
# For periodic materials, we estimate the cost of implementing the QPE algorithm of [#su2021]_
# using Hamiltonians in first quantization and in a plane wave basis. We first need to define the
# number of plane waves, the number of electrons and the volume of the unit cell that constructs
# the periodic material. Let's use dilithium iron silicate :math:`\text{Li}_2\text{FeSiO}_4` as an
# example taken from [#delgado2022]_. For this material, the unit cell contains 156 electrons
# and has dimensions :math:`9.49 \times 10.20 \times 11.83` in atomic units, which amount to a volume
# of :math:`1145 a_0^3`, where :math:`a_0` is the
# `Bohr radius <https://en.wikipedia.org/wiki/Bohr_radius>`_. We also use 100000 plane waves.

planewaves = 100000
electrons = 156
volume = 1145

##############################################################################
# We initiate the :class:`~.pennylane.resource.FirstQuantization` class
algo = qml.resource.FirstQuantization(planewaves, electrons, volume)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits.
print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# We can also plot the estimated numbers as a function of the number of plane waves for different
# target errors

error = [0.05, 0.01, 0.005, 0.001]  # in atomic units
planewaves = [10 ** n for n in range(1, 10)]
n_gates = []
n_qubits = []

for er in error:
    n_gates_ = []
    n_qubits_ = []

    for pw in planewaves:
        algo_ = qml.resource.FirstQuantization(pw, electrons, volume, error=er)
        n_gates_.append(algo_.gates)
        n_qubits_.append(algo_.qubits)
    n_gates.append(n_gates_)
    n_qubits.append(n_qubits_)

fig, ax = plt.subplots(1, 2)

for i in range(len(n_gates)):
    ax[0].plot(planewaves, n_gates[i], 'o', markerfacecolor='none', label=error[i])
    ax[1].plot(planewaves, n_qubits[i], 'o', markerfacecolor='none', label=error[i])

ax[0].set_ylabel('n gates')
ax[1].set_ylabel('n qubits')

for i in [0, 1]:
    ax[i].set_xlabel('n planewaves')
    ax[i].tick_params(axis='x', labelrotation=90)
    ax[0].set_yscale('log')
    ax[i].set_xscale('log')
    ax[i].legend(title='error [Ha]')

fig.tight_layout()

##############################################################################
# Can you interpret the plotted results?
#
# The algorithm uses a decomposition of the Hamiltonian as a linear combination of unitaries.
#
# .. math:: H=\sum_{i} c_i U_i.
#
# The parameter :math:`\lambda=\sum_i c_i` can be interpreted as a 1-norm of the
# Hamiltonian and plays an important role in determining the cost of implementing the QPE algorithm.
# In PennyLane, :math:`\lambda` can be obtained with

print(f'1-norm of the Hamiltonian: {algo.lamb}')

##############################################################################
# Variational quantum eigensolver
# ------------------------------------------
# In variational quantum algorithms such as VQE, the expectation value of an observable is
# typically computed by decomposing the observable into a linear combination of Pauli words,
# which are tensor products of Pauli and Identity operators. The expectation values are calculated
# through linearity by measuring the expectation value for each of these terms and combining the
# results. The number of qubits required for the measurement is trivially determined by
# the number of qubits the observable acts on. The number of gates required to implement the
# variational algorithm is determined by a circuit ansatz that is also known a priori. However,
# estimating the number of circuit evaluations, i.e., the number of shots, required to achieve a certain error in
# computing the expectation value is not as straightforward. Let's now use PennyLane to estimate
# the number of shots needed to compute the expectation value of the water Hamiltonian.
#
# First, we construct the molecular Hamiltonian

H = qml.qchem.molecular_hamiltonian(symbols, geometry)[0]

##############################################################################
# The number of measurements needed to compute :math:`\left \langle H \right \rangle` can be
# obtained with the :func:`~.pennylane.resource.estimate_shots` function which requires the
# Hamiltonian coefficients and observables as input. The number of measurements required to compute
# :math:`\left \langle H \right \rangle` with a target error set to the chemical accuracy, 0.0016
# :math:`\text{Ha}`, is obtained with

m = qml.resource.estimate_shots(H.coeffs, error=0.0016)
print(f'Shots : {m:.2e}')

##############################################################################
# This number corresponds to the measurement process where each term in the Hamiltonian is measured
# independently. The number can be reduced by partitioning the Pauli words into groups of commuting
# terms that can be measured simultaneously.

ops, coeffs = qml.grouping.group_observables(H.ops, H.coeffs)

m = qml.resource.estimate_shots(coeffs, error=0.0016)
print(f'Shots : {m:.2e}')

##############################################################################
# It is also interesting to illustrate how the number of shots depends on the target error.

error = np.array([0.02, 0.015, 0.01, 0.005, 0.001])
m = [qml.resource.estimate_shots(H.coeffs, error=er) for er in error]

e_ = np.linspace(error[0], error[-1], num=50)
m_ = 1.4e4 / np.linspace(error[0], error[-1], num=50)**2

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(error, m, 'o', markerfacecolor='none', color='teal', label='estimated')
ax.plot(e_, m_, ':', markerfacecolor='none', color='olive', label='$ 1.4e4 * 1/\epsilon^2 $')

ax.set_ylabel('shots')
ax.set_xlabel('error [Ha]')
ax.set_yscale('log')
ax.tick_params(axis='x', labelrotation = 90)
ax.legend()
fig.tight_layout()

##############################################################################
# We have added a line showing the dependency of the shots to the error as
# :math:`shots = 1.4e4 * 1/\epsilon^2` for comparison. Can you draw any interesting information form
# the plot?
#
# Conclusions
# -----------
# This tutorial shows how to use the resource estimation functionality in PennyLane to compute the
# total number of non-Clifford gates and logical qubits required to simulate a Hamiltonian with
# quantum phase estimation algorithms. The estimation can be performed for second-quantized
# molecular Hamiltonians obtained with a double low-rank factorization algorithm,
# and first-quantized Hamiltonians of periodic materials in the plane wave basis. We also discuss
# the estimation of the total number of shots required to obtain the expectation value of an
# observable using the variational quantum eigensolver algorithm. The functionality allows to obtain
# interesting results about the cost of implementing important quantum algorithms. For instance, we
# estimated the costs with respect to factors such as the target error in obtaining energies and
# the number of basis functions used to simulate a system. Can you think of other interesting
# information that can be obtained using this PennyLane functionality?!
#
# References
# ----------
#
# .. [#vonburg2021]
#
#     Vera von Burg, Guang Hao Low, Thomas Haner, Damian S. Steiger, *et al.*,
#     "Quantum computing enhanced computational catalysis".
#     `Phys. Rev. Research 3, 033055 (2021)
#     <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.033055>`__
#
# .. [#lee2021]
#
#     Joonho Lee, Dominic W. Berry, Craig Gidney, William J. Huggins, *et al.*,
#     "Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction".
#     `PRX Quantum 2, 030305 (2021)
#     <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`__
#
# .. [#su2021]
#
#     Yuan Su, Dominic W. Berry, Nathan Wiebe, Nicholas Rubin, and Ryan Babbush,
#     "Fault-Tolerant Quantum Simulations of Chemistry in First Quantization".
#     `PRX Quantum 2, 040332 (2021)
#     <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040332>`__
#
# .. [#delgado2022]
#
#     Alain Delgado, Pablo A. M. Casares, Roberto dos Reis, Modjtaba Shokrian Zini, *et al.*,
#     "How to simulate key properties of lithium-ion batteries with a fault-tolerant quantum
#     computer".
#     `arXiv:2204.11890
#     <https://arxiv.org/abs/2204.11890>`__
# About the author
# ----------

##############################################################################
#.. bio:: Soran Jahangiri
#    :photo: ../_static/Soran.png
#
#    Soran Jahangiri is a quantum chemist working at Xanadu. His work is focused on developing and implementing quantum algorithms for chemistry applications.
