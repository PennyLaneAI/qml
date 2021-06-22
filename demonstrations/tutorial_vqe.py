r"""
A brief overview of VQE
=======================

.. meta::
    :property="og:description": Find the ground state of a Hamiltonian using the
        variational quantum eigensolver algorithm in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/pes_h2.png

.. related::

   tutorial_vqe_parallel VQE with parallel QPUs
   tutorial_vqe_qng Accelerating VQE with the QNG
   tutorial_vqt Variational quantum thermalizer

*Author: PennyLane dev team. Last updated: 22 June 2021.*

The Variational Quantum Eigensolver (VQE) [#peruzzo2014]_, [#yudong2019]_ is a flagship algorithm
for quantum chemistry using near-term quantum computers. VQE is an application of the
`Ritz variational principle <https://en.wikipedia.org/wiki/Ritz_method>`_ where a quantum
computer is trained to prepare the ground state of a given molecule.

The inputs to the VQE algorithms are: the qubit Hamiltonian of the molecule, the
parametrized quantum circuit preparing the electronic state of the molecule, and the
cost function to evaluate the expectation value of the target Hamiltonian in the
trial state. The ground state energy is obtained performing an iterative minimization
of the cost function. The optimization problem is solved by a classical optimizer
which uses a quantum computer to evaluate the cost function and its gradient at each
optimization step.

In this tutorial you will learn how to implement the VQE algorithm in a few lines of code.
Without loss of generality, we apply the algorithm to find the ground state of the hydrogen
molecule (:math:`\mathrm{H}_2`). First, we build the molecular Hamiltonian using a minimal
basis set approximation. We continue by defining the quantum circuit preparing the trial
state of the molecule. Then, we define the cost function to evaluate the expectation value
of the qubit Hamiltonian. Finally, we define the classical optimizer, initialize the
circuit parameters and run the VQE algorithm using a PennyLane simulator.

Let's get started!

Building the electronic Hamiltonian
-----------------------------------

The first step is to specify the molecule we want to simulate. This can be done
by providing a list with the atomic symbols and a one-dimensional
array with the nuclear coordinates in
`atomic units <https://en.wikipedia.org/wiki/Hartree_atomic_units>`_.
"""
import numpy as np

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])

##############################################################################
# The molecular structure can also be imported from a external file using
# the :func:`~.pennylane_qchem.qchem.read_structure` function.
#
# Now we can build the electronic Hamiltonian of the hydrogen molecule using the
# :func:`molecular_hamiltonian` function. The outputs of the function are the qubit
# Hamiltonian and the number of qubits needed to represent it:

import pennylane as qml

H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
print("Number of qubits = ", qubits)
print("Hamiltonian is ", H)

##############################################################################
# In this example, we use a `minimal basis set <https://en.wikipedia.org/wiki/STO-nG_basis_sets>`_
# to represent the `molecular orbitals <https://en.wikipedia.org/wiki/Molecular_orbital>`.
# For the :math:`\mathrm{H}_2` molecule, we have a total of four spin orbitals which requires
# four qubits to perform the quantum simulations. Furthermore, we use the Jordan-Wigner
# transformation [#seeley2012]_ to perform the fermionic-to-qubit mapping of the
# electronic Hamiltonian.
#
# For more details on how to use the :func:`molecular_hamiltonian`
# function to build the Hamiltonian of more complicated systems, see the
# tutorial :doc:`tutorial_quantum_chemistry`.
#
# That's it! From here on, we can use PennyLane as usual, employing its entire stack of
# algorithms and optimizers.
#
# Implementing the VQE algorithm
# ------------------------------
# We begin by defining the device, in this case PennyLane’s standard qubit simulator:

dev = qml.device("default.qubit", wires=qubits)

##############################################################################
# Next, we need to define the quantum circuit that prepares the trial state to be
# optimized by the VQE algorithm. Here, we use the qubit states :math:`\vert 0 \rangle`,
# :math:`\vert 1 \rangle` to encode the occupation number of the molecular spin-orbitals.
# For the :math:`\mathrm{H}_2` molecule in a minimal basis set, the *ansatz* for the
# ground state is given by the entangled state,
#
# .. math::
#     \vert \Psi(\theta) \rangle = cos(\theta/2)|1100\rangle -\sin(\theta/2)|0011\rangle),
#
# where :math:`\theta` is the variational parameters. The first term :math:`|1100\rangle`
# represents the `Hartree-Fock (HF) state
# <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`_ where the two electrons in
# the molecule occupy the lowest-energy orbitals. The second term :math:`|0011\rangle`
# encodes a double excitation of the HF state where the particles are excited from qubits
# 0, 1 to 2, 3.
#
# The quantum circuit to prepare the trial state :math:`\vert \Psi(\theta) \rangle` is
# shown in the figure below.
#
# |
#
# .. figure:: /demonstrations/variational_quantum_eigensolver/sketch_circuit.png
#     :width: 50%
#     :align: center
#
# |
#
# The double-excitation gate :math:`G^{(2)}` is implemented in PennyLane as a Givens
# rotations that act on the subspace of four qubits. For more details on the excitation
# operations available in PennyLane see the tutorial :doc:`tutorial_givens_rotations`.
#
# Implementing the circuit above is straightforward. First, we use :func:`hf_state` function
# to generate the vector representing the Hartree-Fock state.

electrons = 2
hf = qml.qchem.hf_state(electrons, qubits)
print(hf)

##############################################################################
# The ``hf`` array is used by the :class:`~.pennylane.BasisState` operation to initialize
# the qubit register. Then, the :class:`~.pennylane.DoubleExcitation` operation is applied
# to create a superposition of the Hartree-Fock and the doubly-excited states.

def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

##############################################################################
# The next step is to create the cost function which is the expectation value
# of the molecular Hamiltonian computed in the trial state prepared by the
# ``circuit`` function. We do this using the :class:`~.ExpvalCost`
# class tailored for VQE optimization. It requires specifying the
# circuit, target Hamiltonian, and the device, and returns a cost function that can
# be evaluated with the circuit parameters:

cost_fn = qml.ExpvalCost(circuit, H, dev)

##############################################################################
# Now we proceed to minimize the cost function to find the ground state of
# the :math:`\mathrm{H}_2` molecule.
#
# First, we need to define the classical optimizer. PennyLane offers different
# built-in optimizers including the quantum natural gradient
# method which can speed up VQE simulations. For example, see the tutorial
# :tutorial:`tutorial_vqe_qng`. Here we use a basic gradient-descent optimizer.

opt = qml.GradientDescentOptimizer(stepsize=0.4)

##############################################################################
# Next, we initialize the circuit parameter :math:`\theta` to zero so we start
# from the Hartree-Fock state.

theta = 0.0

##############################################################################
# We carry out the optimization over a maximum of 100 steps aiming to reach a
# convergence tolerance for the value of the cost function of :math:`\sim 10^{
# -6}`.

# store the values of the cost function
energy = [cost_fn(theta)]

# store the values of the circuit parameter
angle = [theta]

max_iterations = 100
conv_tol = 1e-06

for n in range(max_iterations):
    theta, prev_energy = opt.step_and_cost(cost_fn, theta)

    energy.append(cost_fn(theta))
    angle.append(theta)

    conv = np.abs(energy[-1] - prev_energy)

    if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print()
print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
print(f"Optimal value of the circuit parameter = {angle[-1]:.4f}")

##############################################################################
# We plot the values of the ground state energy of the molecule
# and the gate parameter :math:`\theta` as a function of the optimization step.

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)

# Add energy plot on column 1
E_fci = -1.136189454088
ax1 = fig.add_subplot(121)
ax1.plot(range(n + 2), energy, "go-", ls="dashed")
ax1.plot(range(n + 2), np.full(n + 2, E_fci), color="red")
ax1.set_xlabel("Optimization step", fontsize=13)
ax1.set_ylabel("Energy (Hartree)", fontsize=13)
ax1.text(0.5, -1.1176, r"$E_{HF}$", fontsize=15)
ax1.text(0, -1.1357, r"$E_{FCI}$", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add angle plot on column 2
ax2 = fig.add_subplot(122)
ax2.plot(range(n + 2), angle, "go-", ls="dashed")
ax2.set_xlabel("Optimization step", fontsize=13)
ax2.set_ylabel("Gate parameter $\\theta$", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplots_adjust(wspace=0.3, bottom=0.2)
plt.show()

##############################################################################
# The VQE algorithm converges after thirteen iterations. The optimal value of the
# circuit parameter :math:`\theta^* = 0.208` defines the state
#
# .. math::
#     \vert \Psi(\theta^*) \rangle = 0.994 ~ \vert 1100 \rangle
#                                  - 0.104 ~ \vert 0011 \rangle,
#
# which is precisely the ground state of the :math:`\mathrm{H}_2` molecule in a
# minimal basis set.
#
# We have used the VQE algorithm to correct the Hartree-Fock
# energy of the hydrogen molecule by including the effects of the
# `electronic correlations <<https://en.wikipedia.org/wiki/Electronic_correlation>`_.
# This was done using a simple circuit to account for the double excitation
# :math:`\vert 0011 \rangle` in the trial state. The final value of the
# VQE energy can be used to estimate the *electronic correlation energy*
# :math:`E_\mathrm{corr} = E_\mathrm{VQE} - E_\mathrm{HF} = -0.01883 Ha`.
#
# .. _vqe_references:
#
# References
# ----------
#
# .. [#peruzzo2014]
#
#     Alberto Peruzzo, Jarrod McClean *et al.*, "A variational eigenvalue solver on a photonic
#     quantum processor". `Nature Communications 5, 4213 (2014).
#     <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# .. [#yudong2019]
#
#     Yudong Cao, Jonathan Romero, *et al.*, "Quantum Chemistry in the Age of Quantum Computing".
#     `Chem. Rev. 2019, 119, 19, 10856-10915.
#     <https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803>`__
#
# .. [#seeley2012]
#
#     Jacob T. Seeley, Martin J. Richard, Peter J. Love. "The Bravyi-Kitaev transformation for
#     quantum computation of electronic structure". `Journal of Chemical Physics 137, 224109 (2012).
#     <https://aip.scitation.org/doi/abs/10.1063/1.4768229>`__
