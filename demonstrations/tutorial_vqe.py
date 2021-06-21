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

*Author: PennyLane dev team. Last updated: 8 Apr 2021.*

The Variational Quantum Eigensolver (VQE) [#peruzzo2014]_, [#yudong2019]_ is a flagship algorithm for
quantum chemistry using near-term quantum computers. VQE is an application of the `Ritz variational
principle <https://en.wikipedia.org/wiki/Ritz_method>`_  where a quantum computer is used to
prepare a wave function ansatz of the molecule and estimate the expectation value of its electronic
Hamiltonian while a classical optimizer is used to adjust the quantum circuit parameters in order
to find the molecule's ground state energy.

For example, if we use a minimal basis, the ground state wave function of the hydrogen molecule
:math:`\vert \Psi \rangle = \alpha \vert 1100 \rangle + \beta \vert 0011 \rangle` consists of only
the Hartree-Fock component and a doubly-excited configuration where the two electrons occupy the
highest-energy molecular orbitals. If we use a quantum computer to prepare the four-qubit
entangled state :math:`\vert \Psi \rangle`, the ultimate goal of the VQE algorithm
is to find the values of :math:`\alpha` and :math:`\beta` that minimize the expectation value of
the electronic Hamiltonian.

The PennyLane library allows users to implement the full VQE algorithm using only a few
lines of code. In this tutorial, we guide you through a calculation of the ground-state energy of
the hydrogen molecule. Let's get started! ⚛️

Building the electronic Hamiltonian
-----------------------------------

The first step is to specify the molecule we want to simulate. This can be done
by providing a list with the atomic symbols and a one-dimensional
array with the nuclear coordinates in
`atomic units <https://en.wikipedia.org/wiki/Hartree_atomic_units>`_.
"""
import numpy as np

symbols = ['H', 'H']
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
print('Number of qubits = ', qubits)
print('Hamiltonian is ', H)

##############################################################################
# In this example, we use a `minimal basis set <https://en.wikipedia.org/wiki/STO-nG_basis_sets>`
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

dev = qml.device('default.qubit', wires=qubits)

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
# represents the `Hartree-Fock (HF) state <>`_ where the two electrons in the molecule
# occupy the lowest-energy orbitals. The second term :math:`|0011\rangle` encodes a double
# excitation of the HF state where the particles are excited from qubits 0, 1 to 2, 3.
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
# Now we can proceed to minimize the cost function to find the ground state of
# the :math:`\mathrm{H}_2` molecule.

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
energy = []

# store the values of the circuit parameter
angle = []

max_iterations = 100
conv_tol = 1e-06

for n in range(max_iterations):
    theta, prev_energy = opt.step_and_cost(cost_fn, theta)

    angle.append(theta)
    energy.append(cost_fn(theta))

    # energy = cost_fn(params)
    # conv = np.abs(energy - prev_energy)

    conv = np.abs(energy[-1] - prev_energy)

    if n % 2 == 0:
        print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

    if conv <= conv_tol:
        break

print()
print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
print(f"Optimal value of the circuit parameter = {angle[-1]:.4f}")

##############################################################################
# Success! 🎉🎉🎉 The ground-state energy of the hydrogen molecule has been estimated with chemical
# accuracy (< 1 kcal/mol) with respect to the exact value of -1.136189454088 Hartree (Ha) obtained
# from a full configuration-interaction (FCI) calculation. This is because, for the optimized
# values of the single-qubit rotation angles, the state prepared by the VQE ansatz is precisely
# the FCI ground-state of the :math:`H_2` molecule :math:`|H_2\rangle_{gs} = 0.99 |1100\rangle - 0.10
# |0011\rangle`.

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(12)

# Add energy plot on column 1
E_fci = -1.136189454088
ax1 = fig.add_subplot(121)
ax1.plot(range(n+2), energy, 'go-', ls='dashed')
ax1.plot(range(n+2), np.full(n+2, E_fci), color='red')
ax1.set_xlabel("Optimization step", fontsize=13)
ax1.set_ylabel("Energy (Hartree)", fontsize=13)
ax1.text(0.5, -1.1176, r'$E_{HF}$', fontsize=15)
ax1.text(0, -1.1357, r'$E_{FCI}$', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add angle plot on column 2
ax2 = fig.add_subplot(122)
ax2.plot(range(n+2), angle, 'go-', ls='dashed')
ax2.set_xlabel("Optimization step", fontsize=13)
ax2.set_ylabel('$\\theta$', fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplots_adjust(wspace=0.3)
plt.show()

##############################################################################
# What other molecules would you like to study using PennyLane?
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
