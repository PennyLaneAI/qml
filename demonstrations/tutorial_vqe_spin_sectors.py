r"""
VQE in different spin sectors
=============================

.. meta::
    :property="og:description": Find the lowest-energy states of a Hamiltonian in different spin sectors
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_spectra_h2.png

.. related::
   tutorial_vqe Variational Quantum Eigensolver
   tutorial_vqe_parallel VQE with parallel QPUs

*Author: PennyLane dev team. Last updated: 16 July 2021.*

The Variational Quantum Eigensolver (VQE) algorithm is an approach for finding the
lowest-energy state of a molecule using a quantum computer [#peruzzo2014]_.

In the absence of `spin-orbit coupling <https://en.wikipedia.org/wiki/Spin-orbit_interaction>`_,
the eigenstates of the molecular Hamiltonian can be calculated for specific values of the
spin quantum numbers. For example, this allows us to find the energy of the electronic states
in different sectors of the total-spin projection :math:`S_z`. This is illustrated in the figure
below for the energy spectrum of the hydrogen molecule. In this case, the ground state has total
spin :math:`S=0` while the lowest-lying excited states, with total spin :math:`S=1`, form a triplet
related to the spin components :math:`S_z=-1, 0, 1`.

|

.. figure:: /demonstrations/vqe_spin_sectors/energy_spectra_h2_sto3g.png
    :width: 70%
    :align: center

|

In this tutorial you will learn how to run VQE simulations to find the lowest-energy states
of a molecular Hamiltonian with different values of the total spin.
We illustrate this for the hydrogen molecule although
the same methodology can be applied to other molecules. First, we show how to build
the electronic Hamiltonian and the total-spin operator. 
Next, we use excitation operations implemented in PennyLane as Givens rotations
to prepare the trial states of the molecule. Finally, we run the VQE algorithm to find the
ground and the lowest-lying excited states of the :math:`\mathrm{H}_2` molecule.

Let's get started!

Building the Hamiltonian and the total spin operator :math:`\hat{S}^2`
----------------------------------------------------------------
First, we need to specify the structure of the molecule. This is done by providing a list
with the symbols of the constituent atoms and a one-dimensional array with the corresponding
nuclear coordinates in `atomic units <https://en.wikipedia.org/wiki/Hartree_atomic_units>`_.
"""

import numpy as np

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])

##############################################################################
# The geometry of the molecule can also be imported from an external file using
# the :func:`~.pennylane_qchem.qchem.read_structure` function.
#
# Now, we can build the electronic Hamiltonian. We use a minimal `basis set
# <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`_ approximation to represent
# the `molecular orbitals <https://en.wikipedia.org/wiki/Molecular_orbital>`_. Then,
# the qubit Hamiltonian of the molecule is built using the
# :func:`~.pennylane_qchem.qchem.molecular_hamiltonian` function.

import pennylane as qml

H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)

##############################################################################
# The outputs of the function are the Hamiltonian and the number of qubits
# required for the quantum simulations. For the :math:`\mathrm{H}_2` molecule in a minimal
# basis, we have four molecular **spin**-orbitals, which defines the number of qubits.
#
# The :func:`~.pennylane_qchem.qchem.molecular_hamiltonian` function allows us to define
# additional keyword arguments to simulate more complicated molecules. For more details
# take a look at the tutorial :doc:`tutorial_quantum_chemistry`.
#
# We also want to build the total spin operator :math:`\hat{S}^2`,
#
# .. math::
#
#     \hat{S}^2 = \frac{3}{4} N + \sum_{\alpha, \beta, \gamma, \delta}
#     \langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2
#     \vert \gamma, \delta \rangle ~ \hat{c}_\alpha^\dagger \hat{c}_\beta^\dagger
#     \hat{c}_\gamma \hat{c}_\delta.
#
# In the equation above, :math:`N` is the number of electrons,
# :math:`\hat{c}` and :math:`\hat{c}^\dagger` are respectively the electron annihilation and
# creation operators, and
# :math:`\langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2 \vert \gamma, \delta \rangle`
# is the `matrix element of the two-body spin operator
# <https://pennylane.readthedocs.io/en/stable/code/api/pennylane_qchem.qchem.obs.spin2.html>`_
# :math:`\hat{s}_1 \cdot \hat{s}_2` in the basis of spin orbitals.
#
# We use the :func:`~.pennylane_qchem.qchem.spin2` function to build the
# :math:`\hat{S}^2` observable.

electrons = 2
S2 = qml.qchem.spin2(electrons, qubits)
print(S2)

##############################################################################
# Building the quantum circuit to find the ground state
# -----------------------------------------------------
# We build the variational circuit for the ground state of the hydrogen molecule using the
# :class:`~.pennylane.SingleExcitation` and :class:`~.pennylane.DoubleExcitation` operations
# implemented in PennyLane. These operations act as Givens rotations on the subspace of
# the qubits encoding the single- and double-excitations of the Hartree-Fock reference state
# [#qchemcircuits]. For more details on how to use the excitation operations for
# for quantum chemistry applications see the tutorial :doc:`tutorial_givens_rotations`.
#
# First, we use the :func:`~.pennylane_qchem.qchem.hf_state`
# function to generate the vector representing the Hartree-Fock state
# :math:`\vert 1100 \rangle` of the :math:`\mathrm{H}_2` molecule.

hf = qml.qchem.hf_state(electrons, qubits)
print(hf)

##############################################################################
# Next, we use the :func:`~.pennylane_qchem.qchem.excitations`
# function to generate all single- and double-excitations of the Hartree-Fock state.
# This function allows us to define the keyword argument ``delta_sz``
# to specify the total-spin projection of the excitations with respect to the reference
# state. This is illustrated in the figure below.
#
# |
#
# .. figure:: /demonstrations/vqe_spin_sectors/fig_excitations.png
#     :width: 100%
#     :align: center
#
# |
#
# Therefore, for the ground state of the :math:`\mathrm{H}_2` molecule we choose
# ``delta_sz = 0``.

singles, doubles = qml.qchem.excitations(electrons, qubits, delta_sz=0)
print(singles)
print(doubles)

##############################################################################
# The output lists ``singles`` and ``doubles`` contain the qubit indices involved in the
# single and double excitations. Even and odd indices correspond, respectively, to spin-up
# and spin-down orbitals. For ``delta_sz = 0`` we have two single excitations, one from qubit
# 0 to 2 and the other from qubit 1 to 3, and one double excitation from qubits 0, 1 to 2, 3.
# In order to build the build the variational circuit, we just apply the corresponding
# excitation operations. This is done straightforwardly using the
# :class:`~.pennylane.templates.subroutines.AllSinglesDoubles` function.


def circuit(params, wires):
    qml.templates.AllSinglesDoubles(params, wires, hf, singles, doubles)


##############################################################################
# The circuit above prepares full configuration interaction (FCI) trial states for the
# :math:`\mathrm{H}_2` molecule in a minimal basis set:
#
# .. math::
#
#     \vert \Psi(\theta) =
#     C_\mathrm{HF}(\theta) \vert 1100 \rangle
#     + C_{0123}(\theta) \vert 0011 \rangle
#     + C_{02}(\theta) \vert 0110 \rangle
#     + C_{23}(\theta) \vert 1001 \rangle,
#
# where the coefficients :math:`C` are functions of the variational parameters
# :math:`\theta = (\theta_1, \theta_2, \theta_3)` to be optimized by the VQE algorithm.
# Since the ground state of the hydrogen molecule does not contain any contribution
# from the single excitations, the coefficients :math:`C_{02}, C_{23}` must be zero
# for the optimal set of angles :math:`theta`.

##############################################################################
# Running the VQE simulation
# --------------------------
# Now we proceed to optimize the variational parameters. First, we define the device,
# in this case a qubit simulator:

dev = qml.device("default.qubit", wires=qubits)

##############################################################################
# Next, we use the :class:`~.pennylane.ExpvalCost` class to define the cost function.
# This requires specifying the circuit, target Hamiltonian, and the device. It returns
# a cost function that can be evaluated with the circuit parameters:

cost_fn = qml.ExpvalCost(circuit, H, dev)

##############################################################################
# As a reminder, we also built the total spin operator :math:`\hat{S}^2` for which
# we can now define a function to compute its expectation value:

S2_exp_value = qml.ExpvalCost(circuit, S2, dev)

##############################################################################
# The total spin :math:`S` of the trial state can be obtained from the
# expectation value :math:`\langle \hat{S}^2 \rangle` as,
#
# .. math::
#
#     S = -\frac{1}{2} + \sqrt{\frac{1}{4} + \langle \hat{S}^2 \rangle}.
#
# We define a function to compute the total spin.


def total_spin(params):
    return -0.5 + np.sqrt(1 / 4 + S2_exp_value(params))


##############################################################################
# Now, we proceed to minimize the cost function to find the ground state. We define
# the classical optimizer and initialize the circuit parameters.

opt = qml.GradientDescentOptimizer(stepsize=0.8)
np.random.seed(0)  # for reproducibility
theta = np.random.normal(0, np.pi, len(singles) + len(doubles))
print(theta)

##############################################################################
# We carry out the optimization over a maximum of 100 steps aiming to reach a
# convergence tolerance of :math:`10^{-6}` for the value of the cost function.

max_iterations = 100
conv_tol = 1e-06

for n in range(max_iterations):

    theta, prev_energy = opt.step_and_cost(cost_fn, theta)

    energy = cost_fn(theta)
    spin = total_spin(theta)

    conv = np.abs(energy - prev_energy)

    if n % 4 == 0:
        print(f"Step = {n}, Energy = {energy:.8f} Ha, S = {spin:.4f}")

    if conv <= conv_tol:
        break

print("\n" f"Final value of the ground-state energy = {energy:.8f} Ha")
print("\n" f"Optimal value of the circuit parameters = {theta}")

##############################################################################
# As a result, we have estimated the lowest-energy state of the hydrogen molecule
# with total spin :math:`S = 0` which corresponds to the ground state.
#
# Finding the lowest-lying excited state with :math:`S=1`
# -------------------------------------------------------
# In the last part of the tutorial, we use VQE to find the lowest-lying
# excited state of the hydrogen molecule with total spin quantum numbers :math:`S=1`.
# In this case, we use the :func:`~.pennylane_qchem.qchem.excitations` function to generate
# excitations whose total-spin projection differs by the quantity ``delta_sz=1``
# with respect to the Hartree-Fock reference state.

singles, doubles = qml.qchem.excitations(electrons, qubits, delta_sz=1)
print(singles)
print(doubles)

##############################################################################
# For the :math:`\mathrm{H}_2` molecule in a minimal basis set there are no
# double excitations, but only a spin-flip single excitation from qubit 1 to 2.
# In this case, the circuit will contain only one :class:`~.pennylane.SingleExcitation`
# operation. Furthermore, we initialize the qubit states to encode the double
# excitation :math:`\vert 0011 \rangle`.


def circuit(params, wires):
    qml.templates.AllSinglesDoubles(params, wires, np.flip(hf), singles, doubles)


##############################################################################
# Now the trial states prepared by the circuit above have the form,
#
# .. math::
#
#     \vert \Psi^*(\theta) = C_{03}(\theta) \vert 0101 \rangle
#     + C_{0123}(\theta) \vert 0011 \rangle,
#
# where the first term corresponds to a spin-flip single excitation with :math:`S_z=-1` and the
# second term is a double excitation with :math:`S_z=0`. As it was mentioned in the introduction,
# the electronic Hamiltonian conserves the total spin-projection of the electronic states, meaning
# that one of the two components must be dropped by the VQE algorithm while minimizing the
# cost function. Since the double excitation term increases the energy, the optimized state
# will be :math:`\vert \Psi^*(\theta^*) = C_{03}(\theta^*) \vert 0101 \rangle` which is precisely
# the lowest-energy state in the spin quantum numbers :math:`S=1, S_z=-1`.

cost_fn = qml.ExpvalCost(circuit, H, dev)
S2_exp_value = qml.ExpvalCost(circuit, S2, dev)

##############################################################################
# Then, we generate the new set of initial parameters, and proceed with the VQE algorithm to
# optimize the new variational circuit.

np.random.seed(0)
theta = np.random.normal(0, np.pi, len(singles) + len(doubles))

max_iterations = 100
conv_tol = 1e-06

for n in range(max_iterations):

    theta, prev_energy = opt.step_and_cost(cost_fn, theta)

    energy = cost_fn(theta)
    spin = total_spin(theta)

    conv = np.abs(energy - prev_energy)

    if n % 4 == 0:
        print(f"Step = {n}, Energy = {energy:.8f} Ha, S = {spin:.4f}")

    if conv <= conv_tol:
        break

print("\n" f"Final value of the energy = {energy:.8f} Ha")
print("\n" f"Optimal value of the circuit parameters = {theta}")

##############################################################################
# As expected, the VQE algorithms has found the lowest-energy state with total spin
# :math:`S=1` which is an excited state of the hydrogen molecule.
#
# Conclusion
# ----------
# In this tutorial we have used the standard VQE algorithm to find the ground and the
# lowest-lying excited states of the hydrogen molecule. We have used the excitation operations
# implemented as Givens rotations to prepare the trial states of a molecule. By choosing the
# total-spin projection of the single- and double-excitations VQE ansatz,
# we were able to probe the lowest-energy eigenstates of the molecular Hamiltonian in different
# sectors of the spin quantum numbers. We showed that the optimized states were also eigenstates
# of the total-spin operator :math:`\hat{S}^2`.
#
# References
# ----------
#
# .. [#peruzzo2014]
#
#     A. Peruzzo, J. McClean *et al.*, "A variational eigenvalue solver on a photonic
#     quantum processor". `Nature Communications 5, 4213 (2014).
#     <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# .. [#qchemcircuits]
#
#     J.M. Arrazola, O. Di Matteo, N. Quesada, S. Jahangiri, A. Delgado, N. Killoran.
#     "Universal quantum circuits for quantum chemistry". arXiv:2106.13839, (2021)
