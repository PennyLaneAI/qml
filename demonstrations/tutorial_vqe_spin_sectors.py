r"""
VQE in different spin sectors
=============================

.. meta::
    :property="og:description": Find the lowest-energy states of a Hamiltonian in different spin sectors
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_spectra_h2.png

.. related::
   tutorial_vqe Variational Quantum Eigensolver
   tutorial_vqe_parallel VQE with parallel QPUs

*Author: PennyLane dev team. Last updated: 25 June 2021.*

The Variational Quantum Eigensolver (VQE) algorithm has proven to be a valuable approach
to find the lowest-energy eigenstate of the electronic Hamiltonian of a molecule using a
quantum computer [#peruzzo2014]_.

In the absence of `spin-orbit coupling <https://en.wikipedia.org/wiki/Spin-orbit_interaction>`_,
the eigenstates of the molecular Hamiltonian can be computed in a specific sector of the total
spin quantum numbers. This is illustrated in the figure below for the energy
spectra of the hydrogen molecule. In this case, the ground state corresponds to a singlet state
while the lowest-lying excited states, with total spin :math:`S=1`, form a triplet
related to the spin components :math:`S_z=-1, 0, 1`.

|

.. figure:: /demonstrations/vqe_spin_sectors/energy_spectra_h2_sto3g.png
    :width: 75%
    :align: center

|

In this tutorial you will learn how to run VQE simulations to find the lowest-energy states
of a molecular Hamiltonian in different sectors of the total spin.
For the sake of simplicity, we illustrate this for the hydrogen molecule even though
the same methodology can be applied to more involving molecules. First, we show how to build
the electronic Hamiltonian and the total-spin operator for which we want to compute the
expectation values. Next, we use the unitary coupled-clusters singles and doubles (UCCSD)
*ansatz* [#romero2017]_ to prepare the states of the molecule in different spin sectors.
Finally, we run the VQE algorithm to find the the ground and the lowest-lying excited states
of the :math:`\mathrm{H}_2` molecule.

Let's get started!

Building the Hamiltonian and the spin operator :math:`\hat{S}_z`
----------------------------------------------------------------
First, we need to specify the structure of the molecule. This is done by providing a list
with the symbols of the constituent atoms and a one-dimensional array with the corresponding
nuclear coordinates in `atomic units <https://en.wikipedia.org/wiki/Hartree_atomic_units>`_.
"""

import numpy as np

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])

##############################################################################
# The molecular structure can also be imported from an external file using
# the :func:`~.pennylane_qchem.qchem.read_structure` function.
#
# Now, we can build the electronic Hamiltonian. Here, we use a minimal `basis set
# <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`_ to model the hydrogen molecule.
# In this approximation, the qubit Hamiltonian of the molecule in the Jordan-Wigner
# representation is built using the :func:`~.pennylane_qchem.qchem.molecular_hamiltonian`
# function.

import pennylane as qml

H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)

##############################################################################
# The outputs of the function are the Hamiltonian and the number of qubits
# required for the quantum simulations. For the :math:`\mathrm{H}_2` molecule in a minimal
# basis, we have four molecular **spin**-orbitals which defines the number of qubits.
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
# :math:`\hat{s}_1 \cdot \hat{s}_2` in the basis of *spin* orbitals
#
# We use the :func:`~.pennylane_qchem.qchem.spin2` function to build the
# :math:`\hat{S}^2` observable.

electrons = 2
S2 = qml.qchem.spin2(electrons, qubits)
print(S2)

##############################################################################
# Defining the quantum circuit
# ----------------------------
# In this section we define the variational quantum circuit used to prepare
# the trial states. In this example, we use the first-order Trotter approximation of
# the unitary coupled cluster *ansatz* singles and doubles (UCCSD) [#barkoutsos2018]_,
#
# .. math::
#
#     \hat{U}(\vec{\theta}) = \prod_{p > r} \mathrm{exp}
#     \Big\{\theta_{pr}(\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
#     \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
#     (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}.
#
# In the equation above, the indices :math:`r, s` and :math:`p, q` run respectively over the
# occupied and unoccupied Hartree-Fock orbitals. The operator
# :math:`\hat{c}_p^\dagger \hat{c}_r` creates a single excitation since it
# annihilates an electron in the occupied orbital :math:`r` and creates it in the unoccupied
# orbital :math:`p`. Similarly, the operator
# :math:`\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s` creates a double excitation.
#
# Now, we demonstrate how to use PennyLane functionalities to build up the
# `UCCSD ansatz
# <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.subroutines.UCCSD.html>`_
# for VQE simulations. First, we need to define the Hartree-Fock (HF) reference state
# which the UCCSD unitary acts on. This is done with the :func:`~.pennylane_qchem.qchem.hf_state`
# function. It generates a vector representing the HF state :math:`\vert 1100 \rangle`
# that is used to initialize the qubit register.

hf = qml.qchem.hf_state(electrons, qubits)
print(hf_state)

##############################################################################
# Next, we use the :func:`~.pennylane_qchem.qchem.excitations`
# function to generate all single- and double-excitations of the HF reference state.
# This function allows us to define the keyword argument ``delta_sz``
# to specify the variation of the total-spin projection of the generated excitations
# with respect to the reference state. This is illustrated in the figure below for two
# electrons and four spin-orbitals.
#
# |
#
# .. figure:: /demonstrations/vqe_spin_sectors/fig_excitations.png
#     :width: 75%
#     :align: center
#
# |
#
# To find the ground state of the :math:`\mathrm{H}_2` molecule we choose ``delta_sz = 0``.

singles, doubles = qml.qchem.excitations(electrons, qubits, delta_sz=0)
print(singles)
print(doubles)

##############################################################################
# The output lists ``singles`` and ``doubles`` contain the qubit indices involved in the
# single and double excitations. Even (odd) indices correspond to spin-up (spin-down) orbitals.
# For ``delta_sz = 0`` we have two single excitations, one from qubit 0 to 2 and the other from
# qubit 1 to 3, and one double excitation from qubits 0, 1 to 2, 3.
#
# We use the function :func:`~.pennylane_qchem.qchem.excitations_to_wires` to generate the
# set of wires that the UCCSD circuit will act on. The inputs to this function are the indices
# stored in the ``singles`` and ``doubles`` lists.

s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
print(s_wires)
print(d_wires)

##############################################################################
# Finally, we can use the :func:`~.pennylane.templates.subroutines.UCCSD` function to define
# our VQE ansatz.

ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

##############################################################################
# Running the VQE simulation
# --------------------------
# We begin by defining the device, in this case a qubit simulator:

dev = qml.device("default.qubit", wires=qubits)

##############################################################################
# Next, we use the :class:`~.pennylane.ExpvalCost` class to define the cost function.
# This requires specifying the circuit, target Hamiltonian, and the device. It returns
# a cost function that can be evaluated with the circuit parameters:

cost_fn = qml.ExpvalCost(ansatz, H, dev)

##############################################################################
# As a reminder, we also built the total spin operator :math:`\hat{S}^2` for which
# we can now define a function to compute its expectation value:

S2_exp_value = qml.ExpvalCost(ansatz, S2, dev)

##############################################################################
# The total spin :math:`S` of the trial state can be obtained from the
# expectation value :math:`\langle \hat{S}^2 \rangle` as,
#
# .. math::
#
#     S = -\frac{1}{2} + \sqrt{\frac{1}{4} + \langle \hat{S}^2 \rangle}.
#
# We define a function to compute the total spin


def total_spin(params):
    return -0.5 + np.sqrt(1 / 4 + S2_exp_value(params))


##############################################################################
# Now we proceed to minimize the cost function to find the ground state. We define
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
# In the last part of the tutorial we use VQE to find the lowest-lying
# excited state of the hydrogen molecule with total spin :math:`S=1`. In this case,
# we use the :func:`~.pennylane_qchem.qchem.excitations` function to generate
# excitations whose total-spin projection differs by the quantity ``delta_sz=1`
# with respect to the Hartree-Fock reference state.

singles, doubles = qml.qchem.excitations(electrons, qubits, delta_sz=1)
print(singles)
print(doubles)

##############################################################################
# For the :math:`\mathrm{H}_2` molecule in a minimal basis set there are no
# double excitations but only a spin-flip single excitation from qubit 1 to 2.
# And, that's it!. From this point on the algorithm is the same as described above.

s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)

ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

cost_fn = qml.ExpvalCost(ansatz, H, dev)
S2_exp_value = qml.ExpvalCost(ansatz, S2, dev)

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
# In this case, the VQE algorithms finds the lowest-energy state with total spin
# :math:`S=1` which is an excited state of the hydrogen molecule.
#
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
# .. [#romero2017]
#
#     J. Romero, R. Babbush, *et al.*, "Strategies for quantum computing molecular
#     energies using the unitary coupled cluster ansatz". `arXiv:1701.02691
#     <https://arxiv.org/abs/1701.02691>`_
#
# .. [#barkoutsos2018]
#
#     P. Kl. Barkoutsos, J. F. Gonthier, *et al.*, "Quantum algorithms for electronic structure
#     calculations: particle/hole Hamiltonian and optimized wavefunction expansions".
#     `arXiv:1805.04340. <https://arxiv.org/abs/1805.04340>`_
