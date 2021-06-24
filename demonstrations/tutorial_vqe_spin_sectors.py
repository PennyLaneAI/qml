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

The Variational Quantum Eigensolver (VQE) algorithm [#peruzzo2014]_, [#cao2019]_ has proven
to be a valuable approach to find the lowest-energy eigenstate of the electronic Hamiltonian
of a molecule using a quantum computer [#kandala2017]_.

In the absence of `spin-orbit coupling <https://en.wikipedia.org/wiki/Spin-orbit_interaction>`_,
the eigenstates of the molecular Hamiltonian can be computed in a specific sector of the total
spin-projection quantum number :math:`S_z`. This is illustrated in the figure below for the energy
spectra of the hydrogen molecule. In this case, the ground state coresponds to a singlet state
while the lowest-lying excited states, with total spin :math:`S=1`, form a triplet
related to the spin components :math:`S_z=-1, 0, 1`.

|

.. figure:: /demonstrations/vqe_uccsd_obs/energy_spectra_h2_sto3g.png
    :width: 75%
    :align: center

|

In this tutorial you will learn how run VQE simulations to find the lowest-energy states
of a molecular Hamiltonian in different sectors of the spin quantum numbers.
For the sake of simplicity, we chose the :math:`\mathrm{H}_2` molecule but the
same methodology can be applied to simulate more complicated molecules. First, we build
the electronic Hamiltonian and the total-spin operators for which we want to compute expectation
values. Next, we specify how to build the variational circuits to prepare the trial states
in different spin sectors. Finally, we run the VQE algorithm to find the the ground and the
lowest-lying excited states of the hydrogen molecule.

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
# We also want to build the total spin-projection operator :math:`\hat{S}_z`,
#
# .. math::
#
#     \hat{S}_z = \sum_\alpha s_{z_\alpha} \hat{n}_\alpha.
#
# In the equation above the index :math:`\alpha` runs over the basis of spin-orbitals,
# :math:`\hat{n}_\alpha` is the particle number operator whose
# eigenvalues :math:`0` or :math:`1` gives the occupation of the orbitals, and
# :math:`s_{z_\alpha} = \pm 1/2` denotes the spin-projection quantum number
# of the single-particle state :math:`\vert \alpha \rangle`.
# 
# We can use the :func:`~.pennylane_qchem.qchem.spin_z` function to build the
# :math:`\hat{S}_z` observable. 

Sz = qml.qchem.spin_z(qubits)
print(Sz)

##############################################################################
# Defining the quantum circuit
# ----------------------------
# 
# PennyLane contains the :class:`~.pennylane.ExpvalCost` class to implement the VQE algorithm.
# We begin by defining the device, in this case a qubit simulator:

dev = qml.device("default.qubit", wires=qubits)

##############################################################################
# The next step is to define the variational quantum circuit used to prepare
# the state that minimizes the expectation value of the electronic Hamiltonian.
# In this example, we use the unitary coupled cluster ansatz truncated at
# the level of single and double excitations (UCCSD) [#romero2017]_.
#
# The UCCSD method is a generalization of the traditional CCSD formalism used in quantum chemistry
# to perform post-Hartree-Fock electron correlation calculations. Within the first-order
# Trotter approximation [#suzuki1985]_, the UCCSD ground state of the molecule is built via the exponential ansatz
# [#barkoutsos2018]_,
#
# .. math::
#
#     \hat{U}(\vec{\theta}) = \prod_{p > r} \mathrm{exp}
#     \Big\{\theta_{pr}(\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
#     \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
#     (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}.
#
# In the latter equation, the indices :math:`r, s` and :math:`p, q` run respectively over the
# occupied and unoccupied molecular orbitals. The operator
# :math:`\hat{c}_p^\dagger \hat{c}_r` creates a single excitation [#jensenbook]_ since it
# annihilates an electron in the occupied orbital :math:`r` and creates it in the unoccupied
# orbital :math:`p`. Similarly, the operator
# :math:`\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s` creates a double excitation.
#
# The quantum circuits to exponentiate the excitation operators in the
# Jordan-Wigner representation [#barkoutsos2018]_ are implemented by the
# functions :func:`~.pennylane.templates.subroutines.SingleExcitationUnitary` and
# :func:`~.pennylane.templates.subroutines.DoubleExcitationUnitary` contained
# in the PennyLane templates library. Finally, the parameters :math:`\theta_{pr}` and
# :math:`\theta_{pqrs}` have to be optimized to minimize the expectation value,
#
# .. math::
#
#     E(\vec{\theta}) = \langle \mathrm{HF} \vert \hat{U}^\dagger(\vec{\theta})
#     \hat{H} \hat{U}(\vec{\theta}) \vert \mathrm{HF} \rangle,
#
# where :math:`\vert \mathrm{HF} \rangle` is the Hartree-Fock state. The total number of
# excitations determines the number of parameters :math:`\theta` to be optimized by the
# VQE algorithm as well as the depth of the UCCSD quantum circuit. For more details,
# check the documentation of the
# :func:`~.pennylane.templates.subroutines.SingleExcitationUnitary` and
# :func:`~.pennylane.templates.subroutines.DoubleExcitationUnitary` functions.
#
# Now, we demonstrate how to use PennyLane functionalities to build up the UCCSD
# ansatz for VQE simulations. First, we use the :func:`~.pennylane_qchem.qchem.excitations`
# function to generate the whole set of single- and double-excitations for :math:`N_e`
# ``electrons`` populating ``qubits`` spin orbitals. Furthermore, we can define the selection rules
# :math:`s_{z_p} - s_{z_r} = \Delta s_z` and
# :math:`s_{z_p} + s_{z_q} - s_{z_r} - s_{z_s}= \Delta s_z` for the spin-projection of the
# molecular orbitals involved in the single and double excitations
# using the keyword argument ``delta_sz``. This allows us to prepare a
# correlated state whose total-spin projection :math:`S_z` is different from the one of
# the Hartree-Fock state by the quantity ``delta_sz``. Therefore, we choose ``delta_sz = 0`` to
# prepare the ground state of the :math:`\mathrm{H}_2` molecule.

singles, doubles = qchem.excitations(electrons, qubits, delta_sz=0)
print(singles)
print(doubles)

##############################################################################
# The output lists ``singles`` and ``doubles`` contain the indices representing the single and
# double excitations. For the hydrogen molecule in a minimal basis set we have two single
# and one double excitations. The latter means that in preparing the UCCSD ansatz the
# :func:`~.pennylane.templates.subroutines.SingleExcitationUnitary` function has to be called
# twice to exponentiate the two single-excitation operators while the
# :func:`~.pennylane.templates.subroutines.DoubleExcitationUnitary` function is invoked once to
# exponentiate the double excitation.
#
# We use the function
# :func:`~.pennylane_qchem.qchem.excitations_to_wires` to generate the set of wires that the UCCSD
# circuit will act on. The inputs to this function are the indices stored in the
# ``singles`` and ``doubles`` lists.

s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
print(s_wires)
print(d_wires)

##############################################################################
# Next, we need to define the reference state that the UCCSD unitary acts on, which is
# just the Hartree-Fock state. This can be done straightforwardly with the
# :func:`~.pennylane_qchem.qchem.hf_state`
# function. The output of this function is an array containing the occupation-number vector
# representing the Hartree-Fock state.

hf_state = qchem.hf_state(electrons, qubits)
print(hf_state)

##############################################################################
# Finally, we can use the :func:`~.pennylane.templates.subroutines.UCCSD` function to define
# our VQE ansatz,

ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

##############################################################################
# Next, we use the PennyLane class :class:`~.pennylane.ExpvalCost` to define the cost function.
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
# Wrapping up, we fix an optimizer and randomly initialize the circuit parameters.

opt = qml.GradientDescentOptimizer(stepsize=0.4)
np.random.seed(0)  # for reproducibility
params = np.random.normal(0, np.pi, len(singles) + len(doubles))
print(params)

##############################################################################
# We carry out the optimization over a maximum of 100 steps, aiming to reach a convergence
# tolerance of :math:`\sim 10^{-6}`. Furthermore, we track the value of
# the total spin :math:`S` of the prepared state as it is optimized through
# the iterative procedure.

max_iterations = 100
conv_tol = 1e-06

for n in range(max_iterations):
    params, prev_energy = opt.step_and_cost(cost_fn, params)
    energy = cost_fn(params)
    conv = np.abs(energy - prev_energy)

    spin = total_spin(params)

    if n % 4 == 0:
        print("Iteration = {:},  E = {:.8f} Ha,  S = {:.4f}".format(n, energy, spin))

    if conv <= conv_tol:
        break

print()
print("Final convergence parameter = {:.8f} Ha".format(conv))
print("Final value of the ground-state energy = {:.8f} Ha".format(energy))
print(
    "Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)".format(
        np.abs(energy - (-1.1361894507)), np.abs(energy - (-1.1361894507)) * 627.509474
    )
)

##############################################################################
# Success! 🎉🎉🎉 We have estimated the lowest-energy state with total-spin projection
# :math:`S_z=0` for the hydrogen molecule, which is the ground state, with chemical
# accuracy. Notice also that the optimized UCCSD state is an eigenstate of the total spin
# operator :math:`\hat{S}^2` with eigenvalue :math:`S=0`.
#
# Finding the lowest-energy excited state with :math:`S=1`
# --------------------------------------------------------
# In the last part of the tutorial we want to demonstrate that VQE can also be used to find
# the lowest-energy excited states with total spin :math:`S=1` and :math:`S_z \neq 0`.
# For the hydrogen molecule, this is the case for the states with energy
# :math:`E = -0.4784529844` Ha and :math:`S_z=1` and :math:`S_z=-1`.
#
# Let's consider the case of :math:`S_z=-1` for which we can use the
# :func:`~.pennylane_qchem.qchem.excitations` function with the keyword argument ``delta_sz=1``.

singles, doubles = qchem.excitations(electrons, qubits, delta_sz=1)
print(singles)
print(doubles)

##############################################################################
# Notice that for the hydrogen molecule in a minimal basis set there are no
# double excitations but only a single excitation
# corresponding to the spin-flip transition between orbitals 1 and 2. And, that's it!.
# From this point on the algorithm is the same as described above.

s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
hf_state = qchem.hf_state(electrons, qubits)

ansatz = partial(UCCSD, init_state=hf_state, s_wires=s_wires, d_wires=d_wires)

cost_fn = qml.ExpvalCost(ansatz, H, dev)
S2_exp_value = qml.ExpvalCost(ansatz, S2, dev)

##############################################################################
# Then, we generate the new set of initial parameters, and proceed with the VQE algorithm to
# optimize the new variational circuit.

np.random.seed(0)
params = np.random.normal(0, np.pi, len(singles) + len(doubles))

max_iterations = 100
conv_tol = 1e-06

for n in range(max_iterations):
    params, prev_energy = opt.step_and_cost(cost_fn, params)
    energy = cost_fn(params)
    conv = np.abs(energy - prev_energy)

    spin = total_spin(params)

    if n % 4 == 0:
        print("Iteration = {:},  E = {:.8f} Ha,  S = {:.4f}".format(n, energy, spin))

    if conv <= conv_tol:
        break

print()
print("Final convergence parameter = {:.8f} Ha".format(conv))
print("Energy of the lowest-lying excited state = {:.8f} Ha".format(energy))
print(
    "Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)".format(
        np.abs(energy - (-0.4784529849)), np.abs(energy - (-0.4784529849)) * 627.509474
    )
)

##############################################################################
# As expected, we have successfully estimated the lowest-energy state with total spin
# :math:`S=1` and :math:`S_z=-1` for the hydrogen molecule, which is an excited state.
#
# Now, you can run a VQE simulation to find the degenerate excited state with
# spin quantum numbers :math:`S=1` and :math:`S_z=1`. Give it a try!
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
# .. [#cao2019]
#
#     Y. Cao, J. Romero, *et al.*, "Quantum chemistry in the age of quantum computing".
#     `Chem. Rev. 2019, 119, 19, 10856-10915.
#     <https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803>`__
#
# .. [#kandala2017]
#
#     A. Kandala, A. Mezzacapo *et al.*, "Hardware-efficient variational quantum
#     eigensolver for small molecules and quantum magnets". `arXiv:1704.05018
#     <https://arxiv.org/abs/1704.05018>`_
#
# .. [#romero2017]
#
#     J. Romero, R. Babbush, *et al.*, "Strategies for quantum computing molecular
#     energies using the unitary coupled cluster ansatz". `arXiv:1701.02691
#     <https://arxiv.org/abs/1701.02691>`_
#
# .. [#jensenbook]
#
#     F. Jensen. "Introduction to computational chemistry".
#     (John Wiley & Sons, 2016).
#
# .. [#fetterbook]
#
#     A. Fetter, J. D. Walecka, "Quantum theory of many-particle systems".
#     Courier Corporation, 2012.
#
# .. [#suzuki1985]
#
#     M. Suzuki. "Decomposition formulas of exponential operators and Lie exponentials
#     with some applications to quantum mechanics and statistical physics".
#     `Journal of Mathematical Physics 26, 601 (1985).
#     <https://aip.scitation.org/doi/abs/10.1063/1.526596>`_
#
# .. [#barkoutsos2018]
#
#     P. Kl. Barkoutsos, J. F. Gonthier, *et al.*, "Quantum algorithms for electronic structure
#     calculations: particle/hole Hamiltonian and optimized wavefunction expansions".
#     `arXiv:1805.04340. <https://arxiv.org/abs/1805.04340>`_
