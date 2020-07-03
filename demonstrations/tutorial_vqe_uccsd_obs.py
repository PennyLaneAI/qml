r"""
VQE in different spin sectors with the Unitary Coupled Cluster ansatz
=====================================================================

.. meta::
    :property="og:description": Find the lowest-energy states of a Hamiltonian in different
        spin sectors using the variational quantum eigensolver
        algorithm in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/energy_spectra_h2_sto3g.png

Quantum computers offer a promising avenue to perform first-principles simulations of the
electronic structure of molecules and materials that are currently intractable using classical
high-performance computers. In particular, the Variational Quantum Eigensolver (VQE) algorithm
:ref:`[1, 2]<vqe_uccsd_references>` has proven to be a valuable quantum-classical
approach to find the lowest-energy eigenstate of the electronic Hamiltonian by using Noisy
Intermediate-Scale Quantum (NISQ) devices :ref:`[3]<vqe_uccsd_references>`.

In the absence of `spin-orbit coupling <https://en.wikipedia.org/wiki/Spin-orbit_interaction>`_, the
electronic Hamiltonian matrix is block diagonal in the total spin projection quantum number
:math:`S_z`. This allows us to compute the energy spectrum of the Hamiltonian in a given sector of
:math:`S_z`. For example, the figure below shows the energy spectra of the hydrogen molecule
calculated in different sectors of the total-spin projection. The ground state with
energy :math:`E_\mathrm{gs}=-1.136189` Ha has total spin :math:`S=0` and spin
projection :math:`S_z=0`.
The lowest-lying excited states, with energy :math:`E^*=-0.478453` Ha and total spin
:math:`S=1`, show a three-fold spin degeneracy related to the values of :math:`S_z=-1, 0, 1`.

|

.. figure:: /demonstrations/vqe_uccsd_obs/energy_spectra_h2_sto3g.png
    :width: 75%
    :align: center
|    

Similarly, in the framework of VQE, if the quantum computer can be programmed to prepare many-body
states in a specific sector of the total-spin projection :math:`S_z`, the variational optimization
algorithm will allow us to estimate the energy of the lowest-energy state in this spin sector.
More specifically, if we run a VQE simulation for the :math:`\mathrm{H}_2` molecule in the
subspace of states with :math:`S_z=0`, we will find the ground-state energy of the molecule. If the
VQE simulation is performed in the subspace with :math:`S_z=1`, the optimized state will be in
practice an excited state of the molecule, as shown in the figure above.

At the core of the VQE algorithm is the variational quantum circuit that is optimized to prepare
the desired quantum states. The choice of circuit is crucial for the success of the algorithm. The
unitary coupled cluster ansatz :ref:`[4]<vqe_uccsd_references>` is a powerful quantum circuit that
is believed to outperform the classical coupled cluster method :ref:`[5]<vqe_uccsd_references>`,
traditionally referred to as the gold standard of quantum chemistry.

We demonstrate how different functionalities implemented in PennyLane
can be put together to run VQE simulations in different sectors of the total-spin projection
:math:`S_z`. We also specify how to use the unitary coupled cluster ansatz, restricted to single
and double excitations (UCCSD), as the variational circuit for the algorithm. These functionalities
can be combined to estimate the energies of the ground and the lowest-lying excited states of the 
hydrogen molecule.

Let's get started! ⚛️

Building the Hamiltonian and the total spin observable :math:`\hat{S}^2`
------------------------------------------------------------------------

The first step is to import the required libraries and packages:
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
from pennylane.templates.subroutines import UCCSD

##############################################################################
# The second step is to specify the molecule whose properties we aim to calculate.
# This is done by providing the name, geometry and charge of the molecule. For more details
# take a look at the tutorial :doc:`tutorial_quantum_chemistry`.

name = "h2"

##############################################################################
# The geometry of the molecule can be given in any format recognized by Open Babel.
# Here, we used a locally saved file in
# `xyz format <https://en.wikipedia.org/wiki/XYZ_file_format>`_ specifying the
# three-dimensional coordinates and symbols of the atomic species.

geometry = "h2.xyz"

##############################################################################
# In this example, we consider a neutral molecule:

charge = 0

##############################################################################
# Now, we define two input parameters required to compute the Hartree-Fock (HF)
# electronic structure of the molecule. First, the
# `multiplicity <https://en.wikipedia.org/wiki/Multiplicity_(chemistry)>`_ of the
# `Hartree-Fock state <https://en.wikipedia.org/wiki/Hartree-Fock_method>`_, and
# the second one is the `atomic basis set <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`_
# used to represent the molecular orbitals. In this example, we use the minimal
# basis STO-3g.

multiplicity = 1
basis_set = "sto-3g"

##############################################################################
# PennyLane allows us to define an `active space
# <https://en.wikipedia.org/wiki/Complete_active_space>`_ to expand the second-quantized
# Hamiltonian or any other observable relevant to compute molecular properties.
# The active space, as explained in the tutorial :doc:`tutorial_quantum_chemistry`, is built
# by specifying the number of active electrons and active orbitals.
# For the hydrogen molecule described with a minimal basis set we include all Hartree-Fock orbitals
# in the basis of single-particle states.

n_electrons = 2
n_orbitals = 2

##############################################################################
# Finally, to build the electronic Hamiltonian we have to define the fermionic-to-qubit
# mapping, which can be either Jordan-Wigner (``jordan_wigner``) or Bravyi-Kitaev
# (``bravyi_kitaev``). The outputs of the function :func:`~.generate_hamiltonian` are
# the qubit Hamiltonian of the molecule and the number of qubits needed to represent it:

h, n_qubits = qchem.generate_hamiltonian(
    name,
    geometry,
    charge,
    multiplicity,
    basis_set,
    n_active_electrons=n_electrons,
    n_active_orbitals=n_orbitals,
    mapping="jordan_wigner",
)

print("Number of qubits = ", n_qubits)
print("Hamiltonian is ", h)

##############################################################################
# We also want to build the total spin operator :math:`\hat{S}^2`,
#
# .. math::
#
#     \hat{S}^2 = \frac{3}{4} N_e + \sum_{\alpha, \beta, \gamma, \delta}
#     \langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2
#     \vert \gamma, \delta \rangle ~ \hat{c}_\alpha^\dagger \hat{c}_\beta^\dagger
#     \hat{c}_\gamma \hat{c}_\delta.
#
# In the equation above, :math:`N_e` is the number of active electrons,
# :math:`\hat{c}` and :math:`\hat{c}^\dagger`) are respectively the electron annihilation and
# creation operators, and
# :math:`\langle \alpha, \beta \vert \hat{s}_1 \cdot \hat{s}_2 \vert \gamma, \delta \rangle`
# is the matrix element of the two-particle spin operator
# :math:`\hat{s}_1 \cdot \hat{s}_2` in the basis of Hartree-Fock spin-orbitals
# :ref:`[6]<vqe_uccsd_references>`.
#
# We need to compute these matrix elements. This is achieved by calling the
# :func:`~.get_spin2_matrix_elements`. It reads the Hartree-Fock electronic
# structure, defines the active space, and outputs the first term :math:`\frac{3}{4} N_e`
# as well as the table of matrix elements.

s2_matrix_elements, first_term = qchem.get_spin2_matrix_elements(
    name, "pyscf/sto-3g", n_active_electrons=n_electrons, n_active_orbitals=n_orbitals
)
print(first_term)
print(s2_matrix_elements)

##############################################################################
# We have explicitly inputted the path ``'pyscf/sto-3g'`` to the locally
# saved file ``'pyscf/sto-3g/h2.hdf5'`` storing the HF electronic structure
# of the :math:`\mathrm{H}_2` molecule. This data file was generated by the
# :func:`~.meanfield_data` as it is used by :func:`generate_hamiltonian` function
# to build the Hamiltonian.
#
# Now that we have the two-particle spin matrix elements, we call the
# :func:`~.observable` function to build the Fermionic operator and represent it in
# the basis of Pauli matrices.

s2_obs = qchem.observable(s2_matrix_elements, init_term=first_term, mapping="jordan_wigner")
print(s2_obs)

##############################################################################
# .. note::
#
#     The :func:`~.observable` function can be used to build any second-quantized many-body
#     observable as long as we have access to the matrix elements of the one- and/or
#     two-particle operators in the basis of single-particle states.
#     The keyword argument ``init_term`` contains the contribution
#     of core orbitals, if any, or any other quantity required to initialize the observable.


##############################################################################
# Implementing VQE with the UCCSD ansatz
# --------------------------------------
#
# PennyLane contains the :class:`~.VQECost` class to implement the VQE algorithm.
# We begin by defining the device, in this case a qubit simulator:

dev = qml.device("default.qubit", wires=n_qubits)

##############################################################################
# The next step is to define the variational quantum circuit used to prepare
# the state that minimizes the expectation value of the electronic Hamiltonian.
# In this example, we use the unitary coupled cluster ansatz truncated at
# the level of single and double excitations (UCCSD) :ref:`[4]<vqe_uccsd_references>`.
#
# The UCCSD method is a generalization of the traditional CCSD formalism used in quantum chemistry
# to perform post-Hartree-Fock electron correlation calculations. Within the first-order
# Trotter approximation, the UCCSD ground state of the molecule is built via the exponential ansatz
# :ref:`[7]<vqe_uccsd_references>`,
#
# .. math::
#
#     \hat{U}(\vec{\theta}) = \prod_{p > r} \mathrm{exp}
#     \Big\{\theta_{pr}(\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
#     \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
#     (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}.
#
# In the latter equation, the indices :math:`r, s` and :math:`p, q` run respectively over the
# occupied and unoccupied (virtual) molecular orbitals. The operator
# :math:`\hat{c}_p^\dagger \hat{c}_r` creates a 1particle-1hole (ph)
# excitation :ref:`[8]<vqe_uccsd_references>` since it annihilates a particle in the
# occupied orbital :math:`r` (creates a hole) and creates the particle in the virtual orbital
# :math:`p`. Similarly, the operator
# :math:`\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s` creates 2particle-2hole (pphh)
# excitations. The quantum circuits to exponentiate the excitation operators in the
# Jordan-Wigner representation :ref:`[7]<vqe_uccsd_references>` are implemented by the
# functions :func:`~.pennylane.templates.SingleExcitationUnitary` and
# :func:`~.pennylane.templates.DoubleExcitationUnitary` contained
# in the PennyLane templates library. Finally, the parameters :math:`\theta_{pr}` and
# :math:`\theta_{pqrs}` have to be optimized to minimize the expectation value,
#
# .. math::
#
#     E(\vec{\theta}) = \langle \mathrm{HF} \vert \hat{U}^\dagger(\vec{\theta})
#     \hat{H} \hat{U}(\vec{\theta}) \vert \mathrm{HF} \rangle.
#
# The total number of particle-hole excitations determines the number of parameters
# :math:`\theta` to be optimized by the VQE algorithm as well as the depth of the UCCSD
# quantum circuit. For more details, check the documentation of the
# :func:`~.pennylane.templates.SingleExcitationUnitary` and
# :func:`~.pennylane.templates.DoubleExcitationUnitary` functions.
#
# Now, we demonstrate how to use PennyLane functionalities to build up the UCCSD
# ansatz for VQE simulations. First, we use the :func:`~.sd_excitations` function to generate
# the whole set of particle-hole
# excitations for our active space with ``n_electrons`` in
# ``2*n_orbitals=n_qubits`` *spin*-orbitals. Furthermore, we can enforce the selection rules
# :math:`s_{z_p} - s_{z_r} = \Delta s_z` and
# :math:`s_{z_p} + s_{z_q} - s_{z_r} - s_{z_s}= \Delta s_z` for the spin-projection of the
# molecular orbitals involved in the ph and pphh excitations
# using the keyword argument ``delta_sz``. This is precisely what allows us to prepare a
# correlated trial state whose total-spin projection :math:`S_z` is different from the one of
# the Hartree-Fock state by the quantity ``delta_sz``. Therefore, we choose ``delta_sz = 0`` to
# prepare the ground state of the :math:`\mathrm{H}_2` molecule.

ph, pphh = qchem.sd_excitations(n_electrons, n_qubits, delta_sz=0)
print(ph)
print(pphh)

##############################################################################
# The output lists ``ph`` and ``pphh`` contain the indices representing the one and two
# particle-hole excitations. For the hydrogen molecule in a minimal basis set we have two ph
# excitations and one pphh excitation. The latter means that in preparing the UCCSD ansatz the
# :func:`~.pennylane.templates.SingleExcitationUnitary` function has to be called twice to
# exponentiate the two ph excitation operators while the
# :func:`~.pennylane.templates.DoubleExcitationUnitary` function is invoked once to
# exponentiate the pphh excitation.
#
# We use the function
# :func:`~.excitations_to_wires` to generate the set of wires that the UCCSD circuit will
# act on. The inputs to this function are the indices stored in the ``ph`` and ``pphh``.

ph_wires, pphh_wires = qchem.excitations_to_wires(ph, pphh)
print(ph_wires)
print(pphh_wires)

##############################################################################
# Next, we need to define the reference state that the UCCSD unitary acts on, which is
# just the Hartree-Fock state. This can be done straightforwardly with the :func:`~.hf_state`
# function. The output of this function is an array containing the occupation-number vector
# representing the Hartree-Fock state.

ref_state = qchem.hf_state(n_electrons, n_qubits)
print(ref_state)

##############################################################################
# Finally, we can use the :func:`~.UCCSD` function to define our VQE ansatz,


def vqe_ansatz(params, wires, init_state=ref_state, ph=ph_wires, pphh=pphh_wires):
    UCCSD(params, wires, init_state=ref_state, ph=ph_wires, pphh=pphh_wires)


##############################################################################
# where ``wires`` contains the wires the template act on and ``params`` is a vector
# containing the trainable parameters :math:`\theta_{pr}` and :math:`\theta_{pqrs}`.
#
# Next, we use the PennyLane class :class:`~.VQECost` to define the cost function.
# This requires specifying the circuit, target Hamiltonian, and the device. It returns
# a cost function that can be evaluated with the circuit parameters:

cost_fn = qml.VQECost(vqe_ansatz, h, dev)

##############################################################################
# As a reminder, we also built the total spin operator
# :math:`\hat{S}^2` for which we can now define a function to compute its expectation
# value:

s2_exp_value = qml.VQECost(vqe_ansatz, s2_obs, dev)

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
    return -0.5 + np.sqrt(1 / 4 + s2_exp_value(params))

##############################################################################
# Wrapping up, we fix an optimizer and randomly initialize the circuit parameters.

opt = qml.GradientDescentOptimizer(stepsize=0.4)
np.random.seed(0)  # for reproducibility
params = np.random.normal(0, np.pi, len(ph) + len(pphh))
print(params)

##############################################################################
# We carry out the optimization over a maximum of 100 steps, aiming to reach a convergence
# tolerance of :math:`\sim 10^{-6}`. Furthermore, we track the value of
# the total spin of the prepared state as it is optimized through the iterative procedure.

max_iterations = 100
conv_tol = 1e-06
prev_energy = cost_fn(params)
for n in range(max_iterations):
    params = opt.step(cost_fn, params)
    energy = cost_fn(params)
    conv = np.abs(energy - prev_energy)

    spin = total_spin(params)

    if n % 2 == 0:
        print(
            "Iteration = {:},  energy = {:.8f} Ha,  Total Spin = {:.4f}, epsilon = {"
            ":.8f} Ha".format(n, energy, spin, conv)
        )

    if conv <= conv_tol:
        break

    prev_energy = energy

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
# operator :math:`\hat{S}_2` with eigenvalue :math:`S=0`.
#
#
# Finding the lowest-energy excited state with :math:`S_z=1`
# ----------------------------------------------------------
# In the last part of the tutorial we want to demonstrate that VQE can also be used to find
# the lowest-energy excited states with total-spin projection :math:`S_z \neq 0`.
# For the hydrogen molecule, this is the case for the states with energy
# :math:`E = -0.4784529844` Ha and :math:`Sz=1` and :math:`Sz=-1`.
#
# Let's consider the case of :math:`S_z=1`. We just use the
# :func:`~.sd_excitations` with the keyword argument ``delta_sz=1`` to generate
# particle-hole excitations in this specific sector of :math:`S_z`.

ph, pphh = qchem.sd_excitations(n_electrons, n_qubits, delta_sz=1)
print(ph)
print(pphh)

##############################################################################
# Notice that for the hydrogen molecule in a minimal basis set there are no
# 2particle-2hole (pphh) excitations but only 1particle-1hole excitation (ph)
# corresponding to the spin-flip transition between orbitals 1 and 2. And, that's it!.
# From this point on the algorithm is the same as described above.

ph_wires, pphh_wires = qchem.excitations_to_wires(ph, pphh)
ref_state = qchem.hf_state(n_electrons, n_qubits)


def vqe_ansatz(weights, wires, init_state=ref_state, ph=ph_wires, pphh=pphh_wires):
    UCCSD(weights, wires, init_state=ref_state, ph=ph_wires, pphh=pphh_wires)


cost_fn = qml.VQECost(vqe_ansatz, h, dev)
s2_exp_value = qml.VQECost(vqe_ansatz, s2_obs, dev)

##############################################################################
# Then, we generate the new set of initial parameters, and proceed with the VQE algorithm to
# optimize the new variational circuit.

np.random.seed(0)
params = np.random.normal(0, np.pi, len(ph) + len(pphh))

max_iterations = 100
conv_tol = 1e-06
prev_energy = cost_fn(params)
for n in range(max_iterations):
    params = opt.step(cost_fn, params)
    energy = cost_fn(params)
    conv = np.abs(energy - prev_energy)

    spin = total_spin(params)

    if n % 2 == 0:
        print(
            "Iteration = {:},  energy = {:.8f} Ha,  Total Spin = {:.4f}, epsilon = {"
            ":.8f} Ha".format(n, energy, spin, conv)
        )

    if conv <= conv_tol:
        break

    prev_energy = energy

print()
print("Final convergence parameter = {:.8f} Ha".format(conv))
print("Final value of the ground-state energy = {:.8f} Ha".format(energy))
print(
    "Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)".format(
        np.abs(energy - (-0.4784529849)), np.abs(energy - (-0.4784529849)) * 627.509474
    )
)

##############################################################################
# As expected, we have successfully estimated the lowest-energy state with total-spin projection
# :math:`S_z=1` for the hydrogen molecule, which is an excited state. Notice that
# the optimized UCCSD state is an eigenstate of the total spin operator :math:`\hat{S}_2` with
# eigenvalue :math:`S=1`.
#
# Now, you can run a VQE simulation to find the degenerate excited state with
# spin quantum numbers :math:`S=1` and :math:`S_z=-1`. Give it a try!
#
#
# .. _vqe_uccsd_references:
#
# References
# ----------
#
# 1. A. Peruzzo, J. McClean *et al.*, "A variational eigenvalue solver on a photonic
#    quantum processor". `Nature Communications 5, 4213 (2014).
#    <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# 2. Y. Cao, J. Romero, *et al.*, "Quantum Chemistry in the Age of Quantum Computing".
#    `Chem. Rev. 2019, 119, 19, 10856-10915.
#    <https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803>`__
#
# 3. A. Kandala, A. Mezzacapo *et al.*, "Hardware-efficient Variational Quantum
#    Eigensolver for Small Molecules and Quantum Magnets". `arXiv:1704.05018
#    <https://arxiv.org/abs/1704.05018>`_
#
# 4. J. Romero, R. Babbush, *et al.*,"Strategies for quantum computing molecular
#    energies using the unitary coupled cluster ansatz". `arXiv:1701.02691
#    <https://arxiv.org/abs/1701.02691>`_
#
# 5. F. Jensen. "Introduction to Computational Chemistry".
#    (John Wiley & Sons, 2016).
#
# 6. A. Fetter, J. D. Walecka, "Quantum Theory of many-particle systems".
#    Courier Corporation, 2012.
#
# 7. P. Kl. Barkoutsos, J. F. Gonthier, *et al.*, "Quantum algorithms for electronic structure
#    calculations: particle/hole Hamiltonian and optimized wavefunction expansions".
#    `arXiv:1805.04340. <https://arxiv.org/abs/1805.04340>`_
#
# 8. P. Ring, P. Schuck. "The Nuclear Many-Body Problem".
#    (Springer Science & Business Media, 2004).
