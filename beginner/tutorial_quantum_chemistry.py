r"""
Quantum Chemistry with PennyLane
================================

In quantum chemistry and materials science the term *electronic structure methods* refers to
the approximations used to find the many-electron wave function of polyatomic systems.
Electronic structure methods rely on the Born-Oppenheimer approximation [1],
which allows to write the electronic Hamiltonian of the molecule as an operator which depends
parametrically on the "frozen" nuclear positions.

Once the electronic problem is well defined, estimating the molecular properties with chemical
accuracy requires wave-function based electronic structure calculations. However,
even if we have access to powerful High Performance Computers (HPC), the application of
post-Hartree-Fock electron correlation methods [2] becomes extremely challenging even for
molecular systems with a few atoms.

Quantum computers offer a promising avenue for major breakthroughs in quantum chemistry. For
example, a quantum computer consisting of 50 qubits could naturally encode the wave function of the
water molecule, which on a classical computer would have to be obtained by diagonalizing a
Hamiltonian matrix with dimensions on the order of ​:math:`\sim 10^11`. In particular,
the Variational Quantum Eigensolver (VQE) [3] is a promising hybrid quantum-classical
computational scheme where a quantum computer is used to prepare the trial wave function of a
molecule and to measure the expectation value of the **electronic Hamiltonian**,
while a classical optimizer is used to adjust the quantum circuit parameters in order to find
the lowest eigenvalue of the measured Hamiltonian.

The goal of this tutorial is to highlight PennyLane's quantum chemistry functions and abilities
to build the electronic Hamiltonian of any molecule starting with its geometry and continue all
the way to obtaining the Hamiltonian represented in the basis of Pauli matrices. This tutorial
can also be an opportunity to become more familiar with fundamental concepts in quantum chemistry.

Sit down, brew a hot drink, and let's take a look!

Importing the molecular structure
---------------------------------
The first step is to import PennyLane.
"""
import pennylane as qml

##############################################################################
# In this example, we construct the electronic Hamiltonian of one of the most unique
# molecules: water. We begin by reading the positions of the oxygen and hydrogen atoms.  The
# equilibrium geometry of water can be downloaded from the The `NIST Chemistry WebBook
# <https://webbook.nist.gov/chemistry>`_ data base to the structure-data file
# :download:`h2o.xyz </beginner/h2o.xyz>`.
# We read the equilibrium geometry of water and store it in a list
# containing the symbol and the Cartesian coordinates of each atomic species:

geometry = qml.qchem.read_structure('h2o.xyz')
print("The total number of atoms is: {}".format(len(geometry)))
print(geometry)

##############################################################################
# .. note::
#
#     The xyz format is supported out of the box. If `Open Babel <http://openbabel.org/wiki/Main_Page>`_
#     is installed, any format recognized by Open Babel is also supported
#     by PennyLane, such as ``.mol`` and ``.sdf``.
#
#     Please see the :func:`~.read_structure` and Open Babel documentation
#     for more information on installing Open Babel.
#
# Calling the function :func:`~.read_structure` also creates the file
# ``structure.xyz``, which we can use to visualize our molecule using any molecule editor,
# e.g., `Avogadro <https://avogadro.cc/>`_.
#
# .. figure:: ../beginner/quantum_chemistry/water_structure.png
#     :width: 60%
#     :align: center
#
# Solve the Hartree-Fock equations
# --------------------------------
#
# The next step is to solve the `Hartree-Fock (HF) equations
# <https://en.wikipedia.org/wiki/Hartree-Fock_method>`__ [2] for our molecule. The HF
# method is a mean field approximation, where each electron in the molecule is treated as an
# **independent** particle that moves under the influence of the nuclei Coulomb potential and a
# `self-consistent field (SCF) <https://en.wikipedia.org/wiki/Hartree-Fock_method>`__
# generated by all other electrons. The Hartree-Fock approximation
# is typically the starting point for most electron correlation methods in quantum chemistry, such
# as `Configuration Interaction (CI) <https://en.wikipedia.org/wiki/Configuration_interaction>`__
# and `Coupled Cluster (CC) <https://en.wikipedia.org/wiki/Coupled_cluster>`__ methods among
# others [2].
#
# Before launching the HF calculation using the function :func:`meanfield_data`, we
# need to specify a string to label the molecule and the net charge of the molecule. In this
# example we choose ``'water'`` as the string. On the other hand, although positively or
# negatively charged molecules can be simulated, we choose a neutral system.

name = 'water'
charge = 0

##############################################################################
# In the Hartree-Fock method the many-electron wave function is approximated by a Slater
# determinant [4] that results from occupying the lowest-energy molecular orbitals until all
# electrons in the molecule are accommodated. The way molecular orbitals are occupied matters as
# they determine the self-consistent field.
#
# Let's focus on our water molecule with a total number of ten electrons. For example, the Slater
# determinant resulting from occupying the first five lowest-energy molecular orbitals with two
# *paired* electrons in each orbital, one with spin-up and the other with spin-down, is said to
# be a closed-shell HF state with spin *multiplicity* one. Alternatively, if we define an
# occupation where the first four orbitals are doubly-occupied and the next two are singly
# occupied by *unpaired* electrons with spin-up, this is said to be an open-shell HF state with
# *multiplicity* three.
#
# |
#
# .. figure:: ../beginner/quantum_chemistry/hf_references.png
#     :width: 50%
#     :align: center
#
# |
#
# The take-home message in this context, is that the multiplicity, which we can set as
# :math:`\pm (N_\mathrm{unpaired}^e + 1)` with :math:`N_\mathrm{unpaired}^e` being the number of
# unpaired electrons, determines the occupation of the molecular orbitals in the HF calculations.
# In this tutorial we will consider a closed-shell reference state.

multiplicity = 1

##############################################################################
# Now we need to define the atomic basis set. Hartree-Fock molecular orbitals
# are typically represented as a Linear Combination of Atomic Orbitals (LCAO) which are further
# approximated by using Gaussian function. The `Basis Set Exchange
# <https://www.basissetexchange.org/>`_ data base is an excellent source of Gaussian-type
# orbitals, although many of these basis sets are already incorporated in modern quantum
# chemistry packages. In this example we choose the `minimum basis set
# <https://en.wikipedia.org/wiki/Basis_set_(chemistry)#Minimal_basis_sets>`__ ``'sto-3g'`` of
# Slater-type orbitals (STO) which provides the minimum number of atomic orbitals required to
# accommodate the ten electrons in the water molecule.

basis_set = 'sto-3g'

##############################################################################
# Finally, we can call the function :func:`~.meanfield_data` to launch the mean field
# calculation. At present, the quantum chemistry packages `PySCF
# <https://sunqm.github.io/pyscf/>`_ or `Psi4 <http://www.psicode.org/>`_ can be chosen to solve
# the Hartree-Fock equations. In this example, we choose ``'pyscf'``, which is the default option,
# but the same results can be obtained using ``'psi4'``.

hf_data = qml.qchem.meanfield_data(name, geometry, charge,
                                   multiplicity, basis_set, qc_package='pyscf')

##############################################################################
# Once the calculation is completed,
# the  string variable ``hf_data`` returned by the function stores the path to the directory
# containing the file ``'water.hdf5'`` with the Hartree-Fock electronic structure of the water
# molecule.

import os
print(hf_data)
[print(file) for file in os.listdir(hf_data)]

##############################################################################
# At this stage, we have a basis set of molecular orbitals! Next, we can use the
# function :func:`~.active_space` to define an *active space*. But, what is an active
# space?
#
# Defining an active space
# ------------------------
#
# In general, post-Hartree-Fock electron correlation methods expand the molecule's wave
# function around the Hartree-Fock solution, by adding Slater determinants that result from
# exciting the electrons occupying the HF orbitals below the Fermi level to
# the unoccupied orbitals above it. Despite the fact that there are different techniques to
# truncate this expansion, the number of configurations increases combinatorially and the
# task of finding the wave function expansion coefficients becomes numerically intractable
# should we want to include the full set of molecular orbitals.
#
# In order to circumvent the combinatorial explosion, we can create an active space by classifying
# the molecular orbitals as doubly-occupied, active and external orbitals:
#
# * Doubly-occupied orbitals are always occupied by two electrons.
# * Active orbitals can be occupied by zero, one, or two electrons.
# * The external orbitals are never occupied.
#
# Within this approximation, a certain number of *active electrons* can populate the  *active
# orbitals* from which we can generate a finite-size space of Slater determinants.
#
# .. figure:: ../beginner/quantum_chemistry/sketch_active_space.png
#     :width: 50%
#     :align: center
#
# .. note::
#     The number of active **spin**-orbitals determines the number of qubits required
#     to perform quantum simulations of the electronic structure of the molecule.
#
# For the case of water molecule described using a minimal basis set, we have a total of ten
# electrons occupying the first five out of seven molecular orbitals in the HF reference state.
# Let's partition the HF orbitals to define an active space of four electrons in four active
# orbitals:

d_occ_indices, active_indices = qml.qchem.active_space(name, hf_data,
                                                       n_active_electrons=4,
                                                       n_active_orbitals=4)

##############################################################################
# Viewing the results:

print("List of doubly-occupied molecular orbitals: {:}".format(d_occ_indices))
print("List of active molecular orbitals: {:}".format(active_indices))
print("Number of qubits required for quantum simulation: {:}".format(2*len(active_indices)))

##############################################################################
# Notice that calling the :func:`~.active_space` function without specifying an active
# space, results in no doubly-occupied orbitals --- all molecular orbitals are considered to be active.

d_occ_indices, active_indices = qml.qchem.active_space(name, hf_data)
print("List of doubly-occupied molecular orbitals: {:}".format(d_occ_indices))
print("List of active molecular molecular orbitals: {:}".format(active_indices))
print("Number of qubits required for quantum simulation: {:}".format(2*len(active_indices)))

##############################################################################
# Building the Hamiltonian
# ------------------------
# Once we have an active space defined to generate the correlated wave function of the
# molecule, the next step is to build the second-quantized fermionic Hamiltonian,
#
# .. math::
#     H = \sum_{p,q} h_{pq} c_p^\dagger c_q + 1/2 \sum_{p,q,r,s} h_{pqrs} c_p^\dagger c_q^\dagger
#     c_r c_s
#
# and apply the `Jordan-Wigner
# <https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation>`__ or `Bravyi-Kitaev
# <https://arxiv.org/abs/1208.5986>`__ transformation [5] to map it to a linear combination of
# tensor products of Pauli operators
#
# .. math::
#     \sum_j h_j \prod_i \sigma_i^{(j)},
#
# where :math:`h_j` is a scalar coefficient and :math:`\sigma_i^{(j)}` denotes the :math:`j` th
# Pauli matrix :math:`X`, :math:`Y` or :math:`Z` acting on the :math:`i` th qubit.
# To perform the fermionic-to-qubit transformation of the electronic Hamiltonian, the one-body
# and two-body Coulomb matrix elements :math:`h_{pq}` and :math:`h_{pqrs}` [2] describing the
# fermionic Hamiltonian are retrieved from the previously generated file
# ``'./pyscf/sto-3g/water.hdf5'``.

qubit_hamiltonian = qml.qchem.decompose_hamiltonian(
	name,
	hf_data,
	mapping='jordan_wigner',
	docc_mo_indices=d_occ_indices,
    active_mo_indices=active_indices
)
print("Electronic Hamiltonian of the water molecule represented in the Pauli basis")
print(qubit_hamiltonian)

##############################################################################
# Finally, the the :func:`~.generate_hamiltonian`
# function to automate the construction of the electronic Hamiltonian using
# the functions described above.
#
# An example usage is shown below,
qubit_hamiltonian, n_qubits = qml.qchem.generate_hamiltonian(
	name,
	'h2o.xyz',
	charge,
	multiplicity,
	basis_set,
	qc_package='pyscf',
	n_active_electrons=4,
    n_active_orbitals=4,
    mapping='jordan_wigner'
)

print("Number of qubits required to perform quantum simulations: {:}".format(n_qubits))
print("Electronic Hamiltonian of the water molecule represented in the Pauli basis")
print(qubit_hamiltonian)

##############################################################################
# You have completed the tutorial! Now, select your favorite molecule and build its electronic
# Hamiltonian.
#
# To see how simple it is to implement the VQE algorithm to compute the ground-state energy of
# your molecule using PennyLane, take a look at the tutorial :doc:`/app/tutorial_vqe`.
#
# .. note::
#
#     If you have built your electronic Hamiltonian independently by using `OpenFermion
#     <https://github.com/quantumlib/OpenFermion>`__ tools, no problem! The
#     :func:`~.convert_hamiltonian` function converts the `OpenFermion
#     <https://github.com/quantumlib/OpenFermion>`__ QubitOperator to PennyLane observables.
#
# References
# ----------
#
# 1. Jorge Kohanoff. "Electronic structure calculations for solids and molecules: theory and
#    computational methods". (Cambridge University Press, 2006).
#
# 2. Frank Jensen. "Introduction to Computational Chemistry". (John wiley & sons,
#    2016).
#
# 3. Alberto Peruzzo, Jarrod McClean *et al.*. "A variational eigenvalue solver on a photonic
#    quantum processor". `Nature Communications 5, 4213 (2014).
#    <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# 4. Rolf Seeger, John Pople. "Self‐consistent molecular orbital methods. XVIII. Constraints and
#    stability in Hartree–Fock theory". `Journal of Chemical Physics 66,
#    3045 (1977). <https://aip.scitation.org/doi/abs/10.1063/1.434318>`__
#
# 5. Jacob T. Seeley, Martin J. Richard, Peter J. Love. "The Bravyi-Kitaev transformation for
#    quantum computation of electronic structure". `Journal of Chemical Physics 137, 224109 (2012).
#    <https://aip.scitation.org/doi/abs/10.1063/1.4768229>`__
