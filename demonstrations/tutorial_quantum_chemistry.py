r"""
Building molecular Hamiltonians with PennyLane
==============================================

.. meta::
    :property="og:description": Explore how PennyLane brings modern quantum computing tools
        to build the electronic Hamiltonian of molecules.
    :property="og:image": https://pennylane.ai/qml/_images/water_structure.png

.. related::
   tutorial_vqe Variational quantum eigensolver

*Author: PennyLane dev team. Last updated: 11 June 2021*

The ultimate goal of computational quantum chemistry is to unravel the 
quantum effects that determine the structure and properties
of molecules. In general, reaching this goal is challenging since the characteristic
energies of many chemical phenomena are typically a tiny fraction of the total
energy of the molecules [#jensenbook]_. In other words, numerical accuracy plays
a major role in quantum chemistry simulations.

Accurate molecular properties can be computed by having access
to the wave function describing the interacting electrons in a molecule [#kohanoff2006]_.
The **electronic** wave function :math:`\Psi(r)` satisfies the `Schrodinger
equation <https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation>`_

.. math::
    H_e \Psi(r) = E \Psi(r),

where :math:`H_e` and :math:`E` denote the electronic Hamiltonian and the
total energy of the molecule, respectively. However, solving
the equation above beyond the meanfield approximation to account for the fact that
the electrons can see each other (electronic correlations effects) poses an extremely
challenging computational task even for molecular systems with a few atoms [#jensenbook]_.

Quantum computers offer a promising avenue for major breakthroughs 
in quantum chemistry [#yudong2019]_. The first step to simulate a chemical system using
quantum algorithms is to build a representation of the molecular Hamiltonian
:math:`H_e` whose expectation value can be measured using the quantum device.

In this tutorial, we demonstrate how to use PennyLane's functionalities to build the
electronic Hamiltonian of a molecule. The first step is to read the nuclear coordinates of the
molecule. The atomic nuclei are treated as point particles whose coordinates
are fixed while we solve the electronic structure of the molecule [#BornOpp1927]_. Next,
we explain how to solve the `Hartree-Fock equations
<https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`_ by interfacing with 
classical quantum chemistry packages. Finally, we describe the functions used
to transform the fermionic Hamiltonian into a set of Pauli operators whose expectation
values can be measured in a quantum computer to calculate the total energy of the molecule. 

Let's get started!

Defining the molecular structure
--------------------------------
In this example, we construct the electronic Hamiltonian of the water molecule
consisting of one oxygen and two hydrogen atoms.

.. figure:: ../demonstrations/quantum_chemistry/water_structure.png
    :width: 50%
    :align: center

The structure of a molecule is defined by the symbols and the nuclear coordinates of
its atoms and can be specified using different `chemical file formats
<https://en.wikipedia.org/wiki/Chemical_file_format>`_. Within PennyLane, the molecular
structure is specified by providing a list with the atomic symbols and a one-dimensional
array with the nuclear coordinates in
`atomic units <https://en.wikipedia.org/wiki/Hartree_atomic_units>`_.
"""
import numpy as np

symbols = ['H', 'O', 'H']
coordinates = np.array([-0.0399, -0.0038,  0.0,
                         1.5780,  0.8540,  0.0,
                         2.7909, -0.5159,  0.0])

##############################################################################
# The :func:`~.pennylane_qchem.qchem.read_structure` can also be used to read the
# molecular geometry from a external file.

from pennylane import qchem

symbols, coordinates = qchem.read_structure('h2o.xyz')

##############################################################################
# The xyz format is supported out of the box. If
# `Open Babel <http://openbabel.org/wiki/Main_Page>`_ is installed, any
# format recognized by Open Babel is also supported by PennyLane.
#
# Solve the Hartree-Fock equations
# --------------------------------
# The molecule's electronic Hamiltonian is commonly represented using the
# second-quantization [#fetterbook]_ formalism as it is shown with more details in the
# next section. To that aim, a basis of **single-particle** states needs to be chosen.
# In quantum chemistry these states are the
# **`molecular orbitals <https://en.wikipedia.org/wiki/Molecular_orbital>`_**
# which are the wave functions of a single electron in the molecule.
#
# Molecular orbitals are typically represented as a linear combination of *atomic* orbitals
# [#jensenbook]_. The expansion coefficients in the atomic basis are calculated using the
# `Hartree-Fock method <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>_`.
# In this approximation each electron in the molecule is treated as an **independent**
# particle that moves under the influence of the nuclei Coulomb potential and a mean
# field generated by all other electrons [#pople1977]_. The optimized coefficients are precisely
# the HF molecular orbitals we need to build the second-quantized fermionic Hamiltonian.
#
# We can call the function :func:`~.pennylane_qchem.qchem.meanfield` to solve
# the Hartree-Fock calculation using either the quantum chemistry packages `PySCF
# <https://sunqm.github.io/pyscf/>`_ or `Psi4 <http://www.psicode.org/>`_. In this case
# we use PySCF, which is the default option.

hf_file = qchem.meanfield(symbols, coordinates, name='water')

##############################################################################
# Once the calculation is completed,the string variable ``hf_file`` returned by the
# function stores the absolute path to the hdf5-formatted file ``water`` with the
# Hartree-Fock electronic structure of the water molecule.

print(hf_file)

##############################################################################
# Building the Hamiltonian
# ------------------------
# In the second quantization formalism the electronic wave function of the molecule
# is represented in the occupation number basis. For :math:`M` *spin* molecular
# orbitals the elements of the basis are labeled as
# :math:`\vert n_0, n_1, \dots, n_{M-1} \rangle` where :math:`n_i = 0` or :math:`1`
# indicates the occupation of each orbital. In this representation, the electronic
# Hamiltonian is given by
#
# .. math::
#     H = \sum_{p,q} h_{pq} c_p^\dagger c_q +
#     \frac{1}{2} \sum_{p,q,r,s} h_{pqrs} c_p^\dagger c_q^\dagger c_r c_s,
#
# where :math:`c^\dagger` and :math:`c` are the electron creation
# and annihilation operators, respectively, and the coefficients
# :math:`h_{pq}` and :math:`h_{pqrs}` denote the one- and two-electron
# integrals [#ref_integrals]_ evaluated using the Hartree-Fock orbitals.
#
# We can use the states of :math:`M` qubits to encode any element
# of the occupation number basis
#
# .. math::
#     \vert n_0, n_1, \dots, n_{M-1} \rangle \rightarrow \vert q_0 \rangle
#     \otimes \vert q_1 \rangle \dots \otimes \vert q_{M-1} \rangle.
#
# This implies that we need to map the fermionic operators onto operators
# that act on the qubits. This can be done by using
# the `Jordan-Wigner <https://en.wikipedia.org/wiki/Jordan-Wigner_transformation>`_
# transformation [#seeley2012]_ which allows us to decompose the electronic Hamiltonian
# into a linear combination of the tensor product of Pauli operators
#
# .. math::
#     \sum_j C_j \prod_i \sigma_i^{(j)},
#
# where :math:`C_j` is a scalar coefficient and :math:`\sigma_i` represents the 
# the Pauli group :math:`\{ I, X, Y, Z \}`.
#
# We use the :func:`~.pennylane_qchem.qchem.decompose` function to perform
# the fermionic-to-qubit transformation of the Hamiltonian. This function
# uses `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ 
# functionalities to load the electron integrals from the previously generated file
# ``'./pyscf/sto-3g/water.hdf5'``, build the fermionic Hamiltonian and map it
# to the qubit representation.

qubit_hamiltonian = qchem.decompose(hf_file, mapping="jordan_wigner")
print("Qubit Hamiltonian of the water molecule")
print(qubit_hamiltonian)

##############################################################################
# The :func:`~.pennylane_qchem.qchem.molecular_hamiltonian`
# function can be used to automate the construction of the electronic Hamiltonian using
# the functions described above. An example usage is shown below:

H, qubits = qchem.molecular_hamiltonian(symbols, coordinates)

print("Number of qubits required to perform quantum simulations: {:}".format(qubits))
print("Qubit Hamiltonian of the water molecule")
print(H)

##############################################################################
# We have shown functionalities that allow users to easily build molecular Hamiltonians.
# However, if you have built your electronic Hamiltonian independently using
# `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools it can
# be readily converted to a PennyLane observable using the
# :func:`~.pennylane_qchem.qchem.convert_observable` function.
#
#
# Advanced features
# -----------------
# 
# The :func:`~.pennylane_qchem.qchem.meanfield` allows us to define additional keyword
# arguments to solve the Hartree-Fock equations of more complicated systems.
# The net charge of the molecule may be specified to simulate positively or negatively
# charged molecules. For a neutral system we choose

charge = 0

##############################################################################
# We can also specify the `spin multiplicity
# <https://en.wikipedia.org/wiki/Multiplicity_(chemistry)>`_ of the Hartree-Fock (HF) state.
# In the Hartree-Fock method the electronic wave function is approximated by a `Slater
# determinant <https://en.wikipedia.org/wiki/Slater_determinant>`_ consisting of
# the molecular orbitals occupied by the electrons in the molecule.
#
# For the water molecule, which contains ten electrons, the Slater
# determinant resulting from occupying the five lowest-energy molecular orbitals with two
# *paired* electrons in each orbital is said to be a closed-shell HF state with spin
# **multiplicity** one. Alternatively, if we define an occupation where the first four orbitals
# are doubly occupied and the next two are singly occupied by *unpaired* electrons, this is
# said to be an open-shell HF state with **multiplicity** three.
#
# |
#
# .. figure:: ../demonstrations/quantum_chemistry/hf_references.png
#     :width: 50%
#     :align: center
#
# |
#
# For a closed-shell state we have,

multiplicity = 1

##############################################################################
# As we mentioned above, molecular orbitals are represented as linear combination of atomic
# orbitals which are typically modeled as `Gaussian-type orbitals
# <https://en.wikipedia.org/wiki/Gaussian_orbital>`_. We can specify different types
# of `Gaussian atomic basis <https://www.basissetexchange.org/>`_. In this example we
# choose a `minimal basis set
# <https://en.wikipedia.org/wiki/Basis_set_(chemistry)#Minimal_basis_sets>`_.

basis_set = 'sto-3g'

##############################################################################
# PennyLane also allows us to define an *active space* to perform quantum simulations
# with a reduced number of qubits. But, what is an active space?
#
# In order to account for the electronic correlations in the molecule one needs
# to go beyond the Hartree-Fock approximation [#kohanoff2006]. In the exact limit,
# the electronic wave function is expanded as a linear combination
# of all possible Slater determinants obtained by exciting the electrons
# from the occupied to the unoccupied Hartree-Fock orbitals. This approach, typically
# referred to as the full configuration interaction method (FCI), becomes quickly
# intractable since the number of configurations increases combinatorially with the
# number of electrons and orbitals.
#
# This expansion can be truncated by classifying the molecular orbitals as
# core, active, and external orbitals:
#
# * Core orbitals are always occupied by two electrons.
# * Active orbitals can be occupied by zero, one, or two electrons.
# * The external orbitals are never occupied.
#
# Within this approximation, a certain number of *active electrons* can populate
# the *active orbitals*.
#
# .. figure:: ../demonstrations/quantum_chemistry/sketch_active_space.png
#     :width: 50%
#     :align: center
#
# .. note::
#     The number of *active spin-orbitals* determines the *number of qubits* required
#     to perform quantum simulations of the electronic structure of the molecule.
#
# In this example, for the water molecule we define an active space consisting of four electrons
# in four active orbitals. This is done using the :func:`~.pennylane_qchem.qchem.active_space`
# function:

electrons = 10
orbitals = 7
core, active = qchem.active_space(electrons, orbitals, active_electrons=4, active_orbitals=4)

##############################################################################
# Viewing the results:

print("List of core orbitals: {:}".format(core))
print("List of active orbitals: {:}".format(active))
print("Number of qubits required for quantum simulation: {:}".format(2*len(active)))

##############################################################################
# Finally, we use the :func:`~.pennylane_qchem.qchem.molecular_hamiltonian` function to
# build the **approximated** Hamiltonian of the water molecule:

H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, name='water', charge=charge,
    mult=multiplicity,
    basis=basis_set,
    active_electrons=4,
    active_orbitals=4,
    mapping='jordan_wigner'
)

print("Number of qubits required to perform quantum simulations: {:}".format(qubits))
print("Approximated Hamiltonian of the water molecule represented in the Pauli basis")
print(H)

##############################################################################
# You have completed the tutorial! Now, select your favorite molecule and build its electronic
# Hamiltonian.
#
# To see how simple it is to implement the VQE algorithm to compute the ground-state energy of
# your molecule using PennyLane, take a look at the tutorial :doc:`tutorial_vqe`.
#
# References
# ----------
#
# .. [#jensenbook]
#
#     Frank Jensen. "Introduction to Computational Chemistry". (John Wiley & Sons, 2016).
#
# .. [#kohanoff2006]
#
#     Jorge Kohanoff. "Electronic structure calculations for solids and molecules: theory and
#     computational methods". (Cambridge University Press, 2006).
#
# .. [#yudong2019]
#
#     Yudong Cao, Jonathan Romero, *et al.*, "Quantum Chemistry in the Age of Quantum Computing".
#     `Chem. Rev. 2019, 119, 19, 10856-10915.
#     <https://pubs.acs.org/doi/10.1021/acs.chemrev.8b00803>`__
#
# .. [#BornOpp1927]
#
#     M. Born, J.R. Oppenheimer, "Quantum Theory of the Molecules".
#     `Annalen der Physik 84, 457-484 (1927)
#     <https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.19273892002>`_.
#
# .. [#fetterbook]
#
#     A. Fetter, J. D. Walecka, "Quantum theory of many-particle systems".
#     Courier Corporation, 2012.
#
# .. [#pople1977]
#
#     Rolf Seeger, John Pople. "Self‐consistent molecular orbital methods. XVIII. Constraints and
#     stability in Hartree–Fock theory". `Journal of Chemical Physics 66,
#     3045 (1977). <https://aip.scitation.org/doi/abs/10.1063/1.434318>`__
#
# .. [#ref_integrals]
#
#     J.T. Fermann, E.F. Valeev, "Fundamentals of Molecular Integrals Evaluation".
#     ` arXiv:2007.12057 <https://arxiv.org/abs/2007.12057>`_
#
# .. [#seeley2012]
#
#     Jacob T. Seeley, Martin J. Richard, Peter J. Love. "The Bravyi-Kitaev transformation for
#     quantum computation of electronic structure". `Journal of Chemical Physics 137, 224109 (2012).
#     <https://aip.scitation.org/doi/abs/10.1063/1.4768229>`__
