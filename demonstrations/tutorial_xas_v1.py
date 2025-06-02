"""
X-ray Absorption Spectroscopy Simulation in the Time-Domain 
====================================================
"""
######################################################################
# X-ray absorption spectroscopy (XAS) is a technique for understanding unwanted structures in battery
# materials. This demo will show you how to implement an algorithm developed in the paper "Fast
# simulations of X-ray absorption spectroscopy for battery materials on a quantum computer"
# [Fomichev:2025]. XAS was identified as a potential application for early fault-tolerant quantum
# computers. X-rays are high energy, and so are short wavelength. Therefore, they probe a very
# localized grouping of atoms in a materials. We only need to simulate a small number of molecular
# orbitals to estimate the energy of the orbitals, and the ones accessible by X-rays. However, there
# is a large amount of correlations between these molecular states. This is the sweet spot between not
# having to require too many qubits, but having enough correlations that make the calculation
# intractible on a quantum computer. The ground state is computable, but the highly correlated
# high-energy states are very difficult to compute in a quantum chemistry classical computation


######################################################################
# In this demo we will implement a simplified version of the algorithm as it appears in the paper - we
# will not be applying all of the optimizations present. We will show how to determine and prepare a
# group state, how determine the states that result from acting on the group state by the dipole
# operator (representing the electromagnetic field from the X-ray) and how to code the time-domain
# analysis circuit which can determine the spectrum.
#
# We will be using concepts that were introduced in other PennyLane demos, such as utilizing pyscf
# with PennyLane, initial state preparation of molecules, and building compressed double-factorized
# Hamiltonians. If you haven't checked out those demos yet, it might be best to do so and then come
# back here.


######################################################################
# Why simulate X-ray absorption spectroscopy?
# -------------------------------------------


######################################################################
# - method for studying local electronic structure
# - can directly probe local structure by exciting tightly bound core electrons
# - determining battery degredation relevant mechanisms, such as oxidation states, from an observed
#   spectrum is difficult without simulations
# - the way one could do this is through spectral fingerprinting
# - each oxidized state would have some spectrum, that one would determine through simulation
# - this could be repeated for all possible oxidized states
# - the observed spectrum could then be matched to combinations of these single cluster spectra
# - this 'finger printing' method could allow one to determine the composition of oxidized states in
#   the material


######################################################################
# **insert figure demonstrating fingerprinting** observed spectrum, decomposition into single cluster
# spectra, which come from simulations from a quantum computer


######################################################################
# This method is particularly difficult for classical computers when strongly-correlated transition
# metals are present, such as those typically in lithium-excess materials which are battery cathod
# candidates.


######################################################################
# Algorithm
# ---------


######################################################################
# Absorption cross-section
# ~~~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# In XAS experiments, the spectrum observed is a measure of the absorption cross section as a function
# of the frequency of the incident X-rays :math:`\sigma_A(\omega)`. For our situation, the electrons
# in the molecular cluster start in a ground molecular state :math:`|I\rangle` with energy
# :math:`E_I`, and will be coupled to an excited state :math:`|F\rangle` with energy :math:`E_F`
# through the action of the dipole operator :math:`\hat m_rho`, which represents the effect of the
# radiative field, where :math:`\rho` is any of the Cartesian directions :math:`\{x,y,z\}`.
#
# Using Fermi's golden rule, the absorption cross-section is given by
#
# .. math::  \sigma_A(\omega) = \frac{4 \pi}{3 \hbar c} \omega \sum_{F \neq I}\sum_{\rho=x,y,z} \frac{|\langle F|\hat m_\rho|I \rangle|^2 \eta}{((E_F - E_I)-\omega)^2 + \eta^2}\,, 
#
# where :math:`c` is the speed of light, :math:`\hbar` is Plank's constant, and :math:`\eta` is the
# line broadening which represents the experimental resolution of the spectroscopy, and is typically
# around :math:`1` eV.


######################################################################
# Core-valence separation approximation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# We can restrict the range of frequencies, and consequently the range of final states, by only
# considering final states for which *core* electrons are excited, i.e.'there exists a hole in the
# core orbitals. These are known as core-excited states, and lie significally above the
# valance-excited states in energy. Typically the frequency range is focused on a target atom in a
# molecular cluster, and also near a transition energy, such as targetting core :math:`1s` electrons.
# We will also neglect relativistic corrections, and focus on frequencies for which the dipole
# approximation is valid, which is the assumption that the wavelength of the radiation is large
# compared to the extent of the electronic wavefunction.
#
# Atomic species and oxidations states will determine the energy difference between states with
# different principle quantum numbers, and this difference will show as a peak in spectroscopy, known
# as the *absorption edge*. Focusing spctroscopy near this edge for :math:`1s` to :math:`2p` is called
# the :math:`K`-edge, and in general X-ray absorption near-edge spectroscopy is known as XANES. We
# will focus on simulating spectroscopy in this XANES regime.


######################################################################
# **insert figure which is a single particle energy diagram** should show XANES excitation from core
# to excited compared to UV/vis light excitations of valence electrons to excited


######################################################################
# Time-domain determination of the cross section
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# We will use a mathematical trick of a frequency-domain Green's function to link the absorption cross
# section we want to estimate to the time-propagation of our initial state through a Fourier
# transform. The expectation value of our initial state :math:`m_\rho|I\rangle` propagated various
# amounts in time by the molecular hamiltonian :math:`H` will give us a time-domain Green's function,
# the Fourier transform of which is the frequency-domain Green's function, and that is directly
# related to the absorption cross section.
#
# **Insert intuition for using a Green's function**
#
# **Insert derivation of relation between cross section and Green's function, i.e.'expectation value
# of time evolution operator for various times**


######################################################################
# Below is a step-by-step process of the algorithm: - determine our initial state :math:`|I\rangle` -
# determine how the dipole operator acts on that state to obtain :math:`\hat m_\rho|I\rangle` -
# determine how best to efficiently time propagate that state with the molecular Hamiltonian - run the
# time propagation for various times, and from the expectation value determine the time-domain Green's
# function - Fourier transform (classically) the Green's function to obtain it in the freuency domain,
# which gives us the absorption cross section as a function of frequency


######################################################################
# **insert figure that is a block-diagram outline of this procedure** follows the steps above,
# includes some loops showing repeated shots and various time selections, classical computer computing
# the Fourier transform, final spectrum


######################################################################
# Implementation
# --------------


######################################################################
# Let's look at how to implement these steps in PennyLane. We will make extensive use of the
# ``qml.qchem`` module, as well as modules from ``pyscf``.
#
# For this demo, we are going to use the very simple :math:`H_2` molecule.


######################################################################
# Initial state preparation
# ~~~~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# If you haven't, check out the demo 'Initial state preparation for quantum chemistry'.
#
# We start by creating our molecule object using the Gaussian type orbitals module (``pyscf.gto``),
# and obtaining the reduced Hartree-Fock molecular orbitals with the self-consistent field methods
# (``pyscf.scf``).


######################################################################
# Next we will use the multiconfigurational self-consistent field methods (``pyscf.mcscf``) to solve
# for the expansion of the intitial wavefunction as a linear combination of Slater determinants.
# Running the configuration interaction (CI) method returns the wavefunction as as vector. We will
# filter out small values in the wavefunction.


######################################################################
# Dipole operator action
# ~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# The dipole operator :math:`\hat m_\rho = -q \cdot \hat \rho` will only affect the spatial component
# of the wavefunction ' it does not care about spin. First, we determine the action of the dipole
# operator in the atomic orbital basis, then we will transform to the molecular orbitals. We use two
# resolutions of identity to introduce the atomic orbital wavefunctions :math:`|k\rangle`
#
# .. math::  \hat m_\rho = -q \cdot \hat \rho = -q \sum_{ij} |i\rangle \langle i| \hat \rho |j\rangle \langle j|\,. 
#
# Then, we simplify by noticing that :math:`|i\rangle \langle j| = c_i^\dagger c_j` where
# :math:`c_k^{(\dagger)}` is the annihilation (creation) operator for orbital :math:`k`. That gives us
#
# .. math::  \hat m_\rho = -q \sum_{ij} \langle i|\hat \rho |j\rangle c_i^\dagger c_j\,. 
#
# To calculate the matrix elements :math:`\langle i|\hat \rho |j\rangle`, we again insert sums over
# basis states, this time in the position basis, to this as an integral of orbitals
#
# $$ `\langle `i\|:raw-latex:`\hat `:raw-latex:`\rho `\|j:raw-latex:`\rangle `=
# `\int `d^3r d^3r'
# `\langle `i\|:raw-latex:`\rho`:raw-latex:`\rangle `:raw-latex:`\langle `:raw-latex:`\rho `\|:raw-latex:`\hat `:raw-latex:`\rho `\|
# `\rho `:raw-latex:`\rangle `:raw-latex:`\langle `:raw-latex:`\rho `\| j
# `\rangle `= :raw-latex:`\int `d^3r d^3r' :raw-latex:`\phi`\_i(:raw-latex:`\rho`)
# `\rho `:raw-latex:`\phi`\_j^*(:raw-latex:`\rho`) \,, $$
#
# where :math:`\phi_k(\rho)` are the atomic orbital spatial wavefunctions. These matrix elements can
# be computed in ``pyscf`` using the the ``intor`` method of the ``mol`` object with argument
# ``'int1e_r_cart'`` to specify a one-electron integral with a position :math:`r` factor. Keyword
# argument ``comp=3`` will give us all three components.


######################################################################
# We can then transform to the molecular orbital space using ``np.einsum`` and the ``mo_coeff`` method
# of ``hf``.


######################################################################
# What's left is to code the action of the ladder operators :math:`c_i^\dagger c_j`.


######################################################################
# Finally, we can convert our vector into a PennyLane state vector using
# ``qchem.convert._wfdict_to_statevector``, so that it is ready to be initialzed in a circuit.


######################################################################
# Let's prepare the circuit that will initialize our qubit register with this state


######################################################################
# Molecular Hamiltonian
# ~~~~~~~~~~~~~~~~~~~~~


######################################################################
# Go read the demo 'How to build compressed double-factorized Hamiltonians' if you haven't, because
# that is exactly what we are going to do!
#
# Compressed double-factorized (CDF) molecular Hamiltonians will be perfect for our application ' time
# propagating our state with an electronic Hamiltonian. We will approximate the evolution operator
# :math:`e^{-iHt}` with a Trotter product formula, where the factorized Hamiltonian will allow much
# faster simulation.
#
# Our electronic Hamiltonian is
#
# .. math::  H = E + \sum_{p,q=1}^N \sum_{\gamma\in\{\uparrow,\downarrow\}} (p|\kappa|q) a^\dagger_{p\gamma}a_{q\gamma} + \frac12 \sum_{p,q,r,s=1}^N\sum_{\gamma,\beta\in\{\uparrow,\downarrow\}} (pq|rs) a^\dagger_{p\gamma}a_{q\gamma} a^\dagger_{r\beta}a_{s\beta} \,, 
#
# where :math:`a^{(\dagger)}_{p\gamma}` is the annihilation (creation) operator for a spatial orbital
# :math:`p` and spin :math:`\gamma`, :math:`E` is the enrgy offset, :math:`N` is the number of spatial
# orbitals, and :math:`(p|\kappa|q)` and :math:`(pq|rs)` are the one- and two-electron integrals,
# respectively.
#
# Luckily, the one- and two- eletron integrals can be computed using modules in ``pyscf``. The core
# constant can also be obtained using the method ``energy_nuc()`` of the ``mol`` object.


######################################################################
# We can apply CDF to the two-electron integrals using ``qml``\ 's ``qchem.factorize`` function, with
# ``compressed=True``.


######################################################################
# **Explain one-body extra term as the one-qubit Pauli-Z terms.**
#
# Finally, we add the one-body correction to the one-electron integrals, and use ``np.linalg.eigh`` to
# diagonalize them into the matrix :math:`Z^{(0)}` and obtain the rotation matrices :math:`U^{(0)}`.


######################################################################
# **I think I need to explain the Jordan-Wigner mapping here explicitly, showing the one-body extra,
# and the new rotation matrices and how they can be implemented with Thoulesses' theorem.**
#
# **Explain simplification using Thouless's theorem (here or in earlier section)** Specifically how to
# go form what looks like two U rotations to just one when using the ``qml.BasisRotation``.


######################################################################
# Time-propagation circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~


######################################################################
# The main work of our algorithm will be to apply our Hamiltonian terms as a trotter product, and
# measure the expectation value of that time evolution for various times.
#
# Let's start by writing functions that implement the time evolution for each Hamiltonian term, which
# will be called by our trotter circuit.
#
# One thing to track throughout this implementation is the global phase accrued throughout the time
# evolution.
#
# Starting with the :math:`U` operator rotations, we can write a function that uses
# ``qml.BasisRotation`` to apply the unitary transform we want. We apply this to both spin sections of
# the register.


######################################################################
# Next we write a function to perform the :math:`Z` rotations. To simplify the Trotter function, we
# condition the action in the function on whether it's acting for a one-body or a two-body term.


######################################################################
# Let's define our Trotter step. The function will implement :math:`U` rotations and :math:`Z`
# rotations. By tracking the last :math:`U` rotation used, we can implement two consequtive rotations
# at once as :math:`V^{(\ell)} = U^{(\ell-1)}(U^{(\ell)})^T`, halving the number of rotations required
# per Trotter step.
#
# We will write a function ``LieTrotter`` which will apply the rotations for the one- and two- body
# terms in one order, but can also reverse the order. This can save another rotation step when we
# implement two consecutive Trotter steps in the second-order Trotter scheme.


######################################################################
# Finally, our function ``Strang`` implements the second-order Trotter step, and the function
# ``trotter_circuit`` returns the Trotter step circuit, implementing ``Strang`` and a ``StatePrep`` to
# prepare the register in the previous quantum state.


######################################################################
# Measurement
# -----------


######################################################################
# To measure the expectation value of the time-propagated state, we use a Hadamard test.


######################################################################
# Run Simulation
# ~~~~~~~~~~~~~~


######################################################################
# Let's discuss our choice of parameters when running this simulation. **Discuss :math:`\eta`,
# :math:`j_\mathrm{max}`, the total number of shots :math:`S`, the Hamiltonian norm :math:`||H||`, the
# grid of frequencies and the time step :math:`\tau`**.


######################################################################
# Finally, we can run the simulation, and calculate the spectrum from the measurement results.


######################################################################
# Plot Results
# ~~~~~~~~~~~~


######################################################################
# Plotting the frequency signal, and the spectrum, we see


######################################################################
# Further Optimizations
# ~~~~~~~~~~~~~~~~~~~~~


######################################################################
# There are more optimizations mentioned in the paper that were not implemented here. Below is a list
# of further optimizations: - Randomized Trotter steps - BLISS - Distribution sampling - Double
# measurement


######################################################################
# Conclusion
# ----------
#
# In this tutorial, we have implemented a simplified version of the algorithm as presented in
# [Fomichev:2025]. The algorithm represents a culmination of many optimizations for time-evolving an
# eletronic Hamiltonian. We've also discussed how XAS is a promising candidate for early
# fault-tolerant quantum computers due to its low qubit overhead but high amount of correlations in
# the state space.
# 