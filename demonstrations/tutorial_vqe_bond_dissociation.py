r"""
Modeling chemical reactions using VQE
==============================

.. meta::
    :property="og:description": Construct Potential energy surface for chemical reactions using 
    the variational quantum eigensolver algorithm in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/pes_h2.png

.. related::

   tutorial_vqe
   tutorial_vqe_parallel VQE with parallel QPUs

*Author: PennyLane dev team. Last updated: 29 Apr 2021.*

The term chemical reaction is another name for the transformation of molecules -- breaking and 
forming of bonds -- and this transformation comes with an energy cost. The energy cost determines 
the feasibility of a particular transformation amongst many different alternate possibilities. 
Computational chemistry offers several theoretical methods for determining this energy cost
precisely and a window to predicting the thermodynamic and kinetics aspects of any 
chemical reaction. In this tutorial, you will learn how to use PennyLane to simulate chemical 
reactions. 

In a previous tutorial on :doc:`Variational Quantum Eigensolver (VQE) </demos/tutorial_vqe>`, 
we looked at how it can be used to compute molecular energies. [#peruzzo2014]_ 
Here, we show how VQE can be used to construct potential energy surfaces (PESs) for any general 
molecular transformation. This lends the way to the calculation of important quantities such as
activation energy barriers, reaction energies, and reaction rates. As illustrative examples, we
use tools implemented in PennyLane to study diatomic bond dissociation and reactions involving 
the exchange of hydrogen atoms. Let's get started! 



.. figure:: /demonstrations/vqe_bond_dissociation/h2_pes_pictorial.png
   :width: 50%
   :align: center
   
   An illustration of the potential energy surface of bond dissociation for the hydrogen molecule. 
   On the :math:`y`-axis is the total molecular energy and :math:`x`-axis is the internuclear bond
   distance. By looking at this curve, we can estimate the :math:`H-H` bond length and the energy 
   required to break the :math:`H-H` bond.   


##############################################################################


Potential Energy Surfaces: Hills to die and be reborn 
---------------------------------------------------------------------

Potential energy surfaces (PES) are, in simple words, energy landscapes on which any chemical 
reaction or a general molecular transformation occurs. But what is it? The concept originates
with the fact that the nuclies are much  heavier than electrons, better known as the 
Born-Oppenheimer approximation, and that we can solve for the electronic wavefunction with 
nucleis clamped to their respective positions. This results in the separation of nuclear and 
electronic parts of the Schrodinger equation and we then only solve the electronic part of
the problem

.. math:: H_{el}|\Psi \rangle =  E_{el}|\Psi\rangle.

Thus arises the concept of the electronic energy of the molecule
as a function of interatomic coordinates and angles. The potential energy surface of a 
molecule is a 
:math:`n`-dimensional plot of the energy with the respect to the degrees of freedom. It gives us a 
visual tool to understand chemical reactions where stable molecules are the local minimas in 
the valleys and transition states the *hill peaks* to climb.

To summarize, to build the potential energy surface, we solve the electronic Schrodinger
equation for a given fixed position of the nuclei, and subsequently move the nuclei in incremental steps
to obtain the energy at other configurations of the nuclei. The obtained set of energies are then plotted 
against nuclear positions.

We begin with the simplest of molecule: :math:`H_2`. 
The formation (or breaking) of the :math:`H-H` bond is also the simplest
of all reactions. 

.. math:: H_2 \rightarrow H + H  

Using a minimal basis set (STO-3G), this system can be described by :math:`2` electrons in :math:`4` spin-orbitals. 
When mapped to a qubit representation, we have a total of four qubits. As discussed in the 
previous tutorial, the states involved are the Hartree-Fock (HF) ground state, :math:`|1100\rangle`,
the singly-excited states :math:`|0110\rangle`, :math:`|1001\rangle`, and the doubly-excited 
state :math:`|0011\rangle`. These are the only states out of :math:`2^4 (=16)` possible 
states that matter for this problem and are obtained by single and double excitations
out of the HF state. Below we show how to set up the problem to generate the PES for such a 
reaction. 

The first step is to import the required libraries and packages:
"""

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time

##############################################################################
# The second step is to specify the geometry and charge of the molecule,
# and the spin multiplicity of the electronic configuration. To construct the potential energy
# surface, we need to vary the geometry. So, we keep an :math:`H` atom fixed at origin and vary the
# :math:`x`-coordinate of the other :math:`H` atom such that the bond distance varies from
# :math:`1.0` to :math:`4.0` Bohrs in steps of :math:`0.25` Bohr.

charge = 0
multiplicity = 1
basis_set = "sto-3g"

electrons = 2

active_electrons = 2
active_orbitals = 2

vqe_energy = []

# set up a loop to change internuclear distance
for r_HH in np.arange(0.5, 4.0, 0.1):

    symbols, coordinates = (["H", "H"], np.array([0.0, 0.0, 0.0, 0.0, 0.0, r_HH]))

    # Do a meanfield calculation -> Define a fermionic Hamiltonian -> turn into a qubit Hamiltonian
    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        mult=multiplicity,
        basis=basis_set,
    )

    print("Number of qubits = ", qubits)
    print("Hamiltonian is ", H)

    ##############################################################################
    # Now we need to build the circuit for a general molecular system. We begin by preparing the
    # qubit version of the HF state, :math:`|1100\rangle`. We then identify and add all possible
    # single and double excitations.

    # get all the singles and doubles excitations
    singles, doubles = qchem.excitations(active_electrons, active_orbitals * 2)
    print("Single excitations", singles)
    print("Double excitations", doubles)

    # define the circuit
    def circuit(params, wires):
        # Prepare the HF state |1100> by flipping the qubits 0 and 1
        qml.PauliX(0)
        qml.PauliX(1)
        # Add double excitation
        qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
        # Add single excitations
        qml.SingleExcitation(params[1], wires=[0, 2])
        qml.SingleExcitation(params[2], wires=[1, 3])

    ##############################################################################
    # From here on, we can use optimizers in PennyLane.
    # PennyLane contains the :class:`~.ExpvalCost` class,
    # that we use to obtain the cost function central to the idea of variational optimization
    # of parameters in VQE algorithm. We define the device which is a classical qubit
    # simulator here,
    # a cost function which calculates the expectation value of Hamiltonian operator for the
    # given trial wavefunction and also the gradient descent optimizer that is used to optimize
    # the gate parameters:

    dev = qml.device("default.qubit", wires=qubits)
    cost_fn = qml.ExpvalCost(circuit, H, dev)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    ##############################################################################
    # A related question is what are gate parameters that we seek to optimize?
    # These could be thought of as rotation variables in the gates used which
    # can then be converted into determinant coefficients in the expansion
    # of the exact wavefunction.

    # define and initialize the gate parameters
    params = np.zeros(3)
    dcircuit = qml.grad(cost_fn, argnum=0)
    dcircuit(params)

    ##############################################################################
    # We then begin the VQE iteration to optimize gate parameters.
    # The energy-based convergence criteria is chosen to be :math:`\sim 1E^{-6}`
    # which could be made stricter.

    prev_energy = 0.0

    for n in range(40):

        t1 = time.time()

        params, energy = opt.step_and_cost(cost_fn, params)

        t2 = time.time()

        print("Iteration = {:},  E = {:.8f} Ha, t = {:.2f} S".format(n, energy, t2 - t1))

        # define your convergence criteria, we choose modest value of 1E-6 Ha
        if np.abs(energy - prev_energy) < 1e-6:
            break

        prev_energy = energy

    print("At bond distance \n", r_HH)
    print("The VQE energy is", energy)

    vqe_energy.append(energy)

##############################################################################
# We have the calculated the molecular energy as a function of H-H bond distance, let us plot it.

# Energy as a function of internuclear distance

r = np.arange(0.5, 4.0, 0.1)

fig, ax = plt.subplots()
ax.plot(r, vqe_energy, label="VQE(S+D)")

ax.set(
    xlabel="H-H distance (in Bohr)",
    ylabel="Total energy (in Hartree)",
    title="PES for H$_2$ dissociation",
)
ax.grid()
ax.legend()

plt.show()


##############################################################################
# This is a simple PES (or more appropriately PEC, potential energy curve) for
# the dissociation of hydrogen molecule into two hydrogen atoms. It gives an
# estimate of :math:`H-H` bond distance
# to be :math:`\sim 1.4` Bohrs and the :math:`H-H` bond dissociation energy
# (the difference in energy at equilibrium and energy at dissociation limit)
# as :math:`0.194` Hartrees (:math:`121.8` Kcal/mol). Could these estimates be improved?
# Yes, by using bigger basis sets or using explicitly correlated methods(F12) and
# extrapolating to the complete basis set (CBS) limit. [#motta2020]_
# Now let's move on to something slightly more complicated.
#


##############################################################################
# Hydrogen Exchange Reaction
# -----------------------------
#
# We will try to construct the PES for a simple hydrogen exchange reaction
#
# .. math:: H_2 + H \rightarrow H + H_2
#
# This reaction has a barrier though, the transition state, which it has to cross
# for the exchange of :math:`H` atom to be complete. In a minimal basis like STO-3G,
# this system consists of :math:`3` electrons in :math:`6` spin molecular orbitals.
# This means it is a :math:`6` qubit problem and the ground state (HF state) in
# occupation number representation is :math:`|111000\rangle`.
#
# .. figure:: /demonstrations/vqe_bond_dissociation/h3_mol_movie.gif
#   :width: 50%
#   :align: center
#
# Again, we need to define the molecular parameters.

# Molecular parameters

name = "h3"
basis_set = "sto-3g"

electrons = 3
charge = 0
spin = 1
multiplicity = 2

active_electrons = 3
active_orbitals = 3

##############################################################################
# Then we setup the PES loop, incrementing the :math:`H(1)-H(2)` distance from :math:`1.0`
# to :math:`3.0` Bohrs in steps of :math:`0.1` Bohr.

vqe_energy = []

for r_HH in np.arange(1.0, 3.0, 0.1):

    symbols, coordinates = (
        ["H", "H", "H"],
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, r_HH, 0.0, 0.0, 4.0]),
    )

    # Hamiltonian
    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        mult=multiplicity,
        basis=basis_set,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    # get all the singles and doubles excitations
    singles, doubles = qchem.excitations(active_electrons, active_orbitals * 2)

    def circuit(params, wires):
        # Prepare the HF state |111000> by flipping the qubits 0, 1 and 2
        qml.PauliX(0)
        qml.PauliX(1)
        qml.PauliX(2)
        # All possible double excitations
        for i in range(0, len(doubles)):
            qml.DoubleExcitation(params[i], wires=doubles[i])

        # All single excitations too
        for j in range(0, len(singles)):
            qml.SingleExcitation(params[j + len(doubles)], wires=singles[j])

    #   Now we define the device and initialize the gate parameters.
    dev = qml.device("default.qubit", wires=qubits)

    cost_fn = qml.ExpvalCost(circuit, H, dev)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    # total length of parameters is generally the total no. of determinants considered
    len_params = len(singles) + len(doubles)
    params = np.zeros(len_params)

    ##############################################################################
    # Then we evaluate the costfunction and use the gradient descent algorithm in an iterative
    # optimization of the gate parameters.

    dcircuit = qml.grad(cost_fn, argnum=0)

    dcircuit(params)

    prev_energy = 0.0

    for n in range(40):

        t1 = time.time()

        params, energy = opt.step_and_cost(cost_fn, params)

        t2 = time.time()

        print("Iteration = {:},  E = {:.8f} Ha, t = {:.2f} S".format(n, energy, t2 - t1))

        if np.abs(energy - prev_energy) < 10e-6:
            break

        prev_energy = energy

    ##############################################################################
    # Finally at each point of the PEC, we could print the total VQE energy

    print("At bond distance \n", r_HH)
    print("The VQE energy is", energy)

    vqe_energy.append(energy)

#
##############################################################################
#
# Then we plot the energy as a function of distance between atoms :math:`1` and :math:`2`
# which is also the reaction coordinate and thus, we have the potential energy curve for
# this reaction.

r = np.arange(1.0, 3.0, 0.1)

fig, ax = plt.subplots()
ax.plot(r, vqe_energy, label="VQE(S+D)")

ax.set(
    xlabel="Distance (H-H, in Bohr)",
    ylabel="Total energy (in Hartree)",
    title="PES for H-H + H -> H + H-H reaction",
)
ax.grid()

ax.legend()

plt.show()
#
#
##############################################################################
# Activation energy barriers and reaction rates
# --------------------------------------------
# The utility of potential energy surfaces lies in estimating the
# geometric configurations of key reactants, intermediates, transition states
# and products, as well as the reaction energies and activation energy barriers.
# To be specific about the above PEC, we would like our method to provide
# a good estimate of the energies of the reactants (minima :math:`1`), products (minima :math:`2`)
# and the transition state (maxima). VQE(S+D) reproduces the exact result in the small
# basis (STO-3G). The plot below compares the performance of different methods.
# The PEC obtained from the quantum algorithm overlaps with
# the quantum chemistry methods, CCSD and CISD.
# Another VQE based optimization but restricted to a simpler ansatz is added too.
# DOCI stands for Doubly occupied CI where only pure pair excitations are allowed
# but we have also kept all single excitations.  As we see, VQE(DOCI) does not
# get the energetics of reactants, products and transition states accurately
# and hence is not an ideal method
# for this problem but could become a good starting point for some more difficult problems.
# In a future tutorial, we would show how to build a range of trial wavefunction ansatz
# using tools in Pennylane used here such as qml.DoubleExcitation and qml.SingleExcitation.
#
# The activation energy barrier is defined as the difference between the
# energy of the reactant complex
# and the energy of the
# transition state (:math:`H--H--H`).
#
# .. math:: E_{Activation Barrier} = E_{TS} - E_{Reactant}
#
# In this case, VQE(S+D) reproduces the exact theoretical result in the
# minimal basis. The activation energy barrier is given by
#
# .. math:: E_{Activation Barrier} = 0.0274 Ha = 17.24 Kcal/mol
#
# Though this is the best theoretical estimate in this small basis,
# this is not the *best* theoretical estimate. We would need to do this calculation
# in larger basis, triple and quadruple zeta basis or higher, to reach basis set
# convergence and this would significantly increase the number of qubits required.
# This is a current limitation that would go away with increasing number of logical qubits
# that would hopefully become available in future quantum computers.
#
# The reaction rate constant (k) has an exponential dependence on the activation energy barrier:
#
# .. math:: k = Ae^{-{E_{Activation Barrier}}/RT}
#
# So, in principle, if we know the constant (A) we could calculate the rate of the reaction
# which depends on the rate constant, the concentration of the reactants and the order of the
# reaction.


##############################################################################
# .. figure:: /demonstrations/vqe_bond_dissociation/h3_comparison.png
#     :width: 50%
#     :align: center
#
#
#
#
#
#

##############################################################################
# A model multireference problem: :math:`Be + H_{2} \rightarrow BeH_{2}`
# -------------------------------------------------------------------------------------------
#
#
# A symmetric approach (:math:`C_{2v}`) of :math:`H_2` to :math:`Be` atom to form :math:`BeH_2`
# constitutes a multireference problem. [#purvis1983]_ It needs two different
# HF Slater determinants to qualitatively describe the full potential energy suface
# for the transformation. This is to say that one Slater Determinant is a good HF reference
# for one half of the PES while another determinant is good reference for rest of PES.
#
# Here, we construct the potential energy surface for this reaction and see how
# classical and quantum computing approaches built on single HF reference perform.
# Once we have solved the mean-field problem, we obtain the molecular orbitals
# (:math:`1a,2a,3a,1b ...`) which are then occupied to obtain two principal configurations
# :math:`1a^{2} 2a^{2} 3a^{2}` and :math:`1a^{2} 2a^{2} 1b^{2}`.
#
#
# .. figure:: /demonstrations/vqe_bond_dissociation/beh2_movie.gif
#     :width: 50%
#     :align: center
#
#
# To figure out the reaction coordinate or the trajectory of
# approach of :math:`H_2` to Beryllium atom, we refer to the work by
# Coe et al. [#coe2012]_
# We fix the Beryllium atom at the origin and the coordinates for hydrogen atoms are give by,
# in Bohr, the coordinates :math:`(x, y, 0)` and :math:`(x, −y, 0)` where :math:`y = 2.54 − 0.46x`
# and :math:`x \in [1, 4]`.
# The generation of PES then is straightforward and follows from our previous examples.
# For the sake of saving computational cost, we try a smaller active space of a total of
# :math:`6` spin MOs with core electrons frozen.

# Molecular parameters
name = "beh2"
basis_set = "sto-3g"

electrons = 6
charge = 0
spin = 1
multiplicity = 1

active_electrons = 4
# choosing a smaller active space - 6 spin MOs
active_orbitals = 3
vqe_energy = []


name = "beh2"

for reac_coord in np.arange(1.0, 4.0, 0.25):

    x = reac_coord
    y = np.subtract(2.54, np.multiply(0.46, x))

    symbols, coordinates = (["Be", "H", "H"], np.array([0.0, 0.0, 0.0, x, y, 0.0, x, -y, 0.0]))

    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        mult=multiplicity,
        basis=basis_set,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    # get all the singles and doubles excitations
    singles, doubles = qchem.excitations(active_electrons, active_orbitals * 2)

    def circuit(params, wires):
        # Prepare the HF state |111100> by flipping the qubits 0, 1, 2 and 3
        qml.PauliX(0)
        qml.PauliX(1)
        qml.PauliX(2)
        qml.PauliX(3)
        # All possible double excitations
        for i in range(0, len(doubles)):
            qml.DoubleExcitation(params[i], wires=doubles[i])

        # All possible single excitations

        for j in range(0, len(singles)):
            qml.SingleExcitation(params[j + len(doubles)], wires=singles[j])

    dev = qml.device("default.qubit", wires=qubits)

    cost_fn = qml.ExpvalCost(circuit, H, dev)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    # total length of parameters is the total no. of determinants considered
    len_params = len(singles) + len(doubles)
    params = np.zeros(len_params)

    # compute the gradients

    dcircuit = qml.grad(cost_fn, argnum=0)

    dcircuit(params)

    prev_energy = 0.0

    for n in range(40):

        t1 = time.time()

        params, energy = opt.step_and_cost(cost_fn, params)

        t2 = time.time()

        print("Iteration = {:},  E = {:.8f} Ha, t = {:.2f} S".format(n, energy, t2 - t1))

        if np.abs(energy - prev_energy) < 10e-6:
            break

        prev_energy = energy

    print("At bond distance \n", reac_coord)
    print("The VQE energy is", energy)

    vqe_energy.append(energy)


# PES

r = np.arange(1.0, 4.0, 0.25)

fig, ax = plt.subplots()
ax.plot(r, vqe_energy, c="red", label="VQE(S+D)")

ax.set(
    xlabel="Reaction Coordinate (x, in Bohr)",
    ylabel="Total energy (in Hartree)",
    title="PES for H2 insertion in Be",
)
ax.grid()
ax.legend()

plt.show()


##############################################################################
# ----------
#
# Below is a comparison with other classical quantum chemistry aproaches such
# as CASCI, CCSD and CISD. we can see VQE(S+D) does really well for both sides
# of PES, left and right of the transition state. While single reference approaches
# such as CISD and CCSD suffer in the latter half of PES. We should note though that
# this behavior of CI and CC methods could be corrected if we chose the right Slater
# determinant at each point of PES, the choice of which as we see varies with
# geometry.
#
#
# .. figure:: /demonstrations/vqe_bond_dissociation/H2_Be.png
#     :width: 50%
#     :align: center
#
#
#
# .. _references:
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
# .. [#purvis1983]
#
#    George D. Purvis III, Ron Shepard, Franklin B. Brown and  Rodney J. Bartlett,
#    "C2V Insertion pathway for BeH2: A test problem for the coupled‐cluster single
#    and double excitation model". `International Journal of Quantum Chemistry,
#    23, 835 (1983).
#    <https://onlinelibrary.wiley.com/doi/abs/10.1002/qua.560230307>`__
#
# .. [#coe2012]
#
#     Jeremy P. Coe  and Daniel J. Taylor and Martin J. Paterson, "Calculations of potential
#     energy surfaces using Monte Carlo configuration interaction". `Journal of Chemical
#     Physics 137, 194111 (2012).
#     <https://doi.org/10.1063/1.4767052>`__
#
# .. [#JMOL]
#
#    `Jmol: an open-source Java viewer for chemical structures in 3D. <http://www.jmol.org/>`__
#
# .. [#motta2020]
#
#    Mario Motta, Tanvi Gujarati, and Julia Rice, `"A Tale of Colliding Electrons: Boosting the
#    Accuracy of Chemical Simulations on Quantum Computers" (2020).
#    <https://medium.com/qiskit/a-tale-of-colliding-electrons-boosting-the-accuracy-of-chemical
#    -simulations-on-quantum-computers-50a4b4ee5c64>`__
