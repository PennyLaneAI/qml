r"""
Modeling bond dissociations and chemical reactions using VQE
=======================

.. meta::
    :property="og:description": Construct Potential energy surface for a simple bond dissociation and general chemical     reactions  using the variational quantum eigensolver algorithm in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/pes_h2.png

.. related::

   tutorial_vqe
   tutorial_vqe_parallel VQE with parallel QPUs
   tutorial_vqe_qng Accelerating VQE with the QNG

*Author: PennyLane dev team. Last updated: 25 Apr 2021.*

Chemical reaction is another name for transformation of molecules - breaking and forming of bonds - accompanied with energy cost. This energy cost determines the feasibility of a particular transformation amongst many different alternate possibilities. Wouldn't it be nice to use theoretical tools to model chemical reactions? This is indeed the case and the field of quantum chemistry has several methods to do that. In this tutorial, we will learn how Pennylane could be used to do exactly the same albeit on a quantum computer or a classical simulator. 

Quantum computing aims to revolutionalize this exploration of chemical reactions. We could possibly build the exact energy landscapes and thus unearth the most feasible routes for any general chemical reaction. This could enable us to find new routes for a chemical reaction to occur (i.e reaction mechanism), develop and design new catalysts and create molecules and materials with tailored properties.

In a previous tutorial, we looked at how a hybrid quantum-classical algorithm, Variational Quantum Eigensolver (VQE), [#peruzzo2014]_ is used to compute molecular energies which are the expectation value of the molecular Hamiltonian. Here, we show how VQE can be used as a tool to construct potential energy surfaces(PES) for any general molecular transformation and how this lends the way to the calculation of important quantities such as activation energy barrier, reaction energy and reaction rates. As illustrative examples, we use VQE and other tools implemented in Pennylane library to study a simple diatomic bond dissociation and reactions involving exchange of hydrogen atoms. Let's get started! 


.. figure:: /demonstrations/vqe_bond_dissociation/h2_pes_pictorial.png
   :width: 50%
   :align: center


An illustration of potential energy surface of H-H bond dissociation for hydrogen molecule. On the y-axis is the total molecular energy and x-axis is the H-H bond distance. By looking at this curve, we could estimate the H-H equilibrium bond distance and the energy required to break the H-H bond.   


##############################################################################


Potential Energy Surfaces: Hills to die and be reborn 
---------------------------------------------------------------------

Potential energy surfaces (PES) are, in simple words, energy landscapes on which any chemical reaction or a general molecular transformation occurs. But what is it? The concept originates with the idea that "nuclies are heavier than electron" aka Born-Oppenheimer approximation and that we can solve for the electronic wavefunction with nucleis clamped to their respective positions. This results in separation of nuclear and electronic parts of the Schrodinger equation and we only solve the electronic part 

.. math:: H_{el}|\Psi \rangle =  E_{el}|\Psi\rangle   
From here arises the concept of electronic energy of the molecule, a quantum mechanical system, as a function of interatomic coordinates and angles, and potential energy surface is a n-dimensional plot of the energy with the respect to the degrees of freedom. It gives us a visual tool to undertstand chemical reactions where stable molecules are the local minimas in the valleys and transition states the *hill peaks* to climb.

To summarize, we solve the electronic Schrodinger equation for a given fixed positions of nucleis, and then we move nucleis in incremental step. The obtained set of energies are then plotted against nuclear positions.

We will begin by showing how this works for a simple diatomic molecule such as H2.  H$_2$ is the simplest of the molecules and the formation (or breaking) of the H-H bond is the simplest of all reactions. 

.. math:: H_2 \rightarrow H + H  

In terms of quantum computing terms, this is a 4 qubit problem if considered in a minimal basis set i.e. 2 electron in 4 spin orbitals. And as discussed in the previous tutorial, the states involved are |1100> (also the Hartree-Fock ground state), |1010>, |0101> and |0011>, these are the only states out of 2^4 (=16) states that matter for this problem and are obtained by single and double particle-hole excitation out of the HF state. Below we show how to set upthe problem to generate a PES for such a reaction. 

The first step is to import the required libraries and packages:
"""

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time

##############################################################################
# The second step is to specify the geometry and charge of the molecule,
# and the spin multiplicity of the electronic configuration. To construct the potential energy surface,
# we need to vary the geometry. So, we keep one H atom fixed at origin and vary the x-coordinate of the other
# H atom such that the bond distance varies from 1.0 to 4.0 Bohrs in steps of 0.25 Bohrs.

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
        package="pyscf",
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping="jordan_wigner",
    )

    print("Number of qubits = ", qubits)
    print("Hamiltonian is ", H)

    ##############################################################################
    # Now to build the circuit for a general molecular system. We begin by preparing the
    # qubit version of HF state, :math:`|1100\rangle`.
    # We then identify and add all possible single and double excitations. In this case, there is only one
    # double excitation(:math:`|0011\rangle`) and two single excitations(:math:`|0110\rangle` and :math:`|1001\rangle`)

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
    # PennyLane contains the :class:`~.ExpvalCost` class, specifically
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
    # can then be translated into determinant coefficients in the expansion
    # of the exact wavefunction.

    # define and initialize the gate parameters
    params = np.zeros(3)
    dcircuit = qml.grad(cost_fn, argnum=0)
    dcircuit(params)

    ##############################################################################
    # We then define the VQE optimization iteration and the energy- based convergence 
    #criteria :math:`\sim 10^{-6}`
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
# Once we have the Energy as a function of H-H bond distance, we could the plot it

# Plot the Potential energy surface
# Energy as a function of internuclear distance

r = np.arange(0.5, 4.0, 0.1)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(r, vqe_energy, label="VQE(S+D)")

ax.set(
    xlabel="Distance (H-H, in Angstrom)", ylabel="Total energy", title="PES for H$_2$ dissociation"
)
ax.grid()
ax.legend()

# fig.savefig("test.png")
plt.show()


##############################################################################
# This is a simple PES (or more appropriately PEC, potential energy curve) for 
# dissociation of Hydrogen molecule. It gives an estimate of H-H bond distance 
# to be ~ 1.4 Bohrs and the H-H bond dissociation energy (the difference in energy 
# at equilibrium and energy at dissociation limit)
# as 0.194 Hartrees (121.8 Kcal/mol). Could these estimates be improved? Yes, 
# by using bigger basis sets or using explicitly correlated methods(f12) and
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
# This reaction has a barrier though, transition state, which it has to cross
# for the exchange of H atom to be complete. In a minimal basis like STO-3G,
# this system consists 3 electron in 6 spin molecular orbitals. This means it is
# a 6 qubit problem and the ground state (HF state) is given by |111000>
#
# .. figure:: /demonstrations/vqe_bond_dissociation/h3_mol_movie.gif
#   :width: 50%
#   :align: center
#
# Again, we need to define the molecular parameters

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
# Then we setup the PES loop, incrementing the H(1)-H(2) distance from 1.0 to 3.0 Bohrs in steps of 0.1 Bohr

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
        package="pyscf",
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping="jordan_wigner",
    )

    # get all the singles and doubles excitations

    singles, doubles = qchem.excitations(active_electrons, active_orbitals * 2)
    print("Single excitations", singles)
    print("Double excitations", doubles)

    def circuit(params, wires):
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
    # The we evaluate the costfunction and use the gradient descent algorithm in an iterative optimization
    # of the gate parameters.

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
    #  Finally at each point of the 1D PES, we could print the total VQE energy

    print("At bond distance \n", r_HH)
    print("The VQE energy is", energy)

    vqe_energy.append(energy)

#
##############################################################################
#
# Then we could plot the energy as a function of H-H distance which is also the reaction coordinate
# and thus, we have the potential energy curve for this reaction.

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
# Activation energy barrier and Reaction Rate
# --------------------------------------------
# In the potential energy surface above, we would like to have 
# a good estimate of the energies of the reactants(minima 1), products (minima 2) and
# the transition state (maxima). VQE(S+D) reproduces the exact result in this small
# basis. The plot below compares the performance of many methods with each other
# VQE(S+D), our chosen ansatz reminiscent of UCCSD approach, overlaps with 
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
# transition state ( H --- H --- H).
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
# The reaction rate (k) has an exponential dependence on the activation energy barrier:
#
# .. math:: k = Ae^{-{E_{Activation Barrier}}/RT}
#
# So, in principle, if we know the constant (A) we could calculate the rate of the reaction.


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
# A symmetric insertion of :math:`H_2` with :math:`Be` atom: a model multireference problem
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
# We fix the Beryllium atom at the origin and the coordinates for hydrogen atoms are give by, in Bohr, 
# the coordinates (x, y, 0) and (x, −y, 0) where y = 2.54 − 0.46x and x ∈ [1, 4].
# The generation of PES then is straightforward and follows from our previous examples. 
# For the sake of saving computational cost, we try a smaller active space of a total of
# 6 spin MOs with core electrons frozen.

# Molecular parameters
name = 'beh2'
basis_set = 'sto-3g'

electrons = 6
charge = 0
spin =1
multiplicity=1

active_electrons=4
# choosing a smaller active space - 6 spin MOs
active_orbitals= 3
vqe_energy = []


name = "beh2"

for reac_coord in np.arange(1.0, 4.0, 0.25) :


    x = reac_coord
    y = np.subtract(2.54, np.multiply(0.46, x))

    symbols, coordinates = (["Be", "H","H"], np.array([0.0, 0.0, 0.0, x, y, 0.0, x, -y, 0.0]))

    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        mult=multiplicity,
        basis=basis_set,
        package='pyscf',
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping='jordan_wigner'
    )

    # get all the singles and doubles excitations

    singles, doubles = qchem.excitations(active_electrons, active_orbitals * 2)
    print("Single excitations", singles)
    print("Double excitations", doubles)


    def circuit(params, wires):
        qml.PauliX(0)
        qml.PauliX(1)
        qml.PauliX(2)
        qml.PauliX(3)
        # All possible double excitations
        for i in range(0,len(doubles)):
            qml.DoubleExcitation(params[i], wires=doubles[i])

        # All possible single excitations

        for j in range(0,len(singles)):
            qml.SingleExcitation(params[j+len(doubles)], wires=singles[j])

    dev = qml.device("default.qubit", wires=qubits)

    cost_fn = qml.ExpvalCost(circuit, H, dev)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)


    # total length of parameters is generally the total no. of determinants considered
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

        print("Iteration = {:},  E = {:.8f} Ha, t = {:.2f} S".format(n, energy, t2-t1))

        if (np.abs(energy - prev_energy) < 10E-6 ):
            break

        prev_energy = energy



    print("At bond distance \n", reac_coord)
    print("The VQE energy is", energy)

    vqe_energy.append(energy)


# PES 

r = np.arange(1.0, 4.0, 0.25)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot (r, vqe_energy, c = 'red', label ='VQE(S+D)')

ax.set(xlabel='Reaction Coordinate (in Bohr)', ylabel='Total energy',
       title='PES for H2 insertion in Be reaction')
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
