r"""
Modelling chemical reactions on a quantum computer
====================================================

.. meta::
    :property="og:description": Construct potential energy surfaces for chemical reactions using 
    the variational quantum eigensolver algorithm in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/pes_h2.png

.. related::
   tutorial_quantum_chemistry Quantum Chemistry with PennyLane
   tutorial_vqe Variational Quantum Eigensolver

*Author: PennyLane dev team. Posted: 19 May 2021. Last updated: 19 May 2021.*

The term "chemical reaction" is another name for the transformation of molecules -- the breaking and 
forming of bonds. Such transformations involve energy costs that determine 
the feasibility of a particular transformation amongst many different possibilities. 
Computational chemistry offers several theoretical methods for precisely determining this energy
cost. It also opens a window into predicting the thermodynamic and kinetic aspects of any 
chemical reaction. In this tutorial, you will learn how to use PennyLane to simulate any general 
chemical reaction by constructing the corresponding potential energy surface (a theoretical 
construct and a visual aid to understanding how the reaction proceeds), and estimating the 
relevant energy costs. 

In a previous tutorial on :doc:`Variational Quantum Eigensolver (VQE) </demos/tutorial_vqe>`, 
we looked at how it can be used to compute molecular energies [#peruzzo2014]_.
Here, we show how VQE can be used to construct potential energy surfaces (PESs) for any general 
molecular transformation. This paves the way
for the calculation of important quantities such as activation energy barriers, reaction energies,
and reaction rates. As illustrative examples, we use tools implemented in PennyLane to study diatomic
bond dissociation and reactions involving the exchange of hydrogen atoms. Our first example will show how
to write the code to generate the plot depicting the dissociation of the :math:`H_2` molecule shown
below. 

.. _label_h2_pes:
.. figure:: /demonstrations/vqe_bond_dissociation/h2_pes_pictorial.png
   :width: 90%
   :align: center
   
   Potential energy surface depicting single bond dissociation in hydrogen molecule.


##############################################################################

Potential Energy Surfaces 
---------------------------------------------------------------------

`Potential energy surfaces (PES) <https://en.wikipedia.org/wiki/Potential_energy_surface>`_
are, in simple words, energy landscapes on which chemical
reactions occur. The concept originates from the fact that nuclei are much heavier than 
electrons and thus move much slowly, allowing us to decouple their motion. This is known as the
`Born-Oppenheimer approximation <https://en.wikipedia.org/wiki/Born–Oppenheimer_approximation>`_. 
We can then solve for the electronic wavefunction with
the nuclei clamped to their respective positions. This results in separation of nuclear and
electronic parts of the Schrödinger equation so we need only solve the electronic
equation:

.. math:: H_{el}|\Psi \rangle =  E_{el}|\Psi\rangle.

Thus arises the concept of the electronic energy of the molecule, :math:`E(R)`, 
as a function of interatomic coordinates. The potential energy surface of a 
molecule is a 
:math:`n`-dimensional plot of the energy with the respect to the degrees of freedom. It gives us a 
visual tool to understand chemical reactions by locating stable molecules (reactants and products)
as the local minima and transition states as peaks and marking the possible routes of a plausible
transformation.

To summarize, to build the potential energy surface, we solve the electronic Schrödinger
equation for fixed positions of the nuclei, and subsequently move them in incremental steps
while solving the Schrödinger equation at each such configuration. 
The obtained set of energies thus correspond to a grid of nuclear positions and the plot 
:math:`E(R)` vs :math:`R` is the potential energy surface. 
To really understand the steps involved in making such a plot, let us dive straight into it.

##########################################################

Bond dissociation in Hydrogen molecule 
---------------------------------------------------------

We begin with the simplest of molecules: :math:`H_2`. 
The formation (or breaking) of the :math:`H-H` bond is also the simplest
of all reactions:

.. math:: H_2 \rightarrow H + H.

We now cast this problem in the language of `quantum chemistry 
<https://en.wikipedia.org/wiki/Quantum_chemistry>`_ and then in terms of
quantum circuits. For an introductory discussion, please take a look at 
:doc:`Quantum Chemistry with PennyLane </demos/tutorial_quantum_chemistry>` tutorial.
Using a minimal `basis set <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`_ 
(`STO-3G <https://en.wikipedia.org/wiki/STO-nG_basis_sets>`_), 
this molecular system can be described by :math:`2` electrons in :math:`4` 
spin-orbitals. When mapped to a qubit representation, we need a total of four qubits to represent
it. 
The `Hartree-Fock (HF) <http://vergil.chemistry.gatech.edu/notes/hf-intro/node7.html>`_ 
ground state is  represented as :math:`|1100\rangle`, where the two
lowest-energy orbitals are occupied, and the remaining two are unoccupied. To form the complete 
basis of states, we consider excitations out of the HF state that conserve the spin. 
In this case, where there are two electrons, single and double excitations suffice. The 
singly-excited states are :math:`|0110\rangle`, :math:`|1001\rangle`, and the doubly-excited state 
is :math:`|0011\rangle`. The exact wavefunction (also known as full `configuration interaction
<https://en.wikipedia.org/wiki/Configuration_interaction>`_ or FCI) 
is a linear expansion in terms of these states where 
the expansion coefficients would change as the reaction proceeds and the system moves around 
(figuratively) on the potential energy surface. 
Below we show how to to generate the PES for such a reaction. 

The first step is to import the required libraries and packages:
"""

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##############################################################################
# We then specify the basis set and the atomic symbols of the constituent atoms.
# All electrons and all orbitals are considered in this case.

basis_set = "sto-3g"
symbols = ["H", "H"]

##############################################################################
# To construct the potential energy surface, we need to vary the geometry of the molecule. We keep
# an :math:`H` atom fixed at the origin and vary the
# :math:`x`-coordinate of the other atom such that the bond distance changes from
# :math:`0.5` to :math:`5.0` Bohrs in steps of :math:`0.1` Bohr.
# This covers the range of internuclear distance in which the :math:`H-H` bond is formed
# (equilibrium bond length)
# and also the distance when the bond is broken, which occurs when the atoms
# are far away from each other.
#
#
# Now we set up a loop that incrementally changes the internuclear distance and for each
# such point we generate a molecular Hamiltonian using the
# :func:`~.pennylane_qchem.qchem.molecular_hamiltonian` function.
# At each point, we solve the electronic Schrödinger equation by first solving the
# Hartree-Fock approximation and generating the molecular orbitals (MOs). For a more accurate
# estimation of the molecular wavefunction and energy, we build the
# VQE circuit by first preparing the qubit version of the HF state and then adding all single
# excitation and double excitation gates which use the `Givens rotations
# <https://en.wikipedia.org/wiki/Givens_rotation>`_. This approach is similar to
# the `Unitary Coupled Cluster (UCCSD) <https://youtu.be/sYJ5Ib-8k_8>`_ often used.
#
#
# We use a classical qubit simulator and define
# a cost function which calculates the expectation value of Hamiltonian operator for the
# given trial wavefunction (which is the ground state energy of the molecule) using
# :class:`~.ExpvalCost`.
# For the problems discussed here, we use gradient descent to optimize
# the gate parameters. We initialize the parameters to zero,
# i.e., start from the HF state as the approximation to the exact state.
# The second loop is the variational optimization using VQE algorithm,
# where energy for the trial wavefunction is calculated
# and then used to get a better estimate of gate parameters and improve the trial wavefunction.
# This process is repeated until the energy converges (:math:`E_{n} - E_{n-1} < 10^{-6}` Hartree).
# Once we have the converged VQE energy at the specified internuclear distance, we
# increment the distance and the whole cycle of HF calculation, building quantum circuits and
# the iterative VQE optimization of gate parameters is repeated. After we have covered the grid of
# internuclear distances, we tabulate the results.

vqe_energy = []
pes_point = 0
# set up a loop to change internuclear distance
r_range = np.arange(0.5, 5.0, 0.1).round(1)
for r in r_range:

    coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, r])

    # Obtain the qubit Hamiltonian
    H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, basis=basis_set)

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

    # define the device, cost function and optimizer
    dev = qml.device("default.qubit", wires=qubits)
    cost_fn = qml.ExpvalCost(circuit, H, dev)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    # define and initialize the gate parameters
    params = np.zeros(3)

    # if this is not the first point on PES, initialize with converged parameters
    # from previous point
    if pes_point > 1:
        params = params_old

    # Begin the VQE iteration to optimize gate parameters
    prev_energy = 0.0

    for n in range(50):

        params, energy = opt.step_and_cost(cost_fn, params)
        # print("Iteration = {:},  E = {:.8f} Ha ".format(n, energy))

        # define the convergence criteria
        if np.abs(energy - prev_energy) < 1e-6:
            break

        prev_energy = energy

    # store the converged parameters
    params_old = params
    pes_point = pes_point + 1

    print("At r = {:.1f} Bohrs, number of VQE Iterations required is {:}".format(r, n))
    vqe_energy.append(energy)

# tabulate
list_dist_energy = list(zip(r_range, vqe_energy))
df = pd.DataFrame(list_dist_energy, columns=["H-H distance (in Bohr)", "Energy (in Ha)"])
# display table
print(df)

##############################################################################
# We have calculated the molecular energy as a function of :math:`H-H` bond distance;
# let us plot it.

# Energy as a function of internuclear distance
fig, ax = plt.subplots()
ax.plot(r_range, vqe_energy, label="VQE")

ax.set(
    xlabel="H-H distance (in Bohr)",
    ylabel="Total energy (in Hartree)",
    title="Potential energy surface for H$_2$ dissociation",
)
ax.grid()
ax.legend()

plt.show()


##############################################################################
# This is a simple potential energy surface (or more appropriately, a potential energy curve) for
# the dissociation of a hydrogen molecule into two hydrogen atoms. Exactly the same as shown in
# the illustrated image of :ref:`label_h2_pes`. Let us now understand the utility
# of such a plot.
#
#
# In a diatomic molecule such as :math:`H_2`, the potential energy curve as a function of
# internuclear distance tells us the bond length ---
# the distance between the two atoms when the energy is at a minimum and the system is in
# equilibrium --- and the bond dissociation energy.
# The bond dissociation energy is the amount of energy required to dissociate a bond.
# It is calculated as the difference in energy of the system at equilibrium (minima) and the energy
# of the system at the dissociation limit, where the atoms are far apart and
# the total energy plateaus to a constant: the sum of each atom's individual energy.
# Below we show how our VQE circuit gives an
# estimate of :math:`H-H` bond distance
# to be :math:`\sim 1.4` Bohrs and the :math:`H-H` bond dissociation energy
# as :math:`0.202` Hartrees (:math:`126.79` kcal/mol).

# Energy at equilibrium bond length (minima)
energy_equil = min(vqe_energy)

# Energy at dissociation limit (the point on PES where the atoms are far apart)
energy_dissoc = vqe_energy[-1]

# Bond dissociation energy
bond_dissociation_energy = energy_dissoc - energy_equil
bond_dissociation_energy_kcal = bond_dissociation_energy * 627.5

# H-H bond length is the bond distance at equilibrium geometry
bond_length_index = vqe_energy.index(energy_equil)
bond_length = r_range[bond_length_index]

print("The H-H bond length is {:.1f} Bohrs".format(bond_length))
print(
    "The H-H bond dissociation energy is {:.6f} Hartrees or {:.2f} kcal/mol".format(
        bond_dissociation_energy, bond_dissociation_energy_kcal
    )
)


##############################################################################
# These estimates can be improved
# by using bigger basis sets and extrapolating to the complete basis set limit [#motta2020]_.
# We must also note that our results are subject to gridsize of the span of interatomic
# distances considered. The finer the gridsize, the better the estimate of bond length and
# dissociation energy.
# Now let's move on to a more interesting chemical reaction.
#

##############################################################################
# Hydrogen Exchange Reaction
# -----------------------------
#
# After studying a simple diatomic bond dissociation, we move to a slightly more complicated
# case: a hydrogen exchange reaction.
#
# .. math:: H_2 + H \rightarrow H + H_2.
#
# This reaction has a barrier, the transition state, that it has to cross
# for the exchange of an :math:`H` atom to be complete. In this case, the transition state
# corresponds to a specific linear arrangement of the atoms where one :math:`H-H` bond is
# partially broken and the other :math:`H-H` bond is partially formed.
# The molecular movie (below) is an illustration of the reaction trajectory --- how the distance
# between the hydrogen atoms labelled :math:`1`, :math:`2` and :math:`3` changes as the bond between
# :math:`H(1)` and :math:`H(2)` is broken and another one between :math:`H(2)` and :math:`H(3)`
# is formed. This path along which the reaction proceeds is also known as the `reaction coordinate
# <https://en.wikipedia.org/wiki/Reaction_coordinate>`_.
#
# .. figure:: /demonstrations/vqe_bond_dissociation/h3_mol_movie.gif
#   :width: 50%
#   :align: center
#
# In a minimal basis like STO-3G,
# this system consists of :math:`3` electrons in :math:`6` spin molecular orbitals.
# This translates into a :math:`6` qubit problem and the Hartree-Fock state
# is :math:`|111000\rangle`. As there is an unpaired
# electron, the spin multiplicity is two and needs to be specified.
# As in previous case, all electrons and all orbitals are considered.


# Molecular parameters
basis_set = "sto-3g"

multiplicity = 2

active_electrons = 3
active_orbitals = 3

symbols = ["H", "H", "H"]

##############################################################################
# We setup the PES loop incrementing the :math:`H(1)-H(2)` distance from :math:`1.0`
# to :math:`3.0` Bohrs in steps of :math:`0.1` Bohr.
# We use PennyLane's
# :class:`~.BasisState` operation to construct the HF state and :class:`excitations`
# to obtain the list of allowed single and double excitations out of the HF state.
# The way we build the VQE circuit for this system is generic and can be used to
# build circuits for any molecular system. We then repeat the calculations over the
# whole range of PES and print the converged energies of the whole system at each step.

vqe_energy = []
pes_point = 0

r_range = np.arange(1.0, 3.0, 0.1).round(1)
for r in r_range:

    coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, r, 0.0, 0.0, 4.0])

    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        mult=multiplicity,
        basis=basis_set,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    # get all the singles and doubles excitations
    singles, doubles = qchem.excitations(active_electrons, active_orbitals * 2)

    def circuit(params, wires):
        qml.BasisState(np.array([1, 1, 1, 0, 0, 0]), wires=(0, 1, 2, 3, 4, 5))
        for i in range(0, len(doubles)):
            qml.DoubleExcitation(params[i], wires=doubles[i])
        for j in range(0, len(singles)):
            qml.SingleExcitation(params[j + len(doubles)], wires=singles[j])

    dev = qml.device("default.qubit", wires=qubits)
    cost_fn = qml.ExpvalCost(circuit, H, dev)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    len_params = len(singles) + len(doubles)
    params = np.zeros(len_params)

    # if this is not the first point on PES, initialize with converged parameters
    # from previous point
    if pes_point > 1:
        params = params_old

    prev_energy = 0.0

    for n in range(60):

        params, energy = opt.step_and_cost(cost_fn, params)
        # print("Iteration = {:},  E = {:.8f} Ha ".format(n, energy))

        if np.abs(energy - prev_energy) < 1e-6:
            break

        prev_energy = energy

    # store the converged parameters
    params_old = params
    pes_point = pes_point + 1

    print("At r = {:.1f} Bohrs, number of VQE Iterations required is {:}".format(r, n))
    vqe_energy.append(energy)

# tabulate
list_dist_energy = list(zip(r_range, vqe_energy))
# Converting list into pandas Dataframe
df = pd.DataFrame(list_dist_energy, columns=["H(1)-H(2) distance (in Bohr)", "Energy (in Ha)"])
# display table
print(df)

##############################################################################
# .. note::
#
#     Did you notice a trick we used to speed up the convergence of VQE energy? The converged
#     gate parameters for a particular point on PES are used as the initial guess for the next
#     geometry. With a better guess, the VQE iterations converge relatively quickly and we save
#     considerable time.
#
# After tabulating our results, we plot the energy as a function of distance between atoms
# :math:`1` and :math:`2`, and thus we have the potential energy curve for
# this reaction. The minimas in the curve represent the VQE estimate of the energy and geometry
# of reactants and products respectively while the transition state is represented by the
# local maxima.
# In general, we would like our method (VQE) to provide
# a good estimate of the energies of the reactants (minima :math:`1`), products (minima :math:`2`),
# and the transition state (maxima). We shall compare the VQE results later and find that it
# reproduces the exact energies (FCI) in the chosen basis (STO-3G) throughout the PES.

# plot the PES
fig, ax = plt.subplots()
ax.plot(r_range, vqe_energy)

ax.set(
    xlabel="Distance (H-H, in Bohr)",
    ylabel="Total energy (in Hartree)",
)
ax.grid()
plt.show()
##############################################################################
# Activation energy barriers and reaction rates
# ***************************************************
# The utility of potential energy surfaces lies in estimating
# reaction energies and activation energy barriers, as well as the
# geometric configurations of key reactants,
# `transition states <https://en.wikipedia.org/wiki/Transition_state>`_
# and products.
# The activation energy barrier (:math:`E_{a}`) is defined as the difference between the
# energy of the reactant complex
# and the energy of the
# transition state.
#
# .. math:: E_{a} = E_{TS} - E_{Reactant}
#
# In the case of the hydrogen exchange reaction, the activation energy barrier is
#                   :math:`E_{a} = 0.0275` Ha :math:`= 17.26` kcal/mol
# Below we show how to calculate the activation energy from the above PES.

# Energy of the reactants and products - two minimas on the PES
energy_equil = min(vqe_energy)
energy_equil_2 = min([x for x in vqe_energy if x != min(vqe_energy)])

# Between the two minimas, we have the TS which is a local maxima
bond_length_index_1 = vqe_energy.index(energy_equil)
bond_length_index_2 = vqe_energy.index(energy_equil_2)

# Reaction coordinate at the two minimas
index_1 = min(bond_length_index_1, bond_length_index_2)
index_2 = max(bond_length_index_1, bond_length_index_2)

# Transition state energy
energy_ts = max(vqe_energy[index_1:index_2])

# Activation energy
activation_energy = energy_ts - energy_equil
activation_energy_kcal = activation_energy * 627.5

print(
    "The activation energy is {:.6f} Hartrees or {:.2f} Kcal/mol".format(
        activation_energy, activation_energy_kcal
    )
)

##############################################################################
# Though this is the best theoretical estimate in this small basis,
# this is not the *best* theoretical estimate. We would need to do this calculation
# in larger basis to reach basis set
# convergence and this would significantly increase the number of qubits required.
#
# The reaction rate constant (:math:`k`) has an exponential dependence on the activation energy
# barrier as shown in the `Arrhenius equation <https://en.wikipedia.org/wiki/Arrhenius_equation>`_:
#
# .. math:: k = Ae^{-{E_{a}}/RT}.
#
# So, in principle, if we know the constant (:math:`A`) we could calculate the rate constant
# and the rate of the reaction.
# In general, we desire our method to accurately predict these energy differences
# and the geometries of the key intermediates.
# The plot below compares the performance of VQE with other quantum chemistry methods such as
# Hartree-Fock and Full CI.

##############################################################################
# .. figure:: /demonstrations/vqe_bond_dissociation/h3_comparison.png
#     :width: 90%
#     :align: center
#
# The PEC obtained from the quantum algorithm (VQE) overlaps with
# the FCI result. This is a
# validation of the accuracy of the energy estimates through our VQE circuit --- we exactly
# reproduce the absolute energies of the reactants, transition state and product.

##############################################################################
# A model multi-reference problem: :math:`Be + H_{2} \rightarrow BeH_{2}`
# -------------------------------------------------------------------------------------------
#
# In our previous examples, the Hartree-Fock state was a good approximation to the exact
# ground state for all points on the potential energy surface.
#
# .. math:: \langle \Psi_{HF}| \Psi_{exact}\rangle \simeq 1
#
# Hence, we refer to them as single-reference
# problems. However, there exist situations where the above condition does not hold and
# states other than HF state become considerably important and are
# required for accuracy across the potential energy surface.
# These are called multi-reference problems.
# A symmetric approach of :math:`H_2` to :math:`Be` atom to form :math:`BeH_2`
# (see the animation of reaction trajectory below)
# constitutes a multi-reference problem. [#purvis1983]_
#
#
# .. figure:: /demonstrations/vqe_bond_dissociation/beh2_movie.gif
#     :width: 70%
#     :align: center
#
# Let us first understand the problem in more detail. Once we have solved the mean-field HF
# equations, we obtain the molecular orbitals
# (:math:`1a,2a,3a,1b ...`) which are then occupied to obtain two principal states
# :math:`1a^{2} 2a^{2} 3a^{2}` and :math:`1a^{2} 2a^{2} 1b^{2}`.
# These are the two different
# reference states needed to qualitatively describe the full potential energy surface
# for this reaction. Prior studies have found that one of these states is a good reference
# for one half of the PES while another is good reference for the rest of PES. [#purvis1983]_
# To figure out the reaction coordinate for the
# approach of the :math:`H_2` molecule to the Beryllium atom, we refer to the work by
# Coe et al. [#coe2012]_
# We fix the Beryllium atom at the origin and the coordinates for the hydrogen atoms
# are given by :math:`(x, y, 0)` and :math:`(x, −y, 0)`, where :math:`y = 2.54 − 0.46x`
# and :math:`x \in [1, 4]`. All distances are in Bohr.
#
#
# We set this last section as a challenge problem for the reader. We would like you to
# generate the potential energy surface for this reaction and compare how
# the performance of VQE with Full CI results.
# All you need to do is to traverse along the
# specified reaction coordinate, generate molecular geometries and obtain the converged 
# VQE energies. The code is similar and follows from our previous examples.
#
#
# Below is the PES plot you would be able to generate. We have provided the HF and FCI curves for
# comparison. A sharp maximum could be seen in these curves which is
# due to a sudden switch in the underlying Hartree-Fock reference at the specific reaction
# coordinate.
# The performance of VQE depends on the active space chosen i.e. the number of electrons and
# orbitals that are being considered. For reference, we have included
# the VQE results when the number of active orbitals is constrained to :math:`3` spatial orbitals
# which is equal to :math:`6` spin orbitals. As a simple exercise, try increasing
# the number of active orbitals and see how the performance of our VQE circuit changes.
# Does your VQE circuit
# reproduce the Full CI result shown below if we increase the active orbitals to include
# all the unoccupied orbitals i.e. 12 spin orbitals in total?
#
#
# .. figure:: /demonstrations/vqe_bond_dissociation/H2_Be.png
#     :width: 90%
#     :align: center
#
#
# To summarize, we looked at three different chemical reactions and construct
# their potential energy curves. We calculated the bond dissociation energy,
# activation energy, and bond length for the relevant situations.
# We also saw how quantum computing algorithms such as VQE can in principle
# provide results that match conventional quantum chemistry methods for these systems.
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
