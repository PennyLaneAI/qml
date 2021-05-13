r"""
Modelling chemical reactions using VQE
==============================

.. meta::
    :property="og:description": Construct potential energy surfaces for chemical reactions using 
    the variational quantum eigensolver algorithm in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/pes_h2.png

.. related::

   tutorial_vqe
   tutorial_vqe_parallel VQE with parallel QPUs

*Author: PennyLane dev team. Posted: 13 May 2021. Last updated: 13 May 2021.*

The term "chemical reaction" is another name for the transformation of molecules -- breaking and 
forming of bonds. Such transformations involve energy costs that determine 
the feasibility of a particular transformation amongst many different possibilities. 
Computational chemistry offers several theoretical methods for precisely determining this energy
cost. It also opens a windows into predicting the thermodynamic and kinetics aspects of any 
chemical reaction. In this tutorial, you will learn how to use PennyLane to simulate chemical 
reactions. 

In a previous tutorial on :doc:`Variational Quantum Eigensolver (VQE) </demos/tutorial_vqe>`, 
we looked at how it can be used to compute molecular energies. [#peruzzo2014]_ 
Here, we show how VQE can be used to construct potential energy surfaces (PESs) for any general 
molecular transformation. This paves the way for the calculation of important quantities such as
activation energy barriers, reaction energies, and reaction rates. As illustrative examples, we
use tools implemented in PennyLane to study diatomic bond dissociation and reactions involving 
the exchange of hydrogen atoms. Let's get started! 



.. figure:: /demonstrations/vqe_bond_dissociation/h2_pes_pictorial.png
   :width: 90%
   :align: center
   
   An illustration of the potential energy surface of bond dissociation for the hydrogen molecule. 
   On the :math:`y`-axis is the total energy and :math:`x`-axis is the internuclear bond
   distance. By looking at this curve, we can estimate the :math:`H-H` bond length and the energy 
   required to break the bond.   


##############################################################################


Potential Energy Surfaces: Hills to die and be reborn 
---------------------------------------------------------------------

`Potential energy surfaces (PES) <https://en.wikipedia.org/wiki/Potential_energy_surface>`_
are, in simple words, energy landscapes on which chemical
reactions occur. The concept originates from the fact that nuclei are much heavier than 
electrons and thus move much slowly, allowing us to decouple their motion. This is known as the
`Born-Oppenheimer approximation <https://en.wikipedia.org/wiki/Born–Oppenheimer_approximation>`_. 
We can then solve for the electronic wavefunction with
the nuclei clamped to their respective positions. This results in separation of nuclear and
electronic parts of the Schrödinger equation and we then only solve the electronic
equation:

.. math:: H_{el}|\Psi \rangle =  E_{el}|\Psi\rangle.

Thus arises the concept of the electronic energy of the molecule, :math:`E(R)`, 
as a function of interatomic coordinates. The potential energy surface of a 
molecule is a 
:math:`n`-dimensional plot of the energy with the respect to the degrees of freedom. It gives us a 
visual tool to understand chemical reactions where stable molecules are the local minima 
and transition states are the peaks.

To summarize, to build the potential energy surface, we solve the electronic Schrödinger
equation for a set of positions of the nuclei, and subsequently move them in incremental steps
to obtain the energy at other configurations. The obtained set of energies are then 
plotted against nuclear positions.

We begin with the simplest of molecules: :math:`H_2`. 
The formation (or breaking) of the :math:`H-H` bond is also the simplest
of all reactions:

.. math:: H_2 \rightarrow H + H.

Using a minimal basis set (STO-3G), this system can be described by :math:`2` electrons in :math:`4` 
spin-orbitals. When mapped to a qubit representation, we need a total of four qubits to represent it 
and the Hartree-Fock (HF) ground state is  represented as :math:`|1100\rangle` where two
energetically lowest orbitals are occupied and rest two are unccupied. To form the complete basis of
states, we consider excitations of the HF state that conserve the spin. In this case, where 
there are two electrons, single and double excitations suffice. The singly-excited 
states are :math:`|0110\rangle`, :math:`|1001\rangle`, and the doubly-excited state is 
:math:`|0011\rangle`. The exact wavefunction (also known as full configuration interaction or FCI) 
is a linear expansion in terms of these states where 
the expansion coefficients would change as the reaction proceeds and the  system moves around (figuratively) 
on the potential energy surface. Below we show how to to generate the PES for such a reaction. 

The first step is to import the required libraries and packages:
"""

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# We begin by specifying the basis set and the number of active electrons and active orbitals.
# All electrons and all orbitals are considered in this case.

basis_set = "sto-3g"

active_electrons = 2
active_orbitals = 2

##############################################################################
# To construct the potential energy surface, we need to vary the geometry of the molecule. We keep
# an :math:`H` atom fixed at the origin and vary the
# :math:`x`-coordinate of the other atom such that the bond distance changes from
# :math:`0.5` to :math:`5.0` Bohrs in steps of :math:`0.1` Bohr.
# This covers the range of internuclear distance in which the :math:`H-H` bond is formed
# (equilibrium bond length)
# and also the distance when the bond is broken, which occurs when the atoms
# are far away from each other.
# Now we set up a loop that incrementally changes the internuclear distance and for each
# such point we generate a molecular Hamiltonian using the
# :func:`~.pennylane_qchem.qchem.molecular_hamiltonian` function.
#
#
# We build the VQE circuit by first
# preparing the qubit version of the HF state and then adding all single excitation and
# double excitation gates which use the Givens rotations. This approach is similar to
# the Unitary Coupled Cluster (UCCSD) approach often used.
# We use a classical qubit simulator and define
# a cost function which calculates the expectation value of Hamiltonian operator for the
# given trial wavefunction (which is the ground state energy of the molecule) using
# :class:`~.ExpvalCost`.
# For the problems discussed here, we use gradient descent to optimize
# the gate parameters. We initialize the parameters to zero,
# i.e. start from the HF state as the approximation to the exact state.
# The second loop is the variational optimization using VQE algorithm,
# where energy for the trial wavefunction is calculated
# and then used to get a better estimate of gate parameters and improve the trial wavefunction.
# This process is repeated till convergence.

symbols = ["H", "H"]
vqe_energy = []
# set up a loop to change internuclear distance
for r in np.arange(0.5, 5.0, 0.1):

    coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, r])

    # Obtain the qubit Hamiltonian
    H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, basis=basis_set)

    print("Number of qubits = ", qubits)
    print("Hamiltonian is ", H)

    # get all the singles and doubles excitations
    singles, doubles = qchem.excitations(active_electrons, active_orbitals * 2)

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

    ##############################################################################
    # Begin the VQE iteration to optimize gate parameters.

    prev_energy = 0.0

    for n in range(40):

        params, energy = opt.step_and_cost(cost_fn, params)
        print("Iteration = {:},  E = {:.8f} Ha ".format(n, energy))

        # define the convergence criteria
        if np.abs(energy - prev_energy) < 1e-6:
            break

        prev_energy = energy

    print("At bond distance \n", r)
    print("The VQE energy is", energy)

    vqe_energy.append(energy)

##############################################################################
# We have calculated the molecular energy as a function of :math:`H-H` bond distance;
# let us plot it.

# Energy as a function of internuclear distance
r = np.arange(0.5, 5.0, 0.1)

fig, ax = plt.subplots()
ax.plot(r, vqe_energy, label="VQE")

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
# the dissociation of hydrogen molecule into two hydrogen atoms.
# In a diatomic molecule, the potential energy curve as a function of internuclear distance tells
# us the bond length:
# the distance between the two atoms when the energy is at a minimum and the system is in
# equilibrium.
# The bond dissociation energy is the amount of energy required to dissociate a bond.
# In other words, the difference in energy of the system at equilibrium (minima) and the energy
# of the system at the dissociation limit, where the atoms are far apart and
# the total energy plateaus to a constant: the sum of each atom's individual energy.
# Below we show how our VQE circuit gives an
# estimate of :math:`H-H` bond distance
# to be :math:`\sim 1.4` Bohrs and the :math:`H-H` bond dissociation energy
# as :math:`0.202` Hartrees (:math:`126.79` Kcal/mol).

energy_equil = min(vqe_energy)
energy_dissoc = vqe_energy[-1]

bond_dissociation_energy = energy_dissoc - energy_equil
bond_dissociation_energy_kcal = bond_dissociation_energy * 627.5

bond_length_index = vqe_energy.index(energy_equil)
bond_length = r[bond_length_index]

print("The H-H bond length is {:.1f} Bohrs".format(bond_length))
print(
    "The H-H bond dissociation energy is {:.6f} Hartrees or {:.2f} Kcal/mol".format(
        bond_dissociation_energy, bond_dissociation_energy_kcal
    )
)


##############################################################################
# These estimates can be improved
# by using bigger basis sets and extrapolating to the complete basis set limit. [#motta2020]_
# Now let's move on to a more interesting chemical reaction.
#


##############################################################################
# Hydrogen Exchange Reaction
# -----------------------------
#
# We construct the PES for a simple hydrogen exchange reaction
#
# .. math:: H_2 + H \rightarrow H + H_2.
#
# This reaction has a barrier, the transition state, that it has to cross
# for the exchange of an :math:`H` atom to be complete. In this case, the transition state
# corresponds to a particular linear arrangement of the atoms where one :math:`H-H` bond is
# partially broken and the other :math:`H-H` bond is partially formed.
# In a minimal basis like STO-3G,
# this system consists of :math:`3` electrons in :math:`6` spin molecular orbitals.
# This means it is a :math:`6` qubit problem and the Hartree-Fock state in
# occupation number representation is :math:`|111000\rangle`. As there is an unpaired
# electron, the spin multiplicity is two.
#
# .. figure:: /demonstrations/vqe_bond_dissociation/h3_mol_movie.gif
#   :width: 50%
#   :align: center
#

# Molecular parameters
basis_set = "sto-3g"

multiplicity = 2

active_electrons = 3
active_orbitals = 3

##############################################################################
# Then we setup the PES loop, incrementing the :math:`H(1)-H(2)` distance from :math:`1.0`
# to :math:`3.0` Bohrs in steps of :math:`0.1` Bohr. We use Pennylane
# :class:`~.BasisState` class to construct the HF state in this case.

symbols = ["H", "H", "H"]
vqe_energy = []

for r in np.arange(1.0, 3.0, 0.1):

    coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, r, 0.0, 0.0, 4.0])

    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        mult=multiplicity,
        basis=basis_set,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

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

    prev_energy = 0.0

    for n in range(40):

        params, energy = opt.step_and_cost(cost_fn, params)
        print("Iteration = {:},  E = {:.8f} Ha ".format(n, energy))

        if np.abs(energy - prev_energy) < 1e-6:
            break

        prev_energy = energy

    vqe_energy.append(energy)

#
##############################################################################
#
# Then we plot the energy as a function of distance between atoms :math:`1` and :math:`2`,
# which is also the `reaction coordinate <https://en.wikipedia.org/wiki/Reaction_coordinate>`_,
# and thus we have the potential energy curve for
# this reaction.

r = np.arange(1.0, 3.0, 0.1)

fig, ax = plt.subplots()
ax.plot(r, vqe_energy)

ax.set(
    xlabel="Distance (H-H, in Bohr)",
    ylabel="Total energy (in Hartree)",
)
ax.grid()
plt.show()
#
#
##############################################################################
# Activation energy barriers and reaction rates
# --------------------------------------------
# The utility of potential energy surfaces lies in estimating
# reaction energies and activation energy barriers, as well as the
# geometric configurations of key reactants, intermediates,
# `transition states <https://en.wikipedia.org/wiki/Transition_state>`_
# and products.
# In general, we would like our method (VQE) to provide
# a good estimate of the energies of the reactants (minima :math:`1`), products (minima :math:`2`),
# and the transition state (maxima). VQE reproduces the exact result (FCI) in the small
# basis (STO-3G).
#
# The activation energy barrier (:math:`E_{a}`) is defined as the difference between the
# energy of the reactant complex
# and the energy of the
# transition state.
#
# .. math:: E_{a} = E_{TS} - E_{Reactant}
#
# In this case, the activation energy barrier is
#
# .. math:: E_{a} = 0.0274 Ha = 17.24 Kcal/mol
#
# Below we show how to calculate the activation energy from the above PES.

energy_equil = min(vqe_energy)

energy_equil_2 = min([x for x in vqe_energy if x != min(vqe_energy)])

# Between the two minimas, we have the TS which is a local maxima
bond_length_index_1 = vqe_energy.index(energy_equil)
bond_length_index_2 = vqe_energy.index(energy_equil_2)

index_1 = min(bond_length_index_1, bond_length_index_2)
index_2 = max(bond_length_index_1, bond_length_index_2)

# Transition State energy
energy_ts = max(vqe_energy[index_1:index_2])

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
# in larger basis, triple and quadruple zeta basis or higher, to reach basis set
# convergence and this would significantly increase the number of qubits required.
# This is a current limitation that would go away with increasing number of logical qubits
# that would hopefully become available in future quantum computers.
#
# The reaction rate constant (k) has an exponential dependence on the activation energy barrier
# as shown in the `Arrhenius equation <https://en.wikipedia.org/wiki/Arrhenius_equation>`_:
#
# .. math:: k = Ae^{-{E_{a}}/RT}.
#
# So, in principle, if we know the constant (:math:`A`) we could calculate the rate of the reaction,
# which depends on the rate constant, the concentration of the reactants and the order of the
# reaction.
# In general, we desire our method to very accurately predict these energy differences
# and the geometries of the key intermediates.
# The plot below compares the performance of VQE with other quantum chemistry methods such as
# Hartree-Fock and Full CI.
# The PEC obtained from the quantum algorithm (VQE) overlaps with
# the `FCI <https://en.wikipedia.org/wiki/Configuration_interaction>`_ result.

##############################################################################
# .. figure:: /demonstrations/vqe_bond_dissociation/h3_comparison.png
#     :width: 90%
#     :align: center
#

##############################################################################
# A model multireference problem: :math:`Be + H_{2} \rightarrow BeH_{2}`
# -------------------------------------------------------------------------------------------
#
# In our previous examples, the Hartree-Fock state was a good approximation to the ground state
# for all points on the potential energy surface. Hence, we refer to them as single reference
# problems. However, there exist situations where more than one reference state is
# required across the potential energy surface. These are called multi-reference problems.
# A symmetric approach (:math:`C_{2v}`) of :math:`H_2` to :math:`Be` atom to form :math:`BeH_2`
# constitutes a multireference problem. [#purvis1983]_ It needs two different
# Hartree-Fock Slater determinants to qualitatively describe the full potential energy suface
# for the transformation. This is to say that one Slater determinant is a good reference
# for one half of the PES while another determinant is good reference for the rest of PES.
#
# Here, we construct the potential energy surface for this reaction and see how
# classical and quantum computing approaches built on single HF reference perform.
# Once we have solved the mean-field HF equations, we obtain the molecular orbitals
# (:math:`1a,2a,3a,1b ...`) which are then occupied to obtain two principal states
# :math:`1a^{2} 2a^{2} 3a^{2}` and :math:`1a^{2} 2a^{2} 1b^{2}`.
#
#
# .. figure:: /demonstrations/vqe_bond_dissociation/beh2_movie.gif
#     :width: 70%
#     :align: center
#
#
# To figure out the reaction coordinate for the
# approach of the :math:`H_2` molecule to the Beryllium atom, we refer to the work by
# Coe et al. [#coe2012]_
# We fix the Beryllium atom at the origin and the coordinates for the hydrogen atoms
# are given by :math:`(x, y, 0)` and :math:`(x, −y, 0)`, where :math:`y = 2.54 − 0.46x`
# and :math:`x \in [1, 4]`. All distances are in Bohr.
# Now, it's your turn to generate the potential energy surface. You could follow from our
# previous examples. All you need to do is to traverse along the specified reaction coordinate
# and obtain converged VQE energies.
#
#
# Below is the PES you would be able to generate. We have the HF and FCI curves plotted for
# comparison. We see a sharp maximum which is
# actually the result of a sudden switch in the underlying Hartree-Fock reference.
# The performance of VQE depends on the active space chosen. For reference, we have plotted
# the VQE results when the number of active orbitals is constrained to :math:`3` spatial orbitals
# which is equal to :math:`6` spin orbitals. As a simple exercise, try increasing
# the number of active orbitals and see how the performance of our VQE circuit changes.
# You would notice that our VQE circuit
# reproduces the Full CI result shown below if we increase the active orbitals to include
# all the unoccupied orbitals i.e. 12 spin orbitals in total.
#
#
# .. figure:: /demonstrations/vqe_bond_dissociation/H2_Be.png
#     :width: 90%
#     :align: center
#
#
# To summarize, we looked at three different chemical reactions and constructed
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
