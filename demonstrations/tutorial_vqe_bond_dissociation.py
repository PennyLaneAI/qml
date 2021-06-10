r"""
Modelling chemical reactions on a quantum computer
====================================================

.. meta::
    :property="og:description": Construct potential energy surfaces for chemical reactions
    :property="og:image": https://pennylane.ai/qml/_images/pes_h2.png

.. related::
   tutorial_quantum_chemistry Quantum Chemistry with PennyLane
   tutorial_vqe Variational Quantum Eigensolver

*Author: PennyLane dev team. Posted: 25 May 2021. Last updated: 25 May 2021.*

The term "chemical reaction" is another name for the transformation of molecules -- the breaking and 
forming of bonds. Each reaction is characterized by an energy cost that determines
the likelihood that the reaction occurs. These energy landscapes are the key to understanding,
at the deepest possible level, how chemical reactions occur.

In this tutorial, you will learn how to use PennyLane to simulate
chemical reactions by constructing potential energy surfaces for general
molecular transformations. In the process, you will learn how quantum computers can be used
to calculate activation energy barriers, reaction energies, and reaction rates. As illustrative
examples, we use tools implemented in PennyLane to study diatomic bond dissociation and reactions
involving the exchange of hydrogen atoms.

.. _label_h2_pes:
.. figure:: /demonstrations/vqe_bond_dissociation/h2_pes_pictorial.png
   :width: 90%
   :align: center
   
   Potential energy surface depicting single bond dissociation in hydrogen molecule.


##############################################################################

Potential Energy Surfaces 
---------------------------------------------------------------------

`Potential energy surfaces (PES) <https://en.wikipedia.org/wiki/Potential_energy_surface>`_
are energy landscapes describing the equilibrium energy of molecules for different positions of
its constituent atoms. The concept originates from the fact that nuclei are much heavier than
electrons and thus move more slowly, allowing us to decouple their motion. We can then solve for
the electronic wavefunction with the nuclei clamped to their respective positions. This results
in a separation of the nuclear and electronic parts of the Schrödinger equation, meaning we only
need to solve the electronic equation:

.. math:: H(R)|\Psi \rangle =  E|\Psi\rangle.

Thus arises the concept of the electronic energy of a molecule, :math:`E(R)`,
as a function of nuclear coordinates :math:`R`. The energy :math:`E(R)` is the expectation value
of the Hamiltonian :math:`H(R)`: :math:`E(R)=\langle \Psi_0|H(R)|\Psi_0\rangle` taken over the
ground state :math:`|\Psi_0(R)`. The potential energy surface is
precisely this function :math:`E(R)`, connecting energies to configurations of the molecule. It
gives us a visual tool to understand chemical reactions by associating
stable molecules (reactants and products) with local minima, transition states with peaks,
and by identifying the possible routes for a chemical reaction to occur.

To build the potential energy surface, we solve the electronic Schrödinger
equation for fixed positions of the nuclei, and subsequently move them in incremental steps
while solving the Schrödinger equation at each configuration.
The obtained set of energies corresponds to a grid of nuclear positions and the plot of
:math:`E(R)` gives rise to the potential energy surface. Time to construct them!

##########################################################

Bond dissociation in a Hydrogen molecule 
---------------------------------------------------------

We begin with the simplest of molecules: :math:`H_2`. 
The formation (or breaking) of the :math:`H-H` bond is also the simplest
of all reactions:

.. math:: H_2 \rightarrow H + H.

Using a minimal `basis set <https://en.wikipedia.org/wiki/STO-nG_basis_sets>`_,
this molecular system can be described by two electrons in four
spin-orbitals. When mapped to a qubit representation, we need a total of four qubits.
The `Hartree-Fock (HF) state is represented as :math:`|1100\rangle`, where the two
lowest-energy orbitals are occupied, and the remaining two are unoccupied.

We design a quantum circuit consisting of :class:`~.pennylane.SingleExcitation` and
:class:`~.pennylane.DoubleExcitation` gates applied to the Hartree-Fock state. This circuit
will be optimized to prepare ground states for different configurations of the molecule.
"""

import pennylane as qml
from pennylane import qchem

# Hartree-Fock state
hf = qml.qchem.hf_state(electrons=2, orbitals=4)

def circuit(params, wires):
    # Prepare the HF state: |1100>
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
    qml.SingleExcitation(params[1], wires=[0, 2])
    qml.SingleExcitation(params[2], wires=[1, 3])


##############################################################################
# To construct the potential energy surface, we vary the geometry of the molecule. We keep
# an :math:`H` atom fixed at the origin and vary only the
# :math:`x`-coordinate of the other atom. The potential energy
# surface is then a one-dimensional function depending only on the bond length, i.e., the separation
# between the atoms. For each value of the bond length, we construct the corresponding
# Hamiltonian, optimize the circuit using gradient descent, and obtain the ground-state energy,
# allowing us to build the potential energy surface. We vary the bond length in the range
# :math:`0.5` to :math:`5.0` `Bohrs <https://en.wikipedia.org/wiki/Bohr_radius>`_ in steps of
# :math:`0.1` Bohr. This covers the point where the :math:`H-H` bond is formed
# (equilibrium bond length) and the point where the bond is broken, which occurs when the atoms
# are far away from each other.

# atomic symbols defining the molecule
symbols = ['H', 'H']

# list to store energies
energies = []

# set up a loop to change bond length
r_range = np.arange(0.5, 5.0, 0.1)

# keeps track of which bond length we are at
pes_point = 0

##############################################################################
# We build the Hamiltonian using the :func:`~.pennylane_qchem.qchem.molecular_hamiltonian`
# function, and use standard Pennylane techniques to optimize the circuit. To speed up the
# simulation, it is helpful to start from the optimized parameter of previous points.#

for r in r_range:
    coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, r])

    # Obtain the qubit Hamiltonian 
    H, qubits = qchem.molecular_hamiltonian(symbols, coordinates)

    # define the device, cost function, and optimizer
    dev = qml.device("default.qubit", wires=qubits)
    cost_fn = qml.ExpvalCost(circuit, H, dev)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)

    # initialize the gate parameters
    params = np.zeros(3)

    # initialize with converged parameters from previous point
    if pes_point > 1:
        params = params_old

    prev_energy = 0.0
    for n in range(50):
        # perform optimization step
        params, energy = opt.step_and_cost(cost_fn, params)

        if np.abs(energy - prev_energy) < 1e-6:
            break
        prev_energy = energy

    # store the converged parameters
    params_old = params
    pes_point = pes_point + 1

    energies.append(energy)


##############################################################################
# Let's plot the results

import matplotlib.pyplot as plt

# Energy as a function of internuclear distance
fig, ax = plt.subplots()
ax.plot(r_range, energies)

ax.set(
    xlabel="Bond length (Bohr)",
    ylabel="Total energy (Hartree)",
    title="Potential energy surface for H$_2$ dissociation",
)
ax.grid()
plt.show()


##############################################################################
# This is the potential energy surface for the dissociation of a hydrogen molecule into
# two hydrogen atoms! Let's understand its usefulness.
#
# In a diatomic molecule such as :math:`H_2`, the potential energy surface can be used to obtain
# the equilibrium bond length --- the distance between the two atoms that minimizes the total
# electronic energy. This is simply the minimum of the curve. We can also obtain the bond
# dissociation energy: the difference in energy of the system at
# equilibrium and where the atoms are far apart. At sufficiently large separations, the atoms no
# longer form a molecule, which is therefore dissociated.
#
# We can use our results to compute the equilibrium bond length and the bond dissociation energy:

# equilibrium energy
e_eq = min(energies)
# energy at dissociation
e_dis = energies[-1]

# Bond dissociation energy is their difference
bond_energy = e_dis - e_eq

# Equilibrium bond length
bond_length_index = energies.index(e_eq)
bond_length = r_range[bond_length_index]

print(f"The equilibrium bond length is {bond_length:.1f} Bohrs")
print(f"The H-H bond dissociation energy is {bond_energy:.6f} Hartrees")


##############################################################################
# These estimates can be improved
# by using bigger basis sets and extrapolating to the complete basis set limit [#motta2020]_.
# We must also note that our results are subject to the grid size of the span of interatomic
# distances considered. The finer the grid size, the better the estimates.
#
# .. note::
#
#     Did you notice a trick we used to speed up the calculations? The converged
#     gate parameters for a particular geometry on the PES are used as the initial guess for the VQE
#     calculation at the adjacent geometry. With a better guess, the VQE iterations converge
#     relatively quickly and we save considerable time.

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
# As in the previous case, all electrons and all orbitals are considered.


# Molecular parameters

symbols = ["H", "H", "H"]
multiplicity = 2
basis_set = "sto-3g"


##############################################################################
# We setup the PES loop incrementing the :math:`H(1)-H(2)` distance from :math:`1.0`
# to :math:`3.0` Bohrs in steps of :math:`0.1` Bohr.
# We use to obtain the list of allowed single and double excitations out of the HF state.
# The way we build the VQE circuit for this system is generic and can be used to
# build circuits for any molecular system. We then repeat the calculations over the
# whole range of PES and print the converged energies of the whole system at each step.

energies = []
pes_point = 0

# define circuit

# get all the singles and doubles excitations
electrons = 3
orbitals = 6
singles, doubles = qchem.excitations(electrons, orbitals)

hf = qml.qchem.hf_state(electrons, orbitals)

def circuit(params, wires):
    qml.BasisState(hf, wires=wires)
    for i in range(0, len(doubles)):
        qml.DoubleExcitation(params[i], wires=doubles[i])
    for j in range(0, len(singles)):
        qml.SingleExcitation(params[j + len(doubles)], wires=singles[j])

# loop to change reaction coordinate
r_range = np.arange(1.0, 3.0, 0.1).round(1)
for r in r_range:

    coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, r, 0.0, 0.0, 4.0])

    H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, mult=multiplicity)

    dev = qml.device("default.qubit", wires=qubits)
    cost_fn = qml.ExpvalCost(circuit, H, dev)
    opt = qml.GradientDescentOptimizer(stepsize=1.5)

    len_params = len(singles) + len(doubles)
    params = np.zeros(len_params)

    # if this is not the first geometry point on PES, initialize with converged parameters
    # from previous point
    if pes_point > 1:
        params = params_old

    prev_energy = 0.0

    for n in range(60):

        params, energy = opt.step_and_cost(cost_fn, params)

        if np.abs(energy - prev_energy) < 1e-6:
            break

        prev_energy = energy

    # store the converged parameters
    params_old = params
    pes_point = pes_point + 1

    print("At r = {:.1f} Bohrs, number of VQE Iterations required is {:}".format(r, n))
    energies.append(energy)

# tabulate
list_dist_energy = list(zip(r_range, energies))
# Converting list into pandas Dataframe
df = pd.DataFrame(list_dist_energy, columns=["H(1)-H(2) distance (Bohr)", "Energy (Hartree)"])
# display table
print(df)

##############################################################################
#
# After tabulating our results, we plot the energy as a function of distance between atoms
# :math:`1` and :math:`2`, and thus we have the potential energy curve for
# this reaction. The minima in the curve represent the VQE estimate of the energy and geometry
# of reactants and products respectively, while the transition state is represented by the
# local maximum.
# In general, we would like our method (VQE) to provide
# a good estimate of the energies of the reactants (minimum :math:`1`), products (minimum :math:`2`),
# and the transition state (maximum). We shall revisit the VQE results later and find that it
# reproduces the exact energies (FCI) in the chosen basis (STO-3G) throughout the PES.

# plot the PES
fig, ax = plt.subplots()
ax.plot(r_range, energies)

ax.set(
    xlabel="H(1)-H(2) distance (Bohr)",
    ylabel="Total energy (Hartree)",
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
# We show how to calculate the activation energy of the hydrogen exchange reaction.

# Energy of the reactants and products - two minima on the PES
energy_equil = min(energies)
energy_equil_2 = min([x for x in energies if x != min(energies)])

# Between the two minimas, we have the TS which is a local maxima
bond_length_index_1 = energies.index(energy_equil)
bond_length_index_2 = energies.index(energy_equil_2)

# Reaction coordinate at the two minimas
index_1 = min(bond_length_index_1, bond_length_index_2)
index_2 = max(bond_length_index_1, bond_length_index_2)

# Transition state energy
energy_ts = max(energies[index_1:index_2])

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
# in a larger basis to reach basis set
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
# The PES obtained from the quantum algorithm (VQE) overlaps with
# the FCI result. This is a
# validation of the accuracy of the energy estimates through our VQE circuit --- we exactly
# reproduce the absolute energies of the reactants, transition state and product.

##############################################################################
# A model multi-reference problem: :math:`Be + H_{2} \rightarrow BeH_{2}`
# -------------------------------------------------------------------------------------------
#
# In our previous examples, the Hartree-Fock state was a good approximation to the exact
# ground state for all points on the potential energy surface:
#
# .. math:: \langle \Psi_{HF}| \Psi_{exact}\rangle \simeq 1.
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
# Let us first understand the problem in more detail. Once we have solved the HF
# equations, we obtain the molecular orbitals
# (:math:`1a,2a,3a,1b ...`) which are then occupied to obtain two principal states
# :math:`1a^{2} 2a^{2} 3a^{2}` and :math:`1a^{2} 2a^{2} 1b^{2}`.
# These are the two different
# reference states needed to qualitatively describe the full potential energy surface
# for this reaction. Prior studies have found that one of these states is a good reference
# for one half of the PES while another is good reference for the rest of PES [#purvis1983]_.
# To figure out the reaction coordinate for the
# approach of the :math:`H_2` molecule to the :math:`Be` atom, we refer to the work by
# Coe et al. [#coe2012]_.
# We fix the :math:`Be` atom at the origin and the coordinates for the hydrogen atoms
# are given by :math:`(x, y, 0)` and :math:`(x, −y, 0)`, where :math:`y = 2.54 − 0.46x`
# and :math:`x \in [1, 4]`. All distances are in Bohr.
#
#
# We set this last section as a challenge problem for the reader. We would like you to
# generate the potential energy surface for this reaction and compare how
# the performance of VQE with FCI results.
# All you need to do is to traverse along the
# specified reaction coordinate, generate molecular geometries and obtain the converged
# VQE energies. The code is similar and follows from our previous examples.
#
#
# Below is the PES plot you would be able to generate. We have provided the HF and FCI curves for
# comparison. A sharp maximum could be seen in these curves which is
# due to a sudden switch in the underlying Hartree-Fock reference at the specific reaction
# coordinate.
# The performance of VQE depends on the active space chosen, i.e., the number of electrons and
# orbitals that are being considered. For reference, we have included
# the VQE results when the number of active orbitals is constrained to :math:`3` spatial orbitals,
# which is equal to :math:`6` spin orbitals. As an additional exercise, try increasing
# the number of active orbitals and see how the performance of our VQE circuit changes.
# Does your VQE circuit
# reproduce the FCI result shown below if we increase the active orbitals to include
# all the unoccupied orbitals, i.e., 12 spin orbitals in total?
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
