""".. _toric_code:
Toric Code Topology
===================


.. meta::
    :property="og:description": Investigation of the toric code degenerate ground state and anyon excitations
    :property="og:image": https://pennylane.ai/qml/_images/types_of_loops.png

*Author: Christina Lee. Posted: xx June 2022*

Introduction
------------

The `toric code model <https://arxiv.org/abs/quant-ph/9707021>`__ is a treasure trove of interesting physics and
mathematics. The model sparked the development of the error-correcting `surface codes 
<https://arxiv.org/pdf/1208.0928.pdf>`__ , an essential category of error correction models. But why is the
model so useful for error correction?

To answer that question, we can to delve into mathematics and condensed matter physics.
Viewing the model as a description of spins in a really exotic magnet allows us to start
analyzing the model as a material. What kind of material is it? The toric code is an
example of a topological state of matter.

A state of matter, or phase, cannot become a different phase without some kind of discontinuity
in the physical properties as coefficients in the Hamiltonian change. For example, ice cannot
become water without a discontinuity in density as the temperature changes. The ground state
of a **topological** state of matter cannot be smoothly deformed to a non-entangled state
without a phase transition. Entanglement, and more critically *long-range* entanglement,
is a key hallmark of a topological state of matter.

This exotic phase cannot be detected by local measurements but can only be measured across the 
entire system. To better consider this type of property, consider the parity of the number of 
dancers on a dance floor. Does everyone have a partner, or is there an odd person out? To 
measure that, we have to look at the entire system.

Topology is the study of global properties that are preserved under continuous 
deformations. For example, a coffee cup is equivalent to a donut because they 
both have a single hole. More technically, they both have an 
`Euler characteristic <https://en.wikipedia.org/wiki/Euler_characteristic>`__ of zero. 
When we zoom to a local patch, both a sphere and a torus look the same. Only by considering 
the object as a whole can you detect the single hole.

.. figure:: ../demonstrations/toric_code/torus_to_cup.png
    :align: center
    :width: 70%

In this demo, we will be looking at the degenerate ground state and the
excitations of the toric code model. The toric code was initialized
proposed in “Fault-tolerant quantum computation by anyons” by Kitaev,
and this demo was inspired by “Realizing topologically ordered states on
a quantum processor” by K. J. Satzinger et al. For further reading, I
recommend “Quantum Spin Liquids” by Lucile Savary and Leon Balents. (add
links and proper citations and stuff)

The Model
---------

What is the source of all this interesting physics? The Hamiltonian is:

.. math::
   \mathcal{H} = -\sum_s S_s - \sum_p P_p

.. math::
   S_s = \prod_{i \in s} Z_i \quad P_p = \prod_{j \in p} X_j.

In the literature, the :math:`S_s` terms are called the “star”
operators, and the :math:`P_p` terms are called the “plaquette”
operators. Each star :math:`s` and plaquette :math:`p` is a group of 4 sites.

In the most common formulation of the model, sites live on
the edges of a square lattice. In this formulation, the “plaquette” operators are products of Pauli X operators on all the sites
in a square, and the "star" operators are products of Pauli Z operators on all the sites bordering a vertex.

The model can also be viewed as a checkerboard of alternating square
types. In this formulation, all sites :math:`i` and :math:`j` are the vertices
of a square lattice.
Each square is a group of four sites, and adjacent squares
alternate between the two types of groups. Since the groups on
this checkerboard no longer look like stars and plaquettes, we will call
them the “Z Group” and “X Group” operators in this tutorial.

.. figure:: ../demonstrations/toric_code/stars_plaquettes2.png
    :align: center
    :width: 70%

We will be embedding the lattice on a torus via periodic boundary
conditions. Periodic boundary conditions basically “glue” the bottom of
the lattice to the top of the lattice and the left to the right.

This matching is done with modular arithemetic. Any site at ``(x,y)`` is
the same as a site at ``(x+width, y+height)``.

.. figure:: ../demonstrations/toric_code/converting_to_torus.png
    :align: center
    :width: 70%

On to some practical coding!
"""

import pennylane as qml
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from itertools import product
from dataclasses import dataclass

import numpy as np

np.set_printoptions(suppress=True)

height = 4
width = 6

all_sites = [(i, j) for i, j in product(range(width), range(height))]

######################################################################
# For our wire labels, we will be using an `Immutable Data Class <https://realpython.com/python-data-classes/#immutable-data-classes>`__ .
# PennyLane allows wire labels to be any **hashable** object, but iterable wire labels are currently not supported.
# Therefore we use a frozen dataclass to easily represent individual wires by a row and column position.
#

@dataclass(frozen=True)
class Wire:
    i:int
    j:int

example_wire = Wire(0,0)
print("Example wire: ", example_wire)
print("At coordinates: ", example_wire.i, example_wire.j)


######################################################################
# Setting up Operators
# --------------------
#
# For each type of group operator (X and Z) we will have two different
# lists, the “sites” and the “ops”. The “sites” are tuples and will include virtual
# sites off the edge of the lattice that match up with locations on the
# other side. For example, the site ``(6, 1)`` denotes the real location
# ``(0,1)``. We will use the ``zgroup_sites`` and ``xgroup_sites`` lists
# to help us view the measurements of the corresponding operators.
#
# The "ops" list will contain the tensor observables. We will later
# take the expectation value of each tensor.
#

mod = lambda s: Wire(s[0] % width, s[1] % height)

zgroup_sites = []  # list of sites in each group
zgroup_ops = []  # list of operators for each group

for x, y in product(range(width // 2), range(height)):

    x_shift = 2 * x + (y + 1) % 2

    sites = [(x_shift, y), (x_shift + 1, y), (x_shift + 1, y + 1), (x_shift, y + 1)]

    op = qml.operation.Tensor(*(qml.PauliZ(mod(s)) for s in sites))

    zgroup_sites.append(sites)
    zgroup_ops.append(op)

print("First set of sites: ", zgroup_sites[0])
print("First operator: ", zgroup_ops[0])


######################################################################
# We will later use the X Group operator sites to prepare the ground
# state, so the order here is important. One group needs a slightly
# different order in due to interference with the periodic boundary
# condition.
#

xgroup_sites = []
xgroup_ops = []
for x, y in product(range(width // 2), range(height)):
    x_shift = 2 * x + y % 2

    sites = [(x_shift + 1, y + 1), (x_shift, y + 1), (x_shift, y), (x_shift + 1, y)]

    if x == 2 and y == 1:  # change order for state prep later
        sites = sites[1:] + sites[0:1]

    op = qml.operation.Tensor(*(qml.PauliX(mod(s)) for s in sites))

    xgroup_sites.append(sites)
    xgroup_ops.append(op)

######################################################################
# How can we best visualize these groups of four sites?
#
# We use matplotlib to show each group of four sites as a Polygon patch,
# colored according to the type of group. The ``misc_plot_formatting`` function
# just performs some minor styling improvements that will
# be performed repeated throughout this demo. The dotted horizontal lines and 
# dashed vertical lines denote where we glue our boundaries together.
#


def misc_plot_formatting(fig, ax):
    plt.hlines([-0.5, height - 0.5], -0.5, width - 0.5, linestyle="dotted", color="black")
    plt.vlines([-0.5, width - 0.5], -0.5, height - 0.5, linestyle="dashed", color="black")
    plt.xticks(range(width + 1), [str(i % width) for i in range(width + 1)])
    plt.yticks(range(height + 1), [str(i % height) for i in range(height + 1)])

    for direction in ["top", "right", "bottom", "left"]:
        ax.spines[direction].set_visible(False)

    return fig, ax


fig, ax = plt.subplots()
fig, ax = misc_plot_formatting(fig, ax)

for group in xgroup_sites:
    x_patch = ax.add_patch(Polygon(group, color="lavender", zorder=0))

for group in zgroup_sites:
    z_patch = ax.add_patch(Polygon(group, color="mistyrose", zorder=0))

plt_sites = ax.scatter(*zip(*all_sites))

plt.legend([x_patch, z_patch, plt_sites], ["XGroup", "ZGroup", "Site"], loc="upper left")

plt.show()


######################################################################
# The Ground State
# ----------------
#
# While individual X and Z operators do not commute with each other, the X Group and Z Group operators
# do:
#
# .. math::
#
#      [S_s, P_p] = 0.
#
# Since they commute, the wavefunction can be an eigenstate of each group operator individually. To minimize
# the energy of the Hamiltonian on the system as a whole, we can just minimize the contribution of each group operator.
# Due to the negative coefficients in the Hamiltonian, we need to maximize the
# eigenvalue an individual operator to minimize the contribution of its term.
# The maximum eigenvalue for each operator is simply one. We can turn this
# into a constraint on our ground state:
#
# .. math::
#
#       S_s |G \rangle = +1 |G \rangle \qquad \qquad P_p | G \rangle = +1 |G\rangle.
#
# The wavefunction:
#
# .. math::
#
#    | G \rangle =  \prod_{p} \frac{\mathbb{I} + P_p}{\sqrt{2}} |00\dots 0\rangle = \prod_{p} U_p |00\dots 0 \rangle,
#
# where :math:`P_p` (plaquette) denotes an X Group operator, is a such a state.
#
# .. note::
#
#    For extra understanding, confirm that this ground state obeys the constraints using pen and paper.
#
# This formula is a product of unitaries :math:`U_p`. If we can figure out how to apply a single
# :math:`U_p` using a quantum computer's operations, we can simply apply that decomposition
# for every :math:`p` in the product.
#
# To better understand how to decompose :math:`U_p`, let’s write
# it concretely for a single group of four qubits:
#
# .. math::
#    U |0000 \rangle =
#    \frac{\left(\mathbb{I} + X_1 X_2 X_3 X_4 \right)}{\sqrt{2}} |0000 \rangle
#    = \frac{1}{\sqrt{2}} \left( |0000\rangle + |1111\rangle \right)
#
# This `generalized GHZ state <https://en.wikipedia.org/wiki/Greenberger–Horne–Zeilinger_state>`__
# can be prepared with a Hadamard and 3 CNOT
# gates:
#
# .. figure:: ../demonstrations/toric_code/generalized_ghz_draw.png
#     :align: center
#     :width: 50%
#
# This decomposition for :math:`U_p` holds only when the initial Hadamard
# qubit begins in the :math:`|0\rangle` state, so we need to be careful in
# choosing which qubit to apply the initial Hadamard gate to. This is the
# reason why we rotated the order for a single X Group on the right border
# earlier.
#
# We will also not need to prepare the final X Group that contains the
# four edges of the lattice, as it will already be prepared by preparation
# of the surrounding groups.
#
# Now let’s actually put these together into a circuit!
#

dev = qml.device("lightning.qubit", wires=[Wire(*s) for s in all_sites])


def state_prep():
    for op in xgroup_ops[0:-1]:
        qml.Hadamard(op.wires[0])
        for w in op.wires[1:]:
            qml.CNOT(wires=[op.wires[0], w])


@qml.qnode(dev, diff_method=None)
def circuit():
    state_prep()
    return [qml.expval(op) for op in xgroup_ops+zgroup_ops]


######################################################################
# From this QNode, we can calculate the expectation values of the
# individual operators and the total energy of the system.
#

n_xgroups = len(xgroup_ops)
separate_expvals = lambda expvals: (expvals[:n_xgroups], expvals[n_xgroups:])

xgroup_expvals, zgroup_expvals = separate_expvals(circuit())

E0 = -sum(xgroup_expvals) - sum(zgroup_expvals)

print("X Group expectation values", xgroup_expvals)
print("Z Group expectation values", zgroup_expvals)
print("Total energy: ", E0)


######################################################################
# Excitations
# -----------
#
# Quasiparticles allow physicists to describe complex
# systems as interacting particles in a vacuum. Common examples of
# quasiparticles include electrons and holes in semiconductors, phonons,
# and magnons.
#
# Imagine trying to describe the traffic on a road. We could either:
#
# -  explicitly enumerate the location of each vehicle,
# -  describe the locations and severities of traffic jam.
#
# The first option provides the complete information about the system but
# is much more difficult to work with. For most purposes, we can just work
# with information about how the traffic deviates from a baseline. In
# semiconductors, we don’t write out the wavefunction for every single
# electron. We instead use electrons and holes. Neither quasiparticle
# electrons or holes are fundamental particles like an electron or
# positron in a vacuum. Instead, they are useful descriptions of how the
# wavefunction differs from its ground state.
#
# While the electrons and holes of a metal behave just like electrons and
# positrons in a vacuum, some condensed matter systems contain
# quasiparticles that cannot or do not exist as fundamental particles.
#
# The excitations of the toric code are one such example. To find these quasiparticles,
# we look at states that are *almost* the ground state, such as the ground state
# with a single operator applied to it.
#
# Suppose we apply a pertubation to the ground state in the form of a single
# X gate at location :math:`i` :
#
# .. math::
#
#    | \phi \rangle = X_i | G \rangle.
#
# Two Z group operators :math:`S` contain individual Z operators at that
# same site :math:`i`:. The noise term :math:`X_i` will anti-commute with both
# of these group operators:
#
# .. math::
#
#    S_s X_i = \left( Z_i Z_a Z_b Z_c \right) X_i = - X_i S_s.
#
# Using this relation, we can determine the eigenvalue of the Z group
# operators on the pertubed state:
#
# .. math::
#
#    S_s |\phi\rangle = S_s X_i |G\rangle = - X_i S_s |G\rangle = - X_i |G\rangle = - |\phi\rangle.
#
# :math:`S_s` now has an eigenvalue of :math:`-1`.
#
# Applying a single X operator noise term changes the eigenvalues of *two* Z group operators.
#
# This analysis repeats for the effect of a Z operator on the X Group
# eigenvalue. A single Z operator noise term changes the eigenvalues of *two*
# X group operators.
#
# Each group with a flipped eigenvalue is considered an excitation. In the
# literature, you will often see a Z Group excitation
# :math:`\langle S_s \rangle = -1` called an “electric” :math:`e` excitation and
# an X Group excitation :math:`\langle P_p \rangle = -1` called a
# “magnetic” :math:`m` excitation. You may also see inclusion of an identity
# :math:`\mathbb{I}` particle for the ground state and the combination
# particle :math:`\Psi` consisting of a single :math:`e` and a single :math:`m`
# excitation.
#
# Let’s create a QNode where we can apply these pertubations:
#


@qml.qnode(dev, diff_method=None)
def excitations(x_sites, z_sites):
    state_prep()

    for s in x_sites:
        qml.PauliX(Wire(*s))

    for s in z_sites:
        qml.PauliZ(Wire(*s))

    return [qml.expval(op) for op in xgroup_ops+zgroup_ops]


######################################################################
# What are the expectation values when we apply a single X operation?
# We see we have indeed flipped the eigenvalues for two groups.
#

single_x = [(1, 2)]

x_expvals, z_expvals = separate_expvals(excitations(single_x, []))

print("XGroup: ", x_expvals)
print("ZGroup: ", z_expvals)


######################################################################
# Instead of interpreting the state via the eigenvalues of the operators,
# we can view the state as occupation numbers of the corresponding
# quasiparticles. A group with an eigenvalue of :math:`+1` is in the
# ground state and thus has an occupation number of :math:`0`. If the
# eigenvalue is :math:`-1`, then a quasiparticle exists in that location.
#

occupation_numbers = lambda expvals: 0.5 * (1 - expvals)

def print_info(x_expvals, z_expvals):
    E = -sum(x_expvals) - sum(z_expvals)

    print("Total energy: ", E)
    print("Energy above the ground state: ", E - E0)
    print("X Group occupation numbers: ", occupation_numbers(x_expvals))
    print("Z Group occupation numbers: ", occupation_numbers(z_expvals))


print_info(x_expvals, z_expvals)


######################################################################
# Since we are going to plot the same thing many times, we can group the
# code into a function to easily call later.
#


def excitation_plot(x_excite, z_excite):
    x_color = lambda expval: "navy" if expval < 0 else "lavender"
    z_color = lambda expval: "maroon" if expval < 0 else "mistyrose"

    fig, ax = plt.subplots()
    fig, ax = misc_plot_formatting(fig, ax)

    for expval, sites in zip(x_excite, xgroup_sites):
        ax.add_patch(Polygon(sites, color=x_color(expval), zorder=0))

    for expval, sites in zip(z_excite, zgroup_sites):
        ax.add_patch(Polygon(sites, color=z_color(expval), zorder=0))

    handles = [
        Patch(color="navy", label="X Group -1"),
        Patch(color="lavender", label="X Group +1"),
        Patch(color="maroon", label="Z Group -1"),
        Patch(color="mistyrose", label="Z Group +1"),
        Patch(color="navy", label="Z op"),
        Patch(color="maroon", label="X op"),
    ]

    plt.legend(handles=handles, ncol=3, loc="lower left")

    return fig, ax


fig, ax = excitation_plot(x_expvals, z_expvals)

ax.scatter(*zip(*single_x), color="maroon", s=100)

plt.show()


######################################################################
# Now what if we apply a Z operation instead at the same site? We instead
# get two X Group excitations.
#

single_z = [(1, 2)]

expvals = excitations([], single_z)
x_expvals, z_expvals = separate_expvals(expvals)
print_info(x_expvals, z_expvals)

######################################################################
#

fig, ax = excitation_plot(x_expvals, z_expvals)

ax.scatter(*zip(*single_z), color="navy", s=100)

plt.show()

######################################################################
# What happens if we apply the same pertubation twice at the same
# location? We regain the ground state. 
#
# The excitations of the toric code are
# Majorana particles, particles who are their own antiparticles. While
# postulated to exist in standard particle physics, Majorana particles
# have only been experimentally seen as quasiparticle excitations in
# materials.
#
# We can think of the second operation as creating another set of
# excitations at the same location that annihilate the existing particles.
#

single_z = [(1, 2)]

expvals = excitations([], single_z+single_z)
x_expvals, z_expvals = separate_expvals(expvals)

print_info(x_expvals, z_expvals)


######################################################################
# Moving Excitations and String Operators
# ---------------------------------------
#
# What if we create a second set of particles such that one of the new
# particles overlaps with an existing particle? Then one old particle and
# one new particle annihilate each other. We are left we one of the old
# particles and one new particle, so we still have two particles in total.
#
# We can think about the situation as creating a new pair of particles
# where two particles cancel each other out, but we can also view the
# application of a new pertubation as moving one of the excitations. Let’s
# see what that looks like in code:
#

two_z = [(1, 2), (2, 2)]

expvals = excitations([], two_z)
x_expvals, z_expvals = separate_expvals(expvals)

print_info(x_expvals, z_expvals)

######################################################################
#

fig, ax = excitation_plot(x_expvals, z_expvals)

ax.plot(*zip(*two_z), color="navy", linewidth=10)

plt.show()


######################################################################
# In that example we just moved an excitation a little. How about we try
# moving it even further?
#

long_string = [(1, 2), (2, 2), (3, 2), (4, 1)]

expvals = excitations([], long_string)
x_expvals, z_expvals = separate_expvals(expvals)

print_info(x_expvals, z_expvals)

######################################################################
#

fig, ax = excitation_plot(x_expvals, z_expvals)

ax.plot(*zip(*long_string), color="navy", linewidth=10)

plt.show()

######################################################################
# We end up with these strings of sites that connect pairs of particles.
#
# We can use a branch of topology called `Homotopy <https://en.wikipedia.org/wiki/Homotopy>`__
# to describe the relationship between these strings and the wavefunction.
# Two paths :math:`s_1` and :math:`s_2` are
# **homotopy equivalent** or **homotopic** if they can be continuously deformed into each
# other:
#
# .. math::
#
#    s_1 \sim s_2
#
# For the next picture, assume the red “X” is some kind of defect
# in space, like a tear in a sheet or some kind of object. The two blue
# paths are equivalent to each other because you can smoothly move one
# into the other. You cannot move the blue path into the green path
# without going through the defect, so they are not equivalent to each
# other.
#
# .. figure:: ../demonstrations/toric_code/homotopy.png
#     :align: center
#     :width: 40%
#
# We can divide the set of all possible paths into *Homotopy classes*. A homotopy class
# is an `equivalence class <https://en.wikipedia.org/wiki/Equivalence_class>`__ under
# homotopy. Every member of the same homotopy class can be deformed into every other member of the
# same class, and
# members of different homotopy classes cannot be deformed into each other. All the
# homotopy classes for a given space :math:`S` form its
# `homotopy group <https://en.wikipedia.org/wiki/Homotopy_group>`__, denoted by :math:`\pi_1(S)`.
#
# What if we decided to move the particle to it’s final location via a
# different route?
#
# The below string gets us to the exact same final state.
# Only the homotopy class of the path used to create the excitations influences
# the occupation numbers and total energy of the state.
# As long as the endpoints are the same and the path doesn’t
# wrap around the torus or other particles, the details do not
# impact any observables.
#

equivalent_string = [(1, 2), (2, 1), (3, 1), (4, 1)]

expvals = excitations([], equivalent_string)
x_expvals, z_expvals = separate_expvals(expvals)

print_info(x_expvals, z_expvals)

######################################################################
#

fig, ax = excitation_plot(x_expvals, z_expvals)

ax.plot(*zip(*equivalent_string), color="navy", linewidth=10)

plt.show()


######################################################################
# Looping the torus
# ^^^^^^^^^^^^^^^^^
#
# We can also have a loop of operations that doesn’t create any new
# excitations. The loop creates a pair, moves one around in a circle, and
# then annihilates the two particles again.
#

contractable_loop = [(1, 1), (2, 1), (3, 1), (4, 1), (4, 2), (3, 3), (2, 3), (1, 2)]

expvals = excitations(contractable_loop, [])
x_expvals, z_expvals = separate_expvals(expvals)
print_info(x_expvals, z_expvals)

######################################################################
#

fig, ax = excitation_plot(x_expvals, z_expvals)

ax.plot(*zip(*contractable_loop), color="maroon", linewidth=10)

plt.show()

######################################################################
# The loop doesn’t affect the positions of any excitations, but does it
# affect the state at all?
#
# To answer that question, we will look at the probabilities instead of
# the expectation values of tensor observables.
#


@qml.qnode(dev, diff_method=None)
def probs(x_sites, z_sites):
    state_prep()

    for s in x_sites:
        qml.PauliX(Wire(*s))

    for s in z_sites:
        qml.PauliZ(Wire(*s))

    return qml.probs(wires=[Wire(*s) for s in all_sites])


null_probs = probs([], [])
contractable_probs = probs(contractable_loop, [])

print("Are the probabilities equal? ", np.allclose(null_probs, contractable_probs))


######################################################################
# This result is explained, once again, by the fact that the toric code
# only cares about the homotopy class of the paths. All paths we
# can smoothly deform into each other will give the same result. The
# contractible loop can be smoothly deformed away to nothing, so the
# state with the contractible loop is the same state as a state with no loop.
#


######################################################################
# On the torus, we have four types of unique paths:
#
# -  The trivial path that contracts to nothing
# -  A horizontal loop around the boundaries
# -  A vertical loop around the boundaries
# -  A loop around both the horiztonal and vertical boundaries
#
# .. figure:: ../demonstrations/toric_code/types_of_loops.png
#     :align: center
#     :width: 50%
#
# Each of these paths represents a member of the first homotopy group
# of the torus: :math:`\pi_1(T) = \mathbb{Z}^2`.
#
# All of these do not create any net excitations, so the wavefunction
# remains in the ground state.
#

horizontal_loop = [(i, 1) for i in range(width)]
vertical_loop = [(1, i) for i in range(height)]

expvals = excitations(horizontal_loop + vertical_loop, [])
fig, ax = excitation_plot(*separate_expvals(expvals))

ax.plot(*zip(*horizontal_loop), color="maroon", linewidth=10)
ax.plot(*zip(*vertical_loop), color="maroon", linewidth=10)

plt.show()


######################################################################
# We can compute the probabilities for each of these four types of loops:
#

null_probs = probs([], [])
horizontal_probs = probs(horizontal_loop, [])
vertical_probs = probs(vertical_loop, [])
combo_probs = probs(horizontal_loop + vertical_loop, [])


######################################################################
# While both X and Z operations can change the group operator eigenvalues
# and create quasiparticles, only X operators can change the probability
# distribution. Applying a Z operator would only rotate the phase of the
# state and not change any amplitudes. Hence we only use loops of X
# operators in this section. I encourage you to try this analysis with
# loops of Z operators to confirm that they do not change the probability
# distribution.
#
# We can compare the original state and one with a horizontal loop to see
# if the probability distributions are different:
#

print("Are the probabilities equal? ", qml.math.allclose(null_probs, horizontal_probs))
print("Is this significant?")
print("Maximum difference in probabilities: ", max(abs(null_probs - horizontal_probs)))
print("Maximum probability: ", max(null_probs))

######################################################################
# The size of the difference 
# So this isn’t just random fluctuations and errors.
#
# That was just comparing a horizontal “x” loop with the initial ground
# state. How about the other two types of loops? Let’s loop over all
# combinations of two probability distributions to see if any match.
#

names = ["null", "x", "y", "combo"]
all_probs = [null_probs, horizontal_probs, vertical_probs, combo_probs]

print("\t" + "\t".join(names))

for name, probs1 in zip(names, all_probs):

    comparisons = (format(np.allclose(probs1, probs2), ".0f") for probs2 in all_probs)
    print(name, "\t", "\t".join(comparisons))


######################################################################
# This shows us we have four distinct ground states. More importantly,
# these ground states are separated from each other by long-range
# operations. We have to perform a loop of operations across the entire
# lattice in order to switch ground state.
#
# This four way degeneracy is the source of the error correction in the
# toric code. Instead 24 qubits, we work with 2 logical qubits (4 states)
# that are cleanly separated from each other by topological operations.
#
# .. note::
#
#    I encourage dedicated readers to explore what happens when a path
#    loops the same boundaries twice.
#
# In this section, we've seen that the space of ground states is directly
# equivalent to the first homotopy group of the lattice it is defined on.
# Since our lattice is on a torus, the space of ground states is :math:`\pi_1(T) = \mathbb{Z}^2`.
# What if we defined the model on a differently shaped lattice? Then the space of the 
# ground state would change to reflect the first homotopy group of that space.
# For example, if the model was defined on a sphere, then only a single unique ground state would exist.
# Adding defects like missing sites to the lattice also changes the topology. Error correction
# with the toric code often uses missing sites to add additional logical qubits.
#
#
#


######################################################################
# Mutual Exchange Statistics
# --------------------------
#
# The hole in the center of the donut isn’t the only thing that prevents
# paths from smoothly deforming into each other. We don’t yet know if we
# can deform paths past other particles.
#
# When one indistinguishable fermion of spin 1/2 orbits another fermion of
# the same type, the combined wavefunction picks up a relative phase of negative
# one. When fermions of different types orbit each other, the state is
# unchanged. For example, if an electron goes around a proton and comes
# back to the same spot, the wavefunction is unchanged. If a boson orbits
# around a different type of boson, again the wavefunction is unchanged.
#
# What if a particle went around a different type of particle and
# everything picked up a phase? Would it be a boson or a fermion?
#
# It would be something else entirely: an anyon. An anyon is anything that
# doesn’t cleanly fall into the boson/ fermion categorization of particles.
#
# While the toric code itself is just an extremely useful mathematical
# model, anyons actually exist in real materials. For example, the
# fractional quantum Hall systems have anyonic particles with spin
# :math:`1/q` for different integers :math:`q`.
#
# To measure the exchange statistics of a Z Group excitation and a X group
# excitation, we need to prepare at least one of each type of particle and then
# orbit one around the other.
#
# The following code rotates an X Group excitation around a Z Group excitation.
#

prep1 = [(1, 1), (2, 1)]
prep2 = [(1, 3)]
loop1 = [(2, 3), (2, 2), (2, 1), (3, 1), (3, 2), (2, 3)]

expvals = excitations(prep1, prep2+loop1)
x_expvals, z_expvals = separate_expvals(expvals)

fig, ax = excitation_plot(x_expvals, z_expvals)

ax.plot(*zip(*prep1), color="maroon", linewidth=10)
ax.plot(*zip(*(prep2 + loop1)), color="navy", linewidth=10)

plt.show()

######################################################################
# While we managed to loop one particle around the other, we did not
# extract the relative phase applied to the wavefunction.  To extract
# this information, we will need the *Hadamard test*.


######################################################################
# Hadamard test
# ^^^^^^^^^^^^^
#
# The `Hadamard test <https://en.wikipedia.org/wiki/Hadamard_test_(quantum_computation)>`__
# extracts the real component of a unitary operation
# :math:`\text{Re}\left(\langle \psi | U \rangle \right)`. If the Unitary operation just applies a phase
# :math:`U |\psi\rangle = e^{i \phi} |\psi \rangle`, the measured quantity reduces to :math:`\cos (\phi)`.
#
# The steps in the Hadamard test are:
#
# 1. Prepare the auxiliary qubit into a superposition with a Hadamard
#    gate
# 2. Apply a controlled version of the operation with the auxiliary
#    qubit as the control
# 3. Apply another Hadamard gate to the auxiliary qubit
# 4. Measure the auxiliary qubit in the Z-basis
#
# .. figure:: ../demonstrations/toric_code/Hadamard_test.png
#     :align: center
#     :width: 50%
#
# .. note::
#
#    For extra understanding, validate the Hadamard test algorithm using pen and paper.
#
# Below we implement this algorithm in PennyLane and measure the mutual exchange statistics
# of a X Group excitation and a Z Group excitation.

dev_aux = qml.device("lightning.qubit", wires=[Wire(*s) for s in all_sites] + ["aux"])


def loop(x_loop, z_loop):
    for s in x_loop:
        qml.PauliX(Wire(*s))
    for s in z_loop:
        qml.PauliZ(Wire(*s))


@qml.qnode(dev_aux, diff_method=None)
def hadamard_test(x_prep, z_prep, x_loop, z_loop):
    state_prep()

    for s in x_prep:
        qml.PauliX(Wire(*s))

    for s in z_prep:
        qml.PauliZ(Wire(*s))

    qml.Hadamard("aux")
    qml.ctrl(loop, control="aux")(x_loop, z_loop)
    qml.Hadamard("aux")
    return qml.expval(qml.PauliZ("aux"))


x_around_z = hadamard_test(prep1, prep2, [], loop1)
print("Move x excitation around z excitation: ", x_around_z)


######################################################################
# We just moved two different types of particles around each other and
# picked up a phase. As neither bosons nor fermions behave like this, this
# result demonstrates that the excitations of a toric code are anyons.
#
#
# .. note::
#
#    I encourage dedicated readers to calculate the exchange statistics of:
#
#    * A Z Group excitation and a Z Group Excitation
#    * An X Group excitation and an X Group Excitation
#    * A combination :math:`\Psi` particle and a X Group excitation
#    * A combination :math:`\Psi` particle and a Z Group excitation
#    * A combination :math:`\Psi` particle with another :math:`\Psi` particle
#
#    The combination particle should behave like a standard fermion.
#
# In this demo, we have demonstrated:
#
# 1. How to prepare the ground state of the toric code model on a lattice
#    of qubits
# 2. How to create and move excitations
# 3. The ground state degeneracy of the model on a toric lattice, arising
#    from homotopically distinct loops of operations
# 4. The excitations are anyons due to non-trivial mutual statistics
#
# About the author
# ----------------
#
# .. bio:: Christina Lee
#    :photo: ../_static/authors/christina_lee.JPG
#
#    Christina is currently a quantum software developer at Xanadu working on PennyLane. Outside of coding, she enjoys running, yoga, and piano.
#
