r"""How to build spin Hamiltonians
==================================
Systems of interacting spins provide simple but powerful models for studying problems in physics,
chemistry, and quantum computing. PennyLane provides a powerful set of tools that enables the users
to intuitively construct a broad range of spin Hamiltonians. Here we show you how to use these tools
to easily construct a variety of spin Hamiltonians such as the transverse-field Ising model, the
Fermi-Hubbard model, and the Kitaev honeycomb model.
"""

######################################################################
# Hamiltonian templates
# ---------------------
#
# PennyLane has a range of `functions <https://docs.pennylane.ai/en/latest/code/qml_spin.html#hamiltonian-functions>`__
# for constructing spin model Hamiltonians with minimal input needed from the user. Let’s look
# at the `Fermi-Hubbard <https://en.wikipedia.org/wiki/Hubbard_model>`__ model as an example.
# This model can represent a chain of hydrogen atoms where each atom, or site, can hold one spin-up
# and one spin-down particle. The `Hamiltonian <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.fermi_hubbard.html>`__
# describing this model has two components: the kinetic energy component which is parameterized by a
# hopping parameter, :math:`t`, and the potential energy component parameterized by the on-site
# interaction strength, :math:`U`.
#
# .. math::
#
#     H = -t\sum_{<i,j>, \sigma} c_{i\sigma}^{\dagger}c_{j\sigma} + U\sum_{i}n_{i \uparrow} n_{i\downarrow}.
#
# The terms :math:`c^{\dagger}`, :math:`c` are the creation and annihilation operators,
# :math:`<i,j>` represents the indices of neighbouring spins, :math:`\sigma` is the spin
# degree of freedom, and :math:`n_{i \uparrow}, n_{i \downarrow}` are number operators for spin-up
# and spin-down fermions at site :math:`i`. The Fermi-Hubbard Hamiltonian can then be
# constructed in PennyLane by passing the hoping and interaction parameters to the
# :func:`~.pennylane.spin.fermi_hubbard` function. We also need to specify the shape of the lattice,
# which is ``chain`` in our example. For a full list od supported lattice shapes see the
# :func:`~.pennylane.spin.fermi_hubbard` documentation. The number of sites we would like to include
# in our Hamiltonian should also be defined as a list of integers for
# :math:`x, y, z` directions, depending on the lattice shape. Note that for our ``chain`` model we
# only have one possible direction.

import pennylane as qml

n_cells = [2]
hopping = 0.2
coulomb = 0.3

hamiltonian = qml.spin.fermi_hubbard("chain", n_cells, hopping, coulomb)
hamiltonian

######################################################################
# The :func:`~.pennylane.spin.fermi_hubbard` function is general enough to go beyond the simple
# ``chain`` model and construct
# the Hamiltonian for a wide range of two-dimensional and three-dimensional lattice shapes. For
# those cases, we need to provide the number of sites in each direction of the lattice. We can
# construct the Hamiltonian for a cubic lattice with 3 sites in each direction as

hamiltonian = qml.spin.fermi_hubbard("cubic", [3, 3, 3], hopping, coulomb)

######################################################################
# Similarly, a broad range of other well-investigated spin model Hamiltonians can be constructed
# with the dedicated functions available in the `qml.spin
# <https://docs.pennylane.ai/en/latest/code/qml_spin.html#hamiltonian-functions>`__ module, by just
# providing the lattice information and the Hamiltonian parameters.
#
# Building Hamiltonians manually
# ------------------------------
#
# The Hamiltonian template functions are great and simple tools for someone who just wants to build
# a Hamiltonian quickly. PennyLane also offers tools that can be used to construct
# spin Hamiltonians manually, which are useful for building customized Hamiltonians. Let’s learn
# how to use these tools by constructing the Hamiltonian for the
# `transverse field Ising <https://pennylane.ai/datasets/qspin/transverse-field-ising-model>`__
# model on a two-dimensional lattice.
#
# The Hamiltonian is represented as:
#
# .. math::
#
#     H =  -J \sum_{<i,j>} \sigma_i^{z} \sigma_j^{z} - h\sum_{i} \sigma_{i}^{x},
#
# where :math:`J` is the coupling defined for the Hamiltonian, :math:`h` is the strength of
# transverse magnetic field, and :math:`<i,j> represents the indices of neighbouring spins.
#
# Our approach for doing this is to construct a lattice that represents the spin sites and
# their connectivity. This is done by using the :class:`~.pennylane.spin.Lattice` class that can be
# constructed either by calling the helper function :func:`~.pennylane.spin.generate_lattice` or by
# manually constructing the object. Let's see examples of both methods. First we use
# :func:`~.pennylane.spin.generate_lattice` to construct a square lattice containing
# :math:`9 = 3 \times 3` sites which are all connected to their nearest neighbor.

lattice = qml.spin.generate_lattice('square', [3, 3])

######################################################################
# Let's visualize this lattice to see how it looks. We create a simple function for plotting the
# lattice.

import matplotlib.pyplot as plt

def plot(lattice, figsize=None, showlabel=True):

    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=lattice.n_cells[::-1])

    nodes = lattice.lattice_points

    for edge in lattice.edges:
        start_index, end_index, color = edge
        start_pos, end_pos = nodes[start_index], nodes[end_index]

        x_axis = [start_pos[0], end_pos[0]]
        y_axis = [start_pos[1], end_pos[1]]
        plt.plot(x_axis, y_axis, color='gold')

        plt.scatter(nodes[:,0], nodes[:,1], color='dodgerblue', s=100)

        if showlabel:
            for index, pos in enumerate(nodes):
                plt.text(pos[0]-0.2, pos[1]+0.1, str(index), color='gray')

    plt.axis("off")
    plt.show()

plot(lattice)

######################################################################
# Now, we construct a lattice manually by explicitly defining the positions of the sites in a
# unit cell, the primitive translation vectors defining the lattice and the number of cells in each
# direction [#ashcroft]_. Recall that a unit cell is the smallest repeating unit of a lattice.
#
# Let's create a `square-octagon <https://arxiv.org/abs/1005.3815>`__ lattice manually. Our lattice
# can be constructed in a two-dimensional Cartesian coordinate system with two primitive
# translation vectors defined as vectors along the :math:`x` and :math:`y` directions and four
# lattice point located inside the unit cell. We also assume that the lattice has three unit cells
# along each direction.

from pennylane.spin import Lattice

positions = [[0.2, 0.5], [0.5, 0.2], [0.5, 0.8], [0.8, 0.5]] # coordinate of the lattice cites
vectors = [[1, 0], [0, 1]] # primitive translation vectors
n_cells = [3, 3] # number of unit cells in each direction

lattice = Lattice(n_cells, vectors, positions, neighbour_order=2)

plot(lattice, figsize = (5, 5), showlabel=False)

######################################################################
# Constructing the lattice manually is more flexible while :func:`~.pennylane.spin.generate_lattice`
# only works for some predefined
# `lattice shapes <https://docs.pennylane.ai/en/latest/code/api/pennylane.spin.generate_lattice.html#lattice-details>`__.
#
# Now that we have the lattice, we can use its attributes, e.g., edges and vertices, to construct
# our transverse field Ising model Hamiltonian. For instance, we can access the number of sites
# with ``lattice.n_sites`` and the indices that define each edge with ``lattice.edges_indices``. For
# the full list of attributes, please see the documentation of the :class:`~.pennylane.spin.Lattice`
# class. We also need to define the coupling, :math:`J`, and onsite :math:`h` parameters of the
# Hamiltonian.

from pennylane import X, Y, Z

coupling, onsite = 1.0, 1.0

hamiltonian = 0.0
# add the one-site terms
for vertex in range(lattice.n_sites):
    hamiltonian += -onsite * X(vertex)
# add the coupling terms
for edge in lattice.edges_indices:
    i, j = edge[0], edge[1]
    hamiltonian += - coupling * (Z(i) @ Z(j))

hamiltonian

######################################################################
# In this example we just used the in-built attributes of the lattice we created without further
# customising them. The lattice can be constructed in a more flexible way that allows constructing
# customized Hamiltonians. Let's look at an example.
#
# Adding custom nodes and edges
# -----------------------------
# Now we work on a more complicated Hamiltonian to see how our existing tools allow building it
# intuitively. We construct the anisotropic square-trigonal model, where the coupling parameters
# depend on the orientation of the bonds. We can construct the Hamiltonian by building the
# lattice manually and adding custom edges between the nodes. For instance, to define a custom
# ``XX`` edge with coupling constant 0.5 between nodes 0 and 1, we use:

custom_edge = [(0, 1), ('XX', 0.5)]

######################################################################
# Let's now build our Hamiltonian. We first define the unit cell by specifying the positions of the
# nodes and the translation vectors and then define the number of unit cells in each
# direction.

positions = [[0.1830127018922193, 0.3169872981077807],
             [0.3169872981077807, 0.8169872981077807],
             [0.6830127018922193, 0.1830127018922193],
             [0.8169872981077807, 0.6830127018922193]]

vectors = [[1, 0], [0, 1]]

n_cells = [3, 3]

######################################################################
# Let's plot the lattice to see how it looks like.

plot(Lattice(n_cells, vectors, positions), figsize=(5, 5))

######################################################################
# Now we add custom edges to the lattice. In our example, we define four types of custom
# edges: the first types is the one that connects node 0 to 1, the second type is defined to connect
# node 0 to 2 and the third and forth types connect node 1 to 3 and 2 to 3, respectively. Note that
# this is an arbitrary selection. You can define any type of custom edge you would like.

custom_edges = [[(0, 1), ('XX', 0.5)],
                [(0, 2), ('YY', 0.6)],
                [(1, 3), ('ZZ', 0.7)],
                [(2, 3), ('ZZ', 0.7)]]

lattice = Lattice(n_cells, vectors, positions, custom_edges=custom_edges)

######################################################################
# Let's print the lattice edges and check that our custom edge types are set correctly.

lattice.edges

######################################################################
# You can compare these edges with the lattice plotted above and verify the correct translation of
# the edges over the entire lattice sites.
#
# Now we pass the lattice object to the :func:`~.pennylane.spin.spin_hamiltonian` function, which is
# a helper function that constructs a Hamiltonian from a lattice object.

hamiltonian = qml.spin.spin_hamiltonian(lattice=lattice)

######################################################################
# Alternatively, you can build the Hamiltonian manually by looping over the custom edges
# to build the Hamiltonian.

opmap = {'X': X, 'Y': Y, 'Z': Z}

hamiltonian = 0.0
for edge in lattice.edges:
    i, j = edge[0], edge[1]
    k, l = edge[2][0][0], edge[2][0][1]
    hamiltonian += opmap[k](i) @ opmap[l](j) * edge[2][1]

hamiltonian

######################################################################
# You can see that it is easy and intuitive to construct this anisotropic Hamiltonian with the tools
# available in the `qml.spin <https://docs.pennylane.ai/en/latest/code/qml_spin>`__ module. You can
# use these tools to construct custom Hamiltonians for other interesting systems.
#
# Conclusion
# ----------
# The spin module in PennyLane provides a set of powerful tools for constructing spin
# Hamiltonians. Here we learned how to use these tools to construct pre-defined Hamiltonian
# templates, such as the Fermi-Hubbard model Hamiltonian, and use the
# :class:`~.pennylane.spin.Lattice` object to construct more advanced and customised models such as
# the Kitaev honeycomb Hamiltonian. The versatility of the new spin functions and classes allow you
# to quickly construct any new spin model Hamiltonian intuitively.
#
# References
# ----------
#
# .. [#ashcroft]
#
#     Neil W. Ashcroft, David N. Mermin,
#     "Solid state physics", New York: Saunders College Publishing, 1976
#
# About the author
# ----------------
#
# .. include:: ../_static/authors/diksha_dhawan.txt
#
# .. include:: ../_static/authors/soran_jahangiri.txt
#
