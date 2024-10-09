r"""How to build spin Hamiltonians
==================================
Spin systems provide simple but powerful models for studying problems in physics, chemistry and
quantum computing. The quantum spin models are typically described by a Hamiltonian that can be used
in a quantum algorithm to explore properties of the spin modes that are intractable with classical
computational methods. PennyLane provides a powerful set of tools that enables the users to
intuitively construct a broad range of spin Hamiltonians. Here we show you how to use these tools
for your problem of interest.
"""

######################################################################
# Hamiltonian templates
# ---------------------
#
# PennyLane has a range of functions for constructing spin model Hamiltonians with minimal input
# data needed from the user. Let’s look at the Fermi-Hubbard model as an example. This model can
# represent a chain of hydrogen atoms where each atom, or site, can hold one spin-up and one
# spin-down particle. The Hamiltonian describing this model has two components: the kinetic energy
# component which is parameterized by a hopping parameter and the potential energy component
# parameterized by the on-site interaction strength. The Fermi-Hubbard Hamiltonian can then be
# constructed in PennyLane by passing the hoping and interaction parameters to the fermi_hubbard
# function. We also need to provide information about the number of sites we would like to be
# included in our Hamiltonian.
# 

import pennylane as qml

n = [2]
t = 0.2
u = 0.3

hamiltonian = qml.spin.fermi_hubbard("chain", n, t, u)
hamiltonian

######################################################################
# The fermi_hubbard function is general enough to go beyond the simple “chain” model and construct
# the Hamiltonian for a wide range of two-dimensional and three-dimensional lattice shapes. For
# those cases, we need to provide the number of sites in each direction of the lattice. We can
# construct the Hamiltonian for a cubic lattice with 3 sites on each direction as

hamiltonian = qml.spin.fermi_hubbard("cubic", [3, 3, 3], t, u)

######################################################################
# Similarity, a broad range of other well-investigated spin model Hamiltonians can be constructed
# with the dedicated functions available in the spin module, by just providing the lattice
# information and the Hamiltonian parameters.
#
# Building Hamiltonians manually
# ------------------------------
#
# The Hamiltonian template functions are great and simple tools for someone who just wants to build
# a Hamiltonian quickly. However, PennyLane offers intuitive tools that can be used to construct
# spin Hamiltonians manually which are very handy for building customized Hamiltonians. Let’s learn
# how to use this tools by constructing the Hamiltonian for the transverse field Ising model on a
# two-dimension lattice.
#
# The Hamiltonian is represented as:
#
# .. math::
#
#     \hat{H} =  -J \sum_{<i,j>} \sigma_i^{z} \sigma_j^{z} - h\sum_{i} \sigma_{i}^{x},
#
# where :math:`J` is the coupling defined for the Hamiltonian, :math:`h` is the strength of
# transverse magnetic field and :math:`i,j` represent the indices for neighbouring spins.
#
# Our approach for doing this is to construct a lattice that represents the spin sites and
# their connectivity. This is done by using the Lattice class that can be constructed either by
# calling the helper function generate_lattice or by manually constructing the object. Let's see
# examples of both methods. First we use generate_lattice to construct a square lattice containing 9
# sites which are all connected to their nearest neighbor.

lattice = qml.spin.lattice._generate_lattice('square', [3, 3])

######################################################################
# Let's visualize this lattice to see how it looks. We create a simple function for plotting the
# lattice.

import numpy as np
import matplotlib.pyplot as plt

def plot(lattice):

    plt.figure(figsize=lattice.n_cells[::-1])
    nodes = lattice.lattice_points

    for edge in lattice.edges:
        start_index, end_index, color = edge
        start_pos, end_pos = nodes[start_index], nodes[end_index]

        x_axis = [start_pos[0], end_pos[0]]
        y_axis = [start_pos[1], end_pos[1]]
        plt.plot(x_axis, y_axis, color='gold')

        plt.scatter(nodes[:,0], nodes[:,1], color='dodgerblue', s=100)
        for index, pos in enumerate(nodes):
            plt.text(pos[0]-0.2, pos[1]+0.1, str(index), color='gray')

    plt.axis("off")
    plt.show()

plot(lattice)

######################################################################
# Now, we construct the same lattice manually by explicitly defining the positions of the sites in a
# unit cell, the translation vectors defining the lattice and the number of cells in each direction.

from pennylane.spin import Lattice

nodes = [[0, 0]]  # coordinates of nodes inside the unit cell
vectors = [[1, 0], [0, 1]] # unit cell translation vectors
n_cells = [3, 3] # number of unit cells in each direction

lattice = Lattice(n_cells, vectors, nodes)

######################################################################
# This gives us the same lattice as we created with generate_lattice but constructing the lattice
# manually is more flexible while generate_lattice only works for some predefined lattice shapes.
#
# Now that we have the lattice, we can use its attributes, e.g., edges and vertices, to construct
# our transverse field Isingmodel Hamiltonian. We just need to define the coupling and onsite
# parameters

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
# customising them. The lattice can be constructed in a very flexible way that allows constructing
# customized Hamiltonians. Let's look at an example.
#
# Building customized Hamiltonians
# --------------------------------
# Now we work on a more complicated Hamiltonian to see how our existing tools allow building it
# intuitively. We chose the anisotropic Kitaev Honeycomb model where the coupling
# parameters depend on the orientation of the bonds. [need an image here] We can build the Hamiltonian by building the
# lattice manually and adding custom edges between the nodes. For instance, to define a custom 'XX'
# edge with coupling constant 0.5 between nodes 0 and 1, we use:

custom_edge = [(0, 1), ('XX', 0.5)]

######################################################################
# Let's now build our Hamiltonian. We first define the unit cell by specifying the positions of the
# nodes and the unit cell translation vector and then define the number of unit cells in each
# direction.

nodes = [[0.5, 0.5 / 3 ** 0.5], [1, 1 / 3 ** 0.5]]
vectors = [[1, 0], [0.5, np.sqrt(3) / 2]]
n_cells = [3, 3]

######################################################################
# Let's plot the lattice to see how it looks like.

plot(Lattice(n_cells, vectors, nodes))

######################################################################
# Now we add custom edges to the lattice. We have three different edge orientations that we define
# as

custom_edges = [[(0, 1), ('XX', 0.5)],
                [(1, 2), ('YY', 0.6)],
                [(1, 6), ('ZZ', 0.7)]]

lattice = Lattice(n_cells, vectors, nodes, custom_edges=custom_edges)

######################################################################
# Then we pass the lattice to the spin_hamiltonian function, which is a helper
# function that constructs a Hamiltonian from a lattice.

hamiltonian = qml.spin.spin_hamiltonian(lattice=lattice)

######################################################################
# The spin_hamiltonian function has a simple logic and loops over the custom edges and nodes
# to build the Hamiltonian. In our example, we can also manually do that with a simple code.


opmap = {'X': X, 'Y': Y, 'Z': Z}

hamiltonian = 0.0
for edge in lattice.edges:
    i, j = edge[0], edge[1]
    k, l = edge[2][0][0], edge[2][0][1]
    hamiltonian += opmap[k](i) @ opmap[l](j) * edge[2][1]

hamiltonian

######################################################################
# See how we can easily and intuitively construct the Kitaev model Hamiltonian with the tools in
# these tools.  You can compare the constructed Hamiltonian with the template we already have for
# the Kitaev model and verify that the Hamiltonians are the same.
#
# Conclusion
# ----------
# The spin module in PennyLane provides a set of powerful tools for constructing spin model
# Hamiltonians. Here we learned how to use these tools to construct pre-defined Hamiltonian
# templates, such as the Fermi-Hubbard model Hamiltonian, and use the Lattice object to construct
# more advanced and customised models such as the Kitaev honeycomb Hamiltonian. The versatility of
# the new spin functions and classes allows constructing any new spin model Hamiltonian intuitively.
#
# About the author
# ----------------
#
# .. include:: ../_static/authors/diksha_dhawan.txt
