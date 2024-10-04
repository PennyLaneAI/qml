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
hamiltonian

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
# Our approach for doing this is to construct a Lattice object that represents the spin sites and
# their connectivity. The Lattice object can be constructed by defining the unit cell translation
# vectors, the positions of the sites and the number of unit cells in the lattice. Once we have the
# Lattice object, we can use its attributes, e.g., ``vertices`` and ``edges``, to construct our
# custom Hamiltonian.

from pennylan.spin import Lattice
from pennylane import X, Y, Z
import numpy as np

# The unit cell translation vectors
vectors = [[1, 0], [0, 1]]

# The coordinates of the nodes inside the unit cell
nodes = [[0, 0]]

# Number of unit cells in each direction
n_cells = [2, 2]

# Lattice representing the system
lattice = Lattice(n_cells, vectors, nodes)

# Coupling and onsite parameters
coupling, onsite = 1.0, 10.0

# Construct the Hamiltonian manually
def h_custom(lattice, coupling, onsite):
    hamiltonian = 0.0

    # add two-site terms
    for edge in lattice.edges:
        i, j = edge[0], edge[1]
        hamiltonian += - coupling * (Z(i) @ Z(j))

    # add one-site terms
    for vertex in range(lattice.vertices):
        hamiltonian += -onsite * X(vertex)

    return hamiltonian


h_custom(lattice, coupling, onsite)

######################################################################
# We also have a helper function construct_lattice that helps you construct the lattice object just
# by passing the shape of the lattice without defining the nodes and the translation vectors.
#
# The lattice object is a very flexible and versatile tool that allows you construct more
# complicated Hamiltonians. To show this ability, let’s look at a more advanced example.
#
# Building customized Hamiltonians
# --------------------------------
#
# Now let's look at a more complicated example and see how our existing tools allow building such
# Hamiltonians intuitively. We chose the anisotropic Kitaev Honeycomb model where the coupling
# parameters depend on the orientation of the bonds. We can build the Hamiltonian by building the
# lattice manually and adding custom edges between the nodes. The custom edges can be defined based
# on the nodes they connect. Currently, we use the following format to define a custom 'XX' edge
# with coupling constant 0.5 between nodes 0 and 1:

custom_edge = [[(0, 1), ('XX', 0.5)]

######################################################################
# We can also support a UI where the user defines the neighbour vectors manually. This option is a
# bit complicated and is abstracted away in the current UI. Let's now build our Hamiltonian.
# The unit cell translation vectors
vectors = [[1, 0], [0.5, np.sqrt(3) / 2]]

# The coordinates of the nodes inside the unit cell
nodes = [[0.5, 0.5 / 3 ** 0.5], [1, 1 / 3 ** 0.5]]

# Number of unit cells in each direction
n_cells = [3, 3]

# Add custom edges to the lattice
# the first term adds custom 'XX' edge with
# coupling constant 0.5 between 0 and 1 nodes
custom_edges = [[(0, 1), ('XX', 0.5)],
                [(1, 2), ('YY', 0.6)],
                [(1, 6), ('ZZ', 0.7)],
                [(3, 6), ('XY', 0.8)]]

# Lattice representing the system
lattice = Lattice(n_cells, vectors, nodes, custom_edges=custom_edges)

# Construct the Hamiltonian manually
def h_custom(lattice):
    opmap = {'X': X, 'Y': Y, 'Z': Z}

    hamiltonian = 0.0

    # add two-site terms
    for edge in lattice.edges:
        i, j = edge[0], edge[1]
        k, l = edge[2][0][0], edge[2][0][1]
        hamiltonian += opmap[k](i) @ opmap[l](j) * edge[2][1]
    return hamiltonian

h_custom(lattice)

######################################################################
# You can compare the constructed Hamiltonian with the template we already have for the Kitaev
# model.
#
# Conclusion
# ----------
# To be added ...
#
# About the author
# ----------------
#
# .. include:: ../_static/authors/diksha_dhawan.txt
