r"""A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery
===============================================================================

abstract

Introduction
------------

The game of surface codes [#Litinski]_ is a high-level framework for designing surface code quantum computing architectures.
The game helps us understand space-time trade-offs, where e.g. designs with a higher qubit overhead allow for faster computations and vice versa.
E.g., a space-efficient design might allow a computation with :math:`10^8` T gates to run in :math:`4` hours using :math:`55k` physical qubits, 
whereas and intermediate design may run the same computation in :math:`22` minutes using :math:`120k` physical qubits, 
or a time-optimized design in :math:`1` second using :math:`1500` interconnected quantum computers with :math:`220k` physical qubits, each.

One can draw a rough comparison to microchip design in classical computing, 
where the equivalent game would be about how to arrange the transistors of a chip to perform fast and efficient computations.

The game can be understood entirely from the rules described in the next section. 
However, it still helps to understand the correspondences in physical fault tolerant quantum computing (FTQC) architectures.
First of all it is important to note that we consider surface codes that implement :doc:`(Clifford + T) <compilation/clifford-t-gate-set>` circuits.
In particular, these circuits can be mapped to just performing :doc:`Pauli product measurements <compilation/pauli-product-measurement>`.
This is because all Clifford operations can be moved to the end of the circuit and merged with measurements. 
The remaining non-Clifford gates are realized by :doc:`magic state injection <glossary/what-are-magic-states>` and more Clifford operations that can again be merged with measurements again.
Hence, we mainly care about performing measurements on qubits in arbitrary bases.

We also note that the patches that represent qubits correspond to surface code qubits.
There is a detailed explanation in Appendix A in [#Litinski]_ that describes the surface code realizations of all operations that we are going to see.
These are useful to know in order to grasp the full depth of the game, but are not essential to understanding its rules and concluding design principles explained in this demo.

Rules of the game
-----------------

The game is played on a board of tiles, where patches correspond to qubits.
Underlying these tiles are physical qubits that are statically arranged.
But we should view qubit patches as dynamic entities that appear, move around, deform and disappear again.
The goal of this demo will be to understand the design principles and space-time trade-offs for surface code architectures.
We are going to introduce the necessary rules of the game in this section.

Data qubits as surface code tiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data qubits are realized by patches that at least occupy one tile, but potentially multiple.
They always have four distinct boundaries corresponding to X (dotted) and Z (solid) edges.
Two-qubit patches have 6 distinct boundaries, corresponding to the single X and Z operators of each qubit, as well as the two products XX and ZZ.
This is shown in the figure below.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/qubit_definition.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Qubits are defined as patches of tiles on the board. 
    A single qubit can occupy one tile (a) or multiple tiles (b), where dotted lines correspond to X and solid lines to Z operators.
    Two-qubit patches (c) have 6 boundaries corresponding to the single :math:`X`, :math:`Z` and :math:`XX` and :math:`ZZ` operators.

Basic operations
^^^^^^^^^^^^^^^^

Every operation in the game has an associated time cost that we measure in units of 🕒. These correspond more or less to surface code cycles.
There are some discrepancies but the correspondance is close enough to weigh out space-time trade-offs in architecture designs.
We are not going to give an exhaustive overview of all possible operations, but focus on a few important ones and fill the remaining gaps necessary for the architecture designs in the respective sections below.

- X and Z measurement
- Y measurement



Data blocks design
------------------

Compact data blocks
^^^^^^^^^^^^^^^^^^^

Intermediate data blocks
^^^^^^^^^^^^^^^^^^^^^^^^

Fast data blocks
^^^^^^^^^^^^^^^^


"""
import numpy as np
import pennylane as qml
from pennylane import X, Y, Z, I

import matplotlib.pyplot as plt


##############################################################################
#

##############################################################################
#

##############################################################################
#

##############################################################################
#

##############################################################################
#

##############################################################################
#




##############################################################################
#
# Conclusion
# ----------
#
# asd


##############################################################################
#
# References
# ----------
#
# .. [#Litinski]
#
#     Daniel Litinski
#     "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
#     `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892v3>`__, 2018.
#
#

##############################################################################
# About the author
# ----------------
