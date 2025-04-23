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
First of all it is important to note that we consider surface codes that implement `(Clifford + T) <https://pennylane.ai/compilation/clifford-t-gate-set>`__ circuits.
In particular, these circuits can be compiled to circuits that just perform `Pauli product measurements <https://pennylane.ai/compilation/pauli-product-measurement>`__.
This is because all Clifford operations can be moved to the end of the circuit and merged with measurements. 
The remaining non-Clifford gates are realized by `magic state injection <https://pennylane.ai/qml/glossary/what-are-magic-states>`__ and more Clifford operations, which can be merged with measurements again.
Hence, we mainly care about performing measurements on qubits in arbitrary bases and efficiently distilling and injecting magic states.

We also note that the patches that represent qubits correspond to surface code qubits.
There is a detailed explanation in Appendix A in [#Litinski]_ that describes the surface code realizations of all operations that we are going to see.
These are useful to know in order to grasp the full depth of the game, but are not essential to understanding its rules and concluding design principles that we cover in this demo.

Rules of the game
-----------------

The game is played on a board of tiles, where patches correspond to qubits.
Underlying these tiles are physical qubits that are statically arranged.
But we should view qubit patches as dynamic entities that appear, move around, deform and disappear again.
The goal of this demo will be to understand the design principles and space-time trade-offs for surface code architectures.
We are going to introduce the necessary rules of the game in this section.

Data qubits are realized by patches that at least occupy one tile, but potentially multiple.
They always have four distinct boundaries corresponding to X (dotted) and Z (solid) edges.
This is shown in the figure below.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/qubit_definition_cropped.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Qubits are defined as patches of tiles on the board. 
    A single qubit can occupy one tile (a) or multiple tiles (b), where dotted lines correspond to X and solid lines to Z operators.

Every operation in the game has an associated time cost that we measure in units of 🕒. These correspond more or less to surface code cycles.
There are some discrepancies but the correspondance is close enough to weigh out space-time trade-offs in architecture designs.
We are not going to give an exhaustive overview of all possible operations, but focus on a few important ones and fill the remaining gaps necessary for the architecture designs in the respective sections below.


Arbitrary Pauli product measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the cost of 1🕒 we can measure patches in the X and Z basis. If two patches share a border, one can measure the product of their shared edges as highlighted by the blue region in the figure below.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/ZZ_measurement.png
    :align: center
    :width: 20%
    :target: javascript:void(0)

    Simultaneously measuring the patches of two adjacent patches corresponds to the product of their neighboring edges. Here, we measure $ZZ$.

In particular, if the shared edge contains both Z and X edges, we can measure in the Y basis. In the following example, the upper qubit A has both operator edges $Z_A$ and $X_A$ exposed.
Measuring it together with the auxillary qubit B, initialized in the $|0\rangle$ state below, we measure $(Z_A X_A) \otimes Z_B \propto Y_A \otimes Z_B$ alltogether.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/Y_measurement.png
    :align: center
    :width: 20%
    :target: javascript:void(0)

    $Y$ operators can be measured by having both X and Z edges be exposed with an adjacent auxiliary qubit. The measurement corresponds to the product of all involved operators, involving $Z_A X_A \propto Y_A$.

In practice, we measure a single qubit patch in the Y basis by utilizing an auxiliary qubit. If we start off from a single square patch we first need to deform it at the cost of 1🕒, initialize an auxiliary qubit at no cost, and perform the joint measurement as shown above (1🕒).
The entire protocol is shown below:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/Y_measurement_protocol.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    The protocol for measuring a single qubit in the Y basis involves deforming the patch (Step 2, 1🕒), initializing an auxillary qubit in $|0\rangle$ (0🕒), simultaneously measure both patches (1🕒) and deforming the qubit back again.

Auxiliary qubits play an important role as they allow measuring products Pauli operators on different qubits, 
which is the most crucial operation in this framework, since everything is mapped to `Pauli product measurements <https://pennylane.ai/compilation/pauli-product-measurement>`__.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/PPM.png
    :align: center
    :width: 30%
    :target: javascript:void(0)

    Measuring $Y_1 X_3 Z_4 X_5$ via a joint auxiliary qubit in 1🕒. In principle multi-qubit measurements with many qubits come at the same cost as with fewer qubit, however the requirement of having an auxiliary region connecting all qubits may demand extra formations.

Non-Clifford Pauli rotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Non-Clifford Pauli rotations $e^{-i \frac{\pi}{8} P}$ for some Pauli word $P$ are realized via `magic state distillation and injection <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.
Magic state distillation blocks are a crucial part of the architecture design that we are going to cover later. 
For the moment we assume that we have means to prepare magic states $|m\rangle = |0\rangle + e^{-i \frac{\pi}{4}} |1\rangle$ on special qubit tiles (distillation blocks).
Magic state injection in this case then refers to the following protocol:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/magic_state_injection.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Performing a non-Clifford $\pi/8$ rotation corresponds to performing the joint measurement of the Pauli word and $Z$ on the magic state qubit. The additionally classically controlled Clifford rotations can be merged again with the measurements at the end of the circuit.

Take for example the Pauli word $P = Z_1 Y_2 X_4$ on the architecture layout below. 
This design allows one to directly perform $e^{-i \frac{\pi}{8} P}$ as we have access to all of $X, Y, Z$ on each qubit, as well as the Z edge for the magic state qubit.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/non_clifford_rotation.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Performing $e^{-i \frac{\pi}{8} Z_1 Y_2 X_4}$ by measuring $Z_1 Y_2 X_4 Z_m$. The remaining Clifford Pauli rotations are merged with the terminal measurements at the end of the circuit via compilation.

We are going to see in the next section that one of the biggest problems is performing Y rotations and measurements (same thing, really, in this framework).

Data blocks design
------------------

We now have all the necessary tools to understand different designs and their space-time tradeoffs.

Compact data blocks
^^^^^^^^^^^^^^^^^^^

The compact data block has the following form. The middle aisle is going to be used as an auxiliary qubit region.

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/compact_block.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    The compact data block design is efficient in space. However, only one edge is exposed to the auxiliary qubit region in the middle.

This design only uses $\frac{3}{2}n$ tiles and 3 additional ones for a magic state distillation block.
The biggest drawback is rather obvious: we can only access $Z$ measurements in the auxiliary qubit region. In order to perform joint $X$ measurements,
we can perform a patch rotation at a cost of 3🕒:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/batch_rotation.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    A patch rotation can be used to expose the $X$ edge to the auxiliary qubit region.

An additional problem of this design is the fact that there is no space for qubits to deform to in order to perform Y measurements.
This can be remedied by making use of the identity 

.. math:: e^{i \frac{\pi}{8} Y} = e^{-i \frac{\pi}{4} Z} e^{i \frac{\pi}{8} X} e^{i \frac{\pi}{4} Z}.

The second (first in the circuit) Clifford rotation $e^{i \frac{\pi}{4} Z}$ needs to be explicitly performed in this case. The first one can be merged again with the terminal measurements of the circuit.
Such a rotation $e^{i \frac{\pi}{4} P}$ can be performed with a joint measurement of $P \otimes Y$, similar to the magic state distillation circuit:

.. figure:: ../_static/demonstration_assets/game_of_surface_codes/clifford_rotation.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    A patch rotation can be used to expose the $X$ edge to the auxiliary qubit region.


Intermediate data blocks
^^^^^^^^^^^^^^^^^^^^^^^^

Fast data blocks
^^^^^^^^^^^^^^^^

Distillation blocks design
--------------------------


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
