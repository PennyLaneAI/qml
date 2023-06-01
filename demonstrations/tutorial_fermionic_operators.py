r"""

Fermionic operators
===================

.. meta::
    :property="og:description": Learn how to create and work with fermionic operators
    :property="og:image": https://pennylane.ai/qml/_images/differentiable_HF.png

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe A brief overview of VQE

*Author: Soran Jahangiri — Posted: 01 June 2023. Last updated: 01 June 2023.*

The fermionic creation and annihilation operators are commonly used to describe molecules and spin
systems. The creation operator adds one particle to a given state and the annihilation operator
removes a particle from the state. Imagine a molecule with two orbitals that can each contain one
electron. The quantum state of the molecule can be described by applying creation operators to
create an electron in each orbital. Similarly, applying the annihilation operators to this state
remove the electrons and gives back the original state. These operators have interesting algebraic
properties and commutation relations that make them powerful tools for describing quantum systems
and simulating them with quantum computers. In this tutorial, you will learn how to use PennyLane's
in-built functionalities to build fermionic operators, use them to construct Hamiltonian operators
for interesting systems, and map the resulting operators to the qubit basis so that you can perform
quantum simulations of those systems.

.. figure:: /demonstrations/differentiable_HF/h2.gif
    :width: 60%
    :align: center

    Caption.

Let's get started!

Creating and manipulating fermionic operators
---------------------------------------------


"""
import pennylane as qml
from pennylane import numpy as np


##############################################################################
# ...
#
##############################################################################
# ...
#
# Conclusions
# -----------
# This tutorial introduces ...
#
# References
# ----------
#
# .. [#szabo1996]
#
#     Attila Szabo, Neil S. Ostlund, "Modern Quantum Chemistry: Introduction to Advanced Electronic
#     Structure Theory". Dover Publications, 1996.
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/soran_jahangiri.txt
