r"""

Fermionic Operators
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

.. figure:: /demonstrations/fermionic_operators/creation.jpg
    :width: 60%
    :align: center

    Caption.

Let's get started!

Creation and annihilation operators
-----------------------------------

The fermionic creation and annihilation operators can be easily constructed in PennyLane, similar to
the Pauli operators, with the :class:`~.pennylane.FermiC` and :class:`~.pennylane.FermiA` classes
"""
import pennylane as qml
from pennylane import numpy as np

c = qml.FermiC(0)
a = qml.FermiA(1)

##############################################################################
# Once created, this operators can be multiplied or added to each other to create new operators that
# we can call Fermi word, for the multiplication, and Fermi sentence for the linear combination of
# the Fermi words.

fermi_word = c * a
fermi_sentence = 1.2 * c * a + 2.4 * fermi_word

##############################################################################
# In this simple example, we first created the operator :math:`c^{\dagger}_0 c_1` and then created
# the linear combination :math:`1.2 c^{\dagger}_0 c_1 + 2.4 c^{\dagger}_0 c_1` which is simplified
# to :math:`3.7 c^{\dagger}_0 c_1`. We can create even more complicated operators such as
#
# .. math::
#
#     1.2 \times a_0^{\dagger} a_1 a_2^{\dagger} a_3 - 2.3 \times \left ( a_2^{\dagger} a_2 \right )^2
#
# using only the :class:`~.pennylane.FermiC` and :class:`~.pennylane.FermiA` classes

fermi_sentence = (
    1.2 * qml.FermiC(0) * qml.FermiA(1) * qml.FermiC(2) * qml.FermiA(3)
    - 2.3 * (qml.FermiC(2) * qml.FermiA(2)) ** 2
)
##############################################################################
# This Fermi sentence can be mapped to the qubit basis and reconstructed as a linear combination
# of Pauli operators.

pauli_sentence = fermi_sentence.to_qubit()
#
# Let's learn a bit more about the mapping process.
#
# Mapping fermionic operators
# ---------------------------
#
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
