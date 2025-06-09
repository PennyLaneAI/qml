r"""How to build vibrational Hamiltonians
=========================================
Vibrational motions are crucially important to describe the quantum properties of molecules and
materials. Molecular vibrations can significantly affect the outcome of chemical reactions. There
are also several spectroscopy techniques that rely on the vibrational properties of molecules to
provide valuable insight to understand chemical systems and design new materials. Classical quantum
computations have been routinely implemented to describe vibrational motions of molecules. However,
efficient classical methods typically have fundamental theoretical limitations that prevent their
practical implementation for describing challenging vibrational systems. This makes quantum
algorithms an ideal choice where classical methods are not efficient or accurate.

Quantum algorithms require a precise description of the system Hamiltonian to compute vibrational
properties. In this demo, we learn how to use PennyLane features to construct and manipulate
different representations of vibrational Hamiltonians. We also briefly discuss the implementation of
the Hamiltonian in an interesting quantum algorithm for computing the vibrational dynamics of a
molecule.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# Vibrational Hamiltonian
# -----------------------
# A molecular vibrational Hamiltonian can be defined in terms of the kinetic energy operator of the
# nuclei, :math:`T`, and the potential energy operator, :math:`V`, that describes the interaction  between
# the nuclei as:
#
# .. math::
#
#     H = T + V.
#
# The kinetic energy operator can be written in terms of position and momentum operations or bosonic
# creation and annihilation operations. The potential energy operator is usually obtained by
# expanding the molecular potential energy surface over vibrational normal coordinates :math:`Q`.
#
# .. math::
#
#     V({Q}) = \sum_i V_1(Q_i) + \sum_{ij} V_2(Q_i,Q_j) + ....
#
# This is typically done by performing single-point energy calculations at small distances along the
# normal mode coordinates. Computing the energies for each mode separately provides the term
# :math:`V_1` while displacing atoms along two different modes simultaneously gives the term
# :math:`V_2` and so on.
#
# There are several ways to represent a vibrational Hamiltonian. Here we explain some of these
# representations and provide PennyLane codes for constructing them.
#
# Christiansen representation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Christiansen representation of the vibrational Hamiltonian is a second-quantization form for
# defining the Hamiltonian in term os bosonic creation :math:`b^{\dagger}` and annihilation
# :math:`b` operations:
#
# .. math::
#
#     H = \sum_{i}^M \sum_{k_i, l_i}^{N_i} C_{k_i, l_i}^{(i)} b_{k_i}^{\dagger} b_{l_i} +
#         \sum_{i<j}^{M} \sum_{k_i,l_i}^{N_i} \sum_{k_j,l_j}^{N_j} C_{k_i k_j, l_i l_j}^{(i,j)}
#         b_{k_i}^{\dagger} b_{k_j}^{\dagger} b_{l_i} b_{l_j},
#
# where :math:`M` represents the number of normal modes and :math:`N` is the number of modals. The
# coefficients :math:`C` represent the one-mode and two-mode integrals defined here.
#
# PennyLane provides a set of functions to construct the Christiansen Hamiltonian, either directly
# in one step or by building the Hamiltonian from its building blocks step by step. An important
# step in both methods is to construct the potential energy operator which is done based on
# single-point energy calculations along the normal modes of the molecule. The
# :func:`~.pennylane.qchem.vibrational_pes` function in PennyLane provides an convenient way to
# perform the potential energy scan by just providing minimal input.


import numpy as np
import pennylane as qml

symbols  = ['H', 'F']                                    # define atomic symbols
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # define atom positions
mol = qml.qchem.Molecule(symbols, geometry)              # construct the molecule
pes = qml.qchem.vibrational_pes(mol)

#
#
# Conclusion
# ----------
# The
#
# References
# ----------
#
# .. [#aaa]
#
#     N. W. ,
#     "".
#
# .. [#bbb]
#
#     D. ,
#     "",
#     2008.
#
# About the authors
# -----------------
#
