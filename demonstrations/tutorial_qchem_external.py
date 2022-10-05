r"""

Using external libraries
========================

.. meta::
    :property="og:description": Learn how to use external quantum libraries with PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/water_structure.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE
    tutorial_givens_rotations Givens rotations for quantum chemistry
    tutorial_adaptive_circuits Adaptive circuits for quantum chemistry

*Author: Soran Jahangiri. Posted: 19 September 2022. Last updated: 19 September 2022*

The quantum chemistry module in PennyLane, :mod:`qml.qchem  <pennylane.qchem>`, provides built-in
methods for computing molecular integrals, solving Hartree-Fock equations and constructing
fully-differentiable molecular Hamiltonians. PennyLane also allows users to use external libraries
to perform these tasks and integrate the results within a desired workflow. In this tutorial, you
will learn how to use PennyLane with the electronic structure package
`PySCF <https://github.com/sunqm/pyscf>`_ and
`OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools to compute molecular
integrals and construct molecular Hamiltonians.

Building a molecular Hamiltonian
--------------------------------
In PennyLane, molecular Hamiltonians are built with the
:func:`~.pennylane.qchem.molecular_hamiltonian` function by specifying a backend for solving the
Hartree-Fock equations and then constructing the Hamiltonian. The default backend is the
differentiable Hartree-Fock solver of the :mod:`qml.qchem  <pennylane.qchem>` module. The
:func:`~.pennylane.qchem.molecular_hamiltonian` function can also be used to construct the
molecular Hamiltonian with a non-differentiable backend that uses the
`OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`_ plugin interfaced with the
electronic structure package `PySCF <https://github.com/sunqm/pyscf>`_. This
backend can be selected by setting ``method='pyscf'`` in
:func:`~.pennylane.qchem.molecular_hamiltonian`. # This backend requires the ``OpenFermion-PySCF``
plugin to be installed by the user with
"""
pip install openfermionpyscf

##############################################################################
# The molecular Hamiltonian can then be constructed with

import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "O", "H"]
coordinates = np.array([[-0.0399, -0.0038, 0.0],
                        [1.5780, 0.8540, 0.0],
                        [2.7909, -0.5159, 0.0]])

H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, method="pyscf")
print(H)

##############################################################################
# This generates a PennyLane :func:`~.pennylane.Hamiltonian` that can be used in a VQE workflow or
# converted to a sparse matrix in the computational basis.
#
# Additionally, if you have built your electronic Hamiltonian independently using
# `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools, it can
# be readily converted to a PennyLane observable using the
# :func:`~.pennylane.qchem.import_operator` function. Here is an example:

from openfermion.ops import QubitOperator
H = 0.1 * QubitOperator('X0 X1') + 0.2 * QubitOperator('Z0')
H = qml.qchem.import_operator(H)
print(type(H))
print(H)

##############################################################################
# Computing molecular integrals
# -----------------------------
# The one- and two-electron integrals in the molecular orbital basis are necessary for constructing
# fermionic Hamiltonians which can be mapped to a qubit Hamiltonian. The two-electron integrals
# tensor can be factorized and used to construct factorized Hamiltonians which can be simulated
# with a smaller number of resources. These molecular integrals can be computed with the
# :func:`~.pennylane.qchem.electron_integrals` function of PennyLane. Alternatively, the integrals
# can be computed with the `PySCF <https://github.com/sunqm/pyscf>`_ package and used in PennyLane
# workflows such as quantum resource estimation. Here is an example for water in the 6-311G** basis
# set:
#
# manipulated in different ways to
#
#
# About the author
# ----------

##############################################################################
#.. bio:: Soran Jahangiri
#    :photo: ../_static/Soran.png
#
#    Soran Jahangiri is a quantum chemist working at Xanadu. His work is focused on developing and implementing quantum algorithms for chemistry applications.
