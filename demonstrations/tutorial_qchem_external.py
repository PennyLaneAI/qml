r"""
Using external libraries
========================


.. meta::
    :property="og:description": Learn how to use PennyLane with external quantum libraries.

    :property="og:image": https://pennylane.ai/qml/_images/water_structure.png

.. related::
   tutorial_qchem_external Building molecular Hamiltonians

*Author: Soran Jahangiri. Posted: 19 September 2022. Last updated: 19 September 2022*
"""
#
# OpenFermion-PySCF backend
# -------------------------
# The :func:`~.pennylane.qchem.molecular_hamiltonian` function can also be used to construct the
# molecular Hamiltonian with a non-differentiable backend that uses the
# `OpenFermion-PySCF <https://github.com/quantumlib/OpenFermion-PySCF>`_ plugin interfaced with the
# electronic structure package `PySCF <https://github.com/sunqm/pyscf>`_. This
# backend can be selected by setting ``method='pyscf'`` in
# :func:`~.pennylane.qchem.molecular_hamiltonian`:

import numpy as np
from pennylane import qchem

symbols = ["H", "O", "H"]
coordinates = np.array([[-0.0399, -0.0038, 0.0],
                        [1.5780, 0.8540, 0.0],
                        [2.7909, -0.5159, 0.0]])

H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, method="pyscf")
print(H)

##############################################################################
# This backend requires the ``OpenFermion-PySCF`` plugin to be installed by the user with
#
# .. code-block:: bash
#
#    pip install openfermionpyscf
#
# Additionally, if you have built your electronic Hamiltonian independently using
# `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ tools, it can
# be readily converted to a PennyLane observable using the
# :func:`~.pennylane.qchem.import_operator` function.
#
# You have completed the tutorial! Now, select your favorite molecule and build its electronic
# Hamiltonian.
# To see how simple it is to implement the VQE algorithm to compute the ground-state energy of
# your molecule using PennyLane, take a look at the tutorial :doc:`tutorial_vqe`.
#
# About the author
# ----------

##############################################################################
#.. bio:: Soran Jahangiri
#    :photo: ../_static/Soran.png
#
#    Soran Jahangiri is a quantum chemist working at Xanadu. His work is focused on developing and implementing quantum algorithms for chemistry applications.
