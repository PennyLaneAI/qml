r"""

Qubit tapering
==============

.. meta::
    :property="og:description": Learn how to taper off qubits
    :property="og:image": https://pennylane.ai/qml/_images/ qubit_tapering.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE
    tutorial_givens_rotations Givens rotations for quantum chemistry
    tutorial_adaptive_circuits Adaptive circuits for quantum chemistry


*Author: PennyLane dev team. Posted:  2021. Last updated: XX January 2022*

The performance of variational quantum algorithms is considerably limited by the number of qubits
required to represent the trial wavefunction space. In the context of quantum chemistry, this
limitation hinders the treatment of large molecules with algorithms such as the variational quantum
eigensolver (VQE). Several approaches have been developed to reduce the qubit requirements for
fixed-accuracy electronic structure calculations. In this tutorial, we demonstrate the
symmetry-based qubit tapering method which allows reducing the number of qubits required to perform
molecular quantum simulations by leveraging the symmetries that are present in molecular
Hamiltonians [#bravyi2017]_ [#setia2019]_.
"""

import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "H"]
geometry = np.array([[-0.672943567415407, 0.0, 0.0],
                     [ 0.672943567415407, 0.0, 0.0]], requires_grad=True)

##############################################################################
# We now create a molecule object that stores all the molecular parameters needed to construct the
# molecular electronic Hamiltonian.

mol = qml.hf.Molecule(symbols, geometry)
hamiltonian = qml.hf.generate_hamiltonian(mol)(geometry)
print(hamiltonian)

##############################################################################
# The Hamiltonian contains 15 terms acting on one to four qubits. This Hamiltonian can be
# transformed such that it acts trivially on some of the qubits.

generators, paulix_ops = qml.hf.generate_symmetries(hamiltonian, len(hamiltonian.wires))

paulix_sector = [1, -1, -1]

H_tapered = qml.hf.transform_hamiltonian(hamiltonian, generators, paulix_ops, paulix_sector)

#
# References
# ----------
#
# .. [#bravyi2017]
#
#     Sergey Bravyi, Jay M. Gambetta, Antonio Mezzacapo, Kristan Temme, "Tapering off qubits to
#     simulate fermionic Hamiltonians". `arXiv:1701.08213
#     <https://arxiv.org/abs/1701.08213>`__
#
# .. [#setia2019]
#
#     Kanav Setia, Richard Chen, Julia E. Rice, Antonio Mezzacapo, Marco Pistoia, James Whitfield,
#     "Reducing qubit requirements for quantum simulation using molecular point group symmetries".
#     `arXiv:1910.14644 <https://arxiv.org/abs/1910.14644>`__
