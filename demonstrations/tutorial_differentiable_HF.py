r"""

Autodifferentiable Hartree-Fock solver
======================================

.. meta::
    :property="og:description": Learn how to use the differentiable Hartree-Fock solver
    :property="og:image": https://pennylane.ai/qml/_images/differentiable_HF.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE
    tutorial_givens_rotations Givens rotations for quantum chemistry
    tutorial_adaptive_circuits Adaptive circuits for quantum chemistry


*Author: PennyLane dev team. Posted:  2021. Last updated: XX December 2021*

Variational quantum algorithms aim to calculate the energy of a molecule by constructing a
parametrized quantum circuit and find a set of parameters that minimize the expectation value of the
electronic molecular Hamiltonian. For a given molecule, the electronic Hamiltonian is obtained by
solving the Hartree-Fock calculations which provide a set of one- and two-electron integrals that
are used to construct the Hamiltonian. The constructed Hamiltonian depends on a set of molecular
parameters, such as the atomic coordinates and basis set parameters, which are excluded from the
optimization problem. The ability to optimize these molecular parameters concurrently with the
circuit parameters provides several computational advantages such as efficient molecular geometry
optimization and reaching lower energies without increasing the number of basis functions.

Optimization of the molecular Hamiltonian parameters, at the same time as the circuit parameters,
can be achieved by differentiating the expectation value of the Hamiltonian with respect to the
molecular parameters which can be done with symbolic, numeric, or automatic differentiation.
Symbolic differentiation obtains derivatives of an input function by direct mathematical
manipulation, for example using standard strategies of differential calculus. These can be performed
by hand or with the help of computer algebra software. The resulting expressions are exact, but the
symbolic approach is of limited scope, particularly since many functions are not known in explicit
analytical form. Symbolic methods also suffer from the expression swell problem where careless usage
can lead to exponentially large symbolic expressions. Numerical differentiation is a versatile but
unstable method, often relying on finite differences to calculate approximate derivatives. This is
problematic especially for large computations consisting of many differentiable parameters.
Automatic differentiation is a computational strategy where a function implemented using computer
code is differentiated by expressing it in terms of elementary operations for which derivatives are
known. The gradient of the target function is then obtained by applying the chain rule through the
entire code. In principle, automatic differentiation can be used to calculate derivatives of a
function using resources comparable to those required to evaluate the function itself.

In this tutorial, you will learn how to use the autodifferentiable Hartree-Fock solver implemented
in PennyLane. The Hartree-Fock module in PennyLane provides built-in methods for constructing atomic
and molecular orbitals, building Fock matrices, and solving the self-consistent Hartree-Fock
equations to obtain optimized orbitals, which can be used to construct fully-differentiable
molecular Hamiltonians. PennyLane allows users to natively compute derivatives of all these objects
with respect to the underlying parameters. We will introduce a workflow to jointly optimize circuit
parameters, nuclear coordinates, and basis set parameters in a variational quantum eigensolver
algorithm. Let's get started!

Hartree-Fock method
-------------------

The main goal of the Hartree-Fock method is to obtain molecular spin-orbitals that minimize the energy of a state where
electrons are treated as independent particles occupying the lowest-energy orbitals.
"""

import pennylane as qml
from pennylane import numpy as np

symbols = ["H", "H"]
geometry = np.array([[0.0, 0.0, -0.672943567415407],
                     [0.0, 0.0,  0.672943567415407]], requires_grad=True)

##############################################################################
# We now compute the molecular Hamiltonian in the
# `STO-3G <https://en.wikipedia.org/wiki/STO-nG_basis_sets>`_ basis.
