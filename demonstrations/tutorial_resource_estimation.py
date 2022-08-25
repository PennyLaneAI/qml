r"""

Quantum Resource Estimation
===========================

.. meta::
    :property="og:description": Learn how to estimate quantum resources
    :property="og:image": https://pennylane.ai/qml/_images/differentiable_HF.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief overview of VQE


*Author: Soran Jahangiri. Posted: 17 August 2022. Last updated: 17 August 2022*

Quantum algorithms such s quantum phase estimation and the variational quantum eigensolver
implemented on a suitable quantum hardware are expected to tackle problems that are
intractable for conventional classical computers. In the absence of quantum devices, the
implementation of such algorithms is limited to computationally inefficient classical simulators.
This makes it difficult to properly explore the accuracy and efficiency of these algorithms for
relatively large problem sizes where the actual advantage of quantum algorithms is expected to be
seen. Despite the simulation difficulties, it is possible to estimate the amount of resources
required to implement such quantum algorithms without performing computationally expensive
simulations.

In this demo, we introduce a functionality in PennyLane that allows estimating the total number of
non-Clifford gates and logical qubits required to implement the quantum phase estimation (QPE)
algorithm for simulating molecular Hamiltonians represented in first and second quantization. We
also present the functionality for estimating the total number of measurements needed to compute
expectation values within a given error using algorithms such as the variational quantum eigensolver
(VQE). Estimating the number of gates and qubits is rather straightforward for the VQE algorithm.

Quantum Phase Estimation
------------------------
The QPE algorithm can be used to compute the phase of a unitary operator within an error $\epsilon$.
The unitary operator $U$ can be selected to share eigenvectors $| \Psi \rangle$ with a molecular
Hamiltonian $H$ by having, for example, $U = e^{-iH}$ to compute the eigenvalues of $H$.
A QPE conceptual circuit diagram is shown in the following. The circuit contains a set of target
wires initialized at the eigenstate $| \Psi \rangle$ which encode the unitary operator and a set of
estimation wires initialized in $| 0 \rangle$ which are measured after applying an inverse quantum
Fourier transform. The measurement results give a binary string that can be used to estimate the
phase of the unitary and the ground state energy of the Hamiltonian. The precision in estimating the
phase depends on the number of estimation wires while the number of gates in the circuit is
determined by the unitary operator.

We are interested to estimate the number of logical qubits and the number of non-Clifford gates,
which are hard to implement, for a QPE algorithm that implements a second-quantized Hamiltonian
describing an isolated molecule and a first-quantized Hamiltonian describing a periodic material.
The PennyLane functionality in the ``resource`` module allows estimating such QPE resources by
simply defining system specifications such as atomic symbols and geometries and a target error for
estimating the ground state energy of the Hamiltonian. Let's see how!

Second quantization
*******************
A `second-quantized <https://en.wikipedia.org/wiki/Second_quantization>`_ molecular Hamiltonian is
constructed from one- and two-body electron integrals in the basis of molecular orbitals,
:math:`\phi`,

.. math:: h_{pq} =\int dx \,\phi_p^*(x)\left(-\frac{\nabla^2}{2}-\sum_{i=1}^N\frac{Z_i}{|r-R_i|}\right)\phi_q(x),\\\\
.. math:: h_{pqrs} = \int dx_1 dx_2\, \frac{\phi_p^*(x_1)\phi_q^*(x_2)\phi_r(x_2)\phi_s(x_1)}{|r_1-r_2|}.

as

.. math:: H=\sum_{pq} h_{pq}a_p^\dagger a_q +\frac{1}{2}\sum_{pqrs}h_{pqrs}a_p^\dagger a_q^\dagger a_r a_s,

where :math:`a^\dagger` and :math:`a` are the fermionic creation and annihilation operators,
respectively. This Hamiltonian can then be transformed to the qubit basis and be written as a linear
combination of the tensor products of Pauli and Identity operators

.. math:: H=\sum_{i} c_i P_i.

The cost of computing the ground state energy of this Hamiltonian using the QPE algorithm typically
depends on two parameters: the 1-norm of the Hamiltonian, which in the simplest case is a sum over
the coefficients $c_i$, and the complexity of implementing the unitary operator, which is typically
constructed as $U = e^{-i \arccos (H)}$.


"""
# About the author
# ----------------

##############################################################################
#.. bio:: Soran Jahangiri
#    :photo: ../_static/Soran.png
#
#    Soran Jahangiri is a quantum chemist working at Xanadu. His work is focused on developing and implementing quantum algorithms for chemistry applications.
