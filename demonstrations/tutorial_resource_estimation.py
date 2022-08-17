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
required to implement such quantum algorithms without performing actual classical simulations.

In this demo, we introduce a functionality in PennyLane that allows estimating the total number of
non-Clifford gates and logical qubits required to implement the quantum phase estimation (QPE)
algorithm for simulating molecular Hamiltonians represented in first and second quantization. We
also present a functionality that allows estimating the total number of measurements needed to
implement the variational quantum eigensolver (VQE) for a given molecular Hamiltonian. Estimating
the number of gates and qubits is rather straightforward for the VQE algorithm.

Quantum Phase Estimation
------------------------

"""
# About the author
# ----------

##############################################################################
#.. bio:: Soran Jahangiri
#    :photo: ../_static/Soran.png
#
#    Soran Jahangiri is a quantum chemist working at Xanadu. His work is focused on developing and implementing quantum algorithms for chemistry applications.
