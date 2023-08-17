r"""Linear combination of unitaries and block encodings
=============================================================

.. meta::
    :property="og:description": Master the basics of LCUs and their applications
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_intro_qsvt.png

.. related::

    tutorial_intro_qsvt Intro to QSVT

*Author: Juan Miguel Arrazola, Diego Guala, and Jay Soni — Posted: August, 2023.*

If I (Juan Miguel) had to summarize quantum computing in one sentence, it would be like this: information is
encoded in quantum states, and information is processed using unitary operations []. The art and
science of quantum algorithms is to design and buils these unitaries to performing interesting and
useful tasks. But consider this.
My colleague Nathan Wiebe [] once told me that some of his early research was motivated by a simple
question: Quantum computers are good at implementing products of unitaries -- after all
that's how we build circuits from a universal gate set []. But what about implementing
**sums of unitaries**? 🤔

In this tutorial you will learn the basics of one of the most versatile tools in quantum algorithms:
linear combination of unitaries; or LCUs for short. You will also understand how to
use LCUs to create another powerful tool: block encodings.
Among their many uses, block encodings allow us to transform quantum states by non-unitary operators.
This is useful in a variety of contexts, perhaps most famously in qubitization [] and the quantum
singular value transformation (QSVT) [].

[Main Tarik image here]

LCUs
----
The concept of an LCU is very straightforward; it’s basically already explained in the name: we
decompose operators as a weighted sum of unitaries. Mathematically, this means expresssing
an operator :math:`A` in terms of coefficients $\alpha_k$ and unitaries $U_k$ as

.. math:: A =  \sum_{k=1}^N \alpha_k U_k.

A general way to build LCUs in quantum computing is to employ properties of the **Pauli basis**.
This is the set of all products of Pauli operators $I, X, Y, Z$, which forms a complete basis
for the space of operators on $n$ qubits. Expressing an operator in the Pauli basis immediately
gives an LCU decomposition. PennyLane allows you to compute Pauli-basis LCUs using the
:func:`~.pennylane.pauli_decompose` function:

"""

import numpy as np

# Code to eb added by Diego and Jay
A = np.array([[0, 1]])


##############################################################################
# PennyLane uses a smart implementation based on vectorizing the matrix and exploiting properties of
# the Walsh-Hadamard transform, as described `here <https://quantumcomputing.stackexchange.com/questions/31788/how-to-write-the-iswap-unitary-as-a-linear-combination-of-tensor-products-betw/31790#31790 >`_ ,
# but the cost still scales as :math:`n 4^n`. Be careful.
#
# FOn the other hand, it's good to remember that many types of Hamiltonians are already compactly expressed
# in the Pauli basis, for example in various Ising models [] and for molecular Hamiltonians using the
# Jordan-Wigner transformation []. This is very useful since we get one LCU decomposition
# for free.
#
# Block Encodings
# ---------------
# Going from an LCU to a quantum circuit that applies the associated operator is also straightforward
# once you know the trick. What's the trick? To prepare, select, and unprepare 😈.
#
# Starting from the LCU decomposition :math:`A =  \sum_{k=1}^N \alpha_k U_k`, we define the PREP (prepare)
# operator
#
# .. math:: PREP|0\rangle = \sum_k \sqrt{\frac{|\alpha|_k}{\lambda}}|k\rangle,
#
# and the SEL (select) operator
#
# .. math:: SEL|k\rangle |\psi\rangle = |k\rangle U_k |\psi\rangle.
#
# They are aptly named so that you never forget what they do: PREP is preparing a state whose amplitudes
# are determined by the coefficients of the LCU, and SEL is selecting which unitary is applied to
# the system. In case you're wondering, :math:`\lambda = \sum_k |\alpha_k|` is a normalization
# constant, SEL acts this way on any state :math:`|\psi\rangle`, and we have added auxiliary
# qubits where PREP acts. We are also using :math:`|0\rangle` as shorthand to denote the all-zero
# state of the auxiliary qubits.
#
# The final trick is to combine PREP and SEL and make :math:`A` appear 🪄🎩:
#
# .. math:: \langle 0| \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP} |0\rangle|\psi\rangle = A/\lambda |\psi\rangle.
#
# The way to understand this equation is that if apply PREP, SEL, and then invert PREP. After, if
# measure :math:`|0\rangle` in the auxiliary qubits, the input state will be transformed by
# :math:`A`, up to normalization. If you're up for it, it's illuminating to go through the math yourself.
# (Tip: calculate the action of :math:`\text{PREP}^\dagger on :math:`|0\rangle`, not on the output
# state after :math:`\text{SEL} \cdot \text{PREP}`).
#
# The circuit
#
# .. math:: U = \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP}
#
# is a **block encoding** of :math:`A`, up to normalization.
#
# [Add Oriel's figure here]
#
# The reason for this name is that if we write down
# :math:`U` as a matrix, the operator :math:`A` appears directly in a block defined by all
# states where the auxiliary qubits are in state :math:`|0\rangle`.
#
# PennyLane supports direct implementation of prepare and select operators. Let's go through them
# individually and use them to construct a block encoding circuit.
#
# [Jay and Diego to add code and text here]
#
# Application to QSVT
# -------------------
#
# The QSVT algorithm is a method to transform block-encoded operators. You can learn more about it in
# our demos []. PennyLane supports a :class:`~.pennylane.BlockEncode` operation that is designed for
# use with classical simulators. It works by directly applying matrix representations, which is not scalable for
# large numbers of qubits. Here we show how to implement the QSVT algorithm using an explicit
# quantum circuit for the block encoding, using the same example as in the
# `Intro to QSVT demo <https://pennylane.ai/qml/demos/function_fitting_qsp.html>`_. We describe the
# full workflow obtainig the LCU from a Pauli decomposition, defining prepare and select operators,
# building the block encoding circuit, and using the :class:`~pennylane.QSVT` template to apply
# the polynomial transformation :math:`(5 x^3 - 3x)/2`.
#
#
# [Jay and Diego to add code and text here]
#
#
# References
# ----------
#
#
#

##############################################################################
# About the authors
# ----------------
# .. include:: ../_static/authors/juan_miguel_arrazola.txt
# ..include::../ _static / authors / jay_soni.txt
# ..include::../ _static / authors / diego_guala.txt
