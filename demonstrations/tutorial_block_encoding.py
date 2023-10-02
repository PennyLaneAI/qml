r"""Block encoding
=============================================================

.. meta::
    :property="og:description": Learn how to perform block encoding
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_block_encoding.png

.. related::

     tutorial_intro_qsvt Intro to QSVT

*Author: XXX — Posted: September 29, 2023.*

Prominent quantum algorithms such as Quantum Phase estimation and Quantum Singular Value
Transformation require implementing a non-unitary operator in a quantum circuit. This is problematic
because quantum computers can only perform unitary evolutions. Block encoding is a general technique
that solves this problem by embedding the non-unitary operator in a unitary matrix that can be
implemented in a quantum circuit containing a set of ancilla qubits. In this tutorial, you learn
several block encoding methods that are commonly used in quantum algorithms.

|

.. figure:: ../demonstrations/intro_qsvt/thumbnail_intro_qsvt.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

Block encoding a matrix
-----------------------
We define block encoding as embedding a given non-unitary matrix, :math:`A`, into a matrix :math:`U`

.. math:: U(a) = \begin{pmatrix} A & *\\
                  * & *
                 \end{pmatrix},

such that :math:`U\{\dagger} U = I`. A general recipe for this encoding is to construct :math:`U` as

.. math:: U(a) = \begin{pmatrix} a & \sqrt{1-a^2}\\
                 \sqrt{1-a^2} & -a
                 \end{pmatrix}.

The only condition is that the singular values of :math:`A` should be bounded by 1. This block
encoding can be done in PennyLane using the :class:`~pennylane.BlockEncode` operation. The following
code shows block encoding for a :math:`2 \time 2` matrix.
"""
import pennylane as qml

A = [[0.1, 0.2], [0.3, 0.4]]
U = qml.BlockEncode(A, wires=range(2))
print(qml.matrix(U))

##############################################################################
# We can directly implement this operator in a circuit that will be executed on a quantum simulator.
# This is great but we also need to know how to implement a block encoding unitary using a set of
# quantum gates. In the following sections we learn three main techniques for constructing circuits
# that implement a block encoding unitary for a given matrix.

# Block encoding sparce matrices
# ------------------------------
# Sparse matrices that have specific structures can be efficiently block encoded. A general
# circuit for block encoding s-sparce matrices can be constructed as
#
# Fig 5 of https://arxiv.org/abs/2203.10236
#
# The circuit has n qubits and m + 1 ancilla qubits.
# Ds is defined as HxHxH
# O_A, O_s can be constructed for structured sparse matrices
# Let's look at O_A, O_s for the Banded circulant matrix
#
# Block encoding with FABLE
# -------------------------
# The fast approximate quantum circuits for block encodings (FABLE) is a general method
# for block encoding dense and sparse matrices. The level of approximation in FABLE can be adjusted
# to compress and sparsify the resulting circuits. The general circuit is constructed from a set of
# rotation and C-NOT gates. The rotation angles are obtained from the elements of the encoded
# matrix. ...

# Block encoding with LCU
# -----------------------
# A powerful method for block encoding a matrix is to decompose it into a linear combination of
# unitaries (LCU) and then block encode the LCU. The circuit for this block encoding is constructed
# using two important building blocks: Prepare and Select operations.


# Summary and conclusions
# -----------------------
# Block encoding is a powerful technique in quantum computing that allows implementing a non-unitary
# operation in a quantum circuit typically by embedding the operation in a larger unitary operation.
# Here we reviewed some important block encoding methods with code examples. The choice of the block
# encoding approach depends on the non-unitary operation we want to implement. The functionality
# provided in PennyLane allows you to explore and benchmark different approaches for a desired
# problem. Can you select a matrix and compare different block encoding methods for it?
#
# References
# ----------
#
# .. [#sparse]
#
#     Daan Camps, Lin Lin, Roel Van Beeumen, Chao Yang,
#     "Explicit quantum circuits for block encodings of certain sparse matrices",
#     `arXiv:2203.10236 <https://arxiv.org/abs/2203.10236>`__, 2022
#
#
# .. [#fable]
#
#    Daan Camps, Roel Van Beeumen,
#    "FABLE: fast approximate quantum circuits for block-encodings",
#    `arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`__, 2022
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/xxx.txt
