r"""Intro to QSVT
=============================================================

.. meta::
    :property="og:description": Introduction to the Quantum Singular Value Transformation algorithm
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_pulse_programming.png

Author: Juan Miguel Arrazola — Posted: 2023.

There are few quantum algorithms deserving to be placed in a hall of fame: Shor's algorithm, Grover's algorithm, quantum phase estimation;
maybe even HHL and VQE. While it's still early in its career, there is a new technique with prospects of achieving such celebrity status:
the quantum singular value transformation (QSVT). Since you're reading this, chances are you have at least heard of QSVT and its broad
applicability.

This tutorial introduces the fundamental principles of QSVT with example code from PennyLane. We focus on the basics;
while these techniques may appear intimidating when reading the literature, the fundamentals are relatively easy to grasp. Teaching
you these core principles is the purpose of this tutorial.

|

.. figure:: ../demonstrations/intro_qsvt/QSVT.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

Transforming scalars encoded in matrices
-----------------------------------------
My personal perspective on QSVT is that it is really a result in linear algebra that tells us how to
transform matrices encoded in larger unitary matrices.

Let's start with the simplest example:
we encode a scalar :math:`a` inside a 2x2 unitary :math:`U(a)`. By encoding we mean that the
matrix form of the unitary depends explicitly on :math:`a`. This encoding can be achieved in
multiple ways, for example:

.. math:: U(a) = \begin{pmatrix} a & \sqrt{1-a^2}\\
    \sqrt{1-a^2} & -a
    \end{pmatrix}.

The parameter :math:`a` must lie between -1 and 1 to ensure the operator is unitary, but this is just a matter of rescaling.

We now ask the crucial question that will get everything started:
what happens if we repeatedly alternate multiplication of this unitary by some other unitary? 🤔
There are multiple choices for the "other unitary", for example

.. math:: S(\phi) = \begin{pmatrix} e^{i\phi} & 0\\
    0 & e^{-i\phi}
    \end{pmatrix},

which has the advantage of being diagonal. This is known as the **signal-processing** operator.

The answer to our crucial question is encapsulated in a result known as **quantum signal processing** (QSP).
If we alternate products of :math:`U(a)` and :math:`S(\phi)`, keeping :math:`a` fixed and varying :math:`\phi`,
the top-left corner of the resulting matrix is a polynomial transformation of :math:`a`. Mathematically,

.. math:: S(\phi_0)\prod_{k=1}^d W(a) S(\phi_k) = \begin{pmatrix}
    P(a) & *\\
    * & *
    \end{pmatrix}.

The asterisk :math:`*` is used to indicate that we are not interested in these entries.
The complex polynomial :math:`P(a)` has degree at most :math:`d`, determined by the number of angles,
and its particular form depends on the choice of angles.

The main quantum signal processing theorem states that there exist a sequence of :math:`d+1` angles
that can implement *any* complex polynomial of degree :math:`d`.
Finding the desired angles can be done efficiently in practice, but identifying the best
methods is an active area of research.

For now, let's look at a simple example of how quantum signal processing can be implemented using
PennyLane. We aim to perform a transformation by the Legendre polynomial
:math:`(5 x^3 - 3x)/2`, for which we use pre-computed optimal angles.
As you will soon learn, QSP can be viewed as a special case of QSVT. We thus use the :class:`~pennylane.qsvt`
operation to construct the output matrix and compare the resulting transformation to
the target polynomial.

"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt


def target_poly(a):
    return 0.5 * (5 * a**3 - 3 * a)


# pre-optimized angles
angles = [-0.20409113, -0.91173829, 0.91173829, 0.20409113]


def qsvt_output(a):
    # output matrix
    out = qml.matrix(qml.qsvt(a, angles, wires=[0]))

    return out[0, 0]  # top-left entry


a_vals = np.linspace(-1, 1, 50)
qsvt = [np.real(qsvt_output(a)) for a in a_vals]  # neglect small imaginary part
target = [target_poly(a) for a in a_vals]


plt.plot(a_vals, qsvt, label="target")
plt.plot(a_vals, target, "*", label="qsvt")

plt.legend()
plt.show()


##############################################################################
# It works! 🎉 💃
#
# Quantum signal procesing is a result about multiplication of 2x2 matrices, yet it is the core principle
# underlying the QSVT algorithm. If you've made it this far, you're in great shape for the rest to come 🥇
#
#
# Transforming matrices encoded in matrices
# ------------------------------------------
#
# Time to ask another key question: what if instead of encoding a scalar, we encode an entire matrix :math:`A`? 🧠
# This is trickier since we need to ensure that the larger operator remains unitary. A way to achieve this
# is to use a similar construction as in the scalar case:
#
# .. math:: U(A) = \begin{pmatrix} A & \sqrt{1-A A^\dagger}\\
#     \sqrt{1-A^\dagger A} & -A^\dagger
#     \end{pmatrix}.
#
# This operator is a valid unitary regardless of the form of :math:`A`; it doesn't even have to be a square matrix. We just
# need to ensure that :math:`A` is properly normalized such that its largest singular value is bounded by 1.
#
# Any such method of encoding a matrix inside a larger unitary is known as a **block encoding**. In our construction,
# the matrix :math:`A` is encoded in the top-left block, hence the name. PennyLane supports
# the `:class:`~pennylane.BlockEncode` operation that follows construction above. Let's test
# it out with an example:

# square matrix
A = [[0.1, 0.2], [0.3, 0.4]]
U1 = qml.BlockEncode(A, wires=range(2))
print("U(A) = ", np.round(qml.matrix(U1), 2))

##############################################################################
# rectangular matrix
B = [[0.5, -0.5, 0.5]]
U2 = qml.BlockEncode(B, wires=range(2))
print("U(B) = ", np.round(qml.matrix(U2), 2))


##############################################################################
# Notice that we haven't really made a reference to quantum computing; everything is just linear algebra.
# Told you so! 😈
#
# Quantum kicks in when we construct explicit circuits that implement
# a block-encoding unitary. Much of the challenge of quantum algorithms leveraging these techniques
# lies in finding strategies to implement block encodings. We don't cover such
# methods in detail here, but for reference, a popular approach is to begin from a linear combination of unitaries for :math:`A`
# and define associated :math:`\text{PREP}` (prepare) and :math:`\text{SEL}` (select) operators.
# Then the unitary
#
# .. math::  U=\text{PREP}^\dagger\cdot\text{SEL}\cdot\text{PREP},
#
# is a block-encoding of :math:`A` up to a constant factor.
#
# Time for a third key question: Can we use the same strategy as in QSP to
# polynomially transform a block-encoded matrix? Because that would be fantastic 😎
#
# For this to be possible, we need to generalize the signal-processing operator
# :math:`S(\phi)`. This can be done by using a diagonal unitary where we apply the phase
# :math:`e^{i\phi}` to the subspace determined by the block,
# and the phase :math:`e^{-i \phi}` everywhere else. For example, revisting the square matrix :math:`A` in
# the code above, where :math:`A` is encoded in a two-dimensional subspace, the corresponding operator is
#
# .. math::  \begin{pmatrix} e^{i\phi} & 0 & 0 & 0\\
#    0 & e^{i\phi} & 0 & 0 \\
#    0 & 0 & e^{-i\phi} & 0\\
#    0 & 0 & 0 & e^{-i\phi} \\
#    \end{pmatrix}.
#
# These are known as a **projector-controlled phase gates**, for which we use the symbol :math:`\Pi(\phi)`.
#
# When :math:`A` is not square,
# we have to be careful and define two operators: one acting on the row subspace and another on the
# column subspace. Projector-controlled phase gates are implemented in PennyLane using the
# :class:`~pennylane.PCPhase` operation. Here's a simple example:

dim = 2
phi = np.pi / 2
pi = qml.PCPhase(phi, dim, wires=range(2))
print("Pi = ", np.round(qml.matrix(pi), 2))


##############################################################################
# As you may have guessed, this generalization of QSP does the trick. By cleverly alternating a block-encoding unitary
# and the appropriate projector-controlled phase gates, we can polynomially transform the encoded matrix.
# The result is the QSVT algorithm.
#
# Mathematically, when the polynomial degree :math:`d` is even (number of angles is odd), the QSVT result
# states that
#
# .. math:: \prod_{k=1}^{d/2}\Pi_{\phi_{2k-1}}U(A)^\dagger \tilde{\Pi}_{\phi_{2k}} U(A)=
#    \begin{pmatrix} P(A) & *\\
#    * & *
#    \end{pmatrix}.
#
# The tilde is used in the projector-controlled phase gates to distingush whether they act on the column or row
# subspaces. The polynomial transformation of :math:`A` is defined in terms of its singular value decomposition
#
# .. math:: P(A) = \sum_k P(\sigma_k)|w_k\rangle \langle v_k|,
#
# where we use braket notation to denote the left and right singular vectors.
# For technical reasons, the sequence looks slightly different when the polynomial degree is odd:
#
# .. math:: \Pi_{\phi_1}\prod_{k=1}^{(d-1)/2}\Pi_{\phi_{2k}}U(A)^\dagger \tilde{\Pi}_{\phi_{2k+1}} U(A)=
#    \begin{pmatrix}
#    P(A) & *\\
#    * & *
#    \end{pmatrix}.
#
# As with QSP, it can be shown that there exists angles such that any polynomial transformation up to degree
# :math:`d` can be implemented by the QSVT sequence. In fact, as long as we're careful with conventions,
# we can use the same angles regardless of the dimensions of :math:`A`.
#
# The QSVT construction is a beautiful result. By using a number of operations that grows linearly with the
# degree of the target polynomial, we can transform the singular values of arbitrary block-encoded matrices
# without ever having to actually perform singular value decompositions! If the block encoding circuits can be
# implemented in polynomial time in the number of qubits, the resulting
# quantum algorithm will also run in polynomial time. This is very powerful.
#
# In PennyLane, implementing the QSVT transformation is as simple as using :class:`~pennylane.qsvt`. Let's revisit
# our previous example and transform a matrix according to the same Legendre polynomial. We'll use a diagonal matrix
# with eigenvalues evenly distributed between -1 and 1, allowing us to easily check the transformation.


eigvals = np.linspace(-1, 1, 16)
A = np.diag(eigvals)  # 16-dim matrix
U_A = qml.matrix(qml.qsvt)(A, angles, wires=range(5))  # block-encoded in 5-qubit system

qsvt_A = np.real(np.diagonal(U_A))[:16]  # retrieve transformed eigenvalues

plt.plot(a_vals, target, label="target")
plt.plot(eigvals, qsvt_A, "*", label="qsvt")

plt.legend()
plt.show()


###############################################################################
# The :class:`~pennylane.qsvt` operation is tailored for use in simulators and uses standard forms for block encodings
# and projector-controlled phase shifts. Advanced users can define their own version of these operators
# with explicit quantum circuits, and construct the resulting QSVT algorithm using the :class:`~pennylane.QSVT` template.
#
# Final thoughts
# --------------
# The original paper introducing the QSVT algorithm [#qsvt]_ described a series of potential applications,
# notably Hamiltonian simulation and slinear systems of equations. QSVT has also been used as
# a unifying framework for different quantum algorithms [#unification]_.
# One of my favourite uses of QSVT is to transform a molecular Hamiltonian by a polynomial approximation of a step function [#lintong].
# This sets large eigenvalues to zero, effectively performing a projection onto a low-energy subspace.
#
# The PennyLane team is motivated to create tools that can empower researchers
# worldwide to develop the innovations that will define the present and future of quantum computing.
# We have designed QSVT support to help you master the concepts and perform rapid prototyping of new ideas.
# We look forward to seeing the innovations that will result from your journey.
#
# References
# ----------
#
# .. [#qsvt]
#
#     András Gilyén, Yuan Su, Guang Hao Low, Nathan Wiebe,
#     "Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics",
#     `Proceedings of the 51st Annual ACM SIGACT Symposium on the Theory of Computing <https://dl.acm.org/doi/abs/10.1145/3313276.3316366>`__, 2019
#
#
# .. [#lintong]
#
#    Lin, Lin, and Yu Tong, "Near-optimal ground state preparation",
#    `Quantum 4, 372 <https://quantum-journal.org/papers/q-2020-12-14-372/>`__, 2020
#
#
# ..[#unification]
#
#    John M. Martyn, Zane M. Rossi, Andrew K. Tan, and Isaac L. Chuang,
#    "Grand Unification of Quantum Algorithms",
#    `PRX Quantum 2, 040203 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203>`__, 2021
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/juan_miguel_arrazola.txt
