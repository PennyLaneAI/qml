r"""Intro to QSVT
=============================================================

.. meta::
    :property="og:description": Introduction to the Quantum Singular Value Transformation algorithm
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_pulse_programming.png

Author: Juan Miguel Arrazola — Posted: 2023.

There are few quantum algorithms deserving to be placed in a hall of fame 🏆: Shor's algorithm, Grover's algorithm, quantum phase estimation;
maybe even HHL and VQE. While it's still early in its career, there is a new algorithm with prospects of achieving such celebrity status:
the quantum singular value transformation (QSVT) algorithm. If you're reading this, chances are you have at least heard of this technique and its broad
applicability.

This tutorial outlines the fundamental principles of QSVT and explains how to implement it in PennyLane. We focus on the basics;
while these techniques may appear intimidating when reading the literature, the fundamentals are relatively easy to grasp.

|

.. figure:: ../demonstrations/intro_qsvt/QSVT.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

Transforming scalars encoded in matrices
-----------------------------------------
My personal perspective on QSVT is that at its core it is a result in linear algebra. It tells us how to
transform matrices encoded in larger unitary matrices. In the simplest case,
we encode a scalar :math:`a` inside a 2x2 unitary :math:`U(a)`. By encoding we mean that the unitary depends explicitly
on :math:`a` in its matrix form. This can be done in multiple ways, for example:

.. math:: U(a) = \begin{pmatrix} a & \sqrt{1-a^2}\\
    \sqrt{1-a^2} & -a
    \end{pmatrix}.

Note that we need :math:`a` to be between -1 and 1. We then ask the crucial question that will get everything started:
what happens if we repeatedly multiply this unitary by some other unitary? 🤔

Again there are multiple choices,
for example a single-parameter diagonal unitary

.. math:: S(\phi) = \begin{pmatrix} e^{i\phi} & 0\\
    0 & e^{-i\phi}
    \end{pmatrix}.

This is known as the *signal-processing* operator.
The answer to our question is encapsulated in a result known as *quantum signal processing* (QSP). If we alternate products of
:math:`U(a)` and :math:`S(\phi)`, keeping :math:`a` fixed but varying :math:`\phi` each time, the top-left corner
of the resulting matrix is a polynomial transformation of :math:`a`. Mathematically

.. math:: S(\phi_0)\prod_{k=1}^d W(a) S(\phi_k) = \begin{pmatrix}
    P(a) & *\\
    * & *
    \end{pmatrix}.

We use the asterisk :math:`*` to indicate that right now we are not interested in these entries of the matrix.
This is a complex polynomial that can generally map real numbers to complex numbers.


The specific polynomial :math:`P(a)` has degree at most :math:`d` (determined by the number of angles), with values
between -1 and -1. This happens because every time we multiply by :math:`U(a)`, the degree of the polynomial is increased by one.
Its particular form depends on the choice of angles.
The main quantum signal processing theorem states that there exist :math:`d+1` angles that can implement _any_
complex polynomial of degree :math:`d`. This remains the case even using different conventions for the matrices, which are supported in PennyLane.
Finding the desired angles is feasible in practice, but identifying the best
methods is an active area of research. I invite you to explore our demo on quantum signal
processing if you are interested in more details. [link]

For now, let's look at a simple example of how quantum signal processing can be implemented using
the latest tools in PennyLane. We aim to perform a transformation by the Legendre polynomial
:math:`(5 x^3 - 3x)/2`, and we will be using pre-computed optimal angles.

As you will soon learn, QSP can be viewed as a special case of QSVT, so we use the `qml.qsvt()`
function to construct the output matrix from the QSP sequence. We then compare the transformation to
the target polynomial

"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt


def target_poly(a):

    return 0.5 * (5 * a ** 3 - 3 * a)


# pre-optimized angles
angles = [-0.20409113, -0.91173829, 0.91173829, 0.20409113]


def qsvt_output(a):

    # output matrix
    out = qml.matrix(qml.qsvt(a, angles, wires=[0]))

    return out[0, 0]  # top-left entry of output matrix


a_vals = np.linspace(-1, 1, 50)
qsvt = [np.real(qsvt_output(a)) for a in a_vals]  # neglect small imaginary part
target = [target_poly(a) for a in a_vals]


plt.plot(a_vals, qsvt, label="target")
plt.plot(a_vals, target, "*", label="qsvt")

plt.legend()
plt.show()


##############################################################################
# Quantum signal procesing is a result regarding multiplication of 2x2 matrices, but it is the core result
# underlying the QSVT algorithm. If you've made it this far, you're in great shape for the rest to come. 🥇
#
# Transforming matrices encoded in matrices
# ------------------------------------------
#
# Time to ask another key question: what if instead of encoding a scalar, we encode an entire matrix :math:`A`? 🧠
# This is trickier since we need to ensure that the larger matrix remains unitary. One way to achieve this
# is to generalize the construction for the scalar case. This works and it looks like this
#
# .. math:: U(A) = \begin{pmatrix} A & \sqrt{1-A A^\dagger}\\
#     \sqrt{1-A^\dagger A} & -A^\dagger
#     \end{pmatrix}.
#
# This approach works regardless of the form of :math:`A`; it doesn't even have to be a square matrix. We do
# need that :math:`A` is properly normalized such that its largest singular value is bounded by 1, otherwise
# :math:`U(A)` would not be unitary.
#
# Any such method of encoding a matrix inside a larger unitary is known as a *block encoding*. In our construction,
# the matrix :math:`A` is encoded in the top-left block, hence the name. PennyLane supports
# the `BlockEncode()` operation that follows the construction in the equation above. Let's take a look at how it works

import numpy as np

A = [[0.1, 0.2],
     [0.3, 0.4]]
U1 = qml.BlockEncode(A, wires=range(2))
print("U(A) = ", np.round(qml.matrix(U1), 2))

##############################################################################
# And for a rectangular matrix
B = [[0.5, -0.5, 0.5]]
U2 = qml.BlockEncode(B, wires=range(2))
print("U(B) = ", np.round(qml.matrix(U2), 2))


##############################################################################
# We haven't really made an explicit reference to quantum computing; everything is just linear algebra.
# Told you so! 😈 Quantum kicks in when we consider how to construct a circuit that implements
# a block-encoding unitary. We don't cover these
# methods in detail here, but a popular approach is to begin from a linear combination of unitaries for $A$
# from which we define associated :math:`\text{PREP}` (prepare) and :math:`\text{SEL}` (select) operators.
# Then the unitary
#
# .. math::  U=\text{PREP}^\dagger\cdot\text{SEL}\cdot\text{PREP},
#
# is a block-encoding of :math:`A` up to a constant factor.
#
# Time for a third key question: Can we use the same strategy as in quantum signal processing to
# polynomially transform a block-encoded matrix? Because that would be fantastic 😎
#
# For this to be possible, we first need to generalize the operator
# :math:`S(\phi)`. It turns out that this can also be done following a standard construction where we choose a
# diagonal unitary, but this time apply the phase :math:`e^{i\phi}` to one subspace determined by the block,
# and the phase :math:`e^{-i \phi}` everywhere else. For the example matrix :math:`A` in
# the code above, the corresponding operator is
#
# .. math::  \Pi(\phi)=\begin{pmatrix} e^{i\phi} & 0 & 0 & 0\\
#    0 & e^{i\phi} & 0 & 0 \\
#    0 & 0 & e^{-i\phi} & 0\\
#    0 & 0 & 0 & e^{-i\phi} \\
#    \end{pmatrix}.
#
# These are known as a *projector-controlled phase gates*. The name is less illustrative, but it
# refers to the fact that up to a global phase this operation can be viewed as a phase gate controlled
# on the block-encoding subspace. When :math:`A` is not square,
# we have to be careful and define two operators: one acting on the row subspace and another on the
# column subspace. Projector-controlled phase gates are implemented in PennyLane using the
# qml.PCPhase()` operation, which also includes a decomposition into phase-shift and Pauli X gates. Here's a
# simple example:

dim = 2
phi = np.pi/2
pi = qml.PCPhase(phi, dim, wires=range(2))
print("Pi = ", np.round(qml.matrix(pi), 2))


##############################################################################
# As you may have guessed, this generalization of QSP works! By cleverly alternating a block-encoding unitary
# and the appropriate projector-controlled phase gates, we can polynomially transform the encoded matrix.
# The result is the QSVT algorithm.
#
# Mathematically, when the polynomial degree is even, we have that
#
# .. math:: \prod_{k=1}^{d/2}\Pi_{\phi_{2k-1}}U(A)^\dagger \tilde{\Pi}_{\phi_{2k}} U(A)=
#    \begin{pmatrix} P(A) & *\\
#    * & *
#    \end{pmatrix}.
#
# The tilde is used in the projector-controlled phase gates to distingush whether they act on the column or row
# subspace of the block. The polynomial transformation of :math:`A` is defined in terms of its singular value decomposition
#
# .. math:: P(A) = \sum_k P(\sigma_k)|w_k\rangle \langle v_k|,
#
# where we use braket notation to denote the left and right singular vectors. For technical reasons,
# the sequence looks slightly different when the polynomial degree is odd:
#
# .. math:: \Pi_{\phi_1}\prod_{k=1}^{(d-1)/2}\Pi_{\phi_{2k}}U(A)^\dagger \tilde{\Pi}_{\phi_{2k+1}} U(A)=
#    \begin{pmatrix}
#    P(A) & *\\
#    * & *
#    \end{pmatrix}.
#
# As with QSP, it can be shown that there exists angles such that any polynomial transformation up to degree
# :math:`d` can be implemented by the QSVT algorithm. In fact, as long as we're careful with conventions,
# we can use the same angles regardless of the dimensions of :math:`A`. That includes the QSP case
# when it is just a scalar.
#
# The QSVT construction is an amazing result. Think about it. By using a number of operations that grows linearly with the
# degree of the target polynomial, we can transform the singular values of arbitrary block-encoded matrices
# without ever having to actually perform singular value decompositions! If the block encoding circuits can be
# implemented in polynomial time in the number of qubits, which is often the case, then the resulting
# quantum algorithm runs also in polynomial time.
#
# In PennyLane, implementing the QSVT algorithm is as simple as using `qml.qsvt()`. Let's revisit
# our previous example and transform a matrix according to a Legendre polynomial. We'll use a diagonal matrix
# with eigenvalues evenly distributed between -1 and 1, perform a polynomial transformation using the
# QSVT sequence, then plot the transformed eigenvalues of the matrix, which should fit the target polynomial


# 16-dim matrix, block-encoded in 5-qubit system
eigvals = np.linspace(-1, 1, 16)
A = np.diag(eigvals)

U_A = qml.matrix(qml.qsvt)(A, angles, wires=range(5))
qsvt_A = np.real(np.diagonal(U_A))[:16]

plt.plot(a_vals, target, label="target")
plt.plot(eigvals, qsvt_A, '*', label="qsvt")

plt.legend()
plt.show()


###############################################################################
# The `qml.qsvt()` operation is tailored for use in simulators and uses standard forms for block encodings
# and projector-controlled phase shifts. Advanced users can also define their own version of these operators,
# for instance with explicit quantum circuits, and construct the resulting QSVT algorithm using the `qml.QSVT()` template.

# Final thoughts
# --------------
# The original paper introducing the QSVT algorithm described a series of potential applications,
# notably Hamiltonian simulation and solving linear systems of equations. QSVT has also been used as
# a unifying framework for different quantum algorithms, and it can be used as a subroutine in many others.
# One of my favourite uses of QSVT is to transform a molecular Hamiltonian by a (polynomial approximation of) a step function.
# This sets large eigenvalues to zero, effectively performing a projection onto a low-energy subspace.
#
# The PennyLane development team is motivated to create tools that can empower researchers
# worldwide to develop the innovations that will define the present and future of quantum computing.
# We have designed QSVT functionality to help you in the journey to master the concepts, and we look
# forward to seeing the prototypes and inventions that will come.
#
# References
# ----------
#
# .. [#Mitei]
#
#     Oinam Romesh Meitei, Bryan T. Gard, George S. Barron, David P. Pappas, Sophia E. Economou, Edwin Barnes, Nicholas J. Mayhall
#     "Gate-free state preparation for fast variational quantum eigensolver simulations: ctrl-VQE"
#     `arXiv:2008.04302 <https://arxiv.org/abs/2008.04302>`__, 2020
#
#
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
