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

Transforming matrices that encode scalars
-----------------------------------------
My personal perspective on QSVT is that at its core it is a result in linear algebra. It tells us how to
transform matrices encoded in larger unitary matrices. In the simplest case,
we encode a scalar :math:`a` inside a 2x2 unitary :math:`U(a)`. By encoding we mean that the unitary depends explicitly
on :math:`a` in its matrix form. This can be done in multiple ways, for example:

.. math:: U(a) = \begin{pmatrix} a & \sqrt{1-a^2}
    \sqrt{1-a^2} & -a
    \end{pmatrix}.

Note that we need :math:`a` to be between -1 and 1. We then ask the crucial question that will get everything started:
what happens if we repeatedly multiply this unitary by some other unitary? 🤔

Again there are multiple choices,
for example a single-parameter diagonal unitary

.. math:: S(\phi) = \begin{pmatrix} e^{i\phi} & 0
    0 & e^{-i\phi}
    \end{pmatrix}.

The answer to our question is encapsulated in a result known as *quantum signal processing*. If we alternate products of
:math:`U(a)` and :math:`S(\phi)`, keeping :math:`a` fixed but varying :math:`\phi` each time, the top-left corner
of the resulting matrix is a polynomial transformation of :math:`a`. Mathematically

.. math:: S(\phi_0)\prod_{k=1}^d W(a) S(\phi_k) = \begin{pmatrix}
    P(a) & *\\
    * & *
    \end{pmatrix}.

We use the asterisk :math:`*` to indicate that right now we are not interested in these entries of the matrix.


The specific polynomial :math:`P(a)` has degree at most :math:`d` (determined by the number of angles), with values
between -1 and -1. This happens because every time we multiply by :math:`U(a)`, we increase the degree of the polynomial.
Its particular form depends on the choice of angles.
In fact, the main quantum signal processing theorem states that there exist :math:`d+1` angles that can implement _any_ polynomial
of degree :math:`d`. This remains the case even using different conventions for the matrices.
Finding the desired angles is typically feasible in practice, but identifying the best
methods to do so is an active area of research. I invite you to explore our demo on quantum signal
processing if you are interested in more details. [link]

For now, let's look at a simple example of how quantum signal processing can be implemented using
the latest tools in PennyLane.

While quantum signal procesing is just a result regarding multiplication of 2x2 matrices, it is the core result
underlying the QSVT algorithm. So if you've made it this far, you're in great shape for the rest to come.



"""

import pennylane as qml

# code here


##############################################################################
# Text starts again

# code here

##############################################################################
# Text here. References look like this [#Mitei]_
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
