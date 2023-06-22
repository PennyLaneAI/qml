r"""How to use QSVT
===================

.. meta::
    :property="og:description": Quantum Singular Value Transformation algorithm
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_intro_qsvt.png

.. related::

    tutorial_intro_qsvt Intro to QSVT
    function_fitting_qsp Function Fitting using Quantum Signal Processing

*Author: Jay Soni — Posted: June 22, 2023.*

The Quantum Singular Value Transformation (QSVT) is a powerful tool in the world of quantum
algorithms [#qsvt]_. We have explored the basics of QSVT in :doc:`this demo </demos/tutorial_intro_qsvt>`
and here we show you how to use the PennyLane built-in functionality to implement QSVT in a quantum
circuit. We apply QSVT to solve some interesting problems including matrix inversion.

First let's recall how to implement QSVT in a circuit.
"""

import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(angles):
    qml.qsvt(matrix, angles, wires=[0, 1])
    return qml.state()

###############################################################################
# The QSVT subroutine requires two pieces of information as input. First we need to define a matrix
# for which QSVT is applied and then define a set of phase angles that determine the polynomial
# transformation that we want to apply to the input matrix. There are several ways to find the
# optimal values of the phase angles for a desired problem. For now, we use some random values.

matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
angles = np.array([0.01, 1.21, 2.14, 3.38])

###############################################################################
# We can now execute the circuit and visualize it.

print(circuit(angles))

print(qml.draw(circuit)(angles))

###############################################################################
# The circuit implements QSVT with details that can be obtained by drawing the expanded circuit
# with:

print(circuit.tape.expand().draw())

###############################################################################
# The circuit contains alternating layers of projector-controlled phase gates parameterized by the
# phase angles and the unitary, and its conjugate transpose, that block-encodes the input matrix.
#
# Let's now go back to the phase angles and see how we can obtain them for a problem of interest.

# Obtaining phase angles
# ----------------------
# The specific transformation performed by QSVT depends on the phase angles used. It is,
# unfortunately, not trivial to compute the phase angles which will produce a desired
# transformation. Here we discuss two approaches to obtain the phase angles. The first method uses
# an algorithmic procedure implemented in the `pyqsp <https://github.com/ichuang/pyqsp>`_ library
# and the second method leverages the PennyLane's fully differentiable workflow to optimize the
# phase angles. Let's use both methods to apply the polynomial transformation:
#
# .. math::  p(a) = \frac{1}{\kappa} a^{-1},
#
# where :math:`\kappa` is a constant.
#
# In the optimization approach, we define a set of random initial values for the phase angles. Then,
# we optimize the angles to reduce the difference between the polynomial transformation performed by
# QSVT and the actual polynomial computed manually. To simply the process, we assume that the input
# matrix has a single component and perform the transformation for a set of such scalar values. We
# first define a function that computes the desired polynomial for a given value.

def poly(a):
    return 1 / (kappa * a)  # target polynomial

###############################################################################
# We now define a function that performs QSVT on a given value.

def qsvt(a, angles):
    out = qml.matrix(qml.qsvt(a, angles, wires=[0]))
    return out[0, 0]  # top-left entry

###############################################################################
# Then we define a cost function that we minimize to compute the optimal phase angles.

def cost_function(angles, values):

    sum_square_error = 0

    for a in values:
        qsvt_val = qsvt(a, angles)
        sum_square_error += (np.real(qsvt_val) - poly(a)) ** 2 + (np.imag(qsvt_val)) ** 2

    return sum_square_error / len(values)

###############################################################################
# We can now define a set of initial values for the phase angles and a set of values to apply the
# polynomial transformations to them.

kappa = 10
angles = np.random.rand(4)
values = np.linspace(1/kappa, 1, 50) # Taking 50 samples in (1/kappa, 1) to train

###############################################################################
# We can now perform the optimization.

# Optimization:
opt = qml.AdagradOptimizer(0.3)
cost = 1
n_steps = 50
angles = []

for n in n_steps:

    phi, cost = opt.step_and_cost(cost_function, phi)

    if n % 5 == 0:
        print(f"iter: {n}, cost: {cost}")

###############################################################################
# The
#
# Conclusions
# -----------
# The
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
# .. [#unification]
#
#    John M. Martyn, Zane M. Rossi, Andrew K. Tan, and Isaac L. Chuang,
#    "Grand Unification of Quantum Algorithms",
#    `PRX Quantum 2, 040203 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203>`__, 2021
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/jay_soni.txt
