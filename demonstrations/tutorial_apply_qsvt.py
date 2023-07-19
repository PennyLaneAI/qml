r"""
How to use QSVT (for matrix inversion)
======================================

.. meta::
    :property="og:description": Quantum Singular Value Transformation algorithm
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_intro_qsvt.png

.. related::

    tutorial_intro_qsvt Intro to QSVT
    function_fitting_qsp Function Fitting using Quantum Signal Processing

*Authors: Jay Soni, Jarrett Smalley — Posted: <date>, 2023.*

The Quantum Singular Value Transformation (QSVT) is a powerful quantum algorithm that
provides a method to apply arbitrary polynomial transformations to the singular values
of a given matrix [#qsvt]_. For a refresher on the basics of QSVT checkout our
:doc:`Intro to QSVT </demos/tutorial_intro_qsvt>` tutorial. In this demo, we provide a
practical guide on how to QSVT functionality in PennyLane, focusing on matrix inversion
as a guiding example.

|

.. figure:: ../demonstrations/apply_qsvt/qsvt2_temp.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

Preliminaries
-------------
Let's recall how to apply QSVT in a circuit. This requires two pieces of information as
input. We need to define the matrix that will be transformed and then define a set of
phase angles that determine the polynomial transformation we want to apply. For now, we
use placeholder values for the phase angles; we'll later describe how to optimize them.
The code below shows how to construct a basic QSVT circuit:
"""

import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=[0, 1])

A = np.array([[0.1, 0.2],
              [0.3, 0.4]])
phase_angles = np.array([0.0, 1.0, 2.0, 3.0])


@qml.qnode(dev)
def my_circuit(phase_angles):
    qml.qsvt(A, phase_angles, wires=[0, 1])
    return qml.state()


###############################################################################
# We can now execute the circuit and visualize it.

my_circuit(phase_angles)
print(qml.draw(my_circuit)(phase_angles))

###############################################################################
# We can inspect details by drawing the expanded circuit:

print(my_circuit.tape.expand().draw())

###############################################################################
# Now let's look at an application of QSVT --- solving a linear system of equations!
#
# Given a matrix :math:`A` and a vector :math:`\vec{b}`, we want to solve an equation
# of the form :math:`A \cdot \vec{x} = \vec{b}`. This ultimately
# requires computing:
#
# .. math::  \vec{x} = A^{-1} \cdot \vec{b}
#
# Since computing :math:`A^{-1}` is the same as applying :math:`\frac{1}{x}` to the
# singular values of :math:`A`, we can leverage the power of QSVT to apply this
# transformation.
#
# Obtaining Phase Angles
# ----------------------
# The QSVT phase angles define the polynomial transformation. While we may have a particular
# transformation in mind, it's not easy to know in general which phase angles produce it
# (in fact the angles are not unique). Here we describe two approaches to obtain the phase angles:
#
# 1. Using external packages that provide numerical angle solvers (`pyqsp <https://github.com/ichuang/pyqsp>`_)
# 2. Using PennyLane's differentiable workflow to optimize the phase angles
#
# Let's use both methods to apply a polynomial transformation which approximates:
#
# .. math::  p(a) = \frac{1}{\kappa \cdot a},
#
# where :math:`\kappa` is a re-scaling constant to ensure the function is bounded by :math:`[-1, 1]`.
#
# Phase Angles from PyQSP
# ^^^^^^^^^^^^^^^^^^^^^^^
# There are many methods for computing the phase angles (see [#phaseeval]_, [#machineprecision]_, [#productdecomp]_).
# The computed phase angles can be readily used with PennyLane's QSVT functionality as long as
# the convention used to compute them matches the convention used when applying QSVT. In Pennylane
# this is as simple as specifying the convention as a keyword argument
# :code:`qml.qsvt(A, phases, wires, convention="Wx")`. We demonstrate this by obtaining angles using the
# `pyqsp <https://github.com/ichuang/pyqsp>`_ module.
#
# The phase angles are presented below. Remember that the number of phase angles determines the degree
# of the polynomial approximation. 44 angles were generated from pyqsp which produce a transformation
# of degree 43.

phi_pyqsp = [0.02833, 0.02642, 0.07694, 0.02781, 0.15714, -0.20679, -0.10759, -0.73309, 0.0006, 0.23101, 1.16472, -2.57156, -0.34539, -0.0479, 0.75539, 0.22588, 1.8765, -2.88892, 2.91344, 0.13498, -2.91905, 2.09895, -1.04265, 0.22254, 0.13498, -0.22815, 0.25268, -1.2651, 0.22588, 0.75539, -0.0479, -0.34539, 0.57003, 1.16472, 0.23101, 0.0006, -0.73309, -0.10759, -0.20679, 0.15714, 0.02781, 0.07694, 0.02642, 1.59912]

###############################################################################
# .. note::
#
#     We generated the angles using the following api calls:
#
#     .. code-block:: bash
#
#         >>> pcoefs, scale = pyqsp.poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True)
#         >>> phi_pyqsp = pyqsp.angle_sequence.QuantumSignalProcessingPhases(2 * pcoefs, signal_operator="Wx", tolerance=0.00001)
#
#
# The initial values for :code:`kappa`, the x-axis and the target function are set. Next, iterate
# over each "a" value along the x-axis and compute the matrix for the QSVT for that "a" value. The
# top left entry of this matrix is a complex polynomial approximation whose real component corresponds
# to our target function :math:`p(a)`.

kappa = 4
x_vals = np.linspace(0, 1, 50)
target_y_vals = [1/(kappa * x) for x in np.linspace(1/kappa, 1, 50)]

qsvt_y_vals = []
for a in x_vals:
    poly_a = qml.matrix(qml.qsvt)(a, phi_pyqsp, wires=[0], convention="Wx")  # specify angles convention: `Wx`
    qsvt_y_vals.append(np.real(poly_a[0, 0]))

###############################################################################
# We plot the target function and our approximation generated from QSVT. The
# target function is only plotted from :math:`[\frac{1}{\kappa}, 1]` to match
# the bounds of the QSVT function.

import matplotlib.pyplot as plt

plt.plot(x_vals, np.array(qsvt_y_vals), label="qsvt")
plt.plot(np.linspace(1/kappa, 1, 50), target_y_vals, label="target")

plt.vlines(1/kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -0.1, 1.0, color="black")

plt.legend()
plt.show()

###############################################################################
# Yay! We were able to get an approximation of the function :math:`\frac{1}{\kappa \cdot x}` on the
# domain :math:`[\frac{1}{\kappa}, 1]`.
#
# The QSVT operation, like all other operations in PennyLane, is fully differentiable. We
# can take advantage of this as an alternate approach to obtaining the phase angles. The
# main idea is to use gradient descent optimization to learn the optimal phase angles for
# the target transformation. This approach is very versatile because it does not require
# a polynomial approximation for the target transformation. Rather we can use the target
# function directly to generate a polynomial approximation from QSVT.
#
# Phase Angles from Optimization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First, we need to define a circuit that will prepare our polynomial approximation, this
# could be just a single call to the QSVT subroutine. However, a single QSVT call will
# produce either an even or an odd parity polynomial. We can be clever and get a better
# approximation by using a sum of an even and odd polynomial as our function.
#
# The sum is achieved by using a simple LCU structure. We first split the phase angles
# into two groups (even and odd parity). Next, an ancilla qubit is prepared in equal
# superposition. We apply each QSVT operation, even or odd, conditioned on the ancilla.
# Finally, the ancilla qubit is reset. The top left block of this circuit corresponds
# to the sum of the even and odd polynomial transformations.

def sum_even_odd_circ(a, phi, ancilla_wire, wires):
    phi1, phi2 = phi[:len(phi) // 2], phi[len(phi) // 2:]

    qml.Hadamard(wires=ancilla_wire)

    qml.ctrl(qml.qsvt, control=(ancilla_wire,), control_values=(0,))(a, phi1, wires=wires)
    qml.ctrl(qml.qsvt, control=(ancilla_wire,), control_values=(1,))(a, phi2, wires=wires)

    qml.Hadamard(wires=ancilla_wire)

###############################################################################
# A random set of 101 phase angles is initialized, this implies that the transform we train
# will be a sum of a degree 49 and degree 50 polynomial.

np.random.seed(42)  # set seed for reproducibility
phi = np.random.rand(101)

###############################################################################
# Next, we need to define the loss function to optimize the phase angles over. A mean-squared
# error (MSE) loss function is used. The error is computed using samples from the domain
# :math:`[\frac{1}{\kappa}, 1]` where the target function is defined. Since the polynomial
# produced by the QSVT circuit is complex valued, we compare its real value against our
# target function.
#
# Alternatively, one could also add another term to the loss which forces the imaginary
# component to zero. In this case, we will ignore the imaginary component and instead
# use a simple LCU trick when applying the transformation for matrix inversion.
#

kappa = 5
samples_a = np.linspace(1 / kappa, 1, 70)

def target_func(x):
    return 1 / (kappa * x)

def loss_func(phi):
    sum_square_error = 0
    for s in samples_a:
        qsvt_val = qml.matrix(sum_even_odd_circ)(s, phi, ancilla_wire="ancilla", wires=[0])[0, 0]
        sum_square_error += (np.real(qsvt_val) - target_func(s)) ** 2

    return sum_square_error / len(samples_a)

###############################################################################
# Thanks to PennyLane's fully differentiable workflow, we can execute the optimization
# in just a few lines of code:

# Optimization:
cost = 1
iter = 0
opt = qml.AdagradOptimizer(0.1)

while cost > 0.01:
    iter += 1
    phi, cost = opt.step_and_cost(loss_func, phi)

    if iter % 10 == 0:
        print(f"iter: {iter}, cost: {cost}")

    if iter > 200:
        print("Iteration limit reached!")
        break

print(f"Completed Optimization!")

###############################################################################
# Now we plot the results:

samples_inv = np.linspace(1/kappa, 1, 50)
inv_x = [target_func(a) for a in samples_inv]

samples_a = np.linspace(0, 1, 100)
qsvt_y_vals = [np.real(qml.matrix(qml.qsvt)(x, phi, wires=[0])[0, 0]) for x in samples_a]

plt.plot(samples_a, qsvt_y_vals, label="qsvt")
plt.plot(samples_inv, inv_x, label="target")

plt.vlines(1/kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -0.1, 1.0, color="black")

plt.legend()
plt.show()

###############################################################################
# Awesome, we successfully optimized the phase angles!
#
# Using a higher degree polynomial and training for longer would result in a
# closer fit for both. While we used a standard loss function and optimizer,
# users have the freedom to explore any optimizer, loss function, and sampling
# scheme when training the phase angles for QSVT. Let's take everything we have
# learned and apply it to solve a linear system of equations!
#
# Solving a Linear System with QSVT
# ---------------------------------
# Our goal is to solve the equation :math:`A \cdot \vec{x} = \vec{b}` for a valid :math:`\vec{x}`.
# Let's begin by defining the specific :math:`A` and :math:`\vec{b}` quantities. Our target
# value for :math:`\vec{x}` will be the result of :math:`A^{-1} \cdot \vec{b}`.
#

A = np.array(
    [[ 0.65713691, -0.05349524,  0.08024556, -0.07242864],
     [-0.05349524,  0.65713691, -0.07242864,  0.08024556],
     [ 0.08024556, -0.07242864,  0.65713691, -0.05349524],
     [-0.07242864,  0.08024556, -0.05349524,  0.65713691]]
)

b = np.array([1, 2, 3, 4])
target_x = np.linalg.inv(A) @ b  # the solution to the system!

# Normalize states:
norm_b = np.linalg.norm(b)
normalized_b = b / norm_b

norm_x = np.linalg.norm(target_x)
normalized_x = target_x/norm_x

###############################################################################
# To solve this expression, we construct a quantum circuit that first prepares
# the normalized vector :math:`\vec{b}`. Next we apply the QSVT operation using
# the normalized matrix :math:`A`. This is equivalent to applying :math:`A^{-1}`
# to the prepared state. Finally, we return the state at the end of the circuit.
# The subset of qubits which prepared the :math:`\vec{b}` vector should now be
# transformed to represent our target state (excluding scale factors).


@qml.qnode(qml.device('default.qubit', wires=[0, 1, 2]))
def linear_system_solver_circuit(phi):
    qml.QubitStateVector(normalized_b, wires=[1, 2])
    qml.qsvt(A, phi, wires=[0, 1, 2])
    return qml.state()

trained_phi = stored_phi[-1][0]

transformed_state = linear_system_solver_circuit(trained_phi)[:4]  # first 4 entries of the state vector
rescaled_computed_x = transformed_state * kappa * norm_b
normalized_computed_x = rescaled_computed_x / np.linalg.norm(rescaled_computed_x)

print("target x:", np.round(normalized_x, 3))
print("computed x:", np.round(normalized_computed_x, 3))

A_kappa_inv = qml.matrix(qml.qsvt(A, trained_phi, wires=[0, 1, 2]))[:4, :4]  # top left block
print("\nA @ A^(-1):\n", np.round(A * kappa @ A_kappa_inv, 1))

###############################################################################
# We have solved the linear system 🎉!
#
# Notice that the target state and computed state agree well with only some slight
# deviations. Similarly, the product of :math:`A` with its computed inverse is
# approximately the identity operator.
#
# The deviations can be reduced by improving our approximation of the target function.
# Some methods to accomplish this includes:
#
# -  Using a larger degree polynomial to generate the approximate transformation
# -  Training for more iterations with more sophisticated optimizers and loss functions
# -  Using more sophisticated sampling techniques to sample along the domain of interest
#
# Conclusion
# -------------------------------
# In this demo, we showcased the :code:`qml.qsvt()` functionality in PennyLane,
# specifically to solve the problem of matrix inversion. We showed how to use
# phase angles computed with external packages and how to use PennyLane
# to optimize the phase angles directly.  We finally described how to apply qsvt to an example linear system.
#
# While the matrix we inverted was a simple example, the general problem of matrix
# inversion is often the bottleneck in many applications from regression analysis in
# financial modelling to simulating fluid dynamics for jet engine design. We hope that PennyLane can help you
# along the way to your next big discovery in quantum algorithms.
#
# References
# ----------
#
# .. [#unification]
#
#    John M. Martyn, Zane M. Rossi, Andrew K. Tan, and Isaac L. Chuang,
#    "Grand Unification of Quantum Algorithms",
#    `PRX Quantum 2, 040203 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040203>`__, 2021
#
#
# .. [#phaseeval]
#
#    Dong Y, Meng X, Whaley K, Lin L,
#    "Efficient phase-factor evaluation in quantum signal processing",
#    `Phys. Rev. A 103, 042419 – <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.042419>`__, 2021
#
#
# .. [#machineprecision]
#
#    Chao R, Ding D, Gilyen A, Huang C, Szegedy M,
#    "Finding Angles for Quantum Signal Processing with Machine Precision",
#    `arXiv, 2003.02831 <https://arxiv.org/abs/2003.02831>`__, 2020
#
#
# .. [#productdecomp]
#
#    Haah J,
#    "Product decomposition of periodic functions in quantum signal processing",
#    `Quantum 3, 190 <https://quantum-journal.org/papers/q-2019-10-07-190/>`__, 2019
#
#
# .. [#qsvt]
#
#     András Gilyén, Yuan Su, Guang Hao Low, Nathan Wiebe,
#     "Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics",
#     `Proceedings of the 51st Annual ACM SIGACT Symposium on the Theory of Computing <https://dl.acm.org/doi/abs/10.1145/3313276.3316366>`__, 2019

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/jay_soni.txt
#
# .. include:: ../_static/authors/jarrett_smalley.txt


