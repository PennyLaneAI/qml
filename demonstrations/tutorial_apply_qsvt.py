r"""
QSVT in practice
======================================

.. meta::
    :property="og:description": Quantum Singular Value Transformation algorithm
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_intro_qsvt.png

.. related::

    tutorial_intro_qsvt Intro to QSVT
    function_fitting_qsp Function Fitting using Quantum Signal Processing

*Authors: Jay Soni, Jarrett Smalley [Rolls-Royce] — Posted: <date>, 2023.*

The Quantum Singular Value Transformation (QSVT) is a quantum algorithm that
allows us to apply arbitrary polynomial transformations to the singular values
of a matrix [#qsvt]_. For a refresher on the basics of QSVT check out our
:doc:`Intro to QSVT </demos/tutorial_intro_qsvt>` tutorial. This demo, written in
collaboration between Xanadu and Rolls-Royce, provides a practical guide for the QSVT
functionality in PennyLane, focusing on matrix inversion as a guiding example.

|

.. figure:: ../demonstrations/apply_qsvt/qsvt2_temp.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

Preliminaries
-------------
Let's recall how to apply QSVT in a circuit. This requires two pieces of information as input,
the matrix to be transformed and a set of phase angles which determine the polynomial
transformation. For now, we use placeholder values for the phase angles; we'll later describe
how to optimize them. The code below shows how to construct a basic QSVT circuit on two qubits:
"""

import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=[0, 1])

A = np.array([[0.1, 0.2], [0.3, 0.4]])
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
# We can inspect details by drawing the expanded circuit. The :class:`~.pennylane.QSVT`
# operation is composed of repeated applications of :class:`~.pennylane.BlockEncode` and
# :class:`~.pennylane.PCPhase` operations.

print(my_circuit.tape.expand().draw())

###############################################################################
# Now let's look at an application of QSVT --- solving a linear system of equations.
#
# Matrix Inversion
# ----------------
# Given a matrix :math:`A` and a vector :math:`\vec{b}`, we want to solve an equation of the
# form :math:`A \cdot \vec{x} = \vec{b}`. This ultimately requires computing
# :math:`\vec{x} = A^{-1} \cdot \vec{b}`. Where for simplicity we assume that :math:`A` is
# invertible. Recall that computing :math:`A^{-1}` is the same as inverting the singular
# values of :math:`A`. We can use QSVT on :math:`A` to construct the inverse by applying a
# polynomial approximation to the transformation :math:`\frac{1}{x}`. This may seem simple
# in theory, but in practice there are a few technical details that need to be addressed.
#
# Firstly, it is difficult to approximate :math:`\frac{1}{x}` close to :math:`x = 0`, often
# requiring large degree polynomials (deeper quantum circuits). However, it turns out that
# we only need a good approximation up to the smallest singular value of the target matrix.
# The quantity `\kappa` defines the domain :math:`[\frac{1}{\kappa}, 1]` for which the
# approximation should be good.
#
# Secondly, the QSVT algorithm produces transformations which are bounded
# :math:`|P(x)| \leq 1` for :math:`x \in [-1, 1]`, but :math:`\frac{1}{x} \geq 1` on this
# domain. To remedy this, we instead approximate :math:`s \cdot \frac{1}{x}`, where :math:`s`
# is a scaling factor. Since :math:`s` is fixed beforehand, we can post-process our results
# and rescale them accordingly.
#
# Obtaining Phase Angles
# ----------------------
# The QSVT phase angles :math:`\vec{\phi}` define the polynomial transformation. While we may
# have a particular one in mind, it's not always easy to know which phase angles produce it
# (in fact the angles are not unique). Here we describe two approaches to obtain the phase
# angles:
#
# 1. Using external packages that provide numerical angle solvers (`pyqsp <https://github.com/ichuang/pyqsp>`_)
# 2. Using PennyLane's differentiable workflow to optimize the phase angles
#
# Let's use both methods to apply a polynomial transformation that approximates:
#
# .. math::  P(x) = s \cdot \frac{1}{x},
#
# Phase Angles from PyQSP
# ^^^^^^^^^^^^^^^^^^^^^^^
# There are many methods for computing the phase angles (see [#phaseeval]_,
# [#machineprecision]_, [#productdecomp]_). The computed phase angles can be readily used with
# PennyLane's QSVT functionality as long as the convention used to define the rotations
# matches the one used when applying QSVT. In Pennylane this is as simple as specifying the
# convention as a keyword argument, for example :code:`qml.qsvt(A, phases, wires, convention="Wx")`.
# We demonstrate this by obtaining angles using the `pyqsp <https://github.com/ichuang/pyqsp>`_ module.
#
# The phase angles generated from pyqsp are presented below. A :math:`kappa` of 4 was used
# and the scale factor was extracted from the pyqsp module. Remember that the number of
# phase angles determines the degree of the polynomial approximation. Below we display 44
# angles which produce a transformation of degree 43.

kappa = 4
scale = 0.10145775
phi_pyqsp = [-2.287, 2.776, -1.163, 0.408, -0.16, -0.387, 0.385, -0.726, 0.456, 0.062, -0.468, 0.393, 0.028, -0.567, 0.76, -0.432, -0.011, 0.323, -0.573, 0.82, -1.096, 1.407, -1.735, 2.046, -2.321, 2.569, -2.819, -0.011, 2.71, -2.382, 2.574, 0.028, -2.749, 2.673, 0.062, -2.685, 2.416, 0.385, -0.387, -0.16, 0.408, -1.163, -0.365, 2.426]

###############################################################################
# .. note::
#
#     We generated the angles using the following api calls:
#
#     .. code-block:: bash
#
#         >>> pcoefs, scale = pyqsp.poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True)
#         >>> phi_pyqsp = pyqsp.angle_sequence.QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", tolerance=0.00001)
#
#
# Now that the phase angles are specified, we check that they perform the correct
# transformation. We use the :func:`~.pennylane.matrix()` function to obtain the output matrix
# of the QSVT circuit. The top left entry of this matrix is a polynomial approximation whose
# real component corresponds to our target function :math:`P(x)`.

x_vals = np.linspace(0, 1, 50)
target_y_vals = [scale * (1 / x) for x in np.linspace(scale, 1, 50)]

qsvt_y_vals = []
for x in x_vals:
    poly_x = qml.matrix(qml.qsvt)(
        x, phi_pyqsp, wires=[0], convention="Wx"  # specify angles using convention `Wx`
    )
    qsvt_y_vals.append(np.real(poly_x[0, 0]))

###############################################################################
# We plot the target function and our approximation generated from QSVT. The target function
# is only plotted from :math:`[\frac{1}{\kappa}, 1]` to match the bounds of the QSVT function.

import matplotlib.pyplot as plt

plt.plot(x_vals, np.array(qsvt_y_vals), label="Re(qsvt)")
plt.plot(np.linspace(scale, 1, 50), target_y_vals, label="target")

plt.vlines(1 / kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -0.1, 1.0, color="black")

plt.legend()
plt.show()

###############################################################################
# Yay! We were able to get an approximation of the function :math:`s \cdot \frac{1}{x}` on the
# domain :math:`[\frac{1}{\kappa}, 1]`.
#
#
# Phase Angles from Optimization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The QSVT operation, like all other operations in PennyLane, is fully differentiable. We can
# take advantage of this as an alternate approach to obtaining the phase angles. The main idea
# is to use gradient descent optimization to learn the optimal phase angles for the target
# transformation. This approach is very versatile because it does not require a polynomial
# approximation for the target transformation. Rather, we can use the target function directly
# to generate a polynomial approximation from QSVT.
#
# A single QSVT circuit will produce a transformation that has a fixed parity. We can be
# clever and get a better approximation by using a sum of an even and odd polynomial.
#
# The sum is achieved by using a simple linear combination of unitaries
# (`LCU <https://codebook.xanadu.ai/H.6>`_) structure. We first split the phase angles into
# two groups (even and odd parity). Next, an ancilla qubit is prepared in equal superposition.
# We apply each QSVT operation, even or odd, conditioned on the ancilla. Finally, the ancilla
# qubit is reset.


def sum_even_odd_circ(x, phi, ancilla_wire, wires):
    phi1, phi2 = phi[: len(phi) // 2], phi[len(phi) // 2:]

    qml.Hadamard(wires=ancilla_wire)

    qml.ctrl(qml.qsvt, control=(ancilla_wire,), control_values=(0,))(x, phi1, wires=wires)
    qml.ctrl(qml.qsvt, control=(ancilla_wire,), control_values=(1,))(x, phi2, wires=wires)

    qml.Hadamard(wires=ancilla_wire)


###############################################################################
# We now randomly initialize a total of 101 phase angles. This implies that the resulting
# transformation will be a sum of polynomials with degrees 49 and 50, respectively.

np.random.seed(42)  # set seed for reproducibility
phi = np.random.rand(51)

###############################################################################
# Next, we need to define the loss function. We select a mean-squared error (MSE) loss
# function, although other choices could be made. The error is computed using samples from the
# domain :math:`[\frac{1}{\kappa}, 1]` where the target function is defined. Since the
# polynomial produced by the QSVT circuit is complex valued, we compare its real value against
# our target function. In this case, we ignore the imaginary component and instead use a simple
# LCU trick when applying the transformation for matrix inversion.

samples_x = np.linspace(1 / kappa, 1, 100)

def target_func(x):
    return scale * (1 / x)

def loss_func(phi):
    sum_square_error = 0
    for x in samples_x:
        qsvt_matrix = qml.matrix(sum_even_odd_circ)(x, phi, ancilla_wire="ancilla", wires=[0])
        qsvt_val = qsvt_matrix[0, 0]
        sum_square_error += (np.real(qsvt_val) - target_func(x)) ** 2

    return sum_square_error / len(samples_x)


###############################################################################
# Thanks to PennyLane's fully differentiable workflow, we can execute the optimization in just
# a few lines of code:

# Optimization:
cost = 1
iter = 0
opt = qml.AdagradOptimizer(0.1)

while cost > 0.5e-4:
    iter += 1
    phi, cost = opt.step_and_cost(loss_func, phi)

    if iter % 10 == 0 or iter == 1:
        print(f"iter: {iter}, cost: {cost}")

    if iter > 100:
        print("Iteration limit reached!")
        break

print(f"Completed Optimization! (final cost: {cost})")

###############################################################################
# Now we plot the results:

samples_inv = np.linspace(scale, 1, 50)
inv_x = [target_func(x) for x in samples_inv]

samples_x = np.linspace(0, 1, 100)
qsvt_y_vals = [
    np.real(qml.matrix(sum_even_odd_circ)(x, phi, "ancilla", wires=[0])[0, 0])
    for x in samples_x
]

plt.plot(samples_x, qsvt_y_vals, label="Re(qsvt)")
plt.plot(samples_inv, inv_x, label="target")

plt.vlines(1 / kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -0.1, 1.0, color="black")

plt.legend()
plt.show()

###############################################################################
# Awesome, we successfully optimized the phase angles! While we used a standard loss function
# and optimizer, users have the freedom to explore any optimizer, loss function, and sampling
# scheme when training the phase angles for QSVT.
#
# Let :math:`\hat{U_{qsvt}(\vec{\phi})}` represent the unitary matrix of the QSVT algorithm.
# Both of the methods above produce phase angles :math:`\vec{\phi}` such that:
#
# .. math::
#
#    Re(\hat{U}_{qsvt}(\vec{\phi}, x)) \approx P(x).
#
# In general, the imaginary part of this transformation will *NOT* be zero. In order to perform
# matrix inversion in a quantum circuit, we need an operator which selectively applies the real
# component while ignoring the imaginary component. Note that we can express the real part of
# a complex number as :math:`Re(p) = \frac{1}{2}(p + p^{*})`. Similarly, the operator is
# given by:
#
# .. math::
#
#    \hat{U}_{real}(\vec{\phi}) = \frac{1}{2} \ ( \hat{U}_{qsvt}(\vec{\phi}) + \hat{U}^{\dagger}_{qsvt}(\vec{\phi}) )
#
# Here we use a two term LCU to define the quantum function for this operator:


def real_u(A, phi):
    qml.Hadamard(wires="ancilla1")

    qml.ctrl(sum_even_odd_circ, control=("ancilla1",), control_values=(0,))(A, phi, "ancilla2", [0, 1, 2])
    qml.ctrl(qml.adjoint(sum_even_odd_circ), control=("ancilla1",), control_values=(1,))(A, phi, "ancilla2", [0, 1, 2])

    qml.Hadamard(wires="ancilla1")

###############################################################################
# Let's take everything we have learned and apply it to solve a linear system of equations!
#
# Solving a Linear System with QSVT
# ---------------------------------
# Our goal is to solve the equation :math:`A \cdot \vec{x} = \vec{b}`. Let's begin by
# defining the specific :math:`A` and :math:`\vec{b}` quantities:
#

A = np.array(
    [
        [0.65713691, -0.05349524, 0.08024556, -0.07242864],
        [-0.05349524, 0.65713691, -0.07242864, 0.08024556],
        [0.08024556, -0.07242864, 0.65713691, -0.05349524],
        [-0.07242864, 0.08024556, -0.05349524, 0.65713691],
    ]
)

b = np.array([1, 2, 3, 4], dtype="complex")
target_x = np.linalg.inv(A) @ b  # true solution

# Normalize states:
norm_b = np.linalg.norm(b)
normalized_b = b / norm_b

norm_x = np.linalg.norm(target_x)
normalized_x = target_x / norm_x

###############################################################################
# To solve the linear system we construct a quantum circuit that first prepares the normalized
# vector :math:`\vec{b}` in the working qubit register. Next we call :code:`real_u(A, phi)`
# function. This is equivalent to applying :math:`\frac{1}{\kappa} \cdot A^{-1}` to the
# prepared state. Finally, we return the state at the end of the circuit.
#
# The subset of qubits which prepared the :math:`\vec{b}` vector should be transformed to
# represent :math:`\vec{x}` (up to scaling factors):


@qml.qnode(qml.device("default.qubit", wires=["ancilla1", "ancilla2", 0, 1, 2]))
def linear_system_solver_circuit(phi):
    qml.QubitStateVector(normalized_b, wires=[1, 2])
    real_u(A, phi)
    return qml.state()


transformed_state = linear_system_solver_circuit(phi)[:4]  # first 4 entries of the state vector
rescaled_computed_x = transformed_state * norm_b / scale
normalized_computed_x = rescaled_computed_x / np.linalg.norm(rescaled_computed_x)

print("target x:", np.round(normalized_x, 3))
print("computed x:", np.round(normalized_computed_x, 3))

###############################################################################
# We can additionally verify that we generated the inverse matrix, by computing
# :math:`A \cdot A^{-1}`. We compute the matrix representation of the :code:`real_u()` circuit
# using :code:`qml.matrix()`. We extract the top left block and re-scale it by :math:`\kappa`
# to get :math:`A^{-1}`:

U_real_matrix = qml.matrix(real_u, wire_order=["ancilla1", "ancilla2", 0, 1, 2])(A, phi)
A_inv = U_real_matrix[:4, :4] * (1 / scale)  # top left block
print("\nA @ A^(-1):\n", np.round(A @ A_inv, 1))

###############################################################################
# We have solved the linear system 🎉!
#
# Notice that the target state and computed state agree well with only some slight deviations.
# Similarly, the product of :math:`A` with its computed inverse is the identity operator.
#
# Conclusion
# -------------------------------
# In this demo, we showcased the :func:`~.pennylane.qsvt()` functionality in PennyLane,
# specifically to solve the problem of matrix inversion. We showed how to use phase angles
# computed with external packages and how to use PennyLane to optimize the phase angles
# directly.  We finally described how to apply qsvt to an example linear system.
#
# While the matrix we inverted was a simple example, the general problem of matrix inversion
# is often the bottleneck in many applications from regression analysis in financial modelling
# to simulating fluid dynamics for jet engine design. We hope that PennyLane can help you
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
