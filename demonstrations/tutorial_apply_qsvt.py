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

The Quantum Singular Value Transformation (QSVT) is a powerful tool in the world of quantum
algorithms [#qsvt]_. This algorithm provides a method to apply arbitrary polynomial
transformations onto the singular values of a given matrix; for a refresher on the basics
of QSVT checkout our other :doc:`demo </demos/tutorial_intro_qsvt>`. In this demo provide a
practical guide on how to use the PennyLane built-in QSVT functionality, focusing on the
problem of matrix inversion as a guiding example.

|

.. figure:: ../demonstrations/apply_qsvt/qsvt2_temp.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

Preliminaries
-------------
First let's recall how to apply QSVT in a circuit. This subroutine requires two pieces
of information as input. First we need to define a matrix for which QSVT is applied and
then define a set of phase angles that determine the polynomial transformation we want to
apply to the input matrix. There are several ways to find the optimal values of the phase
angles for a desired transformation. For now, we use some random values.
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
# The circuit implements QSVT with details that can be obtained by drawing the expanded
# circuit with:

print(my_circuit.tape.expand().draw())

###############################################################################
# Now let's look at an application of QSVT; solving a linear system of equations!
#
# Given a matrix :math:`A` and a vector :math:`\vec{b}`, we want to solve an equation
# of the form :math:`A \cdot \vec{x} = \vec{b}` for a valid :math:`\vec{x}`. This ultimately
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
# The phase angles are the keys that unlock the power of QSVT because they define the
# polynomial transformation. While we may have a particular transformation in mind, it's not
# easy to know in general which phase angles produce it (in fact the angles are not unique).
# Here we present two approaches to obtain the phase angles:
#
# -  Using external packages that provide numerical angle solvers (eg. `pyqsp <https://github.com/ichuang/pyqsp>`_)
# -  Using PennyLane's differentiable workflow and optimization to train the optimal phase angles
#
# Let's use both methods to apply the polynomial transformation:
#
# .. math::  p(a) = \frac{1}{\kappa \cdot a},
#
# where :math:`\kappa` is a normalization constant.
#
# Phase Angles from PyQSP
# ^^^^^^^^^^^^^^^^^^^^^^^
# There are many methods for computing the phase angles (see [#phaseeval]_, [#machineprecision]_, [#productdecomp]_).
# This is an active area of research, and we welcome users to take advantage of these resources.
# The computed phase angles can be readily used with PennyLane's QSVT functionality as long as
# the convention used to compute them matches the convention used when applying QSVT. In Pennylane
# this is as simple as specifying a convention as a keyword argument
# :code:`qml.qsvt(A, phases, wires, convention="Wx")`. We demonstrate by obtaining angles using the
# `pyqsp <https://github.com/ichuang/pyqsp>`_ module made by the group of Isaac Chuang.
#
# .. note::
#
#     The following code is part of the pyqsp api; we directly present and use the angles computed below:
#
#     .. code-block:: bash
#
#         >>> import pyqsp
#         >>> from pyqsp import angle_sequence
#         >>>
#         >>> kappa = 4
#         >>> pg = pyqsp.poly.PolyOneOverX()
#         >>> pcoefs, scale = pg.generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True)
#         >>> phiset = angle_sequence.QuantumSignalProcessingPhases(2 * pcoefs, signal_operator="Wx", tolerance=0.00001)
#         >>> phiset
#         [0.02832731632871888, 0.026424896063092316, 0.07693910473903123, 0.027814771098110203, 0.15714250322985934, -0.20678861058826206, -0.10759126877392514, -0.7330914006591908, 0.0005992635704724636, 0.23101434004257504, 1.1647186453026217, -2.5715583495913537, -0.34538771581163785, -0.04789856537188175, 0.7553851256205089, 0.22588274963677946, 1.8764968290283173, -2.888915420007339, 2.91343938625266, 0.13497805753495362, -2.919053234558157, 2.098946659011221, -1.0426459833500663, 0.22253943289532108, 0.1349780142346375, -0.22815322001050908, 0.2526772512477222, -1.2650958285182, 0.2258827273166304, 0.7553851012848343, -0.047898504861988433, -0.3453877134100831, 0.5700343428361438, 1.16471864865887, 0.23101428631502152, 0.0005992569127086789, -0.7330914072726091, -0.10759128593842307, -0.20678861658360936, 0.1571425120509769, 0.02781477441772423, 0.07693910671460868, 0.026424896049645152, 1.599123642563105]
#
#

import matplotlib.pyplot as plt

kappa = 4
phiset = [0.02832731632871888, 0.026424896063092316, 0.07693910473903123, 0.027814771098110203, 0.15714250322985934, -0.20678861058826206, -0.10759126877392514, -0.7330914006591908, 0.0005992635704724636, 0.23101434004257504, 1.1647186453026217, -2.5715583495913537, -0.34538771581163785, -0.04789856537188175, 0.7553851256205089, 0.22588274963677946, 1.8764968290283173, -2.888915420007339, 2.91343938625266, 0.13497805753495362, -2.919053234558157, 2.098946659011221, -1.0426459833500663, 0.22253943289532108, 0.1349780142346375, -0.22815322001050908, 0.2526772512477222, -1.2650958285182, 0.2258827273166304, 0.7553851012848343, -0.047898504861988433, -0.3453877134100831, 0.5700343428361438, 1.16471864865887, 0.23101428631502152, 0.0005992569127086789, -0.7330914072726091, -0.10759128593842307, -0.20678861658360936, 0.1571425120509769, 0.02781477441772423, 0.07693910671460868, 0.026424896049645152, 1.599123642563105]

x_vals = np.linspace(0, 1, 50)
y_vals = [1/(kappa * x) for x in np.linspace(1/kappa, 1, 50)]

qsvt_y_vals = []
for a in x_vals:
    poly_a = qml.matrix(qml.qsvt)(a, phiset, wires=[0], convention="Wx")  # Note make sure the conventions match!
    qsvt_y_vals.append(np.real(poly_a[0, 0]))  # angles were generated with `Wx` convention!

plt.plot(x_vals, np.array(qsvt_y_vals), label="qsvt")
plt.plot(np.linspace(1/kappa, 1, 50), y_vals, label="target")

plt.vlines(1/kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -0.1, 1.0, color="black")

plt.legend()
plt.show()

###############################################################################
# Yay! We were able to get an approximation of the function :math:`\frac{1}{\kappa \cdot x}` on the
# domain :math:`[\frac{1}{\kappa}, 1]`. Now lets explore an alternate approach
# to obtaining the phase angles.
#
# Phase Angles from Optimization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The QSVT operation, like all other operations in PennyLane, is fully differentiable! We
# can take advantage of this as an alternate approach to obtaining the phase angles. The
# main idea is to use gradient descent optimization to learn the optimal phase angles for
# our target transformation. This approach is very versatile because it does not require
# a polynomial approximation for the target transformation. Rather we can use the target
# function directly to generate a polynomial approximation from QSVT.
#

np.random.seed(42)  # set seed for reproducibility

# Initialize parameters and functions:
kappa = 5
cost = 1
epoch = 0
phi = np.random.rand(100)                  # Poly degree 100 approximation
samples_a = np.linspace(1 / kappa, 1, 70)  # Taking 70 samples in (1/kappa, 1)

def target_func(x):
    """The transformation we want to perform with QSVT"""
    return 1 / (kappa * x)

def mean_squared_error(phi):
    """A MSE loss function between the QSVT value to the target function value."""
    sum_square_error = 0
    for s in samples_a:
        qsvt_val = qml.matrix(qml.qsvt)(s, phi, wires=[0])[0, 0]  # block encode in the top left entry
        sum_square_error += (np.real(qsvt_val) - target_func(s)) ** 2 + (np.imag(qsvt_val)) ** 2

    norm = 1 / len(samples_a)
    return norm * sum_square_error

###############################################################################
# The phase angles will be trained to minimize the mean-squared error loss function
# defined above. This function will force the real part of the transformation to
# match our target function, while the imaginary part is forced to 0.
#
# Thanks to PennyLane's fully differentiable workflow, we can execute the optimization
# in just a few lines of code:

# Optimization:
opt = qml.AdagradOptimizer(0.1)
stored_phi = [(phi, cost)]

while cost > 0.01:
    epoch += 1
    phi, cost = opt.step_and_cost(mean_squared_error, phi)

    if epoch % 10 == 0:
        print(f"iter: {epoch}, cost: {cost}")
        stored_phi.append((phi, cost))

    if epoch > 250:
        print("Iteration limit reached!")
        break

print(f"Completed Optimization!")

###############################################################################
# Now we plot the results:

###############################################################################
samples_inv = np.linspace(1/kappa, 1, 50)
inv_x = [target_func(a) for a in samples_inv]

optimized_phi, final_cost = stored_phi[-1]

samples_a = np.linspace(0, 1, 100)
qsvt_y_vals = [qml.matrix(qml.qsvt)(x, optimized_phi, wires=[0])[0, 0] for x in samples_a]
qsvt_y_vals_re = [np.real(y) for y in qsvt_y_vals]
qsvt_y_vals_im = [np.imag(y) for y in qsvt_y_vals]

plt.plot(samples_a, qsvt_y_vals_re, label="Re(qsvt)")
plt.plot(samples_a, qsvt_y_vals_im, label="Im(qsvt)")
plt.plot(samples_inv, inv_x, label="1/(kappa * x)")

plt.vlines(1/kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -0.1, 1.0, color="black")

print(f"Optimized cost: {final_cost}")
plt.legend()
plt.show()

###############################################################################
# Notice that the real component of our transformation approximates our target
# function while the imaginary component is suppressed to 0 in the same domain.
# Using a higher degree polynomial and training for longer would result in a
# closer fit for both. While we used a standard loss function and optimizer,
# users have the freedom to explore any optimizer, loss function and sampling
# scheme when training the phase angles for QSVT. Now let's take everything we have
# learned and apply it to solve a linear system of equations!
#
# Solving a Linear System with QSVT
# ---------------------------------
# Our goal is to solve the equation :math:`A \cdot \vec{x} = \vec{b}` for a valid :math:`\vec{x}`.
# Lets begin by defining the specific :math:`A` and :math:`\vec{b}` quantities. Our target
# value for :math:`\vec{x}` will be the result of :math:`A^{-1} \cdot \vec{b}`.
#

A = np.array(
    [[ 0.51532902, -0.01702354,  0.03901916, -0.0705793 ],
     [-0.01702354,  0.51532902, -0.0705793 ,  0.03901916],
     [ 0.03901916, -0.0705793 ,  0.51532902, -0.01702354],
     [-0.0705793 ,  0.03901916, -0.01702354,  0.51532902]]
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
# transformed to represent our target state (excluding scale factors)!


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
# Yay 🎉! We have solved the linear system!
#
# Notice that the target state and computed state agree well with only some slight
# deviations. Similarly, the product of :math:`A` with its computed inverse is
# approximately the identity operator. The deviations can be attributed to the
# error in our approximation of the :math:`\frac{1}{x}` function. Likewise, the
# non-zero imaginary component in our compute state is caused because the imaginary
# component in our approximation isn't completely 0 on the entire domain.
#
# The deviations can be reduced by improving our approximation of the target function.
# Some methods to accomplish this include:
#
# -  Using a larger degree polynomial to generate the approximate transformation
# -  Training for more iterations with more sophisticated optimizers and loss functions
# -  Using more sophisticated sampling techniques to sample along the domain of interest
#
# Below we provide some plots showing how the error and cost evolved through the training
# process:

lst_epoch = 10 * np.arange(len(stored_phi))
lst_cost = [cost for _, cost in stored_phi]

lst_rescaled_computed_x = [
    linear_system_solver_circuit(phi)[:4] * kappa * norm_b for phi, _ in stored_phi
]

lst_norm = [
    np.linalg.norm(
        normalized_x - (computed_x / np.linalg.norm(computed_x))
    ) for computed_x in lst_rescaled_computed_x
]

###############################################################################
# Plotting the error between the target state and input state over the course of
# the training process:

plt.plot(lst_epoch, lst_norm, "--.b")

plt.xlabel("Training Iterations")
plt.ylabel("Norm(target_state - computed_state)")
plt.show()

###############################################################################
# Plotting the cost value as we trained the phase angles:

plt.plot(lst_epoch, lst_cost, "--.b")

plt.xlabel("Training Iterations")
plt.ylabel("Training Cost")
plt.show()

###############################################################################
# Final Thoughts and Applications
# -------------------------------
# In this demo, we showcased the new :code:`qml.qsvt()` functionality in PennyLane,
# specifically to solve the problem of matrix inversion. We showed that externally
# computed phase angles could be used with this functionality via the "convention"
# keyword argument. We also utilized the fully differentiable nature of the operation
# to optimize the phase angles directly.
#
# We then used this functionality to explicitly solve an example linear system!
# While the matrix we inverted was a trivial example, the general problem of matrix
# inversion is often the bottleneck in many applications from regression analysis in
# financial modelling to simulating fluid dynamics for jet engine design. These
# applications would benefit from efficient matrix inversion.
#
# There are also many unanswered questions when it comes to optimally computing phase angles
# for arbitrary transformations:
#
# -  What degree approximation do I need for a given error threshold?
# -  How many iterations do I need to train my model for?
# -  What kind of loss function should I use?
#
# This is just the tip of the iceberg, there are also many other applications of QSVT in
# general which have yet to be fully explored (hamiltonian simulation, eigenvalue
# filtering, etc.)! We hope you feel inspired to take up the mantel and explore these
# unanswered questions yourself, and we hope that PennyLane can help you along the way
# to the next big discovery. 
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


