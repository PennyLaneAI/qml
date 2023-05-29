r"""
Methods for computing gradients of quantum circuits
=============================================================

.. meta::
    :property="og:description": Compare different methods for computing gradients of quantum circuits.

    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_gradient_methods.png

.. related::

    tutorial_backprop Quantum gradients with backpropagation
    tutorial_adjoint_diff Adjoint Differentiation
    tutorial_spsa Optimization using SPSA
    tutorial_general_parshift Reconstruct and differentiate univariate quantum functions
    tutorial_stochastic_parameter_shift The stochastic parameter-shift rule
    tutorial_here_comes_the_sun Construct and optimize circuits with SU(N) gates
    tutorial_pulse_programming101 Differentiable pulse programming with qubits in PennyLane

*Author: Frederik Wilde — Posted: May 4, 2023. Last updated: May 4, 2023.*
"""

######################################################################
# Variational quantum algorithms are widely used in quantum computing research and applications.
# In these algorithms, a :doc:`variational circuit <../glossary/variational_circuit>` is used
# to prepare a parametrized quantum state. Using measurement outcomes obtained on this state one
# then defines a cost function. By tuning the variational parameters in the circuit, the cost
# function
# can be minimized. When there is a large number of variational parameters, it is beneficial to use
# first-order methods, i.e., optimizers which make use of the gradient of the cost function.
# This gives rise to the need of computing the gradient of quantum circuits.
#
# In this tutorial, we will go over a series of methods which achieve this goal of computing gradients.
# By the end, you will:
#
# * have an overview of available methods,
# * be able to use them in PennyLane,
# * have a high-level understanding of what goes on underneath the hood in each method,
# * know where to look for more detailed and comprehensive information regarding each method.
# 
# .. figure:: ../demonstrations/gradient_methods/vqa-sketch.png
#     :align: center
#     :width: 40%
#     :target: javascript:void(0)
#
# In this tutorial, we denote our parametrized quantum circuit by :math:`U(\theta)`,
# where :math:`\theta \in \mathbb{R}^p` is the vector of variational parameters, i.e., the number of
# variational parameters is :math:`p`. We assume that our
# cost function :math:`C: \mathbb{R}^p \rightarrow \mathbb{R}` is simply given by the expectation
# value of some observable :math:`A` under the state :math:`\vert\psi(\theta)\rangle` that is prepared
# by the quantum circuit. That is
#
# .. math::  C(\theta) = \langle 0\vert U^\dagger(\theta) \,A\, U(\theta) \vert 0\rangle.
#
# Let us create a small example of a cost function in PennyLane to see how many of the methods
# discussed below can be used in practice.
#

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=2)

def qfunc(theta):
    """Apply a sequence of gates."""
    qml.RX(theta[0], wires=0)
    qml.RY(theta[1], wires=1)
    qml.CNOT(wires=(0, 1))

@qml.qnode(dev)
def cost(theta):
    """Our cost function."""
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))

params = np.array([1.0, 2.0])

qml.draw_mpl(cost)(params)
cost(params)

######################################################################
# .. note ::
#
#     In this tutorial we will not go into details about the influence of the number of measurement shots
#     used to estimate a function value. On quantum hardware, this will necessarily be something to
#     consider, since in this case one cannot compute exact expectation values, but only finite-sample
#     estimators thereof.
#

######################################################################
# Table of Contents
# -----------------
#
# .. list-table::
#    :widths: 25 25 25 25
#    :header-rows: 1
#
#    * - Method
#      - Applicable on quantum hardware
#      - Number of function evaluations (for :math:`p` variational parameters)
#      - Exact (when :math:`\#(\mathrm{shots}) \rightarrow \infty`)
#
#    * - Backpropagation
#      - No
#      - :math:`\approx 2\,{}^*`
#      - Yes
#
#    * - Finite Differences
#      - Yes
#      - :math:`p + 1` or :math:`2p`
#      - No
#
#    * - Simultaneous Perturbation Stochastic Approximation
#      - Yes
#      - :math:`\geq 2\,{}^{**}`
#      - No
#
#    * - Parameter-Shift Rule
#      - Yes
#      - :math:`2p`
#      - Yes
#
#    * - Hadamard Test
#      - Yes (with extra qubit and gates)
#      - :math:`p`
#      - Yes
#
#    * - Adjoint Method
#      - No
#      - :math:`\approx 2\,{}^*`
#      - Yes
#
#    * - General Parameter-Shift Rule
#      - Yes
#      - :math:`\leq 2p\,{}^{***}`
#      - Yes
#
#    * - Stochastic Parameter-Shift Rule
#      - Yes
#      - 
#      - No
#
#    * - Multivariate Parameter-Shift Rule
#      - Yes
#      - 
#      - Yes
#
#    * - Parameter-Shift Rule for Pulses
#      - Yes
#      - 
#      - 

######################################################################
# For a fair comparison of the advanced methods with the preceding methods we need to be precise
# about the meaning of the number :math:`p`. Given an arbitrary variational quantum circuit,
# we can always decompose it into a circuit consisting of simple gates of the type
# :math:`\mathrm{e}^{-\mathrm{i}\theta P}`, where the generator :math:`P` has exactly two
# eigenvalues. For the comparison made in the table above we denote the number of such *simple*
# gates by :math:`p`.
#
# :math:`{}^*` In these methods one does not actually compute two function evaluations, but one is able to
# obtain the gradient at a cost of roughly two function evaluations.
#
# :math:`{}^{**}` In SPSA the number :math:`d` of random perturbation vectors can be higher than :math:`1` to
# reduce the statistical error. In this case the number of function evaluation scales as :math:`2d`.
#
# :math:`{}^{***}` When the variational circuit contains complex gates, which would have to be decomposed as
# described above, one can gain a significant advantage by using the general parameter-shift rule.
#
# In the sections below we will briefly explain the basic concepts of the available methods for
# computing gradients. Along with it, we will look at code examples which demonstrate how to use the
# methods in PennyLane. We do this by specifying the ``diff_method`` keyword argument in the
# :func:`~.pennylane.qnode` decorator. The different methods have various parameters, which you can read about in
# the :mod:`~.pennylane.gradients` module of the
# PennyLane documentation. These keyword arguments can be passed to the :func:`~.pennylane.qnode` decorator along
# with the respective ``diff_method``.
#

######################################################################
# Backpropagation
# ^^^^^^^^^^^^^^^
#
# On a simulator, a parametrized quantum circuit is executed by a series of matrix multiplications. In
# this setting one can use standard automatic differentiation to compute the derivative. In machine
# learning this process is called backpropagation since one first computes the function value
# :math:`C(\theta)` and then propagates the derivative beginning from the end all the way through to
# the inputs :math:`\theta`.
#


@qml.qnode(dev, diff_method="backprop")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))

grad_fn = qml.grad(cost)
grad_fn(params)

######################################################################
# The advantage is that this method gives the exact gradient (up to machine precision) and its
# computational complexity is typically on the same order as evaluating the function itself. The
# disadvantage is that we can only use it on a simulator.
#
# *PennyLane Demo:* `Quantum gradients with backpropagation <tutorial_backprop.html>`__
#

######################################################################
# Finite Differences
# ^^^^^^^^^^^^^^^^^^
#
# The most straightforward way, which in general provides an estimate of the gradient is called
# finite differences. Here we shift the parameter of interest by a small amount in the positive and
# negative directions to approximate the difference quotient of the cost function.
#
# .. math::  \partial_i C \approx \frac{C(\theta + \varepsilon e_i) - C(\theta - \varepsilon e_i)}{2\varepsilon}
#
# This is called the central finite difference method. Additionally, there are the forward and
# backward finite differences methods, where one only shifts in one direction and combines the
# result with :math:`C(\theta)`\ , which can be used for all parameters. This reduces the overall
# number of shifts to :math:`p+1` for :math:`p` parameters, as opposed to :math:`2p` for the central
# finite differences rule.
#


@qml.qnode(dev, diff_method="finite-diff")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))

grad_fn = qml.grad(cost)
grad_fn(params)

######################################################################
# The result we obtain differs slightly from the previous one,
# which is a result of choosing a small, but finite :math:`\varepsilon` and the cost function not
# being a linear function.
#
# .. warning::
#
#     Note that this method is highly susceptible to noise, since we are trying to estimate the difference
#     between two numbers that are very close to each other. One might be tempted to simply use a greater
#     :math:`\varepsilon`, however this leads to a larger systematic error of the method. It is generally
#     not advisable to use finite differences on noisy hardware!
#

######################################################################
# Simultaneous Perturbation Stochastic Approximation (SPSA)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When the number of parameters is large, one has to evaluate the cost function many times in order to
# compute the gradient. In such a scenario SPSA can be used to reduce the number of function
# evaluations to two. However, this comes at a cost. As the name suggests, the method only gives a
# highly stochastic approximation of the true gradient.
#
# Specifically, in SPSA one samples a random perturbation vector :math:`\Delta \in \{-1, 1\}^p` which
# is used to shift the parameter into a random direction.
#
# .. math::  \partial_i C \approx \frac{C(\theta + \Delta) - C(\theta - \Delta)}{2\Delta_i}
#


@qml.qnode(dev, diff_method="spsa")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))

grad_fn = qml.grad(cost)

# Compute a lot of samples to make sure the estimator converges
# to the exact value.

grad_estimates = np.array([grad_fn(params) for _ in range(500)])
print(f"Estimate using 5 samples:   {np.mean(grad_estimates[:5], axis=0)}")
print(f"Estimate using 50 samples:  {np.mean(grad_estimates[:50], axis=0)}")
print(f"Estimate using 500 samples: {np.mean(grad_estimates, axis=0)}")

######################################################################
#
# *PennyLane Demo:* `Optimization using SPSA <tutorial_spsa.html>`__
#

######################################################################
# Parameter-Shift Rule
# ^^^^^^^^^^^^^^^^^^^^
#
# The previous two methods only deliver approximations of the gradient. More importantly, in general
# one cannot guarantee that this estimate provided by these methods is unbiased. This means, their
# expectation value, in the limit of many measurement shots does not necessarily equal to the true
# gradient.
#
# This problem is resolved by the parameter-shift rule [#Schuld]_. For simplicity, assume that the
# parametrized gates are Pauli rotations. In this case
#
# .. math::  \partial_j C = C(\theta + s e_j) - C(\theta - s e_j), \quad s = \pi / 4
#
# where :math:`e_j` is the :math:`j`-th canonical unit vector. In fact this rule can be easily
# adapted for any set of gates with generators that have two eigenvalues.
#
# Note that parameter-shift rules can also be derived for higher order derivatives, for instance,
# to compute the Hessian (via :func:`~.pennylane.gradients.param_shift_hessian`) or the
# Fisher-information matrix (via :func:`~.pennylane.metric_tensor`) of the cost function [#Mari]_.
#

@qml.qnode(dev, diff_method="parameter-shift")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))

grad_fn = qml.grad(cost)
grad_fn(params)

######################################################################
# Note how this gives the exact same result as the ``backprop`` method, i.e. it is exact. On quantum
# hardware, however, one would have to estimate the shifted cost function values with a finite number of
# shots, which leads to statistical errors.
#
#
# *PennyLane Demo:* `Quantum gradients with backpropagation <tutorial_backprop.html>`__
#
# *PennyLane Glossary:* `Parameter-shift Rule <../glossary/parameter_shift.html>`__
#

######################################################################
# Hadamard Test
# ^^^^^^^^^^^^^
#
# When writing out the derivative of :math:`C` explicitly as
#
# .. math:: \partial_j C = \langle 0 \vert \partial_j U^\dagger(\theta) \,A\, U(\theta)\vert 0 \rangle + \langle 0\vert U^\dagger(\theta) \,A\, \partial_j U(\theta)\vert 0 \rangle,
# 
# we can observe that it is equal to a very similar expression as :math:`C` itself. In fact, since
# :math:`\partial_j U^\dagger = - (\partial_j U)^\dagger` (assuming :math:`\theta_j` enters the
# circuit in the form
# :math:`\mathrm{e}^{-\mathrm{i}\theta_j P}`, where :math:`P` is an arbitary Hermitian operator),
# the derivative of the cost function is simply given
# by the imaginary part of
# 
# .. math:: 2\, \langle 0\vert U^\dagger(\theta) \,A\, \partial_j U(\theta)\vert 0 \rangle.
#
# This means we need to compute the imaginary part of the overlap of two quantum states, which can
# be done via the Hadamard test [#Guerreschi]_. Note that the Hadamard test requires an ancilla
# qubit. For our example circuit above this means we have to use a device which has three qubits.
#

dev_3qubits = qml.device("default.qubit", wires=3)

@qml.qnode(dev_3qubits, diff_method="hadamard")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))

grad_fn = qml.grad(cost)
grad_fn(params)

######################################################################
# Note that this gives us again the exact gradient, just as in the example for the parameter-shift
# rule. Now, let’s take a look at the two circuits which compute the derivative. We see that we have a controlled
# :math:`X` and a controlled :math:`Y` gate, corresponding to the two parametrized gates in our
# circuit.
#

(grad_tape1, grad_tape2), _ = qml.gradients.hadamard_grad(cost.tape)

qml.drawer.tape_mpl(grad_tape1)
qml.drawer.tape_mpl(grad_tape2)
plt.show()


######################################################################
# Adjoint Method
# ^^^^^^^^^^^^^^
#
# In the previous method we have used the fact that the computation of the gradient of our cost
# function is equal to twice the imaginary part of an overlap of the two vectors
# :math:`\vert\phi_1\rangle` and :math:`\vert\phi_2\rangle`. The adjoint method uses this fact to
# compute the gradient on a device simulator by simply computing this overlap explicitly as a series
# of matrix-vector multiplications [#Jones]_.
#
# It is very similar to ``backprop``, but is only uniquely applicable to quantum circuits. It also
# uses less memory, since it doesn’t have to store any intermediate states.
#


@qml.qnode(dev, diff_method="adjoint")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))

grad_fn = qml.grad(cost)
grad_fn(params)

######################################################################
#
# *PennyLane Demo:* `Adjoint Differentiation <tutorial_adjoint_diff.html>`__
#

######################################################################
# Advanced Methods
# ----------------
#
# The following methods are more advanced regarding their underlying theory. They allow for gradients
# to be computed when more complicated gates are present in the variational quantum circuit. We will
# motivate the mechanism behind these methods without going into all the details and point to the
# relevant sources and demos.
#

######################################################################
# General Parameter-Shift Rule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Another situation one might encounter is a gate which has many distinct eigenvalues (as opposed to
# two in the parameter-shift rule above). For instance, in the `Quantum Approximate Optimization
# Algorithm (QAOA) <tutorial_qaoa_intro.html>`__ one has to apply so-called mixers
# :math:`M(\beta) = \mathrm{e}^{-\mathrm{i}\beta B}`, where :math:`B = \sum_{i=1}^n X_i` and
# :math:`\beta\in\mathbb{R}`. Here :math:`X_i` is the Pauli X matrix, acting on the :math:`i`-th qubit
# and :math:`n` is the number of qubits.
# In order to differentiate the cost function :math:`C` with respect to
# :math:`\beta` one could decompose this gate into simple gates, where each generator only has two
# distinct eigenvalues. However, this leads to :math:`2n` function evaluations, two for each gate.
#
# In this setting, the general parameter-shift rule reduces the required resources [#Wierichs]_.
# For this we need to consider the eigenvalues :math:`\lambda_1, \ldots, \lambda_m` of our
# generator :math:`B` in decreasing order. We then determine the number :math:`R` of *unique*
# eigenvalue differences, i.e., the size of the set
# :math:`\{\lambda_1 - \lambda_2, \lambda_2 - \lambda_3, \ldots, \lambda_{m-1} - \lambda_m\}`.
# Note that in the case of the mixer :math:`B = \sum_{i=1}^n X_i` in QAOA we have
# :math:`R=1`. The general parameter-shift
# rule makes use of the fact that the cost function :math:`C` in terms of a single parameter
# :math:`\theta_i` is always given by a trigonometric polynomial
#
# .. math::  C(\theta_i) = a_0 + \sum_{l=1}^R a_l \cos(\Omega_l \theta_i) + b_l \sin(\Omega_l \theta_i),
#
# where :math:`a_l` and :math:`b_l` are appropriate real-valued coefficients and the
# :math:`\Omega_l`\ ’s are the differences of the eigenvalues of :math:`B`. The functional form of
# :math:`C(\theta_i)`, and thereby its derivative, can then be computed by trigonometric interpolation
# by evaluating the function at :math:`2R+1` points.
#
# The method can also be applied to more complicated gates, where not all eigenvalues are equidistant
# and even for gates of the type considered in the stochastic parameter-shift rule.
#
# To see how PennyLane makes use of this, we need to look a bit deeper into the way PennyLane works.
# Let’s first consider the example with the mixer :math:`B` from above on five qubits.
#

dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev, diff_method="parameter-shift")
def cost(beta):
    for i in range(5):
        qml.RX(beta, wires=i)
    return qml.expval(qml.PauliZ(1))

grad_fn = qml.grad(cost)
beta = np.array(1.0, requires_grad=True)
grad_fn(beta)

######################################################################
# We can see that it computes the gradient. But in principle this would also be possible by
# individually differentiating each of the :class:`~.pennylane.RX` gates with the parameter-shift rule. In this case,
# however, we would need 10 function evaluations. Now let us check how many function evaluations
# PennyLane actually uses by passing the :class:`~.pennylane.tape.QuantumTape` (a raw representation of the
# operations that are executed in the circuit) of our circuit to the parameter-shift method.
#

tape = cost.tape
grad_tapes, processing_fn = qml.gradients.param_shift(tape)
print(f"The number of tapes (circuit evaluations) is {len(grad_tapes)}.")

######################################################################
# We can see that we only have to execute two circuits to compute the gradient, because the
# general parameter-shift rule is used when applicable. Let’s see how it is done, just for
# completeness.
#

outputs = qml.execute(grad_tapes, dev)
sum(processing_fn(outputs))

######################################################################
#
# *PennyLane Demo:* `Generalized parameter-shift rules <tutorial_general_parshift.html>`__
#

######################################################################
# Stochastic Parameter-Shift Rule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When we have a parametrized quantum gate that is more complicated than the ones described above,
# applying the parameter-shift rule becomes expensive or even infeasible. An example of such a case
# is the gate :math:`\mathrm{e}^{-\mathrm{i}(\theta X + Y)}`, where :math:`X` and :math:`Y` could be
# simply Pauli matrices, but they could also be much more complicated generators acting on multiple
# qubits.
#
# In such a case we can make use of the fact that the derivative of the matrix exponential can be
# expressed as an integral
#
# .. math::
#
#        \frac{\partial}{\partial\theta} \mathrm{e}^{X(\theta)} = \int_0^1 \mathrm{d}s~
#        \mathrm{e}^{sX(\theta)} \frac{\partial X}{\partial \theta} \mathrm{e}^{(1-s)X(\theta)},
#
# where in this case :math:`\theta \in \mathbb{R}`.
#
# Applied to an expectation value like in our cost function :math:`C` this results in a method for
# computing the derivative. The disadvantage is that, in principle we have to compute an integral. The
# stochastic parameter-shift rule achieves this through Monte Carlo integration [#Banchi]_. In practice one draws many
# random samples :math:`s` uniformly from the interval :math:`[0, 1]` to compute an estimate of the
# exact derivative.
#

######################################################################
# .. note ::
#
#     We can intuitively motivate this formula in the following way: Imagine
#     :math:`\mathrm{e}^{X(\theta)}` being the time evolution operator corresponding to some linear
#     differential equation. In this case we could chop the evolution into :math:`N` small individual
#     steps:
#
#     .. math::
#
#
#            \mathrm{e}^{X(\theta)} \approx \mathrm{e}^{x(\theta)} \cdots \mathrm{e}^{x(\theta)},
#
#     where :math:`x(\theta) = \frac{X(\theta)}{N}`. Using the exponential series we can simplify each
#     step to
#
#     .. math::
#
#
#            \mathrm{e}^{x(\theta)} = \mathbb{1} + x(\theta) + \mathcal{O}(N^{-2}).
#
#     In the limit of large :math:`N` we can neglect all higher orders. The differentiation with respect
#     to theta then boils down to applying the product rule :math:`N` times:
#
#     .. math::
#
#
#            \frac{\partial}{\partial\theta} \mathrm{e}^{X(\theta)} \approx \sum_{j=1}^N
#            [\mathbb{1} + x(\theta)]^{(j-1)} \frac{\partial x}{\partial \theta} [\mathbb{1} + x(\theta)]^{(N-j)}.
#
#     By taking the continuum limit :math:`N \rightarrow \infty` we arrive at the integral stated above.
#

######################################################################
# 
# *PennyLane Demo:* `The stochastic parameter-shift rule <tutorial_stochastic_parameter_shift.html>`__
#

######################################################################
# Multivariate Parameter-Shift Rule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the case of more complicated gates, as in the stochastic parameter-shift rule one could also
# approach the problem in a different way. Instead of allowing for arbitrary gates, e.g. of the type
# :math:`\mathrm{e}^{-\mathrm{i}(\theta X + Y)}` one can simply use the most general gate possible
# (up to a complex phase),
# which is given by a full parametrization of the special unitary group :math:`\mathrm{SU}(N)` on
# :math:`n=\log_2(N)` qubits.
#
# In this case it is still very complicated to compute the derivative with respect to all the
# individual parameters. But the fact that one is now working with a parametrization of
# :math:`\mathrm{SU}(N)`, one can apply a trick [#Wiersema]_. Let us denote this gate by :math:`U(\theta)`, where
# now :math:`\theta \in \mathbb{R}^{N^2-1}`. First we write out the derivative of the cost function
#
# .. math::
#
#
#        \frac{\partial}{\partial \theta_l} C(\theta) =
#         \langle 0\vert \left[\frac{\partial}{\partial \theta_l} U^\dagger(\theta)\right] A U(\theta) \vert 0\rangle
#         + \langle 0\vert U(\theta) A \left[\frac{\partial}{\partial \theta_l} U(\theta)\right] \vert 0\rangle.
#
# By choosing the right parametrization of the gate :math:`U(\theta)` we can transform this cost
# function into a one-parameter derivative problem of the form
#
# .. math::
#
#
#        \frac{\partial}{\partial \theta_l} C(\theta) =
#        \frac{\mathrm{d}}{\mathrm{d}t}
#        \langle 0\vert \mathrm{e}^{-t\Omega_l} U^\dagger(\theta) \,A\, U(\theta)\, \mathrm{e}^{t\Omega_l} \vert 0\rangle,
#
# which we can solve using the general parameter-shift rule from above. What we need in this case,
# is the tangent vector :math:`\Omega_l` (i.e. an element of the special unitary algebra), defined by
# the derivative of our unitary :math:`U(\theta)`. This can be obtained via automatic differentiation
# on a classical computer, as long as :math:`N` is not too large, i.e., as long as the gate only acts
# on a few qubits. In this way we make a clever distribution of resource requirements between the
# quantum device and the classical computer.
#
# Note again, that this is only a rough sketch of the method and all the details are contained in the
# listed paper and the PennyLane demo.
#
# 
# *PennyLane Demo:* `Here comes the SU(N): multivariate quantum gates and gradients <tutorial_here_comes_the_sun.html>`__
#

######################################################################
# Parameter-Shift Rule for Pulses
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

######################################################################
# On the hardware level, at least in superconducting quantum computers, quantum gates are executed by
# manipulating qubits with finely tuned microwave pulses. On the noisy devices available in today's
# world, it is therefore intuitive to ask whether one could optimize the pulses directly, instead of
# the parameters of abstract quantum gates. It turns out, that one can indeed compute the gradient
# with respect to the pulse parameters in this setting [#Leng]_.
#
# The setting for this is very similar to the one in the stochastic parameter-shift rule. The problem
# can be seen as a parametrized Schrödinger equation.
#
# .. math::
#
#
#    \frac{\mathrm{d}}{\mathrm{d}t} \vert \psi(\theta) \rangle = H(\theta, t) \vert \psi(\theta) \rangle.
#
# When the Hamiltonian is of a specific form
#
# .. math::
#
#
#    H(\theta, t) = H_0 + \sum_{i=1}^m u_i(\theta, t) H_i,
#
# where the :math:`H_i`\ ’s are tensor products of Pauli matrices (i.e., they square to one), we can
# use the same expression that we used for differentiating the matrix exponential in the stochastic
# parameter-shift rule. Essentially, we compute the derivative of the integral of the differential
# equation, by evaluating it at random times :math:`t`.
#
# This allows us to evaluate the derivatives on quantum hardware, by randomly sampling values of
# :math:`t` at which we measure with shifted parameter values. This is in contrast to classical
# pulse-shaping methods, which would require the simulation of the time evolution, which only works up
# to a small number of qubits.
# 
# Interestingly, one of the earlier papers, leading up to the development of the parameter-shift rule
# for variational quantum algorithms, was concerned with the problem of quantum control [#Li]_. I.e.,
# the problem of tuning the parameters of control pulses which are used to operate a quantum computing
# device.
#
# *PennyLane Demo:* `Differentiable pulse programming with qubits in PennyLane <tutorial_pulse_programming101.html>`__
#

######################################################################
# Conclusion
# ----------
# We have learned that computing the gradient of quantum circuits is quite different from classical
# differentiation techniques. One can use approximate methods, such as finite differences and
# simultaneous perturbation stochastic approximation (SPSA), which are also used in classical settings
# when one cannot employ more efficient gradient methods, such as backpropagation.
# However, there are also a variety of ways to compute gradients exactly using parameter-shift
# rules. These methods make use of the fact, that cost functions arising from quantum circuits have
# a specific structure, namely they can be represented by trigonometric polynomials.
# We have seen that PennyLane enables us to use many of these methods, including ones that only work
# on simulators running on classical hardware, such as backpropagation and the adjoint method.
# To specify the gradient method of our choice, we
# simply have to set the ``diff_method`` argument in the :func:`~.pennylane.qnode` decorator.
# 
# We have seen that there are more advanced method relying on deep mathematical insights about quantum
# circuits. Don't worry if you have not understood about these methods from going through this
# tutorial. The aim here is to get a rough idea of the mathamatical basis of these methods
# and in what
# kind of situations they are relevant. For comprehensive details, there are the
# linked PennyLane demos, as well as the respective papers.

######################################################################
# References
# ----------
# .. [#Schuld]
#     Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, Nathan Killoran,
#     "Evaluating analytic gradients on quantum hardware"
#     `Phys. Rev. A 99, 032331 <https://doi.org/10.1103/PhysRevA.99.032331>`__ (2019)
#
# .. [#Mari]
#     Andrea Mari, Thomas R. Bromley, Nathan Killoran,
#     "Estimating the gradient and higher-order derivatives on quantum hardware"
#     `Phys. Rev. A 103, 012405 <https://link.aps.org/doi/10.1103/PhysRevA.103.012405>`__ (2021)
#
# .. [#Guerreschi]
#     Gian Giacomo Guerreschi, Mikhail Smelyanskiy,
#     "Practical optimization for hybrid quantum-classical algorithms"
#     `arxiv:1701.01450 <https://arxiv.org/abs/1701.01450>`__ (2017)
#
# .. [#Jones]
#     Tyson Jones, Julien Gacon,
#     "Efficient calculation of gradients in classical simulations of variational quantum algorithms"
#     `arxiv:2009.02823 <https://arxiv.org/abs/2009.02823>`__ (2020)
#
# .. [#Wierichs]
#     David Wierichs, Josh Izaac, Cody Wang, Cedric Yen-Yu Lin,
#     "General parameter-shift rules for quantum gradients"
#     `Quantum 6, 677 <https://doi.org/10.22331/q-2022-03-30-677>`__ (2022)
#
# .. [#Banchi]
#     Leonardo Banchi, Gavin E. Crooks,
#     "Measuring Analytic Gradients of General Quantum Evolution with the Stochastic Parameter Shift Rule"
#     `Quantum 5, 386 <https://doi.org/10.22331/q-2021-01-25-386>`__ (2021)
#
# .. [#Wiersema]
#     Roeland Wiersema, Dylan Lewis, David Wierichs, Juan Carrasquilla, Nathan Killoran,
#     "Here comes the SU(N): multivariate quantum gates and gradients"
#     `arXiv:2303.11355 <https://arxiv.org/abs/2303.11355>`__ (2023)
#
# .. [#Leng]
#     Jiaqi Leng, Yuxiang Peng, Yi-Ling Qiao, Ming Lin, Xiaodi Wu,
#     "Differentiable Analog Quantum Computing for Optimization and Control"
#     `arxiv:2210.15812 <https://arxiv.org/abs/2210.15812>`__ (2022)
# 
# .. [#Li]
#     Jun Li, Xiaodong Yang, Xinhua Peng, Chang-Pu Sun,
#     "Hybrid Quantum-Classical Approach to Quantum Optimal Control"
#     `Phys. Rev. Lett. 118, 150503 <https://doi.org/10.1103/PhysRevLett.118.150503>`__ (2017)

######################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/frederik_wilde.txt
