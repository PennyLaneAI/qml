r"""Tutorial: Methods for computing gradients of quantum circuits
=============================================================

Frederik Wilde May 4, 2023
"""

######################################################################
# For variational quantum circuits it is often desirable to use first-order optimization, i.e.,
# methods which make use of the gradient of the cost function. Computing derivatives of quantum
# circuits can be done in a variaty of different ways. In this tutorial we will go over a list of
# different methods and discuss their advantages and disadvantages. We will also see how to use these
# methods in Pennylane.
#
# For the purpose of this tutorial we denote our parametrized quantum circuit by :math:`U(\theta)`,
# where :math:`\theta \in \mathbb{R}^p` is the vector of variational parameters. We assume that our
# cost function :math:`C: \mathbb{R}^p \rightarrow \mathbb{R}` is simply given by the expectation
# value of some observable :math:`A` under the state :math:`\vert\psi(\theta)\rangle` that is prepared
# by the quantum circuit. That is
#
# :raw-latex:`\begin{equation}
#     C(\theta) = \langle 0\vert U^\dagger(\theta) A U(\theta) \vert 0\rangle.
# \end{equation}`
#
# Let us create a small example of a cost function in Pennylane to see how many of the methods
# discussed below can be used in practice.
#

import pennylane as qml
import pennylane.numpy as np
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
print(f"Output: {cost(params)}")

######################################################################
# Remark on Code Examples
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# In the sections below we will briefly explain the basic concepts of the available methods for
# computing gradients. Along with it, we will look at a code example which demonstrates how to use the
# method in Pennylane. We do this by specifying the ``diff_method`` keyword argument in the
# :func:`qml.qnode <~pennylane.qnode>` decorator. The different methods have various parameters, which you can read about in
# the :mod:`gradients <~pennylane.gradients>` module of the
# Pennylane documentation. These keyword arguments can be passed to the ``qml.qnode`` decorator along
# with the respective ``diff_method``.
#

######################################################################
# Remark on Measurement Shots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In this tutorial we will not go into details about the influence of the number of measurement shots
# used to estimate a function value. On quantum hardware, this will necessarily be something to
# consider, since in this case one cannot compute exact expectation values, but only finite-sample
# estimators thereof.
#

######################################################################
# Table of Contents
# ~~~~~~~~~~~~~~~~~
#
# .. raw:: html
#
#    <table>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <th>
#
# Method
#
# .. raw:: html
#
#    </th>
#
# .. raw:: html
#
#    <th>
#
# Applicable on Quantum hardware
#
# .. raw:: html
#
#    </th>
#
# .. raw:: html
#
#    <th>
#
# Number of function evaluations (for :math:`p` parametrized gates)
#
# .. raw:: html
#
#    </th>
#
# .. raw:: html
#
#    <th>
#
# Exact (when #\ :math:`(\mathrm{shots})\rightarrow\infty`)
#
# .. raw:: html
#
#    </th>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Backpropagation
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# no
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# :math:`\approx 2~{}^{[1]}`
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Finite Differences
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# :math:`p+1` or :math:`2p`
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# no
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Simultaneous Perturbation Stochastic Approximation
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# 2 or more :math:`{}^{[2]}`
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# no
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Parameter Shift Rule
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# :math:`2p`
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Hadamard Test
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes (w/ extra qubit and gates)
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# :math:`p`
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Adjoint Method
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# no
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# :math:`\approx 2~{}^{[1]}`
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <th>
#
# Advanced Methods
#
# .. raw:: html
#
#    </th>
#
# .. raw:: html
#
#    <th>
#
# .. raw:: html
#
#    </th>
#
# .. raw:: html
#
#    <th>
#
# .. raw:: html
#
#    </th>
#
# .. raw:: html
#
#    <th>
#
# .. raw:: html
#
#    </th>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Stochastic Parameter Shift Rule
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# no
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Generalized Parameter Shift Rule
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# :math:`\leq 2p~{}^{[3]}`
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Multivariate Parameter Shift Rule for :math:`\mathrm{SU}(N)` Gates
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    <tr>
#
# .. raw:: html
#
#    <td>
#
# Pulse-Level Differentiation
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# yes
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    <td>
#
# .. raw:: html
#
#    </td>
#
# .. raw:: html
#
#    </tr>
#
# .. raw:: html
#
#    </table>
#

######################################################################
# The number of parameters :math:`p` is to be understood in the following way. Assume we have a
# variational quantum circuit consisting of arbitarily complex parametrized gates. If we were to
# decompose this circuit into gates of the type :math:`\mathrm{e}^{-\mathrm{i}\theta P}`, where
# :math:`P` has exactly two unique eigenvalues, then :math:`p` refers to the number of such gates in
# the circuit. Many types of circuit ansätze are already of this form, but in order to compare the
# advanced methods to the basic methods in a fair way, we need to take into account all types of
# variational quantum circuits.
#
# [1] In these methods one does not actually compute two function evaluations, but one is able to
# obtain the gradient at a cost of roughly two function evaluations.
#
# [2] In SPSA the number :math:`d` of random perturbation vectors can be higher than :math:`1` to
# reduce the statistical error. In this case the number of function evaluation scales as :math:`2d`.
#
# [3] When the variational circuit contains complex gates, which would have to be decomposed as
# described above, one can gain a significant advantage by using the generalized parameter shift rule.
#

######################################################################
# Backpropagation
# ===============
#
# On a simulator, a parametrized quantum circuit is executed by a series of matrix multiplications. In
# this setting one can use standard automatic differentiation to compute the derivative. In machine
# learning this process is called backpropagation since one first computes the function value
# :math:`C(\theta)` and then propagates the derivative beginning from the end all the way through to
# the inputs :math:`\theta`.
#
# The advantage is that this method gives the exact gradient (up to machine precision) and its
# computational complexity is typically on the same order as evaluating the function itself. The
# disadvantage is that we can only use it on a simulator.
#


@qml.qnode(dev, diff_method="backprop")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))


grad_fn = qml.grad(cost)
print(grad_fn(params))

######################################################################
# References
# ~~~~~~~~~~
#
# -  Pennylane Demo: `Quantum gradients with
#    backpropagation <https://pennylane.ai/qml/demos/tutorial_backprop.html>`__
#

######################################################################
# Finite Differences
# ==================
#
# The most straight-forward way, which in general provides an estimate of the gradient is called
# finite differences. Here we shift the parameter of interest by a small amount to approximate the
# difference quotient of the cost function.
#
# .. math::  \partial_i C \approx \frac{C(\theta + \varepsilon e_i) - C(\theta - \varepsilon e_i)}{2\varepsilon}
#
# This is called the central finited-difference method. Additionally, there is the forward and
# backward finite-differences method, where one only shifts in one direction. This reduces the overall
# number of shifts to :math:`p+1` for :math:`p` parameters, as opposed to :math:`2p` for the central
# finite-differences rule.
#


@qml.qnode(dev, diff_method="finite-diff")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))


grad_fn = qml.grad(cost)
print(grad_fn(params))

######################################################################
# WARNING
# ~~~~~~~
#
# Note that this method is highly susceptible to noise, since we are trying to estimate the difference
# between two numbers that are very close to each other. One might be tempted to simply use a greater
# :math:`\varepsilon`, however this leads to a larger systematic error of the method. It is generally
# not advisable to use finite differences on noisy hardware!
#

######################################################################
# Simultanious Perturbation Stochastic Approximation (SPSA)
# =========================================================
#
# When the number of parameters is large, one has to evaluate the cost function many times in order to
# compute the gradient. In such a scenario SPSA can be used to reduce the number of function
# evaluations to two. However, this comes at a cost. As the name suggests, the method only gives a
# highly stochastic approximation of the true gradient.
#
# Specifically in SPSA one samples a random perturbation vector :math:`\Delta \in \{-1, 1\}^p` which
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
# References
# ~~~~~~~~~~
#
# -  Pennylane Demo: `Optimization using SPSA <https://pennylane.ai/qml/demos/tutorial_spsa.html>`__
#

######################################################################
# Parameter Shift Rule
# ====================
#
# The previous two methods only deliver approximations of the gradient. More importantly, in general
# one cannot guarantee that this estimate provided by these methods is unbiased, i.e., their
# expectation value in the limit of many measurement shots does not equal to the true gradient.
#
# This problem is resolved by the parameter shift rule. In its simplest form it can be formulated for
# a circuit that is parametrized by gates with a two-eigenvalue generator, i.e.,
# :math:`\mathrm{e}^{-\mathrm{i}\theta_j P}`. For instance :math:`P` could be a Pauli matrix. Assume
# that the difference between the two eigenvalues is :math:`2r`. Then
#
# .. math::  \partial_j C = r \big[C(\theta + se_j) - C(\theta - se_j)\big], \quad s = \frac{\pi}{4r},
#
# where :math:`e_j` is the :math:`j`-th canonical unit vector.
#


@qml.qnode(dev, diff_method="parameter-shift")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))


grad_fn = qml.grad(cost)
print(grad_fn(params))

######################################################################
# Note how this gives the exact same result as the ``backprop`` method, i.e. it is exact. On quantum
# hardware, however, one would have to estimate the shifted cost function values with finite number of
# shots, which leads to statistical errors.
#

######################################################################
# References
# ~~~~~~~~~~
#
# -  M. Schuld, et al., `Phys. Rev. A 99, 032331 <https://doi.org/10.1103/PhysRevA.99.032331>`__
#    (2019)
# -  Pennylane Demo: `Quantum gradients with
#    backpropagation <https://pennylane.ai/qml/demos/tutorial_backprop.html>`__
#

######################################################################
# Hadamard Test
# =============
#
# When writing out the derivative of :math:`C` explicitly, we can observe that it is equal to a very
# similar expression as :math:`C` itself. For simplicity, assume here that
# :math:`U(\theta) = V\mathrm{e}^{-\mathrm{i} \theta X}W` and :math:`\theta \in \mathbb{R}`. We then
# get
#
# .. math::
#
#
#        \partial_i C = \mathrm{i} \big[
#            \langle \psi \vert \mathrm{e}^{\mathrm{i} \theta X} X \tilde{A} \mathrm{e}^{-\mathrm{i} \theta X} \vert \psi \rangle
#            - \langle \psi \vert  \mathrm{e}^{\mathrm{i} \theta X} \tilde{A} X \mathrm{e}^{-\mathrm{i} \theta X} \vert \psi \rangle
#        \big]
#        = 2\mathrm{Im} \big[
#            \langle \psi \vert \mathrm{e}^{\mathrm{i} \theta X} \tilde{A} X \mathrm{e}^{-\mathrm{i} \theta X} \vert \psi \rangle
#        \big],
#
# where :math:`\vert \psi \rangle = W \vert 0 \rangle` and :math:`\tilde{A} = V^\dagger A V`. Hence,
# we need to compute the imaginary part of the overlap of two states: \*
# :math:`\vert \phi_1 \rangle = VX\mathrm{e}^{-\mathrm{i} \theta X}W \vert 0\rangle` \*
# :math:`\vert \phi_2 \rangle = V\mathrm{e}^{-\mathrm{i} \theta X}W \vert 0\rangle`.
#
# This can be done via the Hadamard test. Note that the Hadamard test requires an ancilla qubit. For
# our example circuit above this means we have to use a device which has three qubits.
#

dev_3qubits = qml.device("default.qubit", wires=3)


@qml.qnode(dev_3qubits, diff_method="hadamard")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))


grad_fn = qml.grad(cost)
print(grad_fn(params))

######################################################################
# Let’s take a look at the two circuits which compute the derivative. We see that we have a controlled
# :math:`X` and a controlled :math:`Y` gate, corresponding to the two parametrized gates in our
# circuit.
#

(grad_tape1, grad_tape2), _ = qml.gradients.hadamard_grad(cost.tape)

qml.drawer.tape_mpl(grad_tape1)
qml.drawer.tape_mpl(grad_tape2)
plt.show()

######################################################################
# References
# ~~~~~~~~~~
#
# -  G. G. Guerreschi, et al., `arxiv:1701.01450 <https://arxiv.org/abs/1701.01450>`__ (2017)
#

######################################################################
# Adjoint Method
# ==============
#
# In the previous method we have used the fact that the computation of the gradient of our cost
# function is equal to twice the imaginary part of an overlap of the two vectors
# :math:`\vert\phi_1\rangle` and :math:`\vert\phi_2\rangle`. The adjoint method uses this fact to
# compute the gradient on a device simulator by simply computing this overlap explicitly as a series
# of matrix-vector multiplications.
#
# It is very similar to ``backprop``, but is only uniquely applicable to quantum circuits. It also
# uses less memory, since it doesn’t have to store any intermediate states.
#


@qml.qnode(dev, diff_method="adjoint")
def cost(theta):
    qfunc(theta)
    return qml.expval(qml.PauliZ(1))


grad_fn = qml.grad(cost)
print(grad_fn(params))

######################################################################
# References
# ~~~~~~~~~~
#
# -  T. Jones, et al., `arxiv:2009.02823 <https://arxiv.org/abs/2009.02823>`__ (2020)
# -  Pennylane Demo: `Adjoint
#    Differentiation <https://pennylane.ai/qml/demos/tutorial_adjoint_diff.html>`__
#

######################################################################
# Advanced Methods
# ================
#
# The following methods are more advanced regarding their underlying theory. They allow for gradients
# to be computed when more complicated gates are present in the variational quantum circuit. We will
# motivate the mechanism behind these methods without going into all the details and point to the
# relevent sources and demos.
#

######################################################################
# Stochastic Parameter Shift Rule
# ===============================
#
# When we have a parametrized quantum gate that is more complicated than the ones described above,
# applying the parameter-shift rule becomes expensive or even infeasible. An example for such a case
# is the gate :math:`\mathrm{e}^{-\mathrm{i}(\theta X + Y)}`, where :math:`X` and :math:`Y` could be
# simply Pauli matrices, but they could also be much more complicated generators acting an multiple
# qubits.
#
# In such a case we can make use of the fact that the derivative of the matrix exponential can be
# expressed as an integral
#
# .. math::
#
#
#        \frac{\partial}{\partial\theta} \mathrm{e}^{X(\theta)} = \int_0^1 \mathrm{d}s~
#        \mathrm{e}^{sX(\theta)} \frac{\partial X}{\partial \theta} \mathrm{e}^{(1-s)X(\theta)},
#
# where in this case :math:`\theta \in \mathbb{R}`.
#
# Applied to an expectation value like in our cost function :math:`C` this results in a method for
# computing the derivative. The disadvantage is that, in principle we have to compute an integral. The
# stochastic parameter shift rule achieves this by Monte-Carlo integration. In practice one draws many
# random samples :math:`s` uniformly from the interval :math:`[0, 1]` to compute an estimate of the
# exact derivative.
#

######################################################################
# Remark
# ~~~~~~
#
# We can intuitively motivate this formula in the following way: Imagine
# :math:`\mathrm{e}^{X(\theta)}` being the time evolution operator corresponding to some linear
# differential equation. In this case we could chop the evolution into :math:`N` small individual
# steps:
#
# .. math::
#
#
#        \mathrm{e}^{X(\theta)} \approx \mathrm{e}^{x(\theta)} \cdots \mathrm{e}^{x(\theta)},
#
# where :math:`x(\theta) = \frac{X(\theta)}{N}`. Using the exponential series we can simplify each
# step to
#
# .. math::
#
#
#        \mathrm{e}^{x(\theta)} = \mathbb{1} + x(\theta) + \mathcal{O}(N^{-2}).
#
# In the limit of large :math:`N` we can neglect all higher orders. The differentiation with respect
# to theta then boils down applying the product rule :math:`N` times:
#
# .. math::
#
#
#        \frac{\partial}{\partial\theta} \mathrm{e}^{X(\theta)} \approx \sum_{j=1}^N
#        [\mathbb{1} + x(\theta)]^{(j-1)} \frac{\partial x}{\partial \theta} [\mathbb{1} + x(\theta)]^{(N-j)}.
#
# By taking the continuum limit :math:`N \rightarrow \infty` we arrive at the integral stated above.
#

######################################################################
# References
# ~~~~~~~~~~
#
# -  L. Banchi, et al., `arxiv:2005.10299 <https://arxiv.org/abs/2005.10299>`__ (2020)
#

######################################################################
# Generalized Parameter Shift Rule
# ================================
#
# Another situation one might encounter is a gate which has many distinct eigenvalues (as opposed to
# two in the parameter shift rule above). For instance in the Quantum Approximate Optimization
# Algorithm (QAOA) one has to apply so called mixers
# :math:`M(\beta) = \mathrm{e}^{-\mathrm{i}\beta B}`, where :math:`B = \sum_{i=1}^n X_i` and
# :math:`\beta\in\mathbb{R}`. In order to differentiate the cost function :math:`C` with respect to
# :math:`\beta` one could decompose this gate into simple gates, where each generator only has two
# distinct eigenvalues. However, this leads to :math:`2n` function evaluations, two for each gate.
#
# In this setting, the generalized parameter shift rule reduces the required resources. Here, we only
# need to consider the number :math:`R` of differences between eigenvalues of the generator :math:`B`.
# Note that in the case of a simple mixer in QAOA we have :math:`R=1`. The generalized parameter shift
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
# and even for gates of the type considered in the stochastic parameter shift rule.
#
# To see how Pennylane makes use of this, we need to look a bit deeper into the way Pennylane works.
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
print(grad_fn(beta))

######################################################################
# We can see that it computes the gradient. But in principle this would also be possible by
# individually differentiating each of the ``RX`` gates with the parameter shift rule. In this case,
# however, we would need 10 function evaluations. Now let us check how many function evaluation
# Pennylane actually uses by passing the ``qml.tape.QuantumTape`` (a raw representation of the
# operations that are executed in the circuit) of our circuit to the parameter-shift method.
#

tape = cost.tape
grad_tapes, processing_fn = qml.gradients.param_shift(tape)
print(f"The number of tapes (circuit evaluations) is {len(grad_tapes)}.")

######################################################################
# We can see that we only have to execute two circuits to compute the gradient, because the
# generalized parameter shift rule is used when applicable. Let’s see how it is done, just for
# completeness.
#

outputs = qml.execute(grad_tapes, dev)
sum(processing_fn(outputs))

######################################################################
# References
# ~~~~~~~~~~
#
# -  D. Wierichs, et al., `Quantum 6, 677 <https://doi.org/10.22331/q-2022-03-30-677>`__ (2022)
# -  Pennylane Demo: `Generalized parameter-shift
#    rules <https://pennylane.ai/qml/demos/tutorial_general_parshift.html>`__
#

######################################################################
# Multivariate Parameter Shift Rule for :math:`\mathrm{SU}(N)` Gates
# ==================================================================
#
# In the case of more complicated gates, as in the stochastic parameter shift rule one could also
# approach the problem in a different way. Instead of allowing for arbitrary gates, e.g. of the type
# :math:`\mathrm{e}^{-\mathrm{i}(\theta X + Y)}` one can simply use the most general gate possible,
# which is given by a full parametrization of the special unitary group :math:`\mathrm{SU}(N)` on
# :math:`n=\log_2(N)` qubits.
#
# In this case it is still very complicated to compute the derivative with respect to all the
# individual parameters. But the fact that one is now working with a parametrization of
# :math:`\mathrm{SU}(N)`, one can apply a trick. Let us denote this gate by :math:`U(\theta)`, where
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
# function into a one-parameter derivative-problem of the form
#
# .. math::
#
#
#        \frac{\partial}{\partial \theta_l} C(\theta) =
#        \frac{\mathrm{d}}{\mathrm{d}t}
#        \langle 0\vert \mathrm{e}^{-t\Omega_l} U^\dagger(\theta) \,A\, U(\theta)\, \mathrm{e}^{t\Omega_l} \vert 0\rangle,
#
# which we can solve using the generalized parameter shift rule from above. What we need in this case,
# is the tangent vector :math:`\Omega_l` (i.e. an element of the special unitary algebra), defined by
# the derivative of our unitary :math:`U(\theta)`. This can be obtained via automatic differentiation
# on a classical computer, as long as :math:`N` is not too large, i.e., as long as the gate only acts
# on a few qubits. In this way we make a clever distribution of resource requirements between the
# quantum device and the classical computer.
#
# Note again, that this is only a rough sketch of the method and all the details are contained in the
# listed paper and the Pennylane demo.
#

######################################################################
# References
# ~~~~~~~~~~
#
# -  [1] R. Wiersema, et al., `arXiv:2303.11355 <https://arxiv.org/abs/2303.11355>`__ (2023)
# -  Pennylane Demo: `Here comes the SU(N): multivariate quantum gates and
#    gradients <https://pennylane.ai/qml/demos/tutorial_here_comes_the_sun.html>`__
#

######################################################################
# Pulse-Level Differentiation
# ===========================
#

######################################################################
# On the hardware level, at least in superconducting quantum computers, quantum gates are executed by
# manipulating qubits with finely tuned microwave pulses. On the noisy devices available in todays
# world, it is therefore intuitive to ask whether one could optimize the pulses directly, instead of
# the parameters of abstract quantum gates. It turns out, that one can indeed compute the gradient
# with respect to the pulse parameters in this setting.
#
# The setting for this is very similar to the one in the stochastic parameter shift rule. The problem
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
# parameter shift rule. Essentially, we compute the derivative of the integral of the differential
# equation, by evaluating it at random times :math:`t`.
#
# This allows us to evaluate the derivatives on quantum hardware, by randomly sampling values of
# :math:`t` at which we measure with shifted parameter values. This is in contrast to classical
# pulse-shaping methods, which would require the simulation of the time evolution, which only works up
# to a small number of qubits.
#

######################################################################
# References
# ~~~~~~~~~~
#
# -  J. Leng, et al., `arxiv:2210.15812 <https://arxiv.org/abs/2210.15812>`__ (2022)
# -  Pennylane Demo: `Differentiable pulse programming with qubits in
#    PennyLane <https://pennylane.ai/qml/demos/tutorial_pulse_programming101.html>`__
#

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/frederik_picture.txt
