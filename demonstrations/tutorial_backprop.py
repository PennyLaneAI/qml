r"""
Quantum gradients with backpropagation
======================================

.. meta::
    :property="og:description": Compare and contrast the parameter-shift rule with backpropagation.
    :property="og:image": https://pennylane.ai/qml/_images/sphx_glr_tutorial_backprop_002.png

In PennyLane, any quantum device, whether a hardware device or a simulator, can be
trained using the :doc:`parameter-shift rule </glossary/parameter_shift>` to compute quantum
gradients. Indeed, the parameter-shift rule is ideally suited to hardware devices, as it does
not require any knowledge about the internal workings of the device; it is sufficient to treat
the device as a 'black box', and to query it with different input values in order to determine the gradient.

When working with simulators, however, we *do* have access to the internal (classical)
computations being performed. This allows us to take advantage of other methods of computing the
gradient, such as backpropagation, which may be advantageous in certain regimes. In this tutorial,
we will compare and contrast the parameter-shift rule against backpropagation, using
the PennyLane :class:`default.qubit.tf <pennylane.plugins.default_qubit_tf.DefaultQubitTF>`
device.

The parameter-shift rule
------------------------

The parameter-shift rules states that, given a variational quantum circuit :math:`U(\boldsymbol
\theta)` composed of parametrized Pauli rotations, and some measured observable :math:`\hat{B}`, the
derivative of the expectation value

.. math::

    \langle \hat{B} \rangle (\boldsymbol\theta) =
    \langle 0 \vert U(\boldsymbol\theta)^\dagger \hat{B} U(\boldsymbol\theta) \vert 0\rangle

with respect to the input circuit parameters :math:`\boldsymbol{\theta}` is given by

.. math::

   \nabla_{\theta_i}\langle \hat{B} \rangle(\boldsymbol\theta)
      =  \frac{1}{2}
            \left[
                \langle \hat{B} \rangle\left(\boldsymbol\theta + \frac{\pi}{2}\hat{\mathbf{e}}_i\right)
              - \langle \hat{B} \rangle\left(\boldsymbol\theta - \frac{\pi}{2}\hat{\mathbf{e}}_i\right)
            \right].

Thus, the gradient of the expectation value can be calculated by evaluating the same variational
quantum circuit, but with shifted parameter values (hence the name, parameter-shift rule!).

Let's have a go implementing the parameter-shift rule manually in PennyLane.
"""
import numpy as np
import pennylane as qml

# set the random seed
np.random.seed(42)

# create a device to execute the circuit on
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))


##############################################################################
# Let's test the variational circuit evaluation with some parameter input:

# initial parameters
params = np.random.random([6])

print("Parameters:", params)
print("Expectation value:", circuit(params))

##############################################################################
# We can also draw the executed quantum circuit:

print(circuit.draw())


##############################################################################
# Now that we have defined our variational circuit QNode, we can construct
# a function that computes the gradient of the :math:`i\text{th}` parameter
# using the parameter-shift rule.

def parameter_shift_term(qnode, params, i):
    shifted = params.copy()
    shifted[i] += np.pi/2
    forward = qnode(shifted)  # forward evaluation

    shifted[i] -= np.pi
    backward = qnode(shifted) # backward evaluation

    return 0.5 * (forward - backward)

# gradient with respect to the first parameter
print(parameter_shift_term(circuit, params, 0))

##############################################################################
# In order to compute the gradient with respect to *all* parameters, we need
# to loop over the index ``i``:

def parameter_shift(qnode, params):
    gradients = np.zeros([len(params)])

    for i in range(len(params)):
        gradients[i] = parameter_shift_term(qnode, params, i)

    return gradients

print(parameter_shift(circuit, params))

##############################################################################
# We can compare this to PennyLane's *built-in* parameter-shift feature by using
# the :func:`qml.grad <pennylane.grad>` function. Remember, when we defined the
# QNode, we specified that we wanted it to be differentiable using the parameter-shift
# method (``diff_method="parameter-shift"``).

grad_function = qml.grad(circuit)
print(grad_function(params)[0])

##############################################################################
# If you count the number of quantum evaluations, you will notice that we had to evaluate the circuit
# ``2*len(params)`` number of times in order to compute the quantum gradient with respect to all
# parameters. While reasonably fast for a small number of parameters, as the number of parameters in
# our quantum circuit grows, so does both
#
# 1. the circuit depth (and thus the time taken to evaluate each expectation value or 'forward' pass), and
#
# 2. the number of parameter-shift evaluations required.
#
# Both of these factors increase the time taken to compute the gradient with
# respect to all parameters.

#
# Benchmarking
# ~~~~~~~~~~~~
#
# Let's consider an example with a significantly larger number of parameters.
# We'll make use of the :class:`~pennylane.templates.StronglyEntanglingLayers` template
# to make a more complicated QNode.

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, diff_method="parameter-shift", mutable=False)
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


##############################################################################
# Note that we specify that the QNode is **immutable**. This is more restrictive
# than a standard mutable QNode (the quantum circuit cannot change/differ between
# executions); however, it reduces processing overhead.


# initialize circuit parameters
params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=15)
print(params.size)
print(circuit(params))

##############################################################################
# This circuit has 180 parameters. Let's see how long it takes to perform a forward
# pass of the circuit.

import timeit

repeat = 3
number = 10
times = timeit.repeat("circuit(params)", globals=globals(), number=number, repeat=repeat)
forward_time = min(times) / number

print(f"Forward pass (best of {repeat}): {forward_time} sec per loop")


##############################################################################
# We can now estimate the time taken to compute the full gradient vector,

# and see how this compares.

# create the gradient function
grad_fn = qml.grad(circuit)

times = timeit.repeat("grad_fn(params)", globals=globals(), number=number, repeat=repeat)
backward_time = min(times) / number

print(f"Gradient computation (best of {repeat}): {backward_time} sec per loop")


##############################################################################
# Based on the parameter-shift rule, we expect that the amount of time to compute the quantum
# gradients should be approximately :math:`2p\Delta t_{f}` where :math:`p` is the number of
# parameters and :math:`\Delta t_{f}` if the time taken for the forward pass. Let's verify this:

print(2 * forward_time * params.size)


##############################################################################
# Backpropagation
# ---------------
#
#
# An alternative to forward-mode autodifferentiation is `reverse-mode autodifferentiation
# <https://en.wikipedia.org/wiki/Reverse_accumulation>`__; unlike forward-mode differentiation, it
# requires a *single* forward pass of the differentiable function to compute
# the gradient of all variables, at the expense of increased memory usage. During the forward pass,
# the results of all intermediate subexpressions are stored; the computation is

# then traversed *in reverse*, with the gradient computed by repeatedly applying the chain rule. In
# most classical machine learning settings (where we are training loss functions consisting of a large
# number of parameters with a single output), reverse-mode autodifferentiation is the
# preferred method of autodifferentiation---the reduction in computational time enabling larger and
# more complex models to be successfully trained. The backpropagation algorithm is a particular
# special-case of reverse-mode autodifferentiation, which has helped lead to the machine learning
# explosion we see today.
#
# In quantum machine learning, however, the inability to store and utilize the results of
# *intermediate* quantum operations on hardware remains a barrier; while reverse-mode
# autodifferentiation works fine for small quantum simulations, only the
# parameter-shift rule can be used to compute gradients on quantum hardware directly. Nevertheless,
# when training quantum models via classical simulation, it's useful to explore the regimes where
# reverse-mode differentiation may be a better choice than the parameter-shift rule.
#
# Benchmarking
# ~~~~~~~~~~~~
#
# When creating a QNode, :doc:`PennyLane supports various methods of differentiation
# <code/api/pennylane.qnode>`, including ``"parameter-shift"`` (which we used previously),
# ``"finite-diff"``, ``"reversible"``, and ``"backprop"``. While ``"parameter-shift"`` works with all devices,
# simulator and hardware, ``"backprop"`` will only work for specific simulator devices that are
# designed to support reverse-mode differentiation.
#
# One such device is :class:`default.qubit.tf <pennylane.plugins.default_qubit_tf.DefaultQubitTF>`.
# This device is a pure state-vector simulator like ``default.qubit``; however, unlike
# ``default.qubit``, is written using TensorFlow rather than NumPy. As a result, it supports
# classical backpropagation when using the TensorFlow interface.

import tensorflow as tf

dev = qml.device("default.qubit.tf", wires=4)

##############################################################################
# When defining the QNode, we specify ``diff_method="backprop"`` to ensure that
# we are using backpropagation mode. Note that this will be the *default differentiation
# mode* when ``interface="tf"``.


@qml.qnode(dev, diff_method="backprop", interface="tf")
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

# initialize circuit parameters
params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=15)
params = tf.Variable(params)
print(circuit(params))

##############################################################################
# Let's see how long it takes to perform a forward pass of the circuit.

import timeit

repeat = 3
number = 10
times = timeit.repeat("circuit(params)", globals=globals(), number=number, repeat=repeat)
forward_time = min(times) / number
print(f"Forward pass (best of {repeat}): {forward_time} sec per loop")


##############################################################################
# Comparing this to the forward pass from ``default.qubit``, we note that there is some potential
# overhead from using TensorFlow. We can now time a backwards pass.

with tf.GradientTape(persistent=True) as tape:
    res = circuit(params)

times = timeit.repeat("tape.gradient(res, params)", globals=globals(), number=number, repeat=repeat)
backward_time = min(times) / number
print(f"Backward pass (best of {repeat}): {backward_time} sec per loop")

##############################################################################
# Unlike with the parameter-shift rule, the time taken to perform the backwards pass appears
# of the order of a single forward pass!
#
# Time comparison
# ---------------
#
# Let's compare the two differentiation approaches as the number of trainable parameters
# in the variational circuit increases, by timing both the forward and backward pass
# as the number of layers is allowed to increase.
#
# We'll create two devices; one using ``default.qubit`` for the parameter-shift
# rule, and ``default.qubit.tf`` for backpropagation. For convenience, we'll use the TensorFlow
# interface when creating both QNodes.

dev_shift = qml.device("default.qubit", wires=4)
dev_backprop = qml.device("default.qubit.tf", wires=4)

def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

##############################################################################
# We'll continue to use the same ansatz as before, but to reduce the time taken
# to collect the data, we'll reduce the number and repetitions of timings per data
# point. Below, we loop over a variational circuit depth ranging from 0 (no gates/
# trainable parameters) to 20. Each layer will contain :math:`3N` parameters, where
# :math:`N` is the number of wires (in this case, :math:`N=4`).

repeat = 2
number = 3

forward_shift = []
backward_shift = []
forward_backprop = []
backward_backprop = []

for depth in range(0, 21):
    params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=depth)
    num_params = params.size
    params = tf.Variable(params)

    qnode_shift = qml.QNode(circuit, dev_shift, interface="tf", mutable=False)
    qnode_backprop = qml.QNode(circuit, dev_backprop, interface="tf")

    # forward pass timing
    # ===================

    # parameter-shift
    t = timeit.repeat("qnode_shift(params)", globals=globals(), number=number, repeat=repeat)
    forward_shift.append([num_params, min(t) / number])

    # backprop
    t = timeit.repeat("qnode_backprop(params)", globals=globals(), number=number, repeat=repeat)
    forward_backprop.append([num_params, min(t) / number])

    if num_params == 0:
        continue

    # Backward pass timing
    # ====================

    # parameter-shift
    with tf.GradientTape(persistent=True) as tape:
        res = qnode_shift(params)

    t = timeit.repeat("tape.gradient(res, params)", globals=globals(), number=number, repeat=repeat)
    backward_shift.append([num_params, min(t) / number])

    # backprop
    with tf.GradientTape(persistent=True) as tape:
        res = qnode_backprop(params)

    t = timeit.repeat("tape.gradient(res, params)", globals=globals(), number=number, repeat=repeat)
    backward_backprop.append([num_params, min(t) / number])

backward_shift = np.array(backward_shift).T
backward_backprop = np.array(backward_backprop).T
forward_shift = np.array(forward_shift).T
forward_backprop = np.array(forward_backprop).T

##############################################################################
# We now import matplotlib, and plot the results.

from matplotlib import pyplot as plt
plt.style.use("bmh")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(*backward_shift, '.-', label="Parameter-shift")
ax.plot(*backward_backprop, '.-', label="Backprop")
ax.set_ylabel("Time (s)")
ax.set_xlabel("Number of parameters")
ax.legend()

plt.show()

##############################################################################
# .. raw:: html
#
#     <br>
#
# We can see that the computational time for the parameter-shift rule increases with
# increasing number of parameters, as expected, whereas the computational time
# for backpropagation appears much more constant, with perhaps a minute linear increase
# with :math:`p`. Note that the plots are not perfectly linear, with some 'bumpiness' or
# noisiness. This is likely due to low-level operating system jitter, and
# other environmental fluctuations---increasing the number of repeats can help smooth
# out the plot.
#
# For a better comparison, we can scale the time required for computing the quantum
# gradients against the time taken for the corresponding forward pass:

backward_shift[1] /= forward_shift[1, 1:]
backward_backprop[1] /= forward_backprop[1, 1:]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(*backward_shift, '.-', label="Parameter-shift")
ax.plot(*backward_backprop, '.-', label="Backprop")

# perform a least squares regression to determine the linear best fit/gradient
# for the normalized time vs. number of parameters
x = backward_shift[0]
m_shift, c_shift = np.polyfit(*backward_shift, deg=1)
m_back, c_back = np.polyfit(*backward_backprop, deg=1)

ax.plot(x, m_shift * x + c_shift, '--', label=f"{m_shift:.2f}p{c_shift:+.2f}")
ax.plot(x, m_back * x + c_back, '--', label=f"{m_back:.2f}p{c_back:+.2f}")

ax.set_ylabel("Normalized time")
ax.set_xlabel("Number of parameters")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()

plt.show()

##############################################################################
# .. raw:: html
#
#     <br>
#
# We can now see clearly that there is constant overhead for backpropagation with
# ``default.qubit.tf``, but the parameter-shift rule scales as :math:`\sim 2p`.
#
# Footnotes
# ---------
#
# .. [#]
#
#     This is a particularly naïve implementation of forward-mode autodifferentiation; in practice,
#     classical forward-mode differentiation is `implemented using dual numbers
#     <https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers>`__,
#     which reduces the number of forward sweeps, but with an increase in memory.
