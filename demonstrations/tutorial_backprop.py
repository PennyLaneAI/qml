r"""
Quantum gradients with backpropagation
======================================

.. meta::
    :property="og:description": Compare and contrast the parameter-shift rule with backpropagation.
    :property="og:image": https://pennylane.ai/qml/_images/sphx_glr_tutorial_rosalin_002.png

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

The parameter-shift rules states that, given a variational quantum circuit :math:`U(\boldsymbol \theta)`
and some measured observable :math:`\hat{B}`, the derivative of the expectation value

.. math::

	\langle \hat{B} \rangle (\boldsymbol\theta) =
	\langle 0 \mid U(\boldsymbol\theta)^\dagger \hat{B} U(\boldsymbol\theta) \mid 0\rangle

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
# If you count the number of quantum evalutions, you will notice that we had to
# evaluate the circuit ``2*len(params)`` number of times in order to compute the
# quantum gradient with respect to all parameters. While reasonably fast for a small;
# number of parameters, as the number of parameters in our quantum circuit grows,
# so does both the circuit depth (and thus 'forward' circuit evaluation) as well as
# the 'backward' pass (the time taken to compute the gradient with respect to all parameters).
#
# Let's consider an example with a significantly larger number of parameters.
# We'll make use of the :class:`~pennylane.templates.StronglyEntanglingLayers` template
# to make a more complicated QNode.

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

# initialize circuit parameters
params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=15)
print(params.size)

##############################################################################
# This circuit has 180 parameters. Let's see how long it takes to perform a forward
# pass of the circuit.

import timeit

repeat = 3
number = 10
times = timeit.repeat("circuit(params)", globals=globals(), number=number, repeat=repeat)
print(f"best of {repeat}: {min(times) / number} sec per loop")


##############################################################################
# We can now time a backwards pass (the time taken to compute all gradients),
# and see how this compares.

grad_fn = qml.grad(circuit)
times = timeit.repeat("grad_fn(params)", globals=globals(), number=number, repeat=repeat)
print(f"best of {repeat}: {min(times) / number} sec per loop")

##############################################################################
# Backprop
# --------
#
# Let's repeat the above experiment, but this time using the
# :class:`default.qubit.tf <pennylane.plugins.default_qubit_tf.DefaultQubitTF>`
# device. This device is a pure state-vector simulator like ``default.qubit``,
# however unlike ``default.qubit`` is written using TensorFlow rather than NumPy.
# As a result, it supports classical backpropagation when using the
# TensorFlow interface.

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

##############################################################################
# Let's see how long it takes to perform a forward pass of the circuit.

import timeit

repeat = 3
number = 10
times = timeit.repeat("circuit(params)", globals=globals(), number=number, repeat=repeat)
print(f"best of {repeat}: {min(times) / number} sec per loop")


##############################################################################
# This is approximately the same time required when using the NumPy-based default qubit,
# with some potential overhead from using TensorFlow. We can now time a backwards pass.

with tf.GradientTape(persistent=True) as tape:
    res = circuit(params)

times = timeit.repeat("tape.gradient(res, params)", globals=globals(), number=number, repeat=repeat)
print(f"best of {repeat}: {min(times) / number} sec per loop")

##############################################################################
# Unlike with the parameter-shift rule, there is now only a **constant overhead**
# compared to the forward pass!

##############################################################################
# Reversible backprop
# -------------------
#

import tensorflow as tf

dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev, diff_method="reversible")
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

# initialize circuit parameters
params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=15)
print(circuit.__class__.__base__)

##############################################################################
# Let's see how long it takes to perform a forward pass of the circuit.

import timeit

repeat = 3
number = 10
times = timeit.repeat("circuit(params)", globals=globals(), number=number, repeat=repeat)
print(f"best of {repeat}: {min(times) / number} sec per loop")


##############################################################################
# We can now time a backwards pass.

grad_fn = qml.grad(circuit)
times = timeit.repeat("grad_fn(params)", globals=globals(), number=number, repeat=repeat)
print(f"best of {repeat}: {min(times) / number} sec per loop")

