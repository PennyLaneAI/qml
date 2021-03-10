r"""
Using JAX with Pennylane
========================

.. meta::
    :property="og:description": Learn how to use JAX with Pennylane.

*Author: PennyLane dev team. Posted: XX Mar 2021. Last updated: XX Mar 2021.*

JAX is an incredibly powerful deep learning library that has been gaining traction in
the deep learning community. While JAX was originally designed for classical ML,
many of its transformations are also useful for QML, and can be used directly with Pennylane.
"""

##############################################################################
# In this tutorial, we will highlight some ways to use JAX transformations with Pennylane
#
# If this is your first time reading Pennylane code, we recommend going through
# the basic tutorial (ADD LINK) first. It's all in vanilla NumPy, so you should be able to 
# easily transfer what you learn to JA when you come back.
#
# With that said, we begin by importing PennyLane and the JAX-provided version of NumPy, and
# set up a 2-wire qubit device for computations. We'll be using the ``default.qubit.jax`` device
# for this tutorial.

import jax
import jax.numpy as jnp
import pennylane as qml

dev = qml.device("default.qubit.jax", wires=2, analytic=True)

##############################################################################
# Let's start with a simple example circuit, which generates a two-qubit entangled state,
# then evaluates the expectation value of the Pauli Z operator on the first wire.
# This 


@qml.qnode(dev, interface="jax")
def circuit(param):
    # These two gates represent our QML model. 
    qml.RX(param, wires=0)
    qml.CNOT(wires=[0, 1])

    # The expval here will be the "cost function" we try to minimize.
    # Usually, this would be defined by the problem we want to solve.
    return qml.expval(qml.PauliZ(0))


##############################################################################
# We can now execute the circuit just like any other python function.

print(f"circuit(jnp.pi / 2): {circuit(jnp.pi / 2)}")

##############################################################################
# Notice that the output of the circuit is a JAX ``DeviceArray``.
# In fact, when we use the ``default.qubit.jax`` device, the entire computation 
# is done in JAX, so we can use all of the JAX tools out of the box!
#
# Let's start with a simple one. The code we wrote above is entirely continuous, so
# let's calculate its gradient.

grad_circuit = jax.grad(circuit)
print(f"grad_circuit(jnp.pi / 2): {grad_circuit(jnp.pi / 2)}")

##############################################################################
# We can then use this grad_circuit function to optimize the parameter value.

param = 0.123 # Some initial value. 

print(f"Initial param: {param}")
print(f"Initial cost: {circuit(param)}")

for _ in range(100): # Run for 100 steps.
    param -= grad_circuit(param) # Gradient decent param.

print(f"Tuned param: {param}")
print(f"Tuned cost: {circuit(param)}")

#############################################################################
# And that's QML in a nutshell! If you've ever done classical machine learning before,
# the above training loop should feel very familiar to you. The only difference is
# that we used a quantum computer (or rather, a simulation of one) as part of our
# cost calculation. In the end, almost all QML problems involve tuning some
# parameters and making some cost function go down, just like classical ML.
# While classical ML focuses on learning classical systems like language or vision,
# QML is most useful for learning quantum systems like finding chemical ground states
# or learning to sample thermal energy states. (ADD LINKS)

##############################################################################
# Batching and Evolutionary Strategies. 
# -------------------------------------
#
# We just showed how one can use gradient methods to learn a parameter value, 
# but on real QC hardware, calculating gradients can be really expensive and noisy.
# Another approach is to use evolutionary strategies (ES) to learn these parameters.
# Here, we will be using the jax.vmap transform to make running batches of circuits much easier.

# Vectorize the circuit to execute batches in parallel.
vcircuit = jax.vmap(circuit)

##############################################################################
# Now, we call call the circuit with multiple parameters at once and get back 
# batch of expectations.

print(f"Batched result: {vcircuit(jnp.array([1.234, 0.333, -0.971]))}")

##############################################################################
# Let's now setup our ES training loop. The idea is pretty simple. First, we
# calculate the expected values of each of our parameters. The cost values
# then determine the "weight" of that example. The lower the cost, the larger the weight.
# These batches are then used to generate a new set of samples. 

# Needed to do randomness with JAX.
# For more info on how JAX handles randomness, see
# https://jax.readthedocs.io/en/latest/jax.random.html
key = jax.random.PRNGKey(0)

# Generate our first set of samples
params = jax.random.normal(key, (100,))

mean = jnp.average(params)
print(f"Initial value: {mean}")
print(f"Initial cost: {circuit(mean)}")


for _ in range(200):
    costs = vcircuit(params)
    weights = jnp.exp(-costs) # Use exp(-x) here since the costs could be negative.

    mean = jnp.average(params, weights=weights)
    # The variance should decrease as we converge to a solution.
    var = jnp.sqrt(jnp.average((mean - params) ** 2, weights=weights))
    # Split the JAX key.
    key, split = jax.random.split(key)
    params = jax.random.normal(split, (100,)) * jnp.sqrt(var) + mean

print(f"Final value: {mean}")
print(f"Final cost: {circuit(mean)}")


##############################################################################
# Jit, Shots and Sampling with JAX.
# ----------------------------
# 
# JAX was designed to have experiments be as repeatable as possible. Because of this,
# JAX requires us to seed all randomly generated values (as you saw above in the above
# batching example). Sadly, the universe doesn't allow us to seed real quantum computers,
# so if we want our JAX to mimic a real QC, we'll have to handle randomness ourselves.
#
# To set the random number generating key, you'll have to pass the jax.random.PRNGKey
# when constructing the device. Because of this, if you want to use jax.jit with randomness,
# the device construction will have to happen within that jitted method.
# 
# Note: This example only applies if you are using jax.jit. Otherwise, we automatically 
# seed and reset the random number generator for you on each call.


# Let's create our circuit with randomness and a jitting.
@jax.jit
def circuit(key, param):
    # Notice how the device construction now happens within the jitted method.
    dev = qml.device("default.qubit.jax", wires=2, shots=10, prng_key=key)

    # Now we can create our qnode within the circuit function.
    @qml.qnode(dev, interface="jax")
    def my_circuit():
        qml.RX(param, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.sample(qml.PauliZ(0))
    return my_circuit()

key1 = jax.random.PRNGKey(0)
key2 = jax.random.PRNGKey(1)

# Notice that the first two runs return exactly the same results,
# The second run has different results.
print(f"key1: {circuit(key1, jnp.pi/2)}")
print(f"key1: {circuit(key1, jnp.pi/2)}")
print(f"key2: {circuit(key2, jnp.pi/2)}")

