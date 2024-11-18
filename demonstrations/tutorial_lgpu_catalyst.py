"""
How to use Catalyst to just-in-time (QJIT) execute a PennyLane hybrid program on NVIDIA GPUs
============================================================================================

"""

######################################################################
# `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__ is a quantum just-in-time (QJIT) compiler
# and runtime framework to compile, optimize and execute PennyLane hybrid (quantum-classical) programs
# on quantum simulators and hardware. When combined with
# `the lightning.gpu device <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__,
# it enables GPU-accelerated simulation of QJIT compiled programs by leveraging the NVIDIA ``cuQuantum`` SDK.
# This feature is particularly interesting to accelerate quantum execution in Catalyst, bringing us one step
# closer to making quantum computing more accessible and efficient.
#
# In this tutorial, ... 
# 

######################################################################
# QJIT on Lightning-GPU
# ---------------------
# 
# With Catalyst, it is as simple as installing the latest version of `Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/installation.html>`__
# on a CUDA capable GPU of generation SM 7.0 (Volta) and greater, and decorating your existing PennyLane hybrid workflow with ``@qml.qjit``:
# 
# ``` console
#   pip install pennylane-lightning[gpu] pennylane-catalyst
# ```
# 

import pennylane as qml
from jax import numpy as jnp

dev = qml.device("lightning.gpu", wires=2)

@qml.qjit
@qml.qnode(dev)
def circuit(x, y, z):
    qml.RX(jnp.sin(x), wires=[y + 1])
    qml.RY(x ** 2, wires=[z])
    qml.CNOT(wires=[y, z])
    return qml.expval(qml.PauliZ(wires=[z]))

# Under the hood, when circuit is called, Catalyst compiles the entire program in the first call,
# and executes the quantum evolution on the targeted GPU.
#
# Similar to ``lightning.qubit``, the ``lightning.gpu`` device supports quantum circuit gradients
# using the adjoint differentiation method by default. It really shines when used to compile an entire
# workflow using QJIT-compatible optimizers:
#

import jax
import jaxopt

dev = qml.device("lightning.gpu", wires=2)

@qml.qnode(dev, diff_method="adjoint")
def circuit(params):
    qml.Hadamard(0)
    qml.RX(jnp.sin(params[0]) ** 2, wires=1)
    qml.CRY(params[0], wires=[0, 1])
    qml.RX(jnp.sqrt(params[1]), wires=1)
    return qml.expval(qml.PauliZ(1))

@qml.qjit
def cost(param):
    diff = qml.grad(circuit, argnum=0)
    return circuit(param), diff(param)[0]

@qml.qjit
def optimization():
    # initial parameter
    params = jnp.array([0.54, 0.3154])

    # define the optimizer
    opt = jaxopt.GradientDescent(cost, stepsize=0.4, value_and_grad=True)
    update = lambda i, args: tuple(opt.update(*args))

    # perform optimization loop
    state = opt.init_state(params)
    (params, _) = jax.lax.fori_loop(0, 100, update, (params, state))

    return params

# Note that ``lightning.gpu`` offers full feature parity with both ``lightning.qubit`` and ``lightning.kokkos``,
# allowing users to seamlessly swap devices without any changes to your program.
#

######################################################################
# Catalyst: VQE with ``lightning.gpu``
# ------------------------------------
#
# The `Variational Quantum Eigensolver <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__ (VQE) is
# a widely used quantum algorithm with applications in quantum chemistry and portfolio optimization
# problems. It is an application of the `Ritz variational
# principle <https://en.wikipedia.org/wiki/Ritz_method>`__, where a quantum computer is trained to
# prepare the ground state of a given molecule.
#
# Here, we benchmark this algorithm for the trihydrogen cation :math:`H_3^{+}` (three hydrogen
# atoms sharing two electrons) on a Grace-Hopper (GH200) box.
#

######################################################################
# Catalyst: Grover's with ``lightning.gpu``
# -----------------------------------------
# 
# `Grover's algorithm </codebook/#05-grovers-algorithm>`__ is an `oracle
# </codebook/04-basic-quantum-algorithms/02-the-magic-8-ball/>`__-based quantum algorithm, first
# proposed by Lov Grover in 1996 [#Grover1996]_, to solve unstructured search problems using a
# `quantum computer <https://pennylane.ai/qml/quantum-computing/>`__. For example, we could use
# Grover's algorithm to search for a phone number in a randomly ordered database containing
# :math:`N` entries and say (with high probability) that the database contains that number by
# performing :math:`O(\sqrt{N})` queries on the database, whereas a classical search algorithm would
# require :math:`O(N)` queries to perform the same task.
#
# More formally, the *unstructured search problem* is defined as a search for a string of bits in a
# list containing :math:`N` items given an *oracle access function* :math:`f(x).` This function is
# defined such that :math:`f(x) = 1` if :math:`x` is the bitstring we are looking for (the
# *solution*), and :math:`f(x) = 0` otherwise. The generalized form of Grover's algorithm accepts
# :math:`M` solutions, with :math:`1 \leq M \leq N.`
# 
# Here, we benchmark this algorithm for the trihydrogen cation :math:`H_3^{+}` (three hydrogen
# atoms sharing two electrons) on a Grace-Hopper (GH200) box.
#

######################################################################
# Catalyst: Sampling with ``lightning.gpu``
# -----------------------------------------
#

######################################################################
# Conclusion
# -----------
#


######################################################################
# References
# ----------
#

######################################################################
# Footnotes
# ---------
#
