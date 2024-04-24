r"""
Magic state distillation
========================

While many typical quantum algorithms can be described via simple quantum circuits, others require a
more expressive programming model. Beyond the description of unitary operators, an algorithm may
require the probabilistic execution of quantum instructions, real-time feedback from quantum
measurements, real-time classical computation, and unbounded repetitions of program segments. Such
programs are generally also called **hybrid quantum-classical programs**.

`Catalyst <https://github.com/PennyLaneAI/catalyst>`__ for PennyLane brings this powerful
programming model to PennyLane to develop and compile hybrid quantum programs in Python.

One such algorithm that goes beyond the circuit model is the **magic state distillation** routine,
developed to enable practical universal gate sets on fault-tolerant hardware architectures. In this
tutorial, we will see how we can use Catalyst’s tight integration of quantum and classical code,
both within the language and during execution, to develop a magic state distillation routine.

.. figure:: ../_static/demonstration_assets/magic_state_distillation/OGthumbnail_large_magic-state-distillation_2024-04-23.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

Primer
------

The idea behind this algorithm is based on the availability of a quantum computer that is only
capable of running gates from the *Clifford group*, which is generated by the operators
:math:`\{H, S, CNOT\}`. This computer alone is provably not a universal quantum computer, meaning
that there are quantum algorithms it would not be capable of running.

In order to achieve universal quantum computing, only a single additional “non-Clifford” gate is
required, which cannot be constructed from Clifford gates alone. As demonstrated by `Bravyi and
Kitaev in 2005 <https://arxiv.org/abs/quant-ph/0403025>`__, certain noisy qubit states can be
purified into so-called *magic states*, which can in turn be used to implement non-Clifford gates by
consuming the magic state.

In practice, it is not necessarily easy to generate magic states. However, provided we have a method
of generating (noisy) quantum states that are just “close enough” to magic states, we can purify the
noisy states to be arbitrarily close to ideal magic states. This is the procedure performed by the
magic state distillation algorithm.

T-Type Magic States
-------------------

In this tutorial we will produce T-type magic states via purification. T-type states are a
particular type of magic states that are defined as the eigenvectors of the :math:`e^{i\pi/4}SH`
operator (an :math:`H` gate followed by an :math:`S` gate):

.. math:: \frac{e^{i\pi/4}}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ i & -i \end{bmatrix}

In the computational basis, these eigenvectors can be expressed as:

.. math:: | T_0\rangle = \cos(\beta) | 0\rangle + e^{i\pi/4}\sin(\beta) | 1\rangle,

.. math:: | T_1\rangle = \sin(\beta)| 0\rangle - e^{i\pi/4}\cos(\beta)| 1\rangle,

with :math:`\beta = \frac{1}{2}\arccos(\frac{1}{\sqrt{3}})`.

First, let’s write a function to generate a pure :math:`|T_0\rangle` state via a Pauli-Y rotation
and a phase shift:
"""

import pennylane as qml

import jax; jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
from jax import random

b = 0.5 * jnp.arccos(1 / jnp.sqrt(3))

def generate_T0(wire):
    qml.RY(2*b, wires=wire)
    qml.PhaseShift(jnp.pi/4, wires=wire)

######################################################################
# Now, let’s create up a *faulty* :math:`| T_0\rangle` state-generating function. We can do this by
# perturbing the :math:`|T_0\rangle` state by a random component of scale :math:`r`.
#
# Note that since this is a random function, we need to pass it a PRNG key to satisfy `JAX’s stateless
# implementation <https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html>`__.
#

def faulty_generate_T0(wire, key, r):
    key, subkey = random.split(key)
    disturbance = random.uniform(subkey, minval=-r, maxval=r)
    qml.RY(2 * (b + disturbance), wires=wire)

    key, subkey = random.split(key)
    disturbance = random.uniform(subkey, minval=-r, maxval=r)
    qml.PhaseShift(jnp.pi / 4 + disturbance, wires=wire)
    return key

######################################################################
# Purification algorithm
# ----------------------
#
# The purification algorithm for magic states was first introduced by `Bravyi and
# Kitaev <https://arxiv.org/abs/quant-ph/0403025>`__, although we will be using the implementation
# described in `A study of the robustness of magic state distillation against Clifford gate
# faults <https://uwspace.uwaterloo.ca/bitstream/handle/10012/6976/Jochym-O%27Connor_thesis_Final.pdf?sequence=1&isAllowed=y>`__
# by Tomas Jochym-O’Connor. The process is as follows:
#
# -  Prepare five copies of noisy :math:`| T_0\rangle` states;
#
# -  Apply the decoding circuit of the 5-qubit error correction code (refer to the sources above for
#    more details on how this works);
#
# -  Perform a “syndrome” measurement on the first four wires (this is error correction terminology,
#    normally meaning a measurement to detect the presence of errors).
#
# Remarkably, if all the measurements come out as 0, we will have obtained a noisy
# :math:`| T_1\rangle` state of provably higher fidelity than our input states. We can then convert
# the :math:`| T_1\rangle` state into a :math:`| T_0\rangle` state using our Clifford gate set, namely
# a Hadamard gate followed by a Pauli-Y gate. This process can then be repeated to achieve even higher
# fidelities.
#
# Note that if any of the measurements produced a 1, the algorithm failed and we need to restart the
# process. This is where the hybrid quantum programming features come in: we need to obtain real-time
# measurement results, and based on those decide on the next quantum instructions to execute.
#
# First, let’s define the error correction decoding circuit.
#

def error_correction_decoder(wires):
    """Error correction decoder for the 5-qubit error correction code generated
    by stabilizers XZZXI, IXZZX, XIXZZ, ZXIXZ.
    """
    w0, w1, w2, w3, w4 = wires

    qml.CNOT(wires=[w1, w0])
    qml.CZ(wires=[w1, w0])
    qml.CZ(wires=[w1, w2])
    qml.CZ(wires=[w1, w4])

    qml.CNOT(wires=[w2, w0])
    qml.CZ(wires=[w2, w3])
    qml.CZ(wires=[w2, w4])

    qml.CNOT(wires=[w3, w0])

    qml.CNOT(wires=[w4, w0])
    qml.CZ(wires=[w4, w0])

    qml.PauliZ(wires=w0)
    qml.PauliZ(wires=w1)
    qml.PauliZ(wires=w4)

    qml.Hadamard(wires=w1)
    qml.Hadamard(wires=w2)
    qml.Hadamard(wires=w3)
    qml.Hadamard(wires=w4)

######################################################################
# We’ll also define some helper functions to perform a conditional reset of a qubit into the
# :math:`| 0\rangle` state, which we will use whenever the algorithm needs to restart.
#
# Here we use a mid-circuit measurement and a classically-controlled Pauli-X gate:
#

from catalyst import measure

def measure_and_reset(wire):
    """Measure a wire and then reset it back to the |0⟩ state."""

    m = measure(wire)

    if m:
        qml.PauliX(wires=wire)

    return m

######################################################################
# Note in the above:
#
# -  we import and use :func:`catalyst.measure`, rather than using :func:`pennylane.measure`.
#
# -  we use standard Python ``if`` statements. When we quantum just-in-time compile the entire
#    algorithm, we will utilize the :doc:`AutoGraph <catalyst:dev/autograph>` feature of Catalyst to automatically capture Python
#    control flow around quantum operations.
#
# Now we come to the main part of the algorithm, which we will JIT-compile using Catalyst and
# :func:`~pennylane.qjit`!
#
# The structure of the algorithm consists of a *repeat-until-success* loop. This means we execute a
# piece of code with a probabilistic outcome, and if the outcome is not the desired result, we go back
# and repeat the code until we do get the desired result. In our case the desired result is a syndrome
# measurement of 0. We’ll encode this repeat-until-success loop with a while loop around quantum
# operations.
#

dev = qml.device("lightning.qubit", wires=5)

@qml.qjit(autograph=True)
@qml.qnode(dev)
def state_distillation(random_key, r):
    key = random_key
    syndrome = True

    while syndrome:
        # generate 5 faulty T0 states
        key = faulty_generate_T0(0, key, r)
        key = faulty_generate_T0(1, key, r)
        key = faulty_generate_T0(2, key, r)
        key = faulty_generate_T0(3, key, r)
        key = faulty_generate_T0(4, key, r)

        # run the error correction decoding algorithm
        # on the generated faulty states
        error_correction_decoder(wires=(0, 1, 2, 3, 4))

        # measure and reset all wires
        m1 = measure_and_reset(1)
        m2 = measure_and_reset(2)
        m3 = measure_and_reset(3)
        m4 = measure_and_reset(4)

        syndrome = m1 + m2 + m3 + m4 > 0

        if syndrome:
            # reset wire 0 and return to repeat the loop
            measure_and_reset(0)

    # if all measurements were 0, then the loop
    # has exited, and we know that wire 0 is in an approximate T1 state.

    # We can convert the T1 state back to T0
    # by applying a Hadamard and Pauli-Y rotation
    qml.Hadamard(wires=0)
    qml.PauliY(wires=0)

    return qml.state()

######################################################################
# Benchmark
# ---------
#
# To confirm that we are, in fact, successfully distilling T-type magic states, we will measure the
# fidelity of a purified magic state compared to the fidelity of the original noisy state (with
# respect to an ideal :math:`| T_0\rangle` state). The results are averaged over a number of runs to
# account for the randomness in the noise.
#

import matplotlib.pyplot as plt
import os

dev_5_qubits = qml.device("default.qubit", wires=5)

@jax.jit
@qml.qnode(dev_5_qubits)
def T0_state():
    generate_T0(0)
    return qml.state()

@jax.jit
@qml.qnode(dev_5_qubits)
def faulty_T0_state(random_key, r):
    faulty_generate_T0(0, random_key, r)
    return qml.state()

exact_T0_state = T0_state()
perturbations = jnp.linspace(0, 1, 20)
repeats = 200

pres = []
posts = []

for r in perturbations:

    pre_total = 0.
    post_total = 0.

    for i in range(repeats):
        key = random.PRNGKey(i)

        # generate a faulty T0 state
        faulty_qubit_state = faulty_T0_state(key, r)
        # perform magic state distillation on the faulty T0 state
        distilled_qubit_state = state_distillation(key, r)

        # compute the fidelity of the faulty and exact T0 state
        base_fidelity = jnp.abs(jnp.dot(faulty_qubit_state, jnp.conj(exact_T0_state))) ** 2
        pre_total += base_fidelity

        # compute the fidelity of the distilled/purified T0 state and the exact T0 state
        distilled_fidelity = jnp.abs(jnp.dot(distilled_qubit_state, jnp.conj(exact_T0_state))) ** 2
        post_total += distilled_fidelity

    pres.append(pre_total / repeats)
    posts.append(post_total / repeats)

plt.style.use("bmh")
plt.figure(figsize=(10, 6))
plt.plot(perturbations, pres, label="pre-distillation")
plt.plot(perturbations, posts, label="post-distillation")
plt.title("Average Fidelity for T-Type Magic States", fontsize=18)
plt.legend(fontsize=12)
plt.xlabel("noisy perturbation", fontsize=14)
plt.ylabel(r"fidelity w.r.t  $\left|T_0\right\rangle$", fontsize=14)
plt.show()

######################################################################
# From the plot we can see that the distillation procedure can significantly improve the fidelity of
# the magic state, provided that the input state has at least ~82% fidelity.
#
# Conclusion
# ----------
#
# In this tutorial, we have implemented the magic state distillation algorithm to distill a noisy
# T-type magic state into one of higher fidelity using PennyLane and Catalyst. This was done by
# decoding the :math:`| T_0\rangle` state with the 5-qubit error correction code, and selecting for
# the desired measurement outcomes.
#
# This algorithm can be repeated many times, in order to obtain :math:`| T_0\rangle` states of
# arbitrary fidelity. Non-Clifford gates can then be implemented on a fault-tolerant architecture,
# achieving universal quantum computation.
#
# References
# ----------
#
# .. [#bravyi2005]
#
#    Sergey Bravyi and Alexei Kitaev. "Universal quantum computation with
#    ideal Clifford gates and noisy ancillas." `Physical Review A 71.2 (2005).
#    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.022316>`__
#
# .. [#tomas2012]
#
#    Tomas Jochym-O'Connor, et al. "The robustness of magic state
#    distillation against errors in Clifford gates."
#    `arXiv preprint arXiv:1205.6715 (2012) <https://arxiv.org/abs/1205.6715>`__.

######################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/david_ittah.txt
