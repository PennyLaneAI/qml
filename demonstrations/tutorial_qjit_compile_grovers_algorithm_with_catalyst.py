r""".. role:: html(raw)
   :format: html

How to Quantum Just-In-Time Compile Grover's Algorithm with Catalyst
====================================================================

.. meta::
    :property="og:description": This demonstration illustrates how to use
        Catalyst to just-in-time (QJIT) compile a PennyLane circuit implementing
        Grover's algorithm.

.. related::

    tutorial_grovers_algorithm Grover's Algorithm
    tutorial_how_to_quantum_just_in_time_compile_vqe_catalyst How to quantum just-in-time compile VQE with Catalyst

*Author: Joey Carter — Posted: 19 September 2024.*
"""

######################################################################
# `Grover's algorithm <https://pennylane.ai/qml/demos/tutorial_grovers_algorithm/>`__ is an `oracle
# </codebook/04-basic-quantum-algorithms/02-the-magic-8-ball/>`__-based quantum algorithm, first
# proposed by Lov Grover in 1996 [#Grover1996]_, to solve "unstructured search" problems using a
# quantum computer. For example, we could use Grover's algorithm to search for a phone number in a
# randomly ordered database containing :math:`N` entries and say with high probability that the
# database contains the number being searched by performing :math:`O(\sqrt{N})` queries on the
# database, whereas a classical search algorithm would require :math:`O(N)` queries to perform the
# same task.
#
# More formally, the problem is defined as a search for a string of bits in a list containing
# :math:`N` items given an *Oracle access function* :math:`f(x)`. This function is defined such that
# :math:`f(x) = 1` if :math:`x` is the bitstring we are looking for (the *solution*), and
# :math:`f(x) = 0` otherwise. The generalized form of Grover's algorithm accepts :math:`M`
# solutions, with :math:`1 \leq M \leq N`.
#
# In this tutorial, we will implement the generalized Grover's algorithm using `Catalyst
# <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__, a quantum just-in-time
# (QJIT) compiler framework for PennyLane, which makes it possible to compile, optimize, and execute
# hybrid quantum–classical workflows. We will also measure the performance improvement we get from
# using Catalyst with respect to the native Python implementation.


######################################################################
# Generalized Grover's Algorithm with PennyLane
# ---------------------------------------------
#
# In the `Grover's Algorithm <https://pennylane.ai/qml/demos/tutorial_grovers_algorithm/>`__
# tutorial, we saw how to implement the generalized Grover's algorithm in PennyLane. The procedure
# is as follows:
#
# 1. Initialize the system to an equal superposition over all states.
# 2. Perform :math:`r(N, M)` *Grover iterations:*
#
#    1. Apply the unitary *Oracle operator* :math:`U_\omega`, implemented using
#       :class:`~.pennylane.FlipSign`, for each solution index :math:`\omega`.
#    2. Apply the *Grover diffusion operator* :math:`U_D`, implemented using
#       :class:`~.pennylane.GroverOperator`.
#
# 3. Measure the resulting quantum state in the computational basis.
#
# We also saw that the optimal number of Grover iterations to find the solution is given by
# [#NandC2000]_
#
# .. math:: r(N, M) \approx \left \lceil \frac{\pi}{4} \sqrt{\frac{N}{M}} \right \rceil .
#
# For simplicity, throughout the rest of this tutorial we will consider the search for the :math:`M
# = 2` solution states :math:`\vert 0\rangle ^{\otimes n}` and :math:`\vert 1\rangle ^{\otimes n}`,
# where :math:`n = \log_2 N` is the number of qubits, in a "database" of size :math:`N = 2^n`
# containing all possible :math:`n`-qubit states.
#
# First, we'll import the required packages and define the Grover's algorithm circuit, as we did in
# the previous tutorial:

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml


def equal_superposition(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)


def oracle(wires, omega):
    qml.FlipSign(omega, wires=wires)


def num_grover_iterations(N, M):
    return np.ceil(np.sqrt(N / M) * np.pi / 4).astype(int)


def grover_circuit(num_qubits):
    wires = list(range(num_qubits))
    omega = np.array([np.zeros(num_qubits), np.ones(num_qubits)])

    M = len(omega)
    N = 2**num_qubits

    # Initial state preparation
    equal_superposition(wires)

    # Grover iterations
    for _ in range(num_grover_iterations(N, M)):
        for omg in omega:
            oracle(wires, omg)
        qml.templates.GroverOperator(wires)

    return qml.probs(wires=wires)


######################################################################
# We'll begin with a circuit defined using the default state-simulator device, ``"default.qubit"``,
# as our baseline. See the documentation in :func:`~.pennylane.device` for a list of other supported
# devices. To run our performance benchmarks, we'll increase the number of qubits in our circuit to
# :math:`n = 12`.

NUM_QUBITS = 12


@qml.qnode(qml.device("default.qubit", wires=NUM_QUBITS))
def circuit_default_qubit():
    return grover_circuit(NUM_QUBITS)


results = circuit_default_qubit()


######################################################################
# Let's quickly confirm that Grover's algorithm correctly identified the solution states
# :math:`\vert 0\rangle ^{\otimes n}` and :math:`\vert 1\rangle ^{\otimes n}` as the most likely
# states to be measured:

most_probable_states = np.argsort(results)[-2:][::-1]

for i in most_probable_states:
    print(f"Prob of state '{i:0{NUM_QUBITS}b}': {results[i]:.4g}")


######################################################################
# It worked! We are now ready to QJIT compile our Grover's algorithm circuit.


######################################################################
# Quantum Just-In-Time Compiling the Circuit
# ------------------------------------------
#
# At the time of writing, Catalyst does not support the ``"default.qubit"`` state-simulator device,
# so let's first define a new circuit using `Lightning
# <https://docs.pennylane.ai/projects/lightning/en/stable/index.html>`__, which is a PennyLane
# plugin that provides more performant state simulators written in C++, and which is supported by
# Catalyst. The full list of devices supported by Catalyst is available `here
# <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/devices.html>`__.


@qml.qnode(qml.device("lightning.qubit", wires=NUM_QUBITS))
def circuit_lightning():
    return grover_circuit(NUM_QUBITS)


######################################################################
# Then, to QJIT compile our circuit with Catalyst, we simply wrap it with :func:`~pennylane.qjit`:

circuit_qjit = qml.qjit(circuit_lightning)


######################################################################
# We now have our QJIT object ``circuit_qjit``. A small detail to note in this case is that because
# the function ``circuit_lightning`` takes no input arguments, Catalyst will in fact *ahead-of-time*
# (AOT) compile the circuit at instantiation, meaning that when we call this QJIT object for the
# first time, the compilation will have already taken place, and Catalyst will execute the compiled
# code. With JIT compilation, by contrast, the compilation is triggered at the first call site
# rather than at instantiation. The compilation step will incur some runtime overhead, which we will
# measure below. Furthermore, when we call the compiled QJIT object for the first time, there is an
# additional, small overhead incurred to cache the compiled code for faster access later on. Every
# subsequent call to the QJIT object, assuming it has not been altered, will read directly from this
# cache and execute the compiled circuit. See the `Compilation Modes
# <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html#compilation-modes>`__
# documentation in the Catalyst `Quick Start
# <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html>`__ guide for more
# information on the difference between JIT and AOT compilation.


######################################################################
# Profiling the Circuits
# ----------------------
#
# Let's start profiling the circuits we have defined. We have five function executions in total to
# profile:
#
# 1. Executing the circuit using ``"default.qubit"``.
# 2. Executing the circuit using ``"lightning.qubit"``.
# 3. Compiling the circuit with Catalyst, to measure the AOT compilation overhead.
# 4. The first call to the QJIT-compiled circuit, to measure the circuit execution time *with* the
#    caching overhead.
# 5. Subsequent calls to the QJIT-compiled circuit, to measure the circuit execution time *without*
#    the caching overhead.
#
# We'll use the `timeit <https://docs.python.org/3/library/timeit.html>`__ module part of the Python
# Standard Library to measure the runtimes. To improve the statistical precision of these
# measurements, we'll repeat the operations for items (2) and (5) five times; item (1) is slow, and
# items (3) and (4) are only run once by construction, so we will not repeat these operations.

import timeit

NUM_REPS = 5

runtimes_native_default = timeit.repeat(
    "circuit_default_qubit()", globals=locals(), number=1, repeat=1
)
runtimes_native_lightning = timeit.repeat(
    "circuit_lightning()", globals=locals(), number=1, repeat=NUM_REPS
)
runtimes_compilation = timeit.repeat(
    "qml.qjit(circuit_lightning)", globals=locals(), number=1, repeat=1
)
runtimes_first_qjit = timeit.repeat(
    "circuit_qjit()", globals=locals(), number=1, repeat=1
)
runtimes_subsequent_qjit = timeit.repeat(
    "circuit_qjit()", globals=locals(), number=1, repeat=NUM_REPS
)

run_names = [
    "Native (default.qubit)",
    "Native (lightning.qubit)",
    "QJIT compilation",
    "QJIT (first call)",
    "QJIT (subsequent calls)",
]
run_names_display = [name.replace(" ", "\n", 1) for name in run_names]
runtimes = [
    np.mean(runtimes_native_default),
    np.mean(runtimes_native_lightning),
    np.mean(runtimes_compilation),
    np.mean(runtimes_first_qjit),
    np.mean(runtimes_subsequent_qjit),
]


def std_err(x):
    """Standard error = sample standard deviation / sqrt(sample size)"""
    if len(x) == 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(len(x))


runtimes_err = [
    std_err(runtimes_native_default),
    std_err(runtimes_native_lightning),
    std_err(runtimes_compilation),
    std_err(runtimes_first_qjit),
    std_err(runtimes_subsequent_qjit),
]

for i in range(len(run_names)):
    print(f"{run_names[i]} runtime: ({runtimes[i]:.4g} +/- {runtimes_err[i]:.2g}) s")


######################################################################
# Let's plot these runtimes as a bar chart to compare them visually. Note that we're using a
# logarithmic scale for the *y*-axis!

fig = plt.figure(figsize=[8.0, 4.8])
plt.title("Grover's Algorithm Runtime Benchmarks")
bars = plt.bar(run_names_display, runtimes, color="#70CEFF")
plt.errorbar(
    run_names_display, runtimes, yerr=runtimes_err, fmt="None", capsize=2.0, c="k"
)
plt.bar_label(bars, fmt="{:#.2g} s", padding=5)
plt.xlabel("Function Executed")
plt.ylabel("Runtime [s]")
plt.yscale("log")
plt.margins(y=0.15)
plt.text(
    0.98,
    0.98,
    f"Number of qubits, $n = {NUM_QUBITS}$",
    ha="right",
    va="top",
    transform=plt.gca().transAxes,
)
plt.tight_layout()
plt.show()


######################################################################
# This plot illustrates the power of Catalyst: by simply wrapping our Grover's algorithm circuit as
# a QJIT-compiled object (or AOT-compiled in this case) with :func:`~pennylane.qjit`, we have
# achieved execution runtimes several orders of magnitude less than the native-Python PennyLane
# circuit using ``"default.qubit"``. While the compilation step itself does incur some runtime, the
# overall runtime of the QJIT workflow still outperforms even the circuit defined using the
# Lightning state simulator, especially in subsequent calls to the QJIT-compiled circuit. [*]_


######################################################################
# Conclusion
# -----------
#
# This tutorial has demonstrated how to just-in-time compile a quantum circuit implementing the
# generalized Grover's algorithm using `Catalyst
# <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__.
#
# For a circuit with :math:`n = 12` qubits, analogous to a search in a randomly ordered "database"
# containing :math:`N = 2^{12} = 4096` entries, Catalyst offers a runtime performance several orders
# of magnitude better than the same circuit implemented in native Python.


######################################################################
# References
# ----------
#
# .. [#Grover1996]
#
#     L. K. Grover (1996) "A fast quantum mechanical algorithm for database search". `Proceedings of
#     the Twenty-Eighth Annual ACM Symposium on Theory of Computing. STOC '96. Philadelphia,
#     Pennsylvania, USA: Association for Computing Machinery: 212–219
#     <https://dl.acm.org/doi/10.1145/237814.237866>`__.
#     (arXiv: `9605043 [quant-ph] <https://arxiv.org/abs/quant-ph/9605043>`__)
#
# .. [#NandC2000]
#
#     M. A. Nielsen, and I. L. Chuang (2000) "Quantum Computation and Quantum Information",
#     Cambridge University Press.


######################################################################
# -----
#
# .. [*]
#
#     Note that we normally wouldn't execute the same circuit multiple times to perform Grover's
#     algorithm. We have only done so here to illustrate the performance improvement that QJIT
#     compiling with Catalyst offers if it is ever necessary to execute your own quantum circuit
#     multiple times.

######################################################################
# About the author
# ----------------
#
# .. include:: ../_static/authors/joey_carter.txt
