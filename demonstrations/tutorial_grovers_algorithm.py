r""".. role:: html(raw)
   :format: html


Grover's Algorithm
==================

*Author: Ludmila Botelho. Posted: 12 May 2023.*


Grover's Algorithm is an `oracle <https://en.wikipedia.org/wiki/Oracle_machine>`__-based quantum
algorithm, proposed by Lov Grover[1]. In the original description, the author approaches the
following problem: suppose that we are searching for a specific phone number in a randomly-ordered
catalog containing :math:`N` entries. To find such a number with a probability of
:math:`\frac{1}{2}`, a classical algorithm will need to check the list on average
:math:`\frac{N}{2}` times.


In other words, the problem is defined by searching for an item on a list with :math:`N` items given
an Oracle access function :math:`f(x)`. This function has the defining property that
:math:`f(x) = 1` if and only if :math:`x` is the item we are looking for, and :math:`f(x) = 0`
otherwise. The solution to this black-box search problem is proposed as a quantum algorithm that
performs :math:`O(\sqrt{N})` oracular queries to the list with a high probability of finding the
answer, whereas any classical algorithm would require :math:`O(N)` queries for the same problem.

In this tutorial, we are going to implement a search for an n-bit string item using a quantum
circuit based on Grover's Algorithm.


The algorithm can be broken down into the following steps:

1. Prepare the initial state
2. Implement the Oracle
3. Apply the Grover diffusion operator
4. Repeat steps 2 and 3 :math:`\approx \frac{\pi}{4}\sqrt{N}` times
5. Measure



Let's import the usual PennyLane and Numpy libraries to load the necessary functions:

"""

import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np

######################################################################
# Preparing the Initial State
# ---------------------------
#
# To perform the search, we are going to create an n-dimensional system, which has :math:`N = 2^n`
# computational basis states, represented via :math:`N` :math:`n-`\ bit strings
# :math:`x_0,x_2,\cdots, x_{N-1}`. We initialize the system in the uniform superposition over all
# states, i.e., all the amplitudes associated with each of the :math:`N` basis states are equal:
#
#
# .. math:: |s\rangle ={\frac {1}{\sqrt {N}}}\sum _{x=0}^{N-1}|x\rangle .
#
#
# This can be achieved by applying a Hadamard gate to all the wires. We can inspect the circuit using
# :class:`~.Snapshot` to see how the states change on each step. Let us check the probability of finding
# a state in the computational basis for a 2-qubit circuit, writing the following functions and
# QNodes:


NUM_QUBITS = 2
dev = qml.device("default.qubit", wires=NUM_QUBITS)
wires = list(range(NUM_QUBITS))


def equal_supperposition(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)


@qml.qnode(dev)
def circuit():
    qml.Snapshot("Initial state")
    equal_supperposition(wires)
    qml.Snapshot("After applying the Hadamard gates")
    return qml.probs(wires=wires)  # Probability of finding a computational basis state on the wires


results = qml.snapshots(circuit)()

for k, result in results.items():
    print(f"{k}: {result}")

######################################################################
# Let's use a bar plot to better visualize the initial state amplitudes:


y = results["After applying the Hadamard gates"]
bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y))]

plt.xticks(rotation="vertical")
plt.bar(bit_strings, y)
plt.show()

######################################################################
# As expected, they are equally distributed.
#
# The Oracle and Grover's diffusion operator
# ------------------------------------------
#
# To access :math:`f(x)` with an Oracle, we can formulate a unitary operator such that:
#
# .. math::
#    \begin{cases}
#        U_{\omega }|x\rangle =-|x\rangle &{\text{for }}x=\omega {\text{, that is, }}f(x)=1,\\U_{\omega }|x\rangle =|x\rangle &{\text{for }}x\neq \omega {\text{, that is, }}f(x)=0,
#    \end{cases}
#
# where :math:`\omega` corresponds to the state which encondes the solution, and :math:`U_\omega` acts
# by flipping the phase of the solution state while keeping the remaining states untouched. In other
# words, the unitary :math:`U_\omega` can be seen as a reflection around the set of orthogonal states
# to :math:`\vert \omega \rangle`, written as
#
# .. math:: U_\omega = \mathbb{I} - \vert \omega \rangle \langle \omega \vert.
#
# This can be easily implemented with :class:`~.FlipSign`, which takes a binary array and flips the sign
# of the corresponding state.
#
# Let us take a look at the following example: if we pass the array ``[0,1]``, the sign of the state
# :math:`\vert 01 \rangle = \begin{bmatrix} 0 \\1 \\0 \\0 \end{bmatrix}` will flip:


dev = qml.device("default.qubit", wires=NUM_QUBITS)


@qml.qnode(dev)
def circuit():
    # Initial state preparation
    qml.PauliX(1)
    qml.Snapshot("Initial state |01>")
    # Fliping the marked state
    qml.FlipSign([0, 1], wires=wires)
    qml.Snapshot("After fliping it")
    return qml.state()


results = qml.snapshots(circuit)()

results

y1 = results["Initial state |01>"]
y2 = results["After fliping it"]

bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y))]

plt.xticks(rotation="vertical")
plt.bar(bit_strings, y1)
plt.bar(bit_strings, y2)
plt.legend(["Initial state |01>", "After fliping it"])
plt.axhline(y=0.0, color="k", linestyle="-")
plt.show()

######################################################################
# We can see that the amplitude of the state :math:`\vert 01\rangle` flipped. Following, we can prepare
# the Oracle and inspect their action in the circuit.


omega = np.zeros(NUM_QUBITS)

def oracle(wires, omega):
    qml.FlipSign(omega, wires=wires)

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def circuit():
    equal_supperposition(wires)
    qml.Snapshot("Before querying the Oracle")

    oracle(wires, omega)
    qml.Snapshot("After querying the Oracle")

    return qml.probs(wires=wires)
    # return qml.state()


results = qml.snapshots(circuit)()

print(results)

##########################################

y1 = results["Before querying the Oracle"]
y2 = results["After querying the Oracle"]

bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y))]

plt.xticks(rotation="vertical")
plt.bar(bit_strings, y1, alpha = 0.5)
plt.bar(bit_strings, y2, alpha = 0.5)
plt.legend(["Before querying the Oracle", "After querying the Oracle"])
plt.axhline(y=0.0, color="k", linestyle="-")

######################################################################
# We can see that the amplitude corresponding to the state :math:`\vert \omega \rangle` changed.
# However, we need an additional step to find the solution, since the probability of measuring any of
# the states remains equally distributed. This can be solved by applying the *Grover diffusion*
# operator, defined as
#
# .. math::
#    U_D = | s \rangle\langle s| - \mathbb{I}.
#
# The unitary :math:`U_D` also acts as a rotation, but this time through :math:`\vert s \rangle`.
# Finally, the combination of :math:`U_{\omega}` with :math:`U_D` rotates the state
# :math:`\vert s \rangle` by an angle of
# :math:`\theta =2 \arcsin{\tfrac {1}{\sqrt {N}}}`. For more geometric insights
# about the Oracle and the diffusion operator, please refer to this `codebook
# section <https://codebook.xanadu.ai/G.2>`__.
#
#
# .. figure:: ../demonstrations/grovers_algorithm/rotation.gif
#    :align: center
#    :width: 70%
#
#
# In a 2-qubit circuit, the diffusion operator has a specific shape:
#
# .. figure:: ../demonstrations/grovers_algorithm/diffusion_2_qubits.svg
#    :align: center
#    :width: 90%
#
#
# Now, we have the building blocks to implement a search for one item in a 2-qubits circuit. We can
# verify in the circuit below that iterating the *Grover iterator* :math:`U_\omega U_D` once is enough
# to solve the problem.


dev = qml.device("default.qubit", wires=NUM_QUBITS)


def diffusion_operator(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)
        qml.PauliZ(wires=wire)
    qml.ctrl(qml.PauliZ, 0)(wires=1)
    for wire in wires:
        qml.Hadamard(wires=wire)


@qml.qnode(dev)
def circuit():
    equal_supperposition(wires)
    qml.Snapshot("Uniform supperposition |s>")

    oracle(wires, omega)
    qml.Snapshot("State marked by Oracle")
    diffusion_operator(wires)

    qml.Snapshot("Amplitute after diffusion")
    return qml.probs(wires=wires)


qml.snapshots(circuit)()

######################################################################
# Searching for more items in a bigger list
# -----------------------------------------
#
# Now, let us consider the problem with higher :math:`N`, accepting :math:`M` solutions, with
# :math:`1 \leq M \leq N`. In this case, the optimal number of Grover iterations to find the solution
# is given by :math:`r \approx \lceil \frac{\pi}{4} \sqrt{\frac{N}{M}} \rceil`\ [2].
#
# For more qubits, we can use the same function for the Oracle to mark the desired states, and the
# diffusion operator takes a more general form:
#
# .. figure:: ../demonstrations/grovers_algorithm/diffusion_n_qubits.svg
#    :align: center
#    :width: 90%
#
# which is easily implemented using ``qml.template.GroverOperator``.
#
# Finally, we have all the tools to build the circuit for Grover's Algorithm, as we can see in the
# code below. For simplicity, we are going to implement the search for states
# :math:`\vert 0\rangle ^{\otimes n}` and :math:`\vert 1\rangle ^{\otimes n}`, where
# :math:`n = \log_2 N` is the number of qubits.


NUM_QUBITS = 5

omega = np.array([np.zeros(NUM_QUBITS), np.ones(NUM_QUBITS)])

M = len(omega)
N = 2**NUM_QUBITS
wires = list(range(NUM_QUBITS))

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def circuit():
    iterations = int(np.round(np.sqrt(N / M) * np.pi / 4))

    # Initial state preparation
    equal_supperposition(wires)

    # Grover's iterator
    for _ in range(iterations):
        for omg in omega:
            oracle(wires, omg)
        qml.templates.GroverOperator(wires)

    return qml.probs(wires=wires)


results = qml.snapshots(circuit)()
print(results)

######################################################################
# Let us use a bar plot to visualize the probability to find the correct bitstring.


y = results["execution_results"]
bit_strings = [f"{x:0{NUM_QUBITS}b}" for x in range(len(y))]

plt.xticks(rotation="vertical")
plt.bar(bit_strings, results["execution_results"])

######################################################################
# Wrapping up
# -----------
#
# In conclusion, you learned the basic steps of Grover's algorithm and how to implement it to search
# :math:`M` items in a list of size :math:`N` with high probability.
#
# -  Grover's algorithm in principle can be used to speed up more sophisticated computation, for
#    instance, when used as a subroutine for problems that require extensive search;
# -  and is the basis of a whole family of algorithms, such as the `Amplitude
#    amplification <https://en.wikipedia.org/wiki/Amplitude_amplification>`__ technique.
#
# **Next Step:** *Amplitude Amplification*
#

######################################################################
# References
# ----------
#
# [1] Grover, Lov K. (1996). "A fast quantum mechanical algorithm for database search". Proceedings of
# the Twenty-Eighth Annual ACM Symposium on Theory of Computing. STOC '96. Philadelphia, Pennsylvania,
# USA: Association for Computing Machinery: 212–219. arXiv:quant-ph/9605043,
# doi:10.1145/237814.237866
#
# [2] Nielsen, Michael A., and Chuang, Isaac L. (2010). "Quantum computation and quantum information".
# Cambridge: Cambridge University Press. pp. 276–305.
