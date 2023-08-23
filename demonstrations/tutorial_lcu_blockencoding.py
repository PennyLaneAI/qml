r"""Linear combination of unitaries and block encodings
=============================================================

.. meta::
    :property="og:description": Master the basics of LCUs and their applications
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_intro_qsvt.png

.. related::

    tutorial_intro_qsvt Intro to QSVT

*Author: Juan Miguel Arrazola, Diego Guala, and Jay Soni — Posted: August, 2023.*

If I (Juan Miguel) had to summarize quantum computing in one sentence, it would be like this: information is
encoded in quantum states, and information is processed using `unitary operations <https://en.wikipedia.org/wiki/Unitary_operator>`_. The art and
science of quantum algorithms is to design and buils these unitaries to performing interesting and
useful tasks. But consider this.
My colleague `Nathan Wiebe <https://scholar.google.ca/citations?user=DSgKHOQAAAAJ&hl=en>`_ once told me that some of his early research was motivated by a simple
question: Quantum computers are good at implementing products of unitaries -- after all
that's how we build circuits from a `universal gate set <https://en.wikipedia.org/wiki/Quantum_logic_gate#Universal_quantum_gates>`_. But what about implementing
**sums of unitaries**? 🤔

In this tutorial you will learn the basics of one of the most versatile tools in quantum algorithms:
linear combinations of unitaries; or LCUs for short. You will also understand how to
use LCUs to create another powerful tool: block encodings.
Among their many uses, block encodings allow us to transform quantum states by non-unitary operators.
This is useful in a variety of contexts, perhaps most famously in `qubitization <https://arxiv.org/abs/1610.06546)>`_ and the `quantum
singular value transformation (QSVT) <https://pennylane.ai/qml/demos/tutorial_intro_qsvt>`_.

[Main Tarik image here]

LCUs
----
The concept of an LCU is very straightforward; it’s basically already explained in the name: we
decompose operators as a weighted sum of unitaries. Mathematically, this means expresssing
an operator :math:`A` in terms of coefficients $\alpha_k$ and unitaries $U_k$ as

.. math:: A =  \sum_{k=1}^N \alpha_k U_k.

A general way to build LCUs in quantum computing is to employ properties of the **Pauli basis**.
This is the set of all products of Pauli operators $I, X, Y, Z$, which forms a complete basis
for the space of operators on $n$ qubits. Expressing an operator in the Pauli basis immediately
gives an LCU decomposition. PennyLane allows you to compute Pauli-basis LCUs using the
:func:`~.pennylane.pauli_decompose` function. The coefficients :math:`\alpha_k` and the unitaries
:math:`U_k` from the decomposition can then be accessed directly from the result. We will be using
those later.

"""
import numpy as np
import pennylane as qml

a = 0.25
b = 0.75

# matrix to be decomposed
A = np.array(
    [[a,  0, 0,  b],
     [0, -a, b,  0],
     [0,  b, a,  0],
     [b,  0, 0, -a]]
)

LCU = qml.pauli_decompose(A)

print(f"LCU decomposition = {LCU}")

# normalized coefficients
alphas = np.sqrt(LCU.coeffs) / np.linalg.norm(np.sqrt(LCU.coeffs))
# unitaries
ops = LCU.ops

##############################################################################
# PennyLane uses a smart implementation based on vectorizing the matrix and exploiting properties of
# the Walsh-Hadamard transform, as described `here <https://quantumcomputing.stackexchange.com/questions/31788/how-to-write-the-iswap-unitary-as-a-linear-combination-of-tensor-products-betw/31790#31790 >`_ ,
# but the cost still scales as :math:`n 4^n`. Be careful.
#
# On the other hand, it's good to remember that many types of Hamiltonians are already compactly expressed
# in the Pauli basis, for example in various `Ising models <https://en.wikipedia.org/wiki/Ising_model>`_ and for molecular Hamiltonians using the
# `Jordan-Wigner transformation <https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation>`_. This is very useful since we get one LCU decomposition
# for free.
#
# Block Encodings
# ---------------
# Going from an LCU to a quantum circuit that applies the associated operator is also straightforward
# once you know the trick. What's the trick? To prepare, select, and unprepare 😈.
#
# Starting from the LCU decomposition :math:`A =  \sum_{k=1}^N \alpha_k U_k`, we define the PREP (prepare)
# operator
#
# .. math:: PREP|0\rangle = \sum_k \sqrt{\frac{|\alpha|_k}{\lambda}}|k\rangle,
#
# and the SEL (select) operator
#
# .. math:: SEL|k\rangle |\psi\rangle = |k\rangle U_k |\psi\rangle.
#
# They are aptly named so that you never forget what they do: PREP is preparing a state whose amplitudes
# are determined by the coefficients of the LCU, and SEL is selecting which unitary is applied to
# the system. In case you're wondering, :math:`\lambda = \sum_k |\alpha_k|` is a normalization
# constant, SEL acts this way on any state :math:`|\psi\rangle`, and we have added auxiliary
# qubits where PREP acts. We are also using :math:`|0\rangle` as shorthand to denote the all-zero
# state of the auxiliary qubits.
#
# The final trick is to combine PREP and SEL to make :math:`A` appear 🪄🎩:
#
# .. math:: \langle 0| \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP} |0\rangle|\psi\rangle = A/\lambda |\psi\rangle.
#
# The way to understand this equation is that we apply PREP, SEL, and then invert PREP. After, if
# we measure :math:`|0\rangle` in the auxiliary qubits, the input state will be transformed by
# :math:`A` (up to normalization). If you're up for it, it's illuminating to go through the math.
# (Tip: calculate the action of :math:`\text{PREP}^\dagger on :math:`|0\rangle`, not on the output
# state after :math:`\text{SEL} \cdot \text{PREP}`).
#
# The circuit
#
# .. math:: U = \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP},
#
# is a **block encoding** of :math:`A`, up to normalization.
#
# [Add Oriel's figure here]
#
# The reason for this name is that if we write down
# :math:`U` as a matrix, the operator :math:`A` appears directly inside a block. That block is
# defined by the subspace of all states where the auxiliary qubits are in state :math:`|0\rangle`.
#
# PennyLane supports direct implementation of `prepare <https://docs.pennylane.ai/en/latest/code/api/pennylane.StatePrep.html>`_
# and `select <https://docs.pennylane.ai/en/latest/code/api/pennylane.Select.html?highlight=select>`_
# operators. We'll go through them individually and use them to construct a block encoding circuit.
# Prepare circuits can be constructed using the :func:`~.pennylane.StatePrep` operation, which takes
# the normalized target state as input:

dev1 = qml.device("default.qubit", wires=1)

@qml.qnode(dev1)
def prep_circuit():
    qml.StatePrep(alphas, wires=0)
    return qml.state()

print("Target state: ", alphas)
print("Output state: ", np.real(prep_circuit()))

##############################################################################
# Similarly, select circuits can be implemented using :func:`~.pennylane.Select`, which takes the
# target unitaries as input. We specify the control wires directly, but the system wires are inherited
# from the unitaries. Since :func:`~.pennylane.pauli_decompose` uses a canonical wire ordering, we
# first map the wires to those used for the system register in our circuit:
#

dev2 = qml.device("default.qubit", wires=3)

# relabeling wires to act on [1, 2]
unitaries = [qml.map_wires(op, {0: 1, 1: 2}) for op in ops]

@qml.qnode(dev2)
def sel_circuit(state):
    qml.BasisState(state, wires=0)
    qml.Select(unitaries, control_wires=0)
    return qml.expval(qml.PauliZ(2))

# Select flips the last qubit if control is |1>
print(sel_circuit([0]), sel_circuit([1]))

##############################################################################
# We can now combine these to construct a full LCU circuit. Here we make use of :fun:`~.pennylane.adjoint`
# as a convenient way to invert the prepare circuit. We have chosen an input matrix that is already
# normalized, so it can be seen appearing directly in the top-left block of the unitary describing
# the full circuit --- the mark of a successful block encoding.


@qml.qnode(dev2)
def lcu_circuit():
    # PREP
    qml.StatePrep(alphas, wires=0)

    # SEL
    qml.Select(unitaries, control_wires=0)

    # PREP_dagger
    qml.adjoint(qml.StatePrep(alphas, wires=0))
    return qml.state()

fig, ax = qml.draw_mpl(lcu_circuit)()
fig.show()

output_matrix = np.real(np.round(qml.matrix(lcu_circuit)(), 2))
print(output_matrix)
print(A)

##############################################################################
# Application to QSVT
# -------------------
#
# The QSVT algorithm is a method to transform block-encoded operators. You can learn more about it in
# our demos `Intro to QSVT <https://pennylane.ai/qml/demos/tutorial_intro_qsvt>`_ and
# `QSVT in practice <https://pennylane.ai/qml/demos/tutorial_apply_qsvt>`_. Here we show how to
# implement the QSVT algorithm using an explicit construction of the block encoding operator. We also
# need to define projector-controlled phase shifts, which can be done using :func:`~pennylane.PCPhase`.
# The :class:`.~pennylane.QSVT` then uses these as input to implement the QSVT algorithm.

dev2 = qml.device('default.qubit', wires=3)

@qml.qnode(dev2)
def qsvt_circuit(phis):
    # projector-controlled phase shifts
    projectors = [qml.PCPhase(phi, dim=2, wires=[0, 1, 2]) for phi in phis]

    # block encoding operator
    block_encode_op = qml.prod(qml.StatePrep(alphas, wires=0),
                               qml.Select(unitaries, control_wires=0),
                               qml.adjoint(qml.StatePrep(alphas, wires=0)))

    qml.QSVT(block_encode_op, projectors)

    return qml.state()

##############################################################################
# We can do an illustrative check that the algorithm works correctly by choosing angles that start
# small and gradually increase. When angles are all equal to zero we should retrieve the block encoding
# circuit. The output should change only slightly for small angles, whereas differences are more
# pronounced for larger values.


# top-left block of circuit with angles of same magnitude and alternating sign
def out_matrix(theta):
    return np.real(qml.matrix(qsvt_circuit)([theta, -theta, theta, -theta]))[:4, :4]


# angles are zero
print(out_matrix(0))
# angles are small
print(out_matrix(0.1))
# angles are big
print(out_matrix(np.pi / 2))


##############################################################################
# Final thoughts
# -------------------
# LCUs and block encodings are often associated with advanced algorithms that require the full power
# of fault-tolerant quantum computers. But the truth is that they are basic constructions with
# extremely broad applicability in quantum computing. If you're working on quantum algorithms in any
# capacity, these are techniques that you should probably master. PennyLane is equipped with all the
# tools that can help you get there.


##############################################################################
# About the authors
# ----------------
# .. include:: ../_static/authors/juan_miguel_arrazola.txt
# ..include::../ _static / authors / jay_soni.txt
# ..include::../ _static / authors / diego_guala.txt
