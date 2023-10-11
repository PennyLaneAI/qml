r"""Linear combination of unitaries and block encodings
=============================================================

.. meta::
    :property="og:description": Master the basics of LCUs and their applications
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_lcu_blockencoding.png

.. related::

    tutorial_intro_qsvt Intro to QSVT

*Author: Juan Miguel Arrazola, Diego Guala, and Jay Soni — Posted: August, 2023.*

If I (Juan Miguel) had to summarize quantum computing in one sentence, it would be this: information is
encoded in quantum states and processed using `unitary operations <https://en.wikipedia.org/wiki/Unitary_operator>`_.
The challenge of quantum algorithms is to design and build these unitaries to perform interesting and
useful tasks. My colleague `Nathan Wiebe <https://scholar.google.ca/citations?user=DSgKHOQAAAAJ&hl=en>`_
once told me that some of his early research was motivated by a simple
question: Quantum computers can implement products of unitaries --- after all
that's how we build circuits from a `universal gate set <https://en.wikipedia.org/wiki/Quantum_logic_gate#Universal_quantum_gates>`_.
What about **sums of unitaries**? 🤔

In this tutorial you will learn the basics of one of the most versatile tools in quantum algorithms:
linear combinations of unitaries; or LCUs for short. You will also understand how to
use LCUs to create another powerful building block of quantum algorithms: block encodings.
Among their many uses, they allow us to transform quantum states by non-unitary operators.
Block encodings are useful in a variety of contexts, perhaps most famously in `qubitization <https://arxiv.org/abs/1610.06546>`_ and the `quantum
singular value transformation (QSVT) <https://pennylane.ai/qml/demos/tutorial_intro_qsvt>`_.

|

.. figure:: ../demonstrations/lcu_blockencoding/thumbnail_lcu_blockencoding.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

LCUs
----
Linear combinations of unitaries are straightforward --- it’s already explained in the name: we
decompose operators as a weighted sum of unitaries. Mathematically, this means expresssing
an operator :math:`A` in terms of coefficients :math:`\alpha_{k}` and unitaries :math:`U_{k}` as

.. math:: A =  \sum_{k=0}^{N-1} \alpha_k U_k.

A general way to build LCUs is to employ properties of the **Pauli basis**.
This is the set of all products of Pauli matrices :math:`{I, X, Y, Z}`. It forms a complete basis
for the space of operators acting on :math:`n` qubits. Thus any operator can be expressed in the Pauli basis,
which immediately gives an LCU decomposition. PennyLane allows you to decompose any matrix in the Pauli basis using the
:func:`~.pennylane.pauli_decompose` function. The coefficients :math:`\alpha_k` and the unitaries
:math:`U_k` from the decomposition can be accessed directly from the result. We show how to do this
in the code below for a simple example.

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

print(f"LCU decomposition:\n {LCU}")
print(f"Coefficients:\n {LCU.coeffs}")
print(f"Unitaries:\n {LCU.ops}")


##############################################################################
# PennyLane uses a smart Pauli decomposition based on vectorizing the matrix and exploiting properties of
# the `Walsh-Hadamard transform <https://en.wikipedia.org/wiki/Hadamard_transform>`_,
# but the cost still scales as ~ :math:`O(n 4^n)` for :math:`n` qubits. Be careful.
#
# It's good to remember that many types of Hamiltonians are already compactly expressed
# in the Pauli basis, for example in various `Ising models <https://en.wikipedia.org/wiki/Ising_model>`_
# and molecular Hamiltonians using the `Jordan-Wigner transformation <https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation>`_.
# This is very useful since we get an LCU decomposition for free.
#
# Block Encodings
# ---------------
# Going from an LCU to a quantum circuit that applies the associated operator is also straightforward
# once you know the trick: To prepare, select, and unprepare.
#
# Starting from the LCU decomposition :math:`A =  \sum_{k=0}^{N-1} \alpha_k U_k` with positive, real coefficients, we define the prepare
# (PREP) operator:
#
# .. math:: PREP|0\rangle = \sum_k \sqrt{\frac{|\alpha|_k}{\lambda}}|k\rangle,
#
# and the select (SEL) operator:
#
# .. math:: SEL|k\rangle |\psi\rangle = |k\rangle U_k |\psi\rangle.
#
# They are aptly named: PREP prepares a state whose amplitudes
# are determined by the coefficients of the LCU, and SEL selects which unitary is applied.
# 
# .. note::
#
#   Some important details about the equations above:
#
#   * :math:`\lambda` is a normalization constant, defined as :math:`\lambda = \sum_k |\alpha_k|`.
#   * :math:`SEL` acts this way on any state :math:`|\psi\rangle`
#   * We are using :math:`|0\rangle` as shorthand to denote the all-zero state for multiple qubits.
#
# The final trick is to combine PREP and SEL to make :math:`A` appear 🪄:
#
# .. math:: \langle 0| \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP} |0\rangle|\psi\rangle = A/\lambda |\psi\rangle.
#
# If you're up for it, it's illuminating to go through the math and show how :math:`A` comes out on the right
# side of the equation.
# (Tip: calculate the action of :math:`\text{PREP}^\dagger` on :math:`\langle 0|`, not on the output
# state after :math:`\text{SEL} \cdot \text{PREP}`).
#
# Otherwise, the intuitive way to understand this equation is that we apply PREP, SEL, and then invert PREP. If
# we measure :math:`|0\rangle` in the auxiliary qubits, the input state :math:`|\psi\rangle` will be transformed by
# :math:`A` (up to normalization). The figure below shows this as a circuit with four unitaries in SEL.
#
# |
#
# .. figure:: ../demonstrations/lcu_blockencoding/schematic.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0)
#
# |
#
# The circuit
#
# .. math:: U = \text{PREP}^\dagger \cdot \text{SEL} \cdot \text{PREP},
#
# is a **block encoding** of :math:`A`, up to normalization. The reason for this name is that if we write :math:`U`
# as a matrix, the operator :math:`A` is encoded inside a block of :math:`U` as
#
# .. math:: U = \begin{bmatrix} A & \cdot \\ \cdot & \cdot \end{bmatrix}.
#
# This block is defined by the subspace of all states where the auxiliary qubits are in state
# :math:`|0\rangle`.
#
#
# PennyLane supports direct implementation of `prepare <https://docs.pennylane.ai/en/stable/code/api/pennylane.StatePrep.html>`_
# and `select <https://docs.pennylane.ai/en/stable/code/api/pennylane.Select.html>`_
# operators. We'll go through them individually and use them to construct a block encoding circuit.
# Prepare circuits can be constructed using the :class:`~.pennylane.StatePrep` operation, which takes
# the normalized target state as input:

dev1 = qml.device("default.qubit", wires=1)

# normalized square roots of coefficients
alphas = (np.sqrt(LCU.coeffs) / np.linalg.norm(np.sqrt(LCU.coeffs)))


@qml.qnode(dev1)
def prep_circuit():
    qml.StatePrep(alphas, wires=0)
    return qml.state()


print("Target state: ", alphas)
print("Output state: ", np.real(prep_circuit()))

##############################################################################
# Similarly, select circuits can be implemented using :class:`~.pennylane.Select`, which takes the
# target unitaries as input. We specify the control wires directly, but the system wires are inherited
# from the unitaries. Since :func:`~.pennylane.pauli_decompose` uses a canonical wire ordering, we
# first map the wires to those used for the system register in our circuit:

dev2 = qml.device("default.qubit", wires=3)

# unitaries
ops = LCU.ops
# relabeling wires 0 --> 1, and 1 --> 2
unitaries = [qml.map_wires(op, {0: 1, 1: 2}) for op in ops]


@qml.qnode(dev2)
def sel_circuit(qubit_value):
    qml.BasisState(qubit_value, wires=0)
    qml.Select(unitaries, control=0)
    return qml.expval(qml.PauliZ(2))

print(qml.draw(sel_circuit)([0]))
##############################################################################
# Based on the controlled operations, the circuit above will flip the measured qubit
# if the input is :math:`|1\rangle` and leave it in state :math:`|0\rangle` if the 
# input is :math:`|0\rangle`. The output expecation values correspond to these states:

print('Expectation value for input |0>:', sel_circuit([0]))
print('Expectation value for input |1>:', sel_circuit([1]))

##############################################################################
# We can now combine these to construct a full LCU circuit. Here we make use of the :func:`~.pennylane.adjoint` function
# as a convenient way to invert the prepare circuit. We have chosen an input matrix that is already
# normalized, so it can be seen appearing directly in the top-left block of the unitary describing
# the full circuit --- the mark of a successful block encoding.


@qml.qnode(dev2)
def lcu_circuit():  # block_encode
    # PREP
    qml.StatePrep(alphas, wires=0)

    # SEL
    qml.Select(unitaries, control=0)

    # PREP_dagger
    qml.adjoint(qml.StatePrep(alphas, wires=0))
    return qml.state()


output_matrix = qml.matrix(lcu_circuit)()
print("A:\n", A, "\n")
print("Block-encoded A:\n")
print(np.real(np.round(output_matrix,2)))

##############################################################################
# Application: Projectors
# -----------------------
#
# Another operation we can unlock with LCUs is that of projectors. Suppose we wanted to project
# our quantum state :math:`|\psi\rangle` onto the state :math:`|\phi\rangle`, we could
# accomplish this by applying the projector :math:`| \phi \rangle\langle \phi |` to :math:`|\psi\rangle`.
#
# A property of projectors is that they are, by construction, NOT unitary. This prevents us from
# directly applying them as gates in our quantum circuits. We can get around this by using a
# simple LCU decomposition which holds for any projector:
#
# .. math::
#      | \phi \rangle\langle \phi | = \frac{1}{2} \cdot (\mathbb{I}) + \frac{1}{2} \cdot (2 \cdot | \phi \rangle\langle \phi | - \mathbb{I})
#
# Both terms in the expression above are unitary (try proving it for yourself). We can now use this LCU decomposition
# to block-encode the projector! Let's work through an example to block-encode the projector onto the :math:`|0\rangle`
# state:
#
# .. math:: | 0 \rangle\langle 0 | =  \begin{bmatrix}
#                                       1 & 0 \\
#                                       0 & 0 \\
#                                     \end{bmatrix},
#

coeffs = np.array([1/2, 1/2])
alphas = np.sqrt(coeffs) / np.linalg.norm(np.sqrt(coeffs))

# Note the second term in our LCU simplifies to a Pauli Z operation
unitaries = [qml.Identity(0), qml.PauliZ(0)]

def lcu_circuit():  # block_encode
    # PREP
    qml.StatePrep(alphas, wires="ancilla")

    # SEL
    qml.Select(unitaries, control="ancilla")

    # PREP_dagger
    qml.adjoint(qml.StatePrep(alphas, wires="ancilla"))
    return qml.state()


output_matrix = qml.matrix(lcu_circuit)()

##############################################################################
# Application to QSVT
# -------------------
#
# The QSVT algorithm is a method to transform block-encoded operators. You can learn more about it in
# our demos `Intro to QSVT <https://pennylane.ai/qml/demos/tutorial_intro_qsvt>`_ and
# `QSVT in practice <https://pennylane.ai/qml/demos/tutorial_apply_qsvt>`_. Here we show how to
# implement the QSVT algorithm using an explicit construction of the block encoding operator. We also
# need to define projector-controlled phase shifts, which can be done using :class:`~pennylane.PCPhase`.
# The :class:`~pennylane.QSVT` uses these as input to build the full algorithm.

eigen_values = np.linspace(1, 0, 4)  # pick 8 evenly spaced values starting at 0 and ending at 1 
A = np.diag(eigen_values)            # create matrix A using the eigenvalues along the diagonal 


LCU = qml.pauli.pauli_decompose(A)
print(f"LCU decomposition:\n {LCU} \n")

coeffs = np.array([c for c in LCU.coeffs] + [0.0])
alphas = np.sqrt(coeffs / np.sum(coeffs))
unitaries = [qml.map_wires(op, {0: "work1", 1: "work2"}) for op in LCU.ops]

def block_encode_A():
    # PREP
    qml.StatePrep(alphas, wires=["prep1", "prep2"])

    # SEL
    qml.Select(unitaries, control=["prep1", "prep2"])

    # PREP_dagger
    qml.adjoint(qml.StatePrep(alphas, wires=["prep1", "prep2"]))
    
    return qml.state()


output_matrix = qml.matrix(block_encode_A)()
output_matrix = output_matrix[:4, :4]

print("A:\n", np.round(A,3), "\n")
print("Block-encoded A:")
print(np.real(np.round(output_matrix,3)))

##############################################################################
# Then we use the (pre-generated) phase angles that generate this transformation and QSVT to 
# apply the transformation:

# Pre-generated phase angles for the target polynomial transformation:
# phase_angles = [3.78490414, -0.84496266, 3.22264611]
phase_angles=[-0.47136235,  0.76570731, -0.33354304]
projectors = [qml.PCPhase(phi, dim=4, wires=["prep1", "prep2", "work1", "work2"]) for phi in phase_angles]

# Get the block encoding as an operation instead of a qnode
block_encoded_op = qml.prod(block_encode_A)()

# 
QSVT_op = qml.QSVT(block_encoded_op, projectors)

print("QSVT-Block-encoded A:")
print(np.real(np.round(qml.matrix(QSVT_op),3))[:4, :4])


##############################################################################
# Final thoughts
# -------------------
# LCUs and block encodings are often associated with advanced algorithms that require the full power
# of fault-tolerant quantum computers. The truth is that they are basic constructions with
# broad applicability that can be useful for all kinds of hardware and simulators. If you're working
# on quantum algorithms and applications in any capacity, these are techniques that you should
# master. PennyLane is equipped with the tools to help you get there.


##############################################################################
# About the authors
# -----------------
# .. include:: ../_static/authors/juan_miguel_arrazola.txt
# .. include:: ../_static/authors/jay_soni.txt
# .. include:: ../_static/authors/diego_guala.txt
