r"""Block Encodings
=============================================================

.. meta::
    :property="og:description": Learn how to perform block encoding
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_block_encoding.png

.. related::

     tutorial_intro_qsvt Intro to QSVT

*Author: Diego Guala, Soran Jahangiri, Jay Soni,  — Posted: September 29, 2023.*

Prominent quantum algorithms such as Quantum Phase estimation and Quantum Singular Value
Transformation require implementing a non-unitary operator in a quantum circuit. This is problematic
because quantum computers can only perform unitary evolutions. Block encoding is a general technique
that solves this problem by embedding the non-unitary operator in a unitary matrix that can be
implemented in a quantum circuit containing a set of ancilla qubits. In this tutorial, you learn
several block encoding methods that are commonly used in quantum algorithms.

|

.. figure:: ../demonstrations/intro_qsvt/thumbnail_intro_qsvt.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

Block encoding a matrix
-----------------------
We define block encoding as embedding a given non-unitary matrix, :math:`A`, into a matrix :math:`U`

.. math:: U(a) = \begin{pmatrix} A & *\\
                  * & *
                 \end{pmatrix},

such that :math:`U\{\dagger} U = I`. A general recipe for this encoding is to construct :math:`U` as

.. math:: U(a) = \begin{pmatrix} a & \sqrt{1-a^2}\\
                 \sqrt{1-a^2} & -a
                 \end{pmatrix}.

The only condition is that the singular values of :math:`A` should be bounded by 1. This block
encoding can be done in PennyLane using the :class:`~pennylane.BlockEncode` operation. The following
code shows block encoding for a :math:`2 \time 2` matrix.
"""
import pennylane as qml
from pennylane import numpy as qnp

A = [[0.1, 0.2], [0.3, 0.4]]
U = qml.BlockEncode(A, wires=range(2))
print(qml.matrix(U))

##############################################################################
# We can directly implement this operator in a circuit that will be executed on a quantum simulator.
# This is great but we also need to know how to implement a block encoding unitary using a set of
# quantum gates. In the following sections we learn three main techniques for constructing circuits
# that implement a block encoding unitary for a given matrix.

# Block encoding with LCU
# -----------------------
# A powerful method for block encoding a matrix is to decompose it into a linear combination of
# unitaries (LCU) and then block encode the LCU. The circuit for this block encoding is constructed
# using two important building blocks: Prepare and Select operations.

# Block encoding with FABLE
# -------------------------
# One way to create approach block-encoding an arbitrary matrix :math:`A \in \mathbb{C}^{N x N}`, is by relying on
# oracle access to the entries in the matrix (see [#fable]_, [#sparse]_). Suppose we have access to an oracle
# :math:`\hat{O}_{A}` such that:
#
# .. math::
#
#    \hat{O}_{A} |0\rangle |i\rangle |j\rangle \ \rightarrow \ (a_{i,j}|0\rangle + \sqrt{1 - |a_{i,j}|^2})|i\rangle |j\rangle
#
# Given two registers representing the :math:`i`th row and :math:`j`th column, this oracle extracts the matrix entry
# at that position. Where :math:`A_{i,j} = \alpha \cdot a_{i,j}` are rescaled to guarantee it is unitary. Using this
# oracle and a :math:`SWAP` operation, we can construct a quantum circuit for block-encoding :math:`\frac{A}{\alpha}`:
#
# .. figure:: ../demonstrations/qonn/fable_circuit.png
#     :width: 50%
#     :align: center
#
# The fast approximate quantum circuits for block encodings (FABLE) is a general method
# for block encoding dense and sparse matrices. The level of approximation in FABLE can be adjusted
# to compress and sparsify the resulting circuits. The general circuit is constructed from a set of
# rotation and C-NOT gates. The rotation angles are obtained from the elements of the encoded
# matrix. For a :math:`2 \time 2` matrix, the circuit can be constructed as
#
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
#define the matrix
n = 2
N = 2**n
qnp.random.seed(1)
A = qnp.random.randn(N, N)

#turn the matrix into a vector and normalize
Avec = qnp.ravel(A)
alpha = max(1,qnp.linalg.norm(Avec,qnp.inf))
Avec = Avec/alpha

#obtain single qubit rotation angles
alphas = 2*qnp.arccos(Avec)
thetas = compute_theta(alphas)

#define control wires
code = gray_code(N)
num_selections=len(code)

control_wires = control_indices = [
    int(qnp.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
    for i in range(num_selections)
]


#create a circuit that block encodes on 5 qubits, 4x4 matrix
# hard-coded circuit
dev = qml.device('default.qubit', wires=5)
@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    for idx in range(len(thetas)):
        qml.RY(thetas[idx],wires=4)
        qml.CNOT(wires=[control_wires[idx],4])
    qml.SWAP(wires=[0,2])
    qml.SWAP(wires=[1,3])
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    return qml.state()

print(qml.draw(circuit)())
#
# The rotation angles, :math:`\Theta = (\theta_1, ..., \theta_n)`, are obtained with
#
# .. math:: \left ( H^{\otimes 2n} P \right ) \Theta = C,
#
# where P is a permutation that transforms a binary ordering to the Gray code ordering,
# :math:`C = (arccos(A_00), ..., arccos(A_nn))` is obtained from the matrix elements of the matrix A
# and H is defined as

# .. math:: H = \begin{pmatrix} 1 & 1\\
#                               1 & -1
#               \end{pmatrix}.
#
#
# We now compute the matrix representation of the circuit

print(A)
#print top left of circuit unitary
print(alpha*N*qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:N,0:N])

# You can see that the matrix is a block encoding of the original matrix A.
#
# The interesting thing about the Fable method is that one can eliminate those rotation gates that
# have an angle smaller than a pre-defined threshold. This leaves a sequence of C-NOT gates that in
# most cases cancel each other out.

tolerance= 0.07

dev = qml.device('default.qubit', wires=5)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    for idx in range(len(thetas)):
        if abs(thetas[idx])>tolerance:
            qml.RY(thetas[idx],wires=4)
        qml.CNOT(wires=[control_wires[idx],4])
    qml.SWAP(wires=[0,2])
    qml.SWAP(wires=[1,3])
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    return qml.state() 

print(qml.draw(circuit)())

# You can confirm that two C-NOT gates applied to the same wires
# cancel each other. Compressing the circuit in this way is an approximation. Let's see how good
# this approximation is in the case of our example.

tolerance= 0.07

dev = qml.device('default.qubit', wires=5)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    for idx in range(len(thetas)):
        if abs(thetas[idx])>tolerance:
            qml.RY(thetas[idx],wires=4)
        # [add process to remove extra CNOTs]
        qml.CNOT(wires=[control_wires[idx],4])
    qml.SWAP(wires=[0,2])
    qml.SWAP(wires=[1,3])
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    return qml.state()

print(qml.draw(circuit)())

print(A)
print(alpha*N*qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:N,0:N])

# Block-encoding structured sparse matrices
# -----------------------------------------
# The quantum circuit for the oracle :math:`\hat{O}_{A}`, presented above, requires on the order of :math:`~ O(N^{2})`
# gates to implement. In the case where :math:`A` is a structured sparse matrix, we can generate a more efficient
# quantum circuit representation for the oracle.
# Sparse matrices that have specific structures can be efficiently block encoded.  A general
# circuit for block encoding s-sparce matrices can be constructed
#
# Fig 5 of https://arxiv.org/abs/2203.10236
#
# The circuit has n qubits and m + 1 ancilla qubits.
# Ds is defined as HxHxH
# O_A, O_s can be constructed for structured sparse matrices
# Let's look at O_A, O_s for the Banded circulant matrix

# Problem setup: ---------------------------
s = 4
alpha = 0.1
gamma = 0.3 + 0.6j
beta = 0.3 - 0.6j

A = qnp.array([[alpha, gamma, 0, 0, 0, 0, 0, beta],
               [beta, alpha, gamma, 0, 0, 0, 0, 0],
               [0, beta, alpha, gamma, 0, 0, 0, 0],
               [0, 0, beta, alpha, gamma, 0, 0, 0],
               [0, 0, 0, beta, alpha, gamma, 0, 0],
               [0, 0, 0, 0, beta, alpha, gamma, 0],
               [0, 0, 0, 0, 0, beta, alpha, gamma],
               [gamma, 0, 0, 0, 0, 0, beta, alpha]])

print(f"Original A:\n{A}", "\n")

A2 = qml.math.array(A) / s
evs = qnp.linalg.eigvals(A)
print(evs)
g = qnp.linalg.cond(A)
print(max(qnp.abs(evs)), min(qnp.abs(evs)), g, 1 / min(qnp.abs(evs)), "\n")
evs = qnp.linalg.eigvals(A2)
print(evs)
g = qnp.linalg.cond(A2)
print(max(qnp.abs(evs)), min(qnp.abs(evs)), g, 1 / min(qnp.abs(evs)), "\n")
>>>>>>> 04f8302b (messy sparse block encoding section)


# Soln: ---------------------------

def shift_circ(s_wires, shift="L"):
    control_values = [1, 1] if shift == "L" else [0, 0]

    qml.ctrl(qml.PauliX, control=s_wires[:2], control_values=control_values)(wires=s_wires[2])
    qml.ctrl(qml.PauliX, control=s_wires[0], control_values=control_values[0])(wires=s_wires[1])
    qml.PauliX(s_wires[0])


# Sanity check: ~~~~~~~~~~~~~~~~~~~~~~
# with qml.tape.QuantumTape() as t:
#     shift_circ([0, 1, 2], shift="L")

# print(t.draw(wire_order=[2,1,0]))
# print(t.operations)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def oracle_c(wires_l, wires_j):
    qml.ctrl(shift_circ, control=wires_l[0])(wires_j, shift="L")
    qml.ctrl(shift_circ, control=wires_l[1])(wires_j, shift="R")


def oracle_a(ancilla, wire_l, wire_j, a, b, g):
    theta_0 = 2 * qnp.arccos(a - 1)
    theta_1 = 2 * qnp.arccos(b)
    theta_2 = 2 * qnp.arccos(g)

    #     print(theta_0, theta_1, theta_2)

    qml.ctrl(qml.RY, control=wire_l, control_values=[0, 0])(theta_0, wires=ancilla)
    qml.ctrl(qml.RY, control=wire_l, control_values=[1, 0])(theta_1, wires=ancilla)
    qml.ctrl(qml.RY, control=wire_l, control_values=[0, 1])(theta_2, wires=ancilla)


# Sanity check: ~~~~~~~~~~~~~~~~~~~~~~
# with qml.tape.QuantumTape() as t:
#     oracle_a("ancilla", ["l0", "l1"], "j", 0.1, -0.6, 0.3)

# print(t.draw(wire_order=["ancilla", "l1", "l0", "j"]))
# print(t.operations)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


dev = qml.device("lightning.qubit", wires=["ancilla", "l1", "l0", "j2", "j1", "j0"])


@qml.qnode(dev)
def complete_circuit(a, b, g):
    for w in ["l0", "l1"]:  # hadamard transform over |l> register
        qml.Hadamard(w)

    #     qml.Barrier()

    oracle_a("ancilla", ["l0", "l1"], ["j0", "j1", "j2"], a, b, g)

    #     qml.Barrier()

    oracle_c(["l0", "l1"], ["j0", "j1", "j2"])

    #     qml.Barrier()

    for w in ["l0", "l1"]:  # hadamard transform over |l> register
        qml.Hadamard(w)
    return qml.state()


print("Quantum Circuit:")
print(qml.draw(complete_circuit)(alpha, beta, gamma), "\n")

print("BlockEncoded Mat:")
mat = qml.matrix(complete_circuit)(alpha, beta, gamma)[:8, :8] * s
print(mat, "\n")

##############################################################################
# Summary and conclusions
# -----------------------
# Block encoding is a powerful technique in quantum computing that allows implementing a non-unitary
# operation in a quantum circuit typically by embedding the operation in a larger unitary operation.
# Here we reviewed some important block encoding methods with code examples. The choice of the block
# encoding approach depends on the non-unitary operation we want to implement. The functionality
# provided in PennyLane allows you to explore and benchmark different approaches for a desired
# problem. Can you select a matrix and compare different block encoding methods for it?
#
# References
# ----------
#
# .. [#fable]
#
#    Daan Camps, Roel Van Beeumen,
#    "FABLE: fast approximate quantum circuits for block-encodings",
#    `arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`__, 2022
#
#
# .. [#sparse]
#
#     Daan Camps, Lin Lin, Roel Van Beeumen, Chao Yang,
#     "Explicit quantum circuits for block encodings of certain sparse matrices",
#     `arXiv:2203.10236 <https://arxiv.org/abs/2203.10236>`__, 2022
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/diego_guala.txt
#
# .. include:: ../_static/authors/soran_jahangiri.txt
#
# .. include:: ../_static/authors/jay_soni.txt.txt
