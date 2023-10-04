r"""

Block Encoding
==============

.. meta::
    :property="og:description": Learn how to perform block encoding
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_block_encoding.png

.. related::
     tutorial_intro_qsvt Intro to QSVT

*Author: Diego Guala, Soran Jahangiri, Jay Soni — Posted: September 29, 2023.*

Prominent quantum algorithms such as Quantum Phase estimation and Quantum Singular Value
Transformation require implementing a non-unitary operator in a quantum circuit. This is problematic
because quantum computers can only perform unitary evolutions. Block encoding is a general technique
that solves this problem by embedding the non-unitary operator in a unitary matrix that can be
implemented in a quantum circuit containing a set of ancilla qubits. In this [link] demo, we
learned how to block encode a matrix by simply embedding it in a larger unitary matrix using the
:class:`~pennylane.BlockEncode` operation. We also learned [link] a powerful method for block encoding a
matrix by decomposing it into a linear combination of unitaries (LCU) and then block encode the LCU.
In this tutorial we explore a general block encoding method that can be efficiently implemented for
sparse and structured matrices. We first explain the method and then apply it to specific examples
that can be efficiently block-encoded.

A general circuit for block encoding
------------------------------------
An arbitrary matrix :math:`A`, can be block encoded by relying on oracle access to the entries in
the matrix (see [#fable]_, [#sparse]_). A general circuit for block encoding :math:`A` can be
constructed from such oracles as illustrated in the following.

.. figure:: ../demonstrations/block_encoding/general_circuit.png
    :width: 50%
    :align: center

Finding the optimal sequence of the quantum gates that implement the :math:`\hat{O}_{A}` and
:math:`\hat{O}_{C}` oracles is not straightforward for random matrices. In the general case,
:math:`\hat{O}_{C}` can be represented by a set of SWAP gates and :math:`\hat{O}_{A}` can be
constructed from a sequence of uniformly controlled rotation gates:

.. figure:: ../demonstrations/block_encoding/rotation_circuit.png
    :width: 50%
    :align: center

The rotation angles are computed from the matrix elements of the block encoded matrix as
:math:`\theta = arcsin(a_{ij}`. The gate complexity of this circuit is O(N4) which makes its
implementation highly inefficent. We now explain two approaches that provide alternative
constructions of :math:`\hat{O}_{A}` and :math:`\hat{O}_{C}` that can be very efficient for matrices
with specific sparsity and structure.

Block encoding with FABLE
-------------------------
The fast approximate quantum circuits for block encodings (FABLE) is a general method
for block encoding dense and sparse matrices. The level of approximation in FABLE can be adjusted
to compress and sparsify the resulting circuit. For matrices with specific structures, FABLE
provides an efficient circuit without sacrificing accuracy. The general circuit is constructed
from a set of rotation and C-NOT gates where the rotation angles are obtained from a transformation
of the elements of the block encoded matrix. Let's construct a FABLE circuit for a matrix that is
structured.
"""

import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylnae import numpy as qnp

n = 2
N = 2**n

A = qnp.array([[-0.51192128, -0.51192128,  0.6237114 ,  0.6237114 ],
               [ 0.97041007,  0.97041007,  0.99999329,  0.99999329],
               [ 0.82429855,  0.82429855,  0.98175843,  0.98175843],
               [ 0.99675093,  0.99675093,  0.83514837,  0.83514837]])


Avec = qnp.ravel(A)
alpha = max(1,qnp.linalg.norm(Avec,qnp.inf))
Avec = Avec/alpha


alphas = 2*qnp.arccos(Avec)
thetas = compute_theta(alphas)


code = gray_code(N)
num_selections=len(code)

control_wires = control_indices = [
    int(qnp.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
    for i in range(num_selections)
]

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

##############################################################################
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
print(alpha*N*qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:N,0:N])

##############################################################################
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

##############################################################################
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

##############################################################################
# Block-encoding sparse matrices
# ------------------------------
# The quantum circuit for the oracle :math:`\hat{O}_{A}`, presented above, requires on the order of
# :math:`~ O(N^{2})` gates to implement. In the case where :math:`A` is a structured sparse matrix,
# we can generate a more efficient quantum circuit representation for the oracle. Let's construct
# the circuit for a sparse matrix that has repeated entries.

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


def shift_circ(s_wires, shift="L"):
    control_values = [1, 1] if shift == "L" else [0, 0]

    qml.ctrl(qml.PauliX, control=s_wires[:2], control_values=control_values)(wires=s_wires[2])
    qml.ctrl(qml.PauliX, control=s_wires[0], control_values=control_values[0])(wires=s_wires[1])
    qml.PauliX(s_wires[0])


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



dev = qml.device("lightning.qubit", wires=["ancilla", "l1", "l0", "j2", "j1", "j0"])


@qml.qnode(dev)
def complete_circuit(a, b, g):
    for w in ["l0", "l1"]:  # hadamard transform over |l> register
        qml.Hadamard(w)


    oracle_a("ancilla", ["l0", "l1"], ["j0", "j1", "j2"], a, b, g)


    oracle_c(["l0", "l1"], ["j0", "j1", "j2"])


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
# Here we reviewed some important block encoding methods with code examples using PennyLane. The
# efficiency of the block encoding scheme depends on the sparsity and structure of the matrix we
# want to block encode. The block encoding functionality provided in PennyLane allows you to explore
# and benchmark different approaches for a desired problem. Can you select a matrix and compare
# different block encoding methods for it?
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
# .. include:: ../_static/authors/jay_soni.txt
