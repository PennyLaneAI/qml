r"""

Block encoding with matrix access oracles
=========================================

.. meta::
    :property="og:description": Learn how to perform block encoding
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_intro_qsvt.png

.. related::
     tutorial_intro_qsvt Intro to QSVT

*Author: Diego Guala, Jay Soni, Soran Jahangiri — Posted: September 29, 2023.*

Prominent quantum algorithms such as quantum phase estimation and quantum singular value
transformation algorithms require encoding a non-unitary matrix in a quantum circuit. This requires
using sophisticated methods because quantum computers can only perform unitary evolutions. Block
encoding is a general technique that solves this problem by embedding a non-unitary operator in a
unitary matrix that can be implemented in a quantum circuit. In
`this <https://pennylane.ai/qml/demos/tutorial_intro_qsvt#transforming-matrices-encoded-in-matrices>`_
demo, we learned how to block encode a non-unitary matrix by simply embedding it in a larger unitary
matrix using the :class:`~pennylane.BlockEncode` operation. We also
`learned <https://github.com/PennyLaneAI/qml/pull/888>`_ a powerful method for block encoding a
matrix by decomposing it into a linear combination of unitaries (LCU) and then block encode the LCU.
In this tutorial we explore another general block encoding method that can be very efficient for
sparse and structured matrices. We first explain the method and then apply it to some examples.

Circuits with matrix access oracles
-----------------------------------
A general circuit for block encoding an arbitrary matrix :math:`A` can be constructed as shown in
the figure below.

.. figure:: ../demonstrations/block_encoding/general_circuit.png
    :width: 50%
    :align: center

The :math:`H^{\otimes n}` operation is a Hadamard transformation on :math:`n` qubits. The
:math:`U_A` and :math:`U_A` operations, in the most general case, can be constructed from a sequence
of uniformly controlled rotation gates and a set of SWAP gates, respectively. The rotation angles
are computed from the elements of the block encoded matrix as :math:`\theta = \text{arccos}(a_{ij}`.
The gate complexity of this circuit is :math:`O(N^4)` which makes its implementation highly
inefficient.

Finding the optimal sequence of the quantum gates that implement :math:`U_A` and
:math:`U_B` is not always straightforward. We now explain two approaches for the construction of
these oracles that can be very efficient specially for matrices with specific sparsity and
structure.

Block encoding with FABLE
-------------------------
The fast approximate quantum circuits for block encodings (FABLE) is a general method
for block encoding dense and sparse matrices. The level of approximation in FABLE can be adjusted
to simplify the resulting circuit. For matrices with specific structures, FABLE provides an
efficient circuit without reducing accuracy.

The FABLE circuit is constructed from a set of rotation and C-NOT gates. The rotation angles,
:math:`\Theta = (\theta_1, ..., \theta_n)`, are obtained from a transformation of the elements of
the block encoded matrix

.. math:: \left ( H^{\otimes 2n} P \right ) \Theta = C,

where :math:`P` is a permutation that transforms a binary ordering to the Gray code ordering,
:math:`C = (\text{arccos}(A_00), ..., \text{arccos}(A_nn))` is obtained from the matrix elements of
the matrix :math:`A` and :math:`H` is defined as

.. math:: H = \begin{pmatrix} 1 & 1\\
                             1 & -1
               \end{pmatrix}.

Let's now construct the FABLE block encoding circuit for a structured matrix.
"""

import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane import numpy as np

A = np.array([[-0.51192128, -0.51192128,  0.6237114 ,  0.6237114 ],
              [ 0.97041007,  0.97041007,  0.99999329,  0.99999329],
              [ 0.82429855,  0.82429855,  0.98175843,  0.98175843],
              [ 0.99675093,  0.99675093,  0.83514837,  0.83514837]])

##############################################################################
# We now compute the rotation angles and obtain the control wires for the C-NOT gates.

alphas = 2 * np.arccos(A).flatten()
thetas = compute_theta(alphas)

code = gray_code(len(A))
n_selections = len(code)

control_wires = [
    int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % n_selections], 2)))
    for i in range(n_selections)
]

##############################################################################
# We construct the :math:`U_A` and :math:`U_B` oracles as well as an operator representing the
# tensor product of Hadamard gates.

def UA():
    for idx in range(len(thetas)):
        qml.RY(thetas[idx],wires=4)
        qml.CNOT(wires=[control_wires[idx],4])

def UB():
    qml.SWAP(wires=[0,2])
    qml.SWAP(wires=[1,3])

def HN():
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)

##############################################################################
# We construct the circuit using these oracles and draw it.

dev = qml.device('default.qubit', wires=5)
@qml.qnode(dev)
def circuit():
    HN()
    UA()
    UB()
    HN()
    return qml.state()

print(qml.draw(circuit)())

##############################################################################
# We compute the matrix representation of the circuit and print its top-left block to compare it
# with the original matrix.

print(A)
print(len(A) * qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:len(A),0:len(A)])

##############################################################################
# You can easily confirm that the circuit block encodes the original matrix defined above.
#
# The interesting point about the FABLE method is that we can eliminate those rotation gates that
# have an angle smaller than a defined threshold. This leaves a sequence of C-NOT gates that in
# most cases cancel each other out. You can confirm that two C-NOT gates applied to the same wires
# cancel each other. Let's now remove all the rotation gates that have an angle smaller than
# :math:`0.01` and draw the circuit.

tolerance= 0.01

def UA():
    for idx in range(len(thetas)):
        if abs(thetas[idx])>tolerance:
            qml.RY(thetas[idx],wires=4)
        qml.CNOT(wires=[control_wires[idx],4])

print(qml.draw(circuit)())

##############################################################################
# Compressing the circuit by removing some of the rotations is an approximation. We can now see how
# good this approximation is in the case of our example.

def UA():
    nots=[]
    for idx in range(len(thetas)):
        if abs(thetas[idx]) > tolerance:
            for cidx in nots:
                qml.CNOT(wires=[cidx,4])
            qml.RY(thetas[idx],wires=4)
            nots=[]
        if control_wires[idx] in nots:
            del(nots[nots.index(control_wires[idx])])
        else:
            nots.append(control_wires[idx])
    qml.CNOT(nots+[4])

print(qml.draw(circuit)())

print(A)
print(len(A) * qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:len(A),0:len(A)])

##############################################################################
# You can see that the compressed circuit is equivalent to the original circuit. This happens
# because our original matrix is highly structured and many of the rotation angles are zero.
# However, this is not always true for an arbitrary matrix. Can you construct another matrix that
# allows a significant compression of the block encoding circuit without affecting the accuracy?
#
# Block-encoding sparse matrices
# ------------------------------
# The quantum circuit for the oracle :math:`U_A`, presented above, accesses every entry of
# :math:`A` and thus requires :math:`~ O(N^2)` gates to implement the oracle ([#fable]_). In the
# special cases where :math:`A` is structured and sparse, we can generate a more efficient quantum
# circuit representation for :math:`U_A` and :math:`U_B`. Let's look at an example.
#
# Consider the sparse matrix given by:
#
# .. math:: A = \begin{bmatrix}
#       \alpha & \gamma & 0 & \dots & \beta\\
#       \beta & \alpha & \gamma & \ddots & 0 \\
#       0 & \beta & \alpha & \gamma \ddots & 0\\
#       0 & \ddots & \beta & \alpha & \gamma\\
#       \gamma & 0 & \dots & \beta & \alpha \\
#       \end{bmatrix},
#
# where :math:`alpha`, :math:`beta` and :math:`gamma` are real numbers. The following code block
# prepares the matrix representation of :math:`A` for an :math:`8 x 8` sparse matrix.

s = 4       # normalization constant 
alpha, beta, gamma  = 0.1, 0.6, 0.3

A = np.array([[alpha, gamma,     0,     0,     0,     0,     0,  beta],
              [ beta, alpha, gamma,     0,     0,     0,     0,     0],
              [    0,  beta, alpha, gamma,     0,     0,     0,     0],
              [    0,     0,  beta, alpha, gamma,     0,     0,     0],
              [    0,     0,     0,  beta, alpha, gamma,     0,     0],
              [    0,     0,     0,     0,  beta, alpha, gamma,     0],
              [    0,     0,     0,     0,     0,  beta, alpha, gamma],
              [gamma,     0,     0,     0,     0,     0,  beta, alpha]])

print(f"Original A:\n{A}", "\n")

##############################################################################
# The :math:`U_B` oracle for this matrix is defined in terms of the so-called "Left" and "Right"
# shift operators ([#sparse]_). 

def shift_op(s_wires, shift="Left"):
    control_values = [1, 1] if shift == "Left" else [0, 0]

    qml.ctrl(qml.PauliX, control=s_wires[:2], control_values=control_values)(wires=s_wires[2])
    qml.ctrl(qml.PauliX, control=s_wires[0], control_values=control_values[0])(wires=s_wires[1])
    qml.PauliX(s_wires[0])


def UB(wires_l, wires_j):
    qml.ctrl(shift_op, control=wires_l[0])(wires_j, shift="Left")
    qml.ctrl(shift_op, control=wires_l[1])(wires_j, shift="Right")


def UA(ancilla, wire_l, wire_j, a, b, g):
    theta_0 = 2 * np.arccos(a - 1)
    theta_1 = 2 * np.arccos(b)
    theta_2 = 2 * np.arccos(g)

    qml.ctrl(qml.RY, control=wire_l, control_values=[0, 0])(theta_0, wires=ancilla)
    qml.ctrl(qml.RY, control=wire_l, control_values=[1, 0])(theta_1, wires=ancilla)
    qml.ctrl(qml.RY, control=wire_l, control_values=[0, 1])(theta_2, wires=ancilla)

##############################################################################
# We construct our circuit to block encode the sparse matrix.

dev = qml.device("default.qubit", wires=["ancilla", "l1", "l0", "j2", "j1", "j0"])

@qml.qnode(dev)
def complete_circuit(a, b, g):
    for w in ["l0", "l1"]:  # hadamard transform over |l> register
        qml.Hadamard(w)

    UA("ancilla", ["l0", "l1"], ["j0", "j1", "j2"], a, b, g)

    UB(["l0", "l1"], ["j0", "j1", "j2"])

    for w in ["l0", "l1"]:  # hadamard transform over |l> register
        qml.Hadamard(w)

    return qml.state()


print("Quantum Circuit:")
print(qml.draw(complete_circuit)(alpha, beta, gamma), "\n")

print("BlockEncoded Mat:")
mat = qml.matrix(complete_circuit)(alpha, beta, gamma)[:8, :8] * s
print(mat, "\n")

##############################################################################
# You can confirm that the circuit block encodes the original sparse matrix defined above.
#
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
# .. include:: ../_static/authors/jay_soni.txt
#
# .. include:: ../_static/authors/diego_guala.txt
#
# .. include:: ../_static/authors/soran_jahangiri.txt
