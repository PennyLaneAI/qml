r"""

Block encoding with matrix access oracles
=========================================

.. meta::
    :property="og:description": Learn how to perform block encoding
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_intro_qsvt.png

.. related::
     tutorial_intro_qsvt Intro to QSVT

*Author: Jay Soni, Diego Guala, Soran Jahangiri — Posted: September 29, 2023.*

Prominent quantum algorithms such as quantum phase estimation and quantum singular value
transform require encoding a non-unitary matrix in a quantum circuit. This seems problematic
because quantum computers can only perform unitary evolutions. Block encoding is a technique 
that solves this problem by embedding a non-unitary operator as a sub-block of a larger unitary 
matrix. 

In previous demos we have discussed methods for `simulator-friendly <https://pennylane.ai/qml/demos/tutorial_intro_qsvt#transforming-matrices-encoded-in-matrices>`_
encodings and block encodings using `linear combination of unitaries (LCU) decompositions <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`_.
In this tutorial we explore another general block encoding method that can be very efficient for 
sparse and structured matrices.

Circuits with matrix access oracles
-----------------------------------
A general circuit for block encoding an arbitrary matrix :math:`A \in \mathbb{C}^{N x N}` with :math:`N = 2^{n}` 
can be constructed as shown in the figure below.

.. figure:: ../demonstrations/block_encoding/general_circuit.png
    :width: 50%
    :align: center

Where the :math:`H^{\otimes n}` operation is a Hadamard transformation on :math:`n` qubits. The
:math:`U_A` and :math:`U_B` operations are oracles which give us access to the elements of 
the matrix we wish to block encode. The specific action of the oracles are defined below: 

.. math:: U_A |i\rangle |j\rangle \ = ( A_{i, b(i,j)}|0\rangle + \sqrt{1 - |A_{i, b(i,j)}|^2}|1\rangle ) |i\rangle |j\rangle

:math:`U_A`, in the most general case, can be constructed from a sequence of uniformly
controlled rotation gates. The rotation angles are computed from the elements of the block encoded 
matrix as :math:`\theta = \text{arccos}(a_{ij})`. The gate complexity of this circuit is :math:`O(N^4)` 
which makes its implementation highly inefficient.

Let :math:`b(i,j)` be a function such that it takes a column index (:math:`j`) and returns the 
row index for the :math:`i^{th}` non-zero entry in that column of :math:`A`. Note, if :math:`A` 
is treated as completely dense (no non-zero entries), this function simply returns :math:`i`.
We use this to define the oracle :math:`U_B`:

.. math:: U_B \cdot |i\rangle|j\rangle \ = \ |i\rangle |b(i,j)\rangle

Finding an optimal quantum gates decomposition that implements :math:`U_A` and
:math:`U_B` is not always possible. We now explore two approaches for the construction of
these oracles that can be very efficient for matrices with specific sparsity and structure.

Block encoding with FABLE
-------------------------
The "Fast Approximate quantum circuits for BLock Encodings" (FABLE) technique is a general method
for block encoding dense and sparse matrices. The level of approximation in FABLE can be adjusted
to simplify the resulting circuit. For matrices with specific structures, FABLE provides an
efficient circuit without reducing accuracy.

The FABLE circuit is constructed from a set of rotation and C-NOT gates. The rotation angles,
:math:`\Theta = (\theta_1, ..., \theta_n)`, are obtained from a transformation of the elements of
the block encoded matrix

.. math:: \left ( H^{\otimes 2n} P \right ) \Theta = C,

where :math:`P` is a permutation that transforms a binary ordering to the Gray code ordering,
:math:`C = (\text{arccos}(A_{00}), ..., \text{arccos}(A_{nn}))` is obtained from the matrix elements
of the matrix :math:`A`, and :math:`H` is defined as

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

print(f"Original matrix:\n{A}", "\n")
M = len(A) * qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:len(A),0:len(A)]
print(f"Block-encoded matrix:\n{M}", "\n")

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

print(f"Original matrix:\n{A}", "\n")
M = len(A) * qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:len(A),0:len(A)]
print(f"Block-encoded matrix:\n{M}", "\n")

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
# where :math:`\alpha`, :math:`\beta` and :math:`\gamma` are real numbers. The following code block
# prepares the matrix representation of :math:`A` for an :math:`8 \times 8` sparse matrix.

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
# The :math:`U_A` oracle for this matrix is constructed from controlled rotation gates, similar to
# the FABLE circuit.

def UA(ancilla, wire_i, theta):
    qml.ctrl(qml.RY, control=wire_i, control_values=[0, 0])(theta[0], wires=ancilla)
    qml.ctrl(qml.RY, control=wire_i, control_values=[1, 0])(theta[1], wires=ancilla)
    qml.ctrl(qml.RY, control=wire_i, control_values=[0, 1])(theta[2], wires=ancilla)

##############################################################################
# The :math:`U_B` oracle is defined in terms of the so-called "Left" and "Right" shift operators.
# They correspond to the modular arithmetic operations :math:`+1` or :math:`-1` respectively ([#sparse]_).

def shift_op(s_wires, shift="Left"):
    control_values = [1, 1] if shift == "Left" else [0, 0]
    qml.ctrl(qml.PauliX, control=s_wires[:2], control_values=control_values)(wires=s_wires[2])
    qml.ctrl(qml.PauliX, control=s_wires[0], control_values=control_values[0])(wires=s_wires[1])
    qml.PauliX(s_wires[0])


def UB(wires_i, wires_j):
    qml.ctrl(shift_op, control=wires_i[0])(wires_j, shift="Left")
    qml.ctrl(shift_op, control=wires_i[1])(wires_j, shift="Right")

##############################################################################
# We now construct our circuit to block encode the sparse matrix.

dev = qml.device("default.qubit", wires=["ancilla", "i1", "i0", "j2", "j1", "j0"])

@qml.qnode(dev)
def complete_circuit(theta):
    for w in ["i0", "i1"]:  # hadamard transform over |i> register
        qml.Hadamard(w)

    UA("ancilla", ["i0", "i1"], theta)

    UB(["i0", "i1"], ["j0", "j1", "j2"])

    for w in ["i0", "i1"]:  # hadamard transform over |l> register
        qml.Hadamard(w)

    return qml.state()

s = 4  # normalization constant
theta = 2 * np.arccos(np.array([alpha - 1, beta, gamma]))

print("Quantum Circuit:")
print(qml.draw(complete_circuit)(theta), "\n")

print("BlockEncoded Mat:")
mat = qml.matrix(complete_circuit)(theta).real[:8, :8] * s
print(mat, "\n")

##############################################################################
# You can confirm that the circuit block encodes the original sparse matrix defined above.
#
# Conclusion
# -----------------------
# Block encoding is a powerful technique in quantum computing that allows us to implement a non-unitary
# operation in a quantum circuit by embedding the operation in a larger unitary gate.
# In this demo, we reviewed two important block encoding methods with code examples using PennyLane.
# The block encoding functionality provided in PennyLane allows you to explore and benchmark several
# block encoding approaches for a desired problem. The efficiency of the block encoding methods
# typically depends on the sparsity and structure of the original matrix. We hope that you can use 
# these tips and tricks to find a more efficient block encoding for your matrix! 
#
# References
# ----------
#
# .. [#fable]
#
#     Daan Camps, Roel Van Beeumen,
#     "FABLE: fast approximate quantum circuits for block-encodings",
#     `arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`__, 2022
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
