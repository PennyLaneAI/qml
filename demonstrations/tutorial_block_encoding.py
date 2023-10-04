r"""

Block encoding with matrix query oracles
========================================

.. meta::
    :property="og:description": Learn how to perform block encoding
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_block_encoding.png

.. related::
     tutorial_intro_qsvt Intro to QSVT

*Author: Diego Guala, Jay Soni, Soran Jahangiri — Posted: September 29, 2023.*

Prominent quantum algorithms such as Quantum Phase Estimation and Quantum Singular Value
Transformation require implementing a non-unitary operator in a quantum circuit. This is problematic
because quantum computers can only perform unitary evolutions. Block encoding is a general technique
that solves this problem by embedding the non-unitary operator in a unitary matrix that can be
implemented in a quantum circuit using a set of ancilla qubits. In
`this <https://pennylane.ai/qml/demos/tutorial_intro_qsvt#transforming-matrices-encoded-in-matrices>`_
demo, we learned how to block encode a matrix by simply embedding it in a larger unitary matrix
using the :class:`~pennylane.BlockEncode` operation. We also
`learned <https://github.com/PennyLaneAI/qml/pull/888>`_ a powerful method for block encoding a
matrix by decomposing it into a linear combination of unitaries (LCU) and then block encode the LCU.
In this tutorial we explore another general block encoding method that can be particularly
efficient for sparse and structured matrices. We first explain the method and then apply it to
some selected examples.

Circuits with matrix query oracles
----------------------------------
An arbitrary matrix :math:`A`, can be block encoded by relying on oracle access to its entries
(see [#fable]_, [#sparse]_). A general circuit for block encoding :math:`A` can be constructed from
such oracles.

.. figure:: ../demonstrations/block_encoding/general_circuit.png
    :width: 50%
    :align: center

Finding the optimal sequence of the quantum gates that implement the :math:`O_A` and :math:`O_C`
oracles is not straightforward for random matrices. In the general case, :math:`O_A` can be
constructed from a sequence of uniformly controlled rotation gates and :math:`O_C` can be
represented by a set of SWAP gates.

.. figure:: ../demonstrations/block_encoding/fable_circuit.png
    :width: 50%
    :align: center

The rotation angles are computed from the elements of the block encoded matrix as
:math:`\theta = \text{arccos}(a_{ij}`. The gate complexity of this circuit is :math:`O(N^4)` which
makes its implementation highly inefficient. We now explain two approaches that provide alternative
constructions of :math:`O_A` and :math:`O_C` that can be very efficient specially for matrices with
specific sparsity and structure.

Block encoding with FABLE
-------------------------
The fast approximate quantum circuits for block encodings (FABLE) is a general method
for block encoding dense and sparse matrices. The level of approximation in FABLE can be adjusted
to compress and sparsify the resulting circuit. For matrices with specific structures, FABLE
provides an efficient circuit without sacrificing accuracy. The general circuit is constructed
from a set of rotation and C-NOT gates where the rotation angles are obtained from a transformation
of the elements of the block encoded matrix. The rotation angles,
:math:`\Theta = (\theta_1, ..., \theta_n)`, are obtained with

.. math:: \left ( H^{\otimes 2n} P \right ) \Theta = C,

where :math:`P` is a permutation that transforms a binary ordering to the Gray code ordering,
:math:`C = (\text{arccos}(A_00), ..., \text{arccos}(A_nn))` is obtained from the matrix elements of
the matrix :math:`A` and :math:`H` is defined as

.. math:: H = \begin{pmatrix} 1 & 1\\
                             1 & -1
               \end{pmatrix}.

Let's now construct the block encoding circuit for a structured matrix.
"""

import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane import numpy as qnp

A = qnp.array([[-0.51192128, -0.51192128,  0.6237114 ,  0.6237114 ],
               [ 0.97041007,  0.97041007,  0.99999329,  0.99999329],
               [ 0.82429855,  0.82429855,  0.98175843,  0.98175843],
               [ 0.99675093,  0.99675093,  0.83514837,  0.83514837]])


##############################################################################
# We now normalize the matrix and compute the rotation angles.

Avec = qnp.ravel(A)
alpha = max(1,qnp.linalg.norm(Avec,qnp.inf))
Avec = Avec/alpha


alphas = 2*qnp.arccos(Avec)
thetas = compute_theta(alphas)


code = gray_code(len(A))
num_selections=len(code)

control_wires = control_indices = [
    int(qnp.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
    for i in range(num_selections)
]

##############################################################################
# We construct the :math:`O_A` and :math:`O_C` oracles as well as the Ds operator, which is a tensor
# product of Hadamard gates.

def OA():
    for idx in range(len(thetas)):
        qml.RY(thetas[idx],wires=4)
        qml.CNOT(wires=[control_wires[idx],4])

def OC():
    qml.SWAP(wires=[0,2])
    qml.SWAP(wires=[1,3])

def Ds():
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)

##############################################################################
# We construct the circuit using these oracles and draw it.

dev = qml.device('default.qubit', wires=5)
@qml.qnode(dev)
def circuit():
    Ds()
    OA()
    OC()
    Ds()
    return qml.state()

print(qml.draw(circuit)())

##############################################################################
# We compute the matrix representation of the circuit and print its top-left block to compare it
# with the original matrix

print(A)
print(alpha*len(A)*qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:len(A),0:len(A)])

##############################################################################
# You can easily confirm that the circuit block encodes the original matrix defined above.
#
# The interesting thing about the FABLE method is that one can eliminate those rotation gates that
# have an angle smaller than a selected threshold. This leaves a sequence of C-NOT gates that in
# most cases cancel each other out.

tolerance= 0.01

def OA():
    for idx in range(len(thetas)):
        if abs(thetas[idx])>tolerance:
            qml.RY(thetas[idx],wires=4)
        qml.CNOT(wires=[control_wires[idx],4])

print(qml.draw(circuit)())

##############################################################################
# You can confirm that two C-NOT gates applied to the same wires cancel each other. Compressing the
# circuit in this way is an approximation. Let's see how good this approximation is in the case of
# our example.

def OA():
    nots=[]
    for idx in range(len(thetas)):
        if abs(thetas[idx])>tolerance:
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
print(alpha*len(A)*qml.matrix(circuit,wire_order=[0,1,2,3,4][::-1])()[0:len(A),0:len(A)])

##############################################################################
# You can see that the compressed circuit is equivalent to the original circuit simply because our
# matrix is highly structured and many of the rotation angles are zero. This is not the case for any
# given matrix. Can you construct a matrix that also allows a significant compression of the block
# encoding circuit without affecting the accuracy?

##############################################################################
# Block-encoding sparse matrices
# ------------------------------
# The quantum circuit for the oracle :math:`\hat{O}_{A}`, presented above, accesses every entry of 
# :math:`A` and thus requires on the order of :math:`~ O(N^2)` gates to implement ([#fable]_).
# In the case where  :math:`A` is a structured sparse matrix, we can generate a more efficient quantum 
# circuit representation for both oracles. Let's see how we can implement these oracles for a given
# structured sparse matrix.  
#
# Consider the s-sparse matrix given by:
#
# .. math:: A = \begin{bmatrix}
#       \alpha & \gamma & 0 & \dots & \beta\\
#       \beta & \alpha & \gamma & \ddots & 0 \\
#       0 & \beta & \alpha & \gamma \ddots & 0\\
#       0 & \ddots & \beta & \alpha & \gamma\\
#       \gamma & 0 & \dots & \beta & \alpha \\
#       \end{bmatrix}
#
# The following code block prepares the matrix representation for an :math:`8x8` sparse matrix:

s = 4       # normalization constant 
alpha = 0.1
beta  = 0.6
gamma = 0.3

A = qnp.array([[alpha, gamma,     0,     0,     0,     0,     0,  beta],
               [ beta, alpha, gamma,     0,     0,     0,     0,     0],
               [    0,  beta, alpha, gamma,     0,     0,     0,     0],
               [    0,     0,  beta, alpha, gamma,     0,     0,     0],
               [    0,     0,     0,  beta, alpha, gamma,     0,     0],
               [    0,     0,     0,     0,  beta, alpha, gamma,     0],
               [    0,     0,     0,     0,     0,  beta, alpha, gamma],
               [gamma,     0,     0,     0,     0,     0,  beta, alpha]])

print(f"Original A:\n{A}", "\n")

##############################################################################
# The :math:`\hat{O}_{C}` oracle for this matrix is defined in terms of the so called "Left" and "Right"
# shift operators ([#sparse]_). 

def shift_op(s_wires, shift="Left"):
    control_values = [1, 1] if shift == "Left" else [0, 0]

    qml.ctrl(qml.PauliX, control=s_wires[:2], control_values=control_values)(wires=s_wires[2])
    qml.ctrl(qml.PauliX, control=s_wires[0], control_values=control_values[0])(wires=s_wires[1])
    qml.PauliX(s_wires[0])


def oracle_c(wires_l, wires_j):
    qml.ctrl(shift_op, control=wires_l[0])(wires_j, shift="Left")
    qml.ctrl(shift_op, control=wires_l[1])(wires_j, shift="Right")


def oracle_a(ancilla, wire_l, wire_j, a, b, g):
    theta_0 = 2 * qnp.arccos(a - 1)
    theta_1 = 2 * qnp.arccos(b)
    theta_2 = 2 * qnp.arccos(g)

    qml.ctrl(qml.RY, control=wire_l, control_values=[0, 0])(theta_0, wires=ancilla)
    qml.ctrl(qml.RY, control=wire_l, control_values=[1, 0])(theta_1, wires=ancilla)
    qml.ctrl(qml.RY, control=wire_l, control_values=[0, 1])(theta_2, wires=ancilla)

##############################################################################
# Finally, we can bring the oracles together in our circuit to block encode the sparse matrix:

dev = qml.device("default.qubit", wires=["ancilla", "l1", "l0", "j2", "j1", "j0"])

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
# .. include:: ../_static/authors/jay_soni.txt
#
# .. include:: ../_static/authors/soran_jahangiri.txt
