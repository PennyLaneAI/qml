r"""
Stabilizer codes for Quantum Error correction
=================================================

State-of-the-art quantum devices, such as IBM's Condor and Atom Computing's optical lattices, contain more than 
a thousand qubits. Does this qubit count suffice for valuable quantum algorithms with clear speedups?
The reality is that there is more to the story than the sheer number of qubits. As we currently stand, quantum
devices are still prone to errors that increase with device size. For this reason, **quantum error correction**--one of the most important domain in the universe of quantum computing--has 
been gaining traction. 

Quantum error correction is done via schemes known as **error correction codes.**
These are quantum algorithms that come in many varieties that address different error types.
Is there a unified way to understand these codes? The **stabilizer formalism** provides such a framework for
a large class of quantum error correction codes. The so-called
**stabilizer codes**, such as the repetition, Shor, Steane, and surface codes, 
all fall under this formalism. Other codes, like the GKP codes used by Xanadu, lie outside of it.

In this demo, we will introduce the stabilizer formalism using bottom-up approach. We construct and
some well-known codes using the quantum circuit formalism and then derive their **stabilizer generators,** from which
the code can be reconstructed. This enables the construction of a wide range of error correction codes 
directly from their stabilizer generators. 

A toy example: the repetition code
-----------------------------------

To start with, let us explore the general structure of error correction codes using a simple example: the **three-qubit repetition code.** 
We will introduce this code as a quantum circuit with definite steps to gain some intuition on how it corrects errors on qubits. 
The quantum circuit representation is known as the **state picture**. In this formalism, error correction codes follow a simple structure:

- Qubit encoding
- Error detection
- Error correction

Let us describe each element in detail.

Qubit encoding
~~~~~~~~~~~~~~~

The first step in an error correction code is **encoding** one abstract or **logical qubit** into a set of many on-device **physical qubits.** 
The rationale is that, if some external factor changes the state of one of the qubits, the remaining qubits still provide information about the original logical qubit. 
For example, in the three-qubit repetition code, the logical basis-state 
qubits, or **logical codewords**, :math:`\vert \bar{0}\rangle` and :math:`\vert \bar{1}\rangle` are encoded into three physical qubits via

.. math::

    \vert \bar{0} \rangle \mapsto \vert 000 \rangle, \quad \vert \bar{1} \rangle \mapsto \vert 111 \rangle.

A general qubit :math:`\vert \bar{\psi}\rangle = \alpha \vert \bar{0}\rangle + \beta \vert \bar{1}\rangle` is then encoded as

.. math::

    \alpha \vert \bar{0}\rangle + \beta \vert \bar{1}\rangle \mapsto \alpha \vert 000 \rangle + \beta \vert 111\rangle.

This encoding can be done via the following quantum circuit.
 
**INSERT PICTURE**

Let's code this below and verify the output

"""

import pennylane as qml
from pennylane import numpy as np

def encode(alpha, beta):

    qml.StatePrep([alpha, beta], wires = 0)
    qml.CNOT(wires = [0, 1])
    qml.CNOT(wires = [0, 2])


def encoded_state(alpha, beta):

    encode(alpha, beta)
    return qml.state()

encode_qnode = qml.QNode(encoded_state, qml.device("default.qubit"))

alpha = 1 / np.sqrt(2)
beta = 1 / np.sqrt(2)

encode_qnode = qml.QNode(encoded_state, qml.device("default.qubit"))

print("|000> component: ", encode_qnode(alpha, beta)[0])
print("|111> component: ", encode_qnode(alpha, beta)[7])

##############################################################################
#
# Now, suppose that a **bit-flip** error occurs on the second qubit, meaning that the qubit is randomly flipped. This can be modelled as
# an unwanted Pauli-$X$ operator being applied on :math:`\vert \bar{\psi}\rangle:`
#
# .. math::
#
#     X_2 \vert \bar{\psi}\rangle = \alpha \vert 010 \rangle + \beta \vert 101 \rangle
#
# If we are sure that **only one bit-flip error occurred**, and since only the superpositions of :math:`\alpha \vert 000 \rangle` and 
# :math:`\alpha \vert 111 \rangle` are allowed, we can deduce that error occurred on the second qubit and we can fix this error by flipping it back. But there is a problem with this reasoning: 
# in a quantum circuit, we do not have access to the state. To know that a flip did occur,
# we have to measure the state. But this collapses the state of the qubit, rendering it useless for future calculations. Let us see how to get around this.
#
# .. note::
#     Why do we encode qubits in this way, instead of preparing many copies of the state? If the quantum state is known, we could do this,
#     but even state preparation is prone to errors! If the quantum state is not known, the no-cloning
#     theorem states it is impossible to make a copy of the state.
#
# Error detection
# ~~~~~~~~~~~~~~~~
#
# To detect whether a bit-flip error has occurred on one of the physical qubits, we perform a **parity measurement.** A parity measurement
# acts on auxiliary qubits to avoid disturbing the encoded logical state. In the case of the three-qubit repetition code, we measure in the computational
# on two auxiliary qubits after after applying some :math:`\textrm{CNOT}` gates, as shown below.
#
# **NEEDS PICTURE**
#
# The result of the measurements is known as the **syndrome**. It will tell us whether one of the qubits in :math:`\vert \bar{\psi} \rangle` was flipped and moreover,
# it can tell us which qubit was flipped. The following table shows how to interpret the syndromes.
#
# .. list-table::
#    :header-rows: 1
#    :widths: auto 20 20
#    * - Column 1
#      - Column 2
#      - Column 3
#    * - Row 1
#      - :math:`\sqrt{16}`
#      - 9
#    * - Row 2
#      - :math:`x^2`
#      - 16
#
# .. math::
#
#     \begin{tabular}{|c|c|c|}
#     \hline
#     \textbf{Error} & \textbf{Syndrome 0} & \textbf{Syndrome 1} \\ \hline
#      X_0           & 1                   & 0                  \\ \hline
#      X_1          & 1                 & 1                 \\ \hline
#      X_2          & 0                 & 1                 \\ \hline
#     \end{tabular}
#
# Let us verify this by implementing the syndrome measurement in PennyLane

def error_detection():

    qml.CNOT(wires = [0, 3])
    qml.CNOT(wires = [1, 3])
    qml.CNOT(wires = [1, 4])
    qml.CNOT(wires = [2, 4])


@qml.qnode(qml.device("default.qubit", wires = 5, shots = 1)) # A single sample flags error
def syndrome_measurement(error_wire):

    encode(alpha, beta)

    qml.PauliX(wires = error_wire) # Unwanted Pauli Operator

    error_detection()

    return qml.sample(wires = [3,4])


print("Syndrome if error on wire 0: ", syndrome_measurement(0))
print("Syndrome if error on wire 1: ", syndrome_measurement(1))
print("Syndrome if error on wire 2: ", syndrome_measurement(2))

##############################################################################
#
# The measurement outputs confirm the syndrome table.
#
# Error Correction
# ~~~~~~~~~~~~~~~~~
#
# Once a single bit-flip error is detected, correction is straightforward. Since the Pauli-X operator is its own inverse
# (i.e., :math:`X^2 = \mathbb{I}`), applying the :math:`X` operator to the erroneous qubit restores the original state, for example,
# if the syndrome measurement shows the error occurred on the second qubit, we apply 
#
# .. math::
#    
#     X_2 (X_2 \vert \bar{\psi}\rangle) = \vert \bar{\psi} \rangle.
#
# By applying the appropriate corrective operation, the repetition code effectively protects and repairs the encoded quantum information.
# The full workflow is shown in the circuit below.
#
# **NEED FIGURE WITH FULL CIRCUIT**
#
# We can use PennyLane's mid-circuit measurement features to implement the full three-qubit repetition code.

@qml.qnode(qml.device("default.qubit", wires = 5))
def error_correction(error_wire):

    encode(alpha, beta)

    qml.PauliX(wires = error_wire)

    error_detection()

    # Mid circuit measurements

    m3 = qml.measure(3)
    m4 = qml.measure(4)

    # Operations conditional on measurements

    qml.cond(m3 & ~m4 , qml.PauliX)(wires = 0)
    qml.cond(m3 & m4, qml.PauliX)(wires = 1)
    qml.cond(~m3 & m4, qml.PauliX)(wires = 2)

    return qml.density_matrix(wires = [0, 1, 2]) # qml.state not supported, but density matrices are

##############################################################################
#
# At the time of writing, PennyLane circuits with mid-circuit measurements cannot return a quantum state, so we return the density matrix instead.
# With this result, we can verify that the fidelity of the encoded state is the same as the final state after correction
# as follows

dev = qml.device("default.qubit", wires = 5)
error_correction_qnode = qml.QNode(error_correction, dev)
encoded_state = qml.math.dm_from_state_vector(encode_qnode(alpha, beta))

# Compute fidelity of final corrected state with initial encoded state

print("Fidelity if error on wire 0: ", qml.math.fidelity(encoded_state, error_correction_qnode(0)).round(2))
print("Fidelity if error on wire 1: ", qml.math.fidelity(encoded_state, error_correction_qnode(1)).round(2))
print("Fidelity if error on wire 2: ", qml.math.fidelity(encoded_state, error_correction_qnode(2)).round(2))

##############################################################################
#
# The error is corrected no matter which qubit was flipped!
#
# Operator picture and stabilizer generators
# ------------------------------------------
#
# We have worked with a simple example, but it is quite limited. Indeed, the three-qubit code only works for
# bit flip errors, but more powerful codes need more resources. For example, Shor's code, involving 9 qubits, 
# can correct more types of errors on a single logical qubit. Moreover, to avoid errors at an acceptable level,
# the industry standard is 1000 physical qubits per logical qubit. Even with a few qubits, the encoded states and protocols can become increasingly
# complex! To deal with these situations, we resort to a different 
# representation of error correction codes, known as the **operator picture.**
#
# To gain some intuition about the operator picture, let us express the three-qubit repetition code in a different way. Using the 
# identity
#
# **Hadamard and reversed CNOT identities**
#
# and the fact that :math:`HXH = Z,`, the error correction code can be expressed in the following way:
#
# **Three-qubit circuit in the Stabilizer formalism**
#
# This is the same circuit, but the controls are all now in the auxiliary qubits, while the physical qubits act as target qubits. 
# This does not seem desirable--we do not want to change the state of the physical qubits! However, let us observe that the operators
# that act on the logical qubits are :math:`Z_0 Z_1 I_2` and :math:`I_0 Z_1 Z_2,` which leave the logical codewords invariant:
# 
# .. math::
#
#     Z_0 Z_1 I_2 \vert 000 \rangle = \vert 000 \rangle, \quad Z_0 Z_1 I_2 \vert 111 \rangle = \vert 111 \rangle.
#  
# .. math::
#
#     I_0 Z_1 Z_2 \vert 000 \rangle = \vert 000 \rangle, \quad I_0 Z_1 Z_2 \vert 111 \rangle = \vert 111 \rangle.
#
# This is great news. As long as no error has occurred, the logical qubits will be left alone. Otherwise, there will be
# some operations applied on the state, but we will be able to fix them via an error correction scheme. Notably, the invariance property 
# **holds true only for the logical codewords.** For any other three-qubit basis states, at least one of these operators will have eigenvalue 
# :math:`-1`, as shown in the table below.
#
# .. math::
#
#     \begin{align*}
#
#     ZZI \vert 000 \rangle &= + 1 \vert 000 \rangle \qquad IZZ \vert 000 \rangle + 1 \vert 000 \rangle \\
#     ZZI \vert 001 \rangle &= + 1 \vert 001 \rangle \qquad IZZ \vert 001 \rangle - 1 \vert 001 \rangle \\
#     ZZI \vert 010 \rangle &= - 1 \vert 010 \rangle \qquad IZZ \vert 010 \rangle - 1 \vert 010 \rangle \\
#     ZZI \vert 011 \rangle &= - 1 \vert 011 \rangle \qquad IZZ \vert 011 \rangle + 1 \vert 011 \rangle \\
#     ZZI \vert 100 \rangle &= - 1 \vert 100 \rangle \qquad IZZ \vert 100 \rangle + 1 \vert 100 \rangle \\
#     ZZI \vert 101 \rangle &= - 1 \vert 101 \rangle \qquad IZZ \vert 101 \rangle - 1 \vert 101 \rangle \\
#     ZZI \vert 110 \rangle &= + 1 \vert 110 \rangle \qquad IZZ \vert 110 \rangle - 1 \vert 110 \rangle \\
#     ZZI \vert 111 \rangle &= + 1 \vert 111 \rangle \qquad IZZ \vert 111 \rangle + 1 \vert 111 \rangle \\
#
#     \end{align*}
#
# This gives us a new option for characterizing error correction codes. What if instead of building codewords and trying to find
# the syndrome measurement operators from them, we went the the other way round? Namely, we could start by specifying a set of operators, find the states
# that remain invariant under their action, and make these our codewords. These operators are known as **stabilizer generators.**
#
# The stabilizer formalism
# -------------------------
#
# Stabilizer generators
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The stabilizer formalism takes the operator picture representation seriously and uses it to find error correction codes starting from a 
# **stabilizer set** of **Pauli words**--tensor products of Pauli operators. A stabilizer set on :math:`n`
# qubits satisfies the following properties. 
#
# 1. It contains the identity operator :math:`I_0\otimes I_1 \times \ldots I_{n-1},` but does not contain the negative identity.
# 2. All elements of the set commute with each other. 
# 3. The matrix product of two elements in the set yields an element that is also in the set.
# 
# The set can be more succinctly specified set of generators: a minimal set of operators in the stabilizer set that can produce all 
# the other elements through pairwise multiplication. As a simple example, consider the stabilizer set
#
# .. math::
#
#     S = \left\lbrace I_0 \otimes I_1 \otimes I_2, \ Z_0 \otimes Z_1 \otimes I_2, \ Z_0 \otimes I_1 \otimes Z_2, \ I_0 \otimes Z_1 \otimes Z_2 \right\rbrace.
#
# We can check that it satisfies the defining properties 1. to 3. The most cumbersome to check is property 3, where we have to take all
# possible products of the elements and check whether the result is in :math:`S.` For example
# 
# .. math::
# 
#    (Z_0 \otimes Z_1 \otimes I_2)\cdot (I_0 \otimes Z_1 \otimes Z_2) = Z_0 \otimes I_1 \otimes Z_2, \\
#    (Z_0 \otimes Z_1 \otimes I_2 )^2 = I_0 \otimes I_1 \otimes I_2,
# 
# and so on. Note that we can obtain all the elements in :math:`S` just from :math:`Z_0 \otimes Z_1 \otimes I_2` and :math:`I_0 \otimes Z_1 \otimes Z_2.` Because
# of this property, these elements are **stabilizer generators** for :math:`S`. We write this fact as
#
# .. math::
#
#     S = \left\langle Z_0 \otimes Z_1 \otimes I_2, \ I_0 \otimes Z_1 \otimes Z_2 \right\rangle,
#
# which reads :math:`S` *is the stabilizer set generated by the elements* :math:`Z_0 \otimes Z_1 \otimes I_2` *and* :math:`I_0 \otimes Z_1 \otimes Z_2.`
# It turns out that specifying these generators is enough to completely define an error correcting code.
#
# Now that we know how stabilizer generators work, let us create a tool for later use that creates the full stabilizer set from its generators. 
#
import itertools
from pennylane import X, Y, Z
from pennylane import Identity as I

def generate_stabilizer_group(gens, num_wires):
    group = []
    init_op =I(0)
    for i in range(1,num_wires):
      init_op = init_op @ I(i)
    for bits in itertools.product([0, 1], repeat=len(gens)):
        op = init_op
        for i, bit in enumerate(bits):
            if bit:
                op = qml.prod(op, gens[i]).simplify()
        group.append(op)
    return set(group)

generators = [Z(0)@Z(1)@I(2), I(0)@Z(1)@Z(2)]
generate_stabilizer_group(generators, 3)

##############################################################################
#
# Indeed, obtain all the elements of the set by inputting the generators only. Feel free to try out this code with different generators!
#
# Defining the codespace
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# At this point, the pressing question is, given a set of stabilizer operators, how do we recover the error correction circuit? That is, 
# we need to find a way to go from the operator picture to the state picture. Let us recall that the property that inspired using stabilizer generators 
# is that they must leave the codewords invariant. For any stabilizer element :math:`S` and codeword :math:`\vert \psi \rangle`, we must
# have
# 
# .. math::
# 
#     S\vert \psi \rangle = \vert \psi \rangle. 
#
# The **codespace** is defined as the set made up of all states such that :math:`S_i \vert\psi\rangle = \vert \psi\rangle` for all stabilizer
# generators :math:`S_i.` The **codewords** can then be recovered by choosing an orthogonal basis of the codespace.
#
# With this in mind, the error correcting code can be recovered for the set :math:`\left\lbrace S_i \right\rbrace`, as shown below
#
# **FIGURE: GENERAL ERROR CORRECTION CIRCUIT FROM STABILIZERS**
#
# The stabilizer generators act as the controlled operators in the codewords and the measurement in the auxiliary wires yield unique syndromes for the
# Pauli error that the code deals with. 
#
# Logical operators
# ~~~~~~~~~~~~~~~~~~
#
# Thus far, we have defined the stabilizer generators, which correspond to the operators that implement syndrome measurements, and the 
# codewords, which are states invariant under the stabilizers and hence usable as logical qubits. One missing ingredient are gates we can implement
# on logical qubits without leaving the codespace. Such operators would act non-trivially on the codewords, so they cannot be in the
# stabilizer set, but have to preserve the codespace, meaning they **must commute** with all the stabilizer generators. In particular
# we are interested in the logical Pauli :math:`\bar{X}` and :math:`\bar{Z}` operators, defined by
#
# .. math::
#    
#     \bar{X}\vert \bar{0} \rangle = \vert {1} \rangle, \quad \bar{X}\vert \bar{1} \rangle = \vert {0} \rangle \\
#     \bar{Z}\vert \bar{0} \rangle = \vert {0} \rangle, \quad \bar{Z}\vert \bar{1} \rangle = - \vert {1} \rangle
#
# For example, in the three qubit bit flip error correcting code, the logical operators are :math:`\bar{X} = X_0 X_1 X_2` and 
# :math:`\bar{Z} = Z_0 Z_1 Z_2,` but they will not always be this simple.  In general, given a stabilizer set $S$, the logical
# operators for the code satisfy the following properties:
#
# 1. They commute with all elements in :math:`S,` so they leave the codespace invariant,
# 2. They are not in the stabilizer group,
# 3. They anticommute with each other, which means they act in a non-trivial way on the codewords.
#
# The codespace, the logical operators, and the syndrome measurements are all we need to define an error correcting code. We can even write
# a script to classify operators given a set of stabilizer generators. 
#
def classify_pauli(operator, logical_ops, generators, n_wires):

    allowed_wires = set(range(n_wires))
    operator_wires = set(operator.wires)

    assert operator_wires.issubset(allowed_wires), "Operator has wires not allowed by the code"

    operator_names = set([op.name for op in operator.decomposition()])
    allowed_operators = set(['Identity', 'PauliX', 'PauliY', 'PauliZ', 'SProd'])

    assert operator_names.issubset(allowed_operators), "Operator contains an illegal operation"

    stabilizer_group = generate_stabilizer_group(generators, n_wires)

    if operator.simplify() in stabilizer_group:
        return f"{operator} is a Stabilizer."

    if all(qml.is_commuting(operator, g) for g in generators):
        if operator in logical_ops:
            return f"{operator} is a Logical Operator."
        else:
            return f"{operator} commutes with all stabilizers — it's a Logical Operator (or a multiple of one)."

    return f"{operator} is an Error Operator (Destabilizer)."

generators = [Z(0)@Z(1)@I(2), I(0)@Z(1)@Z(2)]
logical_ops = [X(0)@X(1)@X(2), Z(0)@Z(1)@Z(2)]
print(classify_pauli(Z(0)@I(1)@Z(2), logical_ops, generators, 3))
print(classify_pauli(Y(0)@Y(1)@Y(2), logical_ops, generators, 3))
print(classify_pauli(X(0)@Y(1)@Z(2), logical_ops, generators, 3))

##############################################################################
#
# Five-qubit stabilizer code
# ---------------------------
#
# Unlike other error correcting codes, the 5-qubit code does not have a special name, but it holds a special place as the smallest
# error correcting protocol capable of correcting arbitrary Pauli Errors--unwanted applications of :math:`X`, :math:`Y`, or :math:`Z` gates.
# In this section, we will build it starting from its stabilizer generators:
#
# .. math::
#
#     S = \langle S_0, \ S_1, \ S_2, \ S_3 \rangle, 
#
# with 
#
# .. math::
# 
#    S_0 = X_0 Z_1 Z_2 X_3 I_4,\\
#    S_1 = I_0 X_1 Z_2 Z_3 X_4,\\
#    S_2 = X_0 I_1 X_2 Z_3 Z_4,\\
#    S_3 = Z_0 X_1 I_2 Z_3 X_4.
#
# The calculations are a bit cumbersome, but with some patience we can find the common :math:`+1`-eigenspace of the stabilizer generators,
# spanned by the codewords
#
# .. math::
#     
#     \begin{align*}
#     \vert \bar{0}\rangle = &\frac{1}{4}\left(\vert 00000 \rangle \vert 10010 \rangle + \vert 01001 \rangle + \vert 10100 \rangle + \vert 01010 \rangle - \vert 11011 \rangle - \vert 00110 \rangle \right.\\ 
#                            &\left. - \vert 11101 \rangle - \vert 00011\rangle - \vert 11110 \rangle - \vert 01111 \rangle - \vert 10001 \rangle - \vert 01100 \rangle - \vert 10111 \rangle + \vert 00101 \rangle \right)
#     \end{align*}
#
# .. math::
#
#     \vert \bar{1}\rangle = X\otimes X \otimes X \otimes X \otimes X \vert \bar{0}.
#  
# The logical operators bit-flip and phase-flip are for this code are :math:`\bar{X}= X^{\otimes 5}` and :math:`Z^{\otimes 5}.` With these
# defining features in mind, let us build the five-qubit stabilizer code. First, we need to prepare our logical qubit $\vert \bar{0} \rangle,$
# which can be done using the circuit below. 
#
# **INSERT CIRCUIT IMAGE**
#
# This is straightforward to implement in PennyLane.
#

def five_qubit_encode(alpha, beta):

    qml.StatePrep([alpha, beta], wires = 4)
    qml.Hadamard(wires = 0)
    qml.S(wires = 0)
    qml.CZ(wires = [0,1])
    qml.CZ(wires = [0,3])
    qml.CY(wires = [0,4])
    qml.Hadamard(wires = 1)
    qml.CZ(wires = [1,2])
    qml.CZ(wires = [1,3])
    qml.CNOT(wires = [1,4])
    qml.Hadamard(wires = 2)
    qml.CZ(wires = [2,0])
    qml.CZ(wires = [2,1])
    qml.CNOT(wires = [2,4])
    qml.Hadamard(wires = 3)
    qml.S(wires = 3)
    qml.CZ(wires = [3,0])
    qml.CZ(wires = [3,2])
    qml.CY(wires = [3,4])

##############################################################################
#
# Having encoded the logical state, we can use the stabilizers to measure obtain the syndrome table like we did with the three-qubit code.

dev = qml.device("default.qubit", wires = 9, shots = 1)

stabilizers = [X(0)@Z(1)@Z(2)@X(3)@I(4), I(0)@X(1)@Z(2)@Z(3)@X(4),
               X(0)@I(1)@X(2)@Z(3)@Z(4), Z(0)@X(1)@I(2)@X(3)@Z(4)]

@qml.qnode(dev)
def five_qubit_code(alpha, beta, error_type, error_wire):

    five_qubit_encode(alpha, beta)

    if error_type == 'X':
        qml.PauliX(wires = error_wire)

    elif error_type == 'Y':
        qml.PauliY(wires = error_wire)

    elif error_type == 'Z':
        qml.PauliZ(wires = error_wire)

    for wire in range(5,9):
        qml.Hadamard(wires = wire)

    for i in range(len(stabilizers)):

        qml.ctrl(stabilizers[i], control = [i + 5])

    for wire in range(5,9):
        qml.Hadamard(wires = wire)

    return qml.sample(wires = range(5,9))

for wire in (0, 1, 2, 3, 4):

    for error in ('X', 'Y', 'Z'):

        print(f"{error} {wire}", five_qubit_code(1/2, np.sqrt(3)/2, error, wire))
##############################################################################
#
# The syndrome table is printed, and with we can apply the necessary operators to fix the corresponding Pauli errors. The script above is straightforward
# to generalize to any valid set of stabilizers. 
#
#
# References
# -----------
#
# About the author
# -----------------
# 
# 
