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
Is there a unified way to understand these codes? Indeed, The **stabilizer formalism** provides such a framework for
a large class of quantum error correction codes. The so-called
**stabilizer codes**, such as the repetition, Shor, Steane, and surface codes, 
all fall under this formalism. Other codes, like the GKP codes used by Xanadu, lie outside of it.

In this demo, we will introduce the stabilizer formalism using bottom-up approach. We construct and
some well-known codes using the quantum circuit formalism and then derive their **stabilizer generators,** from which
the code can be reconstructed. This enables the construction of a wide range of error correction codes 
directly from their stabilizer generators. 

A toy example: the repetition code
-----------------------------------

To start with, we will explore the general structure of error correction codes using a simple example: the **three-qubit repetition code.** 
We will introduce this code as a quantum circuit with definite steps to gain some intuition on how it corrects errors on qubits. 
The quantum circuit representation is known as the **state picture**. In this formalism, error correction codes follow a simple structure:

- Qubit encoding
- Error detection
- Error correction

Qubit encoding
~~~~~~~~~~~~~~~

The first step in an error correction code is **encoding** one abstract or **logical qubit** into a set of many on-device **physical qubits.** 
The rationale is that, if some external factor changes the state of one of the qubits, we will still have an idea
of what the original logical qubit was by looking at the rest of the qubits. For example, in the three-qubit repetition code, the logical basis-state 
qubits, or **logical codewords**, :math:`\vert \bar{0}\rangle` and :math:`\vert \bar{1}\rangle` are encoded into three physical qubits via

.. math::

    \vert \bar{0} \rangle \mapsto \vert 000 \rangle, \quad \vert \bar{1} \rangle \mapsto \vert 111 \rangle.

A general qubit :math:`\vert \bar{\psi}\rangle = \alpha \vert \bar{0}\rangle + \beta \vert \bar{1}\rangle` is then encoded as

.. math::

    \alpha \vert \bar{0}\rangle + \beta \vert \bar{1}\rangle \mapsto \alpha \vert 000 \rangle + \beta \vert \bar{111}\rangle.

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
# :math:`\alpha \vert 111 \rangle` are allowed, be can fix this error by flipping the qubit back. The problem here is that, to know that this happened,
# we have to measure the state. This collapses the wave function, rendering it useless for future calculations. Let us see how to get around this.
#
# .. note::
#     Why do we encode qubits in this way, instead of preparing many copies of the state? If the quantum state is known, we could do this,
#     but even state preparation is prone to errors! If the quantum state is not known, the no-cloning
#     theorem states it is impossible to make a copy of the state.
#
# Error detection
# ~~~~~~~~~~~~~~~~
#
# To detect whether a bit-flip error has occurred on one of the physical qubits, we perform a **syndrome measurement.** A syndrome measurement
# acts on auxiliary qubits to avoid disturbing the encoded logical state. In the case of the three-qubit repetition code, we measure in the computational
# on two auxiliary qubits after after applying some :math:`\textrm{CNOT}` gates, as shown below
#
# **NEEDS PICTURE**
#
# The results of the measurements will will tell us whether one of the qubits in :math:`\vert \bar{\psi} \rangle` was flipped and moreover,
# they can tell us which qubit was flipped. The following table shows how to interpret the results of the syndrome measurements.
#
# **INSERT TABLE**
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
# Unfortunately, circuits with mid-circuit measurements cannot return a quantum state, so return the density matrix instead.
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
# Operator picture and stabilizers
# ---------------------------------
#
# We have worked with a simple example, but it is quite limited. Indeed, the three-qubit code only works for
# bit flip errors, but more powerful codes need more resources. For example, Shor's code, involving 9 qubits, 
# can correct more types of errors on a single logical qubit. Moreover, to avoid errors at an acceptable level,
# the industry standard is 1000 physical qubits per logical qubit. Even with a few qubits, the encoded states and protocols can become increasingly
# complex! To deal with these situations, we resort to a different 
# representation of error correction codes, known as the operator picture.
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
# some operations done on the state, but we'll still be able to fix them via an error correction scheme. Notably, the invariance property 
# **only holds true for the logical codewords.** For any other three-qubit basis states, at least one of these operators will have eigen
# value :math:`-1`, as shown in the table below.
#
# **INSERT TABLE**
#
# 
#
#
#
# References
# -----------
#
# About the author
# -----------------
# 
# 
