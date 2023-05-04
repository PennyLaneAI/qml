r"""Quantum compilation of circuits
===============================

We can find equivalent representations of quantum circuits using different sets of gates that result
in the same output. We are typically interested in finding the most convenient one for a specific
purpose. For example, in the compilation of quantum circuits, we are interested in finding the one
that will incur the least amount of errors in a specific hardware. This usually implies decomposing
the quantum gates in terms of the native ones of the quantum device, adapting the operations to the
hardrware’s topology, combining them together to reduce the execution time, etc.

In PennyLane, we can apply
```transforms`` <https://docs.pennylane.ai/en/stable/code/qml_transforms.html>`__ to our quantum
functions in order to obtain equivalent representations that may be more convenient for our task. In
this tutorial, we show how to implement the most common ones for circuit compilation.

Circuit transforms
==================

When we run quantum algorithms, it is typically in our best interest that the resulting circuits is
as shallow as possible, specially in the current quantum devices. However, it is often the case that
the they have multiple operations that can be combined together. Here, we introduce three simple
circuit transforms that can be combined together to obtain simpler quantum circuits. The transforms
are based on very basic circuit equivalences:

.. figure:: ../demonstrations/circuit_compilation/circuit_transforms.png
    :align: center
    :width: 90%

In order to illustrate the combined effect of these transforms, let us consider the following
circuit.
"""

import pennylane as qml

dev = qml.device("default.qubit", wires=3)


def circuit(angles):
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.RX(angles[0], 0)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 1])
    qml.RX(angles[2], wires=0)
    qml.RZ(angles[1], wires=2)
    qml.CNOT(wires=[2, 1])
    qml.RZ(-angles[1], wires=2)
    qml.CNOT(wires=[1, 0])
    qml.Hadamard(wires=1)
    qml.CY(wires=[1, 2])
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(wires=0))


angles = [0.2, 0.3, 0.4]
qnode = qml.QNode(circuit, dev)
qml.draw_mpl(qnode, decimals=1)(angles)

######################################################################
# Given an arbitrary quantum circuit, it is some times hard to clearly understand what is really
# happening. To obtain a better picture, we can shuffle the commuting operations to better distinguish
# between groups of single-qubit and two-qubit gates. We can easily do so by applying the
# ```commute_controlled`` <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.commute_controlled.html>`__
# transform.
#

commuted_circuit = qml.transforms.commute_controlled()(circuit)

qnode = qml.QNode(commuted_circuit, dev)
qml.draw_mpl(qnode, decimals=1)(angles)

######################################################################
# With this rearrangement, we can clearly identify a few operations that can be merged together.
# First, the two consecutive CNOT gates from the third to the second qubits will cancel each other
# out. We can remove these with the
# ```cancel_inverses`` <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.cancel_inverses.html>`__
# transform, which removes consecutive inverse operations. Then, we can combine the two ``RX``
# rotations in the first qubit into a single one with the sum of the angles. Finally, the two ``RZ``
# rotations in the third qubit will also cancel each other. We can combine these rotations with the
# ```merge_rotations`` <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.merge_rotations.html>`__
# transform.
#
# Let us see the result of applying both transforms to the rearranged circuit. We start with the
# ``cancel_inverses`` one.
#

cancelled_circuit = qml.transforms.cancel_inverses(commuted_circuit)

qnode = qml.QNode(cancelled_circuit, dev)
qml.draw_mpl(qnode, decimals=1)(angles)

######################################################################
# Now we combine the rotations together.
#

merged_circuit = qml.transforms.merge_rotations()(cancelled_circuit)

qnode = qml.QNode(merged_circuit, dev)
qml.draw_mpl(qnode, decimals=1)(angles)

######################################################################
# Combining these simple circuit transforms, we have reduced the complexity of our original circuit.
# We can apply them at once when we define our quantum function using their decorator forms (beware
# the reverse order!).
#


@qml.qnode(dev)
@qml.transforms.merge_rotations()
@qml.transforms.cancel_inverses
@qml.transforms.commute_controlled()
def q_fun(angles):
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.RX(angles[0], 0)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 1])
    qml.RX(angles[2], wires=0)
    qml.RZ(angles[1], wires=2)
    qml.CNOT(wires=[2, 1])
    qml.RZ(-angles[1], wires=2)
    qml.CNOT(wires=[1, 0])
    qml.Hadamard(wires=1)
    qml.CY(wires=[1, 2])
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(wires=0))


qml.draw_mpl(q_fun, decimals=1)(angles)

######################################################################
# Circuit compilation
# ===================
#
# Rearranging and combining operations is an essential part of circuit compilation. Indeed, it is
# usually performed repeatedly as the compiler does multiple *passess* over the circuit.
#
# We can apply all the transforms introduced above with the
# ```compile`` <https://docs.pennylane.ai/en/stable/code/api/pennylane.compile.html>`__ function,
# which yields the same final circuit.
#

compiled_circuit = qml.compile()(circuit)

qnode = qml.QNode(compiled_circuit, dev)
qml.draw_mpl(qnode, decimals=1)(angles)

######################################################################
# In the resulting circuit, we can identify further operations to be combined, such as the consecutive
# CNOT gates applied from the second to the first qubit. To do so, we would simply need to apply the
# same transforms again in a second pass of the compiler. We can adjust the desired number of passes
# with ``num_passes``.
#
# Let us see the resulting circuit with two passes.
#

compiled_circuit = qml.compile(num_passes=2)(circuit)

qnode = qml.QNode(compiled_circuit, dev)
qml.draw_mpl(qnode, decimals=1)(angles)

######################################################################
# This can be further simplified with an additional pass of the compiler. In this case, we also define
# explicitly the transforms to be applied by the compiler with the ``pipeline`` argument. This allows
# us to control the transforms, their parameters, and the order in which they are applied.
#

compiled_circuit = qml.compile(
    pipeline=[
        qml.transforms.commute_controlled(direction="left"),
        qml.transforms.merge_rotations(atol=1e-6),
        qml.transforms.cancel_inverses,  # Cancel inverses after rotations
    ],
    num_passes=3,
)(circuit)

qnode = qml.QNode(compiled_circuit, dev)
qml.draw_mpl(qnode, decimals=1)(angles)

######################################################################
# Notice how the ``RX`` gate in the first qubit has now been pushed towards the left (it defaults to
# the right), as we have specified in the ``commute_controlled`` transform.
#
# Finally, we can specify a finite basis of gates to describe the circuit providing a ``basis_set``.
# For example, we can choose to only use single-qubit rotations and CNOT operations. The compiler will
# first decompose the gates in terms of our basis and, then, apply the transforms.
#

compiled_circuit = qml.compile(basis_set=["CNOT", "RX", "RY", "RZ"], num_passes=2)(circuit)

qnode = qml.QNode(compiled_circuit, dev)
qml.draw_mpl(qnode, decimals=1)(angles)

######################################################################
# In this tutorial, we have learned the basic principles of circuit transformations and how they apply
# to the compilation of quantum circuits.
#
######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/borja_requena.txt
