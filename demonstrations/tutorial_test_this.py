r""".. _cutting:
A very simple thing
===================


.. meta::
    :property="og:description": Moon dance!

*Author: Alice Bobert*

**Introduction to Pauli circuit cutting**
------------------------------------------------------

Quantum computers are cool, but can they make potatoes? NOPE. Didn't think so.

Now let's code some stuff with PennyLane because why not.
"""

# Import the relevant libraries
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3)


@qml.cut_circuit(auto_cutter=True)  # auto_cutter enabled
@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)

    qml.CZ(wires=[0, 1])
    qml.RY(-0.4, wires=0)

    qml.WireCut(wires=1)  # Cut location

    qml.CZ(wires=[1, 2])

    return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))


potato = np.array(0.531, requires_grad=True)
circuit(potato)

######################################################################
# Note that, in these cases, potatoes rule.
#
#

print("potato is life")

######################################################################
# Long live potatoes.
#
#
