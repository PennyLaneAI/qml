r"""How to use Qualtran with PennyLane
======================================

As interest has grown in quantum computing, so have the number of tools available that help us 
express and analyze quantum circuits. One such tool is Qualtran: a set of abstractions for
representing quantum programs and a library of quantum algorithms expressed in that language.

With the PennyLane-Qualtran integration, we can leverage Qualtran's expansive library of algorithms
as if they were PennyLane operators. We go from abstract, high-level subroutines to well-defined 
operators that can run directly in our quantum circuits.

In the other direction, we can convert PennyLane circuits and operators to Qualtran Bloqs just as
easily, allowing us to use Qualtran's tools and abstractions for expressing and reasoning about
quantum algorithms, programs, and subroutines.
"""

######################################################################
# From Qualtran to Pennylane
# --------------------------
#
# Converting a Qualtran bloq to a PennyLane operator is easy! Just use ``qml.FromBloq``.
# 

import pennylane as qml
from qualtran.bloqs.basic_gates import XGate

bloq_as_op = qml.FromBloq(XGate(), wires=0)
print(bloq_as_op)

######################################################################
# The output is a :class:`~.pennylane.io.FromBloq` instance. In this case, we wrapped Qualtran's
# ``XGate``, which is equivalent to ``qml.X``.
#
# We can verify this by checking its matrix.

print(bloq_as_op.matrix())

######################################################################
# We can convert high-level abstract Bloqs as well. Here we convert a simple ``TextbookQPE`` bloq and
# verify that its decomposition is as expected.

from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE
from qualtran.bloqs.basic_gates import ZPowGate

textbook_qpe = TextbookQPE(ZPowGate(exponent=2 * 0.234), RectangularWindowState(3))

print(qml.FromBloq(textbook_qpe, wires=range(textbook_qpe.signature.n_qubits())).decomposition())

######################################################################
# If you're not sure about the wires to give to ``qml.FromBloq``, you can use the ``qml.bloq_registers``
# helper function. This function creates registers based on the signature of the qualtran Bloq.
# In this instance, we see that it has two registers: a "q" register with 1 qubit and a "qpe_reg" 
# register of 3 qubits.

qml.bloq_register(textbook_qpe)

# In this case, the naming follows the default register names that Qualtran has assigned to the QPE
# Bloq. This information is useful if you want to know, for example, what qubits to measure.

######################################################################
# From PennyLane to Qualtran
# --------------------------
#
# In this section, we discuss how to convert PennyLane objects to Qualtran Bloqs. Similar to
# ``qml.FromBloq``, we use ``qml.to_bloq`` to handle the conversion.

op_as_bloq = qml.to_bloq(qml.X(0))
print(op_as_bloq)

# Unlike `qml.FromBloq`, notice that instead of being wrapped as a `ToBloq` Bloq, the PauliX 
# operator was directly translated to its Qualtran equivalent, ``XGate()``. For `qml.to_bloq`,
# PennyLane always tries its best to translate PennyLane objects to Qualtran-native objects. This
# makes certain Qualtran functionalities, such as gate counting and generalizers, work more
# seamlessly.
# 
# In the following example, we pass a quantum function to `qml.to_bloq`. Here, since the quantum
# function does not have a direct Qualtran equivalent, the circuit is wrapped as a `ToBloq` Bloq.
# We can check its decomposition to see that it follows the circuit description exactly.

from qualtran.drawing import show_bloq

def circ():
    qml.X(0)
    qml.X(1)

qfunc_as_bloq = qml.to_bloq(circ)
print(type(qfunc_as_bloq))
show_bloq(qfunc_as_bloq.decompose_bloq())

######################################################################
# Advanced details: Mapping
# -------------------------
#
# Not all PennyLane operators are as straightforward to map as the PauliX operator. For example, 
# PennyLane's Quantum Phase Estimation could be mapped to a variety of Qualtran Bloqs. In cases
# where the mapping is ambiguous, PennyLane provides what we call a smart default:

from qualtran.drawing import draw_musical_score,  get_musical_score_data

op = qml.QuantumPhaseEstimation(unitary=qml.RY(phi=0.3, wires=[0]), estimation_wires=[1, 2, 3])
qpe_bloq = qml.to_bloq(op)
fig, ax = draw_musical_score(get_musical_score_data(qpe_bloq.decompose_bloq()))
fig.tight_layout()

######################################################################
# Here, the smart default is Qualtran's ``TextbookQPE`` where ``ctrl_state_prep`` is Qualtran's 
# ``RectangularWindowState``. If we wanted to use ``LPResourceState``, rather than 
# ``RectangularWindowState``, we can override the smart default and pass in a custom map.

from qualtran.bloqs.phase_estimation import LPResourceState
from qualtran.bloqs.phase_estimation.text_book_qpe import TextbookQPE

custom_map = {
    op: TextbookQPE(
        unitary=qml.to_bloq(qml.RY(phi=0.3, wires=[0])), 
        ctrl_state_prep=LPResourceState(3)
    )
}

qpe_bloq = qml.to_bloq(op, custom_mapping=custom_map)
fig, ax = draw_musical_score(get_musical_score_data(qpe_bloq.decompose_bloq()))
fig.tight_layout()

######################################################################
# We see that ``RectangularWindowState`` has been switched out for the ``LPResourceState`` we
# defined in the custom map. 
#
# Advanced details: Wrapping
# --------------------------
#
# These two implementations, while similar on a high level, are not exactly the same as the
# implementation in PennyLane. To make our ``qpe_bloq`` exactly the same as its PennyLane
# original, set ``map_ops`` to ``False``. This wraps the ``qpe_bloq`` as a ``ToBloq`` Bloq,
# whose information is based on that of the original PennyLane object.

wrapped_qpe_bloq = qml.to_bloq(op, map_ops=False)
fig, ax = draw_musical_score(get_musical_score_data(wrapped_qpe_bloq.decompose_bloq()))
fig.tight_layout()

######################################################################
# Here, we can clearly see the differences between the two methods. When we map, qubit allocation
# is handled with the use of Qualtran's bookkeeping bloq `Allocate`. Since there is no PennyLane
# equivalent, in the latter version, the 3 wires are explicitly drawn and handled, which leads to
# a more PennyLane-like visualization. The call graphs and resource counts will differ between
# the two methods as well.

from qualtran.drawing import show_counts_sigma

_, mapped_sigma = qpe_bloq.call_graph()
_, wrapped_sigma = wrapped_qpe_bloq.call_graph()

show_counts_sigma(mapped_sigma)
show_counts_sigma(wrapped_sigma)

# Note that while ``ToBloq``'s decomposition always maintains that of the wrapped PennyLane object,
# its call graph may not always match the decomposition. When you use the resource counting
# methods on a ``ToBloq``, you may get resources more optimal than what the original PennyLane
# decomposition might have prescribed.

######################################################################
# Further resources
# -----------------
# 
#
# About the author
# ----------------
#
