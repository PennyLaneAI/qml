r"""How to use Qualtran with PennyLane
======================================

The PennyLane-Qualtran integration allows you to easily convert Qualtran bloqs to PennyLane
operators and vice-versa. Simply wrap any Qualtran bloq using ``qml.FromBloq``, and you can
put it into a quantum function, enabling key PennyLane features such as simulatibility. 
Alternatively, convert a PennyLane object using ``qml.to_bloq`` and access Qualtran features
such as call graphs.
"""

######################################################################
# How to use :class:`~.pennylane.io.FromBloq`
# -----------------------------
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
# We can see that this is true by checking its matrix.

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

# You can also pass in QFuncs. 

def circ():
    qml.X(0)
    qml.X(1)

qfunc_as_bloq = qml.to_bloq(circ)
print(qfunc_as_bloq)

######################################################################
# The Mappening
# -------------
#
# Here, the PauliX operator was directly translated to its Qualtran equivalent, ``XGate()``.
# PennyLane always tries its best to translate operators to a Qualtran-native object. However,
# implementations of the same abstract idea can differ between Qualtran and PennyLane. In ambiguous
# cases, we provide a default mapping, but if that is unsatisfactory, we offer two solutions:
# (1) explicitly map to the specific Qualtran Bloq desired and (2) maintain the original
# PennyLane implementation by setting  ``map_ops`` to ``False``. For example, the implementations
# of Quantum Phase Estimation (QPE) in PennyLane and Qualtran differ. By default,
# we map the PennyLane QPE to its closest equivalent in Qualtran.

from qualtran.drawing import draw_musical_score,  get_musical_score_data

op = qml.QuantumPhaseEstimation(unitary=qml.RY(phi=0.3, wires=[0]), estimation_wires=[1, 2, 3])
qpe_bloq = qml.to_bloq(op)
fig, ax = draw_musical_score(get_musical_score_data(qpe_bloq.decompose_bloq()))
fig.tight_layout()

######################################################################
# We mapped it to Qualtran's ``TextbookQPE`` where ``ctrl_state_prep`` is Qualtran's 
# ``RectangularWindowState``. If we wanted to use ``LPResourceState``,
# rather than ``RectangularWindowState``, we can simply pass in a custom map to ``to_bloq``.

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
# The wrappening
# --------------
#
# These two implementations, while similar on a high level, are not exactly the same as the
# implementation in PennyLane. To make our ``qpe_bloq`` exactly the same as its PennyLane
# original, set ``map_ops`` to ``False``. This wraps the ``qpe_bloq`` as a ``ToBloq`` Bloq,
# whose information is based on that of the original PennyLane object.

qpe_bloq = qml.to_bloq(op, map_ops=False)
fig, ax = draw_musical_score(get_musical_score_data(qpe_bloq.decompose_bloq()))
fig.tight_layout()

######################################################################
# Here, we can clearly see the differences between the two methods. When we map, qubit allocation
# is handled with the use of Qualtran's bookkeeping bloq `Allocate`. Since there is no PennyLane
# equivalent, in the latter version, the 3 wires are explicitly drawn and handled, which leads to
# a more PennyLane-like visualization. When ``to_bloq`` encounters a Bloq that doesn't have a
# corresponding PennyLane equivalent, the Bloq is automatically wrapped as a ``ToBloq`` Bloq.

block_encoding = qml.Hadamard(wires=0)
phase_shifts = [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]
op_as_to_bloq = qml.to_bloq(qml.QSVT(block_encoding, phase_shifts))

# Note that while ``ToBloq``'s decomposition always maintains that of the wrapped PennyLane object,
# its call graph does not always match the decomposition. When you use the resource counting
# methods on a ``ToBloq``, you may get resources more optimal than what the original PennyLane
# decomposition might have prescribed.

######################################################################
# Conclusion
# ----------
# In this demo, we showed how to convert between PennyLane objects and Qualtran bloqs.
#
# About the author
# ----------------
#
