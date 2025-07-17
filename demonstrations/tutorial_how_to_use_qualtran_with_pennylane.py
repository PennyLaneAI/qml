r"""How to use Qualtran with PennyLane
======================================

As interest has grown in quantum computing, so have the number of tools available that help us 
express and analyze quantum circuits. One such tool is Qualtran: a set of abstractions for
representing quantum programs and a library of quantum algorithms expressed in that language.

With the PennyLane-Qualtran integration, we can leverage Qualtran's expansive library of algorithms
as if they were PennyLane operators. We go from abstract, high-level subroutines to well-defined 
operators that can run directly in our quantum circuits.

In the other direction, we can convert PennyLane circuits and operators to Qualtran Bloqs just as
easily. This allows us to use Qualtran's tools and abstractions for expressing and reasoning about
quantum algorithms, programs, and subroutines.
"""

######################################################################
# From Qualtran to Pennylane
# --------------------------
#
# With a little bit of work, you can be using Qualtran Bloqs in your PennyLane circuits in no time!
# You only need one class:
#
# - :class:`~pennylane.io.FromBloq`: wraps an entire Qualtran ``Bloq`` as a 
#   PennyLane operator. It faithfully converts any Bloq, decomposition and all, into a
#   operator you can treat like you would any other.
#
# Let's see how it works!


import pennylane as qml
from qualtran.bloqs.basic_gates import XGate

bloq_as_op = qml.FromBloq(XGate(), wires=0)
print(bloq_as_op)

######################################################################
# .. note ::
#
#    Since Qualtran Bloqs don't know what wires to act on, we need to provide that information to `FromBloq`
#    accordingly. If you're unsure about what wires to provide, you can use the ``qml.bloq_registers``
#    helper function. This function creates registers based on the signature of the qualtran Bloq.
#
#   .. code-block:: python
#        
#       qml.bloq_registers(XGate())
#
#   This will create registers with with the register names in accordance to the Bloq's signature. 
#   Here, we got just one "q" register with a single qubit, which is what we expected for the `XGate`.

######################################################################
# In this simple example, we wrapped Qualtran's `XGate` using `FromBloq``. We can see that
# the output is a :class:`~.pennylane.io.FromBloq` instance, whose properties would be the same
# PennyLane's PauliX operator.
#
# Let's verify this by putting it into a circuit and executing it.

dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit():
    qml.FromBloq(XGate(), wires=[0])
    return qml.state()

######################################################################
# Wow! Like magic, we can use Qualtran's `XGate`` just like we would use the PauliX operator.
# But wait, there's more! We can convert high-level abstract Bloqs as well. Here we convert a 
# simple ``TextbookQPE`` bloq and verify that its decomposition is as expected.

from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE
from qualtran.bloqs.basic_gates import ZPowGate

textbook_qpe = TextbookQPE(ZPowGate(exponent=2 * 0.234), RectangularWindowState(3))

print(qml.FromBloq(textbook_qpe, wires=range(textbook_qpe.signature.n_qubits())).decomposition())

######################################################################
# Amazing! The decomposition is exactly what we expected. It's exactly like using a PennyLane
# operator, except the underlying decomposition is what Qualtran has defined. Neat!
######################################################################
# From PennyLane to Qualtran
# --------------------------
#
# Now, we'll show you how to convert PennyLane objects to Qualtran Bloqs.
#
# Sometimes, there are so many Qualtran Bloqs to choose, it's hard to decide what PennyLane
# operator translates to what Qualtran Bloq. You might not even necessarily want to translate a
# PennyLane operator directly to a Qualtran Bloq. Don't worry, we've got you covered with 3
# flexible options. We'll introduce them here but rest assured, we will cover each option in
# great detail:
#
# - Wrapping: Think of this as the opposite of `FromBloq`. It faithfully converts any operator or
#   Qfunc, decompositions and all, into a Bloq. The output will be a `ToBloq` instance.
#
# - Smart defaults: In this option, you let PennyLane choose what Qualtran Bloq to translate to.
#   If an option exists, we'll give you a Qualtran Bloq that is highly similar to the PennyLane
#   operator. In the case there isn't a smart default, we fallback to the wrapping option.
#
# - Custom mapping: What if you don't like the smart default we've provided? Don't worry, you can
#   custom define what Bloq you want your operator to map to. This is great if you want to really
#   refine the finer details of the Bloqs you want to analyze using Qualtran.
#
# Holding all these options together is our trusty function :func:`~pennylane.io.to_bloq`. In the
# following sections, we'll explore how we can wield this powerful function to get all the
# functionality introduced above.
#
######################################################################
# Smart defaults and custom mapping
# ---------------------------------
# By default, `qml.to_bloq` tries its best to translate 
# PennyLane objects to Qualtran-native objects. This is done through a combination of smart 
# defaults and direct mappings. This makes certain Qualtran
# functionalities, such as gate counting and generalizers, work more seamlessly.

op_as_bloq = qml.to_bloq(qml.X(0), map_ops=True, custom_map=None) # `map_ops` is `True` by default
print(op_as_bloq)

######################################################################
# .. note ::
#
#    When a quantum function or operator does not have a mapping - a direct Qualtran equivalent 
#    or smart default - the circuit is wrapped as a `ToBloq` Bloq.
#
#   .. code-block:: python
#        
#       from qualtran.drawing import show_bloq
#
#       def circ():
#           qml.X(0)
#           qml.X(1)
#
#       qfunc_as_bloq = qml.to_bloq(circ)
#       print(type(qfunc_as_bloq))
#       show_bloq(qfunc_as_bloq.decompose_bloq())
#
#
# Not all PennyLane operators are as straightforward to map as the PauliX operator. For example, 
# PennyLane's Quantum Phase Estimation could be mapped to a variety of Qualtran Bloqs. In cases
# where the mapping is ambiguous, we get the smart default:

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
