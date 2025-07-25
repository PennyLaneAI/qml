r"""How to use Qualtran with PennyLane
======================================

PennyLane can now combine the best features of both PennyLane and Qualtran, using our latest
PennyLane-Qualtran integration.

This integration allows you to:

* **Use Qualtran's subroutines in PennyLane:** Seamlessly incorporate Qualtran's quantum 
    subroutines, known as ``bloqs``, directly in your PennyLane simulations and workflows.
* **Analyze PennyLane circuits with Qualtran:** Leverage Qualtran's advanced analysis tools to
    estimate the computational resource costs of your PennyLane circuits.

In this demo, we want to show you how easy it is to use PennyLane and Qualtran together.
We'll start by focusing on the first key capability: embedding a Qualtran ``Bloq`` in a
PennyLane ``QNode``.
"""

######################################################################
# From Qualtran to PennyLane
# --------------------------
#
# For those unfamiliar, Qualtran (quantum algorithms translator) is a set of abstractions for 
# representing quantum programs and a library of quantum algorithms (Bloqs) expressed in that 
# language.
#
# With a little bit of work, you can be using Qualtran Bloqs in your PennyLane circuits in no time!
# You only need one class: :class:`~pennylane.FromBloq`. It wraps an entire Qualtran ``Bloq`` as a 
# PennyLane operator. It faithfully converts any Bloq, including its decomposition, into a operator
# you can treat like you would any other.
#
# Let's see how it works!


import pennylane as qml
from qualtran.bloqs.basic_gates import XGate

bloq_as_op = qml.FromBloq(XGate(), wires=0)
print(bloq_as_op)

######################################################################
# In this simple example, we wrapped Qualtran's `XGate` using `FromBloq``. We can see that
# the output is a :class:`~.pennylane.io.FromBloq` instance, whose properties would be the same
# PennyLane's PauliX operator.
#
# Since Qualtran Bloqs don't know what wires to act on, we need to provide that information to 
# `FromBloq` accordingly. If you're unsure about what wires to provide, you can use the 
# ``qml.bloq_registers`` helper function. This function creates registers based on the signature 
# of the qualtran Bloq.

print(qml.bloq_registers(XGate()))

######################################################################
# This will create registers with with the register names in accordance to the Bloq's signature. 
# Here, we got just one "q" register with a single qubit, which is what we expected for the `XGate`.
#
# Now, let's verify that `XGate` performs as expected in a PennyLane circuit.

dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit():
    qml.FromBloq(XGate(), wires=[0])
    return qml.state()

print(circuit())
######################################################################
# Wow! Like magic, we can use Qualtran's `XGate`` just like we would use the PauliX operator.
# But wait, there's more! We can convert high-level abstract Bloqs as well. Here, we
# defined some `QubitizationQPE` Bloq in Qualtran. We first do some analysis in Qualtran
# and subsequently hand it off to PennyLane.

import numpy as np
from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
    get_walk_operator_for_hubbard_model,
)
from qualtran.bloqs.phase_estimation import LPResourceState, QubitizationQPE

x_dim, y_dim, t = 2, 2, 2
u = 4 * t
walk = get_walk_operator_for_hubbard_model(x_dim, y_dim, t, u)

algo_eps = t / 100
N = x_dim * y_dim * 2
qlambda = 2 * N * t + (N * u) // 2
qpe_eps = algo_eps / (qlambda * np.sqrt(2))
qubitization_qpe = QubitizationQPE(
    walk, LPResourceState.from_standard_deviation_eps(qpe_eps)
)

# For drawing & analysis
from qualtran.drawing import show_counts_sigma
from qualtran.resource_counting.generalizers import ignore_split_join

q_qpe_g, q_qpe_sigma = qubitization_qpe.call_graph(max_depth=1, generalizer=ignore_split_join)
#show_call_graph(q_qpe_g)
show_counts_sigma(q_qpe_sigma)

n_qubits = qubitization_qpe.signature.n_qubits()
print(n_qubits) # Since the # of qubits is a bit high, we won't run it on a simulator
print(qml.FromBloq(qubitization_qpe, wires=range(n_qubits)).decomposition())
######################################################################
# Amazing! The decomposition is exactly what we expected. It's exactly like using a PennyLane
# operator, except the underlying decomposition is what Qualtran has defined. Neat!
#
# From PennyLane to Qualtran
# --------------------------
#
# Now, we'll show you how to convert PennyLane objects to Qualtran Bloqs. For brevity, we'll 
# mainly cover how operators get converted to Bloqs in our examples, but quantum functions 
# work just the same.
#
# There are three main options for the conversion. We'll briefly introduce them here but go into
# greater detail in subsequent sections:
#
# - Smart defaults: In this option, you let PennyLane choose what Qualtran Bloq to translate to.
#   If an option exists, we'll give you a Qualtran Bloq that is highly similar to the PennyLane
#   operator.
#
# - Custom mapping: What if you don't like the smart default we've provided? Don't worry, you can
#   custom define what Bloq you want your operator to map to. This is great if you want to really
#   refine the finer details of the Bloqs you want to analyze using Qualtran.
#
# - Wrapping: Think of this as an analogue of `FromBloq`. It faithfully converts any operator or
#   Qfunc, decompositions and all, into a Bloq. The output is a :class:`~pennylane.io.ToBloq` instance.
#
# Holding all these options together is our trusty function :func:`~pennylane.to_bloq`. In the
# following sections, we'll explore how we can wield this powerful function to get all the
# functionality introduced above.
#
######################################################################
# Smart defaults
# --------------
# By default, `qml.to_bloq` tries its best to translate 
# PennyLane objects to Qualtran-native objects. This is done through a combination of smart 
# defaults and direct mappings. This makes certain Qualtran
# functionalities, such as gate counting and generalizers, work more seamlessly.
# Here, PennyLane's `PauliX` operator is mapped directly to Qualtran's `XGate`.

op_as_bloq = qml.to_bloq(qml.X(0))
print(op_as_bloq)

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
# ``RectangularWindowState``. But what if we wanted to use a different Bloq for our 
# `ctrl_state_prep`? In this case, we turn to custom mappings.
#  
# Custom mapping
# --------------
# To use ``LPResourceState``, rather than  ``RectangularWindowState``, we can override the smart
# default by passing in a custom map.

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
# When a quantum function or operator does not have a mapping - a direct Qualtran equivalent 
# or smart default - the circuit is wrapped as a `ToBloq` Bloq

def circ():
    qml.X(0)
    qml.X(1)

qfunc_as_bloq = qml.to_bloq(circ)
print(type(qfunc_as_bloq))

######################################################################
# Wrapping
# --------
#
# Functionally, wrapping a quantum function or operator as a `ToBloq` is similar to wrapping a Bloq
# as a `FromBloq`. A wrapped operator or Qfunc now acts like a Bloq, which means it can be analyzed
# using the language of Qualtran to simulate algorithms, estimate resource requirements, draw
# diagrams, and more. 
#
# We can choose to wrap our ``qpe_bloq`` simply by setting ``map_ops`` to ``False``. This wraps the 
# ``qpe_bloq`` as a ``ToBloq`` Bloq, whose information is based on that of the original PennyLane object.

wrapped_qpe_bloq = qml.to_bloq(op, map_ops=False)
fig, ax = draw_musical_score(get_musical_score_data(wrapped_qpe_bloq.decompose_bloq()))
fig.tight_layout()

######################################################################
# Notice the differences between mapping and wrapping. When we map, the drawn musical score is in
# terms of native Qualtran Bloqs such as `Allocate`. When we wrap, the musical score has the 3 
# wires explicitly drawn and handled, because there is no PennyLane `Allocate` operator.
#
# Let's see how mapping and wrapping affects our resource count estimates.

from qualtran.drawing import show_counts_sigma

_, mapped_sigma = qpe_bloq.call_graph()
_, wrapped_sigma = wrapped_qpe_bloq.call_graph()

show_counts_sigma(mapped_sigma)
show_counts_sigma(wrapped_sigma)

# Here, we can clearly see that the resource counts for the two methods are distinctly different.
# This is because the underlying implementations for the two QPE operators differ.
#
# When Qualtran computes the resource counts for a `Bloq`, it first checks if there is a call graph
# defined. If it is, Qualtran uses that call graph to compute the resource count estimates. If it
# is not, Qualtran uses the decomposition to compute the resource count estimates.
#
# For wrapped `ToBloq`s, call graphs are generally not defined. This means the decompositions are
# used to compute the counts. However, since computing decompositions is computationally expensive,
# many PennyLane templates, such as QPE, have call graphs defined even when wrapped as a `ToBloq`.
# By defining these call graphs, you can now efficiently compute resource count estimates for 
# circuits that may require thousands of qubits and trillions of gates. 
#
# There is one caveat to note: for performance reasons, the call graphs sometimes differ from the
# actual decomposition. This means you may get counts more optimal (but still accurate) than what 
# the original PennyLane decomposition might have prescribed. As we continue to develop both the 
# PennyLane-Qualtran integration and PennyLane itself, these call graphs will evolve as well.

######################################################################
# Conclusion
# ----------
# In this how to, we demonstrated how to use the features in the integration with Qualtran. Whether
# that is to leverage Qualtran's powerful subroutines in Pennylane circuits, or to analyze and 
# reason about PennyLane objects using Qualtran, we are confident that this tool will help speed up
# your research and make new advances in quantum computing.
#
# About the author
# ----------------
#
