r"""How to use Qualtran with PennyLane
======================================

Get ready to expand your quantum programming toolkit!
PennyLane and `Qualtran <https://qualtran.readthedocs.io/en/latest/>`_ integrate their best features,
enabling you to to visualize circuits,
count qubits and gates, and simulate outputs for programs built with
`Qualtran bloqs <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_,
PennyLane operations, or even a mix of both.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-qualtran-integration-open-graph.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

This integration allows you to:

* **Simulate Qualtran circuits:** Verify the correctness of your Qualtran programs by simulating
    whole Qualtran circuits and checking their outputs in PennyLane.
* **Expand PennyLane circuits with Qualtran subroutines:** Seamlessly incorporate Qualtran's quantum 
    subroutines, known as `bloqs <https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library>`_,
    directly in your PennyLane simulations.
* **Analyze the resources of PennyLane circuits with Qualtran:** Leverage Qualtran's advanced analysis tools like
    `drawing call graphs <https://qualtran.readthedocs.io/en/latest/reference/qualtran/drawing/show_call_graph.html>`_,
    `counting qubits <https://qualtran.readthedocs.io/en/latest/resource_counting/qubit_counts.html>`_,
    and `counting gates <https://qualtran.readthedocs.io/en/latest/reference/qualtran/drawing/show_counts_sigma.html>`_. 
* **Expand Qualtran circuits with PennyLane gates and subroutines:** Add any PennyLane 
    :doc:`introduction/operations` or
    :doc:`introduction/templates` to a Qualtran circuit.

In this demo, we want to show you how easy it is to use PennyLane and Qualtran together.
We'll start by focusing on the first key capability: simulating Qualtran ``Bloqs`` using
PennyLane ``QNodes``.
"""

######################################################################
# Simulating Qualtran Bloqs and Larger Algorithms
# -----------------------------------------------
#
# For those unfamiliar, Qualtran (quantum algorithms translator) is a set of abstractions for 
# representing quantum programs and a library of quantum algorithms (Bloqs) expressed in that 
# language.
#
# With barely any work, you can drop Qualtran Bloqs into your PennyLane circuits!
# You only need one class: :class:`~pennylane.FromBloq`. It wraps an entire Qualtran ``Bloq`` as a 
# PennyLane operator. It faithfully converts any Bloq, including its decomposition, into an operator
# you can treat like you would any other.
#
# Let's see how it works!


import pennylane as qml
from qualtran.bloqs.basic_gates import XGate

bloq_as_op = qml.FromBloq(XGate(), wires=0)
print(bloq_as_op)

######################################################################
# In this simple example, we wrapped a Qualtran ``XGate`` instance using ``FromBloq``. We can see that
# the output is a :class:`~.pennylane.io.FromBloq` instance, whose properties are the same as
# PennyLane's PauliX operator.
#
# Since Qualtran Bloqs don't know what wires to act on, we need to provide that information to 
# ``FromBloq`` via the ``wires`` argument. Wire requirements will vary depending on the Bloq. You
# can use the :func:`~.pennylane.bloq_registers` helper function to determine what values to 
# provide for ``wires``.

print(qml.bloq_registers(XGate()))

######################################################################
# This will create :func:`~.pennylane.registers` with register names in accordance to the Bloq's signature. 
# Here, the function created one "q" register with a single qubit, as required by the ``XGate``.
#
# Let's look at a more complicated example! Qualtran has a special type of addition, known as
# Galois Field addition (``GF2Addition``) that is not implemented in PennyLane. You can think of it
# as binary addition that doesn't allow carrying. Let's use the combined force of PennyLane and
# Qualtran to bring ``GF2Addition`` to life!

from qualtran.bloqs.gf_arithmetic import GF2Addition
arithmetic_bloq = GF2Addition(4)
wires = qml.bloq_registers(arithmetic_bloq)
five = [0, 1, 0, 1] # 5 in binary
ten = [1, 0, 1, 0] # 10 in binary

@qml.qnode(qml.device('default.qubit', shots=1))
def circuit():
    qml.BasisState(five + ten, wires=wires['x']+wires['y'])
    qml.FromBloq(arithmetic_bloq, wires=wires['x']+wires['y'])
    a = [qml.measure(i) for i in range(len(wires['x']+wires['y']))]
    return qml.sample(a)

binary_string = "".join([str(bit) for bit in circuit()])
print("GF Addition of 5 + 10 =", int(binary_string[len(wires['x']):],2))

######################################################################
# Wow! Just like magic, we used a Qualtran Bloq like a PennyLane template without any additional
# work. With Qualtran's expansive library of quantum algorithms, you can now build a greater
# variety of circuits using a combination of PennyLane templates and Qualtran Bloqs.
#
# Analyzing PennyLane Circuits in Qualtran
# ----------------------------------------
#
# Now, we'll show you how to convert PennyLane objects to Qualtran Bloqs. For brevity, we'll 
# mainly cover how operators get converted to Bloqs in our examples, but functions with operators and
# ``QNodes`` work just the same.
#
# There are three main options for the conversion. We'll briefly introduce them here but go into
# greater detail in subsequent sections:
#
# - Smart defaults: In this option, PennyLane chooses what Qualtran Bloq to translate to.
#   If an option exists, we'll give you a Qualtran Bloq that is highly similar to the PennyLane
#   operator.
#
# - Custom mapping: Want something different from the smart default? Don't worry, you can
#   customize what Bloq you want your operator to map to. This makes it easy to
#   refine the finer details of your algorithm.
#
# - Wrapping: Think of this as an analogue of ``FromBloq``. It faithfully converts any operator or
#   Qfunc, decompositions and all, into a Bloq. The output is a :class:`~pennylane.io.ToBloq` instance.
#
# Holding all these options together is our trusty function :func:`~pennylane.to_bloq`. In the
# following sections, we'll explore how we can wield this powerful function to get all the
# functionality introduced above.
#
######################################################################
# Smart defaults
# --------------
#
# By default, ``qml.to_bloq`` tries its best to translate 
# PennyLane objects to Qualtran-native objects. This makes certain Qualtran
# functionalities, such as gate counting and generalizers, work more seamlessly.
# Here, PennyLane's ``PauliX`` operator is mapped directly to Qualtran's ``XGate``.

op_as_bloq = qml.to_bloq(qml.X(0))
print(op_as_bloq)

# Not all PennyLane operators are as straightforward to map as the ``PauliX`` operator. For example, 
# PennyLane's ``Quantum Phase Estimation`` could be mapped to a variety of Qualtran Bloqs. In cases
# where the mapping is ambiguous, we get the smart default:

from qualtran.drawing import draw_musical_score,  get_musical_score_data

op = qml.QuantumPhaseEstimation(unitary=qml.RY(phi=0.3, wires=[0]), estimation_wires=[1, 2, 3])
qpe_bloq = qml.to_bloq(op)
fig, ax = draw_musical_score(get_musical_score_data(qpe_bloq.decompose_bloq()))
fig.tight_layout()

######################################################################
# Here, the smart default is Qualtran's ``TextbookQPE`` where ``ctrl_state_prep`` is Qualtran's 
# ``RectangularWindowState``. But what if we wanted to use a different Bloq for our 
# ``ctrl_state_prep``? In this case, we turn to custom mappings.
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
# or smart default - the circuit is wrapped as a ``ToBloq`` Bloq.

def circ():
    qml.X(0)
    qml.X(1)

qfunc_as_bloq = qml.to_bloq(circ)
print(type(qfunc_as_bloq))

######################################################################
# Wrapping
# --------
#
# Functionally, wrapping a quantum function or operator as a ``ToBloq`` is similar to wrapping a Bloq
# as a ``FromBloq``. A wrapped PennyLane object acts like a Bloq: it can be analyzed
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
# terms of native Qualtran Bloqs such as ``Allocate``. When we wrap, the musical score has the 3 
# wires explicitly drawn and handled, because there is no PennyLane ``Allocate`` operator.
#
# Let's see how mapping and wrapping affects our resource count estimates.

from qualtran.drawing import show_counts_sigma

_, mapped_sigma = qpe_bloq.call_graph()
_, wrapped_sigma = wrapped_qpe_bloq.call_graph()

show_counts_sigma(mapped_sigma)
show_counts_sigma(wrapped_sigma)

######################################################################
# Here, we can clearly see that the resource counts for the two methods are distinctly different.
# This is because the underlying implementations for the two QPE operators differ.
#
# When Qualtran computes the resource counts for a ``Bloq``, it first checks if there is a call graph
# defined. If it is, Qualtran uses that call graph to compute the resource count estimates. If it
# is not, Qualtran uses the decomposition to compute the resource count estimates.
#
# For wrapped ``ToBloq``s, call graphs are generally not defined. This means the decompositions are
# used to compute the counts. However, since computing decompositions is computationally expensive,
# many PennyLane templates, such as QPE, have call graphs defined even when wrapped as a ``ToBloq``.
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
# In this how to, we took a look at how to use Pennylane with Qualtran. There's so much more that's
# possible that we couldn't cover here. We encourage you to play around with the
# integration: try swapping out PennyLane operators with Qualtran Bloqs and running it on a real
# quantum device! Or convert an existing PennyLane workflow into Qualtran and see how Qualtran
# estimates the number of T gates. If you'd like to see more examples of Qualtran and PennyLane
# or would like to know further technical details, be sure to check out the docs for :func:`~.pennylane.to_bloq`
# and :class:`~.pennylane.FromBloq`!
#
# About the author
# ----------------
#
