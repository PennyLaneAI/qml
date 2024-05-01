r"""How to create dynamic circuits with mid-circuit measurements
================================================================

Measuring qubits in the middle of a quantum circuit execution can be useful in many ways.
From understanding the inner workings of a circuit, hardware characterization,
modeling and error mitigation, to error correction, algorithmic improvements and even up to full
computations encoded as measurements in measurement-based quantum computation (MBQC).

Before turning to any of these advanced topics, it is worthwhile to familiarize ourselves with
the syntax and features around mid-circuit measurements (MCMs). In this how-to, we will focus on
dynamic quantum circuits that use control flow based on MCMs.
Most of the advanced concepts mentioned above incorporate MCMs in this way, making it a
key ingredient to scalable quantum computing.

.. figure:: ../_static/demonstration_assets/how_to_create_dynamic_mcm_circuits/socialsthumbnail_how_to_create_dynamic_mcm_circuits.png
    :align: center
    :width: 50%

"""

######################################################################
# Minimal working example
# -----------------------
#
# We start with a minimal dynamic circuit on two qubits. It rotates one qubit
# about the ``X``-axis and prepares the other qubit in a fixed state.
# After an entangling :class:`~.pennylane.CNOT` gate, the second qubit is measured,
# and if it measured a ``1``, an :class:`~.pennylane.S` gate is applied.
# Finally, the expectation value of the Pauli ``Y`` operator is returned.
#

import pennylane as qml
import numpy as np

dev = qml.device("lightning.qubit", wires=2)

fixed_state = np.array([1.0, np.exp(1j * np.pi / 4)]) / np.sqrt(2)


@qml.qnode(dev, interface="numpy")
def circuit(x):
    qml.RX(x, 0)

    qml.QubitStateVector(fixed_state, 1)
    qml.CNOT(wires=[0, 1])
    mcm = qml.measure(1)
    qml.cond(mcm, qml.S)(wires=0)

    return qml.expval(qml.Y(0))


x = 1.361
print(circuit(x))

######################################################################
# In case you wondered, this circuit implements a so-called
# `T-gadget <https://arxiv.org/abs/quant-ph/0002039>`_ and the ``fixed_state``
# we prepared on the second qubit is called a
# `"magic state" <https://en.wikipedia.org/wiki/Magic_state_distillation#Magic_states>`_,
# but this will not concern us here.
#
# After this minimal working example, we now construct a more complex circuit
# showcasing more features of MCMs and dynamic circuits in PennyLane. We start
# with some short preparatory definitions.
#
#
# VERSION 1
#
# Creating half-filled basis states with a dynamic circuit
# --------------------------------------------------------
#
# We now turn to a more complex example of a dynamic circuit.
# In the following, we will build a circuit that probabilisticly initializes half-filled
# computational basis states, i.e. basis states with as many :math:`1`\ s as :math:`0`\ s.
# The procedure is as follows:
# Single-qubit rotations and a layer of :class:`~.pennylane.CNOT` gates create an entangled
# state on the first three qubits. Afterwards, the qubits are measured and for each qubit
# that has been measured in the state :math:`|0\rangle`, another qubit is excited from the
# :math:`|0\rangle` state to the :math:`|1\rangle` state.
# Finally, we sample the output states to investigate whether the circuit works as intended.
#
# We start by defining a quantum subprogram that creates the initial state:
#


def init_state(x):
    # Rotate the first three qubits
    for w in range(3):
        qml.RX(x[w], w)
    # Entangle the first three qubits
    qml.CNOT([0, 1])
    qml.CNOT([1, 2])
    qml.CNOT([2, 0])


######################################################################
# With this subroutine in our hands, let's define the full :class:`~.pennylane.QNode`.
# For this, we also create a shot-based device.
#

shots = 100
dev = qml.device("default.qubit", shots=shots)


@qml.qnode(dev)
def create_half_filled_state(x):
    init_state(x)
    for w in range(3):
        # Measure one qubit at a time and flip another, fresh qubit if measured 0
        mcm = qml.measure(w)
        qml.cond(~mcm, qml.X)(w + 3)

    return qml.counts(wires=range(6))


######################################################################
# Before running this ``QNode``, let's sample some random input parameters and
# draw the circuit:

np.random.seed(652)
x = np.random.random(3) * np.pi

print(qml.draw(create_half_filled_state)(x))

######################################################################
# We can see the initial state creation and the measure & conditional bit flip
# applied to pairs of qubits.
#
# Great, now let's finally see if it works:
#

counts = create_half_filled_state(x)
print(f"Sampled bit strings:\n{list(counts.keys())}")

######################################################################
# Indeed, we created half-filled computational basis states, each with its own
# probability:
#

print("The probabilities for the bit strings are:")
for key, val in counts.items():
    print(f"    {key}: {val/shots*100:4.1f} %")

######################################################################
# Quiz question: Did we create *all* possible half-filled basis states at
# least once? You can find the answer at the end of this how-to.
#
# Postselecting mid-circuit measurements
# --------------------------------------
# We may select only some of these half-filled states by postselecting on
# measurement outcomes we prefer:
#


@qml.qnode(dev)
def postselect_half_filled_state(x, selection):
    init_state(x)
    for w in range(3):
        # Postselect the measured qubit to match the selection criterion
        mcm = qml.measure(w, postselect=selection[w])
        qml.cond(~mcm, qml.X)(w + 3)

    return qml.counts(wires=range(6))


######################################################################
# As an example, suppose we wanted half-filled states that have a 0
# in the first and a 1 in the third position. We do not postselect on
# the second qubit, which we can indicate by passing ``None`` to the
# ``postselect`` argument of :func:`~.pennylane.measure`.
# Again, before running the circuit, let's draw it first:
#

selection = [0, None, 1]
print(qml.draw(postselect_half_filled_state)(x, selection))

######################################################################
# Note the indicated postselection values next to the drawn mid-circuit
# measurements.
#
# Time to run the postselecting circuit:

counts = postselect_half_filled_state(x, selection)
postselected_shots = sum(counts.values())

print(f"Obtained {postselected_shots} out of {shots} samples after postselection.")
print("The probabilities for the postselected bit strings are:")
for key, val in counts.items():
    print(f"    {key}: {val/postselected_shots*100:4.1f} %")

######################################################################
# We successfully postselected on the desired properties of the computational
# basis state. Note that the number of returned samples is reduced, because
# those samples that do not meet the postselection criterion are discarded
# entirely.
#
# The quiz question from above may have become a bit easier to answer with
# this result...
#
# Replacing postselection by more dynamic gates
# ---------------------------------------------
#
# If we do not want to postselect the prepared states but still would like
# to guarantee some of the bit strings to be in a given state, we may instead
# flip the corresponding pairs of bits:
#


@qml.qnode(dev)
def create_selected_half_filled_state(x, selection):
    init_state(x)
    all_mcms = []
    for w in range(3):
        # Don't postselect on the selection criterion, but store the MCM for later
        mcm = qml.measure(w)
        qml.cond(~mcm, qml.X)(w + 3)
        all_mcms.append(mcm)

    for w, sel, mcm in zip(range(3), selection, all_mcms):
        # If the postselection criterion is not None, flip the corresponding pair
        # of qubits conditioned on the mcm not satisfying the selection criterion
        if sel is not None:
            qml.cond(mcm != sel, qml.X)(w)
            qml.cond(mcm != sel, qml.X)(w + 3)

    return qml.counts(wires=range(6))


print(qml.draw(create_selected_half_filled_state)(x, selection))

######################################################################
# We can see how the measured values are fed not only into the original
# conditioned operation, but also into two more bit flips, as long
# as the selection criterion is not ``None``.
# Let's execute the circuit:

counts = create_selected_half_filled_state(x, selection)
postselected_shots = sum(counts.values())

print(f"Obtained all {postselected_shots} of {shots} samples because we did not postselect")
print("The probabilities for the selected bit strings are:")
for key, val in counts.items():
    print(f"    {key}: {val/postselected_shots*100:4.1f} %")

######################################################################
# Note that we kept all samples because we did not postselect.
# Also, note that we conditionally applied the bit flip operators ``qml.X``
# by comparing a mid-circuit measurement result with the corresponding
# selection criterion (``mcm!=sel``).
# More generally, mid-circuit measurement results can be processed with standard
# arithmetic operations. For details, see the `introduction to MCMs
# <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation of :func:`~.pennylane.measure`.
#
# And this is how to create a dynamic circuit in PennyLane with mid-circuit
# measurements with feedforward control!
#
# Before finishing, here is the answer to the quiz question:
# We did not create all possible half-filled states at least once.
# This is because our circuit forces each of the qubit pairs
# ``(0, 3)``, ``(1, 4)`` and ``(2, 5)`` to be in opposite states.
# However, there are half-filled states that do not have this form,
# as for example ``100110``, which you will not find among the sampled states
# from our circuit.
#
# VERSION 2
#
# Defining quantum subprograms
# ----------------------------
#
# We start by defining two quantum subprograms: blocks of single-qubit
# and two-qubit gates, applied in one layer each.
# We also fix the number of qubits to three.
#

num_wires = 3
wires = list(range(num_wires))


def first_block(x):
    [qml.RX(x, w) for w in wires]
    [qml.CNOT([w, (w + 1) % num_wires]) for w in wires]


def block(param):
    [qml.CRY(param, wires=[w, (w + 1) % num_wires]) for w in wires]
    [qml.Hadamard(w) for w in wires]


######################################################################
# Processing MCMs into boolean conditions
# ---------------------------------------
#
# Next, we define two functions that will process MCMs into a boolean condition
# within the feedforward control flow of the dynamic quantum circuit.
# They are chosen arbitrarily, but showcase that standard arithmetic and
# comparators are supported with MCM values (for more details, consider the
# `introduction to MCMs <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation of :func:`~.pennylane.measure`.
#


def condition1(mcms):
    return np.dot(2 ** np.arange(len(mcms)), mcms) >= 3


def condition2(mcms):
    return np.dot(3 ** np.arange(len(mcms)), mcms) < 3


######################################################################
# Miscellaneous preparations
# --------------------------
#
# To conclude our preparations, we also define a shot-based device
# and a Hamiltonian to be measured.
#

dev = qml.device("default.qubit", shots=100)

ops = [qml.X(0) @ qml.Y(1), qml.Z(1) @ qml.X(2), qml.Y(2) @ qml.Z(0)]
H = qml.dot([0.3, 1.2, -0.5], ops)

######################################################################
# Defining the dynamic quantum circuit
# ------------------------------------
#
# Now we are ready to create a :class:`~.pennylane.QNode`. It will execute blocks
# of quantum gates interleaved with layers of mid-circuit measurements.
# The MCMs are either processed into a condition for whether the next block is
# applied, using the functions ``condition1`` and ``condition2`` from above,
# or they are used for postselection.
#


@qml.qnode(dev)
def circ(x, y, z):
    # Apply the first block of gates
    first_block(x)
    # Measure all qubits w/o resetting them; store the mid-circuit measurement values
    first_mcms = [qml.measure(w) for w in wires]
    # Compute a boolean condition based on the MCMs
    mid_block_condition = condition1(first_mcms)
    # Apply another block of quantum gates if the computed condition is True
    qml.cond(mid_block_condition, block)(y)

    # Measure the first qubit and postselect on having measured "0"
    postselected_mcm = qml.measure(0, postselect=0)
    # Measure the other qubits and reset them
    second_mcms = [qml.measure(w, reset=True) for w in wires[1:]]
    # Compute a boolean condition based on the second set of MCMs
    last_block_condition = condition2(second_mcms)
    # If the second computed condition is True, apply another block.
    # If it is False, instead apply the first block once more
    qml.cond(last_block_condition, block, first_block)(z)

    # Return the (standard) expectation value of the precomputed Hamiltonian,
    # the counters for the two boolean conditions and a common counter
    # for all performed MCMs
    return (
        qml.expval(H),
        qml.counts(mid_block_condition),
        qml.counts(last_block_condition),
        qml.counts([*first_mcms, postselected_mcm, *second_mcms]),
    )


np.random.seed(28)
x, y, z = np.random.random(3)

expval, mid_block_condition, last_block_condition, all_mcms = circ(x, y, z)
print(f"Expectation value of H:\n{expval:.6f}\n")
print(f"Counts for boolean condition for middle block:\n{mid_block_condition}\n")
print(f"Counts for boolean condition for last block:\n{last_block_condition}\n")
all_mcms_formatted = "\n".join(f"    {key}: {val:2d}," for key, val in all_mcms.items())
print(f"Counts for bitstrings of all MCMs:\n{{\n{all_mcms_formatted}\n}}")

######################################################################
# Great, the circuit runs! And it does not only estimate the expectation value of ``H``,
# but it also returns the samples of the dynamic circuit conditions ``mid_block_condition``
# and ``last_block_condition`` as well as all performed measurements individually.
# Note that we only collected ``80`` shots, although the device uses ``100`` shots per
# circuit execution. This is due to the postselection on ``postselected_mcm``, which
# accordingly is registered to return ``0``\ s only.
#
# Visualizing the dynamic circuit
# -------------------------------
#
# Finally, let's look at the circuit we constructed:
#

print(qml.draw(circ, max_length=300)(x, y, z))
fig, ax = qml.draw_mpl(circ)(x, y, z)

######################################################################
# Can you detect all blocks we included and how they are conditioned on
# the MCM values? Note how independent measurement values cross with a
# gap between the double-drawn wires (``═║═``) just like quantum and classical
# wires do (``─║─``), whereas measurement values that are processed
# together are shown without such a gap (``═╬═``).
#
# This concludes our brief how-to on dynamic circuits with mid-circuit measurements
# in PennyLane. For details on MCMs, consider the
# `documentation on MCMs <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_,
# the documentation of :func:`~.pennylane.measure` and other related demos and how-tos
# shown on the right.
