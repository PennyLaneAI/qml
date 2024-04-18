r"""How to collect statistics of mid-circuit measurements
=========================================================

Measuring qubits in the middle of a quantum circuit execution can be useful in many ways.
From understanding the inner workings of a circuit, over hardware characterization,
modeling and error mitigation, to error correction, algorithmic improvements and even up to full
computations encoded as measurements in measurement-based quantum computation (MBQC).

Before turning to any of these advanced topics, it is worthwhile to familiarize ourselves with
the syntax and features around mid-circuit measurements. In this how-to, we will focus on
extracting statistics about measurements that are performed while a quantum circuit is up and
running --- mid-circuit measurement statistics!

.. figure:: ../_static/demonstration_assets/how_to_collect_mcm_stats/socialthumbnail_large_how_to_collect_mcm_stats.png
    :align: center
    :width: 50%

If you are interested in how to use mid-circuit measurements to create dynamic circuits
in PennyLane, also check out the related how-to on that topic!

"""

######################################################################
#
# Defining quantum subprograms
# ----------------------------
#
# We start by preparing some subroutines consisting of a few operations each.
# They will help us to see the circuit structure clearly later on.
# We also define the number of wires and the device already.
#

import pennylane as qml
import numpy as np

np.random.seed(521)

num_wires = 4
wires = list(range(num_wires))

dev = qml.device("default.qubit")


def rx_layer(x):
    """Apply qml.RX on each qubit with a different parameter from the input x."""
    qml.broadcast(qml.RX, pattern="single", parameters=x, wires=wires)


def entangle(direction=None):
    """Apply a layer of CNOTs."""
    if direction == "forward":
        for i in range(num_wires):
            qml.CNOT([i, num_wires])
    elif direction == "backward":
        for i in range(num_wires):
            qml.CNOT([num_wires, i])


######################################################################
#
# Postprocessing mid-circuit measurements
# ---------------------------------------
#
# In contrast to quantum measurements at the end of a QNode, PennyLane supports
# a number of unary and binary operators for MCMs even within ``QNode``\ s.
# We prepare a processing function using such operators and will use it in a
# ``QNode`` below.
#


def process_mcms(mcm_list1, mcm_list2, mcm):
    """Processing function taking two lists of MCMs and an extra MCM.
    It returns an arithmetic expression of the input MCMs."""
    first = np.dot(3 ** np.arange(num_wires), mcm_list1)
    second = np.dot(2 ** np.arange(num_wires), mcm_list2)
    third = 5 * mcm
    return first + second - third


def equality(mcm_list):
    """A function checking whether all measurements in a list of MCMs
    are equal."""
    equal = mcm_list[0] == mcm_list[1]
    for mcm in mcm_list[2:]:
        equal &= mcm_list[0] == mcm
    return equal


######################################################################
#
# Bringing everything together: Stats of postprocessed MCMs
# ---------------------------------------------------------
#
# To sum up everything, consider the following QNode, making use of the quantum subprograms
# and postprocessing functions from above:
#


@qml.qnode(dev)
def stats(x, y, num_wires):
    # Rotate all qubits
    rx_layer(x)
    qml.RY(y, num_wires)
    # Measure last qubit
    mcm1 = qml.measure(num_wires)

    # Entangle all qubits with last qubit
    entangle("backward")
    # Measure all but last qubit and reset them
    mcms2 = [qml.measure(i, reset=True) for i in range(num_wires)]

    # Draw a Barrier to help visualizing circuit structure
    qml.Barrier(only_visual=True)
    # Rotate all but last qubit and apply CNOTs with last qubit
    rx_layer(-x)
    entangle("forward")
    # Measure all but last qubit without reset
    mcms3 = [qml.measure(i) for i in range(num_wires)]

    # Change measurement basis of last qubit
    qml.Hadamard(num_wires)
    # Mesaure last qubit and postselect on measuring a 1
    mcm4 = qml.measure(num_wires, postselect=1)

    # Postprocess all but the first MCM
    value = process_mcms(mcms2, mcms3, mcm4)
    equ = equality(mcms2)

    return (
        qml.expval(qml.X(0) @ qml.Z(3) + 3 * qml.Y(1)),  # Standard expval measurement
        qml.var(mcm1),  # Variance of single MCM
        qml.counts(mcms2[:2]),  # Counter statistics of list of MCMs
        qml.probs(op=mcms3[::2]),  # Probability estimates for some MCMs
        qml.expval(value),  # Postprocessed MCMs
        qml.sample(equ),  # an equality check of a list of MCMs
    )


######################################################################
# Let's draw some random parameters for the ``QNode`` and execute it
# with ``30`` shots:
#

x = np.random.random(num_wires)
y = np.random.random()

stats_ = stats(x, y, num_wires, shots=30)
print(f"Quantum expval of X(0) @ Z(3) + 3 Y(1):       {stats_[0]}")
print(f"Variance of single-qubit MCM:                 {stats_[1]}")
print(f"Counter statistics on first two qubits:       {stats_[2]}")
print(f"Probability estimates for qubits 0 and 2:     {np.round(stats_[3], 4)}")
print(f"Expectation value of postprocessed MCM value: {stats_[4]}")
print(f"Samples of equality condition:\n{stats_[5]}")


######################################################################
# As we can see, only ``11`` of the ``30`` samples have been postselected
# by ``mcm4``, leaving us with a reduced sample size for *all* obtained
# statistics.
#
# Drawing of QNodes is fully supported as well:
#

fig, ax = qml.draw_mpl(stats)(x, y, num_wires)

######################################################################
#
# This concludes our how-to on statistics and postprocessing of
# mid-circuit measurements. Below, you can find some complementary information
# on the supported arithmetic operations for MCMs as well as
# the supported return types with (postprocessed) MCMs.
#
# For more details consider the
# `introduction on measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation of :func:`~.pennylane.measure`.
# For performance considerations, take a look at
# :func:`~.pennylane.defer_measurements` and :func:`~.pennylane.dynamic_one_shot`.
#
# Supported arithmetic operations between MCMs
# --------------------------------------------
#
# A number of unary and binary operators are supported in PennyLane for mid-circuit
# measurements:
#
# The binary arithmetic operators ``+``, ``-``, ``*``, and ``/`` are supported between two
# MCMs and between an MCM and an ``int``, ``float``, ``bool``, or a 0-dimensional ``np.ndarray``.
# The operators are supported "both ways", that is both ``mcm + 4`` and ``4 + mcm`` are valid.
# The same holds for the comparators ``==``, ``<``, ``>``, ``<=``, and ``>=``.
# The boolean "not" (``~``) can be applied to MCMs (or combinations thereof, but it will
# always convert the result to a ``bool``). The bitwise "and" (``&``) and "or" (``|``) operators
# are supported between two MCMs and between and MCM and an ``int`` or ``bool``, but only if
# the MCM is put first, e.g., do ``mcm & 2``, not ``2 & mcm``.
#
# Arithmetic expressions that already contain one or multiple MCMs are supported just like
# a single MCM, allowing for nested arithmetic expressions.
#
# .. warning::
#
#     The bitwise operators ``&`` and ``|`` do not necessarily raise an error when used with
#     ``float``\ s or ``np.ndarray``\ s, even if they return incorrect results!
#
# .. note::
#
#     The bitwise "xor" operator ``^`` currently is not supported but can be obtained by using
#     ``(a | b) - (a & b)``. The Python operators ``and`` and ``or`` are not supported.
#     They usually can be obtained using implicit conversion to integers when applying
#     arithmetics. E.g. ``mcm0 and mcm1`` often is equivalent to ``mcm0 * mcm1``.
#
#
# Supported return types with MCMs
# --------------------------------
#
# Depending on the processing applied to the MCM results, not all return types are supported.
# For example, ``qml.probs(2 * mcm0)`` is not a valid return value, because it is not clear
# which probabilities are being requested.
# Furthermore, as usual the available return types depend on whether or not the device is
# shot-based (``qml.sample`` can not be returned if the device is not sampling).
# Overall, **all combinations of postprocessing and all of**
# :func:`~.pennylane.expval`,
# :func:`~.pennylane.var`,
# :func:`~.pennylane.probs`,
# :func:`~.pennylane.sample`, **and**
# :func:`~.pennylane.counts`,
# **are supported** with the following exceptions:
#
#   - ``qml.sample`` and ``qml.counts`` are not supported for ``shots=None``.
#   - ``qml.probs`` is not supported for MCMs collected in arithmetic expressions. For
#     arithmetic expressions with a single MCM, probabilities according to that of the MCM
#     itself are returned.
#   - ``qml.expval`` and ``qml.var`` are not supported for sequences of MCMs.
#     ``qml.probs``, ``qml.sample``, and ``qml.counts`` are supported for sequences but
#     only if they do not contain arithmetic expressions of these MCMs. That is,
#     ``qml.sample([mcm0, mcm1, mcm2])`` is supported, ``qml.sample([mcm0 + mcm1, mcm2])``
#     is not. You can use multiple return values instead, i.e.
#     ``qml.sample(mcm0 + mcm1), qml.sample(mcm2)``.
#
# **MCM statistics can be returned alongside standard terminal measurements.**
#
