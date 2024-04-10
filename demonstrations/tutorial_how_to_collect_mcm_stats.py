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
# Warmup: Gather statistics on a recycled qubit
# ---------------------------------------------
#
# As a warmup exercise and to (re)familiarize ourselves with measurement processes
# in quantum circuits, we start with a simple example for mid-circuit measurements:
#
#   #. Rotate a single qubit with a ``qml.RY`` gate about some input angle,
#   #. perform a mid-circuit measurement on the qubit with :func:`~.pennylane.measure`,
#   #. repeat the procedure with other input angles and hyperparameters, and
#   #. return statistics about all performed measurements with :func:`~.pennylane.probs`.
#
# If you want to dive into the topic a bit slower, also consider the related tutorials,
# in particular those focusing on the fundamentals of quantum measurements
# and their mid-circuit versions.
#

import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", seed=21)  # seed only used for shot-based evaluations


@qml.qnode(dev, interface="numpy")
def single_qubit_stats(angles):
    mcms = []
    # For each angle, perform a rotation of the qubit and measure it
    qml.RY(angles[0], 0)
    # By default, the qubit is reset to the |0> state, and postselection is off
    mcms.append(qml.measure(0))
    qml.RY(angles[1], 0)
    # We can skip resetting the qubit with `reset=False`
    mcms.append(qml.measure(0, reset=False))
    qml.RY(angles[2], 0)
    # By passing 0 or 1, we can postselect on the corresponding outcome
    mcms.append(qml.measure(0, postselect=1))
    # Return the estimates of the measurement probabilities for each of the MCMs.
    return [qml.probs(op=mcm) for mcm in mcms]


angles = [np.pi / 4, np.pi / 2, np.pi]
stats = single_qubit_stats(angles)
for angle, stat in zip(angles, stats):
    print(f"Probability to measure 0/1 after rotation by {angle:.6f}: {np.round(stat, 6)}")

######################################################################
#
# The ``reset`` keyword argument is crucial to obtain a "cleanly recycled"
# qubit after using ``qml.measure``, and postselection via ``postselect`` can change the
# statistics of measurements performed *before* the postselecting measurement.
#
# If there is *no* chance of measuring a value on which we postselect, the circuit
# will not collect *any* statistics. The result is a ``nan`` value, accompanied
# by a ``RuntimeWarning`` that indicates that the probabilities were not estimated properly.
#


@qml.qnode(dev, interface="numpy")
def node():
    mcm = qml.measure(0, postselect=1)
    return qml.probs(op=mcm)


print(f"Probability to measure 0/1 in state |0> if we measured 1: {np.round(node(), 6)}")

######################################################################
#
# Performance: Deferring measurements vs. one-shot transform
# ----------------------------------------------------------
#
# There are currently two ways of simulating quantum circuits with mid-circuit measurements
# in PennyLane on classical simulator devices. New methods are likely to be added in the
# near future. Here we will not discuss these methods in detail but focus
# on PennyLane's default choices and on how to pick the best performing method.
#
# The first method is :func:`~.pennylane.defer_measurements`.
# It performs particularly well for few mid-circuit measurements and large numbers of
# shots, including analytic evaluations (corresponding to infinitely many shots).
# This method is applied by default if the simulating device
# runs with ``shots=None``, or if it only supports the deferred measurement principle.
#
# The second method is :func:`~.pennylane.dynamic_one_shot`.
# It performs well in the few-shots regime and easily handles large numbers of mid-circuit
# measurements.
# This method is applied by default if the simulating device runs with ``shots!=None``
# and natively supports the method.
#
# Postprocessing mid-circuit measurements within a QNode
# ------------------------------------------------------
#
# In contrast to quantum measurements at the end of a QNode, PennyLane supports
# a number of unary and binary operators for MCMs even within ``QNode``\ s.
# This allows us to return postprocessed statistics directly from the node:
#


@qml.qnode(dev)
def processed_mcms():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    qml.CNOT([0, 2])
    qml.CNOT([0, 3])
    mcms = [qml.measure(w) for w in range(4)]
    # Convert to +-1 based samples and form their product
    prod = np.prod([2 * mcm - 1 for mcm in mcms])
    # Equivalently, return the condition that all MCMs are equal
    equality = (mcms[0] == mcms[1]) & (mcms[0] == mcms[2]) & (mcms[0] == mcms[3])
    # The information about all qubits being measured in the same
    # state can also be extracted from their summed MCM values
    sum_ = sum(mcms)
    return qml.sample(prod), qml.sample(equality), qml.sample(sum_)


print(*processed_mcms(shots=20), sep="\n")

######################################################################
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
# Bringing everything together: Stats of postprocessed MCMs
# ---------------------------------------------------------
#
# To sum up everything, consider the following (somewhat arbitrary) QNode:
#


@qml.qnode(dev)
def stats(x, y, num_wires):
    [qml.RX(x[i], i) for i in range(num_wires)]
    qml.RY(y, num_wires)
    mcm1 = qml.measure(num_wires, reset=False)
    for i in range(num_wires):
        qml.CNOT([num_wires, i])
    mcms2 = [qml.measure(i, reset=False) for i in range(num_wires)]
    [qml.RX(-x[i], i) for i in range(num_wires)]
    for i in range(num_wires):
        qml.CNOT([i, num_wires])
    mcms3 = [qml.measure(i) for i in range(num_wires)]
    qml.Hadamard(num_wires)
    mcm4 = qml.measure(num_wires, postselect=1)
    value = (
        np.dot(3 ** np.arange(num_wires), mcms2)
        + np.dot(2 ** np.arange(num_wires), mcms3)
        - 5 * mcm4
    )

    return (
        qml.expval(qml.X(0) @ qml.Z(3) + 3 * qml.Y(1)),  # Standard expval measurement
        qml.var(mcm1),  # Variance of single MCM
        qml.counts(mcms2[:2]),  # Counter statistics of list of MCMs
        qml.probs(op=mcms3[::2]),  # Probability estimates for some MCMs
        qml.expval(value),  # Postprocessed MCMs
        qml.sample(value),  # The samples that produce the expval above
    )


######################################################################
# Drawing of QNodes is fully supported:
#

import matplotlib.pyplot as plt

np.random.seed(521)
num_wires = 4
x = np.random.random(num_wires)
y = np.random.random()
print(qml.draw(stats, decimals=0, max_length=160)(x, y, num_wires))

fig, ax = qml.draw_mpl(stats)(x, y, num_wires)
plt.show()

######################################################################
# Let's execute the QNode with ``30`` shots:
#

stats_ = stats(x, y, num_wires, shots=30)
print(f"Quantum expval of X(0) @ Z(3) + 3 Y(1):       {stats_[0]}")
print(f"Variance of single-qubit MCM:                 {stats_[1]}")
print(f"Counter statistics on first two qubits:       {stats_[2]}")
print(f"Probability estimates for qubits 0 and 2:     {stats_[3]}")
print(f"Expectation value of postprocessed MCM value: {stats_[4]}")
print(f"Samples of postprocessed MCM value:           {stats_[5]}")

######################################################################
# As we can see, only some of the ``30`` samples have been postselected
# by ``mcm4``, leaving us with a reduced sample size for *all* obtained
# statistics.
#
# This concludes our how-to on statistics and postprocessing of
# mid-circuit measurements. For details consider the
# `introduction on measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation of :func:`~.pennylane.measure`.
# For performance considerations, take a look at
# :func:`~.pennylane.defer_measurements` and
# :func:`~.pennylane.dynamic_one_shot`.
