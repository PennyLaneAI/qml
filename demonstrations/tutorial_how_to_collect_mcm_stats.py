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
in PennyLane, also check out our
:doc:`how-to on creating dynamic circuits using MCMs </demos/tutorial_how_to_create_dynamic_mcm_circuits>`.
If you already did, you may skip the next sections and jump
:ref:`here <end of copied part>`.

"""

######################################################################
# Warmup: Gather statistics on a recycled qubit
# =============================================
#
# As a warmup exercise and to (re)familiarize ourselves with measurement processes
# in quantum circuits, we start with a simple example for mid-circuit measurements:
#
#   #. Rotate a single qubit with a ``qml.RY`` gate about some input angle,
#   #. perform a mid-circuit measurement on the qubit with :func:`~.pennylane.measure`
#      and reset it,
#   #. repeat the procedure with other input angles, and
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
def single_qubit_stats(angles, reset=True, postselect=None):
    mcms = []
    # For each angle, perform a rotation of the qubit and measure it
    for angle in angles:
        qml.RY(angle, 0)
        mcms.append(qml.measure(0, reset=reset, postselect=postselect))
    # Return the estimates of the measurement probabilities for each of the MCMs.
    return [qml.probs(op=mcm) for mcm in mcms]


angles = [np.pi / 4, np.pi / 2, np.pi]
stats = single_qubit_stats(angles)
for angle, stat in zip(angles, stats):
    print(f"Probability to measure 0/1 after rotation by {angle:.6f}: {np.round(stat, 6)}")


######################################################################
# Note that the keyword arguments ``reset`` and ``postselect`` are set to the default values of
# ``qml.measure`` in our function definition.
# Of course one could also obtain these results by executing a circuit with a single
# rotation and final measurement for each angle individually. However, the above can be
# used to condense multiple runs of that experiment into one quantum circuit.
#
# Keyword arguments of ``qml.measure``: ``reset`` and ``postselect``
# ------------------------------------------------------------------
#
# If we change the ``reset`` keyword argument of ``qml.measure`` to ``False``, the qubit remains
# in the state it collapsed into after the measurement. This means that the measured probabilities
# for the different angles will be correlated and we no longer perform a sequence of
# independent experiments:
#

stats = single_qubit_stats(angles, reset=False)
for angle, stat in zip(angles, stats):
    print(f"Probability to measure 0/1 after rotation by {angle:.6f}: {np.round(stat, 6)}")

######################################################################
# This demonstrates that the ``reset`` keyword argument is crucial to obtain a "cleanly recycled"
# qubit after using ``qml.measure``.
#
# The second keyword argument of ``qml.measure`` is ``postselect``. When activated, the remaining
# part of the quantum circuit will only be executed if the measurement outcome matches the
# specified postselection value. Otherwise, the circuit execution will be discarded altogether,
# i.e. samples are not collected and the execution does not contribute to gathered statistics.
# For the circuit and input angles from above, we saw that there is always *some* chance to
# measure ``1``.
# In the example below we only consider the cases in which this happens, so that the
# probability to measure a ``1`` becomes :math:`100\%`.
#

stats = single_qubit_stats(angles, postselect=1)
for angle, stat in zip(angles, stats):
    print(f"Probability to measure 0/1 after rotation by {angle:.6f}: {np.round(stat, 6)}")

######################################################################
# We can think of this experiment as asking the question "What is the probability that we
# measured ``1`` provided that we measured ``1``?". The answer clearly is :math:`100\%`.
#
# There is a singularity in this setup, though: If there is *no* chance of measuring ``1`` in
# the first place but we postselect on exactly this measurement value, we will not collect *any*
# statistics. The result is a ``nan`` value, accompanied by a ``RuntimeWarning`` that
# indicates that the probabilities were not estimated properly.
#

zero_angle = 0.0
stats = single_qubit_stats([zero_angle], postselect=1)[0]
print(f"Probability to measure 0/1 after rotation by {zero_angle:.6f}: {np.round(stat, 6)}")

######################################################################
# Performance: Deferring measurements vs. dynamic one-shots
# =========================================================
#
# There are currently two ways of simulating quantum circuits with mid-circuit measurements
# in PennyLane on classical simulator devices. New methods are likely to be added in the
# near future. Here we will not discuss these methods in detail but focus
# on PennyLane's default choices and on how to pick the best performing method.
#
# The first method is to **defer measurements** until the end of the circuit. Under the hood,
# this allows the simulator to keep the quantum state pure, and **both analytic and
# (many-)shots-based results can easily be computed**. The main drawback of this method is
# that it requires us to simulate one additional qubit per mid-circuit measurement.
# In PennyLane, this method can be used by applying :func:`~.pennylane.defer_measurements`
# to a quantum function or ``QNode``. It is applied by default if the simulating device
# runs with ``shots=None``, or if it only supports the deferred measurement principle.
#
# The second method is to **sample through the mid-circuit measurements for each single shot**,
# or circuit execution. Under the hood, the simulator keeps a pure quantum state by sampling
# the measurement value of each encountered MCM, so that **it does not need any auxiliary qubits.**
# The fact that each circuit execution is sampled individually leads to two drawbacks, though:
# The computational runtime/cost is linear in the shot count, and in particular,
# analytic results are not supported.
# In PennyLane, this method can be activated by applying :func:`~.pennylane.dynamic_one_shot`
# to a quantum function or ``QNode``. It is applied by default if the simulating device
# runs with ``shots!=None`` and it natively supports the method.
#

angles = [0.4, 0.2]
# Automatically uses `qml.defer_measurements` because the device runs with `shots=None`
print(single_qubit_stats(angles, shots=None))
# Automatically uses `qml.dynamic_one_shot` because the device runs with `shots=20!=None`
print(single_qubit_stats(angles, shots=20))

# Manually forces the device to defer measurements although running with `shots=20`
print(qml.defer_measurements(single_qubit_stats)(angles, shots=20))

######################################################################
# It may seem that deferring measurements is the method of choice for MCM simulation, and
# often it is the faster option. This is because ``dynamic_one_shot`` needs to sample its
# way through the circuit for each shot, letting ``node(..., shots=100)`` take ten times as
# many computational resources as ``node(..., shots=10)``!
# However, the fact that ``defer_measurements`` adds qubits in the background implies that
# its computational cost grows *exponentially* with the number of mid-circuit measurements!
# This makes ``dynamic_one_shot`` the faster, if not the only, option for circuits with
# many MCMs or those that have a large qubit count anyways.
#
# We demonstrate this discussion in practice by running our toy circuit with different
# numbers of shots and mid-circuit measurements (controlled by the number of rotation
# angles we put in):

import timeit

rep = 5
print(" " * 28 + "dynamic_one_shot | defer_measurements")

for shots in [10, 1000]:
    for num_mcms in [2, 20]:
        angles = np.random.random(num_mcms)
        time_dyn = timeit.timeit(
            "single_qubit_stats(angles, shots=shots)", number=rep, globals=globals()
        )
        time_defer = timeit.timeit(
            "qml.defer_measurements(single_qubit_stats)(angles, shots=shots)",
            number=rep,
            globals=globals(),
        )
        print(
            f"{shots:4d} shots and {num_mcms:2d} MCMs took   "
            f"{time_dyn/rep:.6f} sec.  |    {time_defer/rep:.6f} sec."
        )

######################################################################
# As anticipated, the QNode using ``dynamic_one_shot`` takes much longer when increasing
# the shot count signficantly, whereas the QNode using ``defer_measurements`` does not
# show any difference in performance. In contrast, the number of MCMs extends the runtime
# of the former QNode only linearly due to the additional circuit depth, whereas the
# latter QNode jumps from milliseconds to seconds of compute time.
# When running circuits with MCMs, keep this difference in strengths and weaknesses
# in mind, and choose your method wisely!
#
# .. _end of copied part:
#
# Postprocessing mid-circuit measurements within a QNode
# ==========================================================
#
# Final measurements in PennyLane (such as expectation values and variances of observables,
# probability estimates and samples) can not be post-processed within a ``QNode`` but need to
# be processed separately. For MCMs, a number of unary and binary operators are supported
# even within ``QNode``\ s. This allows us to return modified statistics directly from the node:
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
# Now consider an approximate version of the above circuit (``qml.Hadamard(0)`` replaced by
# the approximate Hadamard ``[qml.RY(np.pi / 2 - eps, 0), qml.X(0)]``  and ``qml.CNOT(wires)``
# by ``[qml.CRX(np.pi + eps, wires), qml.S(wires[0])]``).
# Above, the three returned values were essentially equivalent, or at least we interpreted
# them  as such. The approximate circuit below produces results that deviate from perfect
# correlation between all four qubits. As they are quite unlikely, still, we make sure to
# see them by switching to ``1000`` shots, and to keep track of what the ``QNode`` returns,
# we make use of ``qml.counts``, which groups the samples conveniently.
#


@qml.defer_measurements  # Faster for few qubits and MCMs
@qml.qnode(dev)
def processed_mcms_approximated(eps):
    qml.RY(np.pi / 2 + eps, 0)
    qml.X(0)
    for wires in ([0, 1], [0, 2], [0, 3]):
        qml.CRX(np.pi + eps, wires)
        qml.S(wires[0])

    mcms = [qml.measure(w) for w in range(4)]
    prod = np.prod([2 * mcm - 1 for mcm in mcms])
    equality = (mcms[0] == mcms[1]) & (mcms[0] == mcms[2]) & (mcms[0] == mcms[3])
    sum_ = sum(mcms)
    return qml.counts(prod), qml.counts(equality), qml.counts(sum_)


print(processed_mcms_approximated(0.6, shots=1000))

######################################################################
# Note that the first two returned ``counts`` disagree on the number of samples that
# violate the perfect correlation: ``qml.counts(prod)`` claims there were ``54`` misses,
# ``qml.counts(equality)`` detected ``58``. The third returned counter, ``qml.counts(sum_)``
# reveals the problem: ``4`` circuit executions had two ``1``\ s and two ``0``\ s,
# leading to the same parity as ``0000`` and ``1111``. The samples of ``prod`` are unable
# to detect this deviation and it incorrectly reports these four samples as perfectly
# correlated. The explicit construction of ``equality``, in contrast, makes sure it only
# counts perfectly correlated measurements of the four qubits.
#
# Supported postprocessing
# ========================
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
# ================================
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
#
# "KILLER APP"
#
######################################################################
######################################################################
