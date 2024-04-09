r"""How to collect statistics of mid-circuit measurements
=========================================================

Measuring qubits in the middle of a quantum circuit execution can be useful in many ways.
From understanding the inner workings of a circuit, over hardware characterization,
modeling and error mitigation, to error correction, algorithmic improvements and even up to full
computations encoded as measurements in measurement-based quantum computation (MBQC).

Before turning to any of these advanced topics, it is worthwhile to familiarize ourselves with
the syntax and features around mid-circuit measurements. In this how-to, we will focus on
dynamic quantum circuits that use control flow based on mid-circuit measurements.
Most of the advanced concepts mentioned above incorporate MCMs in this way, making it a
key ingredient to scalable quantum computing.

.. figure:: ../_static/demonstration_assets/how_to_collect_mcm_stats/socialthumbnail_large_how_to_create_dynamic_mcm_circuits.png
    :align: center
    :width: 50%

If you are interested in how to collect statistics about performed mid-circuit measurements
in PennyLane, also check out our
:doc:`how-to on collecting MCM stats </demos/tutorial_how_to_collect_mcm_stats>`.
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
#
# .. _end of copied part:
#
# NEXT HEADER
# ===========
