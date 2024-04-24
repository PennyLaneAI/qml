r"""How to collect statistics of mid-circuit measurements
=========================================================

Measuring qubits in the middle of a quantum circuit execution can be useful in many ways.
From understanding the inner workings of a circuit, hardware characterization,
modeling and error mitigation, to error correction, algorithmic improvements and even up to full
computations encoded as measurements in measurement-based quantum computation (MBQC).

Before turning to any of these advanced topics, it is worthwhile to familiarize ourselves with
the syntax and features around mid-circuit measurements (MCMs). In this how-to, we will focus on
extracting statistics about measurements that are performed while a quantum circuit is up and
running --- mid-circuit measurement statistics!

.. figure:: ../_static/demonstration_assets/how_to_collect_mcm_stats/socialthumbnail_large_how_to_collect_mcm_stats.png
    :align: center
    :width: 50%


"""

######################################################################
# Defining the circuit ansatz
# ---------------------------
#
# We start with standard imports, setting a randomness seed, and specifying
# wires. As we will treat the first wire different than all other wires, we
# define it as separate variable.
#

import pennylane as qml
import numpy as np

np.random.seed(511)

first_wire = 0
other_wires = [1, 2, 3]
num_wires = 1 + len(other_wires)

######################################################################
# Now we create a quantum circuit ansatz that switches between a layer of
# simple rotation gates (:class:`~.pennylane.RX`),
# mid-circuit measurements(:func:`~.pennylane.measure`),
# and a layer of entangling two-qubit gates (:class:`~.pennylane.CNOT`)
# between the first and all other qubits.
#


def ansatz(x):
    mcms = []

    # Rotate all qubits
    for w, x_ in enumerate(x):
        qml.RX(x_, w)

    # Measure first qubit
    mcms.append(qml.measure(first_wire))

    # Entangle all qubits with first qubit
    for w in other_wires:
        qml.CNOT([first_wire, w])

    # Measure and reset all qubits but the first
    for w in other_wires:
        mcms.append(qml.measure(w))

    return mcms


"""
    # Rotate all but the first qubit and apply CNOTs with first qubit
    for w, x_ in enumerate(x):
        qml.RX(-x_, w)

    # Entangle all qubits with first qubit
    for w in other_wires:
        qml.CNOT([w, first_wire])

    # Measure all qubits but the first
    mcms3 = [qml.measure(w) for w in other_wires]

    # Change measurement basis of first qubit
    qml.Hadamard(first_wire)

    # Measure first qubit and postselect on measuring a 1
    mcm4 = qml.measure(first_wire, postselect=1)
"""

######################################################################
# A quantum circuit with basic MCM statistics
# -------------------------------------------
#
# Before we post-process the mid-circuit measurements in this
# ansatz or expand the ansatz itself, let's construct a simple
# :class:`~.pennylane.QNode` and look at the statistics of the four
# performed MCMs; We compute the probability vector for the MCM
# on the first qubit and count the bit strings sampled from the other
# three MCMs.
# To implement the ``QNode``, we also define a shot-based qubit device.
#

dev = qml.device("default.qubit", shots=100)


@qml.qnode(dev)
def simple_node(x):
    mcm1, *mcms2 = ansatz(x)
    return qml.probs(op=mcm1), qml.counts(mcms2)


######################################################################
# Before executing the circuit, let's draw it! For this, we sample some random
# parameters, one for each qubit, and call the Matplotlib drawer
# :func:`~.pennylane.draw_mpl`.
#

x = np.random.random(num_wires)
fig, ax = qml.draw_mpl(simple_node)(x)
import matplotlib.pyplot as plt

plt.show()

######################################################################
# Neat, let's move on to executing the circuit:
#

probs, counts = qml.defer_measurements(simple_node)(x)
print(f"Probability vector of first qubit MCM: {np.round(probs, 5)}")
print(f"Bit string counts on other qubits: {counts}")

######################################################################
# We see that the first qubit has a probability of about :math:`23.3\%`
# to be in the state :math:`|1\rangle` after the rotation.
# We also observe that we only sampled bit strings from the other three
# qubits for which the second and third bit are identical.
# (Quiz question: Is this an analytic probability or just because we
# did not sample enough?)
#
# Note that we applied the ``defer_measurements`` transform above.
# This is for reproducibility reasons and is not required in general.
#
# Post-processing mid-circuit measurements
# ----------------------------------------
# We now set up a more interesting ``QNode``. It executes the ``ansatz``
# from above twice and compares the obtained MCMs (note that we did not
# define ``comparing_function`` yet, we will get to that shortly):
#


@qml.qnode(dev)
def interesting_qnode(x):
    first_mcms = ansatz(x)
    second_mcms = ansatz(-x)
    output = comparing_function(first_mcms, second_mcms)
    return qml.counts(output)


######################################################################
# Before we can run this more interesting ``QNode``, we need to actually
# specify the ``comparing_function``. We ask the following question:
# Is the measurement on the first qubit equal between the two sets of MCMs,
# and do the other three measured values have the same parity?
#
# In contrast to quantum measurements at the end of a :class:`~.pennylane.QNode`,
# PennyLane supports a number of unary and binary operators for MCMs even *within*
# ``QNode``\ s. This enables us to phrase the question above as a boolean function.
# Consider the
# `introduction on measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation if you want to learn more about the supported operations.
#


def comparing_function(first_mcms, second_mcms):
    """A function that compares two sets of MCM outcomes."""
    equal_first = first_mcms[0] == second_mcms[0]
    # Computing the parity can be done with bitwise "and" with the number 1
    first_parity = sum(first_mcms[1:]) & 1
    second_parity = sum(second_mcms[1:]) & 1
    equal_parity = first_parity == second_parity
    return equal_first & equal_parity


######################################################################
# Now we can run the ``QNode`` and obtain the statistics for our comparison function:

print(qml.defer_measurements(interesting_qnode)(x))

######################################################################
# We find that our question is answered with "yes" and "no" roughly equally often.
# Turning up the number of shots lets us compute this ratio more precisely:
#

num_shots = 10000
counts = qml.defer_measurements(interesting_qnode)(x, shots=num_shots)
p_yes = counts[True] / num_shots
p_no = counts[False] / num_shots
print(f'The probability to answer with "yes" / "no" is {p_yes:.5f} / {p_no:.5f}')

######################################################################
# **This concludes our how-to on statistics and post-processing of
# mid-circuit measurements.**
#
# Additional information
# ----------------------
#
# Below, you can find some complementary information
# on the supported return types with (post-processed) MCMs.
# For more details consider the
# `introduction on measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html#mid-circuit-measurements-and-conditional-operations>`_
# and the documentation of :func:`~.pennylane.measure`.
# For performance considerations, take a look at
# :func:`~.pennylane.defer_measurements` and :func:`~.pennylane.dynamic_one_shot`.
#
# Supported MCM return types
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Depending on the processing applied to the MCM results, not all return types are supported.
# For example, ``qml.probs(2 * mcm0)`` is not a valid return value, because it is not clear
# which probabilities are being requested.
# Furthermore, as usual the available return types depend on whether or not the device is
# shot-based (``qml.sample`` can not be returned if the device is not sampling).
# Overall, **all combinations of post-processing and all of**
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
# As we saw in the ``QNode`` above, MCM statistics can be returned alongside
# standard terminal measurements.
#
