r"""How to use noise models in PennyLane
------------------------------------
"""

######################################################################
# Noise models are essential in understanding and describing the effects of physical errors in a
# quantum computation. They allow one to perform simulations that help emulate imperfections in the
# evolution of and inference from quantum states under the effect of state preparation, measurement
# and environment-based errors.
#
# Here, we introduce how to use the PennyLane :mod:`~.pennylane.noise` module to construct and
# manipulate noise models for noisy simulation. PennyLane supports what is called insertion-based
# noise models that are defined as a mapping from conditions to quantum function-like
# callables and any additional noise-related metadata. Each condition in the mapping evaluates gate
# operations in the quantum circuit based on some of its attributes like name, parameter values or
# wires. Depending on the outcome of the evaluation, the corresponding callable would apply the
# noise operations for the evaluated gate based on user-provided metadata such as hardware topology
# constraints and relaxation/dephasing times.
#

######################################################################
# .. figure:: ../_static/demonstration_assets/noise_models/noise_model_short.jpg
#    :align: center
#    :width: 100%
#
#    ..
#
# Let’s understand this briefly with the help of the above noise model that inserts
# amplitude and phase damping errors for :class:`~.pennylane.RX` and :class:`~.pennylane.RY`
# gates, respectively. It will transform the following circuit by inserting the appropriate
# error operation after each operation that evaluates positively for any of its conditions.
#

######################################################################
# .. figure:: ../_static/demonstration_assets/noise_models/noise_model_long.jpg
#    :align: center
#    :width: 90%
#
#    ..
#

######################################################################
# In the subsequent sections, we will explore how much flexibility is available in defining the
# underlying components for these noise models and then learn how to construct them in PennyLane.
# Finally, we will see how they can be used in noisy simulations.
#

######################################################################
# Boolean functions
# ~~~~~~~~~~~~~~~~~
#
# We want to implement conditions as Boolean functions called `Conditionals` that accept
# an operation and evaluate it to return a boolean output. In PennyLane, such objects are
# constructed as instances of :class:`~.pennylane.BooleanFn` and can combined using standard
# bitwise operations. From the perspective of noise models, we support the following types
# of constructions.
#

######################################################################
# Operation-based conditionals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# These conditionals evaluate an operation based on whether it is a specific type of operation or
# belongs to a specified set of operations. These can be built with :class:`~.pennylane.noise.op_eq`
# and :class:`~.pennylane.noise.op_in` helper methods, where the specific set of operations can be
# provided as a class, instantiated object or by their string representation:
#

import pennylane as qml

cond1 = qml.noise.op_eq("X")
cond2 = qml.noise.op_in(["X", qml.Y, qml.CNOT([0, 1])])

print(f"cond1: {cond1}")
print(f"cond2: {cond2}")

######################################################################

op = qml.Y(0)
print(f"Evaluating conditionals for {op}")
print(f"Result for cond1: {cond1(op)}")
print(f"Result for cond2: {cond2(op)}")

######################################################################
# Wire-based conditionals
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Unlike operation-based conditionals, these evaluate an operation based on whether its
# wires are equal or belong to a specified set of wires. These can be built with
# :class:`~.pennylane.noise.wires_eq` and :class:`~.pennylane.noise.wires_in` helper methods,
# where the specific set of wires can be provided as string or integer inputs. Additionally,
# one can use already built gate operations, and their wires will be automatically extracted
# for constructing the wire set:
#

cond3 = qml.noise.wires_eq("aux")
cond4 = qml.noise.wires_in([0, "c", qml.RX(0.123, wires=["w1"])])

print(f"cond3: {cond3}")
print(f"cond4: {cond4}")

######################################################################

op = qml.X("c")
print(f"Evaluating conditionals for {op}")
print(f"Result for cond3: {cond3(op)}")
print(f"Result for cond4: {cond4(op)}")

######################################################################
# Arbitrary conditionals
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Further precise control over the evaluation of operations can be achieved by defining custom
# conditionals in a functional form and wrapping thema  with :class:`~.pennylane.BooleanFn`
# decorator. Inputs of such custom conditionals must be a single operation for evaluation and
# optional keyword arguments for accepting metadata. For example, a conditional that evaluates
# ``True`` for :math:`RX(\phi)` gate operations with :math:`\phi < 1.0` can be constructed as:
#


@qml.BooleanFn
def rx_condition(op, **metadata):
    return isinstance(op, qml.RX) and op.parameters[0] < 1.0


######################################################################

op1, op2, op3 = qml.RX(0.05, wires=0), qml.RY(0.07, wires=2), qml.RX(2.37, wires="a")

for op in [op1, op2, op3]:
    print(f"Result for {op}: {rx_condition(op)}")

######################################################################
# Combined Conditionals
# ^^^^^^^^^^^^^^^^^^^^^
#
# As noted, the conditionals described above can be combined to form a new conditional via
# bitwise operations such as ``&``, ``|``, ``^``, or ``~``. The resulting conditional will
# evaluate the expression in the order of their combination:
#

and_cond = cond2 & cond4
print(and_cond)

######################################################################

op1, op2, op3 = qml.X(wires=0), qml.CNOT(wires=[2, 3]), qml.RY(0.23, wires="c")

for op in [op1, op2, op3]:
    print(f"Result for {op}: {and_cond(op)}")

######################################################################
# Noisy quantum functions
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# If a conditional evaluates to ``True`` on a given gate operation in the quantum circuit, the
# corresponding quantum function is evaluated and that inserts the error operations for it. One
# can use a helper constructor :func:`~pennylane.noise.partial_wires` function for single
# operation noise insertion or define their own custom quantum function. It should be noted
# that user-defined functions should have the signature ``fn(op, **metadata)``, allowing for
# dependency on both the evaluated operation and metadata specified in the noise model.
#

######################################################################
# Single-instruction noise via partial_wires
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :func:`~pennylane.noise.partial_wires` builds a partially evaluated function based on
# the given operation with all arguments frozen except wires. For example, this can be
# used to create a constant-valued over-rotation:
#

rx_constant = qml.noise.partial_wires(qml.RX(0.1, wires=[0]))

for wire in [0, 2, "w1"]:
    print(f"Over-rotation for {wire}: {rx_constant(wire)}")

######################################################################
# Insertion-based noise
# ^^^^^^^^^^^^^^^^^^^^^
#
# Often, one would want more parameterized control over the noise operations being applied.
# This is possible by defining your own quantum function with the signature we specified
# above. For example, one can use the following for inserting a thermal relaxation error
# based on :math:`T_1` time provided at runtime as a keyword argument (i.e. metadata).
#


def thermal_func(op, **kwargs):
    qml.ThermalRelaxationError(0.4, kwargs["t1"], 0.2, 0.6, op.wires)


for op, t1 in [(qml.X(0), 0.01), (qml.RZ(1.234, "w1"), 0.02), (qml.S("aux"), 0.03)]:
    with qml.queuing.AnnotatedQueue() as q:
        thermal_func(op, t1=t1)
    print(f"Error for {op}: {q.queue[0]}")

######################################################################
# Custom-ordered noise
# ^^^^^^^^^^^^^^^^^^^^
#
# By default, the quantum function defined above would result in an error being inserted after
# the operation that evaluates to ``True``. This might not always be ideal, and there might be
# instances where the error needs more flexibility with the order. One can attain this flexibility
# by specifying the order on their own and queuing the operation being evaluated via
# :func:`~pennylane.apply` within the function definition. For example, the following
# noise function sandwiches the target operation within phase rotations.
#


def sandwich_func(op, **kwargs):
    qml.RZ(op.parameters[0] * 0.05, op.wires)
    qml.apply(op)
    qml.RZ(-op.parameters[0] * 0.05, op.wires)


for op, t1 in [(qml.RX(6.589, 0), 0.01), (qml.RY(4.237, "w1"), 0.02)]:
    with qml.queuing.AnnotatedQueue() as q:
        sandwich_func(op, t1=t1)
    print(f"Error for {op}: {q.queue}")

######################################################################
# Noise Models
# ~~~~~~~~~~~~
#
# Now that we have introduced all the main characters of our story, we can finally
# introduce them in a single plot, i.e., use them to build a noise model in PennyLane
# as a :class:`~.pennylane.NoiseModel` object as given below:
#

# Set up the conditions
fcond1 = qml.noise.op_eq(qml.RX)


@qml.BooleanFn
def fcond2(op, **kwargs):
    return isinstance(op, qml.RY) and op.parameters[0] >= 0.5


fcond3 = qml.noise.op_in(["X", "Y"])

# Set up noisy functions
noise1 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)


def noise2(op, **kwargs):
    qml.RZ(op.parameters[0] * 0.05, op.wires)
    qml.apply(op)
    qml.RZ(-op.parameters[0] * 0.05, op.wires)


def noise3(op, **kwargs):
    qml.ThermalRelaxationError(0.4, kwargs["t1"], kwargs["t2"], 0.6, op.wires)


# Set up noise model
noise_model = qml.NoiseModel({fcond1: noise1, fcond2: noise2, fcond3: noise3}, t1=0.04, t2=0.01)
print(noise_model)

######################################################################
# Adding noise models to your workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The final remaining ingredient of our story is to learn how to use these noise models.
# So, a noise model once built can be attached to a circuit or device via the
# :func:`~pennylane.add_noise` transform. For example, consider the following quantum circuit:
#

from matplotlib import pyplot as plt

qml.drawer.use_style("pennylane")
dev = qml.device("default.mixed", wires=2)


def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RX(params[3], wires=1)
    return qml.counts(wires=[0, 1])


qcircuit = qml.QNode(circuit, dev)
params = [0.1, 0.7, 0.8, 0.4]
qml.draw_mpl(qcircuit, decimals=2)(params)
plt.show()

######################################################################
# Now to attach the noise model to this quantum circuit, we can do the following:
#

noisy_circuit = qml.transforms.add_noise(qcircuit, noise_model)
qml.draw_mpl(noisy_circuit, decimals=2)(params)
plt.show()

######################################################################
# Alternatively, one can also attach the noise model instead to the device itself instead of
# transforming the circuit. For this we can again use the :func:`~pennylane.add_noise` transform:
#

noisy_dev = qml.transforms.add_noise(dev, noise_model)
noisy_dev_circuit = qml.QNode(circuit, noisy_dev)

######################################################################
# We can check that both of these are equivalent by looking at their outcomes:
#

import numpy as np

num_shots = 100000
ideal_circuit_counts = qcircuit(params, shots=num_shots)
noisy_circuit_counts = noisy_circuit(params, shots=num_shots)
noist_dev_counts = noisy_dev_circuit(params, shots=num_shots)
categories = list(noisy_circuit_counts.keys())

width = 0.2
bars1 = np.arange(len(categories))
bars2 = bars1 + width
bars3 = bars1 + 2 * width

counts_ideal = np.array(list(ideal_circuit_counts.values())) / num_shots
counts_ncirc = np.array(list(noisy_circuit_counts.values())) / num_shots
counts_ndevc = np.array(list(noist_dev_counts.values())) / num_shots

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(bars1, counts_ideal, width, label="Ideal Results")
plt.bar(bars2, counts_ncirc, width, label="Noisy Circuit")
plt.bar(bars3, counts_ndevc, width, label="Noisy Device")

# Add labels, title, and legend
plt.xlabel("Bitstring")
plt.ylabel("Probabilities")
plt.xticks(bars1 + width / 2, categories)
plt.legend()

# Show the plot
plt.show()


######################################################################
# Conclusion
# ~~~~~~~~~~
#
# Noise models provide a succinct way to describe the behaviour of the environment on
# quantum computation. In PennyLane, we define such models as mapping conditionals that
# select the target operation and their corresponding noise operations. These can be
# constructed with utmost flexibility, as shown above. We also support converting
# noise models from Qiskit, including the ones from hardware and fake backends to
# PennyLane via our `pennylane-qiskit <https://docs.pennylane.ai/projects/qiskit/en/latest/>`__
# plugin. As such models are instrumental in capturing the working of quantum hardware,
# we will continue to improve these features in PennyLane, allowing one to develop and
# test error correction and noise mitigation strategies that will ultimately pave the
# way towards practical and reliable quantum computations.
#

######################################################################
# About the author
# ----------------
