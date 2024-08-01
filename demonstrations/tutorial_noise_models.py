r"""How to use noise models in PennyLane
------------------------------------
"""

######################################################################
# Noise models are essential for understanding and describing the effects of physical errors
# in a quantum computation, as they allow for simulating the imperfections in state evolution
# arising from environment-based errors, state preparation routines, measurements, and more.
#
# Here, we show how to use the features provided in PennyLane's :mod:`~.pennylane.noise`
# module to construct and manipulate noise models for enabling noisy simulation. In PennyLane,
# noise models are constructed from two main components:
#
# 1. Conditions that help select gate operations by evaluating them based on some of their
#    specific attributes.
# 2. Callables that apply corresponding noise operations for the selected gates using some
#    optional metadata.
#
# The following example shows how a noise model transforms a sample circuit by inserting
# amplitude and phase damping errors for :class:`~.pennylane.RX`
# and :class:`~.pennylane.RY` gates, respectively.
#

######################################################################
# .. figure:: ../_static/demonstration_assets/noise_models/noise_model_long.jpg
#    :align: center
#    :width: 90%
#
#    ..
#

######################################################################
# In the upcoming sections, we further explore the underlying components
# of noise models, learn how to construct them and use them in noisy
# simulations.
#

######################################################################
# Conditionals
# ~~~~~~~~~~~~
#
# We implement conditions as Boolean functions called `Conditionals` that accept an
# operation and evaluate it to return a Boolean output. In PennyLane, such objects are
# constructed as instances of :class:`~.pennylane.BooleanFn` and can be combined using
# standard bitwise operations. For the noise models, we support the following types of
# conditionals.
#

######################################################################
# Operation-based conditionals
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# These conditionals evaluate a gate operation based on its ``type``. They check whether
# it is a specific type of operation or belongs to a specified set of operations. These
# conditionals can be built with the :func:`~.pennylane.noise.op_eq` and
# :func:`~.pennylane.noise.op_in` helper methods:
#

import pennylane as qml

cond1 = qml.noise.op_eq("RX")
cond2 = qml.noise.op_in([qml.RX, qml.RY, qml.CNOT([0, 1])])

print(f"cond1: {cond1}")
print(f"cond2: {cond2}\n")

op = qml.RY(1.23, wires=[0])
print(f"Evaluating conditionals for {op}")
print(f"Result for cond1: {cond1(op)}")
print(f"Result for cond2: {cond2(op)}")

######################################################################
# Wire-based conditionals
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# These conditionals evaluate an operation based on its wires by checking if the wires
# are equal or belong to a specified set of wires. These conditionals can be built with
# the :func:`~.pennylane.noise.wires_eq` and :func:`~.pennylane.noise.wires_in` helper methods,
# where the specific set of wires can be provided as string or integer inputs:
#

cond3 = qml.noise.wires_eq("aux")
cond4 = qml.noise.wires_in([0, "c", qml.RX(0.123, wires=["w1"])])

print(f"cond3: {cond3}")
print(f"cond4: {cond4}\n")

op = qml.X("c")
print(f"Evaluating conditionals for {op}")
print(f"Result for cond3: {cond3(op)}")
print(f"Result for cond4: {cond4(op)}")

######################################################################
# Arbitrary conditionals
# ^^^^^^^^^^^^^^^^^^^^^^
#
# More precise control over the evaluation of operations can be achieved by defining custom
# conditionals in a functional form and wrapping them with a :class:`~.pennylane.BooleanFn`
# decorator. Inputs of such custom conditionals must be a single operation for evaluation.
# For example, a conditional that checks for :math:`R_X(\phi)` gate operations with
# :math:`\phi < 1.0` can be constructed as:
#

@qml.BooleanFn
def rx_condition(op):
    return isinstance(op, qml.RX) and op.parameters[0] < 1.0

op1, op2, op3 = qml.RX(0.05, wires=0), qml.RY(0.07, wires=2), qml.RX(2.37, wires="a")

for op in [op1, op2, op3]:
    print(f"Result for {op}: {rx_condition(op)}")

######################################################################
# Combined Conditionals
# ^^^^^^^^^^^^^^^^^^^^^
#
# The conditionals described above can be combined to form a new conditional
# using bitwise operations such as ``&``, ``|``, ``^``, or ``~``. The resulting
# conditional will evaluate the expression in the order of their combination:
#

and_cond = cond2 & cond4
print(f"and_cond: {and_cond}\n")

op1, op2, op3 = qml.RX(0.42, wires=0), qml.CNOT(wires=[2, 3]), qml.RY(0.23, wires="c")

for op in [op1, op2, op3]:
    print(f"Result for {op}: {and_cond(op)}")

######################################################################
# Callables
# ~~~~~~~~~
#
# Now, let's see how the callables that apply noise operations are defined. They are referred
# to as noise functions, whose definition contains the error operations to be inserted but
# has no return statements. If a conditional evaluates to ``True`` on a given gate operation,
# the noise function is evaluated and its operations are inserted to the circuit. Each noise
# function, will have the signature ``fn(op, **metadata)``, allowing for dependency on both
# the evaluated operation and specified metadata. For the noise models, we support following
# constructions of noise functions.
#

######################################################################
# Single-instruction noise functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For adding a single-instruction noise, one can use :func:`~pennylane.noise.partial_wires`,
# which builds a partially evaluated function based on the given operation with all arguments
# frozen except wires. For example, it can be used to create a callable that instantiate
# a constant-valued over-rotation based on the input wire:
#

rx_constant = qml.noise.partial_wires(qml.RX, 0.1)

for wire in [0, 2, "w1"]:
    print(f"Over-rotation for {wire}: {rx_constant(wire)}")

######################################################################
# User-defined noise functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# More parameterized control over the noise operation being applied is possible by
# defining your own quantum function with the signature we specified above.
# For example, one can use the following for inserting a thermal relaxation error
# based on a :math:`T_1` time provided as a keyword argument (i.e., metadata).
#

def thermal_func(op, **kwargs):
    qml.ThermalRelaxationError(0.4, kwargs["t1"], 0.2, 0.6, op.wires)

######################################################################
# Now to see what error would this noise-function would queue in the circuit using
# an example gate operation:
#

op = qml.X(0)
with qml.queuing.AnnotatedQueue() as q:
    thermal_func(op, t1=0.01)

print(f"Error for {op}: {q.queue[0]}")

######################################################################
# Custom-ordered noise functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# By default, noise functions defined above would insert error operations after the
# evaluated gate operation. More flexibility with the insertion order can be achieved
# by specifying one's own custom order by queuing the operation being evaluated via
# :func:`~pennylane.apply` within the function definition as shown below:
#

def sandwich_func(op, **kwargs):
    qml.RZ(op.parameters[0] * 0.05, op.wires)
    qml.apply(op)
    qml.RZ(-op.parameters[0] * 0.05, op.wires)

######################################################################
# This noise function sandwiches the target operation within some phase rotations as
# we can see below:
#

op = qml.RX(6.58, wires=[0])
with qml.queuing.AnnotatedQueue() as q:
    sandwich_func(op)

print(f"Error for {op}:\n{q.queue}")

######################################################################
# Noise Models
# ~~~~~~~~~~~~
#
# Now that we have introduced all the main ingredients for the noise mdoels,
# we can finally stitch them together to build a PennyLane noise model
# as a :class:`~.pennylane.NoiseModel` object. First we set up the conditionals
#

fcond1 = qml.noise.op_eq(qml.RX)

@qml.BooleanFn
def fcond2(op, **kwargs):
    return isinstance(op, qml.RY) and op.parameters[0] >= 0.5

fcond3 = qml.noise.op_in(["X", "Y"])

######################################################################
# Next, we set up the noise functions:
#

noise1 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

def noise2(op, **kwargs):
    qml.RZ(op.parameters[0] * 0.05, op.wires)
    qml.apply(op)
    qml.RZ(-op.parameters[0] * 0.05, op.wires)

def noise3(op, **kwargs):
    qml.ThermalRelaxationError(0.4, kwargs["t1"], kwargs["t2"], 0.6, op.wires)

######################################################################
# Finally, we build the noise model with some metadata:
#

noise_model = qml.NoiseModel(
    {fcond1: noise1, fcond2: noise2, fcond3: noise3}, t1=0.04, t2=0.01
)
print(noise_model)

######################################################################
# Adding noise models to your workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have seen how to build a noise model, we can learn how to use them.
# A noise model can be utilized for a circuit or device via the :func:`~pennylane.add_noise`
# transform. For example, consider the following circuit:
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
# To attach the ``noise_model`` to this quantum circuit, we can do the following:
#

noisy_circuit = qml.add_noise(qcircuit, noise_model)
qml.draw_mpl(noisy_circuit, decimals=2)(params)
plt.show()

######################################################################
# Alternatively, one can also attach the noise model instead to the device itself instead of
# transforming the circuit. For this we can again use the :func:`~pennylane.add_noise` transform:
#

noisy_dev = qml.add_noise(dev, noise_model)
noisy_dev_circuit = qml.QNode(circuit, noisy_dev)

######################################################################
# We can then use these for running noisy simulations as shown below:
#

import numpy as np

num_shots = 100000
ideal_circ_res = qcircuit(params, shots=num_shots)
noisy_circ_res = noisy_circuit(params, shots=num_shots)
noisy_qdev_res = noisy_dev_circuit(params, shots=num_shots)

size = 0.2
bars = np.arange(len(noisy_qdev_res))
keys = list(noisy_qdev_res.keys())
labels = ["Ideal Results", "Noisy Circuit", "Noisy Device"]

# Create the bar plot
plt.figure(figsize=(10, 4))
for idx, res in enumerate([ideal_circ_res, noisy_circ_res, noisy_qdev_res]):
    counts = np.array(list(res.values())) / num_shots
    plt.bar(bars + idx*size, counts, size, label=labels[idx])

# Add labels, title, and legend
plt.xlabel("Bitstring")
plt.ylabel("Probabilities")
plt.xticks(bars + size / 2, keys)
plt.legend()
plt.show()

######################################################################
# By looking at the closeness of the two noisy results we can confirm
# the equivalence of the two ways the noise models could be added for
# noisy simulations.
#


######################################################################
# Conclusion
# ~~~~~~~~~~
#
# Noise models provide a succinct way to describe the impact of the environment on
# quantum computation. In PennyLane, we define such models as mapping between conditionals
# that select the target operation and their corresponding noise operations. These can be
# constructed with utmost flexibility, as shown above.
#
# As such models are instrumental in capturing the working of quantum hardware,
# we will continue to improve these features in PennyLane, allowing one to
# develop and test error correction and noise mitigation strategies that will
# ultimately pave the way towards practical and reliable quantum computations.
# We encourage users to keep a track of them in the documentation for the
# `noise module <https://docs.pennylane.ai/en/stable/code/qml_noise.html>`__.
#

######################################################################
# About the author
# ----------------
