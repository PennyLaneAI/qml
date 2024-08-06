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
# 1. Boolean conditions that dictate whether or not noise is inserted into the circuit.
# 2. Callables that apply noise operations when a corresponding condition is satisfied.
#
# The following example shows how a noise model transforms a sample circuit by inserting
# amplitude and phase damping errors for :class:`~.pennylane.RX`
# and :class:`~.pennylane.RY` gates, respectively.
#

######################################################################
# .. figure:: ../_static/demonstration_assets/noise_models/noise_model_long.jpg
#    :align: center
#    :width: 85%
#
#    ..
#

######################################################################
# In the upcoming sections, we will first cover the underlying components of
# noise models and learn how to use them to construct one. Finally, we will use
# that noise model to perform noisy simulations.
#

######################################################################
# Conditionals
# ~~~~~~~~~~~~
#
# We implement conditions as Boolean functions called `Conditionals` that accept an
# operation and evaluate it to return a Boolean output. In PennyLane, such objects are
# constructed as instances of :class:`~.pennylane.BooleanFn` and can be combined using
# standard bitwise operations such as ``&``, ``|``, ``^``, or ``~``. We support the
# following types of conditionals:
#
# 1. **Operation-based conditionals:** They evaluate a gate operation based on whether it
#    is a specific type of operation or belongs to a specified set of operations. They are
#    built using the :func:`~.pennylane.noise.op_eq` and :func:`~.pennylane.noise.op_in`.
# 2. **Wire-based conditionals:** They evaluate a gate operation based whether if
#    its wires are equal or belong to a specified set of wires. They are built using the
#    :func:`~.pennylane.noise.wires_eq` and :func:`~.pennylane.noise.wires_in`.
# 3. **Arbitrary conditionals:** Custom conditionals can be defined as a function wrapped
#    with a :class:`~.pennylane.BooleanFn` decorator. Signature for such conditionals must
#    be ``cond_fn(operation: Operation) -> bool``.
#
# For example, here's how we would define a conditional that checks for :math:`R_X(\phi)`
# gate operations with :math:`\phi < 1.0` and wires :math:`\in\, \{0, a\}`:
#

import pennylane as qml

@qml.BooleanFn
def rx_cond(op):
    return isinstance(op, qml.RX) and op.parameters[0] < 1.0

# Combine this arbitrary conditional with a wire-based conditional
cond_fn = rx_cond & qml.noise.wires_in([0, "a"])
for op in [qml.RX(0.05, wires=0), qml.RX(2.37, wires="a")]:
    print(f"Result for {op}: {cond_fn(op)}")

######################################################################
# Callables
# ~~~~~~~~~
#
# Callables that apply noise operations are referred to as noise functions and have the
# signature ``fn(op, **metadata) -> None``, allowing for dependency on both the evaluated
# operation and specified metadata. Their definition has no return statement and contains
# the error operations that are inserted when a gate operation in the circuit satisfies
# the corresponding conditional. We support the following construction of noise functions:
#
# 1. **Single-instruction noise functions:** To add a single-operation noise, one can use
#    :func:`~pennylane.noise.partial_wires`. It performs a partial initialization of the
#    noise operation and queues it on the ``wires`` of the gate operation.
# 2. **User-defined noise functions:** For adding more sophesticated noise, one can define
#    their own quantum function with the signature specified above. This way, one can also
#    specify their own custom order for inserting the noise by queing the operation being
#    evaluated via :func:`~pennylane.apply` within the function definition.
#
# For example, one can use the following for inserting a thermal relaxation error based on
# a :math:`T_1` time provided as a keyword argument and see the error that gets queued in
# the circuit using an example gate operation:
#

def thermal_func(op, **kwargs): # Noise Function
    qml.ThermalRelaxationError(0.4, kwargs["t1"], 0.2, 0.6, op.wires)

op = qml.X(0) # Example operation
with qml.queuing.AnnotatedQueue() as q:
    thermal_func(op, t1=0.01)

print(f"Error for {op}: {q.queue[0]}")

######################################################################
# Noise Models
# ~~~~~~~~~~~~
#
# We can now create a PennyLane :class:`~.pennylane.NoiseModel` by stitching together
# multiple condition-callable pairs that inserts noise operations into the circuit if
# the given condition is satisfied. We will construct a noise model for performing a
# noisy `swap test <https://en.wikipedia.org/wiki/Swap_test>`__ for the two single-qubit
# states. For this purpose, the first pair we construct is to mimick the thermal
# relaxation errors encountered during the state preparation based on ``metadata``
# containing the dephasing and relaxation times for the qubits and the gate time of
# the operation:
#

fcond1 = qml.noise.op_eq(qml.StatePrep)

def noise1(op, **kwargs):
    qml.ThermalRelaxationError(0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"], op.wires)

######################################################################
# The next pair we construct is to add a sandwiching constant-valued rotation
# errors for :class:`~.pennylane.Hadamard` gates on the wires :math:`\in \{1, 2\}`:
#

fcond2 = qml.noise.op_eq("Hadamard") & qml.noise.wires_in([1, 2])

def noise2(op, **kwargs):
    qml.RX(np.pi / 16, op.wires)
    qml.apply(op)
    qml.RY(np.pi / 8, op.wires)

######################################################################
# Another pair for adding a depolarization error for every :class:`~.pennylane.T`
# and :class:`~.pennylane.PhaseShift` gates on the wires :math:`\in \{1, 2\}`:
#

fcond3 = qml.noise.op_in([qml.T, qml.PhaseShift]) & qml.noise.wires_in([1, 2])
noise3 = qml.noise.partial_wires(qml.DepolarizingChannel, 0.01)

######################################################################
# And one last pair for a two-qubit depolarization errors for every
# :class:`~.pennylane.CNOT` gate:
#

import numpy as np
from functools import reduce
from itertools import product

fcond4 = qml.noise.op_eq("CNOT")

pauli_mats = map(qml.matrix, [qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)])
kraus_mats = list(reduce(np.kron, prod, 1.0) for prod in product(pauli_mats, repeat=2))
def noise4(op, **kwargs):
    probs = np.array([1 - kwargs["p"]] + [kwargs["p"] / 15] * 15).reshape(-1, 1, 1)
    qml.QubitChannel(np.sqrt(probs) * np.array(kraus_mats), op.wires)

######################################################################
# Finally, we build the noise model with some required ``metadata``:
#

metadata = dict(t1=0.02, t2=0.03, tg=0.001, p=0.01)  # times unit: sec
noise_model = qml.NoiseModel(
    {fcond1: noise1, fcond2: noise2, fcond3: noise3, fcond4: noise4}, **metadata
)
print(noise_model)

######################################################################
# Adding noise models to your workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have built our noise model, we can learn how to use it.
# A noise model can be applied to a circuit or device via the
# :func:`~pennylane.add_noise` transform. For example, consider
# the following `swap test <https://en.wikipedia.org/wiki/Swap_test>`__
# circuit that compares two single-qubit states prepared based on parameters:
#

from matplotlib import pyplot as plt

qml.drawer.use_style("pennylane")
dev = qml.device("default.mixed", wires=3)
# gives a single-qubit statevector based on a parameter "param"
state = lambda param: np.array([np.cos(-param / 2), 1j * np.sin(-param / 2)])

def swap_test(theta, phi):
    # State preparation
    qml.StatePrep(state(theta), wires=[1])
    qml.StatePrep(state(phi), wires=[2])
    qml.Barrier([0, 1, 2])

    # Swap test with decomposed Fredkin gate
    qml.Hadamard(0)
    qml.CNOT([1, 2])
    qml.Hadamard(1)
    qml.CNOT([1, 2])
    qml.PhaseShift(-np.pi/4, 2)
    qml.CNOT([0, 2])
    qml.T(2)
    qml.CNOT([1, 2])
    qml.PhaseShift(-np.pi/4, 2)
    qml.CNOT([0, 2])
    qml.T(1)
    qml.CNOT([0, 1])
    qml.T(0)
    qml.PhaseShift(-np.pi/4, 1)
    qml.CNOT([0, 1])
    qml.Hadamard(0)

    return qml.expval(qml.Z(0))

swap_circuit = qml.QNode(swap_test, dev)
qml.draw_mpl(swap_circuit)(0.2, 0.3)
plt.show()

######################################################################
# To attach the ``noise_model`` to this quantum circuit, we can do the following:
#

noisy_circuit = qml.add_noise(swap_circuit, noise_model)
print(qml.draw(noisy_circuit, decimals=3, max_length=80)(0.2, 0.3))

######################################################################
# Alternatively, one can also attach the noise model instead to the device itself instead of
# transforming the circuit. For this we can again use the :func:`~pennylane.add_noise` transform:
#

noisy_dev = qml.add_noise(dev, noise_model)
noisy_dev_circuit = qml.QNode(swap_test, noisy_dev)

######################################################################
# We can then use these for running noisy simulations as shown below:
#

theta, phi = np.pi / 3, np.pi / 3
ideal_circ_res = swap_circuit(theta, phi)
noisy_circ_res = noisy_circuit(theta, phi)
noisy_qdev_res = noisy_dev_circuit(theta, phi)

print(f"Ideal v/s Noisy: {ideal_circ_res} and {noisy_circ_res}")
print(f"Noisy Circuit v/s Noisy Device: {noisy_circ_res} and {noisy_qdev_res}")

######################################################################
# Since, both the parameters are equal, the ideal result for the swap test
# is :math:`\approx 1.0`. We see that this is not the case for the
# result obtained from the noisy circuit. Moreover, by looking at the
# closeness of the two noisy results we can confirm the equivalence of the
# two ways the noise models could be added for noisy simulations.
#


######################################################################
# Conclusion
# ~~~~~~~~~~
#
# Noise models provide a succinct way to describe the impact of the environment on
# quantum computation. In PennyLane, we define such models as mapping between conditionals
# that select the target operation and their corresponding noise operations. These
# can be constructed with utmost flexibility, as shown above.
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
