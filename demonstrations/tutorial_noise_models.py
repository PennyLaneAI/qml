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
# the noise model to perform noisy simulations.
#

######################################################################
# Conditionals
# ~~~~~~~~~~~~
#
# We implement conditions as Boolean functions that accept an operation and evaluate it
# to return a Boolean output. In PennyLane, such objects are referred to as conditionals
# and are constructed as instances of :class:`~.pennylane.BooleanFn` and can be combined
# using standard bitwise operations such as ``&``, ``|``, ``^``, or ``~``. We support
# the following types of conditionals:
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
# gate operations with :math:`|\phi| < 1.0` and wires :math:`\in \{1, 2\}`:
#

import pennylane as qml
import numpy as np

@qml.BooleanFn
def rx_cond(op):
    return isinstance(op, qml.RX) and np.abs(op.parameters[0]) < 1.0

# Combine this arbitrary conditional with a wire-based conditional
rx_and_wires_cond = rx_cond & qml.noise.wires_in([0, 1])
for op in [qml.RX(0.05, wires=[0]), qml.RX(2.34, wires=[1])]:
    print(f"Result for {op}: {rx_and_wires_cond(op)}")

######################################################################
# Callables
# ~~~~~~~~~
#
# Callables that apply noise operations are referred to as *noise functions* and have the
# signature ``fn(op, **metadata) -> None``. Their definition has no return statement and
# contains the error operations that are *inserted* when a gate operation in the circuit
# satisfies corresponding conditional. There are a few ways to construct noise functions:
#
# 1. **Single-instruction noise functions:** To add a single-operation noise, one can use
#    :func:`~pennylane.noise.partial_wires`. It performs a partial initialization of the
#    noise operation and queues it on the ``wires`` of the gate operation.
# 2. **User-defined noise functions:** For adding more sophisticated noise, one can define
#    their own quantum function with the signature specified above. This way, one can also
#    specify their own custom order for inserting the noise by queing the operation being
#    evaluated via :func:`~pennylane.apply` within the function definition.
#
# For example, one can use the following for inserting a depolarization error and test it
# with an example gate operation:
#

depol_error = qml.noise.partial_wires(qml.DepolarizingChannel, 0.01)

op = qml.X('w1') # Example operation
print(f"Error for {op}: {depol_error(op)}")

######################################################################
# Noise Models
# ~~~~~~~~~~~~
#
# We can now create a PennyLane :class:`~.pennylane.NoiseModel` by stitching together
# multiple condition-callable pairs, where noisy noise operations are inserted into the
# circuit when their corresponding given condition is satisfied. We will construct a
# noise model for a generic three-qubit quantum circuit. The first condition-callable
# pair we construct will mimic thermal relaxation errors that are encountered during
# state preparation:
#

fcond1 = qml.noise.op_eq(qml.StatePrep)

def noise1(op, **kwargs):
    for wire in op.wires:
        qml.ThermalRelaxationError(0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"], wire)

######################################################################
# By default, noise operations will be inserted *after* the gate that the noise function
# is evaluated over. However, we can circumvent this! For example, we can add a sandwiching
# constant-valued rotation errors for :class:`~.pennylane.Hadamard` gates on the wires
# :math:`\in \{0, 1\}`:
#

fcond2 = qml.noise.op_eq("Hadamard") & qml.noise.wires_in([0, 1])

def noise2(op, **kwargs):
    qml.RX(np.pi / 16, op.wires)
    qml.apply(op)
    qml.RY(np.pi / 8, op.wires)

######################################################################
# Next, we will insert a single-qubit depolarization error for the
# :class:`~.pennylane.RX` gates that act on the wires :math:`\in \{0, 1\}`.
# For this we can reuse the previously constructed ``rx_and_wires_cond``
# and ``depol_error``:
#

fcond3, noise3 = rx_and_wires_cond, depol_error

######################################################################
# And have one last pair for a two-qubit depolarization error for every
# :class:`~.pennylane.CNOT` and :class:`~.pennylane.CZ` gates:
#

from functools import reduce
from itertools import product

fcond4 = qml.noise.op_in(["CNOT", "CZ"])

pauli_mats = map(qml.matrix, [qml.I(0), qml.X(0), qml.Y(0), qml.Z(0)])
kraus_mats = [reduce(np.kron, prod, 1.0) for prod in product(pauli_mats, repeat=2)]
def noise4(op, **kwargs):
    probs = np.array([1 - kwargs["p"]] + [kwargs["p"] / 15] * 15).reshape((-1, 1, 1))
    qml.QubitChannel(np.sqrt(probs) * np.array(kraus_mats), op.wires)

######################################################################
# Finally, we can build the noise model with some required ``metadata``:
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
# the following circuit that performs the evolution and de-evolution
# of a given initial state based on some parameters:
#

from matplotlib import pyplot as plt

qml.drawer.use_style("pennylane")
dev = qml.device("default.mixed", wires=3)
init_state = np.random.RandomState(42).rand(2 ** len(dev.wires))
init_state /= np.linalg.norm(init_state)

def circuit(theta, phi):
    # State preparation
    qml.StatePrep(init_state, wires=[0, 1, 2])
    qml.Barrier(wires=[0, 1, 2])

    # Evolve state
    qml.Hadamard(0)
    qml.RX(theta, 1)
    qml.RX(phi, 2)
    qml.CNOT([1, 2])
    qml.CZ([0, 1])
    qml.Barrier(wires=[0, 1, 2])

    # De-evolve state
    qml.CZ([0, 1])
    qml.CNOT([1, 2])
    qml.RX(-phi, 2)
    qml.RX(-theta, 1)
    qml.Hadamard(0)
    return qml.state()

theta, phi = 0.21, 0.43
ideal_circuit = qml.QNode(circuit, dev)
qml.draw_mpl(ideal_circuit)(theta, phi)
plt.show()

######################################################################
# To attach the ``noise_model`` to this quantum circuit, we can do the following:
#

noisy_circuit = qml.add_noise(ideal_circuit, noise_model)
qml.draw_mpl(noisy_circuit)(theta, phi)
plt.show()

######################################################################
# We can then use these for running noisy simulations as shown below:
#

init_dm = np.outer(init_state, init_state) # density matrix for init_state
ideal_circ_fidelity = qml.math.fidelity(ideal_circuit(theta, phi), init_dm)
noisy_circ_fidelity = qml.math.fidelity(noisy_circuit(theta, phi), init_dm)

print(f"Ideal v/s Noisy: {ideal_circ_fidelity} and {noisy_circ_fidelity}")

######################################################################
# The fidelity for the state obtained from the ideal circuit is :math:`\approx 1.0`.
# We see that this is not the case for the result obtained from the noisy circuit,
# as expected.
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
# Should you have any questions about using noise models in PennyLane, you can consult the
# `noise module documentation <https://docs.pennylane.ai/en/stable/code/qml_noise.html>`__
# or create a post on the `PennyLane Discussion Forum <https://discuss.pennylane.ai>`__.
# You can also follow us on `X (formerly Twitter) <https://twitter.com/PennyLaneAI>`__
# or `LinkedIn <https://www.linkedin.com/company/pennylaneai/>`__ to stay up-to-date with
# the latest and greatest from PennyLane!
#

######################################################################
# About the author
# ----------------
