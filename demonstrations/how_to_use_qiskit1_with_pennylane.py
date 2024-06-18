r"""How to use Qiskit 1.0 with PennyLane
====================================

One of PennyLane’s claims to fame is that it’s *hardware agnostic*; PennyLane doesn’t care about
what you want to execute quantum circuits on. This is made possible by our `plugin
suite <https://pennylane.ai/plugins/#plugins>`__, which provides additional simulator and hardware devices and other functionality that are maintained
by us and the community around the clock 🤝. Everything from Julia backends (`PennyLane-Snowflurry
plugin <https://github.com/calculquebec/pennylane-snowflurry>`__) to IBM’s hardware devices
(`PennyLane-Qiskit plugin <https://docs.pennylane.ai/projects/qiskit/en/latest/>`__) are accessible
with PennyLane as your first point of contact.

On that note of IBM hardware, the PennyLane-Qiskit plugin enables you to integrate your existing
Qiskit code and run circuits on IBM devices with PennyLane, encompassing two real-world scenarios:
(1) working in PennyLane from the start and executing your work on an IBM device and (2) converting
your existing Qiskit code to PennyLane and executing that on *any* device, including IBM devices,
Amazon Braket — you name it!

With the first stable release of Qiskit in February 2024 (Qiskit 1.0), we subsequently shipped some
excellent features and upgrades with the PennyLane-Qiskit plugin, allowing
anyone familiar with
Qiskit (both 1.0 and pre-1.0 versions) to jump into the PennyLane ecosystem and land on both feet.
In this demo, we want to show you how easy these features are to use, letting you make that
aforementioned jump like it’s nothing 😌.
"""

######################################################################
# .. note ::
#
#    To follow along, we recommend installing the PennyLane-Qiskit plugin by doing
#    ``pip install -U pennylane-qiskit`` in your desired environment. This will install PennyLane, the
#    plugin, and the latest Qiskit 1.0 version.
#

######################################################################
# Coding in PennyLane, executing on IBM Quantum devices 📤
# -------------------------------------------------------
#
# If you want to distill how a PennyLane plugin works down to one thing, it’s all in the provided devices! In
# PennyLane, you just :doc:`create your circuit (a quantum function) </introduction/circuits>` and decorate it with
# the QNode decorator :func:`@qml.qnode(dev) <~.qnode>`, where ``dev`` is (one of) the plugin’s device(s).
#
# In PennyLane and its plugins,
# devices are called upon by their short name, and can be loaded via the :func:`~.device` function:
#
#   .. code-block:: python
#
#     qml.device("shortname", *device_options)
#
# If you’ve
# seen PennyLane code before, you’ve probably seen ``"default.qubit"`` or ``"lightning.qubit"`` as
# short names for our Python and C++ statevector simulators, respectively.
#
# In the PennyLane-Qiskit plugin, there are `several IBM
# devices <https://docs.pennylane.ai/projects/qiskit/en/stable/#devices>`__ you can use, but here are
# two heavy-hitters for Qiskit 1.0:
#
# -  ``"qiskit.basicsim"``: uses the Qiskit ``BasicSimulator`` backend from the ``basic_provider``
#    module in Qiskit 1.0.
# -  ``"qiskit.remote"``: lets you run PennyLane code on any Qiskit device, where you can choose between
#    different backends - either simulators tailor-made to emulate the real hardware, or the real
#    hardware itself.
#
# If you want to use any of these devices in PennyLane, simply put those short names into
# ``qml.device`` and any quantum function decorated with ``@qml.qnode(dev)`` will execute on the
# corresponding device.
#

######################################################################
# Let’s say we have this simple circuit in Qiskit 1.0.
#

import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator

qc = QuantumCircuit(2, 1)

qc.h(0)
qc.cx(0, 1)
qc.measure(1, 0)

backend = BasicSimulator()
counts = backend.run(qc).result().get_counts()

print(counts)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     {'0': 523, '1': 501}
#

######################################################################
# In PennyLane, we can execute the exact same circuit on the exact same device and backend like so:
#

import pennylane as qml

dev = qml.device("qiskit.basicsim", wires=2, shots=1024)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=1)

print(circuit())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     {'0': tensor(474, requires_grad=True), '1': tensor(550, requires_grad=True)}
#

######################################################################
# Magic 🪄! With one line of code, you can work inside PennyLane and ship the execution off to your
# favourite IBM device. It’s exactly like using Qiskit 1.0, but you interact with PennyLane instead.
#

######################################################################
# Converting Qiskit 1.0 code to PennyLane 🐛→🦋
# ---------------------------------------------
#
# This is probably what a lot of the audience is wondering: “Can I combine my existing work in Qiskit
# with PennyLane?” YES. And don’t worry, you don’t need to import a ton of things or use a bunch of
# functions — you only need to know *two* things:
#
# 1. `qml.from_qiskit <https://docs.pennylane.ai/en/stable/code/api/pennylane.from_qiskit.html>`__:
#    converts an entire Qiskit ``QuantumCircuit`` — the whole thing — into a PennyLane quantum
#    function. It will faithfully convert Qiskit-side measurements (even mid-circuit measurements), or
#    you can append Pennylane-side measurements directly to it.
# 2. :func:`~.from_qiskit_op`:
#    converts a ``SparsePauliOp`` in Qiskit 1.0 to the equivalent operator in PennyLane.
#
# Both of these functions give you all the functionality you need to access PennyLane’s features and
# user-interface starting from the Qiskit 1.0 side. Let’s look at an example where both of these
# functions are used.
#

######################################################################
# Let’s say you’ve created the following Qiskit code that prepares a modified GHZ state for an
# arbitrary amount of qubits, measures several expectation values of ``SparsePauliOp`` operators.
#

from qiskit.quantum_info import SparsePauliOp

n = 5

def qiskit_GHZ_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
        qc.rx(0.1967, i)
    return qc

qc = qiskit_GHZ_circuit(n)

operator_strings = ["I" * i + "ZZ" + "I" * (n - 2 - i) for i in range(n - 1)]
operators = [SparsePauliOp(operator_string) for operator_string in operator_strings]
print(operators)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     [SparsePauliOp(['ZZIII'],
#                   coeffs=[1.+0.j]), SparsePauliOp(['IZZII'],
#                   coeffs=[1.+0.j]), SparsePauliOp(['IIZZI'],
#                   coeffs=[1.+0.j]), SparsePauliOp(['IIIZZ'],
#                   coeffs=[1.+0.j])]
#

from qiskit.primitives import StatevectorEstimator

estimator = StatevectorEstimator()
job_estimator = estimator.run([(qc, operators)])
result_estimator = job_estimator.result()[0].data.evs

print(result_estimator)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     [0.98071685 0.96180554 0.96180554 0.96180554]
#

######################################################################
# To convert this work into PennyLane, let’s start with the Qiskit-side ``SparsePauliOp`` operators
# and converting them to PennyLane objects with ``qml.from_qiskit_op``.
#

pl_operators = [qml.from_qiskit_op(qiskit_op) for qiskit_op in operators]
print(pl_operators)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     [Z(3) @ Z(4), Z(2) @ Z(3), Z(1) @ Z(2), Z(0) @ Z(1)]
#

######################################################################
# .. note ::
#
#    PennyLane wires are enumerated from left to right, while the Qiskit convention is to
#    enumerate from right to left. This means a ``SparsePauliOp`` term defined by the string “XYZ”
#    applies ``Z`` on wire ``0``, ``Y`` on wire ``1``, and ``X`` on wire ``2``. For more details, see
#    the `String <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Pauli>`__ representation
#    section of the Qiskit documentation for the ``Pauli`` class.
#

######################################################################
# Next, we convert the Qiskit ``QuantumCircuit``, ``qc``, to PennyLane with ``qml.from_qiskit``. We
# can append the measurements — expectation values (:func:`~.expval`) of ``pl_operators`` — with the
# ``measurements`` keyword argument, which accepts a list of PennyLane measurements.
#

measurements = [qml.expval(op) for op in pl_operators]  # expectation values

qc = qiskit_GHZ_circuit(n)
pl_qfunc = qml.from_qiskit(qc, measurements=measurements)

######################################################################
# The last thing to do is make ``pl_func`` a QNode. We can’t decorate ``pl_qfunc`` with
# ``@qml.qnode``, but we can equivalently wrap it with ``qml.QNode`` and supply the device.
#

pl_circuit = qml.QNode(pl_qfunc, device=qml.device("lightning.qubit", wires=n))
pl_circuit()

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     [0.9807168489852623, 0.961805537883582, 0.961805537883582, 0.961805537883582]
#

######################################################################
# What’s really useful about being able to append measurements to the end of a circuit with
# :func:`~.from_qiskit` is being able to measure something that isn’t available in Qiskit 1.0 but is
# available in PennyLane, like the classical shadow measurement protocol, for example. In PennyLane,
# you can measure this with :func:`~.classical_shadow`.
#

measurements = [qml.classical_shadow(wires=range(n))]
pl_qfunc = qml.from_qiskit(qc, measurements=measurements)

pl_circuit = qml.QNode(pl_qfunc, device=qml.device("default.qubit", wires=n))
pl_circuit(shots=5)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     [array([[[0, 0, 0, 0, 1],
#              [1, 1, 1, 0, 1],
#              [0, 1, 0, 0, 0],
#              [1, 1, 0, 0, 0],
#              [0, 0, 0, 0, 0]],
#
#             [[0, 0, 2, 2, 1],
#              [2, 0, 2, 0, 1],
#              [1, 0, 2, 0, 2],
#              [1, 0, 1, 1, 1],
#              [1, 0, 2, 0, 2]]], dtype=int8)]
#

######################################################################
# And that’s it! Now you have a copy of your work in PennyLane, where you can access
# fully-differentiable and hardware-agnostic quantum programming — the entire quantum programming
# ecosystem is at your disposal 💪.
#

######################################################################
# A real-world example 🌎
# -----------------------
#
# One of the things you’ll almost certainly encounter in the wild is a cost function to optimize with
# tunable parameters belonging to a circuit — it’s common 😉! PennyLane is a great option for these
# problems because of its end-to-end differentiability and long list of optimization methods you can
# leverage. So, maybe you have a home-cooked variational circuit written in Qiskit and you want to
# access PennyLane’s seamless differentiability — yep, the PennyLane-Qiskit plugin has you covered
# here, too 🙌.
#
# To keep things simple, let’s use the following Qiskit circuit as our variational circuit.
#

from qiskit.circuit import ParameterVector, Parameter

n = 3

angles1 = ParameterVector("phis", n - 1)
angle2 = Parameter("theta")

qc = QuantumCircuit(3)
qc.rx(angles1[0], [0])
qc.ry(angles1[1], [1])
qc.ry(angle2, [2])

qc.draw("mpl")

######################################################################
# .. figure:: ../_static/demonstration_assets/how_to_use_qiskit_1_with_pennylane/qiskit_parameterized_circuit.png
#     :align: left
#     :width: 20%
#

######################################################################
# This circuit contains two sets of differentiable parameters: ``phis`` (length 2) and ``theta``
# (scalar).
#
# If we give this Qiskit circuit to ``qml.from_qiskit``, we get a quantum function that can
# subsequently be called within a circuit — it’s as if the gates and operations contained within it
# get transferred over to our new QNode.
#

import pennylane.numpy as np

pl_qfunc = qml.from_qiskit(qc)

dev = qml.device("lightning.qubit", wires=n)

@qml.qnode(dev)
def differentiable_circuit(phis, theta):
    pl_qfunc(phis, theta)
    return [qml.expval(qml.Z(i)) for i in range(n)]

phis = np.array([0.6, 0.7])
theta = np.array([0.19])

print(differentiable_circuit(phis, theta))
print(qml.draw(differentiable_circuit)(phis, theta))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     [0.8253356149096783, 0.7648421872844883, 0.9820042351172701]
#     0: ──RX(0.60)─┤  <Z>
#     1: ──RY(0.70)─┤  <Z>
#     2: ──RY(0.19)─┤  <Z>
#

######################################################################
# You’ll notice, too, that ``pl_func`` has the call signature that you would expect:
# ``pl_qfunc(phis, theta)``, just like the name we gave them when we defined them in Qiskit
# (``ParameterVector("phis", n-1)`` and ``Parameter("theta")``).
#
# With the circuit being immediately differentiable to PennyLane, let’s define a dummy cost function
# that will sum the output of ``differentiable_circuit``.
#

def cost(phis, theta):
    return np.sum(differentiable_circuit(phis, theta))

######################################################################
# Now we’re in business and can optimize! You can use any suitable optimizer in PennyLane that you
# like, and by calling its ``step_and_cost`` method you can access the updated parameters and cost
# function value after each optimization step.
#

opt = qml.AdamOptimizer(0.1)

for i in range(100):
    (phis, theta), new_loss = opt.step_and_cost(cost, phis, theta)

print(f"Optimized parameters: phis = {phis}, theta = {theta}")
print(f"Optimized cost function value: {new_loss}")

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     Optimized parameters: phis = [3.12829384 3.12823583], theta = [3.1310224]
#     Optimized cost function val: -2.999796472821245
#

######################################################################
# As we expect, the minimum value that our cost function can take is :math:`-3` when all angles of
# rotation are :math:`\pi` (all qubits are rotated into the :math:`\vert 1 \rangle` state). Of course,
# this is just a toy example of an easy optimization problem. But, you can apply this process to a
# larger, more complicated circuit ansatze without worries.
#

######################################################################
# Further resources 📓
# ====================
#
# There’s so much more to learn about what’s possible in PennyLane, and if you’re coming from Qiskit
# you’re in good hands! The PennyLane-Qiskit plugin is your personal chaperone to the PennyLane
# ecosystem. You can dive deeper into what’s possible with the PennyLane-Qiskit plugin by visiting the
# `plugin homepage <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__. In the
# mean time, if you have any questions about the plugin, PennyLane, or even Qiskit, drop a question on
# our `Discussion Forum <https://discuss.pennylane.ai>`__ and we’ll promptly respond.
#
# Now that you’ve used PennyLane, every road in the wonderful world of quantum programming SDKs is
# open with no set speed limits 🏎️. Visit our website, `pennylane.ai <https://pennylane.ai>`__ to see
# the latest and greatest PennyLane features, demos, and blogs, and follow us on
# `LinkedIn <https://www.linkedin.com/company/pennylaneai/>`__ or `X (formerly
# Twitter) <https://twitter.com/PennyLaneAI>`__ to stay updated!
#

######################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/isaac_de_vlugt.txt
#
