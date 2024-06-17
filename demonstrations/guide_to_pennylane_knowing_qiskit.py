r"""Your Guide to PennyLane if you know Qiskit
===================================================

Greetings fellow quantum programmers 👋! Isn’t it such a wonderful world we live in with so many
different quantum software development kits (SDKs) at our fingertips? I remember the days of (messy)
personalized code bases that took hours and hours to develop just *to be able* to start researching
👴. Nowadays, those tools are there for us to access for free, being developed and maintained by
savvy open-source software developers around the clock. Again, what a wonderful world!

When it comes to quantum programming SDKs, PennyLane and Qiskit (``v1.0`` and ``<=v0.46``) are two
of the most widely-used by the community. PennyLane has a few staples that make it so:

1. **Hardware agnostic**: PennyLane has no opinions on what hardware or simulator backends you want
   to use for your research. You can program an emporium of real hardware and simulator backends all
   from the easy-to-use PennyLane API. This includes IBM’s hardware with the `PennyLane-Qiskit
   plugin <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__.
   
2. **Everything is differentiable**: A quantum circuit in PennyLane is designed to behave like a
   differentiable function, unlocking quantum differentiable programming and allowing to integrate
   seamlessly with your favourite machine learning frameworks.
   
3. **Community-focused**: Let’s face it, you’re going to get stuck at some point when you’re
   researching or learning. That’s why we have a mandate to make our documentation easy to navigate,
   dedicated teams for creating new demonstrations when we release new features, and an active
   discussion forum for answering your questions.

"""

######################################################################
# In this demo, we're going to demonstrate to you the fundamentals of what makes PennyLane awesome with
# the idea in mind that you're coming from Qiskit. If you want to follow along on your computer, you’ll
# need to install PennyLane, the PennyLane-Qiskit plugin, and a couple extras:
# .. code-block:: console
#
#     pip install pennylane pennylane-qiskit torch tensorflow
#
# (the plugin will install Qiskit-1.0). Now,
# let’s get started.
#

######################################################################
# PennyLane 🤝 Qiskit
# -------------------
#
# With the first stable release of Qiskit in February 2024 — `Qiskit 1.0 <https://github.com/Qiskit/qiskit>`__ —
# breaking changes were to be expected. Although learning a new language can be hard, PennyLane has very
# simple tools that will help via the PennyLane-Qiskit plugin: your gateway to the land of PennyLane
# that even lets you keep your existing Qiskit work without even having to know a ton about how PennyLane
# works.
#
# There are two functions you need to know about:
#
# 1. :func:`~.from_qiskit`: converts an entire Qiskit ``QuantumCircuit`` — the whole thing — into
#    PennyLane. It will faithfully convert Qiskit-side measurements (even mid-circuit measurements) or
#    you can append PennyLane-side measurements directly to it.
#
# 2. :func:`~pennylane.from_qiskit_op`: converts a ``SparsePauliOp`` in Qiskit 1.0 to the equivalent operator in
#    PennyLane.
#
# These two functions give you all that you need to access PennyLane’s features and user-interface
# starting from the Qiskit side. As an example, let’s say you have the following code in Qiskit that
# prepares a Bell state.
#

from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

######################################################################
# To convert this circuit to PennyLane, you just use ``qml.from_qiskit``:
#

import pennylane as qml

pl_func = qml.from_qiskit(qc)
print(qml.draw_mpl(pl_func, style="pennylane")())

######################################################################
# .. figure:: ../_static/demonstration_assets/guide_to_pennylane_knowing_qiskit/pennylane_bell_circuit.png
#     :align: left
#     :width: 20%
#

######################################################################
# Want to measure some expectation values of Pauli operators, as well? Use ``qml.from_qiskit_op`` to
# convert a ``SparsePauliOp`` into PennyLane’s equivalent operator.
#

from qiskit.quantum_info import SparsePauliOp

qiskit_pauli_op = SparsePauliOp("XY")
pl_pauli_op = qml.from_qiskit_op(qiskit_pauli_op)

######################################################################
# Then, you can *append* the expectation value measurement — done with ``qml.expval`` — to the PennyLane
# circuit when you create it with ``qml.from_qiskit``:
#

pl_func = qml.from_qiskit(qc, measurements=[qml.expval(pl_pauli_op)])
print(qml.draw_mpl(pl_func)())

######################################################################
# .. figure:: ../_static/demonstration_assets/guide_to_pennylane_knowing_qiskit/pennylane_bell_circuit_measure.png
#     :align: left
#     :width: 20%
#

######################################################################
# And just like that, you’re in PennyLane land! Now you might be asking: “What is ``pl_func`` and how
# do I use it further?” To answer those questions, we need to get to know PennyLane a little better.
#

######################################################################
# Get to Know PennyLane 🌞
# ------------------------
#
# Let’s go back to that Qiskit circuit (``qc``) that we created earlier. If we want to execute that
# circuit in Qiskit and get some results, we can do this:
#

from qiskit.primitives import StatevectorSampler

qc.measure_all()

sampler = StatevectorSampler()

job_sampler = sampler.run([qc], shots=1024)
result_sampler = job_sampler.result()[0].data.meas.get_counts() # or call get_counts()

print(result_sampler)

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     {'11': 530, '00': 494}
#

######################################################################
# When we use ``qml.from_qiskit`` on our Qiskit circuit, this is equivalent to creating this function
# in PennyLane.
#

def pl_func():
    """Equivalent to doing pl_func = qml.from_qiskit(qc)"""
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=[0, 1])

######################################################################
# .. note ::
#
#    Qubits in PennyLane are called “wires”. Why? As was mentioned, PennyLane is a hardware agnostic
#    framework, and “wires” is a hardware agnostic term for quantum degrees of freedom that
#    quantum computers can be based off of.
#
#

######################################################################
# A function like ``pl_func`` is called a **quantum function**. A quantum function in PennyLane just
# contains quantum gates and (optionally) returns a measurement. Measurements in PennyLane are quite
# different than in Qiskit 1.0 — we'll touch on how measurements work in PennyLane shortly. But, in our
# case, :func:`qml.counts(wires=[0, 1]) <pennylane.counts>` is the measurement, which counts the number 
# of times each basis state is sampled.
#
# If we actually want to execute the circuit and see the result of our measurement, we need to define
# what the circuit runs on, just like how in Qiskit when we defined a ``Sampler`` instance. PennyLane’s
# way of doing this is simple: (1) define a device with :func:`qml.device <pennylane.device>` and (2) pair the device with
# the quantum function with :func:`~pennylane.QNode`.
#

dev = qml.device("default.qubit", shots=1024)
pl_circuit = qml.QNode(pl_func, dev)

print(pl_circuit())

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#   .. code-block::
#
#     {'00': tensor(510, requires_grad=True), '11': tensor(514, requires_grad=True)}
#

######################################################################
# Now that we have the full picture of how a circuit gets created and executed in PennyLane, let’s take
# a step back and summarize what’s going on. The first thing you’ll notice is that PennyLane’s primitives
# are Pythonic and array-like; quantum circuits are *functions*, returning measurements that behave
# like NumPy arrays. The function ``pl_circuit`` is called a *quantum node* (QNode), which is the sum
# of two things:
#
# 1. **A quantum function that contains quantum instructions**. This is ``pl_func``, which just contains
#    quantum operations (gates) and returns a measurement. In this case, ``qml.counts(wires=1)`` is
#    the measurement, which counts the number of times each basis state is sampled and returns a dictionary 
#    whose values are NumPy arrays.
# 2. **A device** (e.g., ``qml.device("default.qubit")``). PennyLane has many devices you can choose from, but
#    ``"default.qubit"`` is our battle-tested Python statevector simulator.
#
# As for measurements in PennyLane, they are quite different from Qiskit's ``EstimatorV2`` and ``SamplerV2``
# primitives. :doc:`PennyLane's measurement API <introduction/measurements>`
# comprises ergonomic functions that a QNode can return, like :func:`~.state` (returns the quantum state),
# :func:`~.probs` (returns the probability distribution of the quantum state), and :func:`~.expval` (returns
# the expectation value of a provided operator).
#
# All of this allows for a QNode to be called like a regular Python function, executing on the device
# you specified and returning the measurement you asked for — simple as that 🌈.
#
# Alternatively, because PennyLane’s primitive is a glorified Python function, wrapping a quantum
# function with :func:`qml.QNode <pennylane.QNode>` is the same as *decorating* it with :func:`@qml.qnode(dev) <pennylane.qnode>`:
#

@qml.qnode(dev)
def pl_circuit():
    """Equivalent to doing pl_circuit = qml.QNode(qml.from_qiskit(qc), dev)"""
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=1)

######################################################################
# A minor point, but both approaches work.
#
# What's great about converting your work in Qiskit to PennyLane is that now you have access to all
# of PennyLane's plugins, meaning you can run your Qiskit circuit on more than just IBM hardware! All
# you need to do is install the plugin of interest, and change the name of the device in ``qml.device``.
#

######################################################################
# Further resources 📓
# --------------------
#
# There’s so much more to learn about what’s possible in PennyLane, and if you’re coming from Qiskit
# you’re in good hands. The PennyLane-Qiskit plugin is your personal chaperone to the PennyLane
# ecosystem. You can dive deeper into what’s possible with the PennyLane-Qiskit plugin by visiting the
# `plugin homepage <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__.
#
# Another great thing about the PennyLane ecosystem is that we have an `emporium of up-to-date
# demos <https://pennylane.ai/search/?contentType=DEMO&sort=publication_date>`__ maintained by the
# same people that develop PennyLane. If you’re just starting out, I recommend reading our :doc:`qubit
# rotation tutorial <tutorial_qubit_rotation>`.
#
# Now that you’ve used PennyLane, every road in the wonderful world of quantum programming SDKs is
# open with no set speed limits 🏎️. If you have any questions about the PennyLane-Qiskit plugin,
# PennyLane, or even Qiskit, drop a question on our `Discussion
# Forum <https://discuss.pennylane.ai>`__ and we’ll promptly respond. You can also keep exploring our website,
# `pennylane.ai <https://pennylane.ai>`__, to see the latest and greatest PennyLane features, demos,
# and blogs, or follow us on `LinkedIn <https://www.linkedin.com/company/pennylaneai/>`__ or `X
# (formerly Twitter) <https://twitter.com/PennyLaneAI>`__ to stay updated!
#

######################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/isaac_de_vlugt.txt
#
