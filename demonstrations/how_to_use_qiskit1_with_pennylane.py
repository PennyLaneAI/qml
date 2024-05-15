r"""Greetings fellow quantum programmers 👋! Isn’t it such a wonderful world we live in with so many
different quantum software development kits (SDKs) at our fingertips? I remember the days of (messy)
personalized code bases that took hours and hours to develop just *to be able* to start researching
👴. Nowadays, those tools are there for us to access for free, being developed and maintained by
savvy open-source software developers around the clock. Again, what a wonderful world.

When it comes to quantum programming SDKs, PennyLane and
`Qiskit <https://pennylane.ai/qml/glossary/what-is-qiskit/>`__ are two of the most widely-used by
the community. One of PennyLane’s beloved staples is that it’s hardware agnostic; you can program
any piece of hardware all from the easy-to-use PennyLane API. This includes IBM’s hardware with the
`PennyLane-Qiskit plugin <https://docs.pennylane.ai/projects/qiskit/en/latest/>`__, which has been
maintained and developed by us since 2018.

Continuing with those efforts, the PennyLane-Qiskit plugin was upgraded immediately after the first
stable release of Qiskit in February 2024 — `Qiskit 1.0 <https://github.com/Qiskit/qiskit>`__ —
which shipped with breaking changes from the ``<0.46`` releases. In this demonstration, we want to
show you how we’ve made PennyLane the best its ever been at integrating with Qiskit and interacting
with IBM’s hardware.
"""

######################################################################
# New to PennyLane? 👀
# ====================
# 
# If you’re already very familiar with PennyLane and its ecosystem, you’re free to move on to the next
# section. But, if you’re coming from a Qiskit-heavy background, you will find this section very
# helpful. Here, we’ll go over some basic syntax differences between PennyLane and Qiskit 1.0.
# 
# First, qubits. Qubits in PennyLane are called “wires”. Why? As was mentioned already, PennyLane is
# a hardware agnostic framwork, and “wires” is a more hardware agnostic term for quantum degrees of
# freedom that quantum computers can be based off of. Done! 
# 
# Now let’s break down what circuits look like in PennyLane compared to Qiskit. For that, let’s create
# a simple circuit with 2 qubits that prepares the Bell state
# :math:`\vert \Phi^+ \rangle = \frac{1}{\sqrt{2}} \left( \vert 00 \rangle + \vert 11 \rangle \right)`.
# 

######################################################################
# .. raw:: html
# 
#    <table>
# 
# .. raw:: html
# 
#    <tr>
# 
# .. raw:: html
# 
#    <th>
# 
# PennyLane
# 
# .. raw:: html
# 
#    </th>
# 
# .. raw:: html
# 
#    <th>
# 
# Qiskit
# 
# .. raw:: html
# 
#    </th>
# 
# .. raw:: html
# 
#    </tr>
# 
# .. raw:: html
# 
#    <tr>
# 
# .. raw:: html
# 
#    <td>
# 
# .. code:: python
# 
#    import pennylane as qml
# 
#    dev = qml.device("default.qubit", wires=2)
# 
#    @qml.qnode(dev)
#    def circuit():
#        qml.Hadamard(0)
#        qml.CNOT([0, 1])
#        return qml.counts(wires=1)
# 
#    print(circuit(shots=1000))
#    print(qml.draw_mpl(circuit, style='pennylane')())
# 
# .. raw:: html
# 
#    </td>
# 
# .. raw:: html
# 
#    <td>
# 
# .. code:: python
# 
#    import qiskit
#    from qiskit import QuantumCircuit, transpile
#    from qiskit.providers.basic_provider import BasicSimulator
# 
#    qc = QuantumCircuit(2, 1)
# 
#    qc.h(0)
#    qc.cx(0, 1)
#    qc.measure(1, 0)
# 
#    backend = BasicSimulator()
#    tqc = transpile(qc, backend)
#    counts = backend.run(tqc).result().get_counts()
# 
#    print(counts)
#    qc.draw(output='mpl')
# 
# .. raw:: html
# 
#    </td>
# 
# .. raw:: html
# 
#    </tr>
# 
# .. raw:: html
# 
#    <tr>
# 
# .. raw:: html
# 
#    <td>
# 
# .. code:: pycon
# 
#    {'0': tensor(478, requires_grad=True), 
#     '1': tensor(522, requires_grad=True)}
# 
# .. raw:: html
# 
#    </td>
# 
# .. raw:: html
# 
#    <td>
# 
# .. code:: pycon
# 
#    {'1': 497, '0': 527}
# 
# .. raw:: html
# 
#    </td>
# 
# .. raw:: html
# 
#    </tr>
# 
# .. raw:: html
# 
#    </table>
# 

######################################################################
# The first thing you’ll notice is that PennyLane’s primitives — its foundational building blocks —
# are Pythonic and NumPy-like; quantum circuits are functions that return measurements that behave
# like NumPy arrays. The function ``circuit`` is called a *quantum node* (QNode), which is the sum of
# a couple things:
# 
# 1. A *quantum function* that contains quantum instructions. This is ``circuit`` without the
#    ``@qml.qnode(dev)`` decorator; it just contains quantum operations like gates and returns a
#    measurement (in this case, ``qml.counts(wires=1)``, which counts the number of times a ``0`` or a
#    ``1`` occurs and returns a dictionary).
# 2. A device: ``qml.device("default.qubit")``. ``"default.qubit"`` is the name of our battle-tested
#    statevector simulator.
# 
# When you put a quantum function together with a device by decorating ``circuit`` with
# ``@qml.qnode(dev)``, you have a QNode! The QNode can be called like a regular Python function —
# simple as that 🌈. 
#
# This section is definitely a non-exhaustive look at what PennyLane can do. But, hopefully inspecting
# the differences between Qiskit and PennyLane through this simple example is enough to understand how
# PennyLane roughly works. If you want to learn more about PennyLane, we have an arsenal of amazing
# learning material on our website: https://pennylane.ai/qml/#all-content. If you’re just starting
# out, I recommend `our introductory
# tutorial <https://pennylane.ai/qml/demos/tutorial_qubit_rotation/>`__.
# 
# Let’s get to how you can use the PennyLane-Qiskit plugin.
# 

######################################################################
# The PennyLane-Qiskit plugin 💾
# ==============================
# 
# .. note::
#    To follow along, we recommend installing the PennyLane-Qiskit plugin by doing
#    ``pip install -U pennylane-qiskit`` in your desired environment. This will install PennyLane, the
#    plugin, and the latest pre-1.0 Qiskit version (``0.46.1`` at the time of writing). Luckily, the
#    plugin is compatible with *both* Qiskit 1.0 and pre-1.0. So, you can ``pip install -U qiskit`` in
#    your environment afterwards to have the plugin running alongside Qiskit 1.0. The plugin knows
#    what to do 🧠.
# 
# As was mentioned earlier, the PennyLane-Qiskit plugin allows you to integrate your existing Qiskit
# code and run jobs on IBM devices with PennyLane. This encompasses two real-world scenarios: (1)
# working in PennyLane from the start and executing your work on an IBM device and (2) converting your
# existing Qiskit code to PennyLane and executing that on *any* device, including IBM devices. Let’s
# talk about both.
# 

######################################################################
# Coding in PennyLane, executing on Qiskit 1.0 devices 📤
# -------------------------------------------------------
# 
# If you want to distill how a PennyLane plugin works down to one thing, it’s all in the devices! In
# PennyLane, you just create your circuit (a quantum function) and decorate it with
# ``@qml.qnode(dev)``, where ``dev`` is (one of) a plugin’s device(s). In the PennyLane-Qiskit plugin,
# there are `several Qiskit devices <https://docs.pennylane.ai/projects/qiskit/en/stable/#devices>`__
# you can use, but here are the heavy-hitters for Qiskit 1.0:
# 
# -  ``"qiskit.basicsim"``: uses the Qiskit ``BasicSimulator`` backend from the ``basic_provider``
#    module in Qiskit 1.0.
# -  ``"qiskit.ibmq"``: lets you run PennyLane code on IBM Q hardware, where you can choose between
#    different backends - either simulators tailor-made to emulate the real hardware, or the real
#    hardware itself.
# 
# If you want to use any of these devices in PennyLane, just do
# ``dev = qml.device("<device name>", wires=num_qubits)`` and any QNode decorated with
# ``@qml.qnode(dev)`` will execute on the corresponding device or backend. Here’s a quick example
# showing how simple this is.
# 

import pennylane as qml

dev = qml.device("qiskit.basicsim", wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.expval(qml.Z(0)), qml.expval(qml.X(1))

print(circuit(shots=1000))

######################################################################
# Magic 🪄! With one line of code, you can work inside PennyLane and ship the execution off to your
# favourite IBM device. It’s exactly like using Qiskit 1.0, but you interact with PennyLane instead.
# 

######################################################################
# Converting Qiskit 1.0 code to PennyLane 🦋
# ------------------------------------------
# 
# This is probably what a lot of the audience is wondering: “Can I combine my existing work in Qiskit
# with PennyLane?” YES. And don’t worry, you don’t need to import a ton of things or use a bunch of
# functions — you only need to know *two* things:
# 
# 1. ```qml.from_qiskit`` <https://docs.pennylane.ai/en/stable/code/api/pennylane.from_qiskit.html>`__:
#    converts an entire Qiskit ``QuantumCircuit`` — the whole thing — into a PennyLane quantum
#    function. It will faithfully convert Qiskit-side measurements (even mid-circuit measurements) or
#    you can append Pennylane-side measurements directly to it.
# 2. ```qml.from_qiskit_op`` <https://docs.pennylane.ai/en/stable/code/api/pennylane.from_qiskit_op.html>`__:
#    converts a ``SparsePauliOp`` in Qiskit 1.0 to the equivalent operator in PennyLane.
# 
# Both of these functions give you all the functionality you need to access PennyLane’s features and
# user-interface starting from the Qiskit 1.0 side. Let’s look at an example where both of these
# functions are used.
# 

######################################################################
# Let’s say you’ve created the following Qiskit code that prepares a GHZ state for an arbitrary amount
# of qubits:
# 

from qiskit import QuantumCircuit 

n = 10

def qiskit_GHZ_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

######################################################################
# Next, let’s measure the following ``SparsePauliOp`` operators at the end of the circuit:
# 

from qiskit.quantum_info import SparsePauliOp

operator_strings = ['I' * i + 'XY' + 'I' * (n - 2 - i) for i in range(n-1)]
operators = [SparsePauliOp(operator_string) for operator_string in operator_strings]
operators

######################################################################
# We can convert each ``SparsePauliOp`` into a PennyLane operator with ``qml.from_qiskit_op``:
# 

pl_operators = [qml.from_qiskit_op(qiskit_op) for qiskit_op in operators]
pl_operators

######################################################################
# .. note::
#    What's with the indices starting from the highest and going to the lowest 🤔? PennyLane 
#    orders wires / qubits from *right to left*. This is exactly like the decimal system in that 
#    powers of ten increase from right to left: 
#    :math:`1967 = 1 \times 10^3 + 9 \times 10^2 + 6 \times 10^1 + 7 \times 10^0`. In binary, we 
#    follow the same convention: 
#    :math:`\texttt{int}(1011) = 1 \times 2^3 + 0 \times 2^2 + 1 \times 2^1 + 1 \times 2^0 = 11`. So, 
#    for a list of operator strings like ``'XYIIIIIIII'``, ``X`` is on the tenth qubit (ninth index) 
#    and ``Y`` is on the ninth qubit (eighth index).
# 


######################################################################
# Next, we convert a Qiskit ``QuantumCircuit`` to PennyLane with ``qml.from_qiskit``. We can append
# the measurements — expectation values (``qml.expval``) of ``pl_operators`` — with the
# ``measurements`` keyword argument, which accepts a list of PennyLane measurements.
# 

measurements = [qml.expval(op) for op in pl_operators]

qc = qiskit_GHZ_circuit(n)
pl_qfunc = qml.from_qiskit(qc, measurements=measurements)

######################################################################
# The last thing to do is make ``pl_func`` a QNode. Another way of turning ``pl_qfunc`` into a QNode 
# is by wrapping it with `qml.QNode` and supplying the device.
# 

pl_circuit = qml.QNode(pl_qfunc, device=qml.device("qiskit.basicsim", wires=n))
pl_circuit()

######################################################################
# And that’s it! Now you have a copy of your work in PennyLane, where you can access
# fully-differentiable and hardware-agnostic quantum programming. Heck, you could even run your
# Qiskit-to-PennyLane circuit (``pl_qfunc``) on *other* hardware by specifying a different device.
# 

######################################################################
# Further resources 📓
# ====================
# 
# There’s so much more to learn about what’s possible in PennyLane, and if you’re coming from Qiskit
# you’re in good hands 😌. The PennyLane-Qiskit plugin is your personal chaperone to the PennyLane
# ecosystem. You can dive deeper into what’s possible with the PennyLane-Qiskit plugin by visiting the
# `plugin homepage <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__. In upcoming releases,
# we’ll be refreshing our integration with Qiskit 1.0 runtimes and primitives. Stay tuned for that! In 
# the mean time, if you have any questions about the plugin, PennyLane, or even Qiskit, drop a question
# on our `Discussion Forum <https://discuss.pennylane.ai>`__ and we’ll promptly respond.
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
# # .. include:: ../_static/authors/isaac_de_vlugt.txt
