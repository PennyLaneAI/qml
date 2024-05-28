r"""Your Ultimate Guide to PennyLane if you know Qiskit
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
# With the first stable release of Qiskit in February 2024 — `Qiskit
# 1.0 <https://github.com/Qiskit/qiskit>`__ — breaking changes were to be expected. Don’t abandon your
# code, though! In this demonstration, we want to show you what PennyLane is all about and how you can
# use your existing Qiskit code (``v1.0`` and ``<=v0.46``) with PennyLane.
# 

######################################################################
# If you want to follow along on your computer, you’ll need to install PennyLane, the PennyLane-Qiskit
# plugin, and a couple extras: ``pip install pennylane pennylane-qiskit torch tensorflow``. Now, let’s
# get started.
# 
#    TODO: WHEN THIS GETS RELEASED, NEED TO CHECK WHICH VERSION OF QISKIT SHIPS WITH THE PLUGIN AND
#    WHAT PRE-1.0 VERSIONS OF QISKIT WE STILL SUPPORT.
# 

######################################################################
# PennyLane 🤝 Qiskit
# -------------------
# 
# Learning a new language can be hard, but luckily PennyLane has simple tools to help with the move.
# As it pertains to Qiskit, the PennyLane-Qiskit plugin is your gateway to the land of PennyLane that
# even lets you keep your existing Qiskit work — you can port it over to PennyLane in a few lines of
# code without even having to know a ton about how PennyLane works.
# 
# There are two functions you need to know about for converting your work in Qiskit to PennyLane:
# 
# 1. ``qml.from_qiskit``: converts an entire Qiskit ``QuantumCircuit`` — the whole thing! — into
#    PennyLane. It will faithfully convert Qiskit-side measurements (even mid-circuit measurements) or
#    you can append Pennylane-side measurements directly to it.
# 2. ``qml.from_qiskit_op``: converts a ``SparsePauliOp`` in Qiskit 1.0 to the equivalent operator in
#    PennyLane.
# 
# Both of these functions give you all the functionality you need to access PennyLane’s features and
# user-interface starting from the Qiskit side. As an example, let’s say you have the following code
# in Qiskit that prepares a Bell state.
# 

from qiskit import QuantumCircuit 

qc = QuantumCircuit(2, 1)
qc.h(0)
qc.cx(0, 1);

qc.draw(output='mpl');

######################################################################
# To convert this circuit to PennyLane, you just use ``qml.from_qiskit``:
# 

import pennylane as qml

pl_circuit = qml.from_qiskit(qc)
print(qml.draw_mpl(pl_circuit)())

######################################################################
# Want to measure some expectation values of Pauli operators, as well? Use ``qml.from_qiskit_op`` to
# convert a ``SparsePauliOp`` into PennyLane’s equivalent operator,
# 

from qiskit.quantum_info import SparsePauliOp

qiskit_pauli_op = SparsePauliOp('XY')
pl_pauli_op = qml.from_qiskit_op(qiskit_pauli_op)

######################################################################
# then you can *append* the expectation value measurement — done with ``qml.expval`` — to the
# PennyLane circuit when you create it with ``qml.from_qiskit``:
# 

pl_func = qml.from_qiskit(qc, measurements=[qml.expval(pl_pauli_op)])
print(qml.draw_mpl(pl_func)())

######################################################################
# And just like that, you’re in Pennylane land! Now you might be asking: “What is ``pl_func`` and how
# do I use it further?” To answer those questions, we need to get to know PennyLane a little better.
# 

######################################################################
# Get to Know PennyLane 🌞
# ------------------------
# 
# Quantum Circuits 🎞️
# ~~~~~~~~~~~~~~~~~~~
# 
# Let’s go back to that Qiskit circuit (``qc``) that we created earlier. If we want to execute that
# circuit in Qiskit and get some results, we can do this:
# 

from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator

qc.measure(1, 0)

backend = BasicSimulator()
tqc = transpile(qc, backend)
counts = backend.run(tqc).result().get_counts()

counts

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    {'1': 522, '0': 502}

######################################################################
# When we call ``qml.from_qiskit`` on our Qiskit circuit, this is equivalent to creating this function
# in PennyLane.
# 

def pl_func():
    """Equivalent to doing pl_circuit = qml.from_qiskit(qc)"""
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=1)

######################################################################
#    Qubits in PennyLane are called “wires”. Why? As was mentioned, PennyLane is a hardware agnostic
#    framwork, and “wires” is a more hardware agnostic term for quantum degrees of freedom that
#    quantum computers can be based off of.
# 

######################################################################
# A function like the one above is called a **quantum function**. A quantum function in PennyLane just
# contains quantum gates and (optionally) returns a measurement. In this case, ``qml.counts(wires=1)``
# is the measurement, which counts the number of times a ``0`` or a ``1`` occurs when measuring wire
# ``1``.
# 
# If we actually want to *execute* the circuit and get a result, we need to define what the circuit
# runs on, just like how in Qiskit when we defined a ``BasicSimulator`` instance and then transpiled
# the circuit accordingly. PennyLane’s way of doing this is simple: (1) define a device with
# ``qml.device`` and (2) pair the device with the quantum function with ``qml.QNode``.
# 

dev = qml.device("default.qubit", shots=1024)
pl_circuit = qml.QNode(pl_func, dev)

pl_circuit()

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    {'0': tensor(494, requires_grad=True), '1': tensor(530, requires_grad=True)}

######################################################################
# Now that we have the full picture, let’s take a step back and summarize what’s going on. The first
# thing you’ll notice is that PennyLane’s primitives are Pythonic and array-like; quantum circuits are
# *functions*, returning measurements that behave like NumPy arrays. The function ``pl_circuit`` is
# called a *quantum node* (QNode), which is the sum of two things:
# 
# 1. A *quantum function* that contains quantum instructions. This is ``pl_func``, which just contains
#    quantum operations (gates) and returns a measurement. In this case, ``qml.counts(wires=1)`` is
#    the measurement, which counts the number of times a ``0`` or a ``1`` occurs on wire ``1`` and
#    returns a dictionary whose values are NumPy arrays.
# 2. A device: ``qml.device("default.qubit")``. PennyLane has many devices you can choose from, but
#    ``"default.qubit"`` is our battle-tested Python statevector simulator.
# 
# A QNode can be called like a regular Python function, executing on the device you specified — simple
# as that 🌈. Again, this is equivalent to creating a backend in Qiskit and transpiling your Qiskit
# circuit given that backend. PennyLane just does this in a Python-friendly way instead.
# 
# Alternatively, because PennyLane’s primitive is a glorified Python function, wrapping a quantum
# function with ``qml.QNode`` is the same as *decorating* it with ``@qml.qnode(dev)``:
# 

@qml.qnode(dev)
def pl_circuit():
    """Equivalent to doing pl_circuit = qml.QNode(qml.from_qiskit(qc), dev)"""
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=1)

######################################################################
# A minor point, but both approaches work. Check out our documentation for a full list of `quantum
# operations <https://docs.pennylane.ai/en/stable/introduction/operations.html#qubit-operators>`__ and
# `measurements <https://docs.pennylane.ai/en/stable/introduction/measurements.html>`__ that PennyLane
# offers.
# 
# Now that you know how to make quantum circuits in PennyLane, let’s apply it 👇.
# 

######################################################################
# Quantum Machine Learning 🧠
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Because PennyLane’s primitives are Pythonic and array-like, it follows that you can integrate QNodes
# into hybrid models written using PyTorch, TensorFlow / Keras, or JAX in a familiar-feeling way. For
# PyTorch or TensorFlow / Keras, PennyLane can shapeshift QNodes into their layer types with
# ``qml.qnn.TorchLayer`` and ``qml.qnn.KerasLayer``, respectively. Let’s consider this QNode as an
# example.
# 

dev = qml.device('default.qubit')

@qml.qnode(dev)
def circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=[0, 1], normalize=True)
    qml.RY(weights[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[1], wires=1)
    return qml.vn_entropy(wires=[1])

######################################################################
# To turn ``circuit`` into a PyTorch or a Keras layer, it’s almost a one-liner! Both classes just need
# you to supply the shapes of the trainable parameters — in this case, ``weights`` — via a dictionary
# whose keys are the variable names.
# 

import torch

weight_shapes = {"weights": (2,)}
qlayer_torch = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)

######################################################################
# Now we can stick ``qlayer_torch`` into, say, ``Sequential`` as if it were a typical PyTorch layer:
# 

inputs = torch.rand(4, requires_grad=False)

clayer = torch.nn.Softmax(dim=0)
qlayer_torch(clayer(inputs))
torch_model = torch.nn.Sequential(clayer, qlayer_torch)

torch_model(inputs)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    tensor(0.0424, grad_fn=<ToCopyBackward0>)

######################################################################
# In TensorFlow / Keras, it’s the same story.
# 

import tensorflow as tf

qlayer_keras = qml.qnn.KerasLayer(circuit, weight_shapes=weight_shapes, output_dim=1)

clayer_1 = tf.keras.layers.Dense(4)
inputs = tf.random.uniform((1, 4), minval=0, maxval=1)
tf_model = tf.keras.models.Sequential([clayer_1, qlayer_keras])

tf_model(inputs)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    <tf.Tensor: shape=(1,), dtype=float64, numpy=array([0.30674185])>

######################################################################
# The moral of the story here is that PennyLane is smoother than a hot knife through butter at working
# with your favourite frameworks 🧈. All you need is a couple lines of code!
# 

######################################################################
# Quantum Chemistry ⚛️
# ~~~~~~~~~~~~~~~~~~~~
# 
# PennyLane’s quantum chemistry library is quite feature-rich, but let’s look at the meat and potatoes
# 🥩🥔.
# 

######################################################################
# Fermionic Operators and Mappings
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# The two fundamental fermionic operators in PennyLane are ``qml.FermiA`` and ``qml.FermiC``, which
# are the annihilation and creation operators, respectively. You can use these operators for creating
# fermionic Hamiltonians.
# 

c0 = qml.FermiC(0)
a1 = qml.FermiA(1)

fermi_ham = 0.1 * c0 + 1.3 * a1
print(fermi_ham)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    0.1 * a⁺(0)
#    + 1.3 * a(1)

######################################################################
# From there, you can convert your fermionic Hamiltonian into a qubit Hamiltonian via the
# Bravyi-Kitaev (``qml.barvyi_kitaev``), Jordan-Wigner (``qml.jordan_wigner``), or parity mappings
# (``qml.parity_transform``).
# 

num_qubits = 3

jw = qml.jordan_wigner(fermi_ham)
bk = qml.bravyi_kitaev(fermi_ham, n=num_qubits)
parity = qml.parity_transform(fermi_ham, n=num_qubits)

print(f'Jordan-Wigner: {jw}\nBravyi-Kitaev: {bk}\nParity: {parity}')

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    Jordan-Wigner: 0.05 * X(0) + -0.05j * Y(0) + 0.65 * (Z(0) @ X(1)) + 0.65j * (Z(0) @ Y(1))
#    Bravyi-Kitaev: 0.05 * (X(0) @ X(1) @ X(2)) + -0.05j * (Y(0) @ X(1) @ X(2)) + 0.65 * (Z(0) @ X(1) @ X(2)) + 0.65j * (Y(1) @ X(2))
#    Parity: 0.05 * (X(0) @ X(1) @ X(2)) + -0.05j * (Y(0) @ X(1) @ X(2)) + 0.65 * (Z(0) @ X(1) @ X(2)) + 0.65j * (Y(1) @ X(2))

######################################################################
# Molecular Hamiltonians
# ^^^^^^^^^^^^^^^^^^^^^^
# 
# If you’re doing quantum chemistry, you’ll definitely need to know how to create a molecular
# Hamiltonian. In PennyLane, it’s as easy as specifying the atomic symbols in your molecule and their
# coordinates (in atomic units) and giving that to our aptly-named ``molecular_hamiltonian`` function.
# 
# Let’s take beryllium hydride (:math:`\text{BeH}_{\text{2}}`) as an example.
# 

import pennylane.numpy as np

symbols = ["H", "Be", "H"]

geometry = np.array(
    [
        [2.5, 0, 0], # H 
        [0, 0, 0], # Be 
        [-2.5, 0, 0] # H
    ]
)

hamiltonian, num_qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
num_qubits

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    14

######################################################################
# With this, you can quickly take ``hamiltonian`` and do some VQE to get the ground state. PennyLane
# has a template operation called ``qml.AllSinglesDoubles``, which prepares correlated states of
# molecules by applying all ``SingleExcitation`` and ``DoubleExcitation`` operations to the initial
# Hartree-Fock state. The prerequisites here are:
# 
# 1. Calculate the initial Hartree-Fock state with ``qml.qchem.hf_state``.
# 2. Generate single and double excitations from our initial Hartree-Fock reference state.
# 

active_electrons = 4 # 1 from both H, 2 from Be
hf_state = qml.qchem.hf_state(active_electrons, num_qubits)
singles, doubles = qml.qchem.excitations(active_electrons, num_qubits)

######################################################################
# Now, we can define a quantum circuit ansatz that just contains ``qml.AllSinglesDoubles`` and returns
# the expectation value of the ``hamiltonian``. And, hey, let’s try PennyLane’s performant simulator
# device called ``lightning.qubit`` (it has a C++ backend 🤓).
# 

dev = qml.device('lightning.qubit', wires=num_qubits)

@qml.qnode(dev)
def circuit(params):
    qml.AllSinglesDoubles(params, range(num_qubits), hf_state, singles, doubles)
    return qml.expval(hamiltonian)

######################################################################
# Now optimize 🤖! Just define the trainable parameters for ``AllSinglesDoubles``, an optimizer, and
# do the parameter updates with the optimizer’s ``step_and_cost`` method.
# 

np.random.seed(1967)

params = np.random.normal(0, np.pi, len(singles) + len(doubles))
energies = []

opt = qml.GradientDescentOptimizer(stepsize=0.1)

for _ in range(100):
    params, E = opt.step_and_cost(circuit, params) # returns the new params and circuit(params)
    energies.append(E)

import matplotlib.pyplot as plt

plt.scatter(list(range(100)), energies)
plt.show()

######################################################################
# Quantum chemistry problems ain’t easy, but at least PennyLane won’t get in your way when you want to
# put your ideas to practice 💪.
# 

######################################################################
# Further resources 📓
# ====================
# 
# There’s so much more to learn about what’s possible in PennyLane, and if you’re coming from Qiskit
# you’re in good hands! The PennyLane-Qiskit plugin is your personal chaperone to the PennyLane
# ecosystem. You can dive deeper into what’s possible with the PennyLane-Qiskit plugin by visiting the
# `plugin homepage <https://docs.pennylane.ai/projects/qiskit/en/stable/>`__. In upcoming releases,
# we’ll be refreshing our integration with Qiskit 1.0 `runtimes <https://cloud.ibm.com/quantum>`__ and
# `primitives <https://docs.quantum.ibm.com/api/qiskit/primitives>`__. Stay tuned for that!
# 
# Another great thing about the PennyLane ecosystem is that we have an `emporium of up-to-date
# demos <https://pennylane.ai/search/?contentType=DEMO&sort=publication_date>`__ maintained by the
# same people that develop PennyLane. If you’re just starting out, I recommend reading our `qubit
# rotation tutorial <https://pennylane.ai/qml/demos/tutorial_qubit_rotation/>`__.
# 
# Now that you’ve used PennyLane, every road in the wonderful world of quantum programming SDKs is
# open with no set speed limits 🏎️. If you have any questions about the PennyLane-Qiskit plugin,
# PennyLane, or even Qiskit, drop a question on our `Discussion
# Forum <https://discuss.pennylane.ai>`__ and we’ll promptly respond. You can also visit our website,
# `pennylane.ai <https://pennylane.ai>`__, to see the latest and greatest PennyLane features, demos,
# and blogs, and follow us on `LinkedIn <https://www.linkedin.com/company/pennylaneai/>`__ or `X
# (formerly Twitter) <https://twitter.com/PennyLaneAI>`__ to stay updated!
# 

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/isaac_de_vlugt.txt
