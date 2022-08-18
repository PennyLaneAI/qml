r""".. _quantum_circuit_cutting:
Quantum Circuit Cutting
=======================

.. meta:: 
    :property="og:description": insert cool description here 

*Authors: Gideon Uchehara, Matija Medvidović, Anuj Apte*

Introduction
-------------------------------------

Quantum computers are projected to accomplish tasks that are far beyond
the reach of classical computers. However, today’s quantum computers do
not have the sufficient amount of qubits required to achieve this feat,
since they are limited by decoherence and low fidelity. This means that
we cannot run large quantum circuits on existing quantum computers
without significant errors in our results. An alternative approach is to
simulate a quantum computer, but this requires exponentially more memory
and runtime as the number of qubits increases. This is the reason it is
intractable to simulate an ideal quantum computer classically.

To run a quantum circuit that is larger than current small quantum
computers and intractable to simulate with classical computers, we need
to cut the it into smaller subcircuits. To do this, we need to make many
measurements for the different subcircuits using our small quantum
computer and then recombine the outcomes of our measurements using a
classical computer. But the good news is that we can at least run our
large quantum circuit using our small quantum computer with negligible
errors.

In this demo, we will first introduce the theory behind quantum circuit
cutting based on Pauli measurements and see how it implemented in
PennyLane. This method was first introduced in the paper [#Peng2019]_.
Thereafter, we discuss the theoretical basis on randomized circuit
cutting with two-designs and demonstrate the resulting improvement in
performance compared to Pauli measurement based circuit cutting for an
instance of Quantum Approximate Optimization Algorithm (QAOA).

The main idea behind the algorithm that allows you to simulate large
quantum circuits on a small quantum computer is called *quantum circuit
cutting*.

A quantum circuit cutting algorithm takes a large quantum circuit,
decomposes it into smaller subcircuits that can be executed on a smaller
quantum device. The results from executing the smaller subcircuits are
then recombined through some classical post-processing to obtain the
original result of the large quantum circuit. The PennyLane function
that implements this algorithm is called ``pennylane.cut_circuit``.
Before we delve into PennyLane and work through some examples, let’s
introduce the theory behind the Pauli circuit cutting method.

Background: Understanding the Pauli cutting method
--------------------------------------------------

Consider a :math:`2`-level quantum system in an arbitrary state, with a
density matrix :math:`\rho`. :math:`\rho` can be expressed as a linear
combination of the Pauli matrices as shown below.

.. math::
    \rho = \frac{1}{2}\sum_{i=1}^{8} c_i Tr(\rho O_i)\rho_i

Here, we have denoted Pauli matrices by :math:`O_i`, their
eigenprojectors by :math:`\rho_i` and their corresponding eigenvalues by
:math:`c_i`. In the above equation, :math:`O_1 = O_2 = I`,
:math:`O_3 = O_4 = X`, :math:`O_5 = O_6 = Y` and :math:`O_7 = O_8 = X`.
Also,
:math:`\rho_1 = \rho_7=\left | {0} \right\rangle \left\langle {0} \right |`,
:math:`\rho_2 = \rho_8 = \left | {1} \right\rangle \left\langle {1} \right |`,
:math:`\rho_3 = \left | {+} \right\rangle \left\langle {+} \right |`,
:math:`\rho_4 = \left | {-} \right\rangle \left\langle {-} \right |`,
:math:`\rho_5 = \left | {+i} \right\rangle \left\langle {+i} \right |`,
:math:`\rho_6 = \left | {-i} \right\rangle \left\langle {-i} \right |`
and :math:`c_i = \pm 1`.

To implement the above equation, each term :math:`Tr(\rho O_i)\rho_i` in
the equation is broken into two parts. The first part which corresponds
to the the ``MeasureNode``, :math:`Tr(\rho O_i)` involves measuring the
expectation value of :math:`O_i` for the first fragment
(subcircuit-:math:`u`) of the state :math:`\rho`. The second part which
corresponds to the ``PrepareNode``, :math:`\rho_i` involves preparing
the corresponding eigenstate :math:`\rho_i` for the second fragment
(subcircuit-:math:`v`) of the state :math:`\rho`.

In general, if we picture the time evolution of a qubit state from time
:math:`u` to time :math:`v` as a line between point :math:`u` and point
:math:`v`, if we cut this line at some point, we can recover the qubit
state at point :math:`v` using the above equation, as shown in figure 1.

This means that we only have to do three measurements
:math:`Tr(\rho X), Tr(\rho Y), Tr(\rho Z) \right)` for
subcircuit-:math:`u` and initialize subcircuit-:math:`v` with only four
states: :math:`\left | {0} \right\rangle`,
:math:`\left | {1} \right\rangle`, :math:`\left | {+} \right\rangle` and
:math:`\left | {+i} \right\rangle`. The other two nontrivial expectation
values for states :math:`\left | {-} \right\rangle` and
:math:`\left | {- i} \right\rangle` can be derived with classical
postrocessing.

.. figure:: ../demonstrations/quantum_circuit_cutting/1Qubit-Circuit-Cutting.png
    :align: center
    :width: 80%

    Figure 1. The Pauli circuit cutting method for 1-qubit circuit. The
    first half of the cut circuit on the left (subcircuit-u) is the part
    with ``MeasureNode``. The second half of the cut circuit on the right
    (subcircuit-v) is the part with ``PrepareNode``

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s go through the steps involved in ``qcut`` implementation in
Pennylane using the circuit below.
"""

# Import the relevant libraries
import pennylane as qml
from pennylane import numpy as np
from pennylane.tape import QuantumTape

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)

    qml.CZ(wires=[0, 1])
    qml.RY(-0.4, wires=0)

    qml.CZ(wires=[1, 2])

    return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))


x = np.array(0.531, requires_grad=True)
fig, ax = qml.draw_mpl(circuit)(x)


######################################################################
# With the above 3-qubit quantum circuit, we will assume the following:
# our quantum computer has less than 3 qubits, say we can execute at most
# two qubits on it. This means that we have to cut the circuit such that
# the resulting subcircuits have at most 2 qubits.
#
# Our first decision would be to chose a cut location that separates the
# circuit into subcircuits with at most 2 qubits. By inspection, it is
# obvious that the best cut location would be between the two ``CZ`` gates
# on qubit 1. Hence, we place a ``WireCut`` operation at that location as
# shown below:
#

dev = qml.device("default.qubit", wires=3)

# Quantum Circuit with QNode
@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)

    qml.CZ(wires=[0, 1])
    qml.RY(-0.4, wires=0)

    qml.WireCut(wires=1)  # Cut location

    qml.CZ(wires=[1, 2])

    return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))


x = np.array(0.531, requires_grad=True)  # Defining the parameter x
fig, ax = qml.draw_mpl(circuit)(x)  # Drawing circuit


######################################################################
# The double vertical lines between the two ``CZ`` gates on qubit 1 in the
# above figure shows where we have chosen to cut. Cutting in this position
# gives two subcircuits with 2 qubits each.
#
# Next, we have to apply ``qml.cut_circuit`` operation as a
# decorator to the ``circuit`` function. Under the hood, executing
# ``circuit`` runs multiple configurations of the 2-qubit subcircuits
# which are then postprocessed to give the result of the original circuit
# as follows:
#

dev = qml.device("default.qubit", wires=3)

# Quantum Circuit with QNode
@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)

    qml.CZ(wires=[0, 1])
    qml.RY(-0.4, wires=0)

    qml.WireCut(wires=1)  # Cut location

    qml.CZ(wires=[1, 2])

    return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))


x = np.array(0.531, requires_grad=True)
circuit(x)  # Executing the quantum circuit


######################################################################
# **Graph partitioning and automatic cut placement with Pennylane**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# It was easy to identify the optimal cut location for the above circuit
# by inspection. But this is just one way to cut the circuit. Another
# person may decide to choose more than one cut locations and still
# achieve the same result. The question then is - how are results affected
# by different choices of cut location? The short answer is that the more
# cuts on a circuit, the more classical post-processing overhead you
# incur.
#
# A closer look at QCut would reveal that circuit cutting has a lot to do
# with finding the optimal cut that fragments a circuit such that it
# results in the least classical postprocessing overhead after
# measurement.What happens if the optimal wire-cut placement location(s)
# cannot be determined by mere inspection for an arbitrary circuit? If we
# visualize a quantum circuit as a graph with gates as nodes and qubits as
# edges, this problem becomes synonymous with the graph partitioning
# problem. Graph partitioning involves the fragmentation of the vertices
# of a graph into mutually exclusive groups of smaller subgraphs. This
# problem falls under the category of NP-hard problems. This means that
# only heuristic or approximation algorithms have been developed to solve
# this problem.
#
# To avoid boring you with details, PennyLane has functions that implement
# the graph partitioning operation as part of ``qml.cut_circuit``. One of
# them is the ``auto_cutter``. This option can be enabled in
# ``qml.cut_circuit`` to make attempts in finding an optimal cut. Where it
# is difficult to manually determine the optimal cut location, decorating
# the QNode with the ``cut_circuit()`` batch transform and enabling
# ``auto_cutter`` is the recommended approach into circuit cutting. The
# following examples shows this capability on the same circuit as above
# but with the ``WireCut`` removed.
#

dev = qml.device("default.qubit", wires=3)


@qml.cut_circuit(auto_cutter=True)  # auto_cutter enabled
@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)

    qml.CZ(wires=[0, 1])
    qml.RY(-0.4, wires=0)

    qml.WireCut(wires=1)  # Cut location

    qml.CZ(wires=[1, 2])

    return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))


x = np.array(0.531, requires_grad=True)
circuit(x)


######################################################################
# To get a handle on graph partitioning for circuit cutting, let’s
# interact with one of pennylane’s functionalities —
# ``find_and_place_cuts()``. This is another operation in PennyLane that
# solves graph partitioning for ``cut_circuit``. It makes attempts in
# automatically finding a cut given the device constraints. Using the same
# circuit as above but with the ``WireCut`` removed, the same (optimal)
# cut can be recovered with automatic cutting using
# ``find_and_place_cuts()``.
#
# To use this functionality, we first convert the quantum circuit to a
# ``QuantumTape`` and then to a graph. By converting the quantum circuit
# to a ``QuantumTape``, we can apply the elementary steps in
# ``cut_circuit`` transform and manipulate the quantum tape to perform
# circuit cutting. This is done in Pennylane by passing the quantum
# circuit as a quantum tape to ``qml.transforms.qcut.tape_to_graph``.
# Let’s see how this is done.
#
# First, we define our quantum circuit as a ``QuantumTape`` as show below:
#

# Defining a QuantumTape
with QuantumTape() as uncut_tape:
    qml.RX(0.531, wires=0)
    qml.RY(0.9, wires=1)
    qml.RX(0.3, wires=2)

    qml.CZ(wires=[0, 1])
    qml.RY(-0.4, wires=0)

    qml.CZ(wires=[1, 2])

    qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))

print(uncut_tape.draw())


######################################################################
# Our next step is to convert quantum tape to graph:
#

graph = qml.transforms.qcut.tape_to_graph(uncut_tape)


######################################################################
# Let’s use ``find_and_place_cuts()`` to find the optimal cut given the
# device constraints
#
# The `find_and_place_cuts` function takes as its input, the graph to cut,
# and a cut strategy for optimizing cutting parameters based on device
# constraints. Please refer to
# `the documentation <https://pennylane.readthedocs.io/en/latest/code/api/pennylane.transforms.qcut.find_and_place_cuts.html>`__
# for further details
#

cut_graph = qml.transforms.qcut.find_and_place_cuts(
    graph=graph,
    cut_strategy=qml.transforms.qcut.CutStrategy(max_free_wires=2),
)

print(qml.transforms.qcut.graph_to_tape(cut_graph).draw())  # visualize the cut QuantumTape


######################################################################
# As you can see, the same (optimal) cut has been recovered with automatic
# cutting using ``find_and_place_cuts()``.
#
# Next, we must remove the ``WireCut`` nodes in the graph and replace with
# ``MeasureNode`` and ``PrepareNode`` pairs.
#

qml.transforms.qcut.replace_wire_cut_nodes(cut_graph)

######################################################################
# .. figure:: ../demonstrations/quantum_circuit_cutting/MeasurePrepareNodes.png
#     :align: center
#     :width: 90%
#
#     Figure 2. Replace WireCut with MeasureNode and PrepareNode
#
# The ``MeasureNode`` and ``PrepareNode`` pairs are placeholder operations
# that allow us to cut the circuit graph and then iterate over measurement
# of Pauli observables and preparation of corresponding eigenstates
# configurations at cut locations.
#
# As a recap, after applying ``find_and_place_cuts()``, we identified an
# optimal cut location on the ``QuantumTape`` and applied the ``WireCut``
# node at this location. This node was later replaced with ``MeasureNode``
# and ``PrepareNode`` pairs. Note that we have not separated the fragments
# that resulted after the cut into two different subcircuits.
#
# To separete these fragments into different subcircuits, we use the
# ``fragment_graph()`` function that pulls apart the quantum circuit graph
# into disconnected components as well as returning the
# ``communication_graph`` detailing the connectivity between the
# components.
#

fragments, communication_graph = qml.transforms.qcut.fragment_graph(cut_graph)

######################################################################
# .. figure:: ../demonstrations/quantum_circuit_cutting/separateMeasurePrepareNodes.png
#     :align: center
#     :width: 90%
#
#     Figure 3. Separate fragmments into different subcircuits
#
# Next, we convert the subcircuit fragments back to QuantumTape objects
#

fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]

######################################################################
# Let’s visualize the two subcircuit fragments:
#

print(fragment_tapes[0].draw())  # Subcircuit-u fragment
print(fragment_tapes[1].draw())  # Subcircuit-v fragment

######################################################################
# After separating the subcircuit fragments, we must remap the tape wires
# to match those available on our small quantum device using
# ``remap_tape_wires``. This means that we have to re-number each qubit’s
# wire for the two subcircuit.
#

dev = qml.device("default.qubit", wires=2)
fragment_tapes = [qml.transforms.qcut.remap_tape_wires(t, dev.wires) for t in fragment_tapes]

######################################################################
# Based on the number of cuts and the resulting subcircuits, each circuit
# fragment is expanded over ``MeasureNode`` and ``PrepareNode``
# configurations and a flat list of tapes is created using
# ``expand_fragment_tape``. Recall that from equation
# (`7 <#mjx-eqn-eq7>`__), with each cut, we only have to do four
# measurements
# :math:`\left(Tr(\rho I), Tr(\rho X), Tr(\rho Y), Tr(\rho Z) \right)` for
# subcircuit-:math:`u` and initialize subcircuit-:math:`v` with only four
# states: :math:`\left | {0} \right\rangle`,
# :math:`\left | {1} \right\rangle`, :math:`\left | {+} \right\rangle` and
# :math:`\left | {+i} \right\rangle`
#
# This means that for the subcircuit with the ``MeasureNode``, we measure
# the 4 Pauli observables. This generates 4 different subcircuit
# configurations for subcircuit-:math:`u`. Also, for the subcircuit with
# ``PrepareNode``, we initialize (or parepare) the qubit states with the 4
# different eigenstates (:math:`\left | {0} \right\rangle`,
# :math:`\left | {1} \right\rangle`, :math:`\left | {+} \right\rangle` and
# :math:`\left | {+i} \right\rangle`). This generates 4 different
# subcircuit configuration for subcircuit-:math:`v`. In total, we have
# generated 8 subcircuit configurations by expanding the ``MeasureNode``
# and the ``PrepareNode``.
#

expanded = [
    qml.transforms.qcut.expand_fragment_tape(t) for t in fragment_tapes
]  # expanding MeasureNode
# and PrepareNode

configurations = []  # A list of subcircuit configurations
prepare_nodes = []  # A list of prepareNodes in the circuit
measure_nodes = []  # A list of MeasureNodes in the circuit
for tapes, p, m in expanded:
    configurations.append(tapes)
    prepare_nodes.append(p)
    measure_nodes.append(m)

tapes = tuple(tape for c in configurations for tape in c)

######################################################################
# Let’s visualize the subcircuit configurations of the expanded quantum
# tape fragments. We expect a total of 8 subcircuit configurations.
#
for t in tapes:
    print(qml.drawer.tape_text(t), "\n")

######################################################################
# In this last step, we execute all the tapes (on our small quantum
# computer) by passing them as a tupple to ``execute``. The results from
# executing each subcircuit configuration is passed into
# ``qcut_processing_fn()`` for postprocessing which involves combing the
# results to obtain the original full circuit output via a tensor network
# contraction. This means that each measure for subcircuit-:math:`u` is
# multiplied to the corresponding measurement from subcircuit-:math:`v`.
#

results = qml.execute(tapes, dev, gradient_fn=None)  # executing each subcircuit configuration

qml.transforms.qcut.qcut_processing_fn(
    results,
    communication_graph,
    prepare_nodes,
    measure_nodes,
)

######################################################################
# Randomized Circuit Cutting
# ------------------------------------
#
# After reviewing the standard circuit cutting based on Pauli measurements
# on single qubits, we are now ready to discuss an improved circuit
# cutting protocol that uses randomized measurements to speed up circuit
# cutting. Our description of this method will be based on the recently
# published work [#Lowe2022]_.
#
# The key idea behind this approach is to use measurements in an entagled
# basis that is based on a unitary 2-design to get more information about
# the state with fewer measurements compared to single qubit Pauli
# measurements.
#
# The concept of 2-designs is simple - a unitary 2-design is finite
# collection of unitaries such that the average of any degree 2 polynomial
# function of a linear operator over the design is exactly the same as the
# average over Haar random measure. For further explanation of this measure read
# the `Haar Measure demo <https://pennylane.ai/qml/demos/tutorial_haar_measure.html>`__.
#
# Let :math:`P(U)` be a polynomial with homogeneous degree at most two in
# the entries of a unitary matrix :math:`U`, and degree two in the complex
# conjugates of those entries. A unitary 2-design is a set of :math:`L`
# unitaries :math:`\{U_{L}\}` such that
#
# .. math:: \frac{1}{L} \sum_{l=1}^{L} P(U_l) = \int_{\mathcal{U}(d)} P (U) d\mu(U)~.
#
# The elemements of the Clifford group over the qubits being cut are an
# example of a 2-design. We don’t have a lot of space here to go into too
# many details. But fear not - there is an `entire
# demo <https://pennylane.ai/qml/demos/tutorial_unitary_designs.html>`__
# dedicated to this wonderful topic!
#
# .. figure:: ../demonstrations/quantum_circuit_cutting/flowchart.svg
#     :align: center
#     :width: 90%
#
#     Figure 4. Illustration of Randomized Circuit Cutting based on Two-Designs
#
# If :math:`k` qubits are being cut, then the dimensionality of the
# Hilbert space is :math:`d=2^{k}`. In the randomized measurement circuit
# cutting procedure, we trace out the qubits and a prepare a random basis
# state with probability :math:`d/(2d+1)`. For a linear operator
# :math:`X \in \mathbf{L}(\mathbb{C}^{d})` acting on the :math:`k`-qubits,
# this operation corresponds to the completely depolarizing channel
#
# .. math::  \Psi_{1}(X) = \textrm{Tr}(X)\frac{\mathbf{1}}{d}~.
#
# Otherwise, we perform a randomized measure-and-prepare protocol based on
# a unitary 2-design (e.g. a random Clifford) with probability
# :math:`(d+1)/(2d+1)`, corresponding to the channel
#
# .. math::  \Psi_{0}(X) = \frac{1}{d+1}\left(\textrm{Tr}(X)\mathbf{1} + X\right)~.
#
# The set of Kraus operators for channels :math:`\Psi_{1}, \Psi_{0}` are
#
# .. math::
#
#   \Psi_{1}(X) \xrightarrow{} \left\{ \frac{|i\rangle \langle j|}{\sqrt{d}} \right\} \quad
#   \Psi_{0}(X) \xrightarrow{} \left\{ \frac{\mathbf{1}}{\sqrt{d+1}} \right\} \cup \left\{ \frac{|i\rangle \langle j|}{\sqrt{d+1}} \right\}~,
#
# where indices :math:`i,j` run over the :math:`d` basis elements.
#
# Together these two channels can be used to obtain a resolution of the
# Identity channel on the :math:`k`-qubits as follows
#
# .. math::   X = (d+1)\Psi_0(X)-d\Psi_1(X)~.
#
# By employing this procedure, we can estimate the outcome of the original
# circuit by using the cut circuits. For an error threshold of
# :math:`\varepsilon`, the associated overhead is
# :math:`O(4^{k}(n+k^{2})/\varepsilon^{2})`. When :math:`k` is a small
# constant and the circuit is cut into roughly two equal halves, this
# procecdure effectively doubles the number of qubits which can be
# simulated given a quantum device since the overhead is :math:`O(4^k)`
# compared with the :math:`O(16^k)` overhead of cutting with single-qubit
# measurements. Note that although the overhead incurred is smaller, the
# average depth of the circuit is greater since a random Clifford unitary
# over the :math:`k` qubits has to implemented when randomized measurement
# is performed.
#
# Comparison
# ---------------------
#
# We have seen that looking at circuit cutting through the lens of
# 2-designs can be a source of considerable speedups. A good test case
# where one may care about accurately estiamting an observale is the
# `Quantum Approximate Optimization
# Algorithm <https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html>`__
# (QAOA). In its simplest form, QAOA concerns itself with finding a
# lowest-energy state of a *cost hamiltonian* :math:`H_{\mathcal{C}}`:
#
# .. math::   H_\mathcal{C} = \frac{1}{|E|} \sum _{(i, j) \in E} Z_i Z_j
#
# on a graph :math:`G=(V,E)`, where :math:`Z_i` is a Pauli-:math:`Z`
# operator. The normalization factor is just here so that expectation
# values do not leave the :math:`[-1,1]` interval.
#
# Setup
# ~~~~~
#
# Suppose that we have a specific class of graphs we care about and
# someone already provided us with optimal QAOA angles :math:`\gamma` and
# :math:`\beta` for QAOA of depth :math:`p=1`. Here’s how to map the input
# graph :math:`G` to the QAOA circuit that solves our problem:
#
# .. figure:: ../demonstrations/quantum_circuit_cutting/graph_to_circuit.svg
#     :align: center
#     :width: 90%
#
#     Figure 5. An example of mapping an input interaction graph to a QAOA
#     circuit. (Note: the “stick” gates represent the ZZ rotation gates, to avoid
#     overcrowding the diagram.)
#
# Let’s generate a similar QAOA graph to the one in the figure this using
# `NetworkX <https://networkx.org/>`__!
#

from itertools import product, combinations

import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(1337)

n_side_nodes = 2
n_middle_nodes = 3

top_nodes = range(0, n_side_nodes)
middle_nodes = range(n_side_nodes, n_side_nodes + n_middle_nodes)
bottom_nodes = range(n_side_nodes + n_middle_nodes, n_middle_nodes + 2 * n_side_nodes)

top_edges = list(product(top_nodes, middle_nodes))
bottom_edges = list(product(middle_nodes, bottom_nodes))

graph = nx.Graph()
graph.add_edges_from(combinations(top_nodes, 2), color=0)
graph.add_edges_from(top_edges, color=0)
graph.add_edges_from(bottom_edges, color=1)
graph.add_edges_from(combinations(bottom_nodes, 2), color=1)

nx.draw_spring(graph, with_labels=True)

######################################################################
# For this graph, optimal QAOA parameters read:
#
# .. math::
#
#    \gamma ^* \approx -0.240 \; ; \qquad \beta ^* \approx 0.327 \quad \Rightarrow \quad \left\langle H_\mathcal{C} \right\rangle ^* \approx -0.248
#

optimal_params = np.array([-0.240, 0.327])
optimal_cost = -0.248

######################################################################
# We also define our cost operator :math:`H_{\mathcal{C}}` as a function.
# Because it is diagonal in the computational basis, we only need to
# define its action on computational basis bitstrings.
#


def qaoa_cost(bitstring):
    bitstring = np.atleast_2d(bitstring)
    z = (-1) ** bitstring[:, graph.edges()]
    costs = z.prod(axis=-1).sum(axis=-1)
    return np.squeeze(costs) / len(graph.edges)


######################################################################
# Let’s make a quick and simple QAOA circuit in PennyLane. Before we, we
# have to briefly think about the cut placement. First, we want to apply
# all ZZ rotation gates corresponding to the ``top`` edges, place the wire
# cut, and then the ``bottom``, to ensure that the circuit actually splits
# in two after cutting.
#


def qaoa_template(params):

    gamma, beta = params

    for i in range(len(graph)):  # Apply the Hadamard gates
        qml.Hadamard(wires=i)

    for i, j in top_edges:

        # Apply the ZZ rotation gates
        # corresponding to the
        # green edges in the figure

        qml.MultiRZ(2 * gamma, wires=[i, j])

    qml.WireCut(wires=middle_nodes)  # Place the wire cut

    for i, j in bottom_edges:

        # Apply the ZZ rotation gates
        # corresponding to the
        # purple edges in the figure

        qml.MultiRZ(2 * gamma, wires=[i, j])

    for i in graph.nodes():  # Finally, apply the RX gates
        qml.RX(2 * beta, wires=i)


######################################################################
# Let’s construct the ``QuantumTape`` corresponding to this template and
# draw the circuit:
#

all_wires = list(range(len(graph)))

with QuantumTape() as tape:
    qaoa_template(optimal_params)
    qml.sample(wires=all_wires)

fig, _ = qml.drawer.tape_mpl(tape, expansion_strategy="device")
fig.set_size_inches(12, 6)


######################################################################
# The Pauli cutting method
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# To run fragment subcircuits and combine them into a finite-shot estimate
# of the optimal cost function using the Pauli cut method, we can use
# built-in PennyLane functions. We simply use the ``qml.cut_circuit_mc``
# transform and everything is taken care of for us.
#
# Note that we have already introduced the ``qml.cut_circuit`` transform
# in the previous section. The ``_mc`` appendix stands for Monte Carlo and
# is used to calculate finite-shot estimates of observables. The
# observable itself is passed to the ``qml.cut_circuit_mc`` transform as a
# function mapping a bitstring (circuit sample) to a single number.
#

dev = qml.device("default.qubit", wires=all_wires)


@qml.cut_circuit_mc(classical_processing_fn=qaoa_cost)
@qml.qnode(dev)
def qaoa(params):
    qaoa_template(params)
    return qml.sample(wires=all_wires)


######################################################################
# We can obtain the cost estimate by simply running ``qaoa`` like a
# “normal” ``QNode``. Let’s do just that for a grid of values so we can
# study convergence.
#

n_shots = 10000

shot_counts = np.logspace(1, 4, num=20, dtype=int, requires_grad=False)
pauli_cost_values = np.zeros_like(shot_counts, dtype=float)

for i, shots in enumerate(shot_counts):
    pauli_cost_values[i] = qaoa(optimal_params, shots=shots)


######################################################################
# We will save these results for later and plot them together with results
# of the randomized measurement method.
#
# The randomized channel-based cutting method
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As noted earlier, the easiest way to mathematically represent the
# randomized channel-based method is to write down Kraus operators for the
# relevant channels, :math:`\Psi _0` and :math:`\Psi _1`. Once we have
# represented them in explicit matrix form, we can simply use ``qml.QubitChannel``.
#
#
# To get our matrices, we represent the computational basis set along the
# :math:`k` cut wires as a unit vector
#
# .. math::
#
#   \left\vert i \right\rangle \mapsto (0, \ldots, 1,\ldots,0)
#
# with the 1 positioned at index :math:`i`. Therefore:
#
# .. math::
#
#    \left\vert i \right\rangle \left\langle j \right\vert \mapsto \begin{pmatrix}
#        0 & 0 & \cdots & 0 & 0 \\
#        0 & \ddots & \cdots & 0 & 0 \\
#        \vdots & 0 & 1 & 0 & \vdots \\
#        0 & 0 & \cdots & \ddots & 0 \\
#        0 & 0 & \cdots & 0 & 0 \\
#    \end{pmatrix}
#
# where the 1 sits at column :math:`i` and row :math:`j`.
#
# Given this representation, a neat way to get all Kraus operators’ matrix
# representations is the following:
#


def make_kraus_ops(num_wires: int):

    d = 2**num_wires

    kraus0 = np.identity(d**2).reshape(d**2, d, d)
    kraus0 = np.concatenate([kraus0, np.identity(d)[None, :, :]], axis=0)
    kraus0 /= np.sqrt(d + 1)

    kraus1 = np.identity(d**2).reshape(d**2, d, d)
    kraus1 /= np.sqrt(d)

    return list(kraus0.astype(complex)), list(kraus1.astype(complex))


######################################################################
# Our next task is to generate two new ``QuantumTape`` objects from our
# existing ``tape``, one for :math:`\Psi _0` and one for :math:`\Psi _1`.
# Currently, a ``qml.WireCut`` dummy gate is used to represent the cut
# position and size. So, iterating through gates in ``tape``:
#
# -  If the gate is a ``qml.WireCut``, we apply the ``qml.QubitChannel``
#    corresponding to :math:`\Psi _0` or :math:`\Psi _1` to different new
#    tapes.
# -  Otherwise, just apply the same exisitng gate to both new tapes
#
# In code, this looks like:
#

with QuantumTape(do_queue=False) as tape0, QuantumTape(do_queue=False) as tape1:
    # Record on new "fragment" tapes

    for op in tape:

        if isinstance(op, qml.WireCut):  # If the operation is a wire cut, replace it

            k = len(op.wires)
            d = 2**k

            K0, K1 = make_kraus_ops(k)  # Generate Kraus operators on the fly
            probs = (d + 1) / (2 * d + 1), d / (2 * d + 1)  # Probabilities of the two channels

            psi_0 = qml.QubitChannel(K0, wires=op.wires, do_queue=False)
            psi_1 = qml.QubitChannel(K1, wires=op.wires, do_queue=False)

            qml.apply(psi_0, context=tape0)
            qml.apply(psi_1, context=tape1)

        else:  # Otherwise, just apply the existing gate
            qml.apply(op, context=tape0)
            qml.apply(op, context=tape1)


######################################################################
# Verify that we get expected values:
#

print(f"Cut size: k={k}")
print(f"Channel probabilities: p0={probs[0]:.2f}; p1={probs[1]:.2f}", "\n")

fig, _ = qml.drawer.tape_mpl(tape0, expansion_strategy="device")
fig.set_size_inches(12, 6)

######################################################################
# You may have noticed that both generarated tapes have the same size as
# the original ``tape``. It may seem that no circuit cutting actually took
# place. However, this is just an artefact of the way we chose to
# represent **classical communication** between subcircuits.
# Measure-and-prepare channels at work here are more naturally implemented
# on a mixed-state simulator. On a real quantum device however,
# introducing a classical communication step is equivalent to separating
# the circuit into two.
#
# Given that we are dealing with quantum channels, we need a mixed-state
# simulator. Luckily, PennyLane has just what we need:
#

device = qml.device("default.mixed", wires=tape.wires)

######################################################################
# We only need to run each of the two generated tapes, ``tape0`` and
# ``tape1``, once, collecting the appropriate number of samples. NumPy can
# take care of this for us - we let ``np.choice`` make our decision on
# which tape to run for each shot:
#

samples = np.zeros((n_shots, len(tape.wires)), dtype=int)

rng = np.random.default_rng(seed=1337)
choices = rng.choice(2, size=n_shots, p=probs)

channels, channel_shots = np.unique(choices, return_counts=True)

print("Which channel to run:", choices)
print(f"Channel 0: {channel_shots[0]} times")
print(f"Channel 1: {channel_shots[1]} times.")


######################################################################
# Time to run the simulator!
#

device.shots = channel_shots[0].item()
(shots0,) = qml.execute([tape0], device=device, cache=False, gradient_fn=None)
samples[choices == 0] = shots0

device.shots = channel_shots[1].item()
(shots1,) = qml.execute([tape1], device=device, cache=False, gradient_fn=None)
samples[choices == 1] = shots1

######################################################################
# Now that we have the result stored in ``samples``, we still need to do
# some postprocessing to obtain final estimates of the QAOA cost. In the
# case of a single cut, the general expression discussed earlier
# specializes to:
#
# .. math::
#
#    \left\langle H_\mathcal{C} (x) \right\rangle  = (d +1) \left\langle H_\mathcal{C} (x) \right\rangle _{z=0} - d \left\langle H_\mathcal{C} (x) \right\rangle _{z=1}
#
# where :math:`d=2^k`, :math:`z=0,1` correspond to circuits with inserted
# channels :math:`\Psi _{0,1}`.
#

d = 2**k

randomized_cost_values = np.zeros_like(pauli_cost_values)

signs = np.array([1, -1], requires_grad=False)
shot_signs = signs[choices]

for i, cutoff in enumerate(shot_counts):
    costs = qaoa_cost(samples[:cutoff])
    randomized_cost_values[i] = (2 * d + 1) * np.mean(shot_signs[:cutoff] * costs)

######################################################################
# Let’s plot the results comparing the two methods:
#

fig, ax = plt.subplots(figsize=(12, 6))

ax.semilogx(
    shot_counts,
    pauli_cost_values,
    "o-",
    c="darkorange",
    ms=8,
    markeredgecolor="k",
    label="Pauli",
)

ax.semilogx(
    shot_counts,
    randomized_cost_values,
    "o-",
    c="steelblue",
    ms=8,
    markeredgecolor="k",
    label="Randomized",
)

ax.axhline(optimal_cost, color="k", linestyle="--", label="Exact value")

ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)

ax.set_ylabel("QAOA cost", fontsize=20)
ax.set_xlabel("Number of shots", fontsize=20)

ax.legend(frameon=True, loc="lower right", fontsize=20)

######################################################################
# We see that the randomized method converges faster than the Pauli method
# - fewer shots will get us a better estimate of the true cost function.
# This is even more apparent when we increase the number of shots and go
# to larger graphs and/or QAOA depths :math:`p`. For example, here are
# some results that include cost variances as well as mean values for a
# varying number of shots.
#
# .. figure:: ../demonstrations/quantum_circuit_cutting/shots_vs_cost_p1.svg
#     :align: center
#     :width: 70%
#
# .. figure:: ../demonstrations/quantum_circuit_cutting/shots_vs_cost_p2.svg
#     :align: center
#     :width: 70%
#
#     Figure 6. An example of QAOA cost convergence for a circuit cut both
#     with the Pauli method and randomized channel method.
#
# The randomized method offers a quadratic overhead reduction. In
# practice, for larger cuts, we see that it offers orders of magnitude
# better performance than the Pauli method. For larger circuits, even at
# :math:`10^6` shots, Pauli estimates still sometimes leave the allowed
# interval of :math:`[-1,1]`.
#
# However, these improvements come at the cost of increased circuit depth
# due to inserting random Clifford gates and additional classical
# communication required.
#
# Multiple cuts and mid-circuit measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Careful readers may have noticed that QAOA at depth :math:`p=1` has a
# specific structure of a `Matrix Product
# State <https://pennylane.ai/qml/demos/tutorial_tn_circuits.html>`__
# (MPS) circuit. However, in order to cut a :math:`p=2` QAOA circuit, we
# would need 2 cuts. This introduces some subtleties within the context of
# classical simulation that we point out here.
#
# The measurement performed as a part of the first cut always induces a
# reduced state on the remaining wires. If the circuit has an MPS
# structure, we can just measure all qubits at once - a part of the
# measured bitstring gets passed into the second fragment and the
# remaining bits go directly into the output bitstring. However, when we
# try the same thing on a non-MPS circuit, additional gates need to be
# applied on the wires what now hold a reduced state. This is the other
# reason why it is easier to simulate circuit cutting of a non-MPS circuit
# with a mixed-state simulator.
#
# .. figure:: ../demonstrations/quantum_circuit_cutting/mid_circuit_measure.svg
#     :align: center
#     :width: 90%
#
#     Figure 7. A schematic representation of the mid-circuit measurement
#     “problem”.
#
# Note that, in these cases, memory requirements of classical simulation
# are increased from :math:`O(2^n)` to :math:`O(4^n)`. However, this is
# only a constraint for classical simulation where we have to choose
# between state-vector and density-matrix approaches. Real quantum
# deveices don’t have such limitations, of course.
#
#
# References
# ----------
#
# .. [#Peng2019]
#
#     T. Peng, A. Harrow, M. Ozols, and X. Wu (2019) "Simulating Large Quantum Circuits on a Small Quantum Computer".
#     (`arXiv <https://arxiv.org/abs/1904.00102>`__)
#
# .. [#Lowe2022]
#
#     A. Lowe et. al. (2022) "Fast quantum circuit cutting with randomized measurements".
#     (`arXiv <https://arxiv.org/abs/2207.14734>`__)
#
#
# About the authors
# -----------------
#
# .. bio:: Gideon Uchehara
#    :photo: ../_static/avatar.webp
#
#    Gideon is a super cool person who works at Xanadu.
#
# .. bio:: Matija Medvidović
#    :photo: ../_static/matija_medvidovic.jpeg
#
#    Matija is a PhD student at Columbia University and the Flatiron Institute in New York. He works with machine learning methods to study quantum many-body physics and quantum computers. He is currently a part of the Xanadu residency program. He is a firm believer in keeping bios short and concise.
#
# .. bio:: Anuj Apte
#    :photo: ../_static/avatar.webp
#
#    Anuj is a super cool person who works at Xanadu.
