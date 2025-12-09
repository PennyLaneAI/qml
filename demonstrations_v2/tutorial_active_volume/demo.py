r"""Active volume
=================

- Introduction
    - fundamental problem/challenge of sparsity & idling qubits
    - active volume as new paradigm
    - inherently-quantum phenomena used to solve problem
    - anticipate structure

- Logical circuit to ZX
    - Established language
    - Example circuit: Adder (?)
    - ZX calculus as very useful alternative representation
    - express example circuit in ZX calculus

- ZX to logical blocks
    - ZX diagram is too general/unstructured to use for practical compilation
    - Introduce logical blocks
    - Impose constraints/rule set for logical blocks that guarantees existence of equivalent circuit
    - Express example circuit as oriented ZX diagram

- Logical circuit Interpretation
    - Express logical network as circuit again, using bridge qubits
    - Look at math of bridge qubits
    - reaction time

- Conclusion

- Appendix: Adder circuit?


Quantum compilation bridges the gap between quantum algorithms and
implementable operations on quantum hardware.
In addition to a bare translation between representations, we aim to optimize numerous key
metrics of the quantum executable during compilation, one of which is the spacetime volume of the computation.
Crucially, this **circuit volume** taken up by a computation is not only composed
of the information processing steps that are carried out for computing,
but also of the idling steps that any qubits will perform while they are
not part of the computation.

.. figure:: _static/demo_thumbnails/large_demo_thumbnails/pennylane-demo-active-volume-large-thumbnail.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

In this demo, we will look at the **active volume** of a quantum circuit,
which only measures the spacetime volume of the logical processing steps,
and the active volume aware compilation introduced by Litinski and Nickerson [#Litinski2022]_.
The realization that idling qubits during a quantum computation
represent wasted potential certainly was not made for the first time in this
work. The crucial contribution rather is a solution to this problem,
in the form of a systematic compilation framework that harvests
the wasted potential of idling qubits, or **idle volume**.
This framework combines the language of the ZX calculus with an abstraction of topologically
error-corrected quantum computers as well as the inherently-quantum technique of state teleportation.
The result is a powerful approach to make the cost of quantum computations
proportional to the active volume rather than the circuit volume.

In the following, we will compile the circuit for a subroutine step-by-step into a so-called
**logical network**, the representation of a quantum circuit ingested by an **active volume computer**.
For this, we will transform the circuit into a ZX diagram, impose
a rule set onto the diagram leading to an **oriented** ZX diagram, and then rewrite it as the
mentioned logical network. To understand the state teleportation leveraged during this rather
formal process, we finally take a look at an equivalent quantum circuit representation
of the obtained logical network.

Throughout, we will use a bottom-up approach and focus only on the required steps for our concrete
Clifford circuit example. We will throw another important aspect, the **reaction time**, into
the mix only later.
For a systematic top-down overview of the framework, section 1 of [#Litinski2022]_ is a great
source, and sections 2-6 work through the compilation of increasingly complex circuit components.

The input: a quantum circuit
----------------------------

The active volume framework can compile arbitrary quantum circuits, which we consider to
be the input; Clifford gates are supported directly, as we will see for our example below.
Discrete non-Clifford gates such as ``T`` or ``Toffoli`` are implemented in a surface
code-corrected quantum computer by ingesting magic states. And continuous non-Clifford gates such
as arbitrary-angle Pauli product rotations (PPRs) are supported via standard discretization techniques
like Gridsynth, repeat-until-success circuits, channel mixing, or phase gradient decompositions.
While this already enables the compiler to handle universal quantum circuits, compilation via
an intermediate representation in terms of Pauli product measurements (PPMs) allows for a
smooth integration with other compilation techniques such as those in the
Game of surface codes by Litinski. Correspondingly, you will find many circuit re-expressed in
terms of PPMs in [#Litinski2022]_.

The subroutine we will here compile step by step is a ladder of CNOT gates, which may be used to create
multi-qubit GHZ states, for example. As the CNOT gate is part of the Clifford group, so is the
full ladder. This simple example will allow us to focus on the formalization of active volume compilation into
logical networks and state teleportation as an elementary tool for quantum parallelization.
To make matters concrete, we will be concerned with compiling the following circuit:

.. figure:: ../_static/demonstration_assets/active_volume/example_circuit.png
    :align: center
    :width: 25%
    :target: javascript:void(0)

The labels :math:`|a\rangle` through :math:`|d\rangle` for the four input and output qubits
will be handy to keep track later on.

From circuits to ZX diagrams
----------------------------

The first step we take is to transform the quantum circuit to a ZX diagram, the object representing
linear maps in the ZX calculus.
The elementary building blocks of these diagrams are so-called spiders, drawn as a blue (X-type spider) or
orange (Z-type spider) vertex and a number of legs, each of which is a linear map itself.
Composing these spiders into diagrams is then as easy as to
connect legs of multiple spides, turning them into edges between the vertices. Unconnected legs
of a diagram form inputs and outputs of the represented linear map. ZX diagrams can express universal
quantum circuits (and even more maps that are not circuits), but for active volume compilation we
only need to express Clifford circuits, so that we may restrict ourselves to phase-free spiders:

IMAGE: ADAPT FIG 6.

As we can see, the CNOT gate is easily expressed in terms of an X and a Z spider, each with three legs.
This means that we can just as easily rewrite our CNOT ladder into a ZX diagram:

.. figure:: ../_static/demonstration_assets/active_volume/example_zx_basic.png
    :align: center
    :width: 25%
    :target: javascript:void(0)

We find six spiders, three of each type, with three legs each.

At this point, we not only exchanged circuit symbols for colored circles, though. In the language
of ZX calculus, the internal edges that connect vertices do no longer have a temporal meaning, so
that the geometry of the diagram becomes irrelevant. Only the represented graph carries meaningful
information, as long as we associate the unconnected legs of the diagram with fixed inputs and
outputs.
This allows us to rewrite our diagram as follows (rotating it to make it easier to display):

.. figure:: ../_static/demonstration_assets/active_volume/example_zx_flat.png
    :align: center
    :width: 25%
    :target: javascript:void(0)


Imposing structure: oriented ZX diagrams and logical networks
-------------------------------------------------------------

Expressing the quantum circuit as a less structured ZX diagram allows for convenient manipulation,
but it also causes trouble: ultimately we want to compile the circuit for a quantum computer
that operates with logical qubits implemented as surface code-corrected patches of physical qubits,
so we will want to gain back structure.
For this, [#Litinski2022]_ introduces **oriented ZX diagrams** which must satisfy the following
rules:

    #. Each spider must have two, three or four legs.
    #. Each vertex has six ports (north (N), up (U), east (E), south (S), down (D), and west (W))
       and at most one leg can be connected to each port.
       For each vertex, the two ports in one of three pairs (N, S), (U, D), and (E, W) must both
       be unoccupied. This pair is called the **orientation** of the vertex/spider and we denote it by
       N, U or E accordingly.
    #. Input (output) legs must connect to the down (up) port of a vertex.
    #. Edges (internal legs) must connect to ports at both vertices from the same of the three pairs.
       That is, they must connect to the same port of the two vertices they connect, or to the "opposite" one.
    #. Edges (internal legs) must connect vertices of the same type (X/Z) and same orientation, or
       of different types and different orientations. (For Hadamarded edges, same (different) types with different
       (same) orientations may be connected.)

These rules may seem quite artificial and confusing. This is because they impose constraints
on the--otherwise quite unstructured--ZX diagram that originate in the structure of the surface
code and the arrangement of the overall computation in space and time.
Concretely, the ports named after cardinal directions (N, E, S, W) correspond to the four directions in which a
square patch encoding a logical surface code qubit can interact with other patches, and the
remaining directions (U, D) symbolize the time axis, motivating the input/output constraint (3.).

The contraints on edges then encode the requirements for the oriented ZX diagram to represent
a computation that can be arranged in (2+1) dimensions (surface+time) without collisions and
without inconsistencies in the boundaries of the logical qubits and their parity check measurements.
The achieved consistency then allows us to perform the required joined measurements for lattice surgery
to implement multi-qubit operations.

Before we turn to our example circuit, let's produce a valid oriented ZX diagram for a single CNOT
as a warmup exercise.
The standard ZX diagram has two vertices, so we are tempted to simply take this diagram and assign
ports to each of the three legs, at each vertex.
Due to the input/output constraint, most choices actually are already made for us:

.. figure:: ../_static/demonstration_assets/active_volume/cnot_oriented_zx_incomplete.png
    :align: center
    :width: 25%
    :target: javascript:void(0)

Now, we may choose any of the remaining ports (N, E, S, W) of the Z spider (orange) for the
internal edge. We know that it must connect to the same or opposite port at the X spider (blue) due
to (4.), forcing the orientation of both spiders to be the same. However, as they differ in type, such
a connection is forbidden by (5.).
We thus ran out of options for the internal edge. What can we do to fix this issue?
The solution lies in the identity (marker?) for ZX diagrams; before we orient the diagram,
we may insert additional vertices with two legs, in place for a simple leg.
This allows us to gain more room in order to gradually change the occupied ports of the oriented
vertices before switching between vertex types.
As it turns out, inserting a single vertex in the CNOT diagram is not sufficient, but we need to
insert two vertices:

.. figure:: ../_static/demonstration_assets/active_volume/cnot_oriented_zx_extended.png
    :align: center
    :width: 25%
    :target: javascript:void(0)

Let us try to connect the edges again. We connect the two Z spiders with a (S-N) edge, giving them
both an E-orientation. Coming from the right side, we connect the X spiders with an (E-W) edge,
giving them an N-orientation:

.. figure:: ../_static/demonstration_assets/active_volume/cnot_oriented_zx_extended_1.png
    :align: center
    :width: 25%
    :target: javascript:void(0)

Here we indicated the orientation of the vertices with internal lines, serving as a mnemonic that
these ports are "occupied" already.
Now we can use the N and S ports of the central vertices to route the last edge. We choose both
N ports, because this will be a convenient choice below, and arrive at the oriented ZX diagram
for a single CNOT:

.. figure:: ../_static/demonstration_assets/active_volume/cnot_oriented_zx_extended_1.png
    :align: center
    :width: 25%
    :target: javascript:void(0)

"""


######################################################################
#
# Conclusion
# ----------
#
# References
# -----------
#
# .. [#Litinski2022]
#
#    Daniel Litinski and Naomi Nickerson.
#    "Active volume: An architecture for efficient fault-tolerant quantum computers with limited non-local connections",
#    `arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`__, 2022.
#
# .. [#Gidney2018]
#
#    Craig Gidney. "Halving the cost of quantum addition",
#    `Quantum 2, 74 <https://quantum-journal.org/papers/q-2018-06-18-74/>`__, 2018,
#    `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`__, 2017.
#
