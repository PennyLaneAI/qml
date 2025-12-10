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
In addition to a bare translation between representations, it aims to optimize numerous key
metrics of the quantum executable during compilation, one of which is the spacetime volume
of the computation.
Crucially, this **circuit volume** taken up by a computation is not only composed
of the information processing steps that are carried out for computing,
but also of the idling that qubits will undergo while they are not part of the computation.

.. figure:: _static/demo_thumbnails/large_demo_thumbnails/pennylane-demo-active-volume-large-thumbnail.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

In this demo, we will look at the **active volume** of a quantum circuit,
which only measures the spacetime volume of the logical processing steps,
and the active volume aware compilation introduced by Litinski and Nickerson [#Litinski2022]_.
The realization that idling qubits during a quantum computation
represent wasted potential was not made for the first time in this
work. The crucial contribution rather is a solution to this problem,
in the form of a systematic compilation framework that harvests
the wasted potential of idling qubits, or **idle volume**.
This framework combines the language of the ZX calculus with an abstraction of topologically
error-corrected quantum computers as well as the inherently-quantum technique of state teleportation.
The result is a powerful approach to make the cost of quantum computations
proportional to the active volume rather than the circuit volume, provided that we operate with a
so-called **active volume computer**. We will not go into detail about this abstraction but
emphasize that the compilation framework presented in the following is tailored specifically to this
type of quantum computer.

In the following, we will compile the circuit for a subroutine step-by-step into a so-called
**logical network**, the representation of a quantum circuit ingested by an active volume computer.
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

Without further a-do, let's get started!

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

Our example circuit: CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

ZX diagram for the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    :width: 45%
    :target: javascript:void(0)


Imposing structure: oriented ZX diagrams
----------------------------------------

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

Exercise: oriented ZX diagram of a CNOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we turn to our example circuit, let's produce a valid oriented ZX diagram for a single CNOT
as a warmup exercise.
The standard ZX diagram has two vertices, so we are tempted to simply take this diagram and assign
ports to each of the three legs, at each vertex.
Due to the input/output constraint, most choices actually are already made for us:

.. figure:: ../_static/demonstration_assets/active_volume/cnot_oriented_zx_incomplete.png
    :align: center
    :width: 20%
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
    :width: 35%
    :target: javascript:void(0)

Let us try to connect the edges again. We connect the two Z spiders with a (S-N) edge, giving them
both an E-orientation. Coming from the right side, we connect the X spiders with an (E-W) edge,
giving them an N-orientation:

.. figure:: ../_static/demonstration_assets/active_volume/cnot_oriented_zx_extended_1.png
    :align: center
    :width: 35%
    :target: javascript:void(0)

Here we indicated the orientation of the vertices with internal lines, serving as a mnemonic that
these ports are "occupied" already.
Now we can use the U and D ports of the central vertices to route the last edge. We choose both
U ports, because this will be a convenient choice below, and arrive at the following oriented
ZX diagram for a single CNOT:

.. figure:: ../_static/demonstration_assets/active_volume/cnot_oriented_zx_complete.png
    :align: center
    :width: 35%
    :target: javascript:void(0)

Note that we could also connect the input state :math:`|b\rangle` to the D port of the left X
spider.

Oriented ZX diagram of the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having figured out the oriented ZX diagram for a single CNOT gate, we turn to the
ZX diagram of the CNOT ladder. As you may guess, compared to the unoriented diagram
we arrived at above, we will again need to insert more nodes.
One way to do this would be to simply copy the oriented diagram for the single CNOT
three times and to route outputs of one copy into the inputs of the next.
This would lead to an arrangements with "depth" three.
However, here we want to keep the full circuit in a single layer.
With a similar reasoning as for the single CNOT, we find that we need to insert six nodes
overall, arriving at the oriented ZX diagram

.. figure:: ../_static/demonstration_assets/active_volume/example_oriented_zx_complete.png
    :align: center
    :width: 55%
    :target: javascript:void(0)

As for the single CNOT, this is not a unique solution.


Oriented ZX diagram to logical network
--------------------------------------

The last step required to arrive at logical networks is actually quite small; the major effort
is already done with the orientation of the ZX diagram.

We introduce **logical blocks**, which are just the nodes from an oriented ZX diagram, drawn
as hexagons to mark the six ports more clearly, with a modification of rule (4.) above:

    4'. Edges (internal legs) must connect to **the same port** at both logical blocks.

This turns out to be quite a small modification.
**Logical networks** are diagrams, or networks, of logical blocks. We label the logical blocks
uniquely and replace drawing edged between them by annotating the ports with the labels of the
blocks they connect to.

Logical network for the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looking at the oriented ZX diagram of the CNOT ladder above, we see that the inserted spiders
with two legs are connected to the original three-legged spiders with edges that do not satisfy
the modified rule (4'.) yet. However, in each case, we may simply use the opposite port of those
two-legged spiders. Changing the shape of the vertices, labeling them, and replacing drawn
edges with vertex labels then leads us to a valid logical network already:

.. figure:: ../_static/demonstration_assets/active_volume/example_logical_network.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

And this already concludes the compilation process of this simple Clifford operation.
You may ask what we do with the logical network representation, and why we would prefer twelve
custom blocks--with new complicated rules governing them--over a good old circuit diagram with
three CNOTs. We turn to those questions next.

Why logical networks?
---------------------

The properties of logical blocks and the set of rules we impose on logical networks are made such
that any valid logical network can be realized with surface code-corrected logical qubits.
While this is true for quantum circuit diagrams as well, logical networks allow us to trade space
and time against each other. In a sense, logical networks distill the best out of the two worlds of
quantum circuits and ZX diagrams; ZX diagrams allow for continuous deformations where
"rigid" quantum circuits do not, but the additional structure of logical networks that resembles
quantum circuits ensures that we do not venture too far into the space of abstract representations
as could happen with pure ZX diagrams.

We saw above that a single CNOT gate leads to four vertices in the oriented ZX diagram, and its
logical network has four blocks too, requiring four qubits for execution in a single
time step. While this may seem counterintuitive, it reproduces the footprint of a CNOT implemented
via lattice surgery; there, we need to bring both the X and the Z boundaries of the two patches
encoding the input qubits into contact, leading to a 2-by-2 square of patches overall.

For the example of the CNOT ladder, we saw how a sequence of three Clifford gates can be
transformed into a layer of simultaneously applied logical blocks, i.e., we **parallelized** the
computation even though the individual operations **do not commute**.
As we will explain below in more detail, this powerful transformation is only possible due to
quantum state teleportation, and it becomes _useful_ because
teleportation ultimately is implemented through measurements, the native
language of error corrected qubits.
Note that because we had to add vertices to orient the ZX diagram, we require eight additional qubits
in order to execute the fully parallelized logical network. However, we are not forced to maximize
the parallelization; as we mentioned above, we instead could have concatenated the oriented ZX
diagrams for the three individual CNOT gates, or parallelized only two of them.
This flexibility allows active volume compilation to adjust the logical network to the available
hardware.
In this sense, logical networks form a much more flexible representation of operations on an
error-correct quantum computer, promoting space-time tradeoffs to first-class program
transformations.

Under the hood of parallelization: state teleportation
------------------------------------------------------

The CNOT ladder example circuit has a fundamentally sequential appearance in the circuit picture,
so that parallelizing it seems unintuitive. There are two ingredients needed to understand that
it is possible, and feasible, to do so nonetheless:
the fundamental possibility to parallelize the operations without breaking physics, and the fact
that CNOT gates on surface code-corrected logical qubits can be implemented through measurements
anyways, using lattice surgery.

Physical soundness
~~~~~~~~~~~~~~~~~~

For the first point, let's consider a somewhat more abstract setup, as is done in
Fig. 10b),c) in [#Litinski2022]_. Suppose we want
to sequentially execute two two-qubit gates :math:`A` and :math:`B` that overlap on one qubit
and do not commute:

.. figure:: ../_static/demonstration_assets/active_volume/teleportation_start.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

We also assume that we can track the impact that :math:`B` has on Pauli operators, i.e.,
what happens to a Pauli operator when pulling it through :math:`B`.

For the parallelization, we first insert a state teleportation circuit between :math:`A`
and :math:`B` using an auxiliary qubit, see [link] for details:

.. figure:: ../_static/demonstration_assets/active_volume/teleportation_insert_teleport.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Next, we pull the classically controlled correction gates, which are Pauli operators, through
the gate :math:`B`, and obtain new correction gates :math:`C_Z=B^\dagger X B` and :math:`C_X=B^\dagger Z B`:


.. figure:: ../_static/demonstration_assets/active_volume/teleportation_commuted.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

At this point, we can move :math:`B` to the front, effectively parallelizing it with :math:`A`.
As we can see, the state teleportation together with the ability to track correction gates
through :math:`B` makes it possible to implement the two non-commuting operations :math:`A` and
:math:`B` simultaneously, without violating any physical laws or causal dependencies.

Practical usefulness
~~~~~~~~~~~~~~~~~~~~

The previous discussion shows that it is _possible_ to parallelize non-commuting gates at the cost
of an additional qubit (we need a communication channel "back in time"), a Bell state preparation
and Bell basis measurement (we need to entangle the communication channel), and
classical compute (we need to transform the correction gates by :math:`\text{Ad}_B`).
However, it is not so clear whether this is a useful tradeoff. For example, we could
apply the same technique when using a NISQ computer, but we would potentially even increase the
two-qubit gate depth, and additional qubits are quite expensive to come by.

For the surface code corrected quantum computer, and specifically for the active volume computer,
the creation of qubit pairs in a Bell state and the Bell basis measurement are assumed to be much
cheaper than arbitrary logical operations, because they can just be woven into the measurements, i.e.,
the code cycles, of the error correction code via lattice surgery.
Similarly, Pauli correction gates are anyways tracked in software throughout, so that the only
true price we are paying for the parallelization is the extra memory.
As we are trying to condense a computation by parallelizing it and reducing idle volume, this
tradeoff will often be benefitial.

As we can see, parallelization via state teleportation thus is a very good fit for active volume
computers and the goal of our compilation. This is why the framework by Litinski and
Nickerson [#Litinski2022]_ promotes parallelization to a first-class optimization, as
a direct consequence of the intermediate representation as a ZX diagram.

Teleportation in the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall the logical network into which we compiled the CNOT ladder earlier:

.. figure:: ../_static/demonstration_assets/active_volume/example_logical_network.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

We can identify



Reactive measurements and reaction time
---------------------------------------

to do

Conclusion
----------

to do


Appendix: Compiling a Gidney adder
----------------------------------


References
-----------

.. [#Litinski2022]

   Daniel Litinski and Naomi Nickerson.
   "Active volume: An architecture for efficient fault-tolerant quantum computers with limited non-local connections",
   `arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`__, 2022.

.. [#Gidney2018]

   Craig Gidney. "Halving the cost of quantum addition",
   `Quantum 2, 74 <https://quantum-journal.org/papers/q-2018-06-18-74/>`__, 2018,
   `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`__, 2017.

"""
