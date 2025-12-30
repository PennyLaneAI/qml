r"""Active volume compilation
=============================

In this demo, we will explore the concept of active volume of a quantum computation,
and how compilation can exploit this concept to reduce the resources required to execute the
computation on a suitable machine–an active volume quantum computer.
We will look at circuits, :doc:`ZX diagrams <demos/tutorial_zx_calculus>`, and a new type of
circuit representation called logical networks.
This demo is directly based on a seminal paper by Litinski and Nickerson [#Litinski2022]_.
As the original work already does a great job at presenting the concepts and compilation
techniques at multiple levels of detail and with plenty of visualizations, we will aim to
take a complementary perspective, and walk slowly through a specific compilation example.

For readers unfamiliar with ZX diagrams, it will be useful (but not necessary) to take a look at
our :doc:`introduction to ZX calculus <demos/tutorial_zx_calculus>` first.
Further, it can be instructive to get a :doc:`primer on lattice surgery <demos/tutorial_lattice_surgery>`
and dive into the :doc:`Game of Surface Codes <demos/tutorial_game_of_surface_codes>`, but this
is not a requirement either.

.. figure:: _static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-active-volume-open-graph.png
    :align: center
    :width: 65%
    :target: javascript:void(0)

:doc:`Quantum compilation <compilation/>` is fundamentally about bridging the
gap between high-level descriptions of
quantum algorithms, and low-level instructions that are actually executable on quantum hardware.
In addition to the bare translation between those representations, it also aims to optimize
numerous key metrics of the quantum program during the compilation process. Typically, these
metrics focus on specific properties of the program such as qubit count, and various gate counts,
which often stand in contention against each other. To optimize effectively, we’d ideally like a
metric that expresses the combined cost of all resources used by a program. One such metric is
the **spacetime volume** cost, which in fault-tolerant architectures can be understood as the
total error-corrected qubits (space) taken up by the computation, times the total error
correction cycles (time) required to perform it.

Intuitively, the **circuit volume** can be understood as the total “area” taken up by a
circuit, as depicted below. The crucial insight in the concept of **active volume** then is
the idea that not all of this “area” is dedicated to performing useful computation. In fact,
ladder circuits like those used in reversible arithmetic may consist to a large portion
(if not the majority) of idling qubits!
We can thus partition a circuit into computationally *active* volume and *idle* volume:

.. figure:: _static/demonstration_assets/active_volume/active_vs_idle.png
    :align: center
    :width: 85%
    :target: javascript:void(0)

    Active and idle volumes are represented by areas occupied by gates (green)
    and areas without gates (red), respectively, in a standard circuit diagram.
    Adapted from [1].

In this demo, we will look at how to obtain the active volume of a quantum circuit
(in terms of so-called “logical blocks”), and the systematic compilation framework introduced by
Litinski and Nickerson [#Litinski2022]_, which then can be used to maximize the use of available
computational resources. This framework combines the language of the ZX calculus with an
abstraction of topologically error-corrected quantum computers, as well as the
inherently-quantum technique of state teleportation.
The result is a powerful approach to make the cost of quantum computations proportional to
the active volume rather than the circuit volume, provided that we operate with a so-called
**active volume computer**. We will not go into detail about the definition of such a computer,
but only outline its rough characteristics.

.. admonition:: Active volume computer
    :class: note

    The cost model used by active volume compilation is founded on an abstract error-corrected
    quantum computer, the active volume computer.
    It is made up of a fixed number of *qubit modules*, each of which stores a logical qubit.
    Half of the modules are designated computation modules, in which information is processed,
    while the other half consists of dedicated memory modules, which allow for intermediate
    storage and routing of information between computational steps.
    The following characteristics are crucial assumptions about the computer that shape the cost
    model:

    #. The information content of two qubit modules within range can be
       exchanged quickly/cheaply. They are "quickswappable".
    #. Single qubit modules can be prepared quickly/cheaply in the state :math:`|0\rangle` or :math:`|+\rangle`.
       They can also be measured quickly/cheaply in the Pauli-:math:`X` or Pauli-:math:`Z` basis.
    #. Two qubit modules within some specified range :math:`r` can be prepared quickly/cheaply in the Bell
       state :math:`|\phi\rangle = \tfrac{1}{\sqrt{2}} (|0\rangle +|1\rangle)`.
       They can also be measured quickly/cheaply in the Bell basis.
    #. Choosing the measurement basis of one or two qubits dynamically costs one *reaction time*
       unit. The relation between the reaction time and the time taken by the above cheap
       operations is a property of the computer.
    #. Qubit modules can execute so-called logical blocks, which is a slow/expensive operation.
       Computations are represented by logical networks (see below) made up of connected logical
       blocks. The qubit modules executing connected blocks must be within range.
    #. Qubit modules can be grouped and occupied for some time to distil magic states. This is
       a slow/expensive operation as well.

    A more detailed characterization is provided in Sec. 1 of [1].

    Given the characterization above, active volume compilation aims to maximize the utilization
    of the computational qubit modules, because they execute the slow operations (logical blocks
    and magic state distillation). Memory qubit modules are allowed to sit idle, providing
    intermediate storage and space for routing information through quickswaps. Note that
    long-range communication in the active volume computer is achieved through layers
    of quickswaps, which remain cheap in the above cost model.

In the following, we will demonstrate how to compile the circuit of a subroutine
step-by-step into a so-called **logical network**, which is a representation that allows
the active volume computer to use its computing resources with high efficiency.
For this, we will transform the circuit into a ZX diagram, impose a set of rules that
lead us to an **oriented** ZX diagram, and then rewrite it into the aforementioned
logical network. This representation will allow us to capture the essence of quantum information
processing without the restrictive representation as quantum circuits, while enforcing enough
structure to arrive at a program that can be executed by an active volume computer.

To put this rather formal process into context, we will analyze the same
circuit under the lens of parallelization via quantum state teleportation, and relate the
result to the derived logical network.
The concepts are presented in a bottom-up approach and focus on the required steps to compile
our concrete Clifford circuit example. We will look at another important concept,
the **reaction time**, towards the end of the demo. For a top-down overview of the active
volume framework, section 1 of [1] is a great source, while
sections 2-6 work through the compilation of increasingly complex circuit components.

Without further a-do, let's get started!

The input: a quantum circuit
----------------------------

The active volume framework can compile arbitrary quantum circuits, which we consider to
be the input; Clifford gates are supported directly, as we will see for our example below.
Discrete non-Clifford gates such as ``T`` or ``Toffoli`` are implemented in a surface
code-corrected quantum computer by ingesting magic states via multi-qubit measurements.
And continuous non-Clifford gates such as arbitrary-angle Pauli product rotations (PPRs)
are supported via standard discretization techniques like
`Gridsynth <https://arxiv.org/abs/1403.2975v3>`__,
`repeat-until-success circuits <https://arxiv.org/abs/1404.5320>`__,
`channel mixing <https://quantum-journal.org/papers/q-2023-12-18-1208/>`__, or
:doc:`phase gradient decompositions <compilation/phase-gradient>`.
While this already enables the compiler to handle universal quantum circuits, compilation
via an intermediate representation in terms of Pauli product measurements (PPMs) allows for
a smooth integration with other compilation techniques such as those in the
Game of Surface Codes by Litinski. Correspondingly, you will find many circuits re-expressed
in terms of PPMs in the active volume paper.

Our example circuit: CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The subroutine we will compile step by step is a ladder of CNOT gates, which may be used
to create multi-qubit GHZ states, for example. As the CNOT gate is part of the Clifford group,
so is the full ladder. This simple example will allow us to focus on the formalization of active
volume compilation into logical networks. We will later compare the compilation result to the
composition of the networks for each individual CNOT, parallelized via **state teleportation**.
To make matters concrete, we will be concerned with compiling the following circuit:

.. figure:: _static/demonstration_assets/active_volume/example_circuit.png
    :align: center
    :width: 35%
    :target: javascript:void(0)

The labels :math:`|a\rangle` through :math:`|d\rangle` for the four input and output
qubits will be handy to keep track later on.

From circuits to ZX diagrams
----------------------------

The first step we take is to transform the quantum circuit to a ZX diagram, the object
representing linear maps in the :doc:`ZX calculus <demos/tutorial_zx_calculus>`.
We briefly outline some key building blocks and properties of the calculus here, and defer
to the linked tutorial and its references for more details.

The elementary building blocks of a ZX diagram are so-called spiders, drawn as a
blue (X-type spider) or orange (Z-type spider) vertex and a number of legs. Each spider
represents a linear map itself and composing spiders into diagrams is as easy as connecting their
legs, creating edges between the vertices. Unconnected legs of a
diagram symbolize inputs and outputs of the represented linear map.
Some basic examples are summarized in the following overview figure.

.. figure:: _static/demonstration_assets/active_volume/zx-calculus-overview.png
    :align: center
    :width: 100%
    :target: javascript:void(0)

    Basic building blocks and transformation rules of the ZX calculus.
    Adapted from [1].

ZX diagrams in principle can express universal quantum circuits (and even more maps that are
not circuits), but for active volume compilation we only need to express Clifford circuits
and Pauli product measurements (PPMs), whose measurement bases could be classically conditioned.
Therefore, we may restrict ourselves to phase-free spiders and in particular the building blocks
shown above. The main reason for this is that non-Clifford operations, such as T or Toffoli
gates, are realized via
:doc:`magic state injection <demos/tutorial_mcm_introduction#t-gadget-in-pennylane>`
in surface code computations, and that classical mixtures in the form of classically
conditioned operations can be translated into PPMs with a conditionally chosen basis.

ZX diagram for the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As we can see in the previous figure, the CNOT gate is easily expressed in terms of a diagram
with one X and one Z spider, each with three legs. This means that we can just as easily rewrite
our CNOT ladder into a ZX diagram:

.. figure:: _static/demonstration_assets/active_volume/example_zx_basic.png
    :align: center
    :width: 35%
    :target: javascript:void(0)

We find six spiders, three of each type, with three legs each.

At this point, we didn’t only exchange circuit symbols for colored circles, though. In the
language of the ZX calculus, the internal edges that connect vertices no longer have a temporal
meaning, so that the geometry of the diagram becomes irrelevant. Only the represented
graph carries meaningful information, as long as we associate the unconnected legs of the
diagram with fixed inputs and outputs.
This allows us to rewrite our diagram as follows (rotating it to make it easier to display):

.. figure:: _static/demonstration_assets/active_volume/example_zx_flat.png
    :align: center
    :width: 55%
    :target: javascript:void(0)

Note how the inherent time ordering of the CNOT ladder is no longer represented in this diagram.
While this looks like a small, parallelized implementation of the ladder, we will need to work a
little more to guarantee that the diagram actually represents a map that can be implemented
conveniently on a quantum computer.

Imposing structure: oriented ZX diagrams
----------------------------------------

Expressing the quantum circuit as a ZX diagram with little structure allows for
convenient manipulation, but it also causes trouble: ultimately we want to compile the circuit
for a quantum computer that operates with logical qubits implemented as surface
code corrected patches of physical qubits, so we will need to gain back some structure.
For this, Litinski and Nickerson introduce **oriented ZX diagrams**, which must satisfy the
following rules, illustrated below:

#. Each spider must have two, three or four legs.
#. Each spider has six ports (north (N), up (U), east (E), south (S), down (D), and west (W))
   and at most one leg can be connected to each port.
   For each spider, the two ports in one of three pairs (N, S), (U, D), and (E, W) must both
   be unoccupied. This pair is called the **orientation** of the spider and we denote it
   by N, U or E accordingly.
#. Input (output) legs must connect to the down (up) port of a spider.
#. Edges (internal legs) must connect to ports from the same of the three pairs at both spiders.
   That is, they must connect to the same port of the two spiders they connect, or to
   opposite ports.
#. Edges (internal legs) must connect spiders of the same type (X/Z) and same orientation, or
   of different types and different orientations. (For Hadamarded edges, same (different) types
   with different (same) orientations must be connected.)

[insert image: visualization of rules]

These rules may seem quite artificial and confusing. This is because they impose constraints
on the--otherwise quite unstructured--ZX diagram that originate in the structure of the surface
code and the arrangement of the overall computation in space and time.
Concretely, the ports named after cardinal directions (N, E, S, W) correspond to the four
directions in which a square patch encoding a logical surface code qubit can interact with
other patches, and the remaining directions (U, D) symbolize the time axis, motivating
the input/output constraint (3.).

The constraints on edges then encode the requirements for the oriented ZX diagram to represent
a computation that can be arranged in (2+1) dimensions (surface+time) without collisions and
without inconsistencies in the boundaries of the logical qubits and their parity check measurements.
The achieved consistency allows us to perform the required joint measurements for
lattice surgery to implement multi-qubit operations.

Exercise: oriented ZX diagram of a CNOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we turn to our example circuit, let's produce a valid oriented ZX diagram for a single
CNOT as a warmup exercise. The standard ZX diagram has two spiders, so we are tempted to
simply take this diagram and assign ports to each of the three legs, at each spider.
Due to the input/output constraint, the qubit state labels go into the U and D ports:

.. figure:: _static/demonstration_assets/active_volume/cnot_oriented_zx_incomplete.png
    :align: center
    :width: 20%
    :target: javascript:void(0)

Now, we may choose any of the remaining ports (N, E, S, W) of the Z spider (orange) for
the internal edge. We know that it must connect to the same or opposite port at the X spider
(blue) due to rule (4.), forcing the orientation of both spiders to be the same.
However, as they differ in type (X vs. Z), such a connection is forbidden by rule (5.).
We thus already run out of options for the internal edge. What can we do to fix this issue?
The solution lies in the identity ``─●─ = ──`` for ZX diagrams; before we orient the diagram,
we may insert additional spiders with two legs, in place of a simple leg. This allows us to
gain more room in order to gradually change the occupied ports of the oriented spiders,
before switching between spider types.
As it turns out, inserting a single spider in the CNOT diagram is not sufficient,
and we need to insert two spiders:

.. figure:: _static/demonstration_assets/active_volume/cnot_oriented_zx_extended.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Let us try to connect the edges again. We connect the two Z spiders with a (S-N) edge, giving
them both an E-orientation. Coming from the right side, we connect the X spiders with an
(E-W) edge, giving them an N-orientation:

.. figure:: _static/demonstration_assets/active_volume/cnot_oriented_zx_extended_1.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Here we indicated the orientation of the spiders with internal lines, serving as a mnemonic
that these ports are "occupied" already.
Now we can use the U and D ports of the central spiders to route the last edge. We choose
both U ports, because this will be a convenient choice below, and arrive at the following
oriented ZX diagram for a single CNOT:

.. figure:: _static/demonstration_assets/active_volume/cnot_oriented_zx_complete.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is a valid oriented ZX diagram for a single CNOT gate. Note that the spiders we had
to insert above directly correspond to additional information processing steps, and will take up
additional qubits in the execution of the CNOT. This may come as a surprise, given that
CNOT is a simple two-qubit Clifford gate. However, note that the active volume computer
architecture operates with :doc:`lattice surgery <demos/tutorial_lattice_surgery>`, and
executing a CNOT with lattice surgery also requires two additional logical qubits as intermediate
bridge space. Thus, the oriented ZX diagram simply reproduces the true cost for a CNOT in this
type of architecture.

Note that we could also connect the input state :math:`|b\rangle` to the D port of the left X spider.
This is because we could have inserted the additional blue spider on the output leg of the existing
one, rather than between the inserted orange and the existing blue spiders.
For this, the orientation of the spiders could be kept as-is.

Oriented ZX diagram of the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having figured out the oriented ZX diagram for a single CNOT gate, we turn to the ZX diagram
of the CNOT ladder. As you may guess, compared to the unoriented diagram we arrived at above, we
will again need to insert more nodes.
One way to do this would be to simply copy the oriented diagram for the single CNOT three times
in the vertical (time) direction, and to route outputs of one copy into the inputs of the next.
This would lead to an arrangement with "depth" three.
However, here we want to keep the full circuit in a single layer. With a similar reasoning as for
the single CNOT, we find that we need to insert six nodes overall, arriving at the oriented ZX diagram:

.. figure:: _static/demonstration_assets/active_volume/example_oriented_zx_complete.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

In order to connect the “individual” CNOT networks side-by-side, we have moved some of the input
qubit legs to what was previously an “internal” node. This is allowed as there is no preference
for where to place logical inputs and outputs in a network. Note how some of the qubits appear
to be routed laterally through the network, hinting at the parallelization of what were previously
sequential gates. As for the single CNOT, this is not a unique solution.
We are almost done with the parallelization of the CNOT ladder; one last step will be needed to
transform the oriented ZX diagram into a logical network, which can then be executed on an
active volume computer.

Oriented ZX diagram to logical network
--------------------------------------

The last step required to arrive at logical networks is comparatively small; the major effort
is already done with the orientation of the ZX diagram.

We introduce **logical blocks**, which are just the nodes from an oriented ZX diagram,
drawn as hexagons to mark the six ports more clearly, with a modification of rule (4.) above:

    4'. Edges (internal legs) must connect to **the same port** at both logical blocks.

Depending on the concrete circuit we are compiling, this change might just lead to minor
modifications such as relabeling the ports of two connected blocks, or it might lead to cascades of
re-routing that eventually force us to insert a new block, even.
**Logical networks** are diagrams, or networks, of logical blocks. We label the logical blocks
uniquely and replace drawing edges between them by annotating the ports with the labels of the
blocks they connect to. This allows us to separate logical blocks, which represent computation
and are considered expensive, from the block connectivity, which represents communication between
blocks and is considered cheaper because it is implemented via fast SWAP connections (also see
active volume computer info box).

Logical network for the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looking at the oriented ZX diagram of the CNOT ladder above, we see that the inserted spiders
with two legs are connected to the original three-legged spiders with edges that do not satisfy
the modified rule (4'.) yet. However, in each case, we may simply use the opposite port of those
two-legged spiders. We then arrive at a valid logical network by performing a few cosmetic
changes: first, change the shape of the spiders to hexagons, with each corner corresponding
to a port. Second, enumerate the hexagons for referencing. Any labels work, really. Third and last,
cut the edges between hexagons and instead draw shortened legs, annotated by the label of the
hexagon they connect to.
For our example, we find the logical network

.. figure:: _static/demonstration_assets/active_volume/example_logical_network.png
    :align: center
    :width: 95%
    :target: javascript:void(0)

And this already concludes the compilation process of this simple Clifford operation, arriving at
a fully parallelized computation.

You may ask what we do with the logical network representation,
and why we would prefer twelve custom blocks--with new complicated rules governing them--over a
good old circuit diagram with three CNOTs.
A key advantage of the logical network representation lies in its homogeneous expression of
computational steps within the active volume computer, c.f. the info box at the top;
all computation is made up of logical blocks that can be scheduled densely in the computational
qubit modules of the computer in order to remove idle volume. If logical networks turn out too
large for the space available at a given time step, they can be split into two networks that
are executed sequentially.

For our example circuit, this idea of gradual parallelization generalizes the idea that we are
not forced to maximize parallelization; as we mentioned above, we instead could have concatenated
the oriented ZX diagrams for the three individual CNOT gates, or parallelized only two of them, if
we had less computational space available. Abstractly, we can draw this as sequences of
logicel networks that make up different shapes:

[insert image: tetris-like options for the shape of a network]

Logical networks thus form a much more flexible representation of operations on an error-correct
quantum computer, allowing them to be adjusted to available hardware resources and promoting
space-time tradeoffs to first-class program transformations.

From a mathematical perspective, logical networks distill the best out of the two worlds of
quantum circuits and ZX diagrams; ZX diagrams allow for continuous
deformations where "rigid" quantum circuits do not, but the additional structure of logical
networks that resembles quantum circuits ensures that we do not venture too far into the space
of abstract representations, as could happen with pure ZX diagrams.

Under the hood of parallelization: state teleportation
------------------------------------------------------

In the CNOT ladder example circuit, which has a fundamentally sequential appearance in the
circuit picture, logical network compilation achieved its parallelization by combining the
effects of the three CNOTs into a single effect, creating a “CNOT ladder subroutine”.
If we want to parallelize multiple blocks,
without having to re-compile their joint logical network, we can do this procedurally through
state teleportation, using a so-called bridge qubit. This technique parallelizes non-commuting
operations without breaking physics, and it pays off because Bell state preparation and
measurements are assumed to be fast on an active volume computer (see info box at the top).

Physical soundness
~~~~~~~~~~~~~~~~~~

For the first point, let's consider a somewhat more abstract setup, as is done in Fig. 10b),c)
in [#Litinski2022]_. Suppose we want to sequentially execute two two-qubit gates :math:`A` and
:math:`B` that overlap on one qubit and do not commute:

.. figure:: _static/demonstration_assets/active_volume/teleportation_start.png
    :align: center
    :width: 40%
    :target: javascript:void(0)

We also assume that we can track the impact that :math:`B` has on Pauli operators, i.e., what
happens to a Pauli operator when pulling it through :math:`B`. For our use case, this will be
easily satisfied, because :math:`B` will be a Clifford operation that simply turns Paulis into
new Pauli operators.

For the parallelization, we first insert a state teleportation circuit between :math:`A` and
:math:`B` using a pair of auxiliary qubits prepared in an entangled Bell state
:math:`|\phi\rangle=\tfrac{1}{\sqrt{2}}(|0\rangle +|1\rangle)`, see the
:doc:`demo on state teleportation <tutorial_teleportation>` for details:

.. figure:: _static/demonstration_assets/active_volume/teleportation_insert_teleport.png
    :align: center
    :width: 55%
    :target: javascript:void(0)

Next, we pull the classically controlled correction gates, which are Pauli operators, through
the gate :math:`B`, and obtain new correction gates :math:`C_Z=B^\dagger X B` and
:math:`C_X=B^\dagger Z B`:

.. figure:: _static/demonstration_assets/active_volume/teleportation_commuted.png
    :align: center
    :width: 45%
    :target: javascript:void(0)

At this point, we pulled :math:`B` to the front, effectively parallelizing it with :math:`A`.
As we can see, the state teleportation together with the ability to track correction gates through
:math:`B` makes it possible to implement the two non-commuting operations :math:`A` and :math:`B`
simultaneously, without violating any physical laws or causal dependencies.

Practical usefulness
~~~~~~~~~~~~~~~~~~~~

The previous discussion shows that it is *possible* to parallelize non-commuting gates at the cost
of two additional qubits (we need two copies of the original qubit, plus a communication channel
"back in time"), a Bell state preparation and Bell basis measurement (we need to entangle the
communication channel), and classical compute. However, it is not so clear whether this is a useful tradeoff.
For example, we could apply the same technique when using a NISQ computer, but we would
potentially even increase the two-qubit gate depth, and additional qubits are quite
expensive to come by.

For the surface code corrected quantum computer, and specifically for the active volume computer,
the creation of qubit pairs in a Bell state and the Bell basis measurement are assumed to be
much cheaper than arbitrary logical operations (see info box), because they can just be woven into
the measurements, i.e., the code cycles, of the error correction code via lattice surgery.
Similarly, Pauli correction gates are anyway tracked in software throughout, so that the only
true price we are paying for the parallelization is the extra memory.
As we are trying to condense a computation by parallelizing it and reducing idle volume, this
tradeoff will often be beneficial.

As we can see, parallelization via state teleportation thus is a very good fit for active
volume computers and the goal of our compilation. This is why the framework by Litinski and
Nickerson promotes this technique to a first-class transformation, allowing
us to parallelize networks without having to recompile them.

Teleportation in the CNOT ladder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If teleportation is easily integrated in our compilation framework, one might be tempted
to translate only a minimalistic universal gate set into logical networks and compile any other
subroutines by combining those networks–just like we do with quantum circuits.
However, we will find in the following that this approach will miss out on further optimizations.

To understand the teleportation technique from above in the context of logical networks,
let’s consider our example CNOT ladder and perform the parallelization in the circuit picture.
We begin by inserting 2 teleportation circuits:

.. figure:: _static/demonstration_assets/active_volume/manual_parallel_0.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

Then, we pull the second and third CNOT through the Pauli corrections, turning :math:`X` into
:math:`X\otimes X` and leaving :math:`Z` unchanged:

.. figure:: _static/demonstration_assets/active_volume/manual_parallel_1.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

In the last step, we pull the second factor :math:`X^{(c)}` of the first :math:`X\otimes X`
correction (purple) through the second pair of measurements, leading to a simple classical
postprocessing step. Here we used the general fact that Pauli operations (or corrections) will
only affect the measurement *result* of a Pauli measurement, but not its measurement *basis*,
marked as a classical wire between the two :math:`Z\otimes Z` measurement instruments in
magenta below:

.. figure:: _static/demonstration_assets/active_volume/manual_parallel_2.png
    :align: center
    :width: 40%
    :target: javascript:void(0)

We fully parallelized the CNOT ladder, using :math:`8` instead of the original :math:`4` qubits.

Following the approach mentioned above, we now could
naively plug in the logical network for a single CNOT from earlier. This would lead to yet
another :math:`3\cdot 2` additional qubits, arriving at :math:`12` logical blocks and two idling
blocks (blocks with only U and D ports occupied that can be executed by memory modules),
acting on :math:`14` qubits:

.. figure:: _static/demonstration_assets/active_volume/example_parallelized_explicitly.png
    :align: center
    :width: 95%
    :target: javascript:void(0)

The idling blocks are marked in green.
However, recall the logical network into which we compiled the CNOT ladder earlier:

.. figure:: _static/demonstration_assets/active_volume/example_logical_network.png
    :align: center
    :width: 95%
    :target: javascript:void(0)

As we can see, we require only :math:`12` qubits to realize the network of :math:`12` blocks,
removing the idle memory blocks. This is because the logical network compilation already made
use of simplifications that could be made to the naively composed network.

We see that logical network compilation achieves parallelization without the need for bridge
qubits, making the resulting network more memory efficient than a procedurally parallelized
network (the additional memory is the bridge qubit that idles throughout the execution of the
network).
Thus, even though the number of computational blocks is the same, resynthesizing the composed
network of three CNOTs leads to a slightly cheaper network in terms of routing. It is noteworthy
that this kind of optimization is of second order, because the active volume computer is assumed
to be very fast at routing (see info box at the top). However, for larger computations, e.g.,
already for a Toffoli gate realized via state injection, the number of computational blocks
itself will be reduced, too.

Reactive measurements and reaction time
---------------------------------------

While the compilation techniques presented above are very powerful, there is an important
restriction we haven’t considered so far. Consider the teleportation circuit from the previous
section:

.. figure:: _static/demonstration_assets/active_volume/post-processed_measurement.png
    :align: center
    :width: 15%
    :target: javascript:void(0)

The Bell state measurements determine whether Pauli corrections need to be applied, which in
turn affects the outcome (and only the outcome) of other Bell state measurements, which
determines whether Pauli corrections need to be applied, and so on. But since we’re tracking
Pauli corrections in software [cite pauli frame tracker], there is no immediate need to do
anything with the Bell measurement results, and we can happily continue to apply operations
in parallel or any other order without worrying about data dependencies between them. So do we
*ever* need to affect the quantum state using the information from measurements and Pauli
corrections?

To answer this question, we need to consider non-Clifford operations. As mentioned further up,
these are implemented via magic state injection. As a general rule, operations implemented via
Pauli measurements require a correction that is one level lower in the Clifford hierarchy.
So Clifford gates require a Pauli correction (similar to our corrections in the teleportation
circuit), 1st order non-Clifford gates like T and Toffoli require a Clifford correction, 2nd
order non-Clifford gates require 1st order ones, and so on.

.. figure:: _static/demonstration_assets/active_volume/pprs.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

    The circuit style used in Litinski’s papers color-codes the different members of the
    Clifford hierarchy: Pauli gates (or pi/2 rotations) in gray, Clifford gates (or pi/4 rotations) in
    orange, (1st order) non-Clifford gates (or pi/8 rotations) in green, and measurements in blue.
    Both diagrams show the decomposition of Pauli rotations into Pauli measurements plus associated
    corrections of the lower hierarchy orders. Image source: Daniel Litinski [#Litinski2018]_.

This is where we quickly run into issues, as Clifford corrections are generally already considered
too complex to be tracked classically. That means that we’ll have to physically implement them,
and it is at this point that we require a concrete value of the measurement result (accounting
for any prior Pauli corrections). In this sense, the Clifford corrections impose a limit on the
ability to parallelize, as part of the computation will still need to happen sequentially.

Doing corrections in-line as depicted in the above circuits has been shown to be inefficient
actually, so a key optimization is to move the correction away from the data qubits and onto
the auxiliary magic states. This gives us significantly more flexibility to schedule computation
efficiently as we don’t have to do the corrections right away. We can hold on to the consumed
magic state until it is convenient (or necessary) to apply the correction. Such gate
implementations are called “auto-corrected”, and will look as follows for non-Clifford Pauli
rotations:

.. figure:: _static/demonstration_assets/active_volume/auto-magic.png
    :align: center
    :width: 55%
    :target: javascript:void(0)

    Auto-corrected non-Clifford Pauli rotation. Image source: Daniel Litinski [#Litinski2018]_.

In essence, we have swapped the Clifford correction for a multiplexed measurement, where the
measurement basis is adaptively chosen based on some classical input. A measurement whose basis
depends on prior measurement results is called a **reactive measurement**. The speed at which
these measurements, and associated classical processing, can be performed is a fundamental limiting
factor to how fast a computation can be executed, regardless of how many qubits are available
for parallelism. This limitation is captured in the **reaction time** of an (active volume) quantum
computer. An execution scheme for quantum programs that is limited by these reactive
measurements–rather than, say, the circuit depth–is thus known as implementing
**reaction-limited computation**.

todo
[additional figure to show reaction-limited example circuit]

If you are familiar with the Game of Surface Codes paper referenced earlier, you may wonder
how the scheme presented there differs from what has been presented here. After all, Litinski
already used teleportation back then to parallelize Pauli circuits and achieve reaction-limited
computation (also referred to as “Fowler’s time-optimal scheme”).
Fundamentally, teleportation always trades (execution) time for (memory) space. The crucial
difference, intuitively, is that Litinski’s earlier techniques provide the additional space
in the trade-off as brand new qubits, since each operation is in principle, and through
techniques presented in the paper, assumed to take up the full width of a circuit. Meanwhile,
Active Volume computation tries to reuse already available, but idle, qubit space at a
fine-grained level, and to thus maximize the *efficiency* of the computer.

Conclusion
----------

With this, we conclude our brief introduction to active volume compilation.
We saw how it combines the best of ZX and circuit diagrams into a neat abstraction for logical
compilation steps, to which we then can compile standard circuit components, compressing them to
their essential logic and parallelizing them at the same time. As a consequence, the only idle
volume left in the compiled computation is due to reaction-limited computation that encodes
the fundamental causal structures in the compiled logic.
This compression of quantum circuits reduces their cost dramatically, if we have the right
machine, an active volume computer, in our hands, to which the compilation is tailored. It
forms the second key component of the framework, making the new representation of quantum
computation as logical networks useful. For details on this machine abstraction, many additional
compilation examples, and details about the implementation in a photonic quantum computer,
we recommend to read the original paper as well.
Additionally, you may want to

- dive into the background of :doc:`lattice surgery <demos/tutorial_lattice_surgery>`,
- learn about Pauli-based computation without active volume considerations in the
  :doc:`game of surface codes <demos/tutorial_game_of_surface_codes>`, and
- understand Pauli frame tracking from the perspective
  of :doc:`classical simulation <demos/tutorial_clifford_circuit_simulations>`.

References
-----------

.. [#Litinski2022]

   Daniel Litinski and Naomi Nickerson.
   "Active volume: An architecture for efficient fault-tolerant quantum computers with limited non-local connections",
   `arXiv:2211.15465 <https://arxiv.org/abs/2211.15465>`__, 2022.

.. [#Litinski2018]

   Daniel Litinski
   "A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery"
   `arXiv:1808.02892 <https://arxiv.org/abs/1808.02892v3>`__, 2018.

.. [#Gidney2018]

   Craig Gidney. "Halving the cost of quantum addition",
   `Quantum 2, 74 <https://quantum-journal.org/papers/q-2018-06-18-74/>`__, 2018,
   `arXiv:1709.06648 <https://arxiv.org/abs/1709.06648>`__, 2017.

Attributions
------------

Some images are taken from `Game of Surface Codes <https://quantum-journal.org/papers/q-2019-03-05-128/>`__
by Daniel Litinski [#Litinski2018]_, `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`__.
"""
