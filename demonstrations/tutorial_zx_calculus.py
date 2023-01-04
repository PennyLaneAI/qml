r"""
ZX calculus
===========

.. meta::
    :property="og:description": Investigation of ZX calculus and its applications to quantum computing
    :property="og:image": https://pennylane.ai/qml/_images/zx.png

*Author: Romain Moyard. Posted: January 2023.*


The ZX calculus is a graphical language that can represent any linear map. Therefore, it can be used to reason about
quantum computations and quantum circuits. Its foundations are based on category theory, which makes it a rigorous
framework. It was introduced in 2008 by Coecke and Duncan [#Coecke]_ . As it can handle any linear map, therefore,
it can be considered as a generalization of the circuit representation of quantum computations.

In this tutorial, we first give an overview of the building blocks of the ZX-diagrams and also of the main rewriting
rules, the ZX calculus. Then we will explore how to optimize the number of T-gates of a benchmark circuit with PennyLane
and PyZX [PyZX]_. We also show that simplifying (reducing) a ZX-diagram does not always end up with diagram-like graph,
and that circuit extraction is a main pain point of the ZX framework. Finally, we give some leads about ZX-calculus
advanced uses.

ZX-diagrams
-----------
This introduction follows the works of the [#East2021]_ and [#JvdW2020]_ . We start by introducing ZX diagrams. ZX
diagrams are a graphical depiction of a tensor network representing an arbitrary linear map. Later, we will introduce ZX
rewriting rules, together with diagrams it defines ZX-calculus.

A ZX-diagram is an undirected multi-graph; you can move vertices, and it does not have any effect on the underlying
linear map. The vertices are called Z and X spiders, and it represents two kind of linear maps. The edges are called
wires, and it represents the dimensions on which the linear maps are acting on. Therefore, the edges represent qubits in
quantum computing. The diagram's wires on the left are called inputs, the one leaving on the right are called outputs.

The first building block of the ZX-diagram is the Z spider. In most of the literature, it is depicted as a green vertex.
The Z spider takes a real phase $\alpha \in \mathbb{R}$ and represents the following linear map (it accepts any number
of inputs and outputs):

.. figure:: ../demonstrations/zx_calculus/z_spider.png
    :align: center
    :width: 70%

    The Z-spider.

It is easy to see that the usual Z-gate can be represented with a single-wire Z-gate:

.. figure:: ../demonstrations/zx_calculus/z_gate.png
    :align: center
    :width: 70%

    The Z-gate.


You've already guessed it, the second building block of the ZX-diagram is the X spider. It is usually depicted as a red
vertex. The X spider also takes a real phase $\alpha \in \mathbb{R}$ and it represents the following linear map
(it accepts any number of inputs and outputs):

.. figure:: ../demonstrations/zx_calculus/x_spider.png
    :align: center
    :width: 70%

    The X spider.

It is easy to see that the usual X-gate can be represented with a single-wire X-gate:

.. figure:: ../demonstrations/zx_calculus/x_gate.png
    :align: center
    :width: 70%

    The X gate.

A special case of the Z and X spiders are diagrams with no inputs (or outputs). They are used to represent state which
are unnormalized. If a spider has no inputs and outputs, it simply represents a complex scalar.

The phases are two-pi periodic, when a phase is equal to 0,  we omit to write the zero symbol in the spider.
Therefore, a simple green node is a Z spider with a zero-phase and a simple red node is a X spider with a zero-phase.

You can find the usual representation of quantum states below:

.. figure:: ../demonstrations/zx_calculus/zero_state.png
    :align: center
    :width: 70%

    The zero state.

.. figure:: ../demonstrations/zx_calculus/plus_state.png
    :align: center
    :width: 70%

    The plus state.

Similarly, you get the 1 state and minus state by replacing the zero phase with pi.

We have our two necessary building blocks, now we can compose and stack those tensors. The composition consists in
joining the outputs of a first diagram to the inputs of a second diagram. The tensor product of two diagrams can be done
by stacking them.


Given the rules of stacking and composition we can now build an equivalent CNOT gate (up to a gloabl phase). We first
start by stacking a phaseless Z spider with 1 input wire and two output wires with a single wire.

.. figure:: ../demonstrations/zx_calculus/stack_z_w.png
    :align: center
    :width: 70%

    Phaseless Z with 1 input wire and 2 output wires stacked with a single wire

Then we stack a single wire with a phaseless X spider with 2 input wires and single output wire.

.. figure:: ../demonstrations/zx_calculus/stack_w_x.png
    :align: center
    :width: 70%

    Single wire stacked with a X phaseless spider with two inputs wires and one output wire.

Finally, we compose the two diagrams, meaning that we join the two output of the first diagram with the two inputs of
the second diagram. By doing this we obtain a CNOT gate, you can convince yourself by applying the matrix multiplication
between the two diagrams.

.. figure:: ../demonstrations/zx_calculus/compose_zw_wx.png
    :align: center
    :width: 70%

    The composition of the two diagrams is a CNOT gate.

We've already mentioned it before a ZX-diagram iss an undirected multi-graph. The position of the vertices
does matter as well as the trajectory of the wires. We can move the vertices around, bending,
unbending, crossing, and uncrossing wires, as long as the connectivity and the order of the inputs and outputs is
maintained. It means that ZX-diagrams have all sorts of topological symmetries and all these deformations do not affect
the underlying linear map.

E.g. yhe two following diagrams and the previous one represent the same CNOT linear map.

.. figure:: ../demonstrations/zx_calculus/cnot_moved.png
    :align: center
    :width: 70%

    The composition of the two diagrams is a CNOT gate.

We introduce here the usual way of representing the CNOT gate ( with a vertical wire).

.. figure:: ../demonstrations/zx_calculus/cnot.png
    :align: center
    :width: 70%

    Usual representation of the CNOT gate as a ZX-diagram.


We've just shown that we can express any Z rotation and X rotation with the two spiders. Therefore, it is sufficient to
create any 1-qubit rotation on the Bloch-sphere, therefore any 1-qubit state. By composition and stacking, we can also
create the CNOT gate. Therefore, we have a universal gate set, and we can represent any unitary on any Hilbert space.
We can also create the zero state of any size. Therefore, we can represent any quantum state. Some normalisation can be
needed, it can be performed by adding some complex scalar vertices.

Furthermore, by using the Choi-Jamiolkowski isomorphism, we can represent any linear map L from n wires to m wires as a
ZX-diagram because it can be transformed as a n+m state [#JvdW2020]_. It shows the universality of ZX-diagrams to reason
about any linear map. But it does not mean that the representation is simple.

For a more in depth introduction, see [#Coecke] and [#Coecke2011].

ZX calculus: rewriting rules
----------------------------
The ZX-diagrams coupled with rewriting rules form what is called the ZX-calculus. Previously, we presented the rules
for composition and stacking of diagrams, we've also talked about the topological symmetries. In this section, we show
the main rewriting rules that can be used to simplify the diagrams. This powerful set of rules allows us to transform
the diagrams without changing the underlying linear map. It is very useful for quantum circuit optimization and also
to show that some computation have a very simple form in the ZX framework (e.g. Teleportation).

In the following rules the colours are interchangeable.

0. A first simple rule derived from quantum computing and that helps us to reason about diagrams is that non-phaseless
    vertices of different color do not commute (X gate and Z gate do not commute).

1. The fuse rule can be applied when two spiders of the same type are connected by one or more wires. The connection
    wires are not necessary and therefore can be removed and the spiders are fused. The fusion is simply adding the two
    spider phases.

    .. figure:: ../demonstrations/zx_calculus/f_rule.png
        :align: center
        :width: 70%

        The (f)use rule.

2. The $pi$copy rule describes how an X gate interacts with a Z spider (or a Z gate with an X spider). It shows how
    gates can commute through spiders by copying them on the other side.

    .. figure:: ../demonstrations/zx_calculus/pi_rule.png
        :align: center
        :width: 70%

        The ($pi$)copy rule.

3. The state copy rule shows how a state interact with a spider of opposite colour. It is only valid for states that
    are multiple of $pi$. It shows how certain states can commute through spiders by copying them on the other side.

    .. figure:: ../demonstrations/zx_calculus/c_rule.png
        :align: center
        :width: 70%

        The state (c)opy rule,

4. The identity rule is similar to the rule that Z and X rotation gates which are phaseless are equivalent to the
    identity. The phaseless spiders with one input and one input are equivalent to the identity and therefore can be
    removed. This rule also gives the possibility to get rid of self-loops.

    .. figure:: ../demonstrations/zx_calculus/id_rule.png
        :align: center
        :width: 70%

        The (id)entity removal rule.

5. The bialgebra rule is similar to the fact that the XOR algebra and the COPY coalgebra together form a bialgebra.
    This rule is not straightforward to verify and details can be found in this paper [#JvdW2020]_ .

    .. figure:: ../demonstrations/zx_calculus/b_rule.png
        :align: center
        :width: 70%

        The (b)ialgebra rule.

6. The Hopf rule is similar to the fact that the XOR algebra and the COPY coalgebra satisfying this equation are known
    together as a Hopf algebra. This rule is not straightforward to verify and details can be found in this paper
    [#JvdW2020]_ .

    .. figure:: ../demonstrations/zx_calculus/hopf_rule.png
        :align: center
        :width: 70%

        The (ho)pf rule.

ZXH-diagrams
------------

The universality of the ZX-calculus does not guarantee to have a simple representation of any linear map. For example,
The Toffoli gate (quantum AND gate) has no simple way of being represented, as a ZX-diagram it contains around 25
spiders. Therefore, another generator is introduced: the H-box. It allows for a simple representation of the AND gate.

.. figure:: ../demonstrations/zx_calculus/h_box.png
    :align: center
    :width: 70%

    The H-box.

The parameter $a$ can be any complex number, and the sum  is over all $i1, . . . , im, j1, . . . , jn \in {0, 1}$,
therefore an H-box represents a matrix where all entries are equal to 1, except for the bottom right element, which is \
$a$.

A H-box with one input wire and one output wire, with a=-1 is an Hadamard gate up to global phase, therefore we do not
draw the parameter when it is equal to -1. The Hadamard gate is not always represented as a yellow box, for the sake of
simplicity it is often replaced by a blue edge.

The respective relationship with X and Z spiders are the following:

.. figure:: ../demonstrations/zx_calculus/hxhz.png
    :align: center
    :width: 70%

    How to transform an X spider to a Z spider.

.. figure:: ../demonstrations/zx_calculus/hzhx.png
    :align: center
    :width: 70%

    How to transform an Z spider to a X spider.

Thanks to the introduction of the H-box the Toffoli gate can be represented with three Z spiders and three H-box.

.. figure:: ../demonstrations/zx_calculus/toffoli.png
    :align: center
    :width: 70%

    Toffoli

The ZXH-calculus contains a new set of rewriting rules, for more details you can find these rules in the literature
[#East2021].


Teleportation example:
----------------------

Now that we have all the necessary tools to describe any quantum circuit, let's take a look at how we can describe
teleportation as a ZXH-diagram and simplify it. The results are surprisingly simple!

Teleportation in quantum computing is a protocol for transferring quantum information (the state) from Alice (sender)
placed at a specific location to Bob (receiver) placed at any other distant location.

Alice and Bob need to share a maximally entangled state. The protocol for Alice to send her quantum state to Bob is the
following:

1. Alice applies the CNOT gate followed by the Hadamard gate.
2. Alice measures the two qubits that she has.
3. Alice sends the two measurement results to Bob.
4. Given the results Bob conditionally applies the Z and X gate to.
5. Bob ends up with the same state as Alice previously had. The teleportation is completed.

The procedure is described by the following quantum circuit, it summarizes the protocol from above.

.. figure:: ../demonstrations/zx_calculus/teleportation_circuit.png
    :align: center
    :width: 70%

    The teleportation circuit.

Now we can translate the quantum circuit to a ZXH-diagram. The measurements are represented by the state X-spider
parametrized with boolean parameters a and b. The cup represents the maximally entangled state shared between Alice and
Bob.

.. figure:: ../demonstrations/zx_calculus/teleportation.png
    :align: center
    :width: 70%

    The teleportation ZXH diagram. TODO remove all figures except the first one

Let's simplify the diagram by applying some rewriting rules. The first step is to fuse the a state with the X-spider
of the CNOT. We also merge the H-box with the b state, because together it represents a Z-spider. Then we can fuse the
three Z-spiders by simply adding their phases. After that we see that the Z-spider phase is modulo of two pi and
therefore it can be simplified by using the identity rules. Then we can fuse the two X-spiders by adding their phase.
We notice that the phase is again modulo of two pi and therefore we can use the identity rule and get rid of the last
X-spider. Teleportation is a simple wire connecting Alice and Bob!

.. figure:: ../demonstrations/zx_calculus/teleportation.png
    :align: center
    :width: 70%

    The teleportation simplification.

"""

######################################################################
# ZX-diagrams with PennyLane
# --------------------------
#
#
#

import pennylane as qml
import pyzx

dev = qml.device("default.qubit", wires=2)

@qml.transforms.to_zx
@qml.qnode(device=dev)
def circuit():
    qml.PauliX(wires=0),
    qml.Hadamard(wires=0),
    qml.CNOT(wires=[0, 1]),
    return qml.expval(qml.PauliZ(wires=0))

g = circuit()
pyzx.draw_matplotlib(g)


######################################################################
# Graph simplification and circuit extraction
# -------------------------------------------
# Add the Venne graph for simplification and extraction
#
# Full reduce and teleport reduce

######################################################################
# Example: T-count optimization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Here we give an example of how to use optimization techniques from ZX calculus to reduce the T count of a quantum
# circuit and get back a PennyLane circuit.
# Let’s start by starting with the mod 5 4 circuit from a known benchmark library the expanded circuit before
# optimization is the following QNode:

dev = qml.device("default.qubit", wires=5)

@qml.transforms.to_zx
@qml.qnode(device=dev)
def mod_5_4():
    qml.PauliX(wires=4),
    qml.Hadamard(wires=4),
    qml.CNOT(wires=[3, 4]),
    qml.adjoint(qml.T(wires=[4])),
    qml.CNOT(wires=[0, 4]),
    qml.T(wires=[4]),
    qml.CNOT(wires=[3, 4]),
    qml.adjoint(qml.T(wires=[4])),
    qml.CNOT(wires=[0, 4]),
    qml.T(wires=[3]),
    qml.T(wires=[4]),
    qml.CNOT(wires=[0, 3]),
    qml.T(wires=[0]),
    qml.adjoint(qml.T(wires=[3]))
    qml.CNOT(wires=[0, 3]),
    qml.CNOT(wires=[3, 4]),
    qml.adjoint(qml.T(wires=[4])),
    qml.CNOT(wires=[2, 4]),
    qml.T(wires=[4]),
    qml.CNOT(wires=[3, 4]),
    qml.adjoint(qml.T(wires=[4])),
    qml.CNOT(wires=[2, 4]),
    qml.T(wires=[3]),
    qml.T(wires=[4]),
    qml.CNOT(wires=[2, 3]),
    qml.T(wires=[2]),
    qml.adjoint(qml.T(wires=[3]))
    qml.CNOT(wires=[2, 3]),
    qml.Hadamard(wires=[4]),
    qml.CNOT(wires=[3, 4]),
    qml.Hadamard(wires=4),
    qml.CNOT(wires=[2, 4]),
    qml.adjoint(qml.T(wires=[4]), )
    qml.CNOT(wires=[1, 4]),
    qml.T(wires=[4]),
    qml.CNOT(wires=[2, 4]),
    qml.adjoint(qml.T(wires=[4])),
    qml.CNOT(wires=[1, 4]),
    qml.T(wires=[4]),
    qml.T(wires=[2]),
    qml.CNOT(wires=[1, 2]),
    qml.T(wires=[1]),
    qml.adjoint(qml.T(wires=[2]))
    qml.CNOT(wires=[1, 2]),
    qml.Hadamard(wires=[4]),
    qml.CNOT(wires=[2, 4]),
    qml.Hadamard(wires=4),
    qml.CNOT(wires=[1, 4]),
    qml.adjoint(qml.T(wires=[4])),
    qml.CNOT(wires=[0, 4]),
    qml.T(wires=[4]),
    qml.CNOT(wires=[1, 4]),
    qml.adjoint(qml.T(wires=[4])),
    qml.CNOT(wires=[0, 4]),
    qml.T(wires=[4]),
    qml.T(wires=[1]),
    qml.CNOT(wires=[0, 1]),
    qml.T(wires=[0]),
    qml.adjoint(qml.T(wires=[1])),
    qml.CNOT(wires=[0, 1]),
    qml.Hadamard(wires=[4]),
    qml.CNOT(wires=[1, 4]),
    qml.CNOT(wires=[0, 4]),
    return qml.expval(qml.PauliZ(wires=0))


# The circuit contains 63 gates; 28 `qml.T()` gates, 28 `qml.CNOT()`, 6 `qml.Hadmard()` and 1 `qml.PauliX()`. We applied
# the `qml.transforms.to_zx` decorator in order to transform our circuit to a ZX graph.
#
# You can get the PyZX graph by simply calling the QNode:


g = mod_5_4()
t_count = pyzx.tcount(g)
print(t_count)

# PyZX gives multiple options for optimizing ZX graphs (`pyzx.full_reduce()`, `pyzx.teleport_reduce()`, …). The
# `pyzx.full_reduce()` applies all optimization passes, but the final result may not be circuit-like. Converting back
# to a quantum circuit from a fully reduced graph may be difficult to impossible. Therefore, we instead recommend using
# `pyzx.teleport_reduce()`, as it preserves the circuit structure.

g = pyzx.simplify.teleport_reduce(g)
opt_t_count = pyzx.tcount(g)
print(opt_t_count)


# If you give a closer look, the circuit contains now 53 gates; 8 `qml.T()` gates, 28 `qml.CNOT()`, 6 `qml.Hadmard()`
# and 1 `qml.PauliX()` and 10 `qml.S()`. We successfully reduced the T-count by 20 and have 10 additional S gates.
# The number of CNOT gates remained the same.
#
# The from_zx() transform can now convert the optimized circuit back into PennyLane operations:

qscript_opt = qml.transforms.from_zx(g)

wires = qml.wires.Wires([4, 3, 0, 2, 1])
wires_map = dict(zip(qscript_opt.wires, wires))
qscript_opt_reorder = qml.map_wires(input=qscript_opt, wire_map=wires_map)

@qml.qnode(device=dev)
def mod_5_4():
    for o in qscript_opt_reorder:
        qml.apply(o)
    return qml.expval(qml.PauliZ(wires=0))

######################################################################
# Acknowledgement
# ---------------
# Richard East
# Guillermo Alonso
#
######################################################################
# Math
# ----
#
# .. math::
#
#    S_s X_i = \left( Z_i Z_a Z_b Z_c \right) X_i = - X_i S_s.
#
#
######################################################################
# Advanced use
# ^^^^^^^^^^^^
#
# Derive the parameter shift rule.
#
# References
# ----------
#
# .. [#Duncan2017]
#
#    Ross Duncan, Aleks Kissinger, Simon Perdrix, and John van de Wetering. "Graph-theoretic Simplification of Quantum
#    Circuits with the ZX-calculus"
#    `Quantum Journal <https://quantum-journal.org/papers/q-2020-06-04-279/pdf/>`__.
#
# .. [#Kissinger2021]
#
#    Aleks Kissinger and John van de Wetering. "Reducing T-count with the ZX-calculus."
#    `ArXiv <https://arxiv.org/pdf/1903.10477.pdf>`__.
#
# .. [#Coecke2011]
#
#    Bob Coecke and Ross Duncan. "Interacting quantum observables: categorical algebra and diagrammatics."
#    `New Journal of Physics <https://iopscience.iop.org/article/10.1088/1367-2630/13/4/043016/pdf>`__.
#
#
# .. [#Coecke]
#
#    Bob Coecke and Ross Duncan. "A graphical calculus for quantum observables."
#    `Oxford <https://www.cs.ox.ac.uk/people/bob.coecke/GreenRed.pdf>`__.
#
# .. [#East2021]
#
#    Richard D. P. East, John van de Wetering, Nicholas Chancellor and Adolfo G. Grushin. "AKLT-states as ZX-diagrams:
#    diagrammatic reasoning for quantum states."
#    `ArXiv <https://arxiv.org/pdf/2012.01219.pdf>`__.
#
# .. [#PyZX]
#
#    John van de Wetering. "PyZX."
#    `PyZX GitHub <https://github.com/Quantomatic/pyzx>`__.
#
# .. [#JvdW2020]
#
#    John van de Wetering. "ZX-calculus for the working quantum computer scientist."
#    `ArXiv <https://arxiv.org/abs/2012.13966>`__.
#
# About the author
# ----------------
# .. include:: ../_static/authors/romain_moyard.txt
