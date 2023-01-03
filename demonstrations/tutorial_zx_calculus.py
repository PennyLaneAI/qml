r"""
ZX calculus
===========

.. meta::
    :property="og:description": Investigation of ZX calculus and its applications to quantum computing
    :property="og:image": https://pennylane.ai/qml/_images/zx.png

*Author: Romain. Posted: January 2023.*


The ZX calculus is a graphical language that can represent any linear maps. Therefore, it can be used to reason
quantum computations and quantum circuits. Its foundations are based on category theory, which makes it a rigorous
framework. It was introduced in 2008 by Coecke and Duncan [#Coecke]_ . As it can handle any linear map, it can be
considered as a generalization of the circuit representation of quantum computations.

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


Given the rules of stacking and composition we can now build the CNOT gate.

For more in depth introduction, see:

ZXH-diagrams
------------

[#East2021]_

[#JvdW2020]_

The H spider:

Hadamard is a blue edge in most papers.

.. figure:: ../demonstrations/zx_calculus/h_box.png
    :align: center
    :width: 70%

    The H box

Relationship with X and Z spiders.

Toffoli:

.. figure:: ../demonstrations/zx_calculus/toffoli.png
    :align: center
    :width: 70%

    Toffoli

ZX calculus: rewriting rules
----------------------------
dot of same color commute 

(f)use

.. figure:: ../demonstrations/zx_calculus/f_rule.png
    :align: center
    :width: 70%

    (f)use

($pi$c)opy

.. figure:: ../demonstrations/zx_calculus/pi_rule.png
    :align: center
    :width: 70%

    ($pi$c)opy
    
(b)ialgebra

.. figure:: ../demonstrations/zx_calculus/b_rule.png
    :align: center
    :width: 70%

    (b)ialgebra

(c)opy

.. figure:: ../demonstrations/zx_calculus/c_rule.png
    :align: center
    :width: 70%

    (c)opy

(id)entity

.. figure:: ../demonstrations/zx_calculus/id_rule.png
    :align: center
    :width: 70%

    (id)entity
    
    
(ho)pf

.. figure:: ../demonstrations/zx_calculus/hopf_rule.png
    :align: center
    :width: 70%

    (ho)pf

[#East2021]_

[#JvdW2020]_

Teleportation example:

.. figure:: ../demonstrations/zx_calculus/teleportation.png
    :align: center
    :width: 70%

    Teleportation example 

"""


######################################################################
# ZX-diagrams with PennyLane
# --------------------------

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
# 28

# PyZX gives multiple options for optimizing ZX graphs (`pyzx.full_reduce()`, `pyzx.teleport_reduce()`, …). The
# `pyzx.full_reduce()` applies all optimization passes, but the final result may not be circuit-like. Converting back
# to a quantum circuit from a fully reduced graph may be difficult to impossible. Therefore we instead recommend using
# `pyzx.teleport_reduce()`, as it preserves the circuit structure.

g = pyzx.simplify.teleport_reduce(g)
opt_t_count = pyzx.tcount(g)
print(opt_t_count)

# 8
#
# If you give a closer look, the circuit contains now 53 gates; 8 `qml.T()` gates, 28 `qml.CNOT()`, 6 `qml.Hadmard()`
# and 1 `qml.PauliX()` and 10 `qml.S()`. We successfully reduced the T-count by 20 and have 10 additional S gates.
# The number of CNOT gates remained the same.

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
