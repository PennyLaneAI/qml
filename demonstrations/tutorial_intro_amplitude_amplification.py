r"""Intro to Amplitude Amplification
=============================================================


Search problems have been with us since the dawn of time and finding efficient ways to perform this task is extremely useful.
It is surprising that it was not until the 17th century that we realized that sorting words alphabetically in a
dictionary could make searching easier.

However, the problem becomes complicated again when, given a word definition, we want to find that word in the dictionary.
We have lost the order and there is no possible classical strategy to help us do this quickly. Could we achieve it quantumly?


Preparing the problem
-----------------------

Taking a look to this problem from the point of view of quantum computing, our goal is to generate (and therefore, find)
an unknown state :math:`|\phi\rangle`, which, referring to the previous example, would be the word we just know the definition.

Having no information at all, we have few options, so a good first approach is to generate a state :math:`U|0\rangle := |\Psi\rangle`
with the intention that it contains some amount of the searched state :math:`|\phi\rangle`. Generically we can represent
:math:`|\Psi\rangle` in the computational basis as:

.. math::
    |\Psi\rangle = \sum_i c_i |i\rangle,

but we could represent it with another basis if necessary. In fact, it is not difficult to show that there exists one
where :mat:`|\phi\rangle` is an element of such a basis and :math:`|\phi^{\perp}\rangle` is another element orthogonal
to the previous one such that:

.. math::
    |\Psi\rangle = \alpha |\psi\rangle + \beta |\phi^{\perp}\rangle,

where :math:`\alpha, \beta \in \mathbb{R}`.

.. note::

    An example of :math:`|Psi\rangle` could be the uniform superposition of all the possible words since we know that,
    at least, the one we are looking for is included there. Also in that example :math:`|\phi^{\perp}\rangle` is the
    uniform superposition of the words we are not interested in and :math:`\alpha` can be calculated as :math:`\sqrt{\frac{1}{n}}` where :math:`n` is the size of the dictionary.

A great advantage of representing the state :math:`|\Psi\rangle` in this way is that we can now represent it visually
in a two-dimensional space:

[image]

Our goal, therefore, is to find an operator that moves the initial vector as close to :math:`|\phi\rangle` as we can.

Finding the operator
-----------------------

Ideally, if we could create a rotational gate in this subspace, we would be done. We would simply have to apply a certain
angle to our initial state and we would arrive at :math:`|\phi\rangle`. However, without information about that state,
it is really hard to think that this is possible. This is where a simple but great idea is born: What if instead of
rotations we think of reflections?

Reflection is a well studied operator and can help us to move our state in a controlled way through this subspace.
Looking at our two-dimensional representation we could think of three possible reflections: with respect to
:math:`|\phi^{\perp}\rangle`, :math:`|\phi^{\perp}\rangle` or `|\Psi\rangle`.

With a little intuition we can see that there is a sequence of two reflections that really meets our objective.
The first is the reflection with respect to :math:`|\phi^{\perp}\rangle` and the second with respect to `|\Psi\rangle`.
Let's go step by step:

[image]

The :math:`|\phi^{\perp}\rangle` reflection  may seem somewhat complex to create since we do not have access to such a state.
However, the operator is well defined:

.. math::
     \begin{cases}
     R_{\phi^{\perp}}|\phi\rangle = -|\phi\rangle, \\
     R_{\phi^{\perp}}|\phi^{\perp}\rangle = |\phi^{\perp}\rangle.
     \end{cases}

This operator must be able to change the sign of the solution state.
The way to build it depends on the problem, but in general it is just a classic check if given a state, it is the one
we are looking for.

.. note::

    In the example we have been working on, this operator would take a word as input, look up its definition in the
    dictionary and if it matches ours, it applies a phase to that state.

The second reflection is the one with respect to :math:`|\Psi\rangle`. This operator is much easier to build since
we know the gate :math:`U` that generate it. This can be built directly with :class:`~.pennylane.Reflection`, where you
can find information about its implementation.

[image]

These two rotations are equivalent to rotate the state :math:`2\theta` degrees, where :math:`\theta` is the initial
angle that forms our state. To approach the target state, we must perform this rotation :math:`\text{iters}` times where:

.. math::
    \text{iters} \sim \frac{\pi}{4 \arcsin \alpha}-\frac{1}{2}.

"""
##############################################################################
# About the author
# ----------------
