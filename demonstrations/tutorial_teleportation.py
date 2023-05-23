r"""
Quantum Teleportation
=====================

.. meta::
  :property="og:description": Transmit a quantum state to an entangled qubit.

.. related::

  tutorial_mbqc Measurement-based quantum computation

*Author: Matthew Silverman - Posted: 1 June 2023. Last Updated 1 June 2023.*

This tutorial walks through a popular quantum information technique known as
quantum teleportation. While teleportation has been thought of as the stuff of
sci-fi legend, we are going to prove that it is actually possible today! The
technique leverages many foundational principles of quantum computing, and it has
many useful applications across the entire field. It should be noted that it is
only the quantum *information* being teleported, and not a physical particle.

Suppose there are two researchers named Alice and Bob, and Alice wants to send
her quantum state to Bob. The quantum teleportation protocol will enable Alice to
do exactly this in a very elegant manner. An overview of the protocol can be seen
here:

# TODO: full circuit image here

Problem: The No-Cloning Theorem
-------------------------------

You might be wondering why we need to teleport a state at all. Can't Alice
just make a copy of it and send the copy to Bob? It turns out that copying
arbitrary states is *prohibited*, which you can prove using something called the
**no-cloning theorem**. The proof is surprisingly straightforward. Suppose we
would like to design a circuit (unitary) :math:`U` that can perform the following
action:

.. math::

  U(\vert \psi\rangle \otimes \vert s\rangle ) &= \vert \psi\rangle \otimes \vert \psi\rangle, \\
  U(\vert \varphi\rangle \otimes \vert s\rangle ) &= \vert \varphi \rangle \otimes \vert \varphi \rangle, \\
  \tag{1}

where :math:`\vert \psi\rangle` and :math:`\vert \varphi\rangle` are arbitrary
single-qubit states, and :math:`\vert s \rangle` is some arbitrary starting state.
We will now prove that no such :math:`U` exists! First, note that the inner product
of the left-hand-sides of the two equations is :math:`\langle \psi \vert \varphi \rangle`.
Next, note that the inner product of the right-hand-sides of the two equations is :math:`(\langle \psi \vert \varphi \rangle)^2`.
These inner products must be equal, and they are only equal if they are a value that
squares to itself. The only valid values for the inner product then are 1 and 0. But
if the inner product is 1, the states are the same; on the other hand, if the inner
product is 0, the states are orthogonal. Therefore, we can't clone arbitrary states!

So, what is quantum teleportation?
----------------------------------

Now that we know we can't arbitrarily copy states, we return to the task of
teleporting them. Teleportation relies on Alice and Bob having access to
shared entanglement. The protocol can be divided into roughly four parts. We'll
go through each of them in turn.

# TODO: add 4-part image here

1. State preparation
````````````````````

Teleportation involves three qubits. Two of them are held by Alice, and the
third by Bob. We'll denote their states using a subscript :math:`\vert
\cdot\rangle_A` and :math:`\vert \cdot\rangle_B` for clarity. Together, their starting
state is

.. math::

  \vert 0\rangle_A \vert 0\rangle_A \vert 0\rangle_B.\tag{2}

The first thing Alice does is prepare her first qubit in whichever state :math:`\vert
\psi\rangle` that she'd like to send to Bob, so that their combined state
becomes

.. math::

  \vert \psi\rangle_A \vert 0\rangle_A \vert 0\rangle_B.\tag{3}

2. Shared entanglement
``````````````````````

The reason why teleportation works as it does is the use of an *entangled state*
as a shared resource between Alice and Bob. You can imagine it being
constructed in the circuit, like so, but you can also imagine some sort of
centralized entangled state generator that produces Bell states and sends one
qubit to each party.

Entangling the second and third qubits leads to the combined state

.. math::

  \frac{1}{\sqrt{2}}\left( \vert \psi\rangle_A \vert 0\rangle_A \vert 0\rangle_B + \vert \psi\rangle_A \vert 1\rangle_A \vert 1\rangle_B \right)\tag{4}

3. Change of basis
``````````````````

This is where things get tricky, but also very interesting. The third step of
the protocol is to apply a :math:`CNOT` and a Hadamard to the first two qubits. This is
done prior to the measurements, and labelled "change of basis". But what basis
is this? Notice how these two gates are the *opposite* of what we do to create a
Bell state. If we run them in the opposite direction, we transform the basis
back to the computational one, and simulate a measurement in the *Bell
basis*. The Bell basis is a set of four entangled states

.. math::

  \vert \psi_+\rangle &= \frac{1}{\sqrt{2}} \left( \vert 00\rangle + \vert 11\rangle \right), \\
  \vert \psi_-\rangle &= \frac{1}{\sqrt{2}} \left( \vert 00\rangle - \vert 11\rangle \right), \\
  \vert \phi_+\rangle &= \frac{1}{\sqrt{2}} \left( \vert 01\rangle + \vert 10\rangle \right), \\
  \vert \phi_-\rangle &= \frac{1}{\sqrt{2}} \left( \vert 01\rangle - \vert 10\rangle \right). \\
  \tag{5}

After the basis transform, if we observe the first two qubits to be in the state
:math:`\vert 00\rangle`, this would correspond to the outcome :math:`\vert \psi_+\rangle` in
the bell basis, :math:`\vert 11\rangle` would correspond to :math:`\vert \phi_-\rangle`,
etc. Let's perform this change of basis, one step at a time. Expanding out the terms (and
removing the subscripts for brevity), we obtain

.. math::

  \frac{1}{\sqrt{2}} ( \alpha\vert 000\rangle +
  \beta\vert 100\rangle + \alpha \vert 011\rangle +
  \beta\vert 111\rangle )

Now let's apply a :math:`CNOT` between Alice's two qubits:

.. math::

  \frac{1}{\sqrt{2}} ( \alpha\vert 000\rangle +
  \beta\vert 110\rangle + \alpha \vert 011\rangle +
  \beta\vert 101\rangle )

And then a Hadamard on her first qubit:

.. math::

  \frac{1}{2} ( \alpha \vert 000\rangle + \alpha\vert 100\rangle + \beta\vert 010\rangle - \beta\vert 110\rangle + \alpha \vert 011\rangle + \alpha \vert 111 \rangle + \beta\vert 001\rangle - \beta\vert 101 \rangle ).

Now we need to do some rearranging. We group together the terms based on the first two qubits:

.. math::

  \frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\beta\vert 0\rangle + \alpha\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (-\beta\vert 0\rangle + \alpha\vert 1\rangle).

4. Measurement
``````````````

The last step of the protocol involves applying two controlled operations from
Alice's qubits to Bob, a controlled-:math:`Z`, and a :math:`CNOT`, followed by a
measurement. But why exactly are we doing this before the measurement? In the
previous step, we already performed a basis rotation back to the computational
basis, so shouldn't we be good to go?

Let's take a closer look at the state you should have obtained from the previous
exercise:

.. math::

  \frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\beta\vert 0\rangle + \alpha\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (-\beta\vert 0\rangle + \alpha\vert 1\rangle). \tag{6}

If Alice measures her two qubits in the computational basis, she is equally
likely to obtain any of the four possible outcomes. If she observes the first two
qubits in the state :math:`\vert 00 \rangle`, she would immediately know that Bob's
qubit was in the state :math:`\alpha \vert 0 \rangle + \beta \vert 1 \rangle`, which is
precisely the state we are trying to teleport!

If instead she observed the qubits in state :math:`\vert 01\rangle`, she'd still
know what state Bob has, but it's a little off from the original state. In particular,
we have

.. math::

  \beta \vert 0 \rangle + \alpha \vert 1 \rangle = X \vert \psi \rangle.\tag{7}

Alice could tell Bob, after obtaining these results, to simply apply an :math:`X`
gate to his qubit to recover the original state. Similarly, if she obtained
:math`\vert 10\rangle`, she would tell him to apply a :math:`Z` gate.

In a more ["traditional" version of
teleportation](https://quantum.country/teleportation), this is, in fact, exactly
what happens. Alice would call up Bob on the phone, tell him which state she
observed, and then he would be able to apply an appropriate correction. In this
situation, measurements are happening partway through the protocol, and the
results would be used to control the application of future quantum gates. This is
known as mid-circuit measurement, and such mid-circuit measurements are expressed
in PennyLane using `qml.cond <https://docs.pennylane.ai/en/stable/code/api/pennylane.cond.html>`_.

# TODO: code before explaining deferred measurement principle

Here, we are presenting a slightly different version of teleportation that
leverages the [principle of deferred
measurement](https://en.wikipedia.org/wiki/Deferred_Measurement_Principle). Basically,
we can push all our measurements to the *end* of the circuits.

# TODO: add 4-part image here

You might have wondered why these two gates were included in the measurement box
in the diagrams; this is why. Alice applying controlled operations on Bob's
qubit is performing this same kind of correction *before* any measurements are
made. Let's evaluate the action of the :math:`CNOT` and controlled :math:`Z` on Bob's
qubit, and ensure that Alice's state been successfully teleported. Applying the :math:`CNOT` yields

.. math::

  \frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle)

Then, applying the :math:`CZ` yields

.. math::

  \frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle)

When Alice measures her two qubits at the end, no matter which outcome she
gets, Bob's qubit will be in the state :math:`\alpha\vert 0\rangle + \beta \vert
1\rangle`. This means that our protocol has changed the state of Bob's qubit
into the one Alice wished to send him, which is truly incredible!
"""
