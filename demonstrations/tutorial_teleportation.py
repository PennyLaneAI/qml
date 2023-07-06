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
many useful applications across the entire field. These principles include (but
are not limited to): the no-cloning theorem, quantum entanglement, and the
principle of deferred measurement. Let's dive in!

Suppose there are two researchers named Alice and Bob, and Alice wants to send
her quantum state to Bob. The quantum teleportation protocol enables Alice to
do exactly this in a very elegant manner. It should be noted that it is only
quantum *information* being teleported, and not physical particles. An
overview of the protocol can be seen here:

.. figure:: ../demonstrations/teleportation/teleportation-4part.svg
    :align: center
    :width: 75%

Problem: The No-Cloning Theorem
-------------------------------

You might be wondering why we need to teleport a state at all. Can't Alice
just make a copy of it and send the copy to Bob? It turns out that copying
arbitrary states is *prohibited*, which you can prove using something called the
**no-cloning theorem**. The proof is surprisingly straightforward. Suppose we
would like to design a circuit (unitary) :math:`U` that can perform the following
action:

.. math::

    \begin{align*}
    U(\vert \psi\rangle \otimes \vert s\rangle ) &= \vert \psi\rangle \otimes \vert \psi\rangle \\
    U(\vert \varphi\rangle \otimes \vert s\rangle ) &= \vert \varphi \rangle \otimes \vert \varphi \rangle
    \end{align*}

where :math:`\vert \psi\rangle` and :math:`\vert \varphi\rangle` are arbitrary
single-qubit states, and :math:`\vert s \rangle` is some arbitrary starting state.
We will now prove that no such :math:`U` exists! First, let's take the inner product
of the left-hand sides of the two equations:

.. math::

    (\langle \psi \vert \otimes \langle s \vert) U^\dagger U(\vert \varphi\rangle \otimes \vert s\rangle ) = \langle \psi \vert \varphi\rangle \  \langle s \vert s\rangle

Since :math:`\langle s \vert s\rangle` equals 1, this evaluates to
:math:`\langle \psi \vert \varphi \rangle`. Next, we compare the inner product of the
right-hand sides of the two equations: :math:`(\langle \psi \vert \varphi \rangle)^2`.
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

.. figure:: ../demonstrations/teleportation/teleportation-4part.svg
    :align: center
    :width: 75%

"""

##############################################################################
#
# 1. State preparation
# ````````````````````
#
# Teleportation involves three qubits. Two of them are held by Alice, and the
# third by Bob. We'll denote their states using subscripts:
#
# 1. :math:`\vert\cdot\rangle_A`, Alice's first qubit that she will prepare in
#    some arbitrary state
# 2. :math:`\vert\cdot\rangle_a`, Alice's auxiliary (or "ancilla") qubit that
#    she will entangle with Bob's qubit for communication purposes
# 3. :math:`\vert \cdot\rangle_B`, Bob's qubit that will receive the teleported
#    state
#
# Together, their starting state is:
#
# .. math::
#
#     \vert 0\rangle_A \vert 0\rangle_a \vert 0\rangle_B.
#
# The first thing Alice does is prepare her first qubit in whichever state :math:`\vert
# \psi\rangle` that she'd like to send to Bob, so that their combined state
# becomes:
#
# .. math::
#
#     \vert \psi\rangle_A \vert 0\rangle_a \vert 0\rangle_B.

import pennylane as qml
import numpy as np


def state_preparation(state):
    qml.QubitStateVector(state, wires=["A"])


##############################################################################
#
# 2. Shared entanglement
# ``````````````````````
#
# The reason why teleportation works as it does is the use of an *entangled state*
# as a shared resource between Alice and Bob. You can imagine some process that
# generates a pair of entangled qubits, and sends one qubit to each party. For
# simplicity (and simulation!), we will represent the entanglement process as
# part of our circuit.
#
# Entangling the second and third qubits leads to the combined state:
#
# .. math::
#
#     \frac{1}{\sqrt{2}}\left( \vert \psi\rangle_A \vert 0\rangle_a \vert 0\rangle_B + \vert \psi\rangle_A \vert 1\rangle_a \vert 1\rangle_B \right)\tag{1}


def entangle_qubits():
    qml.Hadamard(wires="a")
    qml.CNOT(wires=["a", "B"])


##############################################################################
#
# 3. Change of basis
# ``````````````````
#
# This is where things get tricky, but also very interesting. The third step of
# the protocol is to apply a :math:`CNOT` and a Hadamard to the first two qubits. This is
# done prior to the measurements, and labelled "change of basis". But what basis
# is this? Notice how these two gates are the *opposite* of what we do to create a
# Bell state. If we run them in the opposite direction, we transform the basis
# back to the computational one, and simulate a measurement in the *Bell
# basis*. The Bell basis is a set of four entangled states
#
# .. math::
#
#     \begin{align*}
#     \vert \psi_+\rangle &= \frac{1}{\sqrt{2}} \left( \vert 00\rangle + \vert 11\rangle \right), \\
#     \vert \psi_-\rangle &= \frac{1}{\sqrt{2}} \left( \vert 00\rangle - \vert 11\rangle \right), \\
#     \vert \phi_+\rangle &= \frac{1}{\sqrt{2}} \left( \vert 01\rangle + \vert 10\rangle \right), \\
#     \vert \phi_-\rangle &= \frac{1}{\sqrt{2}} \left( \vert 01\rangle - \vert 10\rangle \right).
#     \end{align*}
#
# After the basis transform, if we observe the first two qubits to be in the state
# :math:`\vert 00\rangle`, this would correspond to the outcome :math:`\vert \psi_+\rangle` in
# the Bell basis, :math:`\vert 11\rangle` would correspond to :math:`\vert \phi_-\rangle`,
# etc. Let's perform this change of basis, one step at a time.
#
# Suppose we write our initial state :math:`\vert \psi\rangle` as
# :math:`\alpha\vert 0\rangle + \beta\vert 1\rangle`, with :math:`\alpha` and
# :math:`\beta` being complex coefficients. Expanding out the terms from (1) (and
# removing the subscripts for brevity), we obtain:
#
# .. math::
#
#     \frac{1}{\sqrt{2}} ( \alpha\vert 000\rangle +
#     \beta\vert 100\rangle + \alpha \vert 011\rangle +
#     \beta\vert 111\rangle )
#
# Now let's apply a :math:`CNOT` between Alice's two qubits:
#
# .. math::
#
#     \frac{1}{\sqrt{2}} ( \alpha\vert 000\rangle +
#     \beta\vert 110\rangle + \alpha \vert 011\rangle +
#     \beta\vert 101\rangle )
#
# And then a Hadamard on her first qubit:
#
# .. math::
#
#     \frac{1}{2} ( \alpha \vert 000\rangle + \alpha\vert 100\rangle + \beta\vert 010\rangle - \beta\vert 110\rangle + \alpha \vert 011\rangle + \alpha \vert 111 \rangle + \beta\vert 001\rangle - \beta\vert 101 \rangle ).
#
# Now we need to do some rearranging. We group together the terms based on the first two qubits:
#
# .. math::
#
#     \frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\beta\vert 0\rangle + \alpha\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (-\beta\vert 0\rangle + \alpha\vert 1\rangle).\tag{2}
#


def basis_rotation():
    qml.CNOT(wires=["A", "a"])
    qml.Hadamard(wires="A")


##############################################################################
#
# 4. Measurement
# ``````````````
#
# The last step of the protocol involves applying two controlled operations from
# Alice's qubits to Bob, a controlled-:math:`Z`, and a :math:`CNOT`, followed by a
# measurement. But why exactly are we doing this before the measurement? In the
# previous step, we already performed a basis rotation back to the computational
# basis, so shouldn't we be good to go? Not quite, but almost!
#
# Let's take another look at equation (2). If Alice measures her two qubits in the
# computational basis, she is equally likely to obtain any of the four possible
# outcomes. If she observes the first two qubits in the state :math:`\vert 00 \rangle`,
# she would immediately know that Bob's qubit was in the state
# :math:`\alpha \vert 0 \rangle + \beta \vert 1 \rangle`, which is precisely the
# state we are trying to teleport!
#
# If instead she observed the qubits in state :math:`\vert 01\rangle`, she'd still
# know what state Bob has, but it's a little off from the original state. In particular,
# we have:
#
# .. math::
#
#     \beta \vert 0 \rangle + \alpha \vert 1 \rangle = X \vert \psi \rangle.
#
# After obtaining these results, Alice could tell Bob to simply apply an :math:`X`
# gate to his qubit to recover the original state. Similarly, if she obtained
# :math:`\vert 10\rangle`, she would tell him to apply a :math:`Z` gate.
#
# In a more `"traditional" version of
# teleportation <https://quantum.country/teleportation>`__ [#Teleportation1993]_,
# this is, in fact, exactly what happens. Alice would call up Bob on the phone,
# tell him which state she observed, and then he would be able to apply an appropriate
# correction. In this situation, measurements are happening partway through the protocol,
# and the results would be used to control the application of future quantum gates. This
# is known as mid-circuit measurement, and such mid-circuit measurements are expressed
# in PennyLane using `qml.cond <https://docs.pennylane.ai/en/stable/code/api/pennylane.cond.html>`_.


def measure_and_update():
    m0 = qml.measure("A")
    m1 = qml.measure("a")
    qml.cond(m1, qml.PauliX)("B")
    qml.cond(m0, qml.PauliZ)("B")


##############################################################################
#
# We've now defined all the building blocks for the quantum teleportation
# protocol. Let's put it all together!


def teleport(state):
    state_preparation(state)
    entangle_qubits()
    basis_rotation()
    qml.Barrier(["A", "a"], only_visual=True)
    measure_and_update()


state = np.array([1 / np.sqrt(2) + 0.3j, 0.4 - 0.5j])
_ = qml.draw_mpl(teleport, style="sketch")(state)

##############################################################################
#
# There is a neat concept known as the `principle of deferred measurement
# <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`__ [#NandC2000]_,
# and it basically states that we can push all our measurements to the *end*
# of our circuit. This can be useful for a variety of reasons, such as when
# working in a system that does not support mid-circuit measurements. In
# PennyLane, when you bind a circuit to a device that does not support them,
# it will automatically apply the principle of deferred measurement and update
# your circuit to use controlled operations instead.

dev = qml.device("default.qubit", wires=["A", "a", "B"])


@qml.qnode(dev)
def teleport(state):
    state_preparation(state)
    entangle_qubits()
    basis_rotation()
    measure_and_update()
    return qml.state()


_ = qml.draw_mpl(teleport, style="sketch")(state)

##############################################################################
#
# Poof! Our classical signals have been turned into :math:`CZ` and :math:`CNOT` gates.
# You might have wondered why these two gates were included in the measurement box
# in the diagrams; this is why. Alice applying controlled operations on Bob's
# qubit is performing this same kind of correction *before* any measurements are
# made. Let's evaluate the action of the :math:`CNOT` and :math:`CZ` on Bob's
# qubit, and ensure that Alice's state been successfully teleported. Applying
# the :math:`CNOT` yields:
#
# .. math::
#
#     \frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle)
#
# Then, applying the :math:`CZ` yields:
#
# .. math::
#
#     \frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle)\tag{3}
#
# When Alice measures her two qubits at the end, no matter which outcome she
# gets, Bob's qubit will be in the state :math:`\alpha\vert 0\rangle + \beta \vert
# 1\rangle`. This means that our protocol has changed the state of Bob's qubit
# into the one Alice wished to send him, which is truly incredible!
#
# :func:`qml.state <pennylane.state>` will return the state of the overall system,
# so let's inspect it to validate what we've theorized above. Re-arranging equation
# (3), we can see that the final state of the system is:
#
# .. math::
#
#     \frac{1}{2} (\vert 00\rangle + \vert 01\rangle + \vert 10\rangle + \vert 11\rangle) \vert \psi\rangle\tag{4}
#
# Now, we can confirm that our implementation of the quantum teleportation protocol
# is working as expected by reshaping the resulting state to match (4):


def teleport_state(state):
    system_state = teleport(state)
    system_state = qml.math.reshape(system_state, (4, 2))

    if not np.allclose(system_state, state / 2):
        raise ValueError(
            f"Alice's state ({state}) not teleported properly. "
            f"Current system state: {system_state}"
        )
    print("State successfully teleported!")


teleport_state(state)

##############################################################################
#
# References
# ------------
#
# .. [#Teleportation1993]
#
#     C. H. Bennett, G. Brassard, C. Crépeau, R. Jozsa, A. Peres, W. K. Wootters (1993)
#     `"Teleporting an unknown quantum state via dual classical and Einstein-Podolsky-Rosen channels"
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.1895>`__,
#     Phys. Rev. Lett. 70, 1895.
#
# .. [#NandC2000]
#
#     M. A. Nielsen, and I. L. Chuang (2000) "Quantum Computation and Quantum Information",
#     Cambridge University Press.
#
# .. [#Codebook]
#
#     C. Albornoz, G. Alonso, M. Andrenkov, P. Angara, A. Asadi, A. Ballon, S. Bapat, I. De Vlugt,
#     O. Di Matteo, P. Finlay, A. Fumagalli, A. Gardhouse, N. Girard, A. Hayes, J. Izaac, R. Janik,
#     T. Kalajdzievski, N. Killoran, J. Soni, D. Wakeham. (2021) Xanadu Quantum Codebook.

##############################################################################
#
# About the author
# ----------------
#
# .. include:: ../_static/authors/matthew_silverman.txt
