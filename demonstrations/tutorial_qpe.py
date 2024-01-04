r"""Intro to Quantum Phase Estimation
=============================================================

The Quantum Phase Estimation (QPE) algorithm is one of the most important tools in quantum
computing. Maybe THE most important. It solves a deceptively simple task: given an eigenstate of a
unitary operator, find its eigenvalue. Innocent as it may seem, being able to solve this problem is
a superpower that captures the core principles of quantum computing.

.. figure:: ../_static/demonstration_assets/qpe/socialthumbnail_large_Quantum_Phase_Estimation_2023-11-27.png
    :align: center
    :width: 50%

This demo explains the basics of the QPE algorithm. After reading it, you will be able to understand
QPE and how to implement it in PennyLane.

Quantum phase estimation
------------------------

Let's definine the problem more carefully. We are given a unitary
operator :math:`U` and one of its eigenstates :math:`|\psi \rangle`. We can prepare this
eigenstate exactly on a quantum computer (we'll relax that assumption later). The operator is unitary,
so we can write:

.. math::
    U |\psi \rangle = e^{i \phi} |\psi \rangle,

where :math:`\phi` is the *phase* of the eigenvalue. The goal is to estimate :math:`\phi`
(hence the name phase estimation). Our challenge is to design a quantum algorithm to solve this problem.
How would that work? 🧠



Part 1: Encoding the phase
--------------------------
We could solve the problem by finding a quantum circuit that performs the transformation

.. math::
    |\psi \rangle |0\rangle \rightarrow  |\psi \rangle |\phi\rangle.

By measuring the second register we obtain :math:`\phi` directly. We refer to the first register as the **target**,
and the second one is the **estimation** register.

But let's be more careful. Because the
complex exponential has period :math:`2\pi`, technically the phase is not unique. Instead, we
define :math:`\phi = 2\pi \theta` so that :math:`\theta` is a number between 0 and 1; this forces :math:`\phi`
to be between 0 and :math:`2\pi`. We'll refer to :math:`\theta` as the phase from now on.

How can we represent :math:`\theta` on a quantum computer? The answer is the first clever part of the algorithm: we represent
:math:`\theta` in binary. 🧠

Since you probably don't use binary fractions on a daily basis, (or do you?) it's worth stopping for a moment
to make sure we're on the same page.

.. note::
    **Binary fractions**

    When we write the number 0.15625, it is being expressed as a sum of multiples of powers of
    10:

    .. math::
        0.15625 = 1 \times 10^{-1} + 5 \times 10^{-2} + 6 \times 10^{-3} + 2 \times 10^{-4} +  5 \times 10^{-5}.

    But nothing is stopping us from using 2 instead of 10. In binary, the same number is

    .. math::
        0.00101 = 0 \times 2^{-1} + 0 \times 2^{-2} + 1 \times 2^{-3} + 0 \times 2^{-4} +  1 \times 2^{-5}.

    (You can confirm this by computing 1/8 + 1/32 on a calculator). Similarly, 0.5 in decimal is 0.1 in binary,
    and 0.3125 is 0.0101.

Ok, now back to quantum. A binary representation is useful because we can encode it using
qubits, e.g., :math:`|110010\rangle` for :math:`\theta=0.110010`. The phase is retrieved by measuring the qubits.
The **precision** of the estimate is determined by the number of qubits. We've used examples of fractions that can be
conveniently expressed exactly with just a few binary points, but this won't
always be possible. For example, the binary expansion of :math:`0.8` is :math:`0.11001100...` which does not terminate.
From now on, we'll use :math:`n` for the number of estimation qubits.

Part 2: Quantum Fourier Transform
---------------------------------

The second clever part of the algorithm is to follow an advice given to many physicists:
"When in doubt, take the Fourier transform"; or in our case, "When in doubt, take the quantum Fourier transform (QFT)".

.. math::
   \text{QFT}|\theta\rangle = \frac{1}{\sqrt{2^n}}\sum_{k=0} e^{2 \pi i\theta k} |k\rangle.

Note that it is the equiprobable superposition where each state has an additional phase.
If we can prepare that state, then applying the *inverse* QFT would gives
:math:`|\theta\rangle` in the estimation register.
This looks more promising, especially if we notice the appearance of the eigenvalues :math:`e^{2 \pi i\theta}`,
although with an extra factor of :math`k`. We can obtain this factor by applying the unitary :math:`k` times to the state :math:`|\psi\rangle`:

.. math::
   U^k|\psi\rangle =  e^{2\pi i \theta k} |\psi\rangle.

Therefore, we will use :math:`|\psi\rangle` and :math:`U` to generate the factors that are of interest to us in each of the basic states.
If we pay attention carefully we can see that it would be enough to create an operator such that:

.. math::
   |\psi\rangle |k\rangle \rightarrow  U^k |\psi\rangle |k\rangle.


In this way, if we apply this operator to the equiprobable superposition we would have that:

.. math::
   |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0}|k\rangle \rightarrow |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0}U^k|k\rangle =  |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} e^{2 \pi i\theta k} |k\rangle

This is exactly what we want!
In PennyLane we refer to this operator as a :class:`~.ControlledSequence` operation. Let's see how to build it.

Part 3: Controlled sequence
---------------------------

We follow another timeless physics advice: "If stuck, start with the simplest case".
This means just one estimation qubit. After applying a Hadamard, the circuit we need is

.. math::
   \frac{1}{\sqrt{2}} |\psi\rangle |0\rangle + \frac{1}{\sqrt{2}} |\psi\rangle |1\rangle \rightarrow
   \frac{1}{\sqrt{2}} |\psi\rangle |0\rangle + \frac{1}{\sqrt{2}} U|\psi\rangle |1\rangle.

We know how to do this: it's just a controlled-U operation. What about two qubits? Omitting normalization factors,
we need

.. math::
   |\psi\rangle |00\rangle + |\psi\rangle |01\rangle + |\psi\rangle |10\rangle+  |\psi\rangle |11\rangle\rightarrow
   |\psi\rangle |00\rangle + U |\psi\rangle |01\rangle + U^2 |\psi\rangle |10\rangle+ U^3 |\psi\rangle |11\rangle.

Notice something? This is just applying :math:`|\psi\rangle |k\rangle \rightarrow U^k |\psi \rangle |k\rangle` with
:math:`k` expressed in binary. It can be implemented by applying :math:`U` controlled on the first qubit, and
:math:`U^2` controlled on the second qubit. We can extend this idea to any number of qubits.

With six qubits, an example would be

.. math::
   |\psi\rangle |010111\rangle \rightarrow U^1U^2U^4U^{16}|\psi\rangle |010111\rangle = U^{23}|\psi\rangle |010111\rangle.

(Note that 010111 is 23 in binary.)

So we have the answer: apply :math:`U^{2^m}` controlled on the `m`-th estimation qubit.

Bringing it all together, here is the quantum phase estimation algorithm in all its glory:

The QPE algorithm
-----------------

1. Start with the state :math:`|\psi \rangle |0\rangle`. Apply a Hadamard gate to all estimation qubits to implement the
   transformation

   .. math::

       |\psi \rangle |0\rangle \rightarrow |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} |k\rangle.

2. Apply a :class:`~.ControlledSequence` operation, i.e., :math:`U^{2^m}` controlled on the `m`-th estimation qubit.
   This gives

   .. math::

       |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} |k\rangle \rightarrow  |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} e^{2\pi i \theta k}|k\rangle.

3. Apply the inverse quantum Fourier transform to the estimation qubits

   .. math::

      |\psi\rangle \frac{1}{\sqrt{2^n}}\sum_{k=0} e^{2 \pi i \theta k}|k\rangle \rightarrow |\psi\rangle|\theta\rangle.

4. Measure the estimation qubits to recover :math:`\theta`.



Most of the heavy lifting is done by the controlled sequence step. Control-U operations are the heart of the algorithm,
coupled with a clever use of Fourier transforms.

QPE is doing something incredible: it can calculate eigenvalues **without ever diagonalizing
a matrix**. Wow. This is true even if we relax the assumption that the input is an eigenstate. By linearity, for an arbitray
state expanded in the eigenbasis of :math:`U` as

.. math::
   \Psi\rangle = \sum_i c_i |\psi_i\rangle,

QPE outputs the eigenphase :math:`\theta_i` with probability :math:`|c_i|^2`.
This feature is crucial for quantum chemistry applications,
where preparing good initial states is essential [#initial_state]_.
If you want to learn more about this check out our :doc:`demo <tutorial_initial_state_preparation>`.

One more point of importance. Generally it is not possible to represent a given phase exactly using a limited number of
estimation qubits. Thus, there is typically a distribution of possible
outcomes, which induce an error in the estimation. We'll see an example in the code below.
The error *decreases* exponentially with the number of estimation qubits, but the number of controlled-U operations
*increases* exponentially. The math is such that these effects basically cancel out and the cost of estimating a phase
with error :math:`\varepsilon` is poportional to :math:`1/\varepsilon`.


Time to code!
-------------
We already know the three building blocks of QPE; it is time to put them to practice.
We use a single-qubit :class:`~pennylane.PhaseShift` operator :math:`U = R_{\phi}(2 \pi / 5)`
and its eigenstate :math:`|1\rangle`` with corresponding phase :math:`\theta=0.2`.

"""
import pennylane as qml
import numpy as np


def U(wires):
    return qml.PhaseShift(2 * np.pi / 5, wires=wires)

##############################################################################
# We construct a uniform superposition by applying Hadamard gates followed by a :class:`~.ControlledSequence`
# operation. [JM: good to take time here to explain how ControlledSequence works].
# Finally, we perform the adjoint of :class:`~.QFT` and
# return the probability of each computational basis state.


dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit_qpe(estimation_wires):
    # initialize to state |1>
    qml.PauliX(wires=0)

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(U(wires=0), control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)

##############################################################################
# Let's run the circuit and plot the results. We use 8 estimation qubits.


import matplotlib.pyplot as plt

estimation_wires = range(1, 9)

results = circuit_qpe(estimation_wires)
plt.bar(range(len(results)), results)
plt.xlabel("frequency")
plt.show()

##############################################################################
# As mentioned above, since the eigenphase cannot be represented exactly using 8 bits, there is a
# distribution of possible outcomes. [JM: rewrite this to express as binary fraction]. The peak occurs
# at :math:`\phi` is :math:`0.19921875`, which is very close to the exact value of :math:`0.2` 🎊.
#
# Conclusion
# ----------
# This demo presented the "textbook" version of QPE. There are multiple variations, notably iterative QPE that
# uses a single estimation qubit, as well as Bayesian versions that saturate optimal prefactors appearing in the
# total cost. There are also mathematical subtleties about cost and errors that are important but out of
# scope for this demo.
#
# Finally, there is extensive work on how to implement the unitaries themselves. In quantum chemistry,
# the main strategy is to encode a molecular Hamiltonian
# into a unitary such that the phases are invertible functions of the Hamiltonian eigenvalues. This can be done for instance
# through the mapping :math:`U=e^{-iHt}`, which can be implemented using Hamiltonian simulation techniques. QPE can then
# be used to estimate eigenvalues like ground-state energies by sampling them with respect to a distribution
# induced by the input state.
#
# References
# ---------------
#
# .. [#initial_state]
#
#    Stepan Fomichev et al. "Initial state preparation for quantum chemistry on quantum computers",
#    `Arxiv <https://arxiv.org/pdf/2310.18410.pdf/>`__, 2023
#
#
# About the authors
# -----------------
#
