r"""Introduction to mid-circuit measurements
============================================

TODO:

- [ ] Write introduction

.. figure:: ../_static/demonstration_assets/mcm_introduction/socialthumbnail_mcm_introduction.png
    :align: center
    :width: 50%

"""

######################################################################
#
# Prelude: Classical probabilistic states
# ---------------------------------------
# *Disclaimer:* Feel free to skip this section
# if you feel comfortable with probabilistic aspects of quantum mechanics.
#
# Before we dive into the quantum mechanical measurement problem, let's briefly look at a
# purely classical probabilistic example. This will help us to understand **some** part
# of the quantum mechanical processes below. However, it is important that **this
# example does not provide a full analogy to quantum mechanics**.
#
# Imagine the following: a colleague rolls a fair six-sided dice but does not
# reveal the result to us. How can we describe the state of this dice appropriately?
# We know the possible states it *can* be in (:math:`1,\cdots,6`), and we know their
# probabilities (:math:`\frac{1}{6}` each). So unless we
# learn the result of the dice roll, it will be a good idea to use this knowledge to
# describe the state:
#
# .. math:: \vec{p} = \sum_{i=1}^6 \frac{1}{6} \vec{e}_i.
#
# We interpret the individual states :math:`\vec{e}_i` as basis vectors, so that
# :math:`\vec{p}` simply denotes the usual probability vector for the dice.
#
# This description allows us to compute quantities of interest! Maybe the simplest question
# we could ask about the state is the expected number of pips on the dice.
# To compute it, we simply need to multiply the state :math:`\vec{p}` with
# the "pip counting" vector :math:`\vec{\operatorname{count}}=(1, 2, 3, 4, 5, 6)^T`,
# so that
#
# .. math:: \vec{\operatorname{count}}\cdot\vec{p} = \sum_{i=1}^6 \frac{1}{6} i = 3\frac{1}{2}.
#
# Now assume that the colleague reveals the result of their dice roll only
# if it shows, say, :math:`5` pips. In this case, the state simply becomes :math:`\vec{p}=\vec{e}_5`.
# Accordingly, the expectation value will change to
#
# .. math:: \vec{\operatorname{count}}\cdot\vec{p} = 5.
#
# This description of a probabilistic mixture will be useful below to
# understand the difference of superposition between quantum states and purely classical
# probability theory, and why recording a measurement outcome can make a difference.
#
# Measurements in quantum mechanics
# ---------------------------------
#
# Measurements are the concern of important questions in quantum mechanics:
# What is a measurement? How does it affect the measured system? And how can we
# describe a measurement process mathematically?
# Given how fundamental those question are, there is a plethora of learning
# resources on this topic, from textbooks and (video) lectures to
# interactive studying material. Furthermore, discussing measurements quickly
# leads to questions about the interpretation of quantum mechanics and philosophical,
# if not metaphysical, discourse.
# For these reasons, we will not aim at discussing those deep question in great detail,
# but focus on understanding the basics that will help us understand measurements
# performed within quantum circuits, i.e., mid-circuit measurements, and how to realize
# them in PennyLane.
#
# Bare with us, we will briefly look at a definition for measurements but then turn
# to practical examples and some hands-on calculations.
#
# Mathematical description
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We will go with the following definition: A measurement :math:`M` is a process that
# maps a valid quantum state :math:`\rho` to a classical probabilistic mixture
#
# .. math:: M[\rho]=\sum_{i=1}^n p_i \rho_i
#
# of post-measurement quantum states :math:`\rho_i` that are specified by :math:`M`.
# Here, :math:`n` is the number of possible measurement outcomes and :math:`p_i`
# is the probability to measure the outcome :math:`i` associated to :math:`\rho_i`,
# given the input state :math:`\rho`.
#
# The expression above describes the classical mixture after the quantum mechanical
# measurement if we do *not* record the measurement outcome. This is similar
# to the state of the dice in the prelude before our colleague reveals the result
# of the roll. If we do record the measurement outcome,
# we no longer have a probabilistic mixture, but find the state :math:`\rho_i` for
# the observed outcome :math:`i`. This corresponds to our colleague telling
# us the number of pips on the dice.
#
# For the rest of this tutorial, we will restrict ourselves to standard
# measurements commonly found in MCMs, using so-called projection-valued measures (PVMs).
# In this setting, the measurement comes with one projector :math:`\Pi_i` per
# measurement outcome, and all projectors sum to the identity.
# The post-measurement states are given by
#
# .. math:: \rho_i = \frac{\Pi_i \rho \Pi_i}{\operatorname{tr}[\Pi_i \rho]}
#
# and the probabilities are dictated by the Born rule, :math:`p_i=\operatorname{tr}[\Pi_i \rho]`.
# This means that if we do not record the measurement outcome, the state simply is
#
# .. math:: M[\rho] = \sum_{i=1}^n \Pi_i \rho \Pi_i.
#
# To understand this abstract description better, let's look at two simple examples:
#
# Measuring a single qubit
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Consider a single qubit in the state :math:`|+\rangle`, i.e., in the equal superposition
# of :math:`|0\rangle` and :math:`|1\rangle`.
# To get started, let's first implement this state in PennyLane and compute some expectation
# values that will be insightful later on. We follow these steps:
#
# - Import PennyLane and define a device using :func:`~.pennylane.device`. ``"default.qubit"`` will suffice
#   for our purposes.
#
# - Write a quantum function that first creates the :math:`|+\rangle` state using :class:`~.pennylane.Hadamard`
#   and them measures :math:`\langle X\rangle` and :math:`\langle Z\rangle` in this state using :func:`~.pennylane.expval`
#
# - Decorate the function with :func:`~.pennylane.qnode`, turning the quantum function into a quantum node.
#
# - Run the quantum node and show the computed expectation values!
#
# If you'd like more guidance on any of these steps, also have a look at
# :doc:`our tutorial on qubit rotation </demos/tutorial_qubit_rotation>` explaining them in detail.

import pennylane as qml

dev = qml.device("default.qubit")


@qml.qnode(dev)
def before():
    qml.Hadamard(0)  # Create |+> state
    return qml.expval(qml.X(0)), qml.expval(qml.Z(0))


b = before()
print(f"Expectation values before any measurement: {b[0]:.1f}, {b[1]:.1f}")

######################################################################
# The result is not surprising: :math:`|+\rangle` is the eigenstate of :math:`X` for the eigenvalue
# :math:`+1`, and it has the well-known expectation value :math:`\langle +|Z|+\rangle=0`.
#
# Now we bring in a mid-circuit measurement, namely in the computational, or Pauli-:math:`Z`, basis.
# It comes with the projections :math:`\Pi_i=|i\rangle\langle i|`,
# :math:`i\in\{0, 1\}`, onto the computational basis states.
# If we execute the measurement process but do not record the outcome, we find the state
#
# .. math::
#
#     M[\rho]
#     &= p_0 \rho_0 + p_1 \rho_1\\
#     &= |0\rangle\langle 0|+\rangle\langle +|0\rangle\langle 0| + |1\rangle\langle 1|+\rangle\langle +|1\rangle\langle 1|\\
#     &= \frac{1}{2}\mathbb{I}.
#
# where we used the overlaps :math:`\langle +|i\rangle=1/\sqrt{2}`.
# This means that the measurement sends the qubit from a pure state into a mixed state,
# i.e., it not only affects the state but even the *class* of states it is in. And this
# is despite, no because, we did not even record the measurement outcome!
#
# Let's look at this example in PennyLane. We repeat the steps from above but additionally
# include a mid-circuit measurement, using :func:`~.pennylane.measure`. Note that we
# just perform the measurement and do not assign any variable to its outcome.
#


@qml.qnode(dev)
def after():
    qml.Hadamard(0)  # Create |+> state
    qml.measure(0)  # Measure without recording the outcome
    return qml.expval(qml.X(0)), qml.expval(qml.Z(0))


a = after()
print(f"Expectation value after the measurement:  {a[0]:.1f}, {a[1]:.1f}")


######################################################################
# The measurement moved the qubit from the :math:`+1`\-eigenstate :math:`|+\rangle` of
# the Pauli-:math:`X` operator into a mixed state with expectation value zero for all
# Pauli operators, explaining the values we just observed.
#
# Now if we actually record the outcome, say it was a :math:`0`, we find the state
#
# ..math::
#
#    M[\rho]=\rho_0
#    &=\frac{|0\rangle\langle 0|+\rangle\langle +|0\rangle\langle 0|}{\operatorname{tr}[|0\rangle\langle 0|+\rangle\langle +|]}\\
#    &=|0\rangle\langle 0|.
#
# The qubit is in a new, pure state. In PennyLane, we can postselect on the case where we
# measured a :math:`0` using the ``postselect`` keyword argument of ``qml.measure``:
#


@qml.qnode(dev)
def after():
    qml.Hadamard(0)  # Create |+> state
    qml.measure(0, postselect=0)  # Measure and only accept 0 as outcome
    return qml.expval(qml.X(0)), qml.expval(qml.Z(0))


a = after()
print(f"Expectation value after the postselected measurement:  {a[0]:.1f}, {a[1]:.1f}")

######################################################################
# As expected, we find the that the measured, postselected qubit is in the
# :math:`+1`\-eigenstate of the Pauli-:math:`Z` operator, yielding :math:`\langle X\rangle=0`
# and :math:`\langle Z\rangle=1`. For ``postselect=1``, we would have obtained
# the :math:`-1`\-eigenstate of :math:`Z` instead.
#
# Measuring a Bell pair
# ~~~~~~~~~~~~~~~~~~~~~
# Next, we consider a pair of qubits, entangled in a Bell state:
#
# .. math:: |\phi\rangle = \frac{1}{\sqrt{2}} (|00\rangle + |11\rangle).
#
# This is a pure state with density matrix
#
# .. math:: |\phi\rangle\langle \phi | = \frac{1}{2}\left(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|\right).
#
# We again measure only the first qubit, keeping the projections :math:`\Pi_i` the same.
# And we again code this circuit up, using an additional ``CNOT`` gate to create the
# Bell state. We include the postselection as an argument to our quantum function, but do
# not complete the quantum function yet, as we still need to discuss what to return from it.
#


def bell_pair_preparation(postselect):
    qml.Hadamard(0)  # |Make a Bell pair
    qml.CNOT([0, 1])  # |
    qml.measure(0, postselect=postselect)  # Measure and discard outcome or postselect


######################################################################
# Without recording the outcome, i.e., ``postselect=None``, we obtain the state
#
# .. math:: M[\rho] = \frac{1}{2}\left(|00\rangle\langle 00| + |11\rangle\langle 11|\right),
#
# which again could be described by a classical mixture as well.
# If we instead postselect on measuring, say, a :math:`1`, we find :math:`M[\rho] = |11\rangle\langle 11|`.
#
# There are two striking differences between whether we record the measurement outcome
# or not: the state of the qubits changes from a mixed to a pure state, as witnessed
# by the state's *purity*; and its entanglement changes, too, as witnessed by the
# *Von Neumann entanglement entropy*. We can compute both quantities easily in PennyLane, using
# :func:`~.pennylane.purity` and :func:`~.pennylane.vn_entropy`, respectively.
# And those will be the return types to complete our quantum function, so that we can
# turn it into a ``QNode``:
#


@qml.qnode(dev)
def bell_pair(postselect):
    bell_pair_preparation(postselect)
    return qml.purity([0, 1]), qml.vn_entropy(1)


######################################################################
# So let's compare the purities and Von Neumann entropies of the Bell state
# after measurement:
#

without_ps = bell_pair(None)
with_ps = bell_pair(1)
print(f"                     | without ps | with ps ")
print(f"Purity               |     {without_ps[0]:.1f}    |   {with_ps[0]:.1f}")
print(f"Entanglement entropy |     {without_ps[1]:.2f}   |  {with_ps[1]:.1f}")

######################################################################
#
# Qubit reset
# ~~~~~~~~~~~
#
# Another commonly used feature of mid-circuit measurements is to reset the measured
# qubit to the :math:`|0\rangle` state.
#
#
# Dynamically controlling a quantum circuit
# -----------------------------------------
# So far we only talked about MCMs that affect the state of qubits, about postselection, and
# about qubit reset as an additional step after performing the measurement.
# However, the outcomes of a measurement can not only be used to decide whether or not to discard
# a circuit execution. More importantly, as MCMs are performed while the quantum circuit is
# up and running, their outcomes can be used to *conditionally* modify the structure of the
# circuit itself!
#
#
#
# About the author
# ----------------
#
