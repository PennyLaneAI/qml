r"""Constant-depth preparation of matrix product states with dynamic circuits
=============================================================================

Matrix product states (MPS) are an important class of quantum states that is
used in a plethora of applications on quantum many-body systems.
This makes the preparation of MPS on a quantum computer a useful subroutine.
For example, a quantum algorithm could prepare a classically pre-computed MPS
as initial state in order to then perform operations that are no longer
classically tractable.

.. figure:: ../_static/todo
    :align: center
    :width: 50%

In this demo you will learn about a new algorithm to prepare MPS with a dynamic
quantum circuit of constant depth. We will implement the circuit, which makes use
of mid-circuit measurements and conditionally applied operations, for a specific
MPS, and compare it to a (static) sequential circuit from the literature with
linear depth.

This demo closely follows the eponymous article by Smith et al. [#smith]_.

If you would like to first learn about the dynamic circuit tools used in this
algorithm, have a look at our :doc:`introduction to mid-circuit measurements
</demos/tutorial_mcm_introduction>` and learn :doc:`how to collect statics of
mid-circuit measurements </demos/tutorial_how_to_collect_mcm_stats>` and
:doc:`how to create dynamic circuits with mid-circuit measurements
</demos/tutorial_how_to_create_dynamic_mcm_circuits>`.

If you first want to familiarize yourself with tensor network states and MPS
in particular, take a look at our demo on
:doc:`tensor-network quantum circuits </demos/tutorial_tn_circuits>`.
Finally, our demo on :doc:`initial state preparation for quantum chemistry
</demos/tutorial_initial_state_preparation>` shows you how to
proceed to a concrete application, once an MPS is prepared.

Outline
-------

We will start with a brief (no, really!) mathematical overview of the used
objects and subroutines. Then we define them in code for a specific parametric MPS
and verify the conditions for the algorithm to apply.
One of the subroutines is the sequential MPS preparation circuit that we
compare to later.
Afterwards we combine the building blocks into the constant-depth algorithm
by Smith et al. We conclude by computing the correlation length of the prepared
MPS for different parameter values.

Mathematical derivation
-----------------------

We briefly introduce matrix product states and a sequential circuit with linear depth
that prepares them, as well as two tools from quantum information theory
that we will use below, namely fusion of MPS with mid-circuit measurements
and so-called operator pushing.

Matrix product states
~~~~~~~~~~~~~~~~~~~~~

Matrix product states (MPS) are an important class of quantum states.
They can be described and manipulated conveniently on classical computers and
are able to approximate relevant states in quantum many-body systems.
In particular, MPS can accurately describe ground states of (gapped local)
one-dimensional Hamiltonians, which can be found efficiently using the density
matrix renormalization group (DMRG).
For reviews of MPS see [#todo]_, [#todo]_, and [#todo]_.

Following [#smith]_, we will look at translation-invariant MPS
of a quantum :math:`N`-body system where each body, or site, has local (physical)
dimension :math:`d`. We will further restrict to periodic boundary conditions,
corresponding to translational invariance of the MPS on a closed loop.
A general MPS with this properties is given by

.. math::

    |\Psi\rangle = \sum_{\vec{m}} \operatorname{tr}[A^{m_1}A^{m_2}\cdots A^{m_N}]|\vec{m}\rangle,

where :math:`\vec{m}` is a multi-index of :math:`N` indices ranging over :math:`d`,
i.e., :math:`\vec{m}\in\{0, 1 \dots, d-1\}^N`, and :math:`\{A^{m}\}_{m}` are :math:`d`
square matrices of equal dimension. This dimension :math:`D` is called the bond dimension,
and it is a crucial quantity for the expressivitiy and complexity of the MPS.
We see that :math:`|\Psi\rangle` is fully specified by the :math:`d\times D\times D=dD^2`
numbers in the rank-3 tensor :math:`A`.
However, this specification is not unique. We remove some of the redundancies by assuming
that :math:`A` is in so-called left-canonical form, meaning that it satisfies

.. math::

    \sum_m {A^m}^\dagger A^m = \mathbb{I}.

We will not discuss additional redundancies here but refer to [#smith]_ and the
mentioned reviews for more details.

**Example**

As an example, consider an :math:`N`-site qubit (:math:`d=2`) chain and a :math:`D=2` MPS
:math:`|\Psi(g)\rangle` on this system defined by the matrices

.. math::

    A^0 = \eta \left(\begin{matrix} 1 & 0 \\ \sqrt{-g} & 0 \end{matrix}\right)
    \quad
    A^1 = \eta \left(\begin{matrix} 0 & -\sqrt{-g} \\ 0 & 1 \end{matrix}\right)

where :math:`\eta = \sqrt{1-g}^{-1}` and :math:`g\in[-1, 0]` is a freely chosen parameter.

This MPS is an important example because it can be tuned from the long-range correlated
GHZ state :math:`|\Psi(0)\rangle=(|0>^{\otimes N} + |1>^{\otimes N})/\sqrt(2)`
to the state :math:`|\Psi(-1)\rangle=|+\rangle^{\otimes N}` with vanishing correlation length.
In general, the correlation length is given by

.. math::

    \xi = \left|\ln\left(\frac{1+g}{1-g}\right)\right|^{-1}.

Long-range correlated states require linear-depth unitary circuits in general.
Therefore, the constant-depth circuit below for :math:`|\Psi(0)\rangle` will demonstrate
that dynamic quantum circuits are more powerful than unitary operations alone.

The MPS :math:`|\Psi(g)\rangle` will be our working example throughout this demo. Therefore
we warm up our coding by defining the tensor :math:`A` and testing some of its properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pennylane as qml


def A(g):
    """Compute the tensor A in form of d=2 matrices with shape (D, D) = (2, 2) each."""
    eta = 1 / np.sqrt(1 - g)
    A_0 = np.array([[1, 0], [np.sqrt(-g), 0]]) * eta
    A_1 = np.array([[0, -np.sqrt(-g)], [0, 1]]) * eta
    return (A_0, A_1)


g = -0.6
A_0, A_1 = A(g)
is_left_canonical = np.allclose(A_0.conj().T @ A_0 + A_1.conj().T @ A_1, np.eye(2))
print(f"The matrices A^m are in left-canonical form: {is_left_canonical}")

xi = 1 / abs(np.log((1 + g) / (1 - g)))
print(f"For {g=}, the theoretical correlation length is {xi=:.4f}")

######################################################################
# Sequential preparation circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# An MPS like the one above can be prepared in linear depth using an established technique
# by Schön et al. [#schoen]_. We introduce this technique here because the new
# constant-depth construction uses it as a parallelized subroutine, and because we will
# later compare the new construction to the sequential circuit.
#
# The MPS :math:`|\Psi\rangle` above is given by the rank-3 tensor :math:`A`.
# In order to prepare the state in a circuit, we want to construct unitary operations
# from :math:`A`. For this, we build a new rank-4 tensor :math:`U` with an additional
# axis of dimension :math:`d`. If this new physical input axis is in the state
# :math:`|0\rangle`, we define :math:`U` to reproduce one :math:`A^m` for each physical
# output state :math:`|m\rangle`. We leave the remaining action undefined and only require
# it to make :math:`U` unitary overall. In short, this can be written as
#
# .. math::
#
#     U = \sum_m A^m \otimes |m\rangle\!\langle 0| + C_\perp.
#
# Let us denote :math:`U` acting on the :math:`i`-th physical site as :math:`U^{(i)}`
# for a fixed non-physical site.
# The sequential preparation circuit now consists of the following steps.
#
# #. Start in the state :math:`|\psi_0\rangle = |0_d\rangle^{\otimes N}\otimes |00\rangle_D`.
#    The last two sites are non-physical bond sites, encoded by :math:`D`-dimensional qudits.
#
# #. Entangle the bond qudits into the state :math:`|I\rangle = 1/\sqrt{D}\sum_j |jj\rangle_D`.
#
# #. Apply the unitary :math:`U` once to each of the :math:`N` physical sites, with the
#    :math:`D`-dimensional factor always acting on the first bond qudit. This produces the state
#
#    .. math::
#
#        |\psi_1\rangle
#        &= \frac{1}{\sqrt{D}} \sum_j \prod_{i=N-1}^{1} U^{(i)}
#        \sum_{m_N=0}^{d-1} |0_d\rangle^{\otimes N-1}|m_N\rangle A^{m_N}|jj\rangle _D\\
#        &= \frac{1}{\sqrt{D}} \sum_j
#        \sum_{\vec{m}} |\vec{m}\rangle A^{m_1} A^{m_2}\cdots A^{m_N}|j\rangle_D |j\rangle_D\\
#        &= \frac{1}{\sqrt{D}} \sum_{i,j}
#        \sum_{\vec{m}} |\vec{m}\rangle \langle i|A^{m_1} A^{m_2}\cdots A^{m_N}|j\rangle_D |i\rangle_D|j\rangle_D
#
# #. Measure the two bond qudits in the (generalized) Bell basis and postselect on the outcome
#    being :math:`|I\rangle`. Then discard the bond qudits. This collapses :math:`\psi_1\rangle`
#    into the (renormalized) state
#
#    .. math::
#
#        |\psi_2\rangle = \sum_{\vec{m}} \operatorname{tr}[A^{m_1}A^{m_2}\cdots A^{m_N}]|\vec{m}\rangle = |\Psi\rangle
#
#    Note that this step is probabilistic and we only succeed to produce the state if we measured
#    the state :math:`|I\rangle`.
#
# We see that we prepared :math:`|\Psi\rangle` with an entangling operation on the bond qudits, one unitary per
# physical qubit, and a final basis change to measure the Bell basis. Overall, this amounts to a linear
# operation count and circuit depth.
#
# **Example**
#
# For our example MPS :math:`|\Psi(g)\rangle`, let us first find the unitary :math:`U`.
# For this, consider the fixed part of :math:`U`.

E_00 = np.array([[1, 0], [0, 0]])  # |0><0|
E_10 = np.array([[0, 0], [1, 0]])  # |1><0|
U_first_term = np.kron(A_0, E_00) + np.kron(A_1, E_10)

print(np.round(U_first_term, 5))
print(np.linalg.norm(U_first_term, axis=0))

######################################################################
# We see that this fixed part has two norm-1 columns already, and we are
# able to complement this by :math:`C_\perp` with the same columns:

E_01 = np.array([[0, 1], [0, 0]])  # |0><1|
E_11 = np.array([[0, 0], [0, 1]])  # |1><1|
C_perp = np.kron(A_1, E_01) + np.kron(A_0, E_11)

U = U_first_term + C_perp

print(np.round(U, 5))
print(f"U is unitary: {np.allclose(U.conj().T @ U, np.eye(4))}")


######################################################################
# Great! This means that we already found the unitary :math:`U` for this MPS.
# Combining the terms above on paper, we see that
#
# .. math::
#
#     U &= A^0 \otimes (|0\rangle\!\langle 0| + |1\rangle\!\langle 1|) + A^1\otimes (|1\rangle\!\langle 0| + |0\rangle\!\langle 1|)\\
#     &= A^0 \otimes \mathbb{I} + A^1\otimes X
#
# This looks a lot like a :class:`~.pennylane.CNOT` gate plays a part in :math:`U`.
# Removing it from :math:`U` we find
#
# .. math::
#
#     U \operatorname{CNOT}
#     &= A^0 P_0 \otimes \mathbb{I} + A^0P_1 \otimes X + A^1 P_0 \otimes X + A^1P_1 \otimes \mathbb{I}\\
#     &= \left(\begin{matrix} \eta & -\eta\sqrt{-g} \\ \eta\sqrt{-g} & \eta \end{matrix}\right),
#
# where we denoted the computational basis state projectors as :math:`P_i = |i\rangle\!\langle i|`.
# The remaining operation is a Pauli-:math:`Y` rotation by the angle :math:`2\arccos(\eta)`, so
# that we can easily code up the unitary as a small circuit template:


def U_template(eta, bond_idx, phys_idx):
    qml.CNOT([bond_idx, phys_idx])
    qml.RY(2 * np.arccos(eta), bond_idx)


eta = 1 / np.sqrt(1 - g)
template_matrix = qml.matrix(U_template, wire_order=["bond", "phys"])(eta, "bond", "phys")
print(f"The template reproduces U: {np.allclose(template_matrix, U)}")

######################################################################
# Great, now that we have a circuit that realizes the unitary :math:`U`, let's code up the entire
# sequential preparation circuit. We will leave the measurement \& postselect step as a separate
# function for now.


def prepare_bell(wire_0, wire_1):
    """Prepare two qubits in the Bell state (|00>+|11>)/sqrt(2)."""
    qml.Hadamard(wire_0)
    qml.CNOT([wire_0, wire_1])


def project_measure(wire_0, wire_1):
    """Measure two qubits in the Bell basis and
    postselect on (|00>+|11>)/sqrt(2)."""
    # Move bond qubits to Bell basis by undoing the preparation
    qml.adjoint(prepare_bell)(wire_0, wire_1)
    # Measure and postselect |00>
    qml.measure(wire_0, postselect=0)
    qml.measure(wire_1, postselect=0)


def sequential_preparation(g, wires):
    """Prepare the example MPS \Psi(g) on N qubits where N is the length
    of the passed wires minus 2. The bond qubits are still entangled."""
    eta = 1 / np.sqrt(1 - g)
    bond_wires = [wires[0], wires[-1]]  # Bond qubits [0, N+1]
    phys_wires = wires[1:-1]  # Physical qubits [1, 2, ... N]
    prepare_bell(*bond_wires)
    for phys_wire in phys_wires:
        U_template(eta, bond_wires[1], phys_wire)


dev = qml.device("default.qubit")


@qml.qnode(dev)
def sequential_circuit(N, g):
    """Run the preparation circuit and projectively measure the bond qubits."""
    wires = list(range(N + 2))
    sequential_preparation(g, wires)
    project_measure(0, N + 1)
    return qml.probs(wires=wires[1 : N + 1])


######################################################################
# We will verify the prepared state below, when comparing to the constant-depth
# preparation circuit. Let's just draw it for now.

N = 12
qml.draw_mpl(sequential_circuit)(N, g)

######################################################################
#
# Fusion of MPS states with mid-circuit measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The next ingredient to the constant-depth MPS preparation circuit is to fuse
# MPS states on different qubits together into one MPS. Under the hood, this is
# an application of entanglement swapping.
#
# Consider a state :math:`|\Phi\rangle=|\Psi_1\rangle\otimes|\Phi_2\rangle`, where
# :math:`|\Psi_{1,2}\rangle` are MPS prepared with the sequential technique from
# above on :math:`K` and :math:`L` physical qudits, respectively, but with the bond
# qudits left unmeasured. In particular, the postselection step has not been performed
# yet. Using the form of :math:`|\psi_1\rangle` from the recipe above, we can write
# this state as
#
# .. math::
#
#     |\Phi\rangle = \frac{1}{D} \sum_{i,j,\ell,p}
#     \sum_{\vec{m}} |\vec{m}\rangle  |ij\ell p\rangle_D
#     \langle i|A^{m_1} \cdots A^{m_K}|j\rangle_D
#     \langle \ell|A^{m_{K+1}} \cdots A^{m_{K+L}}|p\rangle_D.
#
# Now we want to measure two bond qudits, one of :math:`|\Psi_1\rangle` and
# :math:`|\Psi_2\rangle` each, in an entangled basis. Assume this basis to be given
# in the form of (orthogonal) states
#
# .. math::
#
#     |B^k\rangle=\frac{1}{D} \sum_{j,\ell=1}^D \left(B_{j\ell}^k\right)^\ast |j\ell\rangle,
#
# which in turn are characterized by :math:`D\times D` matrices :math:`B^k`.
# We can change the basis on the bond qudits to be measured by applying the unitary
# :math:`V=\sum_k |k\rangle\!\langle B^k|`, which leads to the state
#
# .. math::
#
#     |\Phi'\rangle = \frac{1}{D} \sum_k \sum_{i,j,\ell,p}
#     \sum_{\vec{m}} |\vec{m}\rangle  |k\rangle_{D^2} |ip\rangle_D
#     \langle i|A^{m_1} \cdots A^{m_K}B^kA^{m_{K+1}} \cdots A^{m_{K+L}}|p\rangle_D,
#
# where we used that :math:`\sum_{j,\ell} |j\rangle B^k_{j\ell}\langle\ell|=B^k`.
# We now can measure the two bond qudits and will know that for a given measurement
# outcome :math:`k`, we obtained the MPS state on :math:`K+L` qubits, except for
# a small "impurity", namely the matrix :math:`B^k` corresponding to the outcome.
#
# In this generality, the fusion strategy may seem complicated, but it can take
# a very simply form, as we will now see in our example:
#
# **Example**
#
# For our example we will use the Bell basis to perform the fusing mid-circuit
# measurement, i.e.,
#
# .. math::
#
#     |B^0\rangle &= \frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)\quad
#     |B^1\rangle = \frac{1}{\sqrt{2}}(|01\rangle+|10\rangle)\\
#     |B^2\rangle &= \frac{1}{\sqrt{2}}(|00\rangle-|11\rangle)\quad
#     |B^3\rangle = \frac{i}{\sqrt{2}}(|01\rangle-|10\rangle).
#
# This choice must seem arbitrary at this point, but will become clear
# later on. There, we will see that in general the matrices :math:`B^k` and the MPS
# tensor :math:`A` have to play well together.
# For the choice above, we find the matrices
#
# .. math::
#
#     B^0 = I \quad B^1 = X \quad B^2 = Z \quad B^3 = Y,
#
# i.e., the standard Pauli matrices.
# The measurement procedure can then be coded up in the following short function,
# which is similar to the ``project_measure`` function from the sequential preparation
# routine. Note, however, that instead of postselecting, we record the measurement
# outcome and will use it further below.


def fuse(wire_0, wire_1):
    """Measure two qubits in the Bell basis and return the outcome
    encoded in two bits."""
    qml.CNOT([wire_0, wire_1])
    qml.Hadamard(wire_0)
    return np.array([qml.measure(i) for i in [wire_0, wire_1]])


######################################################################
#
# Fusing two MPS that have been prepared by the sequential routine now is really simple:


def two_qubit_mps_by_fusion(g):
    sequential_preparation(g, [0, 1, 2])
    sequential_preparation(g, [3, 4, 5])
    mcms = fuse(2, 3)


######################################################################
# However, the produced state is not quite the MPS state on two qubits,
# because of the impurities caused through the fusion measurement. We can check this
# by undoing the preparation with the two qubit sequential circuit and measuring an
# exemplary expectation value.
# If the two sequentially prepared MPS and the fusion step did prepare the correct
# MPS already, the test measurement would just be :math:`\langle 00 | Z_0| 11\rangle=1`.


@qml.qnode(qml.device("default.qubit", wires=6))
def prepare_and_unprepare(g):
    two_qubit_mps_by_fusion(g)
    # The bond qubits for a sequential preparation are just 0 and 5
    # The bond qubits 2 and 3 have been measured out in the fusion protocol
    qml.adjoint(sequential_preparation)(g, [0, 1, 4, 5])
    return qml.expval(qml.PauliZ(1))


test = prepare_and_unprepare(g)
print(f"The test measurement of the fusion preparation + sequential unpreparation is {test:.2f}")

######################################################################
# This means we still need a way to remove the "impurity" matrices :math:`B^k` from the
# state that ended up in the prepared state when we performed the fusion measurement.
# Operator pushing will allow us to do just that!
#
# Operator pushing
# ~~~~~~~~~~~~~~~~
#
# The last building block we need is so-called operator pushing. In fact, whether or not an MPS
# can be prepared with the constant-depth circuit depends on whether certain operator pushing
# relations are satisfied between the matrices :math:`B^k` from the fusion step and the MPS
# tensor :math:`A`. We will not go into detail about these conditions but assume here that
# we work with compatible MPS and measurements.
#
# Operator pushing then allows us to push an impurity operator :math:`B^k` from one bond
# axis through the tensor :math:`A` of the MPS to the other bond axis and to the physical axis.
# While doing so, the operator in general will change, i.e., we end up with different operators
# :math:`C^k` and :math:`D^k` on the bond and physical axes, respectively.
# Visually, we find:
#
# .. image:: ../_static/demonstration_assets/preparation_mps_mcm/operator_pushing.png
#
# For simplicity, we denote a push by :math:`B^k\mapsto (C^k, D^k)`.
#
# **Example**
#
# In the fusion step above we picked the Bell basis as measurement basis, without
# further specifying why. As we saw, this basis leads to Pauli matrices as impurities
# :math:`B^k`. This choice is such that they satisfy pushing relations together with
# the tensor :math:`A` of our example MPS :math:`|\Psi(g)\rangle`.
# In particular, the relations are
#
# .. image:: ../_static/demonstration_assets/preparation_mps_mcm/operator_pushing.png
#
# That is, in the notation above we have
# :math:`X\mapsto (Y, Y)`,
# :math:`Y\mapsto (Y, X)`, and
# :math:`Z\mapsto (I, Z)` as well as the trivial
# :math:`I\mapsto (I, I)`.
# Note that the bond axis operator either is trivial (the identity), or a Pauli-:math:`Y` operator.
# These pushing relations allow us to push the bond axis operator through to an open boundary
# bond axis of the fused MPS, and to remove the physical axis operators by applying the correct
# Pauli operation to the physical site, conditioned on the fusion measurement outcomes.
# If we have an MPS on :math:`L` sites, pushing through a Pauli will have the following effects:
#
# - If it is a Pauli-:math:`Y`, all physical sites need to be corrected by a Pauli-:math:`X`, and
#   a Pauli-:math:`Y` is pushed to the opposite boundary.
#
# - If it is a Pauli-:math:`X`, a Pauli-:math:`Y` is applied to the first physical qubit, and afterwards
#   the case above is recovered, leading to applying Pauli-:math:`X` and pushing through a Pauli-:math:`Y`.
#
# - If it is a Pauli-:math:`Z`, a Pauli-:math:`Z` is applied to the first physical qubit and we are done.
#   No operator is pushed through.
#
# This pushing and correction step is captured by the following function, where we use some additional
# simplifications based on the used encoding with two bits of the pushed operator.


def push_and_correct(op_id, phys_wires):
    """Push an operator from left to right along the bond sites of an MPS and
    conditionally apply correcting operations on the corresponding physical sites.
    - The operator is given by two bits indicating the Pauli operator type.
    - The physical MPS sites are given by phys_wires
    """
    w = phys_wires[0]
    # Apply Y if input is X or Y
    qml.cond(op_id[1] & op_id[0], qml.X)(w)
    # Apply Z if input is Z or Y
    qml.cond(op_id[1] & ~op_id[0], qml.Y)(w)
    qml.cond(~op_id[1] & op_id[0], qml.Z)(w)
    # Apply X on other physical sites if input is X or Y
    for i in phys_wires[1:]:
        qml.cond(op_id[1], qml.X)(i)
    # Push through Y if input is X or Y
    return np.array([op_id[1], op_id[1]])


def correction(idx, q, op_id):
    w = (idx + 1) * q
    op_int = np.array([2, 1]) @ op_id
    # If measurement syndrome is X, apply Y and push Y's along, leading to consecutive X
    qml.cond(op_int == 1, qml.Y)(w)
    for i in range(w + 1, w + q):
        qml.cond(op_int == 1, qml.X)(i)
    # If measurement syndrome is Z, apply Z and terminate
    qml.cond(op_int == 2, qml.Z)(w)
    # If measurement syndrome is Y, apply X and push Y's along, leading to consecutive X
    for i in range(w, w + q):
        qml.cond(op_int == 3, qml.X)(i)
    return np.array([op_id[1], op_id[1]])


######################################################################
#
# Piecing everything together
# ---------------------------
#
# Now that we discussed the sequential preparation algorithm, fusion with mid-circuit measurements,
# and operator pushing, we have all components for the constant-depth preparation algorithm. It works
# as follows (recall that showing existence and finding suitable operator pushing relations is a
# crucial step in general, which goes beyond the scope of the demo).
#
# #. For a block size :math:`q`, prepare :math:`\frac{N}{q}` MPS of size :math:`q` in parallel, using
#    the sequential preparation algorithm (without the final projection step).
#
# #. Fuse together the blocks of MPS using mid-circuit measurements and store the measurement outcomes.
#
# #. Push the "impurity" operators resulting from the measurements to one of the outer bond sites and
#    conditionally apply correction operators to the physical sites.
#
# #. Undo the operator that was pushed to the outer bond site by conditionally applying its inverse.
#
# #. Perform the same projection step as in the sequential preparation algorithm on the two remaining
#    bond sites.
#
# Block size :math:`q`
# ~~~~~~~~~~~~~~~~~~~~
# The size of the blocks that have to be prepared in the first step depends on multiple considerations.
# First, the block must be such that operator pushing relations are available. Usually this will
# determine a minimal block size, and multiples of that size are valid choices as well.
# Second, the depth of the (parallelized) sequential preparation circuits is proportional to :math:`q`,
# determining a key contribution to the overall depth of the algorithm.
# Third, the sequential algorithm requires two bond qudits for each block, leading to
# :math:`\frac{2N}{q}` auxiliary qudits overall.
#
#
# **Example**
#
# For our example MPS :math:`|\Psi(g)\rangle`, we already defined and coded up the sequential preparation
# circuit for a flexible number of qubits, the measurement-based fusion, and the operator pushing
# and correction step. This makes putting the algorithm together quite simple. We only define an XOR
# for our operator IDs and a function that applies a correcting Pauli operator to the final bond
# site after pushing through the operator.
#


def xor(op_id_0, op_id_1):
    """Express logical XOR as "SUM - 2 * AND" on integers."""
    return op_id_0 + op_id_1 - 2 * op_id_0 * op_id_1


def correct_end_bond(bond_idx, op_id):
    """Perform a correction on the end bond site."""
    op_int = np.array([2, 1]) @ op_id
    qml.cond(op_int == 1, qml.X)(bond_idx)
    qml.cond(op_int == 2, qml.Z)(bond_idx)
    qml.cond(op_int == 3, qml.Y)(bond_idx)


def constant_depth(N, g, q):
    """Prepare the MPS |\Psi(g)> in constant depth."""
    num_blocks = N // q
    wires_per_block = q + 2  # 2 bond wires added

    # Step 1: Prepare small block MPS
    for i in range(num_blocks):
        wires = list(range(wires_per_block * i, wires_per_block * (i + 1)))
        sequential_preparation(g, wires)

    # Step 2: Fusion with mid-circuit measurements
    mcms = []
    for i in range(1, num_blocks):
        bond_wires = (wires_per_block * i - 1, wires_per_block * i)
        mcms.append(fuse(*bond_wires))

    # Step 3: Push operators through to highest-index wire and correct phys. sites
    pushed_op_id = np.array([0, 0])  # Start with identity
    for i in range(1, num_blocks):
        phys_wires = list(range(wires_per_block * i + 1, wires_per_block * (i + 1) - 1))
        pushed_op_id = push_and_correct(xor(mcms[i - 1], pushed_op_id), phys_wires)

    # Step 4: Undo the pushed-through operator on the highest-index wire bond site.
    correct_end_bond(num_blocks * wires_per_block - 1, pushed_op_id)


def constant_depth_ansatz(N, g, q):
    """Circuit ansatz for constant-depth preparation routine."""
    num_blocks = N // q
    wires_per_block = q + 2  # 2 bond wires added
    outer_bond_sites = [0, num_blocks * wires_per_block - 1]
    # Steps 1-4
    constant_depth(N, g, q)
    # Step 5: Perform projective measurement on outer-most bond sites.
    project_measure(*outer_bond_sites)
    phys_wires = sum(
        (
            list(range(wires_per_block * i + 1, wires_per_block * (i + 1) - 1))
            for i in range(num_blocks)
        ),
        start=[],
    )
    return phys_wires


@qml.qnode(dev)
def constant_depth_circuit(N, g, q):
    phys_wires = constant_depth_ansatz(N, g, q)
    return qml.probs(wires=phys_wires)


######################################################################
#
# Great, we know built the full constant-depth circuit to prepare the MPS
# :math:`|\Psi(g)\rangle`.
#
# Before we evaluate the states it produces, let's see how the circuit looks.
# We'll add some boxes for the individual subroutines.
#

N = 12
q = 3
g = -0.8
qml.draw_mpl(constant_depth_circuit)(N, g, q)
# plt.show()

######################################################################
# Let's check that the constant-depth circuit produces the same state, or probability
# distribution, as the sequential circuit.
#

p_seq = sequential_circuit(N, g)
p_const = constant_depth_circuit(N, g, q)
print(f"The two circuits agree on the probabilities for {g=:.1f}: {np.allclose(p_seq, p_const)}")


######################################################################
# Great, we now have a constant-depth circuit that reproduces the existing sequential
# preparation circuit, which has linear depth!
#
# Correlation length
# ~~~~~~~~~~~~~~~~~~
#
# When introducing the example MPS :math:`|\Psi(g)\rangle`, we mentioned that the
# parameter :math:`g` controls the correlation length of the state. As extreme examples,
# we can produce a zero-correlation length state for :math:`g=-1` and a long-range
# correlated state, namely the GHZ state, for :math:`g\to 0`.
#

N = 6
g = -1
uniform_probs = constant_depth_circuit(N, g, q)
g = -1e-8
ghz_probs = constant_depth_circuit(N, g, q)
ints = list(range(2**N))
fig, ax = plt.subplots()
ax.bar(ints, uniform_probs, width=1, label="zero correlation length ($|+\\rangle^N$)")
ax.bar(ints, ghz_probs, width=1, label="long correlation length (GHZ)")
ax.set_yscale("log")
ax.set_ylim([1e-3, 1])
ax.legend()
# plt.show()

######################################################################
# In a more nuanced setting, let us produce the MPS in question for a series of values
# for :math:`g`, and measure the correlation length ourselves.
# For this, we measure the two-body correlators
#
# .. math::
#
#     C(j) = \langle Z_0 Z_j\rangle - \langle Z_0\rangle \langle Z_1\rangle
#
# for an increasing distance :math:`j`.


@qml.qnode(dev, interface="numpy")
def pre_correlations(N, g, q):
    phys_wires = constant_depth_ansatz(N, g, q)
    return [qml.expval(qml.Z(1) @ qml.Z(i)) for i in phys_wires] + [
        qml.expval(qml.Z(i)) for i in phys_wires
    ]


def correlations(N, g, q):
    expvals = pre_correlations(N, g, q)
    prods = expvals[:-N]
    expvals = expvals[-N:]
    correls = [prods[i] - expvals[1] * expvals[i] for i in range(N) if i != 1]
    return correls


N = 18
q = 6
gs = [-1 / 5, -2 / 5, -3 / 5, -4 / 5]
fig, ax = plt.subplots()
all_correls = []
for g in gs:
    correls = correlations(N, g, q)
    ax.plot(list(range(N - 1)), correls, label=f"$g={g}$")
    all_correls.append(correls)
ax.set_yscale("log")
ax.set_ylim([5e-8, 1.5])
ax.legend()
plt.show()

######################################################################
# Note that due to the periodic boundary conditions, we have :math:`C(N-j)=C(j)`.
# This is why the correlations increase again after :math:`j=\frac{N}{2}`.
# To obtain the correlation length, we fit an exponential to the measured correlators
# up until :math:`j=\frac{N/2}` and invert the decay rate.
# We can also compare this result to the analytic result for the correlation
# length :math:`xi` mentioned in the beginning:


def exp_fn(x, a, b):
    return a * np.exp(x * b)


print(f"Correlation lengths xi\n===================")
for g, correls in zip(gs, all_correls):
    popt, pcov = curve_fit(exp_fn, list(range(1, N // 2 + 1)), correls[1 : N // 2 + 1])
    xi_numerical = -1 / popt[1]
    xi_predicted = 1 / abs(np.log((1 + g) / (1 - g)))
    print(f"{g=:.1f}")
    print(f"Predicted: {xi_predicted:.5f}")
    print(f"Numerical: {xi_numerical:.5f}")
    print()

######################################################################
# As we can see, the produced MPS states indeed have the predicted correlation lengths,
# up to some numerical imprecisions.
#
# Conclusion
# ----------
#
######################################################################
#
#
# References
# ----------
# .. [#smith]
#
#     Kevin C. Smith, Abid Khan, Bryan K. Clark, S.M. Girvin, Tzu-Chieh Wei
#     "Constant-depth preparation of matrix product states with adaptive quantum circuits",
#     `arXiv:2404.16083 <http://www.arxiv.org/abs/2404.16083>`__, 2024.
#
# .. [#schoen]
#
#     C. Schön, E. Solano, F. Verstraete, J. I. Cirac, M. M. Wolf
#     "Sequential Generation of Entangled Multiqubit States", Physical Review Letters, **95**, 110503,
#     `closed access <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.110503>`__, 2005.
#     `arXiv:quant-ph/0501096 <http://www.arxiv.org/abs/quant-ph/0501096 >`__, 2005.
#
# .. [#todo]
#
#     Todo
#
# About the author
# ----------------
#
