r"""
Measurement optimization
========================

.. meta::
    :property="og:description": Optimize and reduce the number of measurements required to evaluate a variational algorithm cost function.
    :property="og:image": https://pennylane.ai/qml/_images/grouping.png

.. related::

   tutorial_vqe Variational quantum eigensolver
   tutorial_quantum_chemistry Quantum chemistry with PennyLane
   tutorial_qaoa_intro Intro to QAOA

The variational quantum eigensolver (VQE) is the OG variational quantum algorithm. Harnessing
near-term quantum hardware to solve for the electronic structure of molecules, VQE is *the*
algorithm that sparked the variational circuit craze of the last 5 years, and holds the greatest
promise for showcasing a quantum advantage on near-term quantum hardware. It has also inspired
other quantum algorithms such as the :doc:`Quantum Approximate Optimization Algorithm (QAOA)
</demos/tutorial_qaoa_intro>`.

To scale VQE beyond the regime of classical computation, however, we need to use it to solve for the
ground state of excessively larger and larger molecules. A side effect is that the number of
measurements we need to make on the quantum hardware also grows polynomially---a huge bottleneck,
especially when quantum hardware access is limited and expensive.

To mitigate this 'measurement problem', a plethora of recent research dropped over the course of
2019 and 2020 [#yen2020]_ [#verteletskyi2020]_ [#izmaylov2019]_ [#gokhale2020]_, exploring potential
strategies to minimize the number of measurements required. In fact, by grouping qubit-wise
commuting terms of the Hamiltonian, we can significantly reduce the number of measurements
needed---in some cases, reducing the number of measurements by up to 90%(!).

.. figure:: /demonstrations/measurement_optimize/grouping.png
    :width: 90%
    :align: center

In this demonstration, we revisit VQE, see first-hand how the required number of measurements scales
as molecule size increases, and finally use these measurement optimization strategies
to minimize the number of measurements we need to make.

Revisiting VQE
--------------

The study of :doc:`variational quantum algorithms </glossary/variational_circuit>` was spearheaded
by the introduction of the :doc:`variational quantum eigensolver <tutorial_vqe>` (VQE) algorithm in
2014 [#peruzzo2014]_. While classical variational techniques have been known for decades to estimate
the ground state energy of a molecule, VQE allowed this variational technique to be applied using
quantum computers. Since then, the field of variational quantum algorithms has evolved
significantly, with larger and more complex models being proposed (such as
:doc:`quantum neural networks </demos/quantum_neural_net>`, :doc:`QGANs </demos/tutorial_QGAN>`, and
:doc:`variational classifiers </demos/tutorial_variational_classifier>`). However, quantum chemistry
remains one of the flagship uses-cases for variational quantum algorithms, and VQE the standard-bearer.

The appeal of VQE lies almost within its simplicity. A circuit ansatz :math:`U(\theta)` is chosen
(typically the Unitary Coupled-Cluster Singles and Doubles
(:func:`~pennylane.templates.subroutines.UCCSD`) ansatz), and the qubit representation of the
molecular Hamiltonian is computed:

.. math:: H = \sum_i c_i h_i,

where :math:`h_i` are the terms of the Hamiltonian written as a product of Pauli operators :math:`\sigma_n`:

.. math:: h_i = \prod_{n=0}^{N} \sigma_n.

The cost function of VQE is then simply the expectation value of this Hamiltonian after
the variational quantum circuit:

.. math:: \text{cost}(\theta) = \langle 0 | U(\theta)^\dagger H U(\theta) | 0 \rangle.

By using a classical optimizer to *minimize* this quantity, we will be able to estimate
the ground state energy of the Hamiltonian :math:`H`:

.. math:: H U(\theta_{min}) |0\rangle = E_{min} U(\theta_{min}) |0\rangle.

In practice, when we are using quantum hardware to compute these expectation values we expand out
the summation, resulting in separate expectation values that need to be calculated for each term in
the Hamiltonian:

.. math::

    \text{cost}(\theta) = \langle 0 | U(\theta)^\dagger \left(\sum_i c_i h_i\right) U(\theta) | 0 \rangle
                        = \sum_i c_i \langle 0 | U(\theta)^\dagger h_i U(\theta) | 0 \rangle.

.. note::

    How do we compute the qubit representation of the molecular Hamiltonian? This is a more
    complicated story, that involves applying a self-consistent field method (such as Hartree-Fock),
    and then performing a fermionic-to-qubit mapping such as the Jordan-Wigner or Bravyi-Kitaev
    transformations.

    For more details on this process, check out the :doc:`/demos/tutorial_quantum_chemistry`
    tutorial.

The measurement problem
-----------------------

For small molecules, VQE scales and performs exceedingly well. For example, for the
Hydrogen molecule :math:`\text{H}_2`, the final qubit-representation Hamiltonian
has 15 terms that need to be measured. Lets generate this Hamiltonian from the electronic
structure file :download:`h2.xyz </demonstrations/h2.xyz>`, using PennyLane
QChem to verify the number of terms.
"""

import functools
from pennylane import numpy as np
import pennylane as qml

qml.enable_tape()
np.random.seed(42)

H, num_qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")

print("Required number of qubits:", num_qubits)
print(H)

###############################################################################
# Here, we can see that the Hamiltonian involves 15 terms, so we expect to compute 15 expectation values
# on hardware. Let's generate the cost function to check this.

# Create a 4 qubit simulator
dev = qml.device("default.qubit", wires=num_qubits)

# number of electrons
electrons = 2

# Define the Hartree-Fock initial state for our variational circuit
initial_state = qml.qchem.hf_state(electrons, num_qubits)

# Construct the UCCSD ansatz
singles, doubles = qml.qchem.excitations(electrons, num_qubits)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
ansatz = functools.partial(
    qml.templates.UCCSD, init_state=initial_state, s_wires=s_wires, d_wires=d_wires
)

# generate the cost function
cost = qml.ExpvalCost(ansatz, H, dev)

##############################################################################
# If we evaluate this cost function, we can see that it corresponds to 15 different
# QNodes under the hood---one per expectation value:

params = np.random.normal(0, np.pi, len(singles) + len(doubles))
print("Cost function value:", cost(params))
print("Number of quantum evaluations:", dev.num_executions)

##############################################################################
# How about a larger molecule? Lets try the water molecule :download:`h2o.xyz </demonstrations/h2o.xyz>`:

H, num_qubits = qml.qchem.molecular_hamiltonian("h2o", "h2o.xyz")

print("Required number of qubits:", num_qubits)
print("Number of Hamiltonian terms/required measurements:", len(H.ops))

print("\n", H)


##############################################################################
# Simply going from two atoms in :math:`\text{H}_2` to three in :math:`\text{H}_2 \text{O}`
# resulted in 2050 measurements that must be made!
#
# We can see that as the size of our molecule increases, we run into a problem; larger molecules
# result in Hamiltonians that not only require a larger number of qubits :math:`N` in their
# representation, but the number of terms in the Hamiltonian scales like
# :math:`\mathcal{O}(N^4)`! 😱😱😱
#
# We can mitigate this somewhat by choosing smaller `basis sets
# <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`__ to represent the electronic structure
# wavefunction, however this comes with an accuracy cost, and doesn't reduce the number of
# measurements significantly enough to allow us to scale to classically intractable problems.
#
# .. figure:: /demonstrations/measurement_optimize/n4.png
#     :width: 70%
#     :align: center
#
#     The number of qubit Hamiltonian terms required to represent various molecules in the specified
#     basis sets (adapted from `Minimizing State Preparations for VQE by Gokhale et al.
#     <https://pranavgokhale.com/static/Minimizing_State_Preparations_for_VQE.pdf>`__)


##############################################################################
# Simultaneously measuring observables
# ------------------------------------
#
# One of the assumptions we made above was that every term in the Hamiltonian must be measured independently.
# However, this might not be the case. From the `Heisenburg uncertainty relationship
# <https://en.wikipedia.org/wiki/Uncertainty_principle>`__ for two
# observables :math:`\hat{A}` and :math:`\hat{B}`, we know that
#
# .. math:: \sigma_A^2 \sigma_B^2 \geq \frac{1}{2}\left|\left\langle [\hat{A}, \hat{B}] \right\rangle\right|,
#
# where :math:`\sigma^2` the variance of measuring the expectation value of an
# observable, and
#
# .. math:: [\hat{A}, \hat{B}] = \hat{A}\hat{B}-\hat{B}\hat{A}
#
# is the commutator. Therefore,
#
# - If the two observables :math:`\hat{A}` and :math:`\hat{B}` do not commute (:math:`[\hat{A},
#   \hat{B}] \neq 0`), then :math:`\sigma_A^2
#   \sigma_B^2 > 0` and we cannot simultaneously measure the expectation values of the two
#   observables.
#
# ..
#
# - If :math:`\hat{A}` and :math:`\hat{B}` **do** commute (:math:`[\hat{A},
#   \hat{B}] = 0`), then :math:`\sigma_A^2
#   \sigma_B^2 \geq 0` and there exists a measurement basis where we can **simultaneously measure** the
#   expectation value of both observables on the same state.
#
# .. admonition:: Aside: commutativity and shared eigenbases
#     :class: aside
#
#     To explore why commutativity and simultaneous measurement are related, lets assume that there
#     is a a complete, orthonormal eigenbasis :math:`|\phi_n\rangle` that *simultaneously
#     diagonalizes* both :math:`\hat{A}` and :math:`\hat{B}`:
#
#     .. math::
#
#         ① ~~ \hat{A} |\phi_n\rangle &= \lambda_{A,n} |\phi_n\rangle,\\
#         ② ~~ \hat{B} |\phi_n\rangle &= \lambda_{B,n} |\phi_n\rangle.
#
#     where :math:`\lambda_{A,n}` and :math:`\lambda_{B,n}` are the corresponding eigenvalues.
#     If we pre-multiply the first equation by :math:`\hat{B}`, and the second by :math:`\hat{A}`
#     (denoted in blue):
#
#     .. math::
#
#         \color{blue}{\hat{B}}\hat{A} |\phi_n\rangle &= \lambda_{A,n} \color{blue}{\hat{B}}
#           |\phi_n\rangle = \lambda_{A,n} \color{blue}{\lambda_{B,n}} |\phi_n\rangle,\\
#         \color{blue}{\hat{A}}\hat{B} |\phi_n\rangle &= \lambda_{B,n} \color{blue}{\hat{A}}
#           |\phi_n\rangle = \lambda_{A,n} \color{blue}{\lambda_{B,n}} |\phi_n\rangle.
#
#     We can see that assuming a simultaneous eigenbasis requires that
#     :math:`\hat{A}\hat{B}|\phi_n\rangle = \hat{B}\hat{A}|\phi_n\rangle`. Or, rearranging,
#
#     .. math:: (\hat{A}\hat{B} - \hat{B}\hat{A}) |\phi_n\rangle = [\hat{A}, \hat{B}]|\phi_n\rangle = 0.
#
#     Our assumption that :math:`|\phi_n\rangle` simultaneously diagonalizes both :math:`\hat{A}` and
#     :math:`\hat{B}` only holds true if the two observables commute.
#
# So far, this seems awfully theoretical. What does this mean in practice?
#
# In the realm of variational circuits, we typically want to compute expectation values of an
# observable on a given state :math:`|\psi\rangle`. If we have two commuting observables, we also know that
# they share a simultaneous eigenbasis:
#
# .. math::
#
#     \hat{A} &= \sum_n \lambda_{A, n} |\phi_n\rangle\langle \phi_n|\\
#     \hat{B} &= \sum_n \lambda_{B, n} |\phi_n\rangle\langle \phi_n|.
#
# Substituting this into the expression for the expectation values:
#
# .. math::
#
#     \langle\hat{A}\rangle &= \langle \psi | \hat{A} | \psi \rangle = \langle \psi | \left( \sum_n
#         \lambda_{A, n} |\phi_n\rangle\langle \phi_n| \right) | \psi \rangle = \sum_n \lambda_{A,n}
#         |\langle \phi_n|\psi\rangle|^2,\\
#     \langle\hat{B}\rangle &= \langle \psi | \hat{B} | \psi \rangle = \langle \psi | \left( \sum_n
#         \lambda_{B, n} |\phi_n\rangle\langle \phi_n| \right) | \psi \rangle = \sum_n \lambda_{B,n}
#         |\langle \phi_n|\psi\rangle|^2.
#
# So, assuming we know the eigenvalues of the commuting observables in advance, if we perform a
# measurement in their shared eigenbasis, we only need to perform a **single measurement** of the
# probabilities :math:`|\langle \phi_n|\psi\rangle|^2` in order to recover both expectation values! 😍
#
# Fantastic! But, can we use this to reduce the number of measurements we need to perform in VQE?
# To do so, we need to be able to answer two simple sounding questions:
#
# 1. How do we determine which terms of the cost Hamiltonian are commuting?
#
# 2. How do we rotate the circuit into the shared eigenbasis prior to measurement?
#
# The answers to these questions aren't necessarily easy nor straightforward. Thankfully, there are
# some recent techniques we can harness to address both.

##############################################################################
# Qubit-wise commuting Pauli terms
# --------------------------------
#
# Back when we summarized VQE, we saw that each term of the Hamiltonian is generally represented
# as a tensor product of Pauli terms:
#
# .. math:: h_i = \prod_{n=0}^{N} \sigma_n.
#
# Luckily, this allows us to take a bit of a shortcut. Rather than consider **full commutativity**,
# we can consider a subset known as **qubit-wise commutativity** (QWC).
#
# To start with, let's consider single Pauli operators. We know that the Pauli operators
# commute with themselves as well as the identity, but they do *not* commute with
# each other:
#
# .. math::
#
#     [\sigma_i, I] = 0, ~~~ [\sigma_i, \sigma_i] = 0, ~~~ [\sigma_i, \sigma_j] = c \sigma_k \delta_{ij}.
#
# Now consider two tensor products of Pauli terms, for example :math:`X\otimes Y \otimes I` and
# :math:`X\otimes I \otimes Z`. We say that these two terms are qubit-wise commuting, since, if
# we compare each subsystem in the tensor product, we see that every one commutes:
#
# .. math::
#
#     \begin{array}{ | *1c | *1c | *1c | *1c | *1c |}
#       X &\otimes &Y &\otimes &I\\
#       X &\otimes &I &\otimes &Z
#     \end{array} ~~~~ \Rightarrow ~~~~ [X, X] = 0, ~~~ [Y, I] = 0, ~~~ [I, Z] = 0.
#
# As a consequence, both terms must commute:
#
# .. math:: [X\otimes Y \otimes I, X\otimes I \otimes Z] = 0.
#
# .. important::
#
#     Qubit-wise commutativity is a **sufficient** but not **necessary** condition
#     for full commutativity. For example, the two Pauli terms :math:`Y\otimes Y` and
#     :math:`X\otimes X` are not qubit-wise commuting, but do commute (have a go verifying this!).
#
# Once we have identified a qubit-wise commuting pair of Pauli terms, it is also straightforward to
# find the gates to rotate the circuit into the shared eigenbasis. To do so, we simply rotate
# each wire one-by-one depending on the Pauli operator we are measuring on that wire:
#
# .. raw:: html
#
#     <style>
#         .docstable {
#             max-width: 300px;
#         }
#         .docstable tr.row-even th, .docstable tr.row-even td {
#             text-align: center;
#         }
#         .docstable tr.row-odd th, .docstable tr.row-odd td {
#             text-align: center;
#         }
#     </style>
#     <div class="d-flex justify-content-center">
#
# .. rst-class:: docstable
#
#     +------------------+-----------------------+
#     |    Observable    | Rotation gate         |
#     +==================+=======================+
#     | :math:`X`        | :math:`H`             |
#     +------------------+-----------------------+
#     | :math:`Y`        | :math:`H S^{-1}=HSZ`  |
#     +------------------+-----------------------+
#     | :math:`Z`        | :math:`I`             |
#     +------------------+-----------------------+
#     | :math:`I`        | :math:`I`             |
#     +------------------+-----------------------+
#
# .. raw:: html
#
#     </div>
#
# Therefore, in this particular example:
#
# * Wire 0: we are measuring both terms in the :math:`X` basis, apply the Hadamard gate
# * Wire 1: we are measuring both terms in the :math:`Y` basis, apply the :math:`H S^{-1}` gates
# * Wire 2: we are measuring both terms in the :math:`Z` basis, no gate needs to be applied.
#
# Let's use PennyLane to verify this.


obs = [
    qml.PauliX(0) @ qml.PauliY(1),
    qml.PauliX(0) @ qml.PauliZ(2)
]


##############################################################################
# First, lets naively use two separate circuit evaluations to measure
# the two QWC terms.


dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit1(weights):
    qml.templates.StronglyEntanglingLayers(weights, wires=range(3))
    return qml.expval(obs[0])


@qml.qnode(dev)
def circuit2(weights):
    qml.templates.StronglyEntanglingLayers(weights, wires=range(3))
    return qml.expval(obs[1])


weights = qml.init.strong_ent_layers_normal(n_layers=3, n_wires=3)

print("Expectation value of XYI = ", circuit1(weights))
print("Expectation value of XIZ = ", circuit2(weights))

##############################################################################
# Now, lets use our QWC approach to reduce this down to a *single* measurement
# of the probabilities in the shared eigenbasis of both QWC observables:

@qml.qnode(dev)
def circuit_qwc(weights):
    qml.templates.StronglyEntanglingLayers(weights, wires=range(3))

    # rotate wire 0 into the shared eigenbasis
    qml.Hadamard(wires=0)

    # rotate wire 1 into the shared eigenbasis
    qml.S(wires=1).inv()
    qml.Hadamard(wires=1)

    # wire 2 does not require a rotation

    # measure probabilities in the computational basis
    return qml.probs(wires=range(3))


rotated_probs = circuit_qwc(weights)
print(rotated_probs)


##############################################################################
# We're not quite there yet; we have only calculated the probabilities of the variational circuit
# rotated into the shared eigenbasis; :math:`|\langle \phi_n |\psi\rangle|^2`. To recover the
# *expectation values* of the two QWC observables from the probabilities, recall that we need one
# final piece of information; their eigenvalues :math:`\lambda_{A, n}` and :math:`\lambda_{B, n}`.
#
# We know that the Pauli operators have eigenvalues :math:`(1, -1)`, while the identity
# operator has eigenvalues :math:`(1, 1)`; we can make use of ``np.kron`` to quickly
# generate the probabilities of the full Pauli terms.

eigenvalues_XYI = np.kron(np.kron([1, -1], [1, -1]), [1, 1])
eigenvalues_XIZ = np.kron(np.kron([1, -1], [1, 1]), [1, -1])

# Taking the linear combination of the eigenvalues and the probabilities
print("Expectation value of XYI = ", np.dot(eigenvalues_XYI, rotated_probs))
print("Expectation value of XIZ = ", np.dot(eigenvalues_XIZ, rotated_probs))


##############################################################################
# Compare this to the result when we used two circuit evaluations. We have successfully used a
# single circuit evaluation to recover both expectation values!
#
# Luckily, PennyLane automatically performs this QWC under-the-hood. We simply
# return the two QWC Pauli terms from the QNode:

@qml.qnode(dev)
def circuit(weights):
    qml.templates.StronglyEntanglingLayers(weights, wires=range(3))
    return [
        qml.expval(qml.PauliX(0) @ qml.PauliY(1)),
        qml.expval(qml.PauliX(0) @ qml.PauliZ(2))
    ]


print(circuit(weights))


##############################################################################
# Behind the scenes, PennyLane is making use of our built-in
# :mod:`qml.grouping <pennylane.grouping>` module, which contains functions for diagonalizing QWC
# terms:

rotations, new_obs = qml.grouping.diagonalize_qwc_pauli_words(obs)

print(rotations)
print(new_obs)


##############################################################################
# Check out the :mod:`qml.grouping <pennylane.grouping>` documentation for more details on its
# provided functionality and how it works.
#
# What happens, though, if we (in a moment of reckless abandonment!) ask a QNode to simultaneously
# measure two observables that *aren't* qubit-wise commuting? For example, lets consider
# :math:`X\otimes Y` and :math:`Z\otimes Z`:
#
# .. code-block:: python
#
#     @qml.qnode(dev)
#     def circuit(weights):
#         qml.templates.StronglyEntanglingLayers(weights, wires=range(3))
#         return [
#             qml.expval(qml.PauliZ(0) @ qml.PauliY(1)),
#             qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
#         ]
#
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     pennylane.qnodes.base.QuantumFunctionError: Only observables that are qubit-wise commuting
#     Pauli words can be returned on the same wire
#
# The QNode has detected that the two observables are not qubit-wise commuting, and
# has raised an error.
#
# So, a strategy begins to take shape: given a Hamiltonian containing a large number of Pauli terms,
# can we somehow partition the terms into **fewest** number of QWC groups, to minimize the measurements
# we need to take?

##############################################################################
# Grouping QWC terms
# ------------------
#
# Say we have the following Hamiltonian defined over four qubits:
#
# .. math:: H = Z_0 + Z_0 Z_1 + Z_0 Z_1 Z_2 + Z_0 Z_1 Z_2 Z_3 + X_2 X_3 + Y_0 X_2 X_3 + Y_0 Y_1 X_2 X_3,
#
# where we are using the shorthand :math:`P_0 P_2 = P\otimes I \otimes P \otimes I` for brevity.
# If we go through and work out which Pauli terms are qubit-wise commuting, we can represent
# this in a neat way using a graph:
#
# .. figure:: /demonstrations/measurement_optimize/graph1.png
#     :width: 70%
#     :align: center
#
# In the above graph, every node represents an individual Pauli term of the Hamiltonian, with
# edges connecting terms that are qubit-wise commuting. Groups of qubit-wise commuting terms are
# represented as **complete subgraphs**. Straight away, we can make an observation:
# there is no unique solution for partitioning the Hamiltonian into groups of qubit-wise commuting
# terms! In fact, there are several solutions:
#
# .. figure:: /demonstrations/measurement_optimize/graph2.png
#     :width: 90%
#     :align: center
#
# Of course, of the potential solutions above, there is one that is more optimal than the others;
# on the bottom left, we have partitioned the graph into *two* complete subgraphs, as opposed to the
# other solutions that require three complete subgraphs. If we were to go with this solution,
# we would be able to measure the expectation value of the Hamiltonian using two circuit evaluations.
#
# This problem---finding the minimum number of complete subgraphs of a graph---is actually quite well
# known in graph theory, where it is referred to as the `minimum clique cover problem
# <https://en.wikipedia.org/wiki/Clique_cover>`__ (with 'clique' being another term for a complete subgraph).
#
# Unfortunately, that's where our good fortune ends---the minimum clique cover is known to
# be `NP-hard <https://en.wikipedia.org/wiki/NP-hardness>`__, meaning there is no known (classical)
# solution to finding the optimum/minimum clique cover in polynomial time.
#
# Thankfully, there is a silver lining---we know of polynomial-time algorithms for finding
# *approximate* solutions to the minimum clique cover problem. These heuristic approaches, while
# not guaranteed to find the optimum solution, scale quadratically with the number of nodes in the
# graph/terms in the Hamiltonian [#yen2020]_, so work reasonably well in practice.
#
# .. figure:: /demonstrations/measurement_optimize/graph3.png
#     :width: 100%
#     :align: center
#

##############################################################################
# Beyond VQE
# ----------

##############################################################################
# References
# ----------
#
# .. [#peruzzo2014]
#
#     Alberto Peruzzo, Jarrod McClean *et al.*, "A variational eigenvalue solver on a photonic
#     quantum processor". `Nature Communications 5, 4213 (2014).
#     <https://www.nature.com/articles/ncomms5213?origin=ppub>`__
#
# .. [#yen2020]
#
#     Tzu-Ching Yen, Vladyslav Verteletskyi, and Artur F. Izmaylov. "Measuring all compatible
#     operators in one series of single-qubit measurements using unitary transformations." `Journal of
#     Chemical Theory and Computation 16.4 (2020): 2400-2409.
#     <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.0c00008>`__
#
# .. [#verteletskyi2020]
#
#     Vladyslav Verteletskyi, Tzu-Ching Yen, and Artur F. Izmaylov. "Measurement optimization in the
#     variational quantum eigensolver using a minimum clique cover." `The Journal of Chemical Physics
#     152.12 (2020): 124114. <https://aip.scitation.org/doi/10.1063/1.5141458>`__
#
# .. [#izmaylov2019]
#
#    Artur F. Izmaylov, et al. "Unitary partitioning approach to the measurement problem in the
#    variational quantum eigensolver method." `Journal of Chemical Theory and Computation 16.1 (2019):
#    190-195. <https://pubs.acs.org/doi/abs/10.1021/acs.jctc.9b00791>`__
#
# .. [#gokhale2020]
#
#    Pranav Gokhale, et al. "Minimizing state preparations in variational quantum eigensolver by
#    partitioning into commuting families." `arXiv preprint arXiv:1907.13623 (2019).
#    <https://arxiv.org/abs/1907.13623>`__
