r"""Shadow Hamiltonian Simulation
=================================

We provide a beginner friendly introduction to the new kid on the block: shadow Hamiltonian simulation [#SommaShadow]_.

Introduction
------------

Shadow Hamiltonian simulation is a new approach to quantum simulation on quantum computers [#SommaShadow]_.
Despite its name, it has little to do with :doc:`classical shadows </demos/tutorial_diffable_shadows>`.
In quantum simulation, the goal is typically to simulate the expectation values of observables 
:math:`O_m` for :math:`m=1,..,M` under some unitary evolution.
The common approach is to evolve the wave function :math:`|\psi\rangle` and then measure the desired observables after the evolution.

In shadow Hamiltonian simulation, we instead directly encode the expectation values in a proxy state, the shadow state,
and evolve that state accordingly. Specifically for time evolution, we can write a shadow Schrödinger equation that governs the
dynamics of the shadow state.

This is fundamentally different to the common approach as the dimensionality of the 
shadow system no longer depends on the number of constituents :math:`n` of the system,
but on the number of observables :math:`M` that we are interested in and require, 
as there are conditions of completeness for the shadow encoding to succeed (called invariance property in the original paper).
Further, since the
expectation values are encoded in the amplitudes of states, we cannot directly measure them anymore, but need to resort to some 
form of state tomography.
On the other hand, this gives us entirely new possibilities by letting us sample from :math:`p_m = |\langle O_m \rangle|^2`,
and, in particular simultaneously all absolute values of observables.

In this demo we are going to introduce the basic concepts of shadow Hamiltonian simulation alongside some easy-to-follow code snippets.
We will also later see how shadow Hamiltonian simulation comes down to :doc:`demo on g-sim </demos/tutorial_liesim>`, 
a Lie algebraic classical simulation tool, but run on a quantum computer and with some simplifications due to considering specifically Hamiltonian simulation.

Shadow Hamiltonian simulation
-----------------------------

In common quantum Hamiltonian simulation, we evolve a state vector :math:`|\psi(t)\rangle` according to the Schrödinger equation

.. math:: \frac{\text{d}}{\text{dt}} |\psi(t)\rangle = -i H |\psi(t)\rangle

by some Hamiltonian :math:`H`, and then compute expectation values of the evolved state through measurement.
In Shadow Hamiltonian simulation, we encode a set of expectation values in the amplitudes of a quantum state,
and evolve those according to some shadow Schrödinger equation.

For that, we first need to define the shadow state

.. math:: |\rho\rangle = \frac{1}{\sqrt{A}} \begin{pmatrix} \langle O_1 \rangle \\ \vdots \\ \langle O_M \rangle \end{pmatrix}

for a set of operators :math:`S = \{O_m\}` with normalization constant :math:`A = \sum_m |\langle O_m \rangle|^2`.
So we can encode those :math:`M` operator expectation values with :math:`n_S` qubits such that :math:`2^{n_S} \geq M`.

The shadow state evolves according to its shadow Schrödinger equation

.. math:: \frac{\text{d}}{\text{dt}} |\rho\rangle = - i H_S |\rho\rangle.

The Hamiltonian matrix :math:`H_S` is given by the commutation relations between the system Hamiltonian :math:`H` and
the operators in :math:`S = \{O_m\}`. In particular, it is implicitly defined by

.. math:: [H, O_m] = - \sum_{m'=1}^M \left( H_S \right)_{m m'} O_{m'}.

A vector :math:`\boldsymbol{v}` can always be decomposed in an orthogonal basis :math:`\boldsymbol{e}_j` via
:math:`\boldsymbol{v} = \sum_j \frac{\langle \boldsymbol{e}_j, \boldsymbol{v}\rangle}{||\boldsymbol{e}_j||^2} \boldsymbol{e}_j`.
Since the operators under consideration are elements of the vector space of Hermitian operators, we can use that to compute :math:`H_S`.

In particular, with the trace inner product this amounts to

.. math:: [H, O_m] = \sum_{m'=1}^M \frac{\text{tr}\left( O_{m'} [H, O_m] \right)}{|| O_{m'} ||^2} O_{m'},

from which we can read off the matrix elements of :math:`H_S`, i.e.

.. math:: (H_S)_{m m'} = -\frac{\text{tr}\left( O_{m'} [H, O_m] \right)}{|| O_{m'} ||^2}.

How this related to g-sim
~~~~~~~~~~~~~~~~~~~~~~~~~

In :doc:`demo on g-sim </demos/tutorial_liesim>`
[#Somma]_ [#Somma2]_ [#Galitski]_ [#Goh]_, we have operators :math:`\{ g_i \}` that are potential generators or observables for a parametrized quantum circuit,
e.g. :math:`U(\theta) = \prod_\ell \exp(-i \thata_\ell g_\ell)`.
For that, we are looking at the so-called dynamical Lie algebra (DLA)
:math:`\mathfrak{g} = \langle \{ g_i \} \rangle_\text{Lie} = \{ g_1, .., g_{|\mathfrak{g}|} \}` of the circuit as well as
the adjoint representation
:math:`(-i ad_{g_\gamma})_{\alpha \beta} = f^\gamma_{\alpha \beta}`, where :math:`f^\gamma_{\alpha \beta}` are the structure
constants of the dynamical Lie algebra :math:`\mathfrak{g}`.
In g-sim, we also evolve expectation vectors :math:`(\vec{g})_i = \langle g_i \rangle`.
In particular, the circuit of evolving a state according to :math:`U(\theta)` and computing expectation values 
:math:`\langle g_i \rangle` then corresponds then corresponds to evolving :math:`\vec{g}` by :math:`\prod_\ell \exp(-i \thata_\ell \text{ad}_{g_\ell})`.

Shadow Hamiltonian simulation can then be viewed as g-sim
but run on a quantum computer with a single, specific gate generated by :math:`H`.

In particular, :math:`H_S` corresponds to the adjoint representation :math:`\text{ad}_H` of the 
:doc:`dynamical Lie algebra </demos/tutorial_liealgebra>`
:math:`\langle \{ H \} \cup S \rangle_\text{Lie}`. We explain the concept of the adjoint representation in our
:doc:`demo on g-sim </demos/tutorial_liesim>` that makes extensive use of it.

One striking difference is that because
we only have one specific "gate", we do not need the full Lie closure of the operators that we want to compute the expectation values of.
Instead, here it is sufficient to choose :math:`O_m` such that they support all :math:`[H, O_m]`.
This is a significant difference, as the Lie closure in most cases leads to an exponentially large DLA [#Wiersema]_ [#Aguilar]_.

A simple example
----------------

The abstract concepts of shadow Hamiltonian simulation are best illustrated with a simple and concrete example.
We are interested in simulating the Hamiltonian evolution of 

.. math:: H = X + Y

after :math:`t = 1` and compute the expectation values of :math:`\{X, Y, Z, I \}`.
In the standard formulation we simply evolve the initial quantum state :math:`|\psi(0)\rangle = |0\rangle` by :math:`H` in the
following way.

"""

import pennylane as qml
import numpy as np
from pennylane import X, Y, Z, I

dev = qml.device("default.qubit")

n = 1
S = [X(0), Y(0), Z(0), I(0)]
H = X(0) + Y(0)

@qml.qnode(dev)
def evolve(H, t):
    qml.evolve(H, t)
    return [qml.expval(Om) for Om in S]

t = 1.
O_t_standard = np.array(evolve(H, t))
O_t_standard

##############################################################################
#
# We evolved a :math:`2^n = 2` dimensional quantum state and performed :math:`3` independent measurements (non-commuting).
#
# In shadow Hamiltonian simulation, we encode :math:`4` expectation values in a :math:`2^2 = 4` dimensional
# quantum state, i.e. :math:`n_S = 2`.
# (TODO comment on whether or not that is practical. At first it seems
# redundant, but this technique is allowing new things like sampling from :math:`p_m = \langle O_m \rangle`
# that we get simultaneously, so no need to perform 3 independent measurements, i.e. saving on measurement resources)
#
# Let us first construct the initial shadow state :math:`\boldsymbol{O}(t=0)` by computing
# :math:`\langle O_m \rangle_{t=0} = \text{tr}\left(O_m |\psi(0)\rangle \langle \psi(0)| \right)`
# with :math:`|\psi(0)\rangle = |0\rangle`.
# The ``pauli_rep`` of PennyLane operators in form of :class:`~.pennylane.pauli.PauliSentence` instances let us efficiently
# compute the trace and we use the trick that :math:`|0 \rangle \langle 0| = (I + Z)/2`.
# 

S_pauli = [op.pauli_rep for op in S]

O_0 = []

for Om in S_pauli:
    psi0 = (I(0) + Z(0)).pauli_rep

    expval_Om = (psi0 @ Om).trace()
    O_0.append(expval_Om)

O_0 = np.array(O_0)
A = np.linalg.norm(O_0)

O_0

##############################################################################
# There is a variety of methods to encode this vector in a qubit basis. We will later just use
# :class:`~StatePrep`.
#
# We now go on to construct the shadow Hamiltonian :math:`H_S` by means of computing the elements
# :math:`(H_S)_{m m'} = \frac{\text{tr}\left( O_{m'} [H, O_m] \right)}{|| O_{m'} ||^2}`.
# We again make use of the :meth:`~.pennylane.pauli.PauliSentence.trace` method.
#

H_pauli = H.pauli_rep

H_S = np.zeros((len(S), len(S)), dtype=complex)

for m, Om in enumerate(S_pauli):
    com = H_pauli.commutator(Om)
    for mt, Omt in enumerate(S_pauli):
        # v = ∑ (v · e_j / ||e_j||^2) * e_j

        value = (Omt @ com).trace()
        value = value / (Omt @ Omt).trace()  
        H_S[m,mt] = value

H_S = -H_S # definition eq. (2) in [1]

##############################################################################
# In order for the shadow evolution to be unitary and implementable on a quantum computer,
# we need :math:`H_S` to be Hermitian.

np.all(H_S == H_S.conj().T)

##############################################################################
# Knowing that, we can write the formal solution to the shadow Schrödinger equation as
# 
# .. math:: \boldsymbol{O}(t) = \exp\left(-i t H_S \right) \boldsymbol{O}(0)

from scipy.linalg import expm

O_t = expm(-1j * t * H_S) @ O_0
O_t

##############################################################################
# Up to this point, this is equivalent to :doc:`g-sim </demos/tutorial_liesim>` if we were doing classical simulation.
# Now the main novelty for shadow Hamiltonian simulation is to perform this on a quantum computer by encoding the 
# expectation values of :math:`\langle O_m \rangle` in the amplitude of a quantum state, and to translate :math:`H_S`
# accordingly.
#
# This can be done by decomposing the numerical matrix :math:`H_S` into Pauli operators, which then in turn can
# be implemented on a quantum computer.
#

H_S_qubit = qml.pauli_decompose(H_S)
H_S_qubit

##############################################################################
# Using all these ingredients, we now are able to formulate the shadow Hamiltonian simulation as a quantum algorithm.
# For the amplitude encoding, we need to make sure that the state is normalized. We use that normalization factor to then
# later retrieve the correct result.
#

@qml.qnode(dev)
def shadow_evolve(H_S_qubit, O_0, t):
    qml.StatePrep(O_0 / A, wires=range(2))
    qml.evolve(H_S_qubit, t)
    return qml.state()

O_t_quantum = shadow_evolve(H_S_qubit, O_0, t) * A

print(O_t_standard)
print(O_t_quantum)

##############################################################################
# We see that the results match with both approaches.
#
# The first result is coming from three independent measurements on a quantum computer after evolution with system Hamiltonian :math:`H`.
# This is conceptually very different from the second result where
# :math:`\boldsymbol{O}` is encoded in the state of the shadow system, which we evolve according to :math:`H_S`.
#
# In the first case, the measurement is directly obtained, however, 
# in the shadow Hamiltonian simulation, we need to access the amplitudes of the underlying state (note the return value ``qml.state()`` in the qnode).
# This can be done naively with state tomography, but in instances where we know 
# that :math:`\langle O_m \rangle \in [0, 1]`, we can just sample bitstrings according to
# :math:`p_m = |\langle O_m\rangle|^2`. The ability to sample from such a distribution 
# :math:`p_m = |\langle O_m\rangle|^2` is a unique and new feature to shadow Hamiltonian simulation.
#
# We are making use of the abstract quantum sub-routines :func:`~evolve` and :class:`~StatePrep`, which each warrant their
# specific implementation. For example, :class:`~StatePrep` can be realized by :class:`~MottonenStatePreparation` and :func:`~evolve` can be realized
# by :class:`TrotterProduct`, though that shall not be the focus of this demo.

##############################################################################
#
# Second example
# --------------
#
# Something a bit more involved? I am not sure if this is actually necessary. The above example illustrates the basic concepts already quite nicely and keeps things concise. Happy to hear feedback here.

##############################################################################
#
# Green's functions and other correlators
# ---------------------------------------
#
# Perhaps include something about correlators here?

##############################################################################
# 
# Conclusion
# ----------
#
# We have introduced the basic concepts of shadow Hamiltonian simulation and seen how this fundamentally differs from the common approach to Hamiltonian simulation.
#
# We have seen how classical Hamiltonian simulation is tightly connected to g-sim, but run on a quantum computer.
# A significant difference comes from the fact that the authors in [#SommaShadow]_ specifically look at Hamiltonian simulation :math:`\exp(-i t H)`
# which allows us to just look at the support of :math:`[H, O_m]`, instead of the full Lie closure, which in most cases leads to an exponential amount of operators [#Wiersema]_ [#Aguilar]_.
#
# In [#SommaShadow]_, the authors specifically consider Hamiltonian simulation :math:`\exp(-i t H)`, which corresponds to just one single
# "gate", which, in some cases can significantly lower the required dimensionality compared to g-sim that requires the full Lie closure of 
# the considered observables and operators, as we will later see.



##############################################################################
# 
# References
# ----------
#
# .. [#SommaShadow]
#
#     Rolando D. Somma, Robbie King, Robin Kothari, Thomas O'Brien, Ryan Babbush
#     "Shadow Hamiltonian Simulation"
#     `arXiv:2407.21775 <https://arxiv.org/abs/2407.21775>`__, 2024.
#
# .. [#Somma]
#
#     Rolando D. Somma
#     "Quantum Computation, Complexity, and Many-Body Physics"
#     `arXiv:quant-ph/0512209 <https://arxiv.org/abs/quant-ph/0512209>`__, 2005.
#
# .. [#Somma2]
#
#     Rolando Somma, Howard Barnum, Gerardo Ortiz, Emanuel Knill
#     "Efficient solvability of Hamiltonians and limits on the power of some quantum computational models"
#     `arXiv:quant-ph/0601030 <https://arxiv.org/abs/quant-ph/0601030>`__, 2006.
#
# .. [#Galitski]
#
#     Victor Galitski
#     "Quantum-to-Classical Correspondence and Hubbard-Stratonovich Dynamical Systems, a Lie-Algebraic Approach"
#     `arXiv:1012.2873 <https://arxiv.org/abs/1012.2873>`__, 2010.
#
# .. [#Goh]
#
#     Matthew L. Goh, Martin Larocca, Lukasz Cincio, M. Cerezo, Frédéric Sauvage
#     "Lie-algebraic classical simulations for variational quantum computing"
#     `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__, 2023.
#
# .. [#Cerezo]
#
#     M. Cerezo, Martin Larocca, Diego García-Martín, N. L. Diaz, Paolo Braccia, Enrico Fontana, Manuel S. Rudolph, Pablo Bermejo, Aroosa Ijaz, Supanut Thanasilp, Eric R. Anschuetz, Zoë Holmes
#     "Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink variational quantum computing"
#     `arXiv:2312.09121 <https://arxiv.org/abs/2312.09121>`__, 2023.
#
# .. [#Wiersema]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
#
# .. [#Aguilar]
#
#     Gerard Aguilar, Simon Cichy, Jens Eisert, Lennart Bittel
#     "Full classification of Pauli Lie algebras"
#     `arXiv:2408.00081 <https://arxiv.org/abs/2408.00081>`__, 2024.

#
# .. [#Mazzola]
#
#     Guglielmo Mazzola
#     "Quantum computing for chemistry and physics applications from a Monte Carlo perspective"
#     `arXiv:2308.07964 <https://arxiv.org/abs/2308.07964>`__, 2023.
#
# .. [#Park]
#
#     Chae-Yeun Park, Minhyeok Kang, Joonsuk Huh
#     "Hardware-efficient ansatz without barren plateaus in any depth"
#     `arXiv:2403.04844 <https://arxiv.org/abs/2403.04844>`__, 2024.
#

##############################################################################
# About the author
# ----------------
#
