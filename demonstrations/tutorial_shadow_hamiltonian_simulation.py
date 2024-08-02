r"""Shadow Hamiltonian Simulation
=================================

abstract

Introduction
------------

content

Shadow Hamiltonian simulation
-----------------------------

In common quantum Hamiltonian simulation, we evolve a state vector :math:`|\psi(t)\rangle` according to the Schrödinger equation
by some Hamiltonian :math:`H`, and then compute expectation values of the evolved state through measurement.
In Shadow Hamiltonian simulation, we encode a set of expectation values in the amplitudes of a quantum state,
and evolve those according to some shadow Schrödinger equation.

For that, we first need to define the shadow state

.. math:: |\rho\rangle = \frac{1}{\sqrt{A}} \begin{pmatrix} \langle O_1 \\ \vdots \\ O_M \rangle \end{pmatrix}

for a set of operators :math:`S = \{O_m\}` with normalization constant :math:`A = \sum_m \langle O_m \rangle`.
So we can encode those :math:`M` operator expectation values with :math:`n_S` qubits such that :math:`2^{n_S} >= M`.

The shadow state evolves according to its shadow Schrödinger equation

.. math:: \frac{d}{dt} |\rho\rangle = - i H_S |\rho\rangle.

The Hamiltonian matrix :math:`H_S` is given by the commutation relations between the system Hamiltonian :math:`H` and
the operators in :math:`S = \{O_m\}`. In particular, it is implicitly defined by

.. math:: [H, O_m] = - \sum_{m'=1}^M \left( H_S \right)_{m m'} O_{m'}.

Knowing that the Hermitian operators under consideration form a vector space, we can use standard projection
techniques like 
:math:`\boldsymbol{v} = \sum_j \frac{\langle \boldsymbol{e}_j, \boldsymbol{v}\rangle}{||\boldsymbol{e}_j||^2} \boldsymbol{e}_j`
for some vector :math:`\boldsymbol{v}` and an orthogonal basis :math:`\boldsymbol{e}_j` (need not be normalized, hence the normalization factor in the projection).

For operators using the trace inner product this amounts to

.. math:: [H, O_m] = - \sum_{m'=1}^M \frac{\text{tr}\left( O_{m'}, [H, O_m] \right)}{|| O_{m'} ||^2} O_{m'},

from which we can read off the matrix elements of :math:`H_S`, i.e.

.. math:: (H_S)_{m m'} = \frac{\text{tr}\left( O_{m'}, [H, O_m] \right)}{|| O_{m'} ||^2}.

Note that this is just the adjoint representation :math:`\text{ad}_H` of the :doc:`dynamical Lie algebra </demos/tutorial_liealgebra>`
:math:`\langle \{ H \} \union S \rangle_\text{Lie}`, see our :doc:`demo on g-sim </demos/tutorial_liesim>` that makes extensive use of the adjoint representation.

A simple example
----------------

These abstract concepts are best illustrated with a simple but concrete example.
We are interested in simulating the Hamiltonian evolution of 

.. math:: H = X + Y

after :math:`t = 1s` and compute the expectation values of :math:`\{X, Y, Z, I \}`.
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
res = evolve(H, t)
res

##############################################################################
#
# We evolved a :math:`2^n = 2` dimensional quantum state and performed :math:`3` independent measurements (non-commuting).
# In shadow Hamiltonian simulation, we encode :math:`4` expectation values in a :math:`2^2 = 4` dimensional
# quantum state, i.e. :math:`n_S = 2`.
# (TODO comment on whether or not that is practical. At first it seems
# redundant, but this technique is allowing new things like sampling from :math:`p_m = \langle O_m \rangle`
# that we get simultaneously, so no need to perform 3 independent measurements, i.e. saving on measurement resources)
#
# Let us first construct the initial shadow state :math:`\boldsymbol{O}(t=0)` by computing
# :math:`\langle O_m \rangle_{t=0} = \text{tr}\left(O_m |\psi(0)\rangle \langle \psi(0)| \right)`
# with :math:`|\psi(0)\rangle = |0\rangle`.
# The `pauli_rep` of PennyLane operators in form of :class:`~PauliSentence` instances let us efficiently
# compute the trace and we use the trick that :math:`|0 \rangle \langle 0| = (I + Z)/2`.
# 

S_pauli = [op.pauli_rep for op in S]

O_0 = []

for Om in S_pauli:
    psi0 = (I(0) + Z(0)).pauli_rep

    res = (psi0 @ Om).trace()
    O_0.append(res)

O_0 = np.array(O_0)
A = np.linalg.norm(O_0)

##############################################################################
# There is a variety of method to encode this vector in a qubit basis. We will later just use
# :class:`~StatePrep` as the focus 
#
# We now go on to construct the shadow Hamiltonian :math:`H_S` by means of computing the elements
# :math:`(H_S)_{m m'} = \frac{\text{tr}\left( O_{m'}, [H, O_m] \right)}{|| O_{m'} ||^2}`.
# We again make use of :meth:`~PauliSentence.trace`.
#

H_pauli = H.pauli_rep

H_S = np.zeros((len(S), len(S)), dtype=complex)

for m, Om in enumerate(S_pauli):
    com = H_pauli.commutator(Om)
    for mt, Omt in enumerate(S_pauli):
        # value = (1j * (Omt @ com).trace()).real
        value = (Omt @ com).trace()
        value = value / (Omt @ Omt).trace()  # v = ∑ (v · e_j / ||e_j||^2) * e_j
        H_S[m,mt] = value

H_S = -H_S # definition eq. (2) in [1]

##############################################################################
# In order for the shadow evolution to be unitary and implementable on a quantum computer,
# we need :math:`H_S` to be Hermitian.

H_S == H_S.conj().T

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

res, O_t_quantum

##############################################################################
# We see that the results match. The first result is three independent measurements after evolving a quantum state
# according to the state Hamiltonian :math:`H`. This is conceptually very different from the second result where
# :math:`\boldsymbol{O}` is encoded in the state of the shadow system, which we evolve according to `H_S`.
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
# Conclusion
# ----------
#
# Shadow Hamiltonian simulation is g-sim on a quantum computer.



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
