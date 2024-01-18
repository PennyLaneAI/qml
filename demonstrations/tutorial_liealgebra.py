r"""Introducing (Dynamical) Lie Algebras for quantum practitioners
==================================================================


We are going to introduce the basics of (dynamical) Lie algebras with a focus on quantum computing.
In particular, we are going to motivate and introduce Lie algebras and Lie groups from a persepctive that is amenable to
quantum scientists, engineers, and practitioners that are new to Lie theory.
Let's see what the fuzz is all about, shall we?

Introduction
------------

Lie algebras offer a fresh perspective on some of the established ideas in quantum physics and become more and more important
in quantum computing. Let us recap some of the key concepts of quantum mechanics and how they relate to and give rise to Lie
algebras.

Most physicists know quantum physics in terms of wavefunctions :math:`|\psi\rangle`
that live in a `Hilbert space <https://en.wikipedia.org/wiki/Hilbert_space>`_ :math:`\mathcal{H}`, 
as well as (bounded) linear operators :math:`\hat{O}` that live in the space of linear operators on
that Hilbert space, :math:`\mathcal{L}(\mathcal{H})`. For finite dimensional systems (think, :math:`n` number of qubits)
we have complex valued state vectors (wavefunctions) in :math:`\mathcal{H} = \mathbb{C}^{2^n}` with norm 1 and 
square matrices (linear operators) in :math:`\mathcal{L}(\mathcal{H}) = \mathbb{C}^{2^n \times 2^n}`.

Two very important sub-classes of linear operators in quantum mechanics are unitary and Hermitian operators.
Hermitian operators :math:`H` are self-adjoint, :math:`H^\dagger = H`, and describe observables that can be measured. Unitary operators are norm-preserving
such that :math:`\langle \psi | U^\dagger U | \psi \rangle = \langle \psi | \psi \rangle`, in particular we have
:math:`U^{-1} = U^\dagger`. They describe how quantum states are transformed and ensure that their norm is preserved.

A unitary operator can always be written as

.. math:: U = e^{-i H},

where we say that :math:`H` is the generator of :math:`U`. Take for example a single qubit rotation
:math:`U(\phi) = e^{-i \frac{\phi}{2} X}`. :math:`U(\phi)` is the unitary evolution that rotates a quantum
state in Hilbert space around the x-axis on the `Bloch sphere <https://en.wikipedia.org/wiki/Bloch_sphere>`_, 
and is generated by the Pauli :math:`X` `matrix <https://en.wikipedia.org/wiki/Pauli_matrices>`_.

The space of all such unitary operators
forms the so-called special unitary group :math:`SU(N)`, where for qubit systems we have :math:`N=2^n` with :math:`N` the dimension of the group
and `n` the number of qubits.
In quantum computing, we are typically dealing with the Hilbert space :math:`\mathcal{H} = \mathbb{C}^{2^n}` and for full
universality we require the available gates to span all of :math:`SU(2^n)`. That means when we have all unitaries of :math:`SU(2^n)`
available to us, we can reach any state in Hilbert space from any other state.

This Lie group has an associated Lie algebra to it, called :math:`\mathfrak{su}(2^n)` (more on that later).
In some cases, it is more convenient to work with the associated Lie algebra rather than the Lie group.

So if you are familiar with quantum computing but knew nothing about Lie algebras and Lie groups before this demo, 
the good news is that you actually already know the elements of both. Roughly speaking, the relevant Lie group in quantum computing
is the space of unitaries, and the relevant Lie algebra is the space of Hermitian matrices. Further, they are related to each other: The Lie algebra (Hermitian matrices)
generates the Lie group (unitaries) via the exponential map. There are, however, some subtleties if we want to be mathematically precise, as we will explore more in depth now.

Lie algebras
------------

After some motivation and connections to concepts we are already familiar with, let us formally introduce Lie algebras.
An `algebra <https://en.wikipedia.org/wiki/Algebra_over_a_field>`_ is a vector space equipped with a bilinear operation.
A `Lie algebra <https://en.wikipedia.org/wiki/Lie_algebra>`_ :math:`\mathfrak{g}` is a special case where the bilinear operation behaves like a commutator.
In paricular, the bilinear operation :math:`[\bullet, \bullet]: \mathfrak{a} \times \mathfrak{g} \rightarrow \mathfrak{g}` needs to satisfy

* :math:`[x, x] = 0 \ \forall x \in \mathfrak{a}` (alternativity)
* :math:`[x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0 \ \forall x,y,z \in \mathfrak{a}` (Jacobi identity)
* :math:`[x, y] = - [y, x] \ \forall x,y \in \mathfrak{a}` (anti-commutativity)

The last one, anti-commutativity, technically is not an axiom but follows from bilinearity and alternativity, but is so crucial that it is worth highlighting.
These properties generally define the so-called Lie bracket, where the commutator is just one
special case thereof. A different example would be the `cross-product <https://en.wikipedia.org/wiki/Cross_product>`_ between vectors in :math:`\mathbb{R}^3`.
Note also that we are talking about a **vector** space in the mathematical sense, and the elements ("vectors") in :math:`\mathfrak{g}` are actually operators (matrices) in our case looking at quantum physics.

One very relevant Lie algebra for us is the special unitary algebra :math:`\mathfrak{su}(N)`, the space of :math:`N \times N` skew-Hermitian matrices with trace zero.
The fact that we look at skew-Hermitian (:math:`H^\dagger = - H`) instead of Hermitian (:math:`H^\dagger = H`) matrices is a technical detail (see note below). For all practical purposes you
can just think of Hermitian operators with an imaginary factor and note that linear cominations are strictly over the reals. In fact, you may sometimes
find references to :math:`\mathfrak{su}(N)` being the Hermitian matrices in physics literature (see `wikipedia <https://en.wikipedia.org/wiki/Special_unitary_group#Fundamental_representation>`_).

.. note::

    The result of a commutator between two Hermitian operators :math:`H_1` and :math:`H_2` is always skew-Hermitian due to the commutator's anti-commutativity, i.e.

    .. math:: [H_1, H_2]^\dagger = [H_2^\dagger, H_1^\dagger] = - [H_1, H_2].

    This means that Hermitian operators are not closed under commutation, and thus do not form a Lie algebra
    (because the commutator maps outside the set of Hermitian matrices). But instead, skew-Hermitian operators do.
    Note that the algebra of :math:`N \times N` skew-Hermitian matrices is called the unitary algebra :math:`\mathfrak{u}(N)`, whereas
    the additional property of the trace being zero making it the `special` unitary algebra :math:`\mathfrak{su}(N)`. They generate
    the unitary group :math:`U(N)` and the special unitary group `SU(N)` with determinant 1, respectively.


The Pauli matrices :math:`\{iX, iY, iZ\}` span the :math:`\mathfrak{su}(2)` algebra that we can associate with single qubit dynamics.
For multiple qubits we have 

.. math:: \mathfrak{su}(2^n) = \text{span}_{\mathbb{R}}\left(\{iX_0, .., iY_0, .., iZ_0, .., iX_0 X_1, .. iY_0 Y_1, .., iZ_0 Z_1, ..\}\right),

where the span is over the reals :math:`\mathbb{R}`. In particular, we cannot do a complex span, since this could destroy the anti-commutativity again. Another way
of thinking about this is that Lie algebra elements "live" in the exponent of a unitary operator, and having that exponent become Hermitian instead of skew-Hermitian
destroys the unitary property.

Let us briefly test some of these properties numerically.
First, let us do a linear combination of :math:`\{iX, iY, iZ\}` with some real values and check unitarity after putting them in the exponent.
"""
import numpy as np
import pennylane as qml
from pennylane.pauli import PauliWord

X, Y, Z = PauliWord({0:"X"}), PauliWord({0:"Y"}), PauliWord({0:"Z"})
su2 = [1j * X, 1j * Y, 1j * Z]

coeffs = [1., 2., 3.]
exponent = sum([c * P for c,P in zip(coeffs, su2)])
U = qml.math.expm(exponent.to_mat())
print(np.allclose(U.conj().T @ U, np.eye(2)))

##############################################################################
# If we throw complex values in the mix, the resulting matrix is not unitary anymore.

coeffs = [1., 2.+1j, 3.]
exponent = sum([c * P for c,P in zip(coeffs, su2)])
U = qml.math.expm(exponent.to_mat())
print(np.allclose(U.conj().T @ U, np.eye(2)))

##############################################################################
# 
# Relation to Lie groups
# ----------------------
# We said earlier that the Lie group :math:`SU(N)` is generated by the Lie algebra :math:`\mathfrak{su}(N)`.
# But what do we actually mean by that?
# Essentially, for every unitary matrix :math:`U \in SU(N)` there is a (real) linear combination of elements :math:`iP_j \in \mathfrak{su}(N)` such that
#
# .. math:: U = e^{i \sum_{j=1}^N \lambda_j P_j}
#
# for some real coefficients :math:`\lambda_j \in \mathbb{R}`.
# 
# In quantum computing we are interested in unitary gates, that, composed together, 
# realize a complicated unitary evolution :math:`U`. That could, for example, be
# a unitary that prepares the ground state of a Hamiltonian from the 
# :math:`|0\rangle^{\otimes n}` state or perform a sub-routine like the quantum 
# Fourier transform. In particular, we are not composing quantum circuits via creating
# superpositions of Lie algebra elements as is done in the last equation.
#
# Luckily, beyond the relation above, we also know that any unitary matrix :math:`U \in SU(2^n)`
# can be decomposed in a finite product of elements from a universal gate set :math:`\mathcal{U}`,
#
# .. math:: U = \prod_j U_j
#
# for :math:`U_j \in \mathcal{U}`. A universal gate set is formed exactly when the generators of its elements
# form :math:`\mathfrak{su}(2^n)`. Note that in this equation the product may feature a large number of gates :math:`U_j`, 
# so universality does not guarantee an efficient decomposition but rather just a finite one.
# 
# Dynamical Lie Algebras
# ~~~~~~~~~~~~~~~~~~~~~~
#
# A different way of looking at this is taking a set of generators :math:`\{iG_j\}` and asking what kind of 
# unitary evolutions they can generate. This naturally introduces the so-called Dynamical Lie Algebra (DLA),
# originally coined in quantum optimal control theory and recently re-emerging in the quantum computing literature.
# The DLA :math:`i\mathfrak{g}` is given by all possible nested commutators between the generators :math:`\{iG_j\}`,
# until no new and linearly independent skew-Hermitian operator is generated. This is called the Lie-closure and is
# written like
#
# .. math:: i \mathfrak{g} = \langle iG_1, iG_2, iG_3,.. \rangle_\text{Lie}.
#
# Let us do a quick example and compute the Lie closure of :math:`\{iX, iY\}` (more examples later).

def commutator(x, y):
    return x @ y - y @ x

print(commutator(1j * X, 1j * Y))

##############################################################################
# We know that the commutator between :math:`iX` and :math:`iY` yields a new operator :math:`\propto iZ`.
# So we add :math:`iZ` to our list of operators and continue to take commutators between them.

list_ops = [1j * X, 1j * Y, 1j * Z]
for op1 in list_ops:
    for op2 in list_ops:
        print(commutator(op1, op2))

##############################################################################
# Since no new operators have been created we know the lie closure is complete and our dynamical Lie algebra
# is :math:`\langle\{iX, iY\}\rangle_\text{Lie} = \{iX, iY, iZ\}( = \mathfrak{su}(2))`.
#
# On one hand, the Lie closure ensures that the DLA is closed under commutation.
# But you can also think of the Lie closure as filling the missing operators to describe the possible dynamics in terms of its Lie algebra.
# Let us stick to the example above and imagine for a second that we dont take the Lie closure but just take the two generators :math:`\{iX, iY\}`.
# These two generators suffice for universality (for a single qubit) in that we can write any evolution in :math:`SU(2)` as a finite product of 
# :math:`e^{-i \phi X}` and :math:`e^{-i \phi Y}`. For example, let us write a Pauli-Z rotation at non-trivial angle :math:`0.5` as a product of them.

U_target = qml.matrix(qml.RZ(-0.5, 0))
decomp = qml.ops.one_qubit_decomposition(U_target, 0, rotations="XYX")
print(decomp)

##############################################################################
# We can check that this is indeed a valid decomposition by computing the trace distance.

U = qml.matrix(decomp[0])
U = U @ qml.matrix(decomp[1])
U = U @ qml.matrix(decomp[2])
1 - np.real(np.trace(U_target @ U))/2

##############################################################################
# So we see that a finite set of generators :math:`iX` and :math:`iY` suffice to express the target unitary. However, we cannot write
# :math:`U = e^{-i(\lambda_1 X + \lambda_2 Y)}` since we are missing the :math:`iZ` from the DLA 
# :math:`i\mathfrak{g} = \langle iX, iY \rangle_\text{Lie} = \{iX, iY, iZ\}`.
#
# so(2^n)
# ~~~~~~~
# Let us work through another example to get some exercise. Let us look at the generators :math:`\{iX_0 X_1, iZ_0, iZ_1\}`. 
# You may recognize them as the terms in the transverse field Ising model (here for the simple case of :math:`n=2`)
#
# .. math:: H_\text{Ising} = \sum_{\langle i, j \rangle} X_i X_j + \sum_{j=1}^n Z_j
#
# where :math:`\langle i, j \rangle` indicates a sum over nearest neighbors in the system's topology.
# Let us compute the first set of commutators for those generators.

XX = PauliWord({0:"X", 1:"X"})
Z0 = PauliWord({0:"Z"})
Z1 = PauliWord({1:"Z"})

generators = [1j * XX, 1j * Z0, 1j * Z1]
dla = generators.copy()
for i, op1 in enumerate(generators):
    for op2 in generators[i+1:]:
        res = commutator(op1, op2)
        print(f"[{op1}, {op2}] = {res}")
        if next(iter(res.values()))!=0. and res/2. not in dla and -1* res/2 not in dla:
            dla.append(res/2.)

##############################################################################
# We obtain two new operators :math:`iY_0 X_1` and :math:`iX_0 Y_1` and append the list of operators of the DLA.
# We then continue with depth-1 nested commutators (as :math:`iY_0 X_1 \propto [iX_0 X_1, iZ_0]`).

for i, op1 in enumerate(dla.copy()):
    for op2 in dla.copy()[i+1:]:
        res = commutator(op1, op2)
        print(f"[{op1}, {op2}] = {res}")

        # add new operator to dla, normalize for convenience
        if next(iter(res.values()))!=0. and res/2. not in dla and -1* res/2 not in dla:
            dla.append(res/2.)

##############################################################################
# We could continue this process with a second nesting layer but will find that no new operators are added past this point.
# We finally end up with the DLA :math:`\{X_0 X_1, Z_0, Z_1, iY_0 X_1 iX_0 Y_1, iY_0 Y_1\}`

for op in dla:      
    print(op)

##############################################################################
# Curiously, even though both :math:`iZ_0` and :math:`iZ_1` are in the DLA, :math:`iZ_0 Z_1` is not.
# Hence, products of generators are not necessarily in the DLA.
#
# The DLA obtained from the Ising generators form the so-called special orthogonal Lie algebra
# :math:`\mathfrak{so}(2n-1)`, which has the dimension :math:`(2n-1) (2n-2) = 6`, equal to the number of operators we obtain in the DLA.
# This DLA is special as it is one of the few DLAs that has a polynomial size and is thus efficiently simulatable [#Goh]_.
#
# For one spatial dimension, there is a full classification of all translationally invariant systems [#Wiersma]_. Less common but also relevant
# is the `symplectic algebra <https://en.wikipedia.org/wiki/Symplectic_group>`_ :math:`\mathfrak{sp}(2N)` that is also polynomial in size.
#
# Symmetries
# ----------
#
# With this new knowledge we are now able to understand what is meant when some Hamiltonian models are said to be symmetric under some symmetry group.
# Specifically, let us look at the spin-1/2 Heisenberg model Hamiltonian in 1D with nearest neighbor interactions,
#
# .. math:: H_\text{Heis} = \sum_{j=1}^{n-1} J_j \left(X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1} \right)
#
# with some coupling constants :math:`J_j \in \mathbb{R}`. First it is important to understand that the generators here are made up of the whole
# sum of operators :math:`X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1}`, and not each individual term.
# This Hamiltonian is said to be :math:`SU(2)` invariant, but what does that mean?
#
# First, the system describes a chain of coupled spins. With no external field, as is the case in the model description above, the total spin components
#
# .. math:: S_\text{tot}^{x} = \sum_{j=1}^n X_j ; \ S_\text{tot}^{y} = \sum_{j=1}^n Y_j ; \ S_\text{tot}^{z} = \sum_{j=1}^n Z_j
#
# must be preserved. That means that expectation values of the spin components cannot change under evolution of the system Hamiltonian.
# Mathematically, this is expressed by identifying so-called charges :math:`Q` that commute with the Hamiltonian. 
# Let us check that briefly for a small example.

XX = PauliWord({0:"X", 1:"X"}) ; IXX = PauliWord({1:"X", 2:"X"})
YY = PauliWord({0:"Y", 1:"Y"}) ; IYY = PauliWord({1:"Y", 2:"Y"})
ZZ = PauliWord({0:"Z", 1:"Z"}) ; IZZ = PauliWord({1:"Z", 2:"Z"})

H0 = XX + YY + ZZ
H1 = IXX + IYY + IZZ
SX = PauliWord({0:"X"}) + PauliWord({1:"X"}) + PauliWord({2:"X"})
SY = PauliWord({0:"Y"}) + PauliWord({1:"Y"}) + PauliWord({2:"Y"})
SZ = PauliWord({0:"Z"}) + PauliWord({1:"Z"}) + PauliWord({2:"Z"})

print(commutator(H0, SX).simplify()) # simplify removes words with 0 coefficient
print(commutator(H0, SY).simplify())
print(commutator(H0, SZ).simplify())
print(commutator(H1, SX).simplify())
print(commutator(H1, SY).simplify())
print(commutator(H1, SZ).simplify())

##############################################################################
# We can see how this generalizes to arbitary indices pairs ``(i, i+1)``. 
# So overall we have the three charges :math:`S_\text{tot}^{x}, S_\text{tot}^{y}, S_\text{tot}^{z}`
# and they span a representation of :math:`\mathfrak{su}(2)`. This may be a bit confusing because earlier we said :math:`\mathfrak{su}(2) = \{iX, iY, iZ\}`.
# What is really meant by that is that these generators span a `representation` of :math:`\mathfrak{su}(2)`, where :math:`\{S_\text{tot}^{x}, S_\text{tot}^{y}, S_\text{tot}^{z}\}`
# is just another one.
#
# Another thing that may be confusing is the fact that the Hamiltonian is commuting with elements that form a Lie algebra, but we usually associate symmetries with the respective group.
# That comes down to terminology and the fact that we often care about invariance: An observable made up of the charges :math:`\hat{O} = c_x S^x_\text{tot} + c_x S^y_\text{tot} + c_x S^z_\text{tot}`
# is invariant under any evolution of the Hamiltonian. That is because any state :math:`|\psi\rangle` is confined to the symmetry sector of the associated charge in Hilbert space. This can be seen from
# 
# .. math:: \langle \psi | e^{i t H_\text{Heis}} \hat{O} e^{-i t H_\text{Heis}} |\psi\rangle = \langle \psi | e^{i t H_\text{Heis}} e^{-i t H_\text{Heis}} \hat{O} |\psi\rangle = \langle \psi | \hat{O} |\psi\rangle 
#
# where we use the fact that :math:`\hat{O}` is made up of charges that each commute with :math:`H_\text{Heis}`, and thus with :math:`U=e^{-i t H_\text{Heis}}`.
# So overall, the evolution of :math:`H_\text{Heis}` is invariant under that representation of :math:`SU(2)`, which is generated by :math:`\{S_\text{tot}^{x}, S_\text{tot}^{y}, S_\text{tot}^{z}\}`.
#
# .. note::
#     Symmetries play a big role in quantum phase transitions:
#     Imagine preparing the ground state at zero temperature of a system that has a symmetry. Accordingly, the ground state must
#     be invariant under that symmetry. However, it may happen that by adiabatically (very slowly) changing the system parameters 
#     (while staying at zero temperature), the expectation value of the associated charge changes. That is what is called the spontaneous breaking of the symmetry
#     and it is associated with a quantum phase transition.
#


##############################################################################
# 
# Conclusion
# ----------
#
# With this introduction, we hope to clarify some terminology, introduce the basic concepts of Lie theory and motivate their relevance in quantum physics by touching on universality and symmetries.
# While Lie theory and symmetries are playing a central role in established fields such as quantum phase transitions (see note above) and `high energy physics <https://en.wikipedia.org/wiki/Standard_Model>`_,
# they have recently also emerged in quantum machine learning with the onset of geometric quantum machine learning [#Meyer]_ [#Nguyen]_ (see our recent :doc:`~tutorial_geometric_qml`.
# Further, DLAs have recently become instrumental in classifying criteria for barren plateaus [#Fontana]_ [#Ragone]_ and designing simmulators based on them [#Goh]_.
#



##############################################################################
# 
# References
# ----------
#
# .. [#Wiersma]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
#
# .. [#Meyer]
#
#     Johannes Jakob Meyer, Marian Mularski, Elies Gil-Fuster, Antonio Anna Mele, Francesco Arzani, Alissa Wilms, Jens Eisert
#     "Exploiting symmetry in variational quantum machine learning"
#     `arXiv:2205.06217 <https://arxiv.org/abs/2205.06217>`__, 2022.
#
# .. [#Nguyen]
#
#     Quynh T. Nguyen, Louis Schatzki, Paolo Braccia, Michael Ragone, Patrick J. Coles, Frederic Sauvage, Martin Larocca, M. Cerezo
#     "Theory for Equivariant Quantum Neural Networks"
#     `arXiv:2210.08566 <https://arxiv.org/abs/2210.08566>`__, 2022.
#
# .. [#Fontana]
#
#     Enrico Fontana, Dylan Herman, Shouvanik Chakrabarti, Niraj Kumar, Romina Yalovetzky, Jamie Heredge, Shree Hari Sureshbabu, Marco Pistoia
#     "The Adjoint Is All You Need: Characterizing Barren Plateaus in Quantum Ansätze"
#     `arXiv:2309.07902 <https://arxiv.org/abs/2309.07902>`__, 2023.
#
# .. [#Ragone]
#
#     Michael Ragone, Bojko N. Bakalov, Frédéric Sauvage, Alexander F. Kemper, Carlos Ortiz Marrero, Martin Larocca, M. Cerezo
#     "A Unified Theory of Barren Plateaus for Deep Parametrized Quantum Circuits"
#     `arXiv:2309.09342 <https://arxiv.org/abs/2309.09342>`__, 2023.
#
# .. [#Goh]
#
#     Matthew L. Goh, Martin Larocca, Lukasz Cincio, M. Cerezo, Frédéric Sauvage
#     "Lie-algebraic classical simulations for variational quantum computing"
#     `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__, 2023.
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
