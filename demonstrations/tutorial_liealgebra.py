r"""A gentle introduction to (Dynamical) Lie Algebras in quantum computing
==========================================================================


If you have come across the term Dynamical Lie Algebra (DLA) in a quantum physics paper and are not quite sure what that is:
This is the right place for you. We are going to introduce the basics of (dynamical) Lie algebras, which are relevant for circuit 
expressivity, universality and symmetries in quantum physics.

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
:math:`U^{-1} = U^\dagger`. They describe how quantum states are transformed.

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
universality we require the available gates to form the special unitary Lie group :math:`SU(2^n)`.
It is often more convenient to work with the associated Lie algebra :math:`\mathfrak{su}(2^n)` that generates :math:`SU(2^n) = e^{-i \mathfrak{su}(2^n)}`
via the exponential map.

In summary, we have Hermitian operators and an imaginary factor in the exponent that form elements of a Lie algebra, which together generate
elements of a Lie group.

Lie algebras
------------

After some motivation and connections to concepts we are already familiar with, let us formally introduce Lie algebras.
An `algebra <https://en.wikipedia.org/wiki/Algebra_over_a_field>`_ is a vector space equipped with a bilinear operation.
A `Lie algebra <https://en.wikipedia.org/wiki/Lie_algebra>`_ :math:`\mathfrak{g}` is a special case where the bilinear operation behaves like a commutator.
In paricular, the bilinear operation :math:`[\bullet, \bullet]: \mathfrak{g} \times \mathfrak{g} \rightarrow \mathfrak{g}` needs to satisfy

* :math:`[x, x] = 0 \ \forall x \in \mathfrak{g}` (alternativity)
* :math:`[x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0 \ \forall x,y,z \in \mathfrak{g}` (Jacobi identity)
* :math:`[x, y] = - [y, x] \ \forall x,y,z \in \mathfrak{g}` (anti-commutativity)

The last one, anti-commutativity, technically is not an axiom but follows from bilinearity and alternativity, but is so crucial that it is worth highlighting.
These properties generally define the so-called Lie bracket, where the commutator is just one
special case thereof. A different example would be the `cross-product <https://en.wikipedia.org/wiki/Cross_product>`_ between vectors in :math:`\mathbb{R}^3`.

One very relevant Lie algebra for us is the special unitary algebra :math:`\mathfrak{su}(N)`, the space of :math:`N \times N` skew-Hermitian matrices with trace zero.
The fact that we look at skew-Hermitian (:math:`H^\dagger = - H`) instead of Hermitian (:math:`H^\dagger = H`) matrices is a technical detail. For all practical purposes you
can just think of Hermitian operators with an imaginary factor and note that linear cominations are strictly over the reals. In fact, you may sometimes
find references to :math:`\mathfrak{su}(N)` being the Hermitian matrices in physics literature (see `wikipedia <https://en.wikipedia.org/wiki/Special_unitary_group#Fundamental_representation>`_).

.. note::

    The result of a commutator between two Hermitian operators :math:`H_1` and :math:`H_2` is always skew-Hermitian due to the commutator's anti-commutativity, i.e.

    .. math:: [H_1, H_2]^\dagger = [H_2^\dagger, H_1^\dagger] = - [H_1, H_2].

    This means that Hermitian operators do not form a Lie algebra because the commutator maps outside the set of Hermitian matrices. But instead, skew-Hermitian operators do.
    Note that the algebra of :math:`N \times N` skew-Hermitian matrices is called the unitary algebra :math:`\mathfrak{u}(N)`, whereas
    the additional property of the trace being zero making it the `special` unitary algebra :math:`\mathfrak{su}(N)`. They generate
    the unitary group :math:`U(N)` and the special unitary group `SU(N)` with determinant 1, respectively.


The Pauli matrices :math:`\{iX, iY, iZ\}` span the :math:`\mathfrak{su}(2)` algebra that we typically associate with a single qubit.
For multiple qubits we have 

.. math:: \mathfrak{su}(2^n) = \text{span}_{\mathbb{R}}\left(\{iX_0, .., iY_0, .., iZ_0, .., iX_0 X_1, .. iY_0 Y_1, .., iZ_0 Z_1, ..\}\right),

where the span is over the reals :math:`\mathbb{R}`. In particular, we do cannot do a complex span, since this could destroy the anti-commutativity again. Another way
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
# .. math:: U = e^{i \sum_j \lambda_j P_j}
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
# form :math:`\mathfrak{su}(2^n)`.
# 
# Dynamical Lie Algebras
# ~~~~~~~~~~~~~~~~~~~~~~
#
# A different way of looking at this is taking a set of generators :math:`\{G_j\}` and asking what kind of 
# unitary evolutions they can generate. This naturally introduces the so-called Dynamical Lie Algebra (DLA),
# originally coined in quantum optimal control theory and recently re-emerging in the quantum computing literature.
# The DLA :math:`i\mathfrak{g}` is given by all possible nested commutators between the generators :math:`\{G_j\}`,
# until no new and linearly independent skew-Hermitian operator is generated. This is called the Lie-closure and is
# written like
#
# .. math:: i \mathfrak{g} = \langle iG_1, iG_2, iG_3,.. \rangle_\text{Lie}
#
# On one hand, the Lie closure ensures that the DLA is closed under commutation.
# But you can also think of the Lie closure as filling the missing operators to describe the possible dynamics in terms of its Lie algebra.
# Take for example the set of generators :math:`\{iX, iY\}`.
# The Lie closure yields another :math:`iZ` operator because :math:`[X, Y] = 2iZ`. But lets imagine for a second that we dont take the Lie closure but just take the two generators.
# We can still write a Pauli-Z rotation (here exemplarily with angle :math:`0.5`) as a product of them.

U_target = qml.matrix(qml.RZ(-0.5, 0))
decomp = qml.ops.one_qubit_decomposition(U_target, 0, rotations="XYX")
print(decomp)

##############################################################################
# We can check that this is indeed a valid decomposition by computing the trace distance

U = qml.matrix(decomp[0])
U = U @ qml.matrix(decomp[1])
U = U @ qml.matrix(decomp[2])
1 - np.real(np.trace(U_target @ U))/2

##############################################################################
# So we see that the generators :math:`iX` and :math:`iY` suffice to express the target unitary. However, we cannot write
# :math:`U = e^{-i(\lambda_1 X + \lambda_2 Y)}` since we are missing the :math:`iZ` from the DLA 
# :math:`i\mathfrak{g} = \langle iX, iY \rangle_\text{Lie} = \{iX, iY, iZ\}`.
#
# Another handy example are the generators :math:`\{X_0 X_1, Z_0, Z_1\}`. The Lie closure yields
# :math:`\text{tbc}`.
#
# so(2^n)
# ~~~~~~~
# 
# sp(2^n)
# ~~~~~~~
#
# Symmetries
# ----------
# Heisenberg model as an example of something that is considered SU(2) symmetric. Explain what is meant by that, should be eye-opening to many.




##############################################################################
# 
#


##############################################################################
# 
# Conclusion
# ----------
#
# conclusion



##############################################################################
# 
# References
# ----------
#
# .. [#Kottmann]
#
#     Korbinian Kottmann, Nathan Killoran
#     "Evaluating analytic gradients of pulse programs on quantum computers"
#     `arXiv:2309.16756 <https://arxiv.org/abs/2309.16756>`__, 2023.
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
