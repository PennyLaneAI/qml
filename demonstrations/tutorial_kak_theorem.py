r"""The KAK theorem
===================

The KAK theorem is a beautiful mathematical result from Lie theory, with
particular relevance for quantum computing research. It can be seen as a
generalization of the singular value decomposition, and therefore falls
under the large umbrella of matrix factorizations. This allows us to
use it for quantum circuit decompositions. However, it can also
be understood from a more abstract point of view, as we will see.

In this demo, we will discuss so-called symmetric spaces, which arise from
subgroups of Lie groups. For this, we will focus on the algebraic level
and introduce Cartan decompositions, horizontal
and vertical subspaces, as well as (horizontal) Cartan subalgebras.
With these tools in our hands, we will then learn about the KAK theorem
itself.

As an application, we will get to know a handy decomposition of arbitrary
two-qubit unitaries into rotation gates. We will use this example throughout
to accompany the mathematical derivation in code.


.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_kak_theorem.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

.. admonition:: Prerequisites
    :class: note

    In the following we will assume a basic understanding of vector spaces,
    linear maps, and Lie algebras. For the former two, we recommend a look
    at your favourite linear algebra material, for the latter see our
    :doc:`introduction to (dynamical) Lie algebras </demos/tutorial_liealgebra/>`.


Introduction
------------

Basic mathematical objects
--------------------------

Introduce the mathematical objects that will play together to yield
the KAK theorem.

(Semi-)simple Lie algebras
~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, we will assume a basic understanding of the mathematical objects
we will use. To warm up, however, let us briefly talk about Lie algebras (for details
see our :doc:`intro to (dynamical) Lie algebras </demos/tutorial_liealgebra/>`).

A *Lie algebra* :math:`\mathfrak{g}` is a vector space with an additional operation
between two vectors, called the *Lie bracket*, that yields a vector again.
For our purposes, the vectors will always be matrices and the Lie bracket is the matrix
commutator.

**Example**

Our working example in this demo will be the *special unitary* algebra :math:`\mathfrak{su}(2)`.
It consists of traceless complex-valued skew-Hermitian :math:`2\times 2` matrices, i.e.,

.. math::

    \mathfrak{su}(2) = \left\{x \in \mathbb{C}^{(2\times 2)} \large| x^\dagger = -x , \text{tr}[x]=0\right\}.

We will look at a slightly more complex example at the end of the demo.

.. admonition:: Mathematical detail
    :class: note

    :math:`\mathfrak{su}(n)` is a *real* Lie algebra, i.e., it is a vector space over the
    real numbers :math:`\mathbb{R}`. This means that scalar-vector multiplication is
    only valid between vectors (complex-valued matrices) and real scalars.

    There is a simple reason to see this; Multiplying a skew-Hermitian matrix
    :math:`x\in\mathfrak{su}(n)` by a complex number :math:`c\in\mathbb{C}` will yield
    :math:`(cx)^\dagger=\overline{c} x^\dagger=-\overline{c} x`, so that
    the result might no longer be in the algebra! If we keep it to real scalars
    :math:`c\in\mathbb{R}` only, we have :math:`\overline{c}=c`, so that
    :math:`(cx)^\dagger=-cx` and we're fine.

Let us set up :math:`\mathfrak{su}(2)` in code. For this, we create a basis for traceless
Hermitian :math:`2\times 2` matrices, which is given by the Pauli operators.
Note that the algebra itself consists of *skew*-Hermitian matrices, but we will work
with the Hermitian counterparts as inputs.
We can check that :math:`\mathfrak{su}(2)` is closed under commutators, by
computing all nested commutators, the so-called *Lie closure*, and observing
that the closure is not larger than :math:`\mathfrak{su}(2)` itself.
Of course we could also check the closure manually for this small example.
"""

from itertools import product, combinations
import pennylane as qml
import numpy as np

su2 = [qml.X(0), qml.Y(0), qml.Z(0)]
print(f"su(2) is {len(su2)}-dimensional")

all_hermitian = all(qml.equal(qml.adjoint(op).simplify(), op) for op in su2)
print(f"The operators are all Hermitian: {all_hermitian}")

su2_lie_closed = qml.pauli.dla.lie_closure(su2)
print(f"The Lie closure of su(2) is {len(su2_lie_closed)}-dimensional.")

traces = [op.pauli_rep.trace() for op in su2]
print(f"All operators are traceless: {np.allclose(traces, 0.)}")

######################################################################
# We find that :math:`\mathfrak{su}(2)` indeed is closed, and that it is a 3-dimensional
# space. We also picked a correct representation with traceless operators.
#
# .. admonition:: Mathematical detail
#     :class: note
#
#     Our main result for this demo will be the KAK theorem, which applies to
#     so-called *semisimple* Lie algebras. We will not go into detail about this notion, but
#     it often is sufficient to think of them as the algebras that are composed from
#     three types of *simple* building blocks, namely
#     (1) special orthogonal algebras :math:`\mathfrak{so}(n)`, (2) unitary symplectic algebras
#     :math:`\mathfrak{sp}(n)`, and (3) special unitary algebras :math:`\mathfrak{su}(n)`.
#     In particular, our example here is of the latter type, so it is not only semisimple,
#     but even simple.
#
# Group and algebra interaction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The topic of Lie groups and Lie algebras is a large field of study and there are many
# things we could talk about in this section. For the sake of brevity, however, we will
# only list a few important properties that are needed further below. For more details
# and proofs, refer to your favourite Lie theory book, which might be #TODO
#
# The Lie group :math:`\mathcal{G}` associated to a Lie algebra :math:`\mathfrak{g}` is given
# by the exponential map applied to the algebra:
#
# .. math::
#
#     \exp : \mathfrak{g} \to \exp(\mathfrak{g})=\mathcal{G}, \ x\mapsto \exp(x)
#
# We will only consider Lie groups arising from Lie algebras here, which we denote by
# :math:`\exp(\mathfrak{g})` for an algebra :math:`\mathfrak{g}`.
# As we usually think about the unitary algebras :math:`\mathfrak{u}` and their
# subalgebras, the correspondence is well-known to quantum practitioners: Exponentiate
# a skew-Hermitian matrix to obtain a unitary operation, i.e., a quantum gate.
#
# We will make use of a particular interaction between the algebra :math:`\mathfrak{g}` and
# its group :math:`\mathcal{G}`, called the *adjoint action* of :math:`\mathcal{G}` on :math:`\mathfrak{g}`.
# It is given by
#
# .. math::
#
#     \text{Ad}: \mathcal{G} \times \mathfrak{g} \to \mathfrak{g},
#     \ (\exp(x),y)\mapsto \text{Ad}_{\exp(x)}(y) = \exp(x) y\exp(-x).
#
# Similarly, we can interpret the Lie bracket as a map of :math:`\mathfrak{g}` acting on itself,
# which is called the *adjoint representation* of :math:`\mathfrak{g}` on itself:
#
# .. math::
#
#     \text{ad}: \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g},
#     \ (x, y) \mapsto \text{ad}_x(y) = [x, y].
#
# The adjoint group action and adjoint algebra representation do not only carry a very
# similar name, they are intimately related:
#
# .. math::
#
#     \text{Ad}_{\exp(x)}(y) = \exp(\text{ad}_x) (y),
#
# where we applied the exponential map to :math:`\text{ad}_x` via its series representation.
# We will refer to this relationship as *adjoint identity*.
# We talk about Ad and ad in more detail in the box below, and refer to our tutorial on
# :doc:`g-sim: Lie algebraic classical simulations </demos/tutorial_liesim/>` for
# further discussion.
#
# .. admonition:: Derivation: Adjoint representations
#     :class: note
#
#     TODO
#     An important operation in :math:`\mathcal{G}` is the *adjoint action* on itself,
#     given by
#
#     .. math::
#
#         \text{Ad}: \mathcal{G} \times \mathcal{G} \to \mathcal{G},
#         \ (\exp(x),\exp(y))\mapsto \text{Ad}_{\exp(x)}(\exp(y)) = \exp(x) \exp(y)\exp(-x).
#
#     Consider a curve through :math:`\mathcal{G}` given by
#
# Subalgebras and Cartan decomposition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A *subalgebra* :math:`\mathfrak{k}` of a Lie algebra :math:`\mathfrak{g}` is a
# vector subspace that is closed under the Lie bracket. Overall, this means that
# :math:`\mathfrak{k}` is closed under addition, scalar multiplication, and the Lie bracket.
# The latter often is simply written as :math:`[\mathfrak{k}, \mathfrak{k}]\subset \mathfrak{k}`.
#
# The algebras we are interested in come with an *inner product* between its elements.
# For our purposes, it is sufficient to assume that it is
#
# .. math::
#
#     \langle x, y\rangle = \text{tr}[x^\dagger y].
#
# Let's implement the inner product and an orthogonality check based on it:
#


def inner_product(op1, op2):
    """Compute the trace inner product between two operators."""
    # Use two wires to reuse it in the second example on two qubits later on
    return qml.math.trace(qml.matrix(qml.adjoint(op1) @ op2, wire_order=[0, 1]))


def is_orthogonal(op, basis):
    """Check whether an operator is orthogonal to a space given by some basis."""
    return np.allclose([inner_product(op, basis_op) for basis_op in basis], 0)


######################################################################
# Given a subalgebra :math:`\mathfrak{k}\subset \mathfrak{g}`, the inner product allows
# us to define an orthogonal complement
#
# .. math::
#
#     \mathfrak{p} = \{x\in\mathfrak{g} | \langle x, y\rangle=0 \forall y\mathfrak{k}\}.
#
# In this context, :math:`\mathfrak{k}` is commonly called the *vertical space*,
# :math:`\mathfrak{p}` accordingly is the *horizontal space*.
# The KAK theorem will apply to scenarios in which these spaces satisfy additional
# commutation relations, which do not hold for all subalgebras:
#
# .. math::
#
#     [\mathfrak{k}, \mathfrak{p}] \subset& \mathfrak{p} \qquad \text{(Reductive property)}\\
#     [\mathfrak{p}, \mathfrak{p}] \subset& \mathfrak{k} \qquad \text{(Symmetric property)}.
#
# The first property tells us that :math:`\mathfrak{p}` is left intact by the adjoint action of
# :math:`\mathfrak{k}` and that :math:`\mathfrak{p}` behaves like the "opposite" of a subalgebra, i.e.,
# all commutators lie in its complement, the subalgebra :math:`\mathfrak{k}`.
# Due to the adjoint identity from above, the first property also holds for group elements acting on
# algebra elements; For all :math:`x\in\mathfrak{p}` and :math:`y\in\mathfrak{k}`, we have
#
# .. math::
#
#     K x K^\dagger
#     = \exp(y) x \exp(-y)
#     = \exp(\text{ad}_y) (x)
#     = \sum_{n=0}^\infty \frac{1}{n!} \underset{\in\mathfrak{p}}{\underbrace{(\text{ad}_y)^n (x)}}
#     \in \mathfrak{p}.
#
# If the reductive property holds, the quotient space :math:`G/K` of the groups
# of :math:`\mathfrak{g}` and :math:`\mathfrak{k}` is called a *reductive homogeneous space*.
# If both properties hold, :math:`(\mathfrak{k}, \mathfrak{p})` is called a
# *Cartan pair* and we call :math:`\mathfrak{g}=\mathfrak{k} \oplus \mathfrak{p}` a *Cartan decomposition*.
# :math:`(\mathfrak{g}, \mathfrak{k})` is named a *symmetric pair*
# and the quotient :math:`G/K` is a *symmetric space*.
# Symmetric spaces are relevant for a wide range of applications in physics
# and have been studied a lot throughout the last hundred years.
#
# .. admonition:: Nomenclature
#     :class: warning
#
#     Depending on context and field, there are sometimes additional requirements
#     for :math:`\mathfrak{g}=\mathfrak{k}\oplus\mathfrak{p}` to be called a Cartan decomposition.
#
# **Example**
#
# For our example, we consider the subalgebra :math:`\mathfrak{k}=\mathfrak{u}(1)`
# of :math:`\mathfrak{su}(2)` generating Pauli-Z rotations:
#
# .. math::
#
#     \mathfrak{k} = \mathbb{R} iZ.
#
# Let us define it in code, and check whether it gives rise to a Cartan decomposition.
# As we want to look at another example later, we wrap everything in a function.
#


def check_cartan_decomposition(g, k, space_name):
    """Given an algebra g and an operator subspace k, verify that k is a subalgebra
    and gives rise to a Cartan decomposition."""
    # Check Lie closure of k
    k_lie_closed = qml.pauli.dla.lie_closure(k)
    print(
        f"The Lie closure of k is as large as k itself: {len(k_lie_closed)==len(k)}."
    )

    # Orthogonal complement of k, assuming that everything is given in the same basis.
    p = [g_op for g_op in g if is_orthogonal(g_op, k)]
    print(
        f"k has dimension {len(k)}, p has dimension {len(p)}, which combine to "
        f"the dimension {len(su2)} of su(2): {len(k)+len(p)==len(su2)}"
    )

    # Check reductive property
    k_p_commutators = [qml.commutator(k_op, p_op) for k_op, p_op in product(k, p)]
    k_p_coms_in_p = [is_orthogonal(com, k) for com in k_p_commutators]

    print(
        f"All commutators in [k, p] are in p (orthogonal to k): {all(k_p_coms_in_p)}."
    )
    if all(k_p_coms_in_p):
        print(f"{space_name} is a reductive homogeneous space.")

    # Check symmetric property
    p_p_commutators = [
        qml.commutator(p_op, p_op) for p_op, p_op in combinations(p, r=2)
    ]
    p_p_coms_in_k = [is_orthogonal(com, p) for com in p_p_commutators]

    print(
        f"All commutators in [p, p] are in k (orthogonal to p): {all(p_p_coms_in_k)}."
    )
    if all(p_p_coms_in_k):
        print(f"{space_name} is a symmetric space.")

    return p


u1 = [qml.Z(0)]
space_name = "SU(2)/U(1)"
p = check_cartan_decomposition(su2, u1, space_name)

"""
# Subalgebra
k = [qml.X(0), qml.Y(0), qml.Z(0), qml.X(1), qml.Y(1), qml.Z(1)]
# Check Lie closure of k
k_lie_closed = qml.pauli.dla.lie_closure(k)
print(f"The Lie closure of k is as large as k itself: {len(k_lie_closed)==len(k)}.")

# Orthogonal complement of k
p = [op for op in su4 if np.allclose([inner_product(op, k_op) for k_op in k], 0)]
print(
    f"k has dimension {len(k)}, p has dimension {len(p)}, which combine to "
    f"the dimension {len(su4)} of su(4): {len(k)+len(p)==len(su4)}"
)

# Check reductive property
k_p_commutators = [
    k_op.pauli_rep.commutator(p_op.pauli_rep) for k_op, p_op in product(k, p)
]
k_p_coms_in_p = [
    np.allclose([inner_product(com, k_op) for k_op in k], 0)
    for com in k_p_commutators
]
print(f"All commutators in [k, p] are in p (orthogonal to k): {all(k_p_coms_in_p)}.")
if all(k_p_coms_in_p):
    print("SU(4)/(SU(2)xSU(2)) is a reductive homogeneous space.")

# Check symmetric property
p_p_commutators = [
    op1.pauli_rep.commutator(op2.pauli_rep) for op1, op2 in combinations(p, r=2)
]
p_p_coms_in_k = [
    np.allclose([inner_product(com, p_op) for p_op in p], 0)
    for com in p_p_commutators
]
print(f"All commutators in [p, p] are in k (orthogonal to p): {all(p_p_coms_in_k)}.")
if all(p_p_coms_in_k):
    print("SU(4)/(SU(2)xSU(2)) is a symmetric space.")
"""

######################################################################
# Cartan subalgebras
# ~~~~~~~~~~~~~~~~~~
#
# The symmetric property of a Cartan decomposition (:math:`[\mathfrak{p}, \mathfrak{p}]\subset\mathfrak{k}`)
# tells us that :math:`\mathfrak{p}` is very
# far from being a subalgebra. This also gives us information about potential subalgebras
# *within* :math:`\ \mathfrak{p}`. Assume we have a subalgebra :math:`\mathfrak{a}\subset\mathfrak{p}`. Then the commutator
# between any two elements :math:`x, y\in\mathfrak{a}` must satisfy
#
# .. math::
#
#     [x, y] \in \mathfrak{a} \subset \mathfrak{p} &\Rightarrow [x, y]\in\mathfrak{p} \text{(subalgebra property)} \\
#     [x, y] \in [\mathfrak{a}, \mathfrak{a}] \subset [\mathfrak{p}, \mathfrak{p}]\subset \mathfrak{k} &\Rightarrow [x, y]\in\mathfrak{k}\ \text{(symmetric property)}.
#
# That is, the commutator must lie in both orthogonal complements :math:`\mathfrak{k}` and :math:`\mathfrak{p}`,
# which only have the zero vector in common. This tells us that *all* commutators in :math:`\mathfrak{a}`
# vanish, making it an *Abelian* subalgebra:
#
# .. math::
#
#     [\mathfrak{a}, \mathfrak{a}] = \{0\}.
#
# Such an Abelian subalgebra is a (horizontal) *Cartan subalgebra (CSA)* if it is *maximal*,
# i.e., if it can not be made any larger (higher-dimensional) without leaving :math:`\mathfrak{p}`.
#
# .. admonition:: Nomenclature
#     :class: warning
#
#     Depending on context and field, there are inequivalent notions of Cartan subalgebras.
#     In particular, there is a common notion of Cartan subalgebras which are not contained
#     in a horizontal space. Throughout this demo, we always mean a *horizontal*
#     maximal Abelian subalgebra :math:`\mathfrak{a}\subset\mathfrak{p}`.
#
# How many different CSAs are there? Given a CSA :math:`\mathfrak{a}`, we can pick a vertical element
# :math:`y\in\mathfrak{k}` and apply the corresponding group element :math:`K=\exp(y)` to
# all elements of the CSA, using the adjoint action we studied above.
# This will yield a valid CSA again. First, :math:`K\mathfrak{a} K^\dagger` remains in :math:`\mathfrak{p}`
# due to the reductive property, as we discussed when introducing the Cartan decomposition.
# Second the adjoint action will not change the Abelian property because
#
# .. math::
#
#     [K x_1 K^\dagger, K x_2 K^\dagger] = K [x_1, x_2] K^\dagger = 0 \quad \forall\ x_{1, 2}\in\mathfrak{a}.
#
# Finally, we are guaranteed that :math:`K\mathfrak{a} K^\dagger` remains maximal.
#
# .. admonition:: Mathematical detail
#     :class: note
#
#     The reason that :math:`K\mathfrak{a} K^\dagger` is maximal if :math:`\mathfrak{a}` was, is
#     that we assume :math:`\mathfrak{g}` to be a semisimple Lie algebra, for which the
#     adjoint representation is faithful. This in turn implies that linearly
#     independent elements of :math:`\mathfrak{g}` will not be mapped to linearly dependent
#     elements by the adjoint action of :math:`K`.
#
# For most :math:`y\in\mathfrak{k}`, applying :math:`K=\exp(y)` in this way will yield a
# *different* CSA, so that we find a whole continuum of them.
# It turns out that they *all* can be found by starting with *any*
# :math:`\mathfrak{a}` and applying all of :math:`\exp(\mathfrak{k})` to it.
#
# *This is what powers the KAK theorem.*
#
# **Example**
#
# For our example, we established the decomposition :math:`\mathfrak{su}(2)=\mathfrak{u}(1)\oplus \mathfrak{p}`
# with the two-dimensional horizontal space :math:`\mathfrak{p} = \text{span}_{\mathbb{R}}\{iX, iY\}`.
# Starting with the subspace :math:`\mathfrak{a}=\mathbb{R} iY`, we see that we immediately
# reach a maximal Abelian subalgebra, i.e., a CSA,
# because :math:`[Y, X]\neq 0`. Applying a rotation :math:`\exp(i\eta Z)` to this CSA gives us a new CSA via
#
# .. math::
#
#     \mathfrak{a}'=\exp(i\eta Z) (\mathbb{R} iY) \exp(-i\eta Z) = \mathbb{R} (\cos(2\eta) iY + \sin(2\eta) iX).
#
# The vertical group element :math:`\exp(i\eta Z)` simply rotates the CSA within :math:`\mathfrak{p}!`
# Let us not forget to define the CSA in code.

# CSA generator: iY
a = p[1]

# Rotate CSA by applying vertical group element
eta = 0.6
# The factor -2 compensates the convention -1/2 in the RZ gate
# TODO
# a_prime = qml.simplify(qml.prod(qml.RZ(-2 * eta, 0), a, qml.RZ(2 * eta, 0)))
# a_prime = qml.RZ(-2 * eta, 0).pauli_rep @ a.pauli_rep
# print(f"The rotated CSA a' is generated by {a_prime}")

a_prime_expected = np.cos(2 * eta) * a + np.sin(2 * eta) * p[0]
# a_primes_equal = qml.equal(a_prime_expected, a_prime)
# print(f"This matches the expected rotated CSA from theory: {a_primes_equal}")

######################################################################
# Involutions
# ~~~~~~~~~~~
#
# In practice, there often is a more convenient way to a Cartan decomposition
# than by specifying the subalgebra :math:`\mathfrak{k}` or its horizontal counterpart
# :math:`\mathfrak{p}` manually. It goes as follows.
#
# We will look at a map :math:`\theta` from the total Lie algebra :math:`\mathfrak{g}`
# to itself. We demand that :math:`\theta` has the following properties, for
# :math:`x, y\in\mathfrak{g}` and :math:`c\in\mathbb{R}`.
#
# #. It is linear, i.e., :math:`\theta(x + cy)=\theta(x) +c \theta(y)`
# #. It is compatible with the commutator, i.e., :math:`\theta([x, y])=[\theta(x),\theta(y)]`, and
# #. It is an *involution*, i.e., :math:`\theta(\theta(x)) = x`.
#
# Put compactly, we demand that :math:`\theta` be an *involutive automorphism* of :math:`\mathfrak{g}`.
# As an involution, :math:`\theta` only can have the eigenvalues :math:`\pm 1`, with associated
# eigenspaces :math:`\mathfrak{g}_\pm`. Let's see what happens when we compute commutators between
# elements :math:`x_\pm\in\mathfrak{g}_\pm`:
#
# .. math::
#
#     \theta([x_+, x_+]) = [\theta(x_+), \theta(x_+)] = [x_+, x_+] &\ \Rightarrow\ [x_+, x_+]\in\mathfrak{g}_+\\
#     \theta([x_+, x_-]) = [\theta(x_+), \theta(x_-)] = -[x_+, x_-] &\ \Rightarrow\ [x_+, x_-]\in\mathfrak{g}_-\\
#     \theta([x_-, x_-]) = [\theta(x_-), \theta(x_-)] = (-1)^2 [x_-, x_-] &\ \Rightarrow\ [x_-, x_-]\in\mathfrak{g}_+.
#
# Or, in other words, :math:`[\mathfrak{g}_+, \mathfrak{g}_+] \subset \mathfrak{g}_+`, :math:`[\mathfrak{g}_+, \mathfrak{g}_-] \subset \mathfrak{g}_-`,
# and :math:`[\mathfrak{g}_-, \mathfrak{g}_-] \subset \mathfrak{g}_+`.
# So an involution is enough to find us a Cartan decomposition, with :math:`\mathfrak{k}=\mathfrak{g}_+`
# and :math:`\mathfrak{p}=\mathfrak{g}_-`.
#
# 🤯
#
# We might want to call such a :math:`\theta` a *Cartan involution*.
#
# .. admonition:: Nomenclature
#     :class: warning
#
#     Some people do so, some people again require more properties for such an
#     involution to be called Cartan involution.
#     For our purposes, let's go with the more general definition and call all
#     involutions with the properties above Cartan involution.
#
# Conversely, if we have a Cartan decomposition based on a subalgebra :math:`\mathfrak{k}`,
# we can define the map
#
# .. math::
#
#     \theta_{\mathfrak{k}}(x) = \Pi_{\mathfrak{k}}(x)-\Pi_{\mathfrak{p}}(x),
#
# where :math:`\Pi` are the projectors onto the two vector subspaces.
# Clearly, :math:`\theta_{\mathfrak{k}}` is linear because projectors are.
# It is also compatible with the commutator due to the commutation relations
# between :math:`\mathfrak{k}` and :math:`\mathfrak{p}` (see box).
#
# .. admonition:: Mathematical detail
#     :class: note
#
#     To see that :math:`\theta_{\mathfrak{k}}` is compatible with the commutator, we compute
#     the action of the projectors on it:
#
#     .. math::
#
#         \Pi_{\mathfrak{k}}([x, y])
#         &= \Pi_{\mathfrak{k}}([\Pi_{\mathfrak{k}}(x) + \Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y) + \Pi_{\mathfrak{p}}(y) \\
#         &= \Pi_{\mathfrak{k}}(\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)]}})
#         \Pi_{\mathfrak{k}}(\underset{\in \mathfrak{p}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{p}}(y)]}})
#         \Pi_{\mathfrak{k}}(\underset{\in \mathfrak{p}}{\underbrace{[\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y)]}})
#         \Pi_{\mathfrak{k}}(\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{p}}(y)]}})\\
#         &= [\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)] + [\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{p}}(y)]\\
#         \Pi_{\mathfrak{p}}([x, y])
#         &= \Pi_{\mathfrak{p}}([\Pi_{\mathfrak{k}}(x) + \Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y) + \Pi_{\mathfrak{p}}(y) \\
#         &= \Pi_{\mathfrak{p}}(\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)]}})
#         \Pi_{\mathfrak{p}}(\underset{\in \mathfrak{p}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{p}}(y)]}})
#         \Pi_{\mathfrak{p}}(\underset{\in \mathfrak{p}}{\underbrace{[\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y)]}})
#         \Pi_{\mathfrak{p}}(\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{p}}(y)]}})\\
#         &= [\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{p}}(y)] + [\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y)]
#
#     Here we used :math:`\mathbb{I}_{\mathfrak{g}} = \Pi_{\mathfrak{k}} + \Pi_{\mathfrak{p}}` and the
#     commutation relations between :math:`\mathfrak{k}` and :math:`\mathfrak{p}`.
#
#     We can put thes pieces together to get
#
#     .. math::
#
#         \theta_{\mathfrak{k}} ([x, y])
#         &=\Pi_{\mathfrak{k}}([x, y]) - \Pi_{\mathfrak{p}}([x, y])\\
#         &=[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)] + [\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{p}}(y)]
#         - [\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{p}}(y)] - [\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y)]\\
#         &=[\Pi_{\mathfrak{k}}(x) -\Pi_{\mathfrak{p}}(x), \Pi_{\mathfrak{k}}(y)-\Pi_{\mathfrak{p}}(y)]\\
#         &=[\theta_{\mathfrak{k}} (x),\theta_{\mathfrak{k}} (y)].
#
# Finally, :math:`\theta_{\mathfrak{k}}` is an involution because
#
# .. math::
#
#     \theta_{\mathfrak{k}}^2=(\Pi_{\mathfrak{k}}-\Pi_{\mathfrak{p}})^2
#     = \Pi_{\mathfrak{k}}^2-\Pi_{\mathfrak{k}}\Pi_{\mathfrak{p}}-\Pi_{\mathfrak{p}}\Pi_{\mathfrak{k}}+\Pi_{\mathfrak{p}}^2
#     =\Pi_{\mathfrak{k}}-\Pi_{\mathfrak{p}}
#     = \mathbb{I}_{\mathfrak{g}},
#
# where we used the projector's properties.
#
# This shows us that we can easily switch between a Cartan involution and a Cartan
# decomposition, in either direction!
#
# **Example**
#
# In our example, an involution that reproduces our choice :math:`\mathfrak{k}=\mathbb{R} iZ` is
# :math:`\theta_Z(x) = Z x Z` (Convince yourself that it is an involution that respects commutators).


def theta_Z(x):
    return qml.simplify(qml.Z(0) @ x @ qml.Z(0))


theta_of_u1 = [theta_Z(x) for x in u1]
u1_is_su2_plus = all(
    qml.equal(x, theta_of_x) for x, theta_of_x in zip(u1, theta_of_u1)
)
print(f"U(1) is the +1 eigenspace: {u1_is_su2_plus}")

theta_of_p = [theta_Z(x) for x in p]
p_is_su2_minus = all(
    qml.equal(-x, theta_of_x) for x, theta_of_x in zip(p, theta_of_p)
)
print(f"p is the -1 eigenspace: {p_is_su2_minus}")

######################################################################
#
# We can easily get a new subalgebra by modifying the involution, say, to :math:`\theta_Y(x) = Y x Y`,
# expecting that :math:`k_Y=\mathbb{R} iY` becomes the new subalgebra.


def theta_Y(x):
    return qml.simplify(qml.Y(0) @ x @ qml.Y(0))


eigvals = []
for x in su2:
    if qml.equal(theta_Y(x), x):
        eigvals.append(1)
    elif qml.equal(theta_Y(x), -x):
        eigvals.append(-1)
    else:
        raise ValueError("Operator not purely in either eigenspace.")

print(
    f"Under the involution theta_Y, the operators\n{su2}\nhave the eigenvalues\n{eigvals}"
)

######################################################################
# This worked! a new involution gave us a new subalgebra and Cartan decomposition.
#
# .. adminition:: Mathematical detail
#     :class: note
#
#     You might already see that the two different decompositions created by :math:`\theta_Z`
#     and :math:`\theta_Y` are very similar. There is a whole field of study
#     characterizing---and even fully classifying---the possible Cartan decompositions
#     of semisimple Lie algebras. We will not go into detail here, but this classification
#     plays a big role when talking about decompositions without getting stuck on details
#     like the choice of basis or the representation of the algebra.
#
# KAK theorem
# ~~~~~~~~~~~
#
# Now that we covered all prerequisites, we are ready for our main result. It consists of two steps
# that are good to know my themselves, so we will look at both of them in sequence. We will not conduct
# formal proofs but leave those to the literature references.
# In the following, let :math:`\mathfrak{g}` be a compact real semisimple Lie algebra and :math:`\mathfrak{k}`
# a subalgebra such that :math:`\mathfrak{g}=\mathfrak{k}\oplus \mathfrak{p}` is a Cartan decomposition.
#
# The first step is a decomposition of the Lie group :math:`\mathcal{G}=\exp(\mathfrak{g})` into the Lie subgroup
# :math:`\mathcal{K}=\exp(\mathfrak{k})` and the exponential of the horizontal space, :math:`\mathcal{P}=\exp(\mathfrak{p})`,
# *which is not a group*. The decomposition is a simple product within :math:`\mathcal{G}`:
#
# .. math::
#
#     \mathcal{G} = \mathcal{K}\mathcal{P}, \text{ or }\ \forall\ G\in\mathcal{G} \exists K\in\mathcal{K}, x\in\mathcal{m}: G = K \exp(x)
#
# This "KP" decomposition can be seen as the "group version" of
# :math:`\mathfrak{g} = \mathfrak{k} \oplus\mathfrak{p}`.
#
# The second step is the further decomposition of the space :math:`\mathcal{P}=\exp(\mathfrak{p})`.
# We start by fixing a Cartan subalgebra (CSA) :math:`\mathfrak{a}\subset\mathfrak{p}`.
# Given a horizontal vector :math:`x\in\mathfrak{p}`, we can construct a second CSA
# :math:`\mathfrak{a}_x\subset\mathfrak{p}` that contains :math:`x`. Now, recall that for any two
# CSAs there is a subalgebra element :math:`y\in\mathfrak{k}` such that the adjoint action of
# :math:`\exp(y)` maps one CSA to the other. In particular, there is a :math:`y\in\mathfrak{k}`
# so that
# .. math::
#
#     \exp(y)\mathfrak{a}_x\exp(-y)=\mathfrak{a}\quad\Rightarrow\quad x\in(\exp(-y) \mathfrak{a}\exp(y).
#
# Generalizing this statement across all horizontal elements :math:`x\in\mathfrak{p}`, we find
#
# .. math::
#
#     \mathfrak{p} \subset \{\exp(-y) \mathfrak{a} \exp(y) | y\in\mathfrak{k}\}.
#
# As we discussed, the converse inclusion also must hold for a reductive space, so that we may even replace
# :math:`\subset` by an equality.
# Now we can use :math:`\exp(\text{Ad}_{\exp(-y)} x)=\text{Ad}_{\exp(-y)}\exp(x)` to move this statement
# to the group level,
#
# .. math::
#
#     \mathcal{P}
#     = \{\exp(\exp(-y) \mathfrak{a} \exp(y)) | y\in\mathfrak{k}\}
#     = \{K^{-1} \mathcal{A} K | K\in\mathcal{K}\},
#
# where we abbreviated :math:`\mathcal{A} = \exp(\mathfrak{a})`.
#
# Chaining the two steps together and combining the left factor :math:`K^{-1}` with the group
# :math:`\mathcal{K}` in the "KP" decomposition, we obtain the *KAK theorem*
#
# .. math::
#
#     \mathcal{G} &= \mathcal{K} \mathcal{A} \mathcal{K} \quad\textbf{(KAK Theorem).}
#
# It teaches us that any group element can be decomposed into two factors from the Lie subgroup and
# the exponential of a CSA element, i.e., of commuting elements from the horizontal subspace
# :math:`\mathfrak{p}`. This may already hint at the usefulness of the KAK theorem for matrix
# factorizations in general, and for quantum circuit decompositions in particular.
#
# **Example**
#
# Applying what we just learned to our example on :math:`\mathfrak{su}(2)`, we can state that
# any single-qubit gate can be implemented by running a gate from
# :math:`\mathcal{K}=\{\exp(i\eta Z) | \eta\in\mathbb{R}\}`, a CSA gate
# :math:`\mathcal{A}=\{\exp(i\varphi Y) | \eta\in\mathbb{R}\}`, and another gate from :math:`\mathcal{K}`.
# We rediscovered a standard decomposition of an arbitrary :class:`~.pennylane.Rot` gate!

print(qml.Rot(0.5, 0.2, -1.6, wires=0).decomposition())

######################################################################
# Other choices for involutions or---equivalently---subalgebras :math:`\mathfrak{k}` will
# lead to other decompositions of ``Rot``. For example, using :math:`\theta_Y` from above
# together with the CSA :math:`\mathfrak{a_Y}=\mathbb{R} iX`, we find the decomposition
#
# .. math::
#
#     \text{Rot}(\phi, \theta, \omega) = R_Y(\eta_1) R_X(\vartheta) R_Y(\eta_2).
#
# And that's it for our main discussion. We conclude this demo by applying the
# KAK theorem to the group of arbitrary two-qubit gates.
#
# Two-qubit KAK decomposition
# ---------------------------
#
# - Algebra/subalgebra :math:`\mathfrak{g} =\mathfrak{su}(4) | \mathfrak{k} =\mathfrak{su}(2) \oplus \mathfrak{su}(2)`
# - Involution: EvenOdd
# - CSA: :math:`\mathfrak{a} = \langle\{XX, YY, ZZ\}\rangle_{i\mathbb{R}}`
# - KAK decomposition :math:`U= (A\otimes B) \exp(i(\eta_x XX+\eta_y YY +\eta_z ZZ)) (C\otimes D)`.
# - [optional] Mention Cartan coordinates

######################################################################
# Conclusion
# ----------
#
# In this demo we learned about the KAK theorem and how it uses a Cartan
# decomposition of a Lie algebra to decompose its Lie group.
# This allows us to break down arbitrary quantum gates from that group,
# as we implemented in code for the group of two-qubit gates :math:`SU(4)`.
#
# If you are interested in other applications of Lie theory in the field of
# quantum computing, you are in luck! It has been a handy tool throughout the last
# decades, e.g., for the simulation and compression of quantum circuits, # TODO: REFS
# in quantum optimal control, and for trainability analyses. For Lie algebraic
# classical simulation of quantum circuits, check the
# :doc:`g-sim </demos/tutorial_liesim/>` and
# :doc:`(g+P)-sim </demos/tutorial_liesim_extension/>` demos, and stay posted for
# a brand new demo on compiling Hamiltonian simulation circuits with the KAK theorem!
#
# References
# ----------
#
# .. [#khaneja_glaser]
#
#     Navin Khaneja, Steffen Glaser
#     "Cartan decomposition of SU(2^n), constructive controllability of spin systems and universal quantum computing"
#     `arXiv:quant-ph/0010100 <https://arxiv.org/abs/quant-ph/0010100>`__, 2000
#
# About the author
# ----------------
