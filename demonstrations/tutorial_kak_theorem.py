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

.. note::

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

Our working example in this demo will be the *special unitary* algebra :math:`\mathfrak{su}(4)`.
It consists of traceless complex-valued skew-Hermitian :math:`4\times 4` matrices, i.e.,

.. math::

    \mathfrak{u}(4) = \left\{x \in \mathbb{C}^{(4\times 4)} | x^\dagger = -x , \text{tr}[x]=0\right\}.

.. note::

    :math:`\mathfrak{su}(4)` is a *real* Lie algebra, i.e., it is a vector space over the
    real numbers :math:`\mathbb{R}`. This means that scalar-vector multiplication is
    only valid between vectors (complex-valued matrices) and real scalars.

    There is a simple reason to see this; Multiplying a skew-Hermitian matrix
    :math:`x\in\mathfrak{u}(4)` by a complex number :math:`c\in\mathbb{C}` will yield
    :math:`(cx)^\dagger=\overline{c} x^\dagger=-\overline{c} x`, so that
    the result might no longer be in the algebra! If we keep it to real scalars
    :math:`c\in\mathbb{R}` only, we have :math:`\overline{c}=c`, so that
    :math:`(cx)^\dagger=-cx` and we're fine.

Let us set up :math:`\mathfrak{su}(4)` in code. For this, we create a basis for traceless
Hermitian :math:`4\times 4` matrices, which is given by the Pauli basis on two qubits.
Note that the algebra itself consists of *skew-*Hermitian matrices, but we will work
with the Hermitian counterparts as inputs.
We can check that :math:`\mathfrak{su}(4)` is closed under commutators, by
computing all nested commutators, the so-called *Lie closure*, and observing
that the closure is not larger than :math:`\mathfrak{su}(4)` itself.
"""

from itertools import product, combinations
import pennylane as qml
import numpy as np

su4 = [op for op in qml.pauli.pauli_group(2)][1:]
print(f"su(4) is {len(su4)}-dimensional")

all_skew_hermitian = all(qml.equal(qml.adjoint(op).simplify(), op) for op in su4)
print(f"The operators are all Hermitian: {all_skew_hermitian}")

su4_lie_closed = qml.pauli.dla.lie_closure(su4)
print(f"The Lie closure of su(4) is {len(su4_lie_closed)}-dimensional.")

traces = [op.pauli_rep.trace() for op in su4]
print(f"All operators are traceless: {np.allclose(traces, 0.)}")

######################################################################
# We find that :math:`\mathfrak{su}(4)` indeed is closed, and that it is a 15-dimensional
# space. We also picked a correct representation with traceless operators.
#
# .. note::
#
#     Our main result for this demo will be the KAK theorem, which applies to
#     so-called *semisimple* Lie algebras. We will not go into detail about this notion, but
#     it often is sufficient to think of them as the algebras that are composed from
#     three types of *simple* building blocks, namely
#     (1) special orthogonal algebras :math:`\mathfrak{so}(n)`, (2) unitary symplectic algebras
#     :math:`\mathfrak{sp}(n)`, and (3) special unitary algebras :math:`\mathfrak{su}(n)`.
#     In particular, the unitary :math:`\mathfrak{u}(4)` is not *quite* semisimple;
#     it contains a *center*, which is a non-simple component. This is why we here will be
#     studying the *special* unitary algebra :math:`\mathfrak{su}(4)`.
#     Fortunately, the center of :math:`\mathfrak{u}(4)` is encoded in the trace of
#     the operators and generates global phases when used in a quantum gate. As we
#     usually can discard global phases anyways, :math:`\mathfrak{su}(4)` is indeed the
#     algebra we care about.
#
# - [optional] In particular mention that the adjoint representation is faithful for semisimple algebras.
#
# Group and algebra interaction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# - Exponential map
# - adjoint action of group on algebra
# - adjoint action of algebra on algebra -> adjoint representation
# - adjoint identity (-> g-sim demo)
#
# Subalgebras and Cartan decomposition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A *subalgebra* :math:`\mathfrak{k}` of a Lie algebra :math:`\mathfrak{g}` is a
# vector subspace that is closed under the Lie bracket. Overall, this means that
# :math:`\mathfrak{k}` is closed under addition, scalar multiplication, and the Lie bracket.
# The latter often is simply written as :math:`[\mathfrak{k}, \mathfrak{k}]\subset \mathfrak{k}`.
#
# The algebras we are interested in come with an inner product between its elements.
# For our purposes, it is sufficient to assume that it is
#
# .. math::
#
#     \langle x, y\rangle = \text{tr}[x^\dagger y].
#
# Given a subalgebra :math:`\mathfrak{k}\subset \mathfrak{g}`, the inner product allows
# us to define an orthogonal complement
#
# .. math::
#
#     \mathfrak{m} = \{x\in\mathfrak{g} | \langle x, y\rangle=0 \forall y\mathfrak{k}\}.
#
# In this context, :math:`\mathfrak{k}` is commonly called the *vertical space*,
# :math:`\mathfrak{m}` accordingly is the *horizontal space*.
# The KAK theorem will apply to scenarios in which these spaces satisfy additional
# commutation relations, which do not hold for all subalgebras:
#
# .. math::
#
#     [\mathfrak{k}, \mathfrak{m}] \subset& \mathfrak{m} \qquad \text{(Reductive property)}\\
#     [\mathfrak{m}, \mathfrak{m}] \subset& \mathfrak{k} \qquad \text{(Symmetric property)}.
#
# The first property tells us that :math:`\mathfrak{m}` is left intact by the adjoint action of
# :math:`\mathfrak{k}` and that :math:`\mathfrak{m}` behaves like the "opposite" of a subalgebra, i.e.,
# all commutators lie in its complement, the subalgebra :math:`\mathfrak{k}`.
# Due to the adjoint identity from above, the first property also holds for group elements acting on
# algebra elements:
#
# .. math::
#
#     K x K^\dagger = \exp(y) x \exp(-y) \in \mathfrak{m} \quad \forall\ x\in\mathfrak{m}, y\in\mathfrak{k},
#
# .. note::
#
#     If the reductive property holds, the quotient space :math:`G/K` of the groups
#     of :math:`\mathfrak{g}` and :math:`\mathfrak{k}` is called a *reductive homogeneous space*.
#     If both properties hold, :math:`(\mathfrak{k}, \mathfrak{m})` is called a
#     *Cartan pair* and we call :math:`\mathfrak{g}=\mathfrak{k} \oplus \mathfrak{m}` a *Cartan decomposition*.
#     :math:`(\mathfrak{g}, \mathfrak{k})` is named a *symmetric pair*
#     and the quotient :math:`G/K` is a *symmetric space*.
#     Symmetric spaces are relevant for a wide range of applications in physics
#     and have been studied a lot throughout the last hundred years.
#
# .. warning::
#
#     Depending on context and field, there are sometimes additional requirements
#     for :math:`\mathfrak{g}=\mathfrak{k}\oplus\mathfrak{m}` to be called a Cartan decomposition.
#
# For our example, we consider the subalgebra :math:`\mathfrak{k}=\mathfrak{su}(2)\oplus\mathfrak{su}(2)`
# of :math:`\mathfrak{su}(4)`, consisting of independent single-qubit operations acting
# on either of the two qubits. Concretely,
#
# .. math::
#
#     \mathfrak{k} = \text{span}_{i\mathbb{R}}\left\{X_0, Y_0, Z_0, X_1, Y_1, Z_1\right\}.
#
# Let us define it in code, and check whether it gives rise to a Cartan decomposition.
#


def inner_product(op1, op2):
    mat1 = qml.matrix(op1, wire_order=[0, 1])
    mat2 = qml.matrix(op2, wire_order=[0, 1])
    return qml.math.trace(mat1.conj().T @ mat2)


# Subalgebra
k = [qml.X(0), qml.Y(0), qml.Z(0), qml.X(1), qml.Y(1), qml.Z(1)]
# Check Lie closure of k
k_lie_closed = qml.pauli.dla.lie_closure(k)
print(f"The Lie closure of k is as large as k itself: {len(k_lie_closed)==len(k)}.")

# Orthogonal complement of k
m = [op for op in su4 if np.allclose([inner_product(op, k_op) for k_op in k], 0)]
print(
    f"k has dimension {len(k)}, m has dimension {len(m)}, which combine to "
    f"the dimension {len(su4)} of su(4): {len(k)+len(m)==len(su4)}"
)

# Check reductive property
k_m_commutators = [
    k_op.pauli_rep.commutator(m_op.pauli_rep) for k_op, m_op in product(k, m)
]
k_m_coms_in_m = [
    np.allclose([inner_product(com, k_op) for k_op in k], 0)
    for com in k_m_commutators
]
print(f"All commutators in [k, m] are in m (orthogonal to k): {all(k_m_coms_in_m)}.")
if all(k_m_coms_in_m):
    print("SU(4)/(SU(2)xSU(2)) is a reductive homogeneous space.")

# Check symmetric property
m_m_commutators = [
    op1.pauli_rep.commutator(op2.pauli_rep) for op1, op2 in combinations(m, r=2)
]
m_m_coms_in_k = [
    np.allclose([inner_product(com, m_op) for m_op in m], 0)
    for com in m_m_commutators
]
print(f"All commutators in [m, m] are in k (orthogonal to m): {all(m_m_coms_in_k)}.")
if all(m_m_coms_in_k):
    print("SU(4)/(SU(2)xSU(2)) is a symmetric space.")

######################################################################
# Cartan subalgebras
# ~~~~~~~~~~~~~~~~~~
#
# The symmetric property of a Cartan decomposition (:math:`[\mathfrak{m}, \mathfrak{m}]\subset\mathfrak{k}`)
# tells us that :math:`\mathfrak{m}` is very
# far from being a subalgebra. This also gives us information about potential subalgebras
# *within* :math:`\ \mathfrak{m}`. Assume we have a subalgebra :math:`\mathfrak{h}\subset\mathfrak{m}`. Then the commutator
# between any two elements :math:`x, y\in\mathfrak{h}` must satisfy
#
# .. math::
#
#     [x, y] \in \mathfrak{h} \subset \mathfrak{m} &\Rightarrow [x, y]\in\mathfrak{m} \text{(subalgebra property)} \\
#     [x, y] \in [\mathfrak{h}, \mathfrak{h}] \subset [\mathfrak{m}, \mathfrak{m}]\subset \mathfrak{k} &\Rightarrow [x, y]\in\mathfrak{k}\ \text{(symmetric property)}.
#
# That is, the commutator must lie in both orthogonal complements :math:`\mathfrak{k}` and :math:`\mathfrak{m}`,
# which only have the zero vector in common. This tells us that *all* commutators in :math:`\mathfrak{h}`
# vanish, making it an *Abelian* subalgebra:
#
# .. math::
#
#     [\mathfrak{h}, \mathfrak{h}] = \{0\}.
#
# Such an Abelian subalgebra is a (horizontal) *Cartan subalgebra (CSA)* if it is *maximal*,
# i.e., if it can not be made any larger (higher-dimensional) without leaving :math:`\mathfrak{m}`.
#
# .. warning::
#
#     Depending on context and field, there are inequivalent notions of Cartan subalgebras.
#     In particular, there is a common notion of Cartan subalgebras which are not contained
#     in a horizontal space. Throughout this demo, we always mean a *horizontal*
#     maximal Abelian subalgebra :math:`\mathfrak{h}\subset\mathfrak{m}`.
#
# How many different CSAs are there? Given a CSA :math:`\mathfrak{h}`, we can pick a vertical element
# :math:`y\in\mathfrak{k}` and apply the corresponding group element :math:`K=\exp(y)` to
# all elements of the CSA, using the adjoint action we studied above.
# This will yield a valid CSA again. First, :math:`K\mathfrak{h} K^\dagger` remains in :math:`\mathfrak{m}`
# due to the reductive property, as we discussed when introducing the Cartan decomposition.
# Second the adjoint action will not change the Abelian property because
#
# .. math::
#
#     [K x_1 K^\dagger, K x_2 K^\dagger] = K [x_1, x_2] K^\dagger = 0 \quad \forall\ x_{1, 2}\in\mathfrak{h}.
#
# Finally, we are guaranteed that :math:`K\mathfrak{h} K^\dagger` remains maximal.
#
# .. note::
#
#     The reason that :math:`K\mathfrak{h} K^\dagger` is maximal if :math:`\mathfrak{h}` was, is
#     that we assume :math:`\mathfrak{g}` to be a semisimple Lie algebra, for which the
#     adjoint representation is faithful. This in turn implies that linearly
#     independent elements of :math:`\mathfrak{g}` will not be mapped to linearly dependent
#     elements by the adjoint action of :math:`K`.
#
# For most :math:`y\in\mathfrak{k}`, applying :math:`K=exp(y)` in this way will yield a
# *different* CSA, so that we find a whole continuum of them.
# It turns out that they *all* can be found by starting with *any*
# :math:`\mathfrak{h}` and applying all of :math:`\exp(\mathfrak{k})` to it.
#
# TODO: BETTER HIGHLIGHT
# *This fact is what powers the KAK theorem.*
#
# Involutions
# ~~~~~~~~~~~
#
# In practice, there often is a more convenient way to a Cartan decomposition
# than by specifying the subalgebra :math:`\mathfrak{k}` or its horizontal counterpart
# :math:`\mathfrak{m}` manually. It goes as follows.
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
# and :math:`\mathfrak{m}=\mathfrak{g}_-`.
#
# 🤯
#
# We might want to call such a :math:`\theta` a *Cartan involution*.
#
# .. warning::
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
#     \theta_{\mathfrak{k}}(x) = \Pi_{\mathfrak{k}}(x)-\Pi_{\mathfrak{m}}(x),
#
# where :math:`\Pi` are the projectors onto the two vector subspaces.
# Clearly, :math:`\theta_{\mathfrak{k}}` is linear because projectors are.
# It is also compatible with the commutator due to the commutation relations
# between :math:`\mathfrak{k}` and :math:`\mathfrak{m}` (see box).
#
# .. note::
#
#     To see that :math:`\theta_{\mathfrak{k}}` is compatible with the commutator, we compute
#     the action of the projectors on it:
#
#     .. math::
#
#         \Pi_{\mathfrak{k}}([x, y])
#         &= \Pi_{\mathfrak{k}}([\Pi_{\mathfrak{k}}(x) + \Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{k}}(y) + \Pi_{\mathfrak{m}}(y) \\
#         &= \Pi_{\mathfrak{k}}(\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)]}})
#         \Pi_{\mathfrak{k}}(\underset{\in \mathfrak{m}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{m}}(y)]}})
#         \Pi_{\mathfrak{k}}(\underset{\in \mathfrak{m}}{\underbrace{[\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{k}}(y)]}})
#         \Pi_{\mathfrak{k}}(\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{m}}(y)]}})\\
#         &= [\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)] + [\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{m}}(y)]\\
#         \Pi_{\mathfrak{m}}([x, y])
#         &= \Pi_{\mathfrak{m}}([\Pi_{\mathfrak{k}}(x) + \Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{k}}(y) + \Pi_{\mathfrak{m}}(y) \\
#         &= \Pi_{\mathfrak{m}}(\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)]}})
#         \Pi_{\mathfrak{m}}(\underset{\in \mathfrak{m}}{\underbrace{[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{m}}(y)]}})
#         \Pi_{\mathfrak{m}}(\underset{\in \mathfrak{m}}{\underbrace{[\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{k}}(y)]}})
#         \Pi_{\mathfrak{m}}(\underset{\in \mathfrak{k}}{\underbrace{[\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{m}}(y)]}})\\
#         &= [\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{m}}(y)] + [\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{k}}(y)]
#
#     Here we used :math:`\mathbb{I}_{\mathfrak{g}} = \Pi_{\mathfrak{k}} + \Pi_{\mathfrak{m}}` and the
#     commutation relations between :math:`\mathfrak{k}` and :math:`\mathfrak{m}`.
#
#     We can put thes pieces together to get
#
#     .. math::
#
#         \theta_{\mathfrak{k}} ([x, y])
#         &=\Pi_{\mathfrak{k}}([x, y]) - \Pi_{\mathfrak{m}}([x, y])\\
#         &=[\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{k}}(y)] + [\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{m}}(y)]
#         - [\Pi_{\mathfrak{k}}(x), \Pi_{\mathfrak{m}}(y)] - [\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{k}}(y)]\\
#         &=[\Pi_{\mathfrak{k}}(x) -\Pi_{\mathfrak{m}}(x), \Pi_{\mathfrak{k}}(y)-\Pi_{\mathfrak{m}}(y)]\\
#         &=[\theta_{\mathfrak{k}} (x),\theta_{\mathfrak{k}} (y)].
#
# Finally, :math:`\theta_{\mathfrak{k}}` is an involution because
#
# .. math::
#
#     \theta_{\mathfrak{k}}^2=(\Pi_{\mathfrak{k}}-\Pi_{\mathfrak{m}})^2
#     = \Pi_{\mathfrak{k}}^2-\Pi_{\mathfrak{k}}\Pi_{\mathfrak{m}}-\Pi_{\mathfrak{m}}\Pi_{\mathfrak{k}}+\Pi_{\mathfrak{m}}^2
#     =\Pi_{\mathfrak{k}}-\Pi_{\mathfrak{m}}
#     = \mathbb{I}_{\mathfrak{g}},
#
# where we used the projector's properties.
#
# This shows us that we can easily switch between a Cartan involution and a Cartan
# decomposition, in either direction!
#
# KAK theorem
# ~~~~~~~~~~~
#
# - KP decomposition
# - KAK decomposition
# - [optional] implication: KaK on algebra level
#
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
