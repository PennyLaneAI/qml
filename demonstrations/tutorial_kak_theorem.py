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
and introduce Cartan involutions/decompositions, horizontal
and vertical subspaces, as well as horizontal Cartan subalgebras.
With these tools in our hands, we will then learn about the KAK theorem
itself.
We conclude with a famous application of the theorem to circuit decomposition
by Khaneja and Glaser [#khaneja_glaser]_, which provides a circuit
template for arbitrary unitaries on any number of qubits, and proved for
the first time that single and two-qubit gates are sufficient to implement them.

While this demo is of more mathematical nature than others, we will include
hands-on examples throughout.

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

- Introduce the notion of a Lie algebra very briefly, refer to existing demo(s).
- Focus on vector space notion being clear.
- [optional] Briefly say what a simple/semisimple Lie algebra is.
- [optional] In particular mention that the adjoint representation is faithful for semisimple algebras.

Group and algebra interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Exponential map
- adjoint action of group on algebra
- adjoint action of algebra on algebra -> adjoint representation
- adjoint identity (-> g-sim demo)

Subalgebras and Cartan pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Introduce the notion of a subalgebra.
- Explain that there can be vector subspaces that are not subalgebras.
- Define Cartan pairs via commutation relations

Cartan subalgebras
~~~~~~~~~~~~~~~~~~

- Define Cartan subalgebras of :math:`m`.
- Dimension of Cartan subalgebras
- Transition between Cartan subalgebras via :math:`K`

Involutions
~~~~~~~~~~~

- Explain linear maps on (matrix) algebras (-> homomorphism)
- Define involutions.
- Involutions define Cartan pairs (:math:`k = +1 | m = -1` eigenspaces)
- Cartan pairs define involutions :math:`\theta = \Pi_{\mathfrak{k}} - \Pi_{\mathfrak{m}}`

KAK theorem
~~~~~~~~~~~

- KP decomposition
- KAK decomposition
- [optional] implication: KaK on algebra level


Two-qubit KAK decomposition
---------------------------

- Algebra/subalgebra :math:`\mathfrak{g} =\mathfrak{su}(4) | \mathfrak{k} =\mathfrak{su}(2) \oplus \mathfrak{su}(2)`
- Involution: EvenOdd
- CSA: :math:`\mathfrak{a} = \langle\{XX, YY, ZZ\}\rangle_{i\mathbb{R}}`
- KAK decomposition :math:`U= (A\otimes B) \exp(i(\eta_x XX+\eta_y YY +\eta_z ZZ)) (C\otimes D)`.
- [optional] Mention Cartan coordinates

Khaneja-Glaser decomposition
----------------------------

- Important first recursive decomposition showing universality of single- and two-qubit operations
- Used for practical decompositions, replaced by other, similar decompositions by now

A recursive decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~

- Show recursion on qubit count
- display resulting decomposition structure
- Mention that a two-qubit interaction is enough to get the CSA elements
- Universality

The recursion step in detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Two substeps in each recursion step:
    - Algebra/subalgebra :math:`\mathfrak{g}=\mathfrak{su}(2^n) | \mathfrak{k} = \mathfrak{su}(2^{n-1}) \oplus \mathfrak{su}(2^{n-1})`
    - Involution TBD
    - CSA TBD
    - Algebra/subalgebra :math:`\mathfrak{g}=\mathfrak{su}(2^{n-1}) \oplus \mathfrak{su}(2^{n-1}) | \mathfrak{k} = \mathfrak{su}(2^{n-1})`
    - Involution TBD
    - CSA TBD

Overview of resulting decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Count blocks
- [optional] CNOT count


Conclusion
----------

In this demo we learned about the KAK theorem and how it uses a Cartan
decomposition of a Lie algebra to decompose its Lie group.
A famous immediate application of this result is the circuit decomposition, or
parametrization, for arbitrary qubit numbers by Khaneja and Glaser. It also allowed
us to prove universality of single and two-qubit unitaries for quantum computation.

If you are interested in other applications of Lie theory in the field of
quantum computing, you are in luck! It has been a handy tool throughout the last
decades, e.g., for the simulation and compression of quantum circuits, # TODO: REFS
in quantum optimal control, and for trainability analyses. For Lie algebraic
classical simulation of quantum circuits, check the
:doc:`g-sim </demos/tutorial_liesim/>` and
:doc:`(g+P)-sim </demos/tutorial_liesim_extension/>` demos, and stay posted for
a brand new demo on compiling Hamiltonian simulation circuits with the KAK theorem!


The props
---------

Adjoint representation
~~~~~~~~~~~~~~~~~~~~~~

"""

import pennylane as qml

######################################################################
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
