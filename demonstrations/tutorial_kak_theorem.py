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


The stage
---------

Intro




The actors
----------

(Semi-)simple Lie algebras
~~~~~~~~~~~~~~~~~~~~~~~~~~
 Brief!

Subalgebras
~~~~~~~~~~~

Cartan involution
~~~~~~~~~~~~~~~~~

Symmetric space
~~~~~~~~~~~~~~~




The props
---------

Exponential map
~~~~~~~~~~~~~~~

Adjoint representation
~~~~~~~~~~~~~~~~~~~~~~




The plot
--------

Cartan involutions create subalgebras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subalgebras create Cartan involutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What happens in horizontal space...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So many Cartan subalgebras
~~~~~~~~~~~~~~~~~~~~~~~~~~

Finale: KAK theorem
~~~~~~~~~~~~~~~~~~~


The sequel, but quantum
-----------------------

A recursive decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~

The recursion step in detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local unitaries are universal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Conclusion
----------

In this demo we learned about the KAK theorem and how it uses a Cartan
decomposition of a Lie algebra to decompose its Lie group.
A famous immediate application of this result is the circuit decomposition, or
parametrization, for arbitrary qubit numbers by Khaneja and Glaser. It also allowed
us to prove universality of single and two-qubit unitaries for quantum computation.

If you are interested in other applications of Lie theory in the field of 
quantum computing, you are in luck! It has been a handy tool throughout the last
decades, e.g., for the simulation and compression of quantum circuits, 
in quantum optimal control, and for trainability analyses. For Lie algebraic
classical simulation of quantum circuits, check the
:doc:`g-sim </demos/tutorial_liesim/>` and
:doc:`(g+P)-sim </demos/tutorial_liesim_extension/>` demos, and stay posted for
a brand new demo on compiling Hamiltonian simulation circuits with the KAK theorem!
"""

import pennylane as qml

######################################################################
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
