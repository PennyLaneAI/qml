r"""Fixed Depth Hamiltonian Simulation via Cartan Decomposition
===============================================================

abstract

Introduction
------------

The KAK theorem is an important result from Lie theory that states that any Lie group element :math:`U` can be decomposed
as :math:`U = K_1 A K_2`, where :math:`K_{1, 2}` and :math:`A` are elements of two separate special sub-groups
:math:`\mathcal{K}` and :math:`\mathcal{A}`, respectively. You can think of this KAK decomposition as a generalization of
the singular value decomposition to Lie groups.

For that, recall that the singular value decomposition states that any
matrix :math:`M \in \mathbb{C}^{m \times n}` can be decomposed as :math:`M = U \Lambda V^\dagger`, where :math:`\Lambda`
are the diagonal singular values and :math:`U \in \mathbb{C}^{m \times \mu}` and :math:`V^\dagger \in \mathbb{C}^{\mu \times n}`
are left- and right-unitary with :math:`\mu = \min(m, n)`.

In the case of the KAK decomposition, :math:`\mathcal{A}` is an Abelian subgroup such that all its elements are commuting,
just as is the case for diagonal matrices.

Unitary gates in quantum computing are described by the special orthogonal Lie group :math:`SU(2^n)`, so we can use the KAK
theorem to decompose quantum gates into :math:`U = K_1 A K_2`. While the mathematical statement is rather straight-forward,
actually finding this decomposition is not. We are going to follow the recipe prescribed in 
`Fixed Depth Hamiltonian Simulation via Cartan Decomposition <https://arxiv.org/abs/2104.00728>`__ [#Kökcü]_.

Let us walk through an explicit example, doing theory and a concrete example side-by-side.
For that we are going to use the Heisenberg model generators and Hamiltonian for :math:`n=4` qubits.
The foundation to a KAK decomposition is a Cartan decomposition of the associated Lie algebra :math:`\mathfrak{g}`.
For that, let us first construct it and import some libraries that we are going to use later.


"""
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pennylane as qml
from pennylane import X, Y, Z

import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)

n_wires = 4
gens = [X(i) @ X(i+1) for i in range(n_wires-1)]
gens += [Y(i) @ Y(i+1) for i in range(n_wires-1)]
gens += [Z(i) @ Z(i+1) for i in range(n_wires-1)]

H = qml.sum(*gens)

g = qml.lie_closure(gens)
g = [op.pauli_rep for op in g]

##############################################################################
# 
# Cartan decomposition
# --------------------
# 
# First let us recap the necessary ingredients to mathematically obtain the KAK decomposition.
# 
# 
# A Cartan decomposition is a bipartition :math:`\mathfrak{g} = \mathcal{k} \oplus \mathcal{m}` into a vertical subspace
# :math:`\mathfrak{k}` and an orthogonal horizontal subspace :math:`\mathfrak{m}`. In practice, it can be induced by an
# involution function :math:`\Theta` that fulfils :math:`\Theta(\Theta(g)) = g \forall g \in \mathfrak{g}`. Different 
# involutions lead to different types of Cartan decompositions, which have been fully classified by Cartan, 
# see `wikipedia <https://en.wikipedia.org/wiki/Symmetric_space#Classification_result>`__.
# 
# .. note::
#     Note that :math:`\mathfrak{k}` is the small letter k in
#     `Fraktur <https://en.wikipedia.org/wiki/Fraktur>`__ and a 
#     common - not our - choice for the vertical subspace in a Cartan decomposition.
#
# One common choice of involution is the so-called even-odd involution for Pauli words
# :math:`P = P_1 \otimes P_2 .. \otimes P_n` where `P_j \in \{I, X, Y, Z\}`.
# It essentially counts whether the number of non-identity Pauli operators in the Pauli word is even or odd.

def even_odd_involution(op):
    """Generalization of EvenOdd to sums of Paulis"""
    [pw] = op.pauli_rep
    parity = len(pw) % 2

    return parity

even_odd_involution(X(0)), even_odd_involution(X(0) @ Y(3))

##############################################################################
# 
# The vertical and horizontal subspaces are the two eigenspaces of the involution, corresponding to the :math:`\pm 1` eigenvalues.
# In particular, we have :math:`\Theta(\mathfrak{k}) = \mathfrak{k}` and :math:`\Theta(\mathfrak{m}) = - \mathfrak{m}`
# So in order to perform the Cartan decomposition :math:`\mathfrak{g} = \mathcal{k} \oplus \mathcal{m}`, we simply
# sort the operators by whether or not they yield a plus or minus sign from the involution function.

def cartan_decomposition(g, involution):
    """Cartan Decomposition g = k + m
    
    Args:
        g (List[PauliSentence]): the (dynamical) Lie algebra to decompose
        involution (callable): Involution function :math:`\Theta(\cdot)` to act on PauliSentence ops, should return ``0/1`` or ``True/False``.
    
    Returns:
        k (List[PauliSentence]): the even parity subspace :math:`\Theta(\mathfrak{k}) = \mathfrak{k}`
        m (List[PauliSentence]): the odd parity subspace :math:`\Theta(\mathfrak{m}) = \mathfrak{m}` """
    m = []
    k = []

    for op in g:
        if involution(op): # odd parity
            k.append(op)
        else: # even parity
            m.append(op)
    return k, m

k, m = cartan_decomposition(g, even_odd_involution)
len(k), len(m)


##############################################################################
# 

##############################################################################
# 

##############################################################################
# 

##############################################################################
# 

##############################################################################
# 
#


##############################################################################
# 
# Conclusion
# ----------
#
# With this introduction, we hope to clarify some terminology, introduce the basic concepts of Lie theory and motivate their relevance in quantum physics by touching on universality and symmetries.
# While Lie theory and symmetries are playing a central role in established fields such as quantum phase transitions (see note above) and `high energy physics <https://en.wikipedia.org/wiki/Standard_Model>`_,
# they have recently also emerged in quantum machine learning with the onset of geometric quantum machine learning [#Meyer]_ [#Nguyen]_
# (see our recent :doc:`introduction to geometric quantum machine learning <tutorial_geometric_qml>`).
# Further, DLAs have recently become instrumental in classifying criteria for barren plateaus [#Fontana]_ [#Ragone]_ and designing simulators based on them [#Goh]_.
#



##############################################################################
# 
# References
# ----------
#
# .. [#Kökcü]
#
#     Efekan Kökcü, Thomas Steckmann, Yan Wang, J. K. Freericks, Eugene F. Dumitrescu, Alexander F. Kemper
#     "Fixed Depth Hamiltonian Simulation via Cartan Decomposition"
#     `arXiv:2104.00728 <https://arxiv.org/abs/2104.00728>`__, 2021.
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
# .. [#Somma]
#
#     Rolando D. Somma
#     "Quantum Computation, Complexity, and Many-Body Physics"
#     `arXiv:quant-ph/0512209 <https://arxiv.org/abs/quant-ph/0512209>`__, 2005.
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
