r"""(g + P)-sim: Extending g-sim by non-DLA observables and gates
=================================================================

In a :doc:`previous demo </demos/tutorial_liesim>`, we have introduced the core concepts of
Lie-algebraic simulation techniques :math:`\mathfrak{g}`-sim [#Somma]_ [#Somma2]_ [#Galitski]_, such as :math:`\mathfrak{g}`-sim [#Goh]_.
With that, we can compute quantum circuit expectation values using the so-called dynamical Lie algebra (DLA) of the circuit generators and observables.
The complexity of :math:`\mathfrak{g}`-sim is determined by the dimension of :math:`\mathfrak{g}`.
Adding operators to :math:`\mathfrak{g}` can significantly increase its dimension, but we show here that
when one is using only few and a specific kind of non-DLA gates and observables, the increase in size can still be managable.

.. note::
    
    The contents of this demo are self-contained. However, we highly recommend reading our previous demos on
    :doc:`dynamical Lie algebras </demos/tutorial_liealgebra>` and :doc:`g-sim in PennyLane </demos/tutorial_liesim>`.

Introduction
------------

Lie-algebraic simulation techniques such as :math:`\mathfrak{g}`-sim can be handy in the niche cases where the
:doc:`dynamical Lie algebras (DLA) </demos/tutorial_liealgebra>` scales polynomially with
the number of qubits. That is for example the case for the transverse field Ising model (TFIM) and variants thereof in 1D [#Wiersema]_.

We are interested in cases where the majority of gates and observables are described by a small DLA but there are few gates and/or observables
that are outside the DLA. Take for example the TFIM with :math:`n` qubits and a dimension of
:math:`\text{dim}(\mathfrak{g}) = 2n(2n-1)/2` (see "Ising-type Lie algebras" :doc:`here </demos/tutorial_liealgebra>`).
Let us assume we want to expand the DLA by :math:`P = h_{\alpha_1} h_{\alpha_2} \notin \mathfrak{g}` in order to use it as a gate or include it in an observable.
Adding product operators and computing their new Lie closure often results in the associative algebra of :math:`\mathfrak{g}`, which in this case
would yield a dimension of :math:`2(2^{2n-2}-1)`, an exponential increase.

Here, we show how to do the same computation without going to the exponentially large associative algebra, but instead make use of the fact that :math:`P` is
a product of DLA elements. We do so by looking at `moments` of :math:`\mathfrak{g}` instead. The :math:`m`-th order moments are products of :math:`(m+1)` DLA elements.
E.g. :math:`P = h_{\alpha_1} h_{\alpha_2} \notin \mathfrak{g}` is a first order moment. Depending on their order, every non-DLA moment gate increases
the maximum moment order :math:`M_\text{max}` considered in the computation. The overall cost scales with the maximum order :math:`\text{dim}(\mathfrak{g})^{M_\text{max}}`.

In the worst case, each moment expands the space of operators by a factor :math:`\text{dim}(\mathfrak{g})`, such that for :math:`M` moments,
we are dealing with a :math:`\text{dim}(\mathfrak{g})^{M+2}` dimensional space. In that sense, this is similar to
:doc:`Clifford+T simulators </demos/tutorial_clifford_circuit_simulations>` where
expensive :math:`T` gates come with an exponential cost.

In [#Goh]_, the authors already hint at the possibility of extending :math:`\mathfrak{g}`-sim by expectation values
of products of DLA elements. In this demo, we extend this notion to `gates` generated by moments of the DLA.

:math:`\mathfrak{g}`-sim
------------------------

Let us briefly recap the core principles of :math:`\mathfrak{g}`-sim. We consider a Lie algebra :math:`\mathfrak{g} = \{h_1, .., h_d\}` that is closed
under commutation (see :func:`~pennylane.lie_closure`). We know that gates :math:`e^{-i \theta h_\alpha}` transform Lie algebra elements to Lie
algebra elements,

.. math:: e^{i \theta h_\mu} h_\alpha e^{-i \theta h_\mu} = \sum_\beta e^{-i \theta f^\mu_{\alpha \beta}} h_\beta.

This is the adjoint identity with the adjoint representation of the Lie algebra given by the :func:`~pennylane.structure_constants`,
:math:`\left(\text{ad}_{h_\mu}\right)_{\alpha \beta} = f^\mu_{\alpha \beta}`.

This lets us evolve any expectation value of DLA elements using the adjoint representation of the DLA.
For that, we define the expectation value vector :math:`(\boldsymbol{e})_\alpha = \text{tr}[h_\alpha \rho]`.
Also, let us write :math:`\tilde{U} = e^{-i \theta \text{ad}_{h_\mu}}` corresponding to a unitary :math:`U = e^{-i \theta h_\mu}`.
Using the adjoint identity above and the cyclic property of the trace, we can write an evolved expectation value vector as

.. math:: \text{tr}\left[h_\alpha U \rho U^\dagger\right] = \sum_\beta \tilde{U}_{\alpha \beta} \text{tr}\left[h_\beta \rho \right].

Hence, the expectation value vector is simply transformed by matrix multiplication with :math:`\tilde{U}` and we have

.. math:: \boldsymbol{e}^1 = \tilde{U} \boldsymbol{e}^0

for some input expectation value vector :math:`\boldsymbol{e}^0`.

A circuit comprised of multiple unitaries :math:`U` then simply corresponds to evolving the ecpectation value vector
according to the corresponding :math:`\tilde{U}`.

We are going to concretely use the DLA of the transverse field Ising model,

.. math:: H_\text{Ising} = J \sum_{i=1}^{n-1} X_i X_j + h \sum_{i=1}^n Z_i.

This is one of the few systems that yield a polynomially sized DLA.
Let us construct its DLA via :func:`~pennylane.lie_closure`.

"""

import pennylane as qml
import numpy as np

from pennylane import X, Y, Z, I
from pennylane.pauli import PauliSentence, PauliWord
from scipy.linalg import expm

import copy 

# TFIM generators
def TFIM(n):
    generators = [
        X(i) @ X(i+1) for i in range(n-1)
    ]
    generators += [
        Z(i) for i in range(n)
    ]
    generators = [op.pauli_rep for op in generators]

    dla = qml.pauli.lie_closure(generators, pauli=True)
    dim_dla = len(dla)
    return generators, dla, dim_dla

generators, dla, dim_g = TFIM(n=4)

##############################################################################
# In regular :math:`\mathfrak{g}`-sim, the unitary evolution :math:`U` of the expectation vector
# is simply generated by the adjoint representation :math:`\tilde{U}`.
# the DLA :math:`\mathfrak{g}` an generate

adjoint_repr = qml.structure_constants(dla)
gate = adjoint_repr[-1]
theta = 0.5

Utilde = expm(theta * gate)

##############################################################################
#
# :math:`(\mathfrak{g}+P)`-sim
# ----------------------------
# 
# We now want to extend :math:`\mathfrak{g}`-sim by operators that are not in the DLA, but a product
# of DLA operators. Note that while the DLA is closed under commutation, it is not under multiplication,
# such that products of DLA elements are in general not in :math:`\mathfrak{g}`.
# 
# The basic assumption is that the adjoint action of such a gate generates higher moments of DLA
# elements that we need to keep track of.
# In particular, for a gate generated by `p = h_{\mu_1} h_{\mu_2} \notin \mathfrak{g}`,
# we assume the adjoint action to be of the form
# 
# .. math:: e^{i \theta h_{\mu_1} h_{\mu_2}} h_\alpha e^{-i \theta h_{\mu_1} h_{\mu_2}} = \sum_\beta \tilde{P^0}_{\alpha \beta} h_\beta + \sum_{\beta_1 \beta_2} \tilde{P^1}_{\alpha \beta_1 \beta_2} h_{\beta_1} h_{\beta_2} + ...
# 
# Here, :math:`\tilde{P^\ell}` correspond to the contributions of the :math:`\ell`-th `moments` in :math:`\mathfrak{g}`.
# Let us look at the case where only first order moments contribute to the adjoint action of :math:`P = e^{i \theta h_{\mu_1} h_{\mu_2}}`. 
#
# For that, let us construct a concrete example. First we pick two elements from :math:`\mathfrak{g}` such that their product is not in :math:`\mathfrak{g}`.

p = dla[-5] @ dla[-2]
dla_vspace = qml.pauli.PauliVSpace(dla, dtype=complex)
dla_vspace.is_independent(p.pauli_rep)


##############################################################################
# Now, we compute :math:`e^{i \theta h_{\mu_1} h_{\mu_2}} h_\alpha e^{-i \theta h_{\mu_1} h_{\mu_2}}` and decompose it in the DLA and first moments.
#
# Note that since the product basis is overcomplete, we only keep track of the linearly independent elements and ignore the rest.

def exppw(theta, ps):
    # assert that it is indeed a pure pauli word, not a sentence
    assert (len(ps) == 1 and isinstance(ps, PauliSentence)) or isinstance(ps, PauliWord)
    return np.cos(theta) * PauliWord({}) + 1j * np.sin(theta) * ps

theta = 0.5

Up = exppw(theta, p)
Up_dagger = exppw(-theta, p) # complex conjugate

Ptilde0 = np.zeros((dim_g, dim_g), dtype=float)

for i, h1 in enumerate(dla):
    res = Up @ h1 @ Up_dagger
    for j, h2 in enumerate(dla):
        # decompose the result in terms of DLA elements
        # res = ∑ (res · h_j / ||h_j||^2) * h_j 
        value = (res @ h2).trace().real
        value = value / (h2 @ h2).trace()
        Ptilde0[i, j] = value

Ptilde1 = np.zeros((dim_g, dim_g, dim_g), dtype=float)

for i, h1 in enumerate(dla):
    res = Up @ h1 @ Up_dagger
    dla_and_M1_vspace = copy.deepcopy(dla_vspace)
    for j, h2 in enumerate(dla):
        for l, h3 in enumerate(dla):
            prod = h2 @ h3
            
            if not dla_and_M1_vspace.is_independent(prod):
                continue

            # decompose the result in terms of products of DLA elements
            # res = ∑ (res · p_j / ||p_j||^2) * p_j 
            value = (res @ prod).trace().real
            value = value / (prod @ prod).trace().real
            Ptilde1[i, j, l] = value
            dla_and_M1_vspace.add(prod)

##############################################################################
# We want to confirm that the adjoint action of P is indeed fully described by the first moments.
#
# For that, we reconstruct the transformed DLA elements and compare them with the decomposition.

for i, h1 in enumerate(dla):
    res = Up @ h1 @ Up_dagger
    res.simplify()

    reconstruct = sum([Ptilde0[i, j] * dla[j] for j in range(dim_g)])
    reconstruct += sum([Ptilde1[i, j, l] * dla[j] @ dla[l] for j in range(dim_g) for l in range(dim_g)])
    reconstruct.simplify()

    if res != reconstruct:
        print(f"Missmatch: {res}, {reconstruct}")

##############################################################################
# Now that we have successfully constructed a :math:`P` gate,
# let us look how entering it in a circuit transforms DLA elements (and therefore expectation value vector elements):
# 
# .. math::
# 
#     \begin{align*}
#     (\boldsymbol{e})_\alpha & = \text{tr}\left[h_\alpha P \rho P^\dagger \right] = \text{tr}\left[P^\dagger h_\alpha P \rho \right] \\
#     \ & = \sum_\beta \tilde{P^0}_{\alpha \beta} \text{tr}\left[ h_\beta \rho \right] + \sum_{\beta_1 \beta_2} \tilde{P^1}_{\alpha \beta_1 \beta_2} \text{tr}\left[ h_{\beta_1} h_{\beta_2} \rho \right] \\
#     \ & = \sum_\beta \tilde{P^0}_{\alpha \beta} (\boldsymbol{e})_\beta + \sum_{\beta_1 \beta_2} \tilde{P^1}_{\alpha \beta_1 \beta_2} (\boldsymbol{E}^1)_{\beta_1 \beta_2}
#     \end{align*}
# 
# Here we have defined the expectation tensor
# :math:`(\boldsymbol{E}^m)_{\alpha \beta_1 , .. , \beta_{m+1}} = \text{tr}\left[ h_{\beta_1} .. h_{\beta_{m+1}} \rho \right]` for the :math:`m`-th moment.
#
#
# Such a computation corresponds to the branching off from the original diagram, with an extra contribution coming from the higher moments.
# 
# .. figure:: /_static/demonstration_assets/liesim3/first_split.png
#    :width: 35%
#    :align: center
# 
# When inserting an arbitrary number of DLA gates :math:`\tilde{U} = \tilde{U}_1 \tilde{U}_2 \tilde{U}_3 ..` before and after the :math:`P` gate,
# we obtain the following diagram.
# 
# .. figure:: /_static/demonstration_assets/liesim3/first_order_diagram.png
#    :width: 45%
#    :align: center
# 
# Note that in one vertical column the :math:`\tilde{U}` correspond to the same matrices.
#
#
# Example
# ~~~~~~~
#
# Let us compute an example. For that we start by computing the initial expectation vector and tensor.

# TODO use pauli arithmetic instead
@qml.qnode(qml.device("default.qubit"))
def qnode(op):
    return qml.expval(op)

e_in = np.zeros((dim_g), dtype=float)
E_in = np.zeros((dim_g, dim_g), dtype=float)

for i, hi in enumerate(dla):
    e_in[i] = qnode(hi.operation())

for i, hi in enumerate(dla):
    for j, hj in enumerate(dla):
        # prod = strip(hi @ hj)
        prod = hi @ hj
        op = (prod).operation()
        op = op if op != I() else I(0)
        E_in[i, j] = qnode(op)

##############################################################################
#
# Now we need to compute the two branches from the diagram above.

# contract first branch
# - Ptilde - Utilde - e_in
res0 = Utilde @ e_in
res0 = Ptilde0 @ res0

# contract second branch

# --Utilde-==-+------+   -+------+
#             | E_in | =  | res  |
# --Utilde-==-+------+   -+------+
res = np.einsum("ij,jl->il", Utilde, E_in)
res = np.einsum("kl,il->ik", Utilde, res)

#    +--------+-==-+------+
#  --| Ptilde |    | res  |
#    +--------+-==-+------+
res = np.einsum("ijl,jl->i", Ptilde1, res)

res = res + res0

##############################################################################
# As a sanity check, let us compare this to the same circuit but using our default state vector simulator in PennyLane.

@qml.qnode(qml.device("default.qubit"))
def true():
    qml.exp(-1j * theta * dla[-1].operation())
    qml.exp(-1j * 0.5 * p.operation())
    return [qml.expval(op.operation()) for op in dla]

true_res = np.array(true())

##############################################################################
# We find that indeed both results coincide and expectation vector values are correctly transformed in :math:`(\mathfrak{g}+P)`-sim.

##############################################################################
# Higher moments
# ~~~~~~~~~~~~~~
#
# We can extend the above approach by more than one :math:`P` gate in the circuit.
# This leads to contributions from up to the fourth order. The diagram for a circuit :math:`P \tilde{U} P \tilde{U}` has the following five branches.
#
# First, the first and second order contribution
#
# .. figure:: /_static/demonstration_assets/liesim3/2P_first_second.png
#    :width: 75%
#    :align: center
#
# Then we have the two third order contributions.
#
# .. figure:: /_static/demonstration_assets/liesim3/2P_thirds.png
#    :width: 75%
#    :align: center
#
# And finally the fourth order contribution.
#
# .. figure:: /_static/demonstration_assets/liesim3/2P_fourth.png
#    :width: 75%
#    :align: center

##############################################################################
# Moment vector space
# -------------------
#
# - there is a maximum moment
# - no point in using the above when reaching the maximum moment
# - dimension of this explodes relatively quickly, showcase dimension scaling to show just how niche this all is

##############################################################################
# 
# Conclusion
# ----------
#
# Great success
#



##############################################################################
# 
# References
# ----------
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
# .. [#Wiersema]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
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
# .. [#Cerezo]
#
#     M. Cerezo, Martin Larocca, Diego García-Martín, N. L. Diaz, Paolo Braccia, Enrico Fontana, Manuel S. Rudolph, Pablo Bermejo, Aroosa Ijaz, Supanut Thanasilp, Eric R. Anschuetz, Zoë Holmes
#     "Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink variational quantum computing"
#     `arXiv:2312.09121 <https://arxiv.org/abs/2312.09121>`__, 2023.
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
