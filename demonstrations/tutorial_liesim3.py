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
    :doc:`dynamical Lie algebras </demos/tutorial_liealgebra` and :doc:`g-sim in PennyLane </demos/tutorial_liesim>`.

Introduction
------------

Lie-algebraic simulation techniques such as :math:`\mathfrak{g}`-sim can be handy in the niche cases where the
:doc:`dynamical Lie algebras (DLA) </demos/tutorial_liealgebra>` scales polynomially with
the number of qubits. That is for example the case for the transverse field Ising model (TFIM) and variants thereof in 1D [#Wiersema]_.

We are interested in cases where the majority of gates and observables are described by a small DLA but there are few gates and/or observables
that are outside the DLA. Take for example the TFIM for :math:`n = 6` qubits with a dimension of
:math:`\text{dim}(\mathfrak{g}) = 2n(2n-1)/2 = 66` (see "Ising-type Lie algebras" :doc:`here </demos/tutorial_liealgebra`).
Let us assume we want to expand the DLA by :math:`P = Y_0 Z_1 X_2 Y_4 Y_5 \notin \mathfrak{g}` in order to use it as a gate or include it in an observable.
This would yield a dimension of :math:`2(2^{2n-2}-1) = 2046`, an exponential increase (or :math:`31\times` for this specific example).

Instead, we can identify :math:`P` as the product of two DLA elements, :math:`P = h_{\alpha_1} h_{\alpha_2}`. We show in this demo that in this special
case we can work in the significantly smaller vector space of first-order products (first "moments") of DLA elements,
that are of size :math:`O(n^3)` (here :math:`561` in this particular example).

In [#Goh]_, the authors already hint at the possibility of extending :math:`\mathfrak{g}`-sim by expectation values
of products of DLA elements. In this demo, we extend this notion to `gates` generated by moments of the DLA.
Furthermore, we introduce a more efficient representation of this extension.

The final algorithm lets us compute expectation values of moments as well as circuits comprised of gates that are generated by higher moments.
The required order of the moments under consideration is determined by the number of non-DLA gates and their order.

In the worst case, each moment expands the space of operators by a factor :math:`\text{dim}(\mathfrak{g})`, such that for :math:`M` moments,
we are dealing with a :math:`\text{dim}(\mathfrak{g})^{M+2}` dimensional space. In that sense, this is similar to
:doc:`Clifford+T simulators </demos/tutorial_clifford_circuit_simulations>` where
expensive :math:`T` gates come with an exponential cost.

At the same time, the actual expansion is much sparser in practice, which we see in the example below and which we take advantage of with a 
more efficient representation.

The final computation is analogous to :math:`\mathfrak{g}`-sim, just using the vector space of moments instead of the DLA.

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

:math:`(\mathfrak{g}+P)`-sim
----------------------------

We now want to extend :math:`\mathfrak{g}`-sim by operators that are not in the DLA, but a product
of DLA operators. Note that while the DLA is closed under commutation, it is not under multiplication,
such that products of DLA elements are in general not in :math:`\mathfrak{g}`.

The basic assumption is that the adjoint action of such a gate generates higher moments of DLA
elements that we need to keep track of.
In particular, for a gate generated by `p = h_{\mu_1} h_{\mu_2} \notin \mathfrak{g}`,
we assume the adjoint action to be of the form

.. math:: e^{i \theta h_{\mu_1} h_{\mu_2}} h_\alpha e^{-i \theta h_{\mu_1} h_{\mu_2}} = \sum_\beta \tilde{P^0}_{\alpha \beta} h_\beta + \sum_{\beta_1 \beta_2} \tilde{P^1}_{\alpha \beta_1 \beta_2} h_{\beta_1} h_{\beta_2} + ...

Here, :math:`\tilde{P^\ell}` correspond to the contributions of the :math:`\ell`-th `moments` in :math:`\mathfrak{g}`.
Let us look at the case where only first order moments contribute to the adjoint action of :math:`P = e^{i \theta h_{\mu_1} h_{\mu_2}}`. We have

.. math::

    \begin{align*}
    (\boldsymbol{e})_\alpha & = \text{tr}\left[h_\alpha P \rho P^\dagger \right] = \text{tr}\left[P^\dagger h_\alpha P \rho \right] \\
    \ & = \sum_\beta \tilde{P^0}_{\alpha \beta} \text{tr}\left[ h_\beta \rho \right] + \sum_{\beta_1 \beta_2} \tilde{P^1}_{\alpha \beta_1 \beta_2} \text{tr}\left[ h_{\beta_1} h_{\beta_2} \rho \right] \\
    \ & = \sum_\beta \tilde{P^0}_{\alpha \beta} (\boldsymbol{e})_\beta + \sum_{\beta_1 \beta_2} \tilde{P^1}_{\alpha \beta_1 \beta_2} (\boldsymbol{E}^1)_{\beta_1 \beta_2}
    \end{align*}

Here we have defined the expectation tensor
:math:`(\boldsymbol{E}^m)_{\alpha \beta_1 , .. , \beta_{m+1}} = \text{tr}\left[ h_{\beta_1} .. h_{\beta_{m+1}} \rho \right]` for the :math:`m`-th moment.
This corresponds to the branching off from the original diagram, with an extra contribution coming from the higher moments.

.. figure:: /_static/demonstration_assets/liesim3/first_split.png
   :width: 35%
   :align: center

We can append arbitrary regular DLA gates before and after the :math:`P` gate.

.. figure:: /_static/demonstration_assets/liesim3/first_order_diagram.png
   :width: 45%
   :align: center

"""

import pennylane as qml
from pennylane import X, Y, Z, I
from pennylane.pauli import PauliSentence, PauliWord, PauliVSpace

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
# Moments
# -------
#
# A Lie algebra :math:`\mathfrak{g}` is closed under commutation, but not necessarily under multiplication. Hence, a product
# :math:`h_{\alpha_1} h_{\alpha_2}` of elements in :math:`\mathfrak{g}` is in general not in :math:`\mathfrak{g}`.
# Whenever :math:`h_{\alpha_1} h_{\alpha_2} \notin \mathfrak{g}`, we call it a first moment of :math:`\mathfrak{g}`.
# We can collect all first order moments in a vector space :math:`\mathcal{M}^1` (note that :math:`\mathcal{M}^0 = \mathfrak{g}`).
#
# Second order moments can be computed by looking at :math:`h_{\alpha_1} h_{\alpha_2} h_{\alpha_3} \notin \mathfrak{g} \cup \mathcal{M}^1`.
# Alternatively and more simply, we can check :math:`h_{\alpha_1} \tilde{h}_{\alpha_2} \notin \mathfrak{g} \cup \mathcal{M}^1`
# with :math:`\tilde{h}_{\alpha_2} \in \mathcal{M}^1`, as we are doing in the ``Moment_step`` function below.
#
# We can continue this process until eventually the maximum order is reached and no new operators are generated. We have then successfully constructed the 
# associative Lie algebra of :math:`\mathfrak{g}`.
#
# We now set up a vector space starting from the DLA and keep adding linearly independent product operators.

def Moment_step(ops, dla):
    MomentX = PauliVSpace(ops.copy())
    for i, op1 in enumerate(dla):
        for op2 in ops[i+1:]:
            prod = op1 @ op2
            # ignore scalar coefficient
            pw = next(iter(prod.keys()))

            MomentX.add(pw)
    
    return MomentX.basis

Moment0 = dla.copy()
Moment = [Moment0]
dim = [len(Moment0)]
for i in range(1, 5):
    Moment.append(Moment_step(Moment[-1], dla))
    dim.append(len(Moment[-1]))

dim

##############################################################################
# We see the growing dimension of the intermediate moment spaces. Eventually they saturate when reaching the maximum moment,
# which here is :math:`3`. The associative algebra has dimension :math:`127`.
#
# It is important to recall that the (intermediate) moments generally do not form a Lie algebra. This is because
# they are not closed under commutation, which can be seen by comparing its dimension with
# that of its Lie closure.

Moment1_closure = qml.lie_closure(Moment[1])
len(Moment1_closure), len(Moment[1])

##############################################################################
#
# We now look at the scaling of the first moment spaces compared to the Lie algebra :math:`\langle \mathfrak{g} + P \rangle_\text{Lie}`
# (where :math:`\langle \cdot \rangle_\text{Lie}` refers to :func:`~pennylane.lie_closure`) for some first order moment :math:`P`.
# For that, we compute the :func:`~pennylane.lie_closure` of the DLA extended by some :math:`P = h_{\alpha_1} h_{\alpha_2} \notin \mathfrak{g}`,
# as well as the space of all the first moments.

import matplotlib.pyplot as plt
import numpy as np

dims_dla = []
dims_moment1 = []
dims_g_plus_P_dla = []

ns = np.arange(3, 7)

for n in ns:
    _, dla, dim_g = TFIM(n)
    moments = Moment_step(dla, dla)
    g_plus_P_dla = qml.lie_closure(dla + [moments[-1]], pauli=True)

    dims_dla.append(dim_g)
    dims_moment1.append(len(moments))
    dims_g_plus_P_dla.append(len(g_plus_P_dla))

plt.title("Dimension of $\\langle g + P \\rangle_{{Lie}}$ vs $\mathcal{M}^1$")

plt.plot(ns, dims_g_plus_P_dla, "x--", label="${{dim}}(\\langle g + P \\rangle_{{Lie}})$", color="tab:blue")
plt.plot(ns, 2 * (2**(2*ns - 2) - 1), "-", label="$2(2^{{2n-2}}-1)$", color="tab:blue")

plt.plot(ns, dims_moment1, "x--", label="${{dim}}(\mathcal{M}^1)$", color="tab:orange")
coeff = np.polyfit(ns, dims_moment1, 3) # polynomial fit of order 3
plt.plot(ns, np.poly1d(coeff)(ns), "-", label="$O(n^3)$", color="tab:orange")

plt.yscale("log")
plt.legend()
plt.xlabel("n")
plt.show()

##############################################################################
#
# We can see the exponential scaling of the Lie algebra :math:`\langle \mathfrak{g} + P \rangle_\text{Lie}`,
# whereas the vector space of the first moments scales as :math:`O(n^3)`.
# 
# :math:`(\mathfrak{g}+P)`-sim
# ----------------------------
#
# The goal of :math:`(\mathfrak{g}+P)`-sim is to incorporate moments (product operators) as gates (called :math:`P` gates).
# The resulting algorithm is surprisingly simple and analogous to :math:`\mathfrak{g}`-sim with just
# the DLA :math:`\mathfrak{g}` exchanged for a suitably chosen moment vector space.
#
# This works under the condition that we choose a suitably large overall moment space
# :math:`\mathfrak{g} \cup \mathcal{M}^1 \cup \mathcal{M}^2 .. ` depending on how many gates and non-DLA operators we use in the circuit and observables.
#
# Let us work out an intuition how to choose the maximal moment to keep track of.
# For that, we look at the adjoint action of a :math:`P_\mu` gate,
#
# .. math:: e^{i \theta P_\mu} P_\alpha e^{-i \theta P_\mu} = \hat{O}(\mathfrak{g}) + \hat{O}(\mathcal{M}^1) + \hat{O}(\mathcal{M}^2) + .. + \hat{O}(\mathcal{M}^{d_\text{max}}).
#
# Each :math:`\hat{O}` indicates the space the adjoint action maps to, with the maximal degree being the associatice algebra with degree :math:`d_\text{max}`.
# For example, for both :math:`P_\alpha \in \mathfrak{g}` and :math:`P_\mu \in \mathfrak{g}` we just map to :math:`\mathfrak{g}` again with all other contributions being zero.
# This is what happens in :math:`\mathfrak{g}`-sim.
#
# For :math:`P_\alpha \in \mathfrak{g}` and :math:`P_\mu \in \mathcal{M^1}` (i.e. :math:`P_\mu = h_{\mu_1} h_{\mu_2}`), we get nonzero contributions in \hat{O}(\mathfrak{g}) + \hat{O}(\mathcal{M}^1).
# Hence, for one such first-order :math:`P` gate in the circuit and only DLA observables, we need to track up to first order moments in the computation.
#
# If we were to also track first-order observables, we need to track up to second order moments.
# asd
#
# First, we compute the initial expectation value vector :math:`\boldsymbol{e}` for not just the DLA
# but the degree of moments we are considering. For now, let us just use the first moments
# (which will permit us to run :math:`(\mathfrak{g} + P)`-sim with one :math:`P`-gate later).

from scipy.linalg import expm

def comp_e_in(pick_moment):
    # compute initial expectation vector
    e_in = np.zeros(dim[pick_moment], dtype=float)

    for i, h_i in enumerate(Moment[pick_moment]):
        # initial state |0x0| = (I + Z)/2, note that trace function
        # below already normalizes by the dimension,
        # so we can ommit the explicit factor /2
        rho_in = qml.prod(*(I(i) + Z(i) for i in h_i.wires))
        rho_in = rho_in.pauli_rep

        e_in[i] = (h_i @ rho_in).trace()
    return e_in

pick_moment = 1 # order of moments (=number of P gates)
e_in = comp_e_in(pick_moment)

##############################################################################
#
# Now, :math:`(\mathfrak{g} + P)`-sim works in the same way as regular :math:`\mathfrak{g}`-sim,
# just extended to the appropriately chosen moment vector space.
#
# Let us define a callable that computes the final expectation value of the TFIM Hamiltonian H_\text{Ising}
# with :math:`J=h=0.5`. For comparison, we also define a ``qnode`` that computes the same expectation value
# using a state vector simulator.

def run_gP_sim(e_in, coeff, gates):
    # simulation
    e_t = e_in.copy()
    for i in range(len(gates)):
        e_t = expm(coeff[i] * gates[i]) @ e_t

    # final expectation value
    # H = 0.5 @ generators (TFIM Hamiltonian)
    weights = np.zeros(dim[pick_moment], dtype=complex)
    weights[:len(generators)] = 0.5 

    return weights @ e_t

H = 0.5 * qml.sum(*[op.operation() for op in generators])

@qml.qnode(qml.device("default.qubit"))
def qnode(coeff, gates):
    for i in range(len(gates)):
        qml.exp(
            -1j * coeff[i] * gates[i]
        )
    return qml.expval(H)

##############################################################################
# We now set a random circuit by picking random generators from the DLA but allow for the central gate
# to be a :math:`P` gate from the first moments.

depth = 10
coeff = np.random.rand(depth)

adjoint_repr = qml.structure_constants(Moment[pick_moment])

for i in range(10):
    gate_choice = np.random.choice(dim_g, size=depth)

    # one P gate in the circuit
    gate_choice[depth//2] = np.random.choice(range(dim[0], dim[pick_moment]), size=(1,))[0]
    gates = adjoint_repr[gate_choice]

    result_g_sim = run_gP_sim(e_in, coeff, gates)

    pl_gate_generators = [Moment[pick_moment][i].operation() for i in gate_choice]
    true_res = qnode(coeff, pl_gate_generators)

    if not np.allclose(result_g_sim, true_res):
        print(f"FAIL: g-sim res: {result_g_sim}, exact res: {true_res}")
    else:
        print(f"SUCCESS: g-sim res: {result_g_sim}, exact res: {true_res}")

##############################################################################
# If we want to succeed with two :math:`P^1` gates, we need to go up to the 2nd moments.

pick_moment = 2
adjoint_repr = qml.structure_constants(Moment[pick_moment])

for i in range(10):
    gate_choice = np.random.choice(dim_g, size=depth)

    # 1st P^1 gate
    gate_choice[depth//2] = np.random.choice(range(dim[0], dim[1]), size=(1,))[0]
    # 2nd P^1 gate
    gate_choice[depth//2+1] = np.random.choice(range(dim[0], dim[1]), size=(1,))[0]

    gates = adjoint_repr[gate_choice]

    result_g_sim = run_gP_sim(comp_e_in(pick_moment), coeff, gates)

    pl_gate_generators = [Moment[pick_moment][i].operation() for i in gate_choice]
    true_res = qnode(coeff, pl_gate_generators)

    if not np.allclose(result_g_sim, true_res):
        print(f"FAIL: g-sim res: {result_g_sim}, exact res: {true_res}")
    else:
        print(f"SUCCESS: g-sim res: {result_g_sim}, exact res: {true_res}")

##############################################################################
# We also need the 2nd moments if we want to succeed with one :math:`P^2` gate.

for i in range(10):
    gate_choice = np.random.choice(dim_g, size=depth)

    # one P^2 gate
    gate_choice[depth//2] = np.random.choice(range(dim[1], dim[2]), size=(1,))[0]
    gates = adjoint_repr[gate_choice]

    result_g_sim = run_gP_sim(comp_e_in(pick_moment), coeff, gates)

    pl_gate_generators = [Moment[pick_moment][i].operation() for i in gate_choice]
    true_res = qnode(coeff, pl_gate_generators)

    if not np.allclose(result_g_sim, true_res):
        print(f"FAIL: g-sim res: {result_g_sim}, exact res: {true_res}")
    else:
        print(f"SUCCESS: g-sim res: {result_g_sim}, exact res: {true_res}")

##############################################################################
#
# Computational bottlenecks
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As we go to larger moment spaces, the bottlebeck soon becomes computing :func:`~pennylane.structure_constants`.
# This function iterates over :math:`\tilde{d}^3/2` combinations, where :math:`\tilde{d}` is the dimension of the vector space
# of the moments we are considering. Instead of computing the adjoint representation for every element of the moment space
# we can also directly compute the adjoint action of every gate.
#
# We do that by simply projecting :math:`U^\dagger h_\alpha U` for every :math:`h_\alpha \in \mathcal{M}^\mu` onto the basis of :math:`\mathcal{M}^\mu`.
# Let us recall that even though the :math:`\mu`-th moment space :math:`\mathcal{M}^\mu` generally does not form a Lie algebra, it is a valid 
# vector space by construction.
#
# We still need to iterate through :math:`\tilde{d}^2` elements, but can do that individually for only the :math:`D` gates in the circuit.
# So the cost here is :math:`D\tilde{d}^2`, which is worth while whenever 
# :math:`D \leq \tilde{d}/2`. Additionally, because we are dealing with Pauli words, we can avoid computing the exponential
# of the adjoint representation because there is an efficient formula
# :math:`e^{-i \theta \bigotimes_j P_j} = \cos(\theta) \mathbb{I} -i \sin(\theta) \bigotimes_j P_j` for the exponential of them.

def exppw(theta, ps):
    # assert that it is indeed a pure pauli word, not a sentence
    assert (len(ps) == 1 and isinstance(ps, PauliSentence)) or isinstance(ps, PauliWord)
    return np.cos(theta) * PauliWord({}) + 1j * np.sin(theta) * ps

pick_moment = 1 # setting back to 1 as we are only using 1 P^1 gate in this example

for i in range(10):
    gate_choice = np.random.choice(dim_g, size=depth)
    gate_choice[depth//2] = np.random.choice(range(dim[0], dim[pick_moment]), size=(1,))[0] # one P^1 gate
    gates = np.array(Moment[pick_moment])[gate_choice].tolist()
    # Compute adjoint actions of all gates
    adj_gates = []

    for i, t in enumerate(gates):
        theta = coeff[i]

        T = exppw(theta, t)
        Td = exppw(-theta, t) # complex conjugate

        # adjoint action of T, decomposed in elements of the vector space
        T_adj = np.zeros((dim[pick_moment], dim[pick_moment]), dtype=float)

        for i, h1 in enumerate(Moment[pick_moment]):
            res = T @ h1 @ Td
            for j, h2 in enumerate(Moment[pick_moment]):
                # decompose the result in terms of DLA elements
                # res = ∑ (res · h_j / ||h_j||^2) * h_j 
                value = (res @ h2).trace().real
                value = value / (h2 @ h2).trace()
                T_adj[i, j] = value

        adj_gates.append(T_adj)

    # simulation
    e_t = e_in.copy()
    for i in range(depth):
        e_t = adj_gates[i] @ e_t

    # H = 0.5 @ generators (not full dla)
    weights = np.zeros(dim[pick_moment], dtype=complex)
    weights[:len(generators)] = 0.5 

    result_g_sim = weights @ e_t

    pl_gate_generators = [Moment[pick_moment][i].operation() for i in gate_choice]
    true_res = qnode(coeff, pl_gate_generators)

    if not np.allclose(result_g_sim, true_res):
        print(f"FAIL: g-sim res: {result_g_sim}, exact res: {true_res}")
    else:
        print(f"SUCCESS: g-sim res: {result_g_sim}, exact res: {true_res}")


##############################################################################
# Alternatively, we can also speed up :func:`~pennylane.structure_constants` itself
# by making use of the fact that all computations in the outer loop are independent
# and use `embarrassing parallelism <https://en.wikipedia.org/wiki/Embarrassingly_parallel>`__.
#
# In python, this can be done with ``multiprocessing`` as follows.

import multiprocessing as mp
import concurrent.futures
from itertools import combinations

max_workers = 8 # number of CPU cores to distribute the task over

gtilde = Moment[pick_moment]
dtilde = len(gtilde)

# compute adjoint representation using embarrassing parallelism
chunk_size = dtilde // max_workers
chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(max_workers)]
chunks[-1] = (chunks[-1][0], dtilde)

commutators = {}
for (j, op1), (k, op2) in combinations(enumerate(gtilde), r=2):
    res = op1.commutator(op2)
    if res != PauliSentence({}):
        commutators[(j, k)] = res

def _wrap_run_job(chunk):
    rep = np.zeros((np.diff(chunk)[0], len(gtilde), len(gtilde)), dtype=float)
    for idx, i in enumerate(range(*chunk)):
        op = gtilde[i]
        for (j, k), res in commutators.items():
            value = (1j * (op @ res).trace()).real
            value = value / (op @ op).trace()  # v = ∑ (v · e_j / ||e_j||^2) * e_j
            rep[idx, j, k] = value
            rep[idx, k, j] = -value
    return chunk, rep

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('fork')) as executor:
    exec_map = executor.map(_wrap_run_job, chunks)
    results = tuple(circuit for circuit in exec_map)

rep = np.zeros((len(gtilde), len(gtilde), len(gtilde)), dtype=float)
for chunk, repi in results:
    rep[range(*chunk)] = repi

adjoint_repr_alt = rep
adjoint_repr = qml.structure_constants(Moment[pick_moment])
np.allclose(adjoint_repr_alt, adjoint_repr)

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
