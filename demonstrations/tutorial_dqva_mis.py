r""".. _dqva_mis:

Dynamic Quantum Variational Ansatz (DQVA) for Combinatorial Optimization
========================================================================

.. meta::
    :property="og:description": Dynamic Quantum Variational Ansatz (DQVA) for Combinatorial Optimization
    :property="og:image": https://pennylane.ai/qml/_images/mixer-unitary.png

.. related::
   tutorial_qaoa_intro Introduction to QAOA
   tutorial_qaoa_maxcut QAOA for MaxCut

*Author: Priya Angara*

This demo discusses the `Dynamic Quantum Variational Ansatz (DQVA) <https://arxiv.org/abs/2010.06660>`__ [#Saleem2020]_ and the `Quantum Alternating Operator Ansatz (QAO-Ansatz) <https://arxiv.org/abs/1709.03489)>`__ [#Hadfield2019]_ for constrained quantum approximate optimization in the context of solving the Maximum Independent Set problem. 

"""

######################################################################
#QAO-Ansatz and DQVA
# --------------------------
#
# The **Quantum Approximate Optimization Algorithm (QAOA)**, a hybrid
# quantum-classical technique due to `Farhi et
# al. <https://arxiv.org/abs/1411.4028>`__ [#Farhi2015]_  is an approach to solving combinatorial optimization problems using quantum computers. The
# quantum part evaluates the objective function and involves alternating between a *cost Hamiltonian*,
# :math:`H_C`, and a *mixer Hamiltonian*, :math:`H_M`. A classical optimization loop updates the ansatz
# parameters. You can learn more about this in `PennyLane's tutorial on
# QAOA <https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html>`__.
#
# For solving constrained combinatorial optimization problems such as maximal independent set (MIS) 
# using QAOA, constraints can be imposed in two ways:
#
# 1. By adding a penalty term to the problem's objective function (the Lagrange multiplier approach)
# 2. By constructing the variational ansatz in a way such that constraints are satisfied at all times
#
# Hadfield et al., extend QAOA by introducing the **Quantum
# Alternating Operator Ansatz (QAO-Ansatz)** which alternates between more
# general families of unitary operators. This has the potential to narrow the focus 
# of the algorithm on a more useful set of states. For
# example, the mixing operator :math:`U_M (\beta)` can be a product
# of partial mixers :math:`U_{M, x}(\beta)` which may not commute. 
# In contrast, the original proposal for QAOA used a mixing operator
# which was a simple product of single-qubit gates.
#
# .. figure:: ../demonstrations/dqva_mis/partial-mixers.png
#    :align: center
#    :width: 90%
#
#    ..
#
#    Structure of a partial mixer
#
# The QAO-Ansatz can be used to guarantee that the state of the circuit never leaves the set of
# feasible states. In contrast, the penalty term approach requires an
# additional pruning step since the output can correspond to an
# infeasible solution. However, the QAO-Ansatz requires more complicated
# quantum circuits. In the case of MIS, one must apply
# Multi-Controlled Toffoli gates that require high connectivity between
# qubits, limiting the practicality of the QAO-Ansatz for large graphs on 
# near-term quantum computers.
#
# This brings us to the **Dynamic Quantum Variational Ansatz (DQVA)** that
# maximizes the performance of the QAO-Ansatz by:
#
# 1. *Warm starting the optimization* by starting with an initial state that is a feasible state
# or a superposition of feasible states. A feasible state can be found by
# using a classical approximate polynomial-time algorithm.
# 2. *Updating the ansatz dynamically* by turning partial mixers *on* and *off*
# 3. *Randomizing the ordering of the partial mixers* in the mixer unitaries
#
# Now that we have the basics in place, let's implement the QAO-Ansatz and
# the DQVA!


######################################################################
# Formulating the Maximum Independent Set (MIS) problem
# -----------------------------------------------------
#
# Given a graph, :math:`G = (V, E)` where :math:`V` is the set of vertices
# and :math:`E` is the set of edges, an *independent set* is a subset of
# vertices, :math:`V' \subset V`, such that no two vertices in :math:`V'`
# share an edge. A *Maximum Independent Set* (MIS) is an independent set of
# the largest possible size. In this demo, our objective is to find the
# MIS of a given graph. The following image shows some independent sets of
# a graph with 5 vertices:
#
# .. figure:: ../demonstrations/dqva_mis/independent-sets.png
#    :align: center
#    :width: 100%
#
#    ..
#
#    Independent sets of a graph with the MIS highlighted.
#
# Here, independent sets can be represented as bitstrings
# :math:`b = \{b_1, b_2, \cdots b_n\} \in \{0, 1\}^n`, where the i'th bit being one represents inclusion of the ith vertex in the set. This is convenient as the outputs of 
# quantum computations are also bitstrings!
# In the graph shown above :math:`00000`, :math:`10000`, :math:`01001`, :math:`10101` are
# independent sets of size 0, 1, 2, and 3 respectively.
#
# We will start with the QAO-Ansatz and then re-use most components to
# formulate the DQVA.
#
# First, we start with the necessary imports.
#

import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import copy
from typing import List, Optional


######################################################################
# Next, we define the graph shown above with 5 nodes for which we would
# like to find the Maximum Independent Set.
#

edges = [(0, 1), (0, 3), (1, 2), (1, 3), (3, 4)]
graph = nx.Graph(edges)
nx.draw(graph, with_labels=True)
plt.show()


######################################################################
# We will also instantiate a device with the number of wires being one
# more than the number of nodes (the extra wire is an ancilla).
#

wires = len(graph.nodes) + 1
dev = qml.device("default.qubit", wires=range(wires))


######################################################################
# For MIS, our goal is to maximize the number of vertices that form an
# independent set, which is equivalent to maximizing the `Hamming
# weight <https://en.wikipedia.org/wiki/Hamming_weight>`__ of the
# bitstring. Therefore, our objective function is encoded by the Hamming weight
# operator,
#
# .. math::
#
#
#    C_{obj} = H = \sum_{i \in V} b_i,
#
# where
#
# .. math::
#
#
#    b_i = \frac{1}{2} (I - Z_i).
#
# Here :math:`Z_i` is the Pauli-Z operator acting on the :math:`i`-th
# qubit.
#


######################################################################
# We need a couple of helper functions here: ``hamming_weight``, which
# calculates the Hamming weight of a given bitstring and ``is_indset``,
# which verifies whether a given bitstring is a valid independent set.
#


def hamming_weight(bitstr):
    return sum([1 for bit in bitstr if bit == "1"])


# def is_indset(bitstr, G):
#     nodes = list(G.nodes)
#     ind_set = []
#     for idx, bit in enumerate(bitstr):
#         if bit == "1":
#             cur_neighbors = list(G.neighbors(idx))
#             for node in ind_set:
#                 if node in cur_neighbors:
#                     return False
#             else:
#                 ind_set.append(idx)
#     return True

def is_indset(bitstr, G):
    for edge in list(G.edges):
        if bitstr[edge[0]] == "1" and bitstr[edge[1]] == "1":
            return False
    return True
######################################################################
# Next, we define a cost unitary that incorporates the Hamming weight operator,
# :math:`H` and is parameterized by :math:`\gamma`:
#
# .. math::
#
#
#    U_C(\gamma) = e^{i \gamma H}.
#


def cost_layer(gamma):
    for qb in graph.nodes:
        qml.RZ(2 * gamma, wires=qb)


######################################################################
# The mixer unitary for the QAO-Ansatz is defined as 
#
# .. math::
#
#
#       U_M(\beta) = \prod_j e^{i \beta M_j}
#
# where the product is over each node :math:`j`,
#
# .. math::
#
#
#    M_j = X_j \tilde{B},
#
# and
#
# .. math::
#
#
#    \tilde{B} = \prod_{k=1}^{l} \tilde{b}_{v_k}.
#
# Here, :math:`v_k` are the neighbors, :math:`l` is the number of
# neighbors for the :math:`j`-th node and
#
# .. math::
#
#
#    \tilde{b}_{v_j} = \frac{I+Z_{v_j}}{2}.
#
# The mixer unitary can also be written as a product of :math:`N` partial
# mixers :math:`V_i`:
#
# .. math::
#
#
#    U_M(\beta) = \prod_{i=1}^N V_i (\beta) =\prod_{i=1}^N  (I + (e^{-i\beta X_i} - I)\tilde{B}) 
#
# As mentioned earlier, the partial mixers may not commute with each other, i.e.,
# :math:`[V_i, V_j] \neq 0`. Therefore, the variational ansatz is defined
# up to a permutation:
#
# .. math::
#
#
#    U_M(\beta) \simeq \mathcal{P}(V_1(\beta)V_2(\beta)\cdots V_n(\beta))
#
# where P is the permutation's function of labels from :math:`1` to
# :math:`N`.
#
# These partial mixers are implemented using parameterized
# controlled-:math:`X` rotations sandwiched between two `multi-controlled
# Toffoli
# gates <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.MultiControlledX.html>`__.
# The number of qubits we require is one more than the number of vertices.
# This extra qubit is used as an ancillary qubit to serve as the target
# for the multi-controlled Toffoli gates.
#
# .. figure:: ../demonstrations/dqva_mis/mixer-unitary.png
#    :align: center
#    :width: 80%
#
#    ..
#
#    Formulation of the mixer unitary
#
# In the figure above, a PauliX rotation is applied to qubit :math:`\vert i \rangle` between two multi-controlled Toffoli gates
# controlled by the neighbors of the :math:`i`'th vertex.

def mixer_layer(beta: List, ancilla: int, mixer_order: Optional[List]=None):
    """
    Builds the QAO-Ansatz mixer layer for max independent set problem using PennyLane operations
    
    Args:
        beta (List): A list of values for the mixing parameter
        ancilla (int): The qubit to be used as the ancilla
        mixer_order (Optional[List]=None): The desired permutation of the partial mixers
        
    Returns: 
        None
    """
    # Permute the order of mixing unitaries
    if mixer_order is None:
        mixer_order = list(graph.nodes)

    for qubit in list(mixer_order):
        neighbors = list(graph.neighbors(qubit))

        # Apply the multi-controlled Toffoli, targetting the ancilla qubit
        ctrl_qubits = [i for i in neighbors]
        qml.MultiControlledX(
            control_wires=ctrl_qubits, wires=[ancilla], control_values="0" * len(neighbors)
        )

        qml.CRX(2 * beta, wires=[ancilla, qubit])

        # Uncompute the ancilla
        qml.MultiControlledX(
            control_wires=ctrl_qubits, wires=[ancilla], control_values="0" * len(neighbors)
        )

######################################################################
# We are now ready to build the circuit consisting of alternating cost and
# mixer layers. A list of ``params`` includes parameters for the mixer
# layer (even elements) and cost layer (odd elements).
#


def qaoa_ansatz(P, params=[], init_state=None, mixer_order=None):
    nq = len(graph.nodes)
    # Initialize
    if init_state is None:
        init_state = "0" * nq

    else:
        for qb, bit in enumerate(reversed(init_state)):
            if bit == "1":
                qml.PauliX(wires=qb)
    if len(params) != 2 * P:
        raise ValueError("Incorrect number of parameters!")

    betas = [a for i, a in enumerate(params) if i % 2 == 0]
    gammas = [a for i, a in enumerate(params) if i % 2 == 1]

    for beta, gamma in zip(betas, gammas):
        ancilla = nq  # Use the last qubit as an ancilla
        mixer_layer(beta, ancilla, mixer_order)
        cost_layer(gamma)


######################################################################
# Now, let's build the circuit that finds the probabilities of measuring
# each bitstring.
#


@qml.qnode(dev)
def probability_circuit(P, params=[], init_state=None, mixer_order=None):
    qaoa_ansatz(P, params, init_state, mixer_order)
    return qml.probs(wires=range(wires - 1))


######################################################################
# Finally, we run the quantum-classical loop for ``m`` different mixer
# permutations. The expectation value is also calculated in the cost
# function ``f`` by running the probability circuit and calculating the
# Hamming weight (which is defined as the cost). In each iteration, an
# ansatz is constructed based on the mixer order, with optimal parameters
# from classical minimization of the cost function using gradient descent:
#
# .. math::
#
#
#    \langle \beta, \gamma \vert C_{obj} \vert \beta, \gamma \rangle.
#


def solve_mis_qaoa(init_state, P=1, m=1, mixer_order=None, threshold=1e-5, cutoff=1):
    # Select an ordering for the partial mixers
    if mixer_order == None:
        cur_permutation = np.random.permutation(list(graph.nodes)).tolist()
    else:
        cur_permutation = mixer_order

    # Define the function to be optimized
    # Note that we are returning -cost since this is a minimization
    def f(params):
        probs = probability_circuit(
            P, params=params, init_state=cur_init_state, mixer_order=cur_permutation
        )
        avg_cost = 0
        for sample in range(0, len(probs)):

            x = [int(bit) for bit in list(np.binary_repr(sample, len(graph.nodes)))]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        return -avg_cost

    # Begin outer optimization loop
    best_indset = best_init_state = cur_init_state = init_state
    best_params = None
    best_perm = copy.copy(cur_permutation)

    # Randomly permute the order of mixer unitaries m times
    for mixer_round in range(1, m + 1):

        new_hamming_weight = hamming_weight(cur_init_state)
        inner_round = 1

        while inner_round < 2:
            print(
                f"Start round {mixer_round}.{inner_round}, Initial state = {cur_init_state}"
            )

            # Begin inner variational loop
            num_params = 2 * P
            print("\tNum params =", num_params)

            init_params = np.random.uniform(low=-np.pi, high=np.pi, size=num_params)
            print("\tCurrent Mixer Order:", cur_permutation)

            # Optimize parameters
            optimizer = qml.GradientDescentOptimizer(stepsize=0.5)
            cur_params = init_params.copy()

            for i in range(70):
                cur_params, opt_cost = optimizer.step_and_cost(f, cur_params)

            opt_params = cur_params.copy()

            print("\tOptimal cost:", opt_cost)

            # Obtain probabilites
            probs = probability_circuit(
                P, params=opt_params, init_state=cur_init_state, mixer_order=cur_permutation
            )

            # Sort bitstrings by decreasing probability
            top_counts = list(
                map(lambda x: np.binary_repr(x, len(graph.nodes)), np.argsort(probs))
            )[::-1]

            best_hamming_weight = hamming_weight(best_indset)
            better_strs = []

            print(top_counts[:cutoff])
            for bitstr in top_counts[:cutoff]:

                this_hamming = hamming_weight(bitstr)
                if is_indset(bitstr, graph) and this_hamming > best_hamming_weight:
                    better_strs.append((bitstr, this_hamming))

            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)

            # If no improvement was made, break and go to next mixer round
            if len(better_strs) == 0:
                print(
                    "\tNone of the measured bitstrings had higher Hamming weight than:", best_indset
                )
                break

            # Otherwise, save the new bitstring and repeat
            best_indset, new_hamming_weight = better_strs[0]
            best_init_state = cur_init_state
            best_params = opt_params.copy()
            best_perm = copy.copy(cur_permutation)
            cur_init_state = best_indset
            print(
                f"\tFound new independent set: {best_indset}, Hamming weight = {new_hamming_weight}"
            )

            inner_round = inner_round + 1
        # Choose a new permutation of the mixer unitaries
        cur_permutation = np.random.permutation(list(graph.nodes)).tolist()

    print("\tRETURNING, best hamming weight:", new_hamming_weight)
    return best_indset, best_params, best_init_state, best_perm


######################################################################
# Let's run this to find the MIS!
#

base_str = "0" * len(graph.nodes)

out = solve_mis_qaoa(base_str, P=1, m=4, threshold=1e-5, cutoff=1)
print(f"Init string: {base_str}, Best MIS: {out[0]}")
print()


######################################################################
# Starting with an all-zero initial string will give us an independent
# set, but this may not be the maximum. Now, we run this with a few
# different initial states and also increase the cutoff value. ``cutoff``
# indicates the number of bitstrings (sorted by the highest probabilities)
# that we consider that may improve the Hamming weight.
#

base_str = "0" * len(graph.nodes)
for i in range(len(graph.nodes)):
    init_str = list(base_str)
    init_str[i] = "1"
    out = solve_mis_qaoa("".join(init_str), P=1, m=4, threshold=1e-5, cutoff=2)
    print(f"Init string: {init_str}, Best MIS: {out[0]}")
    print()


######################################################################
# Dynamic Quantum Variational Ansatz
# ----------------------------------
# We will now formulate the MIS using the DQVA Ansatz [#Saleem2020]. The cost function
# is the same as the QAO-Ansatz (the Hamming weight operator).
#
# In the DQVA, the way mixers are defined is slightly different from the
# QAO-Ansatz and are allowed to be independent.
#
# .. math::
#
#
#    U_M^k(\alpha_k) = \mathcal{P}(V_1^k(\alpha_k^1)V_2^k(\alpha_k^2)\cdots V_N^k(\alpha_k^N))
#
# where :math:`k = 1, 2 \cdots p`.
#
# In the DQVA, whenever the :math:`j`-th bit of the initial state is one,
# the corresponding parameter :math:`\alpha_k^j` is set to 0. For example,
# if the initial state is :math:`\vert 01101 \rangle`, then
#
# .. math::
#
#
#    U_M^k(\alpha_k) = \mathcal{P}(V_1^k(\alpha_k^1) I_2 I_3 V_4^k(\alpha_k^4) I_5).
#
# Therefore, we are dynamically turning off parameters thereby improving
# the utilization of quantum resources.
#


def mixer_dqva(alpha, ancilla, init_state, mixer_order):

    # Permute the order of mixing unitaries
    if mixer_order is None:
        mixer_order = list(graph.nodes)

    pad_alpha = [None] * len(init_state)
    next_alpha = 0

    for qubit in mixer_order:
        bit = list(init_state)[qubit]
        if bit == "1" or next_alpha >= len(alpha):
            continue
        else:
            pad_alpha[qubit] = alpha[next_alpha]
            next_alpha += 1

    for qubit in mixer_order:

        if pad_alpha[qubit] == None or not graph.has_node(qubit):
            # Turn off mixers for qubits which are already 1
            continue

        neighbors = list(graph.neighbors(qubit))

        # Apply the multi-controlled Toffoli, targetting the ancilla qubit
        ctrl_qubits = [i for i in neighbors]
        qml.MultiControlledX(
            control_wires=ctrl_qubits, wires=[ancilla], control_values="0" * len(neighbors)
        )

        qml.CRX(2 * pad_alpha[qubit], wires=[ancilla, qubit])

        # Uncompute the ancilla
        qml.MultiControlledX(
            control_wires=ctrl_qubits, wires=[ancilla], control_values="0" * len(neighbors)
        )

######################################################################
# The structure of the cost and mixer unitaries is similar QAO-Ansatz,
# however, we now alternate between :math:`p` applications of the mixing
# unitary :math:`U_M^k(\alpha_k)` and :math:`p` applications of the cost
# unitary :math:`U_C^k(\gamma_k)`:
#
# .. math::
#
#
#    \vert \alpha, \gamma \rangle = U_C^p(\gamma_p)U_M^p(\alpha_p)\cdots U_C^1(\gamma_p)U_M^1(\alpha_1)\vert c_1 \rangle
#
# where :math:`\vert c_1 \rangle` is an initial state.
#


def dqva_ansatz(P, params=[], init_state=None, mixer_order=None):
    nq = len(graph.nodes)

    # Step1: Initialize
    if init_state is None:
        init_state = "0" * nq

    else:
        for qb, bit in enumerate(reversed(init_state)):
            if bit == "1":
                qml.PauliX(wires=qb)

    num_nonzero = nq - hamming_weight(init_state)
    assert len(params) == (nq + 1) * P, "Incorrect number of parameters!"
    alpha_list = []
    gamma_list = []
    last_idx = 0
    for p in range(P):
        chunk = num_nonzero + 1
        cur_section = params[p * chunk : (p + 1) * chunk]
        alpha_list.append(cur_section[:-1])
        gamma_list.append(cur_section[-1])
        last_idx = (p + 1) * chunk

    alpha_list.append(params[last_idx:])

    for i in range(len(alpha_list)):
        ancilla = nq
        alphas = alpha_list[i]
        mixer_dqva(alphas, ancilla, init_state, mixer_order)

        if i < len(gamma_list):
            gamma = gamma_list[i]
            cost_layer(gamma)



######################################################################
# Let's also define the probability circuit for the DQVA ansatz
#


@qml.qnode(dev)
def probability_dqva(P, params=[], init_state=None, mixer_order=None):
    dqva_ansatz(P, params, init_state, mixer_order)
    return qml.probs(wires=dev.wires[:-1])


######################################################################
# The quantum-classical loop for DQVA also maximizes
#
# .. math::
#
#
#    \langle \alpha, \gamma \vert C_{obj} \vert \alpha, \gamma \rangle.
#
# The dynamic ansatz update includes:
#
# 1. Optimization of parameters using Gradient Descent and finding the Hamming weight with new parameters
# 2. If the Hamming weight of the new state is larger than the initial state, the initial state gets updated to this new state
# 3. Based on this new state, partial mixers are updated (i.e., turned off for ones and turned on for zeros)
# 4. Steps 2 and 3 are repeated unti                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  l the Hamming weight can no longer be improved
# 5. If no new Hamming weight is obtained, the partial mixers are randomized and steps 2 and 3 are repeated to check if
# a better Hamming weight is found. The number of randomizations is controlled via a hyperparameter.
#
#

def solve_mis_dqva(init_state, P=1, m=1, mixer_order=None, threshold=1e-5, cutoff=1):

    # Select an ordering for the partial mixers
    if mixer_order == None:
        cur_permutation = np.random.permutation(list(graph.nodes)).tolist()
    else:
        cur_permutation = mixer_order

    def f(params):

        probs = probability_dqva(
            P, params=params, init_state=cur_init_state, mixer_order=cur_permutation
        )
        avg_cost = 0
        for sample in range(0, len(probs)):

            x = [int(bit) for bit in list(np.binary_repr(sample, len(graph.nodes)))]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        return -avg_cost

    # Begin outer optimization loop
    best_indset = init_state
    best_init_state = init_state
    cur_init_state = init_state
    best_params = None
    best_perm = copy.copy(cur_permutation)

    # Randomly permute the order of mixer unitaries m times
    for mixer_round in range(1, m + 1):

        inner_round = 1
        new_hamming_weight = hamming_weight(cur_init_state)

        while True:
            print(
                "Start round {}.{}, Initial state = {}".format(
                    mixer_round, inner_round, cur_init_state
                )
            )

            # Begin inner variational loop
            num_params = P * (len(graph.nodes()) + 1)
            print("\tNum params =", num_params)
            init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=num_params)
            print("\tCurrent Mixer Order:", cur_permutation)

            # Optimize parameters
            optimizer = qml.GradientDescentOptimizer(stepsize=0.5)
            cur_params = init_params.copy()

            for i in range(70):
                cur_params, opt_cost = optimizer.step_and_cost(f, cur_params)

            opt_params = cur_params.copy()

            print("\tOptimal cost:", opt_cost)

            probs = probability_dqva(
                P, params=opt_params, init_state=cur_init_state, mixer_order=cur_permutation
            )

            top_counts = list(
                map(lambda x: np.binary_repr(x, len(graph.nodes)), np.argsort(probs))
            )[::-1]

            best_hamming_weight = hamming_weight(best_indset)
            better_strs = []

            for bitstr in top_counts[:1]:
                this_hamming = hamming_weight(bitstr)
                if is_indset(bitstr, graph) and this_hamming > best_hamming_weight:
                    better_strs.append((bitstr, this_hamming))
            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)

            # If no improvement was made, break and go to next mixer round
            if len(better_strs) == 0:
                print(
                    "\tNone of the measured bitstrings had higher Hamming weight than:", best_indset
                )
                break

            # Otherwise, save the new bitstring and repeat
            best_indset, new_hamming_weight = better_strs[0]
            best_init_state = cur_init_state
            best_params = opt_params
            best_perm = copy.copy(cur_permutation)
            cur_init_state = best_indset
            print(
                "\tFound new independent set: {}, Hamming weight = {}".format(
                    best_indset, new_hamming_weight
                )
            )
            inner_round += 1

        # Choose a new permutation of the mixer unitaries
        cur_permutation = np.random.permutation(list(graph.nodes)).tolist()

    print("\tRETURNING, best hamming weight:", new_hamming_weight)
    return best_indset, best_params, best_init_state, best_perm


######################################################################
# We also run the algorithm for several initial states (instead of
# just one) to check what is the best Maximum Independent Set we obtain.
# The initial states are a set of independent sets of size one (this is
# trivial - any bitstring with a single "1" is a valid independent set)
#

base_str = "0" * len(graph.nodes)
all_init_strs = []
for i in range(len(graph.nodes)):
    init_str = list(base_str)
    init_str[i] = "1"
    out = solve_mis_dqva("".join(init_str), P=1, m=4, threshold=1e-5, cutoff=1)
    print("Init string: {}, Best MIS: {}".format("".join(init_str), out[0]))
    print()


######################################################################
# And voila, we have found an independent set of size 3 which is the state
# ``10101``!
#


######################################################################
# Conclusion
# ----------
#
# In this demo, we looked into solving a constrained combinatorial
# optimization problem using the QAO-Ansatz and the DQVA. The QAO-Ansatz
# defines mixer unitaries with partial mixers in such a way that we never
# leave the set of feasible states. However, this comes at a high cost of
# using multi-controlled Toffoli gates. To overcome this, we looked at the
# DQVA which turned partial mixers on and off, and randomized the partial
# mixer ordering. Furthermore, Saleem et al., recommend starting with a
# set of initial states that are independent sets found by any classical
# polynomial time approximation algorithm to "warm start" the
# initialization. They also propose the first useful application
# of `circuit-cutting techniques <https://arxiv.org/abs/2107.07532>`__ [#Saleem2021]_ to
# solve MIS, which involves classical partitioning of a large graph into
# sub-graphs, finding a solution using DQVA on a subgraph, and then
# preparing a set of states to be passed as an input, essentially
# stitching together solutions.
#


######################################################################
#
# References
# ----------
#
# .. [#Saleem2020] Z. H. Saleem, T. Tomesh, B. Tariq, M. Suchara. (2020) "Approaches to Constrained Quantum Approximate Optimization",
#     `arXiv preprint arXiv:2010.06660 <https://arxiv.org/abs/2010.06660>`__.
#
# .. [#Farhi2014] E. Farhi, J. Goldstone, S. Gutmann. (2014) "A Quantum Approximate Optimization Algorithm",
#     `arXiv preprint arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`__.
# 
# .. [#Hadfield2019] S. Hadfield,  Z. Wang, B. O’Gorman, E. Rieffel, D. Venturelli, R. Biswas. (2019) "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz",
#     `arXiv preprint arXiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__.
#
# .. [#Saleem2021] Z. H. Saleem, T. Tomesh, M. A. Perlin, P. Gokhale M. Suchara. (2021) "Quantum Divide and Conquer for Combinatorial Optimization and Distributed Computing",
#     `arXiv preprint arXiv:2107.07532 <https://arxiv.org/abs/2107.07532>`__.
