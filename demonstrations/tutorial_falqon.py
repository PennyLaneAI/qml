r"""
Feedback-Based Quantum Optimization (FALQON)
=========================

.. meta::
    :property="og:description": Solve combinatorial problems with no classical optimizer
    :property="og:image": https://pennylane.ai/qml/_images/qaoa_layer.png

.. related::

   tutorial_qaoa_intro Intro to QAOA
   tutorial_qaoa_maxcut QAOA for MaxCut

*Authors: David Wakeham and Jack Ceroni. Posted: XXX. Last updated: XXX.*

-----------------------------

Note: Before reading this tutorial, we strongly recommend that you check out the
`Intro to QAOA tutorial <https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html>`__, as many of the same ideas
carry over, and will be assumed throughout this demonstration!

-----------------------------

While the `Quantum Approximate Optimization Algorithm (QAOA) <https://arxiv.org/pdf/1411.4028.pdf>`__
is one of the best-known processes for solving combinatorial optimization problems with quantum computers,
it has drawbacks. Since QAOA relies on optimization over a parameter space,
convergence isn't guaranteed, and the optimization can become "stuck" in local minima

.. figure:: ../demonstrations/falqon/global_min.png
    :align: center
    :width: 70%

In this demo, we'll be implementing the FALQON algorithm: a feedback-based algorithm
for quantum optimization, introduced by `Magann, Rudinger, Grace & Sarovar (2021) <https://arxiv.org/pdf/2103.08619.pdf>`__.
It is similar in spirit to QAOA, but instead uses iterative feedback steps rather than a global optimization
over parameters. We will show how to implement FALQON in PennyLane
and test its performance on the **MaxClique** problem in graph theory!

Theory
------

To solve combinatorial optimization problems using a quantum computer, a typical strategy is to encode
the solution to the problem as the ground state of *cost Hamiltonian* :math:`H_c`, and choose some strategy to drive
the system from a known initial state into this ground state. FALQON falls under this broad scheme!

Imagine a quantum system governed by a Hamiltonian of the form :math:`H = H_c + \beta(t) H_d`. These kinds of
Hamiltonians come up quite often in the theory of `quantum control <https://quantiki.org/wiki/quantum-control-theory>`__,
which studies how we may go about driving a quantum system from one state to another. The choice of :math:`\beta(t)` allows
us to decide which state we want a system governed by such a Hamiltonian to evolve towards.

The time-dependent Schrodinger equation tells us that the dynamics of the system are given by:

.. math:: i \frac{d}{dt} |\psi(t)\rangle = (H_c + \beta(t) H_d) |\psi(t)\rangle

where we set :math:`\hbar = 1`. Now, let us suppose that the objective of our quantum control experiment is to drive our system
to the state :math:`|\psi\rangle`: the ground state of :math:`H_c`. Phrased differently, we would like to minimize the expectation
value :math:`\langle H_c\rangle`. Therefore, a reasonable goal is to construct our system such that the expectation decreases with time:

.. math:: \frac{d}{dt} \langle H_c\rangle_t = \frac{d}{dt} \langle \psi(t)|H_c|\psi(t)\rangle = i \beta(t)\langle [H_d, H_c] \rangle_t \leq 0

where we used the product rule and Schrodinger's equation. Recall that our control experiment depends on the choice of :math:`\beta(t)`. Thus,
if we pick :math:`\beta(t) = -\langle i[H_d, H_c] \rangle_t`, so that

.. math:: \frac{d}{dt} \langle H_c\rangle_t = -|\langle i[H_d, H_c] \rangle_t|^2 \leq 0

then :math:`\langle H_c \rangle` is guaranteed to strictly decrease, as desired!
(Note that we bring the :math:`i` into the expectation to give a Hermitian operator.)

Using `techniques from control theory <https://arxiv.org/pdf/1304.3997.pdf>`__, it is possible to rigorously show this will
eventually drive the system into the ground state! Thus, if we evolve some initial state :math:`|\psi_0\rangle` under the
time-evolution operator corresponding to :math:`H`, given by :math:`U(t) = e^{-iHt}`, then we will arrive at the ground state of :math:`H_c`!
This is exactly the procedure used by FALQON!"""

######################################################################
# Simulating FALQON with PennyLane
# --------------------------------
# To begin, we import the necessary dependencies:
#

import pennylane as qml
import numpy as np
from matplotlib import pyplot as plt
from pennylane import qaoa as qaoa
import networkx as nx

######################################################################
# FALQON is an algorithm for combinatorial optimization. Therefore, we must select some kind of combinatorial problem
# to solve, using FALQON. In this demonstration, we will be solving the
# `maximum clique (MaxClique) problem <https://en.wikipedia.org/wiki/Clique_problem>`__: finding the
# largest complete subgraph of some graph :math:`G`. For example, the following graph's maximum clique is coloured in red:
#
# .. figure:: ../demonstrations/falqon/max_clique.png
#     :align: center
#     :width: 90%
#
# In this demonstration, we attempt to find the maximum cliques (there are two) of the following graph:
#

edges = [(0, 1), (1, 2), (2, 0), (2, 3), (1, 4)]
graph = nx.Graph(edges)
nx.draw(graph)

######################################################################
# We must first encode this combinatorial problem into a cost Hamiltonian :math:`H_c`. This ends up being given by:
#
# .. math:: H_c = 3 \sum_{(i, j) \in E(\bar{G})} (Z_i Z_j - Z_i - Z_j) + \displaystyle\sum_{i \in V(G)} Z_i
#
# where each qubit is a node in the graph, and the states :math:`|0\rangle` and :math:`|1\rangle` represent if the vertex
# has been "marked" as part of the clique, as is the case for most standard QAOA encoding schemes.
#
# In addition, to :math:`H_c`, we also require a driver Hamiltonian :math:`H_d`, which doesn't commute with :math:`H_c`, and
# is able to "mix up" our system usfficiently so that it may be driven towards the ground state (similar to the mixer Hamiltonian in QAOA).
# To keep things simple, we choose a sum over Pauli-X operations in each qubit:
#
# .. math:: H_{D} = \displaystyle\sum_{i \in V(G)} X_i
#
# These Hamiltonians come nicely bundled together in the PennyLane QAOA module:
#

cost_h, driver_h = qaoa.max_clique(graph, constrained=False)

print("Cost Hamiltonian")
print(cost_h)
print("Driver Hamiltonian")
print(driver_h)

######################################################################
# As you may recall from the Theory section, one of the main ingredients in the FALQON algorithm is the operator :math:`i [H_d, H_c]`. In
# the case of MaxClique, we can write down the commutator :math:`[H_d, H_c]` explicitly, exploiting its bilinearity:
#
# .. math:: [H_d, H_c] = \Big[ \displaystyle\sum_{i \in V(G)} X_i, 3 \sum_{(i, j) \in E(\bar{G})} (Z_i Z_j - Z_i - Z_j) +
#           \displaystyle\sum_{i \in V(G)} Z_i \Big]
# .. math:: = 3 \displaystyle\sum_{k \in V(G)} \displaystyle\sum_{(i, j) \in E(\bar{G})} \big( [X_k, Z_i Z_j] - [X_k, Z_i]
#           - [X_k, Z_j] \big) + 3 \displaystyle\sum_{i \in V(G)} \displaystyle\sum_{j \in V(G)} [X_i, Z_j]
#
# Clearly, there are two distinct commutators that we must calculate, :math:`[X_k, Z_j]` and :math:`[X_k, Z_i Z_j]`.
# This is easy, as we know exactly what the
# `commutators of the Pauli matrices <https://en.wikipedia.org/wiki/Pauli_matrices#Commutation_relations>`__ are.
# This gives us:
#
# .. math:: [X_k, Z_j] = -2 i \delta_{kj} Y_k \ \ \ \text{and} \ \ \ [X_k, Z_i Z_j] = -2 i \delta_{ik} Y_k Z_j - 2i \delta_{jk} Z_i Y_k
#
# where :math:`\delta_{kj}` is the `Kronecker delta <https://en.wikipedia.org/wiki/Kronecker_delta>`__. Therefore, it follows that from substitution
# into the above equation, and multiplication by :math:`i` that:
#
# .. math:: i [H_d, H_c] = 6 \displaystyle\sum_{k \in V(G)} \displaystyle\sum_{(i, j) \in E(\bar{G})} \big( \delta_{ki} Y_k Z_j +
#          \delta_{kj} Z_{i} Y_{k} - \delta_{ki} Y_k - \delta_{kj} Y_k \big) + 6 \displaystyle\sum_{i \in V(G)} Y_{i}
#
# This new operator has quite a few terms! Therefore, we write a short method which computes it for us, and return a :class:`~.pennylane.Hamiltonian` object:
#

def build_commutator(graph):
    H = qml.Hamiltonian([], [])

    # Computes the complement of the graph
    graph_c = nx.complement(graph)

    for k in graph_c.nodes:
        # Adds the terms in the first sum
        for edge in graph_c.edges:
            i, j = edge
            if k == i:
                H += 6 * (qml.PauliY(k) @ qml.PauliZ(j) - qml.PauliY(k))
            if k == j:
                H += 6 * (qml.PauliZ(i) @ qml.PauliY(k) - qml.PauliY(k))
        # Adds the terms in the second sum
        H += 6 * qml.PauliY(k)

    return H


print("MaxClique Commutator")
print(build_commutator(graph))

######################################################################
# We now must build the actual FALQON algorithm. Our goal is to evolve some initial state under the Hamiltonian :math:`H`, with our
# chosen :math:`\beta(t)`. In general, implementing a time-evolution unitary of the form :math:`U(t) = e^{-iHt}` in a quantum circuit
# is difficult, so we use a Trotter-Suzuki decomposition to perform approximate time-evolution:
#
# .. math:: U(t) \approx U_d(\beta_n) U_c U_d(\beta_{n-1}) U_c\cdots U_d(\beta_1) U_c, \quad U_c =
#           e^{-iH_c \Delta t}, \quad U_D(\beta_k) = e^{-i\beta_k H_d \Delta t}
#
# where :math:`\Delta t` is a small step in time, and :math:`\beta_k = \beta(k\Delta t)`.
# We can now easily define a layer of the FALQON circuit, which is of the form :math:`U_d(\beta_k) U_c`:

def falqon_layer(beta_k, cost_h, driver_h, delta_t):
    qaoa.cost_layer(delta_t, cost_h)
    qaoa.mixer_layer(delta_t * beta_k, driver_h)

######################################################################
# We then define a method which returns a FALQON ansatz corresponding to a particular cost Hamiltonian, driver
# Hamiltonian, and :math:`\Delta t`. This simply involves repeating the FALQON layer multiple times. Note that the
# initial state of our circuit is an even superposition:

def build_maxclique_ansatz(cost_h, driver_h, delta_t):
    def ansatz(beta, **kwargs):
        layers = len(beta)
        for w in dev.wires:
            qml.Hadamard(wires=w)
        qml.layer(
            falqon_layer,
            layers,
            beta,
            cost_h=cost_h,
            driver_h=driver_h,
            delta_t=delta_t
        )

    return ansatz

######################################################################
# Finally, before we put everything together and run the circuit, there is still one issue that must be addressed.
# Namely, for each layer, we need to know the value :math:`\beta_k`, is dependent on the state of the system, at some time, as
# we have defined:
#
# .. math:: \beta(t) = - \langle \psi(t) | i [H_d, H_c] | \psi(t) \rangle
#
# Our strategy is to use the value of :math:`A(t) := i\langle [H_d, H_c] \rangle_t` obtained by evaluating the circuit for the previous time-step:
#
# .. math:: \beta_{k+1} = -A_k = -A(k\Delta t)
#
# This leads immediately to the FALQON algorithm as a recursive process that feeds back into itself.
# On step $k$, we perform the following three substeps:
#
# 1. Prepare the state :math:`|\psi_k\rangle = U_d(\beta_k) U_c \cdots U_d(\beta_1) U_c|\psi_0\rangle`,
# 2. Measure the expectation value :math:`A_k = \langle i[H_c, H_d]\rangle_k`.
# 3. Set :math:`\beta_{k+1} = -A_k`.
#
# We repeat for all :math:`k` from :math:`1` to :math:`n`. At the final step, we evaluate :math:`\langle H_c \rangle`.
#
# .. figure:: ../demonstrations/falqon/falqon.png
#     :align: center
#     :width: 90%
#
# Implementing this recursive process isn't too difficult, we simply make use of the methods defined above:

def max_clique_falqon(graph, n, beta_1, delta_t, dev):
    hamiltonian = build_commutator(graph) # Builds the commutator
    cost_h, driver_h = qaoa.max_clique(graph, constrained=False) # Builds H_c and H_d
    ansatz = build_maxclique_ansatz(cost_h, driver_h, delta_t) # Builds the FALQON ansatz

    beta = [beta_1]
    energies = []

    for i in range(n):
        # Creates a function which can evaluate the expectation value of the commutator
        cost_fn = qml.ExpvalCost(ansatz, hamiltonian, dev)

        # Creates a function which returns the expectation value of the cost Hamiltonian
        cost_fn_energy = qml.ExpvalCost(ansatz, cost_h, dev)

        # Adds a value of beta to the list and evaluates the cost function
        beta.append(-1 * cost_fn(beta))
        energy = cost_fn_energy(beta)
        energies.append(energy)

        print("Step {} Done, Cost = {}".format(i + 1, energy))

    return beta, energies

######################################################################
# Note that we return both the list of :math:`\beta_k` values, as well as the expectation value of the cost Hamiltonian
# for each step.
#
# Without further delay, we can run FALQON for our MaxClique problem! It is important that we choose :math:`\Delta t` small enough
# such that our approximate time-evolution is close enough to the real time-evolution, otherwise we the expectation value of
# :math:`H_c` may not strictly decrease. For this demonstration, we set :math:`\Delta t = 0.03`, :math:`n = 40`, and :math:`\beta_1 = 0`:

n = 40
beta_1 = 0.0
delta_t = 0.03

dev = qml.device("default.qubit", wires=graph.nodes) # Creates a device for the simulation
res_beta, res_energies = max_clique_falqon(graph, n, beta_1, delta_t, dev)

######################################################################
# As expected, the expectation value of the cost Hamiltonian strictly decreases!
#

plt.plot(range(n+1)[1:], res_energies)

######################################################################
# To get a better idea of how well
# FALQON did, we can create a graph showing the probability of measuring each possible bitstring. We create
# following circuit, feeding in the optimal values of :math:`\beta_k`:

@qml.qnode(dev)
def prob_circuit():
    build_maxclique_ansatz(cost_h, driver_h, delta_t)(res_beta)
    return qml.probs(wires=dev.wires)

######################################################################
# Running this circuit gives us the following probability distribution:
#

probs = prob_circuit()
plt.bar(range(2**len(dev.wires)), probs)

######################################################################
# Clearly, the bitstring occurring with the highest probability is the state :math:`|28\rangle = |11100\rangle`.
# This is precisely the triangle in our graph, which is clearly the maximum clique.
# FALQON has solved the MaxClique problem!
#

######################################################################
# Benchmarking FALQON
# -------------------
#
# To conclude this demonstration, we will benchmark FALQON.

######################################################################
# Bonus: Seeding QAOA with FALQON (Bird Seed)
# -------------------------------------------
#
# QAOA and FALQON have many similarities, most notably, their circuit structure. Both involve alternating layers
# of time-evolution operators corresponding to a cost and a mixer/driver Hamiltonian. As it turns out, this
# will allow us to combine FALQON and QAOA to make something even more powerful!
#
#
#
