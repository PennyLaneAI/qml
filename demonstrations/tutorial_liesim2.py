r"""(g + P)-sim: Extending g-sim by non-DLA observables and gates
=================================================================

In a previous demo we introduced :math:`\mathfrak{g}`-sim, a Lie-theoretic simulation technique
of observables and gates comprised of Lie algebra elements. We extend :math:`\mathfrak{g}`-sim by
non-DLA elements comprised of products of DLA elements.

.. note::
    
    The contents of this demo are self-contained. However, we highly recommend reading our previous demos on
    :doc:`(dynamical) Lie algebras </demos/tutorial_liealgebra/` and :doc:`g-sim in PennyLane </demos/tutorial_liesim/`.

Introduction
------------

asd

:math:`\mathfrak{g}`-sim
------------------------

asd

(:math:`\mathfrak{g}`+P)-sim
----------------------------

asd

PennyLane implementation
------------------------

asd

"""

import pennylane as qml
from pennylane import X, Y, Z, I
from pennylane.pauli import PauliSentence, PauliWord, PauliVSpace
import numpy as np

from scipy.linalg import expm

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

##############################################################################
#
# DLA
# ~~~
#
# We take the generators from the transverse-field Ising model (TFIM) in 1D with nearest neighbors.
# This is one of the few systems that yield a polynomially sized DLA.

# TFIM generators
n = 4
generators = [
    X(i) @ X(i+1) for i in range(n-1)
]
generators += [
    Z(i) for i in range(n)
]
generators = [op.pauli_rep for op in generators]

dla = qml.pauli.lie_closure(generators, pauli=True)
dim_g = len(dla)

##############################################################################
#
# Moments
# ~~~~~~~
#
# We now compute the higher moments of DLA operators. A :math:`t`-th moment is given
# by a product of :math:`n+1` operators. E.g. :math:`h_j` is a :math:`0`-th moment,
# :math:`(h_i h_j) \notin \mathfrak{g}` is a :math:`1`st moment and so on.
# For that we start from the 
# vector space of :math:`\mathfrak{g}` itself and keep adding linearly independent
# operators comprised of products.

def Moment_step(ops):
    MomentX = PauliVSpace(ops.copy())
    for i, op1 in enumerate(dla):
        for op2 in ops[i+1:]:
            prod = op1 @ op2
            pw = next(iter(prod.keys())) # ignore scalar coefficient

            MomentX.add(pw)
    
    return MomentX.basis
Moment0 = dla.copy()
Moment = [Moment0]
dim = [len(Moment0)]
for i in range(1, 3):
    Moment.append(Moment_step(Moment[-1]))
    dim.append(len(Moment[-1]))

dim, 4**n-1

##############################################################################
#
# It is important to recall that the moments generally do not form a Lie algebra. This is because
# they are not closed under commutation, which can be seen by comparing the dimension with
# that of its Lie closure.

Moment1_closure = qml.lie_closure(Moment[1])
len(Moment1_closure), len(Moment[1])

##############################################################################
#
# So we see that the first moments do not form a Lie algebra. However, they still form a valid
# vector space. This is important because this lets us use their pseudo structure constants
# to form the adjoint action of the associated gates.
#
# We now compute the initial expectation value vector :math:`\vec{e}` for not just the DLA
# but the degree of moments we are considering. For simplicity, let us just use the first moments
# (which will permit us to run :math:`(\mathfrak{g} + P)`-sim with one :math:`P`-gate later).

pick_moment = 1 # order of moments (=number of P gates)

# compute initial expectation vector
e_in = np.zeros(dim[pick_moment], dtype=float)

for i, h_i in enumerate(Moment[pick_moment]):
    # initial state |0x0| = (I + Z)/2, note that trace function
    # below already normalizes by the dimension,
    # so we can ommit the explicit factor /2
    rho_in = qml.prod(*(I(i) + Z(i) for i in h_i.wires))
    rho_in = rho_in.pauli_rep

    e_in[i] = (h_i @ rho_in).trace()

##############################################################################
#
# We can now run :math:`(\mathfrak{g} + P)`-sim with random circuits that contain
# `one` P gate (that we also randomly set here)

depth = 10
coeff = np.random.rand(depth)

adjoint_repr = qml.structure_constants(Moment[pick_moment])

for i in range(10):
    gate_choice = np.random.choice(dim_g, size=depth)
    # one (random) P gate at the center of the
    gate_choice[depth//2] = np.random.choice(range(dim[0], dim[pick_moment]), size=(1,))[0]
    gates = adjoint_repr[gate_choice]

    # simulation
    e_t = e_in.copy()
    for i in range(depth):
        e_t = expm(coeff[i] * gates[i]) @ e_t

    # final expectation value

    # H = 0.5 @ generators (not full dla)
    weights = np.zeros(dim[pick_moment], dtype=complex)
    weights[:len(generators)] = 0.5 

    result_g_sim = weights @ e_t

    H = 0.5 * qml.sum(*[op.operation() for op in generators])

    @qml.qnode(qml.device("default.qubit"))
    def qnode():
        for i, mu in enumerate(gate_choice):
            qml.exp(
                -1j * coeff[i] * Moment[pick_moment][mu].operation()
            )
        return qml.expval(H)

    true_res = qnode()
    if not np.allclose(result_g_sim, true_res):
        print(f"FAIL: g-sim res: {result_g_sim}, exact res: {true_res}")
    else:
        print(f"SUCCESS: g-sim res: {result_g_sim}, exact res: {true_res}")

##############################################################################
#
# The bottleneck here is the computation of the adjoint action via :func:`~pennylane.structure_constants`
# that iterates over :math:`\tilde{d}^3/2` combinations, where `\tilde{d}` is the dimension of the vector space
# of the moments we are considering. Instead of computing the adjoint representation for every element of the moment space
# we can also directly compute the adjoint action of every gate as follows.
#
# The cost here is :math:`D\tilde{d}^2`, where :math:`D` is the depth of the circuit. So this is worth while whenever 
# :math:`D \leq \tilde{d}/2`. Additionally, because we are dealing with Pauli words, we can avoid computing the exponential
# of the adjoint representation because there is an efficient formula
# :math:`e^{-i \theta \otimes_j P_j} = \cos(\theta) \mathbb{I} -i \sin(\theta) \otimes_j P_j` for the exponential of them.

def exppw(theta, ps):
    # assert that it is indeed a pure pauli word, not a sentence
    assert (len(ps) == 1 and isinstance(ps, PauliSentence)) or isinstance(ps, PauliWord)
    return np.cos(theta) * PauliWord({}) + 1j * np.sin(theta) * ps

for i in range(10):
    gate_choice = np.random.choice(dim_g, size=depth)
    gate_choice[depth//2] = np.random.choice(range(dim[0], dim[pick_moment]), size=(1,))[0] # one product gate
    gates = np.array(Moment[pick_moment])[gate_choice].tolist()
    # Compute adjoint actions of all gates
    adj_gates = []

    for i, t in enumerate(gates):
        theta = coeff[i]

        T = exppw(theta, t)
        Td = exppw(-theta, t) # complex conjugate

        T2 = np.zeros((dim[pick_moment], dim[pick_moment]), dtype=float)

        for i, h1 in enumerate(Moment[pick_moment]):
            res = T @ h1 @ Td
            for j, h2 in enumerate(Moment[pick_moment]):
                # decompose the result in terms of DLA elements
                # res = ∑ (res · h_j / ||h_j||^2) * h_j 
                value = (res @ h2).trace().real
                value = value / (h2 @ h2).trace()
                T2[i, j] = value

        adj_gates.append(T2)

    # simulation
    e_t = e_in.copy()
    for i in range(depth):
        e_t = adj_gates[i] @ e_t

    # final expectation value

    # H = 0.5 @ generators (not full dla)
    weights = np.zeros(dim[pick_moment], dtype=complex)
    weights[:len(generators)] = 0.5 

    result_g_sim = weights @ e_t

    H = 0.5 * qml.sum(*[op.operation() for op in generators])

    @qml.qnode(qml.device("default.qubit"))
    def qnode():
        for i, mu in enumerate(gate_choice):
            qml.exp(
                -1j * coeff[i] * Moment[pick_moment][mu].operation()
            )
        return qml.expval(H)

    true_res = qnode()
    if not np.allclose(result_g_sim, true_res):
        print(f"FAIL: g-sim res: {result_g_sim}, exact res: {true_res}")
    else:
        print(f"SUCCESS: g-sim res: {result_g_sim}, exact res: {true_res}")


##############################################################################
# Alternatively, we can also speed up the process of the :func:`~pennylane.structure_constants`
# computation by making use of the fact that all computations in the outer loop are independent
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

adjoint_repr2 = rep
np.allclose(adjoint_repr2, adjoint_repr)

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
# .. [#Kottmann]
#
#     Korbinian Kottmann
#     "Introducing (Dynamical) Lie Algebras for quantum practitioners"
#     `PennyLane Demos <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__, 2024.
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
# .. [#Cerezo]
#
#     M. Cerezo, Martin Larocca, Diego García-Martín, N. L. Diaz, Paolo Braccia, Enrico Fontana, Manuel S. Rudolph, Pablo Bermejo, Aroosa Ijaz, Supanut Thanasilp, Eric R. Anschuetz, Zoë Holmes
#     "Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink variational quantum computing"
#     `arXiv:2312.09121 <https://arxiv.org/abs/2312.09121>`__, 2023.
#
# .. [#Wiersema]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
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
