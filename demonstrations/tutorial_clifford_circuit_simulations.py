r"""
.. _clifford_circuit_simulations:

Efficient Simulation of Clifford Circuits
=========================================

.. meta::
    :property="og:description": This tutorial demonstrate how to efficiently simulate Clifford circuits with PennyLane.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/clifford_simulation/thumbnail_tutorial_clifford_simulation.png

.. related::

   tutorial_mbqc Measurement-based quantum computation
   tutorial_unitary_designs Unitary designs

.. figure:: ../_static/demonstration_assets/clifford_simulation/thumbnail_tutorial_clifford_simulation.jpg
   :align: center
   :width: 45%
   :target: javascript:void(0)

"""

#######################################################################
# In this tutorial, we take a deep dive into learning about Clifford gates and Clifford circuits,
# which are known to be efficiently classically simulable and play an essential role in the
# practical implementation of quantum computation. As a bonus, we will also see how to
# perform these simulations with PennyLane for circuits scaling up to thousands of qubits.
#

# Imports for the tutorial
from timeit import time
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

import pennylane as qml

qml.drawer.use_style("pennylane")

######################################################################
# Universal Gate Set
# ------------------
#

######################################################################
# Just like how in classical computation once can define a set logic gate operations
# ``{AND, NOT, OR}`` that can be used to perform any boolean function, in quantum computation as well,
# we define a universal set of quantum gates, ``{H, S, CNOT, T}``, with which one can approximate any
# unitary transformation to a desired accuracy. This gate set is also referred to as the *Clifford +
# T* owing to the fact that the elements ``{H, S, CNOT}`` are generators of the *Clifford group*
# :math:`\mathcal{C}` which is the
# `normalizer <https://en.wikipedia.org/wiki/Centralizer_and_normalizer>`__ of Pauli group
# :math:`\mathcal{P}`, :math:`\mathcal{C}_n = \{C \in U_{2^n}\ |\ C \mathcal{P}_n C^{\dagger} = \mathcal{P}_n\}`,
# i.e., its elements transforms :math:`n`-qubit *Pauli* operations to other *Pauli* operations.
#

######################################################################
# Clifford Gates
# ~~~~~~~~~~~~~~
#
# The elements of the *Clifford group* are called the *Clifford gates* and they include the
# following commonly used quantum gate operations supported in PennyLane -
#
# 1. Single-qubit Pauli gates: :class:`~.pennylane.I`, :class:`~.pennylane.X`, :class:`~.pennylane.Y`, :class:`~.pennylane.Z`
# 2. Other single-qubit gates: :class:`~.pennylane.S`, :class:`~.pennylane.H`
# 3. The two-qubit ``controlled`` Pauli gates: :class:`~.pennylane.CNOT`, :class:`~.pennylane.CY`, :class:`~.pennylane.CZ`
# 4. Other two-qubit gates: :class:`~.pennylane.SWAP`, :class:`~.pennylane.iSWAP`
# 5. Adjoints of the above gate operations via :func:`~pennylane.adjoint`
#
#
# Each of the *Clifford gates* can be uniquely visualized by a *Clifford Tableau*,
# which represents how they transform the Pauli words. Let us try to compute this tableau
# for some of the gates we have listed above.
#


def clifford_tableau(op):
    """Prints a Clifford Tableau representation for a given operation."""
    # set up Pauli operators
    num_wires = len(op.wires)
    pauli_ops = [
        [pauli(wire) for pauli in [qml.PauliX, qml.PauliZ]] for wire in range(num_wires)
    ]
    # conjugate the Pauli operators
    conjugate = [
        [
            qml.pauli_decompose(qml.prod(qml.adjoint(op), pauli, op).matrix(), pauli=True)
            for pauli in pauli_ops[wire]
        ]
        for wire in range(num_wires)
    ]
    # Print the tableau
    print(f"Tableau: {op.label()}({', '.join(map(str, op.wires))})")
    for pauli_op, conjug in zip(pauli_ops, conjugate):
        for pauli, conj in zip(pauli_op, conjug):
            phase = "+" if list(conj.values())[0] > 0 else "-"
            label = f"{pauli.label()}({', '.join(map(str, pauli.wires))})"
            print(label, "—>", phase, list(conj.keys())[0])


######################################################################

clifford_tableau(qml.Hadamard(0))  # Hadamard

######################################################################

clifford_tableau(qml.ISWAP([0, 1]))  # ISWAP

######################################################################
# As you see, we now have a definition of both ``Hadamard`` and ``ISWAP`` in terms of how they
# perform conjugation of Pauli words. This will come handy when we learn more about using such
# Tableau structure for simuling Clifford gates later on in this tutorial.
#


######################################################################
# Clifford Decomposition
# ----------------------
#
# In PennyLane, one can perform decomposition of any quantum circuit into the `Clifford + T`
# basis using the :func:`~pennylane.clifford_t_decomposition`. This transform under the hood,
# decomposes the entire circuit up to a desired operator norm error :math:`\epsilon` using
# :func:`~pennylane.ops.sk_decomposition` that employs an iter-recursive variant of the Solovay-Kitaev
# algorithm described in `Dawson and Nielsen (2005) <https://arxiv.org/abs/quant-ph/0505030>`__ .
# Let's see this in action for the following two-qubit parameterized circuit -
#

@qml.qnode(qml.device("default.qubit"))
def original_circuit(x, y):
    qml.RX(x, 0)
    qml.CNOT([0, 1])
    qml.RY(y, 0)
    return qml.probs()

######################################################################

x, y = np.pi / 2, np.pi / 4
qml.draw_mpl(original_circuit, decimals=2)(x, y)
plt.show()

######################################################################

unrolled_circuit = qml.transforms.clifford_t_decomposition(original_circuit)
qml.draw_mpl(unrolled_circuit, decimals=2)(x, y)
plt.show()

######################################################################
# In this *unrolled* quantum circuit, we can see that the non-Clifford rotation gates ``qml.RX`` and
# ``qml.RY`` at the either side of ``qml.CNOT`` have been replaced by the sequence of single-qubit
# Clifford gates depending on their parameter values. In order to ensure that the performed
# decomposition is correct, we can compare the measurement results of the unrolled and original
# circuit.
#

original_probs = original_circuit(x, y)
unrolled_probs = unrolled_circuit(x, y)
print(qml.math.allclose(original_probs, unrolled_probs, atol=1e-3))


######################################################################
# As we see, that the output of both the circuits are equivalent, which allows us to see how an
# arbirtary unitary transformation can be approximated using the universal gate set.
#


######################################################################
# Efficient Classical Simulability
# --------------------------------
#

######################################################################
# Efficiency of a classical computer to perform a quantum simulation, can be speculated
# by determining whether there exists a classical description for the simulation of the
# relevant quantum state, such that one can apply unitary operations to it and perform
# measurements from it efficiently in a polynomial number of operations. Therefore,
# efficient simulability of a problem relies on the fact that whether it requires some
# additional quantum resource that would inhibit such a description and hence would allow
# the showcase of an advantage [#supremecy_exp1]_, [#supremecy_exp2]_.
# 

######################################################################
# Gottesman-Knill theorem
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The quantum circuits that consist only of *Clifford gates* are called
# *Clifford group circuits* (or more generally *Clifford circuits*).
# These make up an extremely important class of circuits as they are efficiently
# classically simulable by *Gottesman-Knill* theorem, which says that :math:`n`-qubit
# Clifford circuits with :math:`m` Clifford gates can be simulated in time :math:`poly(m, n)`
# on a probabilistic classical computer. A key consequence that emerges from this is that
# the non-Clifford *T* gate represents the additional quantum resource required for
# universal quantum computation that would inhibit efficient classical simulability
# of a quantum circuit and therefore needed for a quantum advantage..
#

######################################################################
# Stabilizer Circuits
# -------------------
#

######################################################################
# Another important class of the circuit that are efficiently classically simulable via the
# Gottesman-Knill Theorem are the *stabilizer circuits*. These circuits are nothing but *Clifford
# circuits* with single-qubit measurment gates and the state such circuit can evolve to are called
# *stabilizer states*. These are of quite significance as they are commonly found in the literature
# related to quantum error correction [cite] and measurement-based quantum computation [cite]. So it
# is important to know how one can not just simulate these circuits efficiently but also to obtain
# quantities of interest from them. While there exist quite a few techniques that enable us to do so,
# one of the more popular ones is the `CHP formalism <https://quantum-journal.org/papers/q-2019-09-02-181/>`__\
# (or the *phase-sensitive* formalism), where we represent a stabilizer state as a subgroup.
# We store the *global phase* in addition to the *generators* of these subgroup as a *Tableau* and
# update them to replicate application of the Clifford gates on the state.
#

######################################################################
# Simulating with PennyLane
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# As noted in the previous section, studying stabilizer circuit is crucial for understanding theory
# of quantum computation, and hence it is crucial to have tools to do so. With this as motivation,
# we introduce a new ``default.clifford``
# `device <https://docs.pennylane.ai/en/latest/code/api/pennylane.devices.default_clifford.html>`_
# that enables efficient simulation of large-scale Clifford circuits defined in PennyLane through
# the use of `stim <https://github.com/quantumlib/Stim>`__ as an underlying backend [#stim]_,
# which is based on an improvised *CHP formalism* mentioned above. We can use it to run
# *Clifford circuits* in the same way we run any other normal circuit -
#

dev = qml.device("default.clifford", tableau=True, wires=2)

@qml.qnode(dev)
def circuit(ret_state=True):
    qml.PauliX(wires=[0])
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=[0])
    qml.Hadamard(wires=[1])
    return [
        qml.expval(op=qml.PauliX(0) @ qml.PauliX(1)),
        qml.var(op=qml.PauliZ(0) @ qml.PauliZ(1)),
        qml.probs(),
    ] + ([] if not ret_state else [qml.state()])


expval, var, probs, state = circuit(ret_state=True)
print(expval, var)

######################################################################
# One can use this device to obtain the usual range of PennyLane measurements like ``expval``,
# with or without shots, in addition to the state represented in the following stabilizer
# *Tableau* form [#aaronson-gottesman2004]_ -
#
# .. figure:: ../_static/demonstration_assets/clifford_simulation/stabilizer-tableau.jpeg
#    :align: center
#    :width: 90%
#    :target: javascript:void(0)
#
# Here, the first and the last :math:`n` rows of the represents the generators for the
# ``destabilizers`` and ``stabilizers`` for the state as a `binary
# vector <https://docs.pennylane.ai/en/latest/code/api/pennylane.pauli.binary_to_pauli.html>`__,
# respectively, and the last column contains the binary variable regarding the phase of each
# generator. We can obtain these evolved tableaus for the executed circuit using ``qml.state()``
# when the device is initialzied with ``tableau=True`` keyword argument. For example, the tableau
# for the above circuit is -
#

print(state)

######################################################################
# Looking at the tableaus in a matrix form could be difficult. Instead, one can look at the generator
# set of *destabilizers* and *stabilizers* which would uniquely represent each tableau. This provide a
# more human readable way of visualizing tableaus and the following methods help us do so -
#


def tableau_to_pauli_group(tableau):
    """Get stabilizers, destabilizers and phase from a Tableau"""
    num_qubits = tableau.shape[0] // 2
    stab_mat, destab_mat = tableau[num_qubits:, :-1], tableau[:num_qubits, :-1]
    stabilizers = [qml.pauli.binary_to_pauli(stab) for stab in stab_mat]
    destabilizers = [qml.pauli.binary_to_pauli(destab) for destab in destab_mat]
    phase = tableau[:, -1].T.reshape(-1, min(2, num_qubits))
    return stabilizers, destabilizers, phase


def tableau_to_pauli_rep(tableau):
    """Get a string representation for stabilizers and destabilizers from a Tableau"""
    wire_map = {idx: idx for idx in range(tableau.shape[0] // 2)}
    stabilizers, destabilizers, phase = tableau_to_pauli_group(tableau)
    stab_rep, destab_rep = [], []
    for index in wire_map:
        phase_rep = ["+" if not p else "-" for p in phase[:, index]]
        stab_rep.append(
            phase_rep[1] + qml.pauli.pauli_word_to_string(stabilizers[index], wire_map)
        )
        destab_rep.append(
            phase_rep[0] + qml.pauli.pauli_word_to_string(destabilizers[index], wire_map)
        )
    return {"Stabilizers": stab_rep, "Destabilizers": destab_rep}


tableau_to_pauli_rep(state)

######################################################################
# Now, let us briefly understand how does the stabilizer tableau formalism actually work. We will use
# the the *generator* set representation for describing the stabilizer state and *tableau*
# representation for the Clifford gate that is applied on it. So, to use the circuit described above,
# we first transform it in a way that we can access its state before and after application of gate
# operation using ``qml.snapshots``.
#


@qml.transform
def state_at_each_step(tape):
    """Transforms a circuit to access state after every operation"""
    num_ops = len(tape.operations)
    operations = list(
        it.chain.from_iterable(zip([qml.Snapshot()] * num_ops, tape.operations))
    ) + [qml.Snapshot()]
    new_tape = type(tape)(operations, tape.measurements, shots=tape.shots)

    def postprocessing(results):
        return results[0]

    return [new_tape], postprocessing


snapshots = qml.snapshots(state_at_each_step(circuit))()

######################################################################
# We can now access the tableau state via the ``snapshots`` dictionary, where the integer keys
# representing each step. The step ``0`` corresponds to the initial all zero :math:`|00\rangle` state
# -
#

print(snapshots[0])
print(tableau_to_pauli_rep(snapshots[0]))

######################################################################
# The evolution of the stabilizer tableau after application of each clifford gate operation can be
# understood by how the *generator* set is transformed based on the Clifford tableau that we saw
# previously. The first operation ``qml.PauliX(0)`` in the circuit has the following tableau -
#

clifford_tableau(qml.PauliX(0))

######################################################################
# Based on its Clifford tableau, we expect the evolution of tableau by ``qml.PauliX(0)`` to correspond
# to the transformation of its generators of stabilizers to ``-Z`` from ``+Z`` and destabilizers
# remaining the same. Let’s check if this is actually true by accessing the state at step ``1`` -
#

print(snapshots[1])
print(tableau_to_pauli_rep(snapshots[1]))

######################################################################
# So, to track and compute the evolved state one just simply need to know the transformation rules for
# the gate operation, which makes the tableau formalism much more efficient than the state vector
# formalism where a more computationally expensive matrix-vector multiplication has to be performed at
# each step. Let’s look at the remianing operations to confirm it -
#

circuit_ops = circuit.tape.operations
print(circuit_ops)

for step in range(1, len(circuit_ops)):
    print("--" * 5 + f" Step {step} - {circuit_ops[step]} " + "--" * 5)
    clifford_tableau(circuit_ops[step])
    print(f"Before - {tableau_to_pauli_rep(snapshots[step])}")
    print(f"After  - {tableau_to_pauli_rep(snapshots[step+1])}\n")

######################################################################
# Sampling with PennyLane
# ~~~~~~~~~~~~~~~~~~~~~~~
#

######################################################################
# In addition to the analytic results, one can use ``default.clifford`` for obtaining samples from the
# stabilizer circuits. We support all the standard sample-based measurements on this device as we do
# for ``default.qubit``. For example, let us the simulate circuit above with :math:`10,000` shots and
# compare the probability distribution with the analytic case -
#

sampled_result = circuit(ret_state=False, shots=10000)
sampled_expval, sampled_var = sampled_result[:2]

print(sampled_expval, sampled_var)

######################################################################

# Define computational basis states
basis_states = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

# Plot the probabilities
bar_width, bar_space = 0.25, 0.01
bar_original = plt.bar(
    np.arange(4), probs, width=bar_width, color="#C756B2", label="Analytic"
)
bar_unrolled = plt.bar(
    np.arange(4) + bar_width + bar_space, sampled_result[2],
    width=bar_width, color="#70CEFF", label="Statistical"
)

# Add bar labels
for bar in [bar_original, bar_unrolled]:
    plt.bar_label(bar, padding=1, fmt="%.3f", fontsize=8)

# Add labels and show
plt.title("Comparing Probabilities from Circuits", fontsize=11)
plt.xlabel("Basis States")
plt.ylabel("Probabilities")
plt.xticks(np.arange(4) + bar_width / 2, basis_states)
plt.ylim(0.0, 0.30)
plt.legend(loc="upper center", ncols=2, fontsize=9)
plt.show()


######################################################################
# As we see, the stochastic case matches pretty much with our analytic results, letting us be
# confident about our capabilities for sampling for stabilizers circuit.
#

######################################################################
# Benchmarking
# ------------
#

######################################################################
# Now that we have learnt that ``default.clifford`` can allow us to execute stabilizer circuits and
# compute various measurements of interest from them both analytically and stochastically, let us now
# try to benchmark its capabilities. To do so, we look at a set of experiments with the
# follwing :math:`n`-qubit
# `Greenberger-Horne-Zeilinger state <https://en.wikipedia.org/wiki/Greenberger-Horne-Zeilinger_state>`_
# (GHZ state) preparation circuit -

dev = qml.device("default.clifford")


@qml.qnode(dev)
def GHZStatePrep(num_wires):
    """Prepares the GHZ State"""
    qml.Hadamard(wires=[0])
    for wire in range(num_wires):
        qml.CNOT(wires=[wire, wire + 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(num_wires - 1))


print(GHZStatePrep(num_wires=6))

######################################################################
# In our experiments, we will vary the number of qubits to see how both does it
# impact the execution timings for the circuit in the anaylytic and finite-shots
# cases.
#

dev = qml.device("default.clifford")

num_shots = [None, 100000]
num_wires = [10, 100, 1000, 10000]

shots_times = np.zeros((len(num_shots), len(num_wires)))
for ind, num_shot in enumerate(num_shots):

    # Iterate over different number of qubits
    for idx, num_wire in enumerate(num_wires):
        exec_time = []
        for _ in range(5):
            start = time.time()
            GHZStatePrep(num_wires=num_wire, shots=num_shot)
            ended = time.time()
            exec_time.append(ended - start)

        shots_times[ind][idx] = np.mean(exec_time)

# Figure set up
fig = plt.figure(figsize=(10, 5))

# Plot the data
bar_width, bar_space = 0.3, 0.01
colors = ["#70CEFF", "#C756B2"]
labels = ["Analytical", "100k shots"]
for idx, num_shot in enumerate(num_shots):
    bars = plt.bar(
        np.arange(len(num_wires)) + idx * bar_width + bar_space,
        shots_times[idx],
        width=bar_width,
        label=labels[idx],
        color=colors[idx],
    )
    plt.bar_label(bars, padding=1, fmt="%.2f", fontsize=9)

# Add labels and titles
plt.xlabel("#qubits")
plt.ylabel("Time (s)")
plt.gca().set_axisbelow(True)
plt.grid(axis="y", alpha=0.5)
plt.xticks(np.arange(len(num_wires)) + bar_width / 2, num_wires)
plt.title("Execution Times with varying shots")
plt.legend(fontsize=9)
plt.show()


######################################################################
# From this result, we can clearly see that both huge analytic and sampling simulations
# can be performed using ``default.clifford``. Estimatede time remains pretty much same,
# especially when number of qubits scales up. Therefore, this device is clearly much
# more performant than statevector-based device like ``default.qubit`` for simulating
# stabilizer circuits.
#

######################################################################
# Conclusion
# ----------
#

######################################################################
# We conlcude that stabilizers circuits are important class of quantum circuits that find usecase in
# quantum error correction and benchmarking performance of quantum hardware. Therefore, their
# efficient simulations allows for an effective way of verficiation of hardware. The
# ``default.clifford`` device in PennyLane enables such simulations of large-scale Clifford circuits
# that would enable all such usecases.
#

##############################################################################
# References
# ----------
#
# .. [#supremecy_exp1]
#
#     C. Huang, F. Zhang, M. Newman, J. Cai, X. Gao, Z. Tian, and *et al.*
#     "Classical Simulation of Quantum Supremacy Circuits"
#     `arXiv:2005.06787 <https://arxiv.org/abs/2005.06787>`__, 2020.
#
# .. [#supremecy_exp2]
#
#     D. Maslov, S. Bravyi, F. Tripier, A. Maksymov, and J. Latone
#     "Fast classical simulation of Harvard/QuEra IQP circuits"
#     `arXiv:2402.03211 <https://arxiv.org/abs/2402.03211>`__, 2024.
#
# .. [#aaronson-gottesman2004]
#    S. Aaronson and D. Gottesman
#    "Improved simulation of stabilizer circuits"
#    `Phys. Rev. A 70, 052328 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.052328>`__, 2004.
#
# .. [#stim]
#    C. Gidney
#    "Stim: a fast stabilizer circuit simulator"
#    `Quantum 5, 497 <https://doi.org/10.22331/q-2021-07-06-497>`__, 2021.
#

######################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/utkarsh_azad.txt
