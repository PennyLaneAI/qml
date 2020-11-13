"""
Using qsim for supremacy
========================

In the paper `Quantum supremacy using a programmable superconducting processor
<https://www.nature.com/articles/s41586-019-1666-5>`_ Google showed that their
quantum computer could complete a task that would take a classical computer
thousands of years. For their simulation benchmarks they used a simulator called
qsim.

In this demonstarion, we will walk you through how their benchmarks and provide
an example of what their simulations looked like. We will be using PennyLane
along with the PennyLane-Cirq plugin and the qsim-device, running via the Cirq
backend.

"""

import pennylane as qml
from pennylane_cirq import ops

import cirq
import numpy as np


######################################################################
# To start we need to define the qubit grid that we will use for mimics Google's
# Sycamore chip, although we will only use 12 qubits instead of the 54 that the
# actual chip has. The reason for this is simply because of performance
# constraints.
#
# We define the 12 qubits in a rectangular grid, and label the corresponding
# wires in a way that makes it easier to reference the specific qubits later.
#

qubits = sorted([
    cirq.GridQubit(3, 3),
    cirq.GridQubit(3, 4),
    cirq.GridQubit(3, 5),
    cirq.GridQubit(3, 6),
    cirq.GridQubit(4, 3),
    cirq.GridQubit(4, 4),
    cirq.GridQubit(4, 5),
    cirq.GridQubit(4, 6),
    cirq.GridQubit(5, 3),
    cirq.GridQubit(5, 4),
    cirq.GridQubit(5, 5),
    cirq.GridQubit(5, 6),
])

wires = len(qubits)

# create a mapping between wire number and Cirq qubit
qb2wire = {i: j for i, j in zip(qubits, range(wires))}


######################################################################
# Now let's create the qsim-device, available via the Cirq plugin,
# together with the ``wires`` and ``qubits`` keywords that we defined
# above. qsim is a Schrödinger full state-vector simulator that was used
# for the cross entropy benchmarking in
# `Google's supremacy experiment <https://www.nature.com/articles/s41586-019-1666-5>`__
#

dev = qml.device('cirq.qsim', wires=wires, qubits=qubits, shots=1)


######################################################################
# Several gates that are not natively supported in PennyLane are needed.
# Some of them are made available through the Cirq plugin, since they
# already are implemented in Cirq, and thus are supported by qsim. To
# simplify the circuit definition, we define the remaining gates before
# the circuit is created.
#
# For the single qubit gates we need the :math:`\sqrt{X}` gate, which
# can be written as :math:`RX(\pi/2)`, the :math:`\sqrt{Y}` gate which
# can be written as :math:`RY(\pi/2)`, as well as the
# :math:`\sqrt{W}`, where :math:`W = \frac{X + Y}{2}`, which is
# easiest to define by it's unitary matrix
#
# .. math::
#
#    \frac{1}{\sqrt{2}}
#    \begin{matrix}
#       1 & \sqrt{i}  \\
#       \sqrt{-i} & 1 \\
#    \end{matrix}
#

sqrtXgate = lambda wires: qml.RX(np.pi / 2, wires=wires)

sqrtYgate = lambda wires: qml.RY(np.pi / 2, wires=wires)

sqrtWgate = lambda wires: qml.QubitUnitary(
    np.array([[1,  -np.sqrt(1j)],
              [np.sqrt(-1j), 1]]) / np.sqrt(2), wires=wires
)

single_qubit_gates = [sqrtXgate, sqrtYgate, sqrtWgate]


######################################################################
# For the two-qubit gates we need the iSWAP gate
#
# .. math::
#
#    \begin{matrix}
#       1 & 0 & 0 & 0 \\
#       0 & 0 & i & 0 \\
#       0 & i & 0 & 0 \\
#       0 & 0 & 0 & 1
#    \end{matrix}
#
# as well as the CPhase gate
#
# .. math::
#
#    \begin{matrix}
#       1 & 0 & 0 & 0 \\
#       0 & 1 & 0 & 0 \\
#       0 & 0 & 1 & 0 \\
#       0 & 0 & 0 & e^{-i\phi}
#    \end{matrix}
#
# These two gates have been made accesible via the Cirq plugin.
#


######################################################################
# Here comes a bit of a tricky part. The way the paper decides which
# qubits the two-qubit gates should be applied to depends on how they are
# connected to each other. In an alternating pattern, each pair of
# neighbouring qubits gets labeled with a letter A-D, where A and B
# correspond to all horizontally neighbouring qubits, and C and D to the
# vertically neighbouring qubits.
#
# The logic below iterates through all connections and returns a
# dictionary ``d`` with list of tuples containing two neighbouring qubits
# with the key as their connection label.
#

from itertools import combinations

d = {"A":[], "B":[], "C":[], "D":[]}
for i, j in combinations(qubits, 2):
    wire_1 = qb2wire[i]
    wire_2 = qb2wire[j]
    if i in j.neighbors():
        if i.row == j.row and i.col % 2 == 0:
            d["A"].append((wire_1, wire_2))
        elif i.row == j.row and j.col % 2 == 0:
            d["B"].append((wire_1, wire_2))
        elif i.col == j.col and i.row % 2 == 0:
            d["C"].append((wire_1, wire_2))
        elif i.col == j.col and j.row % 2 == 0:
            d["D"].append((wire_1, wire_2))


######################################################################
# At this point we can define a gate sequence, which is the order in which
# qubit-pairs the two-qubit gates are applied to. E.g. ``["A", "B"]``
# would mean that the two-qubit gates are first applied to all qubits
# connected with label A, and then, during the next full cycle, the
# two-qubit gates are applied to all qubits connected with label B. This
# would then correspond to a 2-cycle run.
#
# While we can define any patterns we'd like, the two gate sequences below
# are the ones that are used in the supremacy paper. The shorter one is
# used for their classically verifiable benchmarking, while the slightly
# longer sequence is much harder to simulate classically and is used for
# estimating the cross-entroby fidelity in what they call the supremacy
# regime. We will use the shorter gate sequence for the following
# demonstration, although feel free to play around with other combinations
# if you wish.
#

m = 14  # number of cycles

# gate_sequence = np.resize(["A", "B", "C", "D", "C", "D", "A", "B"], m)
gate_sequence = np.resize(["A", "B", "C", "D"], m)


######################################################################
# Each circuit-loop consists of alternating layers of single-qubit gates
# and two-qubit gates, referred to as a full cycle.
#
# The single-qubit gates are randomly selected and applied to each qubit
# in the circuit, while the two-qubit gates are applied to the qubits
# connected by A, B, C or D as defined above. Before and after each two
# qubit gate sequence (consisting of an iSWAP and a CPhase gate) RZ gates
# are applied to the participating qubits. The circuit finally ends with a
# half-cycle, consisting of only a layer of single-qubit gates.
#
# We define the circuit, naming it ``lossless_circuit`` since we'll
# attempt adding noise to it later, and decorate it with the QNode
# decorator, binding it to the qsim simulator device.
#

@qml.qnode(dev)
def lossless_circuit(seed=42):
    np.random.seed(seed)
    # m full cycle - single-qubit gates & two-qubit gate
    for gs in gate_sequence:
        for w in range(wires):  # TODO: avoid same gate twice on a qubit
            np.random.choice(single_qubit_gates)(wires=w)

        for qb_1, qb_2 in d[gs]:
            r = 40*np.pi * (2*np.random.random(4) - np.random.random(4))

            qml.RZ(r[0], wires=qb_1)
            qml.RZ(r[1], wires=qb_2)

            ops.ISWAP(wires=(qb_1, qb_2))
            ops.CPhase(-np.pi/6, wires=(qb_1, qb_2))

            qml.RZ(r[2], wires=qb_1)
            qml.RZ(r[3], wires=qb_2)

    # one half-cycle - single-qubit gates only
    for w in range(wires):
        np.random.choice(single_qubit_gates)(wires=w)

    return qml.probs(wires=range(wires))


######################################################################
# The benchmark that is used in the paper, and the one that we will use in
# this demo, is called the linear cross-entropy benchmarking fidelity of
# the circuit. It's defined as
#
# .. math::
#
#    2^{n}\left<P(x_i)\right>_i - 1
#
# where :math:``n`` is the number of qubits, :math:``P(x)i)`` is the
# probability of bitstring :math:``x_i`` computed for the ideal quantum
# circuit, and the average is over the observed bitstrings.
#

def fidelity(number_of_samples, probs, p=None):
    sampled_probs = np.random.choice(probs, p=p, size=number_of_samples, replace=True)

    return 2**wires * np.mean(sampled_probs) - 1


######################################################################
# For an ideal circuit, i.e. one without any noise, the output of this
# function should be close to 1, while if any errors have occurred in the
# circuit, the value will be closer to 0. The former would correspond to a
# an exponential probability distribution, while the former would
# correspond to a normal distribution.
#

probs = lossless_circuit()

# not setting the sampling probabilities `p` would correspond to sampling from a normal distribution
f_normal = fidelity(1000, probs)
print(f"Normal distribution:     {f_normal:2f}")

# while sampling from the circuits probability distribution would correspond to an exponential one
f_circuit = fidelity(1000, probs, p=probs)
print(f"Exponential distribution: {f_circuit:2f}")


######################################################################
# We want to use a lossy circuit for this demo, to be able to compare the
# ideal probabilities with the ones where errors have occured with certain
# probabilities. To be able to do this, we must separate the random seeds
# for the randomized gate application and the error probability. Luckily,
# we can define separate random states for these two cases, and set a
# random seed for gates. This is crucial, since we want to apply the same
# gates each time we run the circuit, while still being able to randomize
# errors.
#

gate_random_state = np.random.RandomState()
error_random_state = np.random.RandomState()

probability_samples = 100000000
gate_seed = np.random.randint(0, 42424242)


######################################################################
# Next, we define the lossy circuit, and simply name it ``circuit``. We
# also define an ``error`` template that simply applies either a bitflip
# (``PauliX`` gate) or a phaseflip (``PauliZ`` gate) on each wire with a
# certain probability.
#
# We then apply the error template after each random single qubit gate as
# well as after each two qubit gate, as well as at the end of the circuit,
# corresponding to measurement errors. Here, ``s_prob``, ``t_prob`` and
# ``m_prob`` correspond to single-qubit gate errors, two-qubit gate errors
# and measurement/readout errors respectively.
#

@qml.template
def error(error_probability, wires):
    for w in wires:
        if error_random_state.random() < error_probability:
            np.random.choice([
                qml.PauliX(wires=w),
                qml.PauliZ(wires=w),
            ])

@qml.qnode(dev)
def circuit(r, phi, gate_seed=42, error_seed=42, s_prob=0, t_prob=0, m_prob=0):
    # get the same gates each time (but change errors)
    gate_random_state.seed(gate_seed)
    error_random_state.seed(error_seed)

    # m full cycle - single-qubit gates & two-qubit gate
    for gs in gate_sequence:
        for w in range(wires):  # TODO: avoid same gate twice on a qubit
            gate_random_state.choice(single_qubit_gates)(wires=w)
        error(s_prob, wires=range(wires))  # apply random single-qubit error

        for qb_1, qb_2 in d[gs]:
            qml.RZ(r[0], wires=qb_1)
            qml.RZ(r[1], wires=qb_2)

            ops.ISWAP(wires=(qb_1, qb_2))
            error(t_prob, wires=(qb_1, qb_2))  # apply random two-qubit error

            ops.CPhase(phi, wires=(qb_1, qb_2))
            error(t_prob, wires=(qb_1, qb_2))  # apply random two-qubit error

            qml.RZ(r[2], wires=qb_1)
            qml.RZ(r[3], wires=qb_2)

    # one half-cycle - single-qubit gates only
    for w in range(wires):
        gate_random_state.choice(single_qubit_gates)(wires=w)
    error(s_prob, wires=range(wires))  # apply random single-qubit error

    error(m_prob, wires=range(wires))  # apply random measurement error

    return qml.probs(wires=range(wires))


######################################################################
# *a bit unsure about the next step; work in progress*
#
# To find the parameters for the ``CPhase`` gate and the ``RZ`` gates that
# we will use in the circuit, we need to figure out which values provide a
# cross-entropy fidelity close to 1 for the pure circuit. Since this is a
# bit stochastic, we optimize the circuit using the NLopt library to find
# the optimal parameters to be used in the circuit.
#

import nlopt

wires = 4
steps = 1000

print(f"seed: {gate_seed}\n")

opt_algorithm = nlopt.LN_BOBYQA  # gradient-free optimizer

opt = nlopt.opt(opt_algorithm, wires+1)

min_cost = 100
num_evals = 0
def cost(params, grad=[]):
    global min_cost
    global num_evals
    global best_params

    phi = params[0]
    r = params[1:]

    probs_ideal = circuit(r, phi, gate_seed=gate_seed)

    fide = fidelity(probability_samples, probs=probs_ideal, p=probs_ideal)

    cost_val = np.abs(1 - fide)

    if cost_val < min_cost:
        min_cost = cost_val
        best_params = params.copy()

    num_evals += 1
    print(f"\revals: {num_evals:3d}    cost: {min_cost:.7f}", end="")

    return float(cost_val)

opt.set_min_objective(cost)

opt.set_lower_bounds(np.append(-np.pi, -40*np.pi * np.ones(wires)))
opt.set_upper_bounds(np.append(np.pi, 40*np.pi * np.ones(wires)))

opt.set_maxeval(steps)

params = np.pi * (2*np.random.random(wires+1) - np.random.random(wires+1))

try:
    best_params = opt.optimize(params)
except:
    pass
print(f"\noptimal parameters: {best_params}")


######################################################################
# Let's print the parameters along with the optimized cross-entropy
# fidelity.
#

phi = best_params[0]
r = best_params[1:]
print(f"r = {r}, phi = {phi}")

probs_ideal = circuit(r, phi, gate_seed=gate_seed)

print(f"ideal fidelity: {fidelity(probability_samples, probs=probs_ideal, p=probs_ideal)}")


######################################################################
# As we can see, the fidelity is now fairly close to 1, which is what we
# want. We now save these parameters, along with the gate seed defined
# earlier, so that we can use this exact circuit for the simulations
# below.
#


######################################################################
# Finally, we can calculate the fidelity for the circuits using different
# probabilities for errors to occur. The values are taken directly from
# the supremacy paper, and inserted into the circuits to calculate the
# ideal and the observed (i.e. with errors) probabilities. This is done
# several times, and the mean fidelity is then printed.
#

# Average error            Isolated        Simultaneous
#
# Single-qubit (e1)        0.15%           0.16%
# Two-qubit (e2)           0.36%           0.62%
# Two-qubit, cycle (e2c)   0.65%           0.93%
# Readout (er)             3.10%           3.80%

e1 = 0.0016
e2 = 0.0062
e2c = 0.0093
er = 0.038

fidelity_samples = 100

fidelities = []
for i in range(fidelity_samples):
    seed = np.random.randint(0, 42424242)

    probs_ideal = circuit(r, phi, gate_seed=gate_seed, error_seed=seed)
    probs_observed = circuit(r, phi, gate_seed=gate_seed, error_seed=seed, s_prob=e1, t_prob=e2, m_prob=er)

    f_xeb = fidelity(probability_samples, probs=probs_observed, p=probs_ideal)
    fidelities.append(f_xeb)

    id_fide = fidelity(probability_samples, probs=probs_ideal, p=probs_ideal)
    print(f"\r{i+1} / {fidelity_samples}    fidelity = {np.mean(fidelities):4f}    ideal fidelity: {id_fide:.7f}", end="")
