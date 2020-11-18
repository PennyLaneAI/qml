"""
Quantum supremacy using qsim
============================

In the paper `Quantum supremacy using a programmable superconducting processor
<https://www.nature.com/articles/s41586-019-1666-5>`__ Google showed that
their quantum computer could complete a task that would take a classical
computer thousands of years. For their simulation benchmarks they used a
simulator called qsim, which we will also use here provided via our
PennyLane-Cirq plugin.

To construct a task that could be run on the Sycamore chip and compared
to similar simulations, a pseudo-random circuit was constructed by
alternating single-qubit and two-qubit gates in a specific, semi-random
pattern. The circuit output was then sampled a number of times,
producing a set of bitstrings corresponing to the measured state of the
circuit. The probability distribution from which these bitstrings were
sampled becomes more and more difficult to simulate classically the more
qubits there are and the deeper the circuit is, and can thus provide a
comparable measure against the outputs of the physical quantum chip.

In this demonstarion, we will walk you through how their benchmarks were
constructed and run, and provide an example of what their simulations
looked like. We will be using PennyLane along with the PennyLane-Cirq
plugin and the qsim-device, running via the Cirq backend.

"""


######################################################################
# Implementing the circuit
# ------------------------
#
# As always, we begin by importing the necessary modules. We will use
# PennyLane, along with some PennyLane-Cirq specific operation, as well as
# Cirq.
#

import pennylane as qml
from pennylane_cirq import ops

import cirq
import numpy as np


######################################################################
# To start we need to define the qubit grid that we will use for mimics
# Google's Sycamore chip, although we will only use 12 qubits instead of
# the 54 that the actual chip has. The reason for this is simply because
# of performance constraints.
#
# We define the 12 qubits in a rectangular grid, and label the
# corresponding wires in a way that makes it easier to reference the
# specific qubits later. Feel free to play around with different grids and
# number of qubits. Just keep in mind that the grid needs to stay
# connected. You could, for example, remove the final row (last four
# qubits in the list) to simulate an 8 qubit system.
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
# We also need to define the number of shots per circuit instance to be
# used. This corresponds to the number of times that the circuit is
# sampled. This will be used later when calculating the cross-entropy
# benchmark fidelity. The more shots, the more accurate the results will
# be. We will use 500.000 shots; the same number of samples that are used
# in the supremacy paper, but feel free to change this to whichever value
# you wish (depending on your own hardware restrictions).
#

shots = 500000
dev = qml.device('cirq.qsim', wires=wires, qubits=qubits, shots=shots)


######################################################################
# The next step would be to prepare the gates that will be used in the
# circuit. Several gates that are not natively supported in PennyLane are
# needed. Some of them are made available through the Cirq plugin, since
# they already are implemented in Cirq, and thus are supported by qsim. To
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
#    \begin{bmatrix}
#       1 & \sqrt{i}  \\
#       \sqrt{-i} & 1 \\
#    \end{bmatrix}
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
#    \begin{bmatrix}
#       1 & 0 & 0 & 0 \\
#       0 & 0 & i & 0 \\
#       0 & i & 0 & 0 \\
#       0 & 0 & 0 & 1
#    \end{bmatrix}
#
# as well as the CPhase gate
#
# .. math::
#
#    \begin{bmatrix}
#       1 & 0 & 0 & 0 \\
#       0 & 1 & 0 & 0 \\
#       0 & 0 & 1 & 0 \\
#       0 & 0 & 0 & e^{-i\phi}
#    \end{bmatrix}
#
# These two gates have already been made accesible via the Cirq plugin.
#


######################################################################
# Here comes one of the tricky parts. The way the paper decides which
# qubits the two-qubit gates should be applied to depends on how they are
# connected to each other. In an alternating pattern, each pair of
# neighbouring qubits gets labeled with a letter A-D, where A and B
# correspond to all horizontally neighbouring qubits, and C and D to the
# vertically neighbouring qubits.
#
# The logic below iterates through all connections and returns a
# dictionary ``d`` with list of tuples containing two neighbouring qubits
# with the key as their connection label. We will use this dictionary
# inside the circuit to iterate through the different qubit pairs and
# apply the two two-qubit gates that we just defined above. The way we
# iterate through the dictionary will depend on a gate sequence defined in
# the next section.
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
# At this point we can define the gate sequence, which is the order in
# which qubit-pairs the two-qubit gates are applied to. For example,
# ``["A", "B"]`` would mean that the two-qubit gates are first applied to
# all qubits connected with label A, and then, during the next full cycle,
# the two-qubit gates are applied to all qubits connected with label B.
# This would then correspond to a 2-cycle run (or a circuit with a depth
# of 2).
#
# While we can define any patterns we'd like, the two gate sequences below
# are the ones that are used in the supremacy paper. The shorter one is
# used for their classically verifiable benchmarking, while the slightly
# longer sequence is much harder to simulate classically and is used for
# estimating the cross-entropy fidelity in what they call the supremacy
# regime. We will use the shorter gate sequence for the following
# demonstration, although feel free to play around with other combinations
# if you wish.
#

m = 14  # number of cycles

# gate_sequence = np.resize(["A", "B", "C", "D", "C", "D", "A", "B"], m)
gate_sequence = np.resize(["A", "B", "C", "D"], m)


######################################################################
# Finally, we can define the circuit itself and create a QNode that we
# will use for circuit evaluation with the qsim device.
#
# Each circuit-loop consists of alternating layers of single-qubit gates
# and two-qubit gates, referred to as a full cycle. The single-qubit gates
# are randomly selected and applied to each qubit in the circuit, while
# the two-qubit gates are applied to the qubits connected by A, B, C or D
# as defined above. The circuit finally ends with a half-cycle, consisting
# of only a layer of single-qubit gates. Note that the last half-cycle is
# only applied once after the gate sequence is completed.
#
# We define the circuit, letting it return the state probabilities, and
# decorate it with the QNode decorator, binding it to the qsim simulator
# device. Later, we will also extract samples directly from the device
# without needing to add it as a return statement in the circuit.
#

@qml.qnode(dev)
def circuit(seed=42):
    np.random.seed(seed)

    # m full cycle - single-qubit gates & two-qubit gate
    for gs in gate_sequence:
        for w in range(wires):  # TODO: avoid same gate twice on a qubit
            np.random.choice(single_qubit_gates)(wires=w)

        for qb_1, qb_2 in d[gs]:
            ops.ISWAP(wires=(qb_1, qb_2))
            ops.CPhase(-np.pi/6, wires=(qb_1, qb_2))

    # one half-cycle - single-qubit gates only
    for w in range(wires):
        np.random.choice(single_qubit_gates)(wires=w)

    return qml.probs(wires=range(wires))


######################################################################
# The cross-entropy benchmark fidelity
# ------------------------------------
#
# The benchmark that is used in the paper, and the one that we will use in
# this demo, is called the linear cross-entropy benchmarking fidelity of
# the circuit. It's defined as
#
# .. math::
#
#    2^{n}\left<P(x_i)\right>_i - 1
#
# where :math:`n` is the number of qubits, :math:`P(x)i)` is the
# probability of bitstring :math:`x_i` computed for the ideal quantum
# circuit, and the average is over the observed bitstrings.
#

def fidelity_xeb(samples, probs):
    sampled_probs = []
    for sam in samples:
        bitstring = "".join(sam.astype(str))
        bitstring_idx = int(bitstring, 2)

        sampled_probs.append(probs[bitstring_idx])

    return 2**samples.shape[1] * np.mean(sampled_probs) - 1


######################################################################
# For an ideal circuit, i.e. one without any noise, the output of this
# function should be close to 1, while if any errors have occurred in the
# circuit, the value will be closer to 0. The former would correspond to
# sampling from a an exponential probability distribution, while the
# latter would correspond to sampling from a normal distribution.
#
# We set a random seed and use it to calculated the probability for all
# the possible 12 qubit states. We can then sample from the device by
# calling ``dev.generate_samples`` and similarly, sample random bitstrings
# from a normal distribution by generating all basis states, and their
# corresponding bitstrings, and sampling directly from them using NumPy.
#

seed=np.random.randint(0, 42424242)
# seed=42

probs = circuit(seed=seed)

# the theoretical value should be 2^wires * 2 / (2^wires + 1) - 1
print(f"Theoretical:              {2**wires*(2/(2**wires+1)) - 1:.7f}")

# while sampling from the circuit's probability distribution would
# correspond to an exponential one and should give a fidelity close to 1
samples = dev.generate_samples()
f_circuit = fidelity_xeb(samples, probs)
print(f"Exponential distribution: {f_circuit:.7f}")

# sampling from a normal distribution should give a fidelity close to 0
basis_states = dev.generate_basis_states(wires)
samples = np.array([basis_states[i] for i in np.random.randint(0, len(basis_states), size=shots)])
f_normal = fidelity_xeb(samples, probs)
print(f"Normal distribution:      {f_normal:.7f}")

f_circuit = []
print(f"Theoretical:              {2**wires*(2/(2**wires+1)) - 1:.7f}")

num_of_evaluations = 10
for i in range(num_of_evaluations):
    seed=np.random.randint(0, 42424242)

    probs = circuit(seed=seed)
    samples = dev.generate_samples()

    f_circuit.append(fidelity_xeb(samples, probs))
    print(f"\r{i+1:4d} / {num_of_evaluations}               {np.mean(f_circuit):.7f}", end="")
print(f"\rObserved:                 {np.mean(f_circuit):.7f}")
