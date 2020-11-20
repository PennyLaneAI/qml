"""
Quantum supremacy using qsim
============================

In the paper `Quantum supremacy using a programmable superconducting
processor <https://www.nature.com/articles/s41586-019-1666-5>`__ Google
showed that their quantum computer could complete a task that would take a
classical computer potentially thousands of years. For their simulation
benchmarks they used a simulator called `qsim
<https://github.com/quantumlib/qsim>`__, which we will also use here,
provided via our `PennyLane-Cirq plugin
<https://pennylane-cirq.readthedocs.io/en/latest/>`__.

To construct a task that was able to both be run on the Sycamore chip and
simulated classically, a pseudo-random circuit was constructed by
alternating single-qubit and two-qubit gates in a specific, semi-random
pattern. This is basically a way of building a random unitary that is also
compatible with their hardware. The circuit output could then be sampled a
number of times, producing a set of bitstrings corresponing to the
measurements of the circuit. The more qubits there are, and the deeper the
circuit is, the more difficult it becomes to simulate and sample this
probability distribution, from which these bitstrings come from,
classically. By comparing run-times for the classical simulations along
with the outputs of the physical quantum chip for smaller circuits, and
then extrapolating classical run-times for a larger circuits, this is
explained as a suffient measure of quantum supremacy. [1]

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
# To start we need to define the qubit grid that we will use for mimicking
# Google's Sycamore chip, although we will only use 12 qubits instead of
# the 54 that the actual chip has. The reason for this is simply because of
# performance constraints, since we want you to be able to run this demo
# without having access to a super-computer.
#
# We define the 12 qubits in a rectangular grid setting the coordinates for
# each qubit following the paper's suplementary dataset [3]. We also create
# a mapping between the wire number and the Cirq qubit to easier reference
# specific qubits later. Feel free to play around with different grids and
# number of qubits. Just keep in mind that the grid needs to stay
# connected. You could, for example, remove the final row (last four qubits
# in the list) to simulate an 8 qubit system.
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
# Now let's create the qsim-device, available via the Cirq plugin, together
# with the ``wires`` and ``qubits`` keywords that we defined above. qsim is
# a Schrödinger full state-vector simulator that was used for the cross
# entropy benchmarking in Google's supremacy experiment [1].
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
# simplify the circuit definition, we define the remaining gates before the
# circuit is created.
#
# For the single qubit gates we need the :math:`\sqrt{X}` gate, which can
# be written as :math:`RX(\pi/2)`, the :math:`\sqrt{Y}` gate which can be
# written as :math:`RY(\pi/2)`, as well as the :math:`\sqrt{W}`, where
# :math:`W = \frac{X + Y}{2}`, which is easiest to define by it's unitary
# matrix
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
# The logic below iterates through all connections and returns a dictionary
# ``d`` with list of tuples containing two neighbouring qubits with the key
# as their connection label. We will use this dictionary inside the circuit
# to iterate through the different qubit pairs and apply the two two-qubit
# gates that we just defined above. The way we iterate through the
# dictionary will depend on a gate sequence defined in the next section.
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
# At this point we can define the gate sequence, which is the order the
# two-qubit gates are applied to the different qubit-pairs . For example,
# ``["A", "B"]`` would mean that the two-qubit gates are first applied to
# all qubits connected with label A, and then, during the next full cycle,
# the two-qubit gates are applied to all qubits connected with label B.
# This would then correspond to a 2-cycle run (or a circuit with a depth of
# 2).
#
# While we can define any patterns we'd like, the two gate sequences below
# are the ones that are used in the supremacy paper. The shorter one is
# used for their classically verifiable benchmarking, while the slightly
# longer sequence, which is much harder to simulate classically, is used
# for estimating the cross-entropy fidelity in what they call the supremacy
# regime. We will use the shorter gate sequence for the following
# demonstration, although feel free to play around with other combinations
# if you wish.
#

m = 14  # number of cycles

# gate_sequence = np.resize(["A", "B", "C", "D", "C", "D", "A", "B"], m)
gate_sequence = np.resize(["A", "B", "C", "D"], m)


######################################################################
# Finally, we can define the circuit itself and create a QNode that we will
# use for circuit evaluation with the qsim device.
#
# Each circuit-loop consists of alternating layers of single-qubit gates
# and two-qubit gates, referred to as a full cycle. The single-qubit gates
# are randomly selected and applied to each qubit in the circuit, while the
# two-qubit gates are applied to the qubits connected by A, B, C or D as
# defined above. The circuit finally ends with a half-cycle, consisting of
# only a layer of single-qubit gates. Note that the last half-cycle is only
# applied once after the gate sequence is completed.
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
#    F_{XEB} = 2^{n}\left<P(x_i)\right>_i - 1
#
# where :math:``n`` is the number of qubits, :math:`P(x_i)` is the
# probability of bitstring :math:`x_i` computed for the ideal quantum
# circuit, and the average is over the observed bitstrings.
#
# The idea behind using this fidelity is that it will be close to 1 for
# samples obtained from random quantum circuits, such as the one we defined
# above, and close to zero for all other probability distributions that can
# be effectively sampled from classically (such as a normal distribution).
# Let us briefly calculate the expected theoretical value for the fidelity
# based on the number of qubits in the circuit. According to the
# supplementary information for both the supremacy paper [2] and a related
# paper [4], sampling a bitstring from a random quantum circuit would
# follow the Porter-Thomas distribution, given by
#
# .. math::
#
#    Pr(p) = (N - 1)(1- p)^{N-2}
#
# where :math:`N = 2^\text{wires}` is the number of possible bitstrings.
# From this we can then calculate the expectation value
# :math:`\left<P(x_i)\right>` as follows:
#
# .. math::
#
#    \left<P(x_i)\right> = \int_0^1 p^2 N (N-1)(1-p)^{N-2}dp = \frac{2}{N+1}
#
# which leads to the theoretical fidelity
#
# .. math::
#
#    F_{XEB} = 2^{n}\left<P(x_i)\right>_i - 1 = \frac{2N}{N+1} - 1
#
# We implement this fidelity as the funtion below, where ``samples`` is a
# list of samples with each sample being a list of ``0`` and ``1``, and
# ``probs`` is a list with correspondning sampling probabilities for the
# same noise-less circuit.
#

def fidelity_xeb(samples, probs):
    sampled_probs = []
    for sam in samples:
        #  create a single bitstring string and convert it to an integer
        bitstring = "".join(sam.astype(str))
        bitstring_idx = int(bitstring, 2)

        # retrieve the corresponding probability for the bitstring
        sampled_probs.append(probs[bitstring_idx])

    return 2**samples.shape[1] * np.mean(sampled_probs) - 1


######################################################################
# We set a random seed and use it to calculated the probability for all the
# possible 12 qubit states. We can then sample from the device by calling
# ``dev.generate_samples`` and similarly, sample random bitstrings from a
# normal distribution by generating all basis states, along with their
# corresponding bitstrings, and sample directly from them using NumPy.
#

seed=np.random.randint(0, 42424242)
probs = circuit(seed=seed)

# sampling from the circuit's probability distribution should give a fidelity close to 1
samples = dev.generate_samples()
f_circuit = fidelity_xeb(samples, probs)
print("Circuit's distribution:", f"{f_circuit:.7f}".rjust(12))

# sampling from a normal distribution should give a fidelity close to 0
basis_states = dev.generate_basis_states(wires)
samples = np.array([basis_states[i] for i in np.random.randint(0, len(basis_states), size=shots)])
f_normal = fidelity_xeb(samples, probs)
print("Normal distribution:", f"{f_normal:.7f}".rjust(15))


######################################################################
# We can also calculate the theoretical result obtained from the equation
# above and compare it to the value obtained from sampling from the
# circuit. This should be close to the fidelity calculated from the circuit
# samples.
#

N = 2 ** wires
theoretical_value = 2 * N / (N + 1) - 1

print("Theoretical:", f"{theoretical_value:.7f}\n".rjust(24))


######################################################################
# The values above might seem a bit arbitrary. To show that the fidelity
# from the circuit sampling actually tends towards the theoretical value we
# can run several different random circuits, calculate their respective
# cross-entropy benchmark fidelities and then calculate the mean fidelity.
# This value should get closer to the theoretical value the more
# evaluations we do.
#
# .. note::
#
#    The following mean fidelity calculations can be interesting to play
#    around with. You can change the qubit grid at the top of this demo
#    using, e.g., 8 or 4 qubits; change the number of shots used; as well
#    as the number of circuit evaluations below. Running the following code
#    snippet, the mean fidelity should still tend towards the theoretical
#    value (which will be lower for fewer qubits).
#

print("Theoretical:", f"{2**wires*(2/(2**wires+1)) - 1:.7f}".rjust(24))

f_circuit = []
num_of_evaluations = 10
for i in range(num_of_evaluations):
    seed=np.random.randint(0, 42424242)

    probs = circuit(seed=seed)
    samples = dev.generate_samples()

    f_circuit.append(fidelity_xeb(samples, probs))
    print(f"\r{i + 1:4d} / {num_of_evaluations:4d}{' ':17}{np.mean(f_circuit):.7f}", end="")
print("\rObserved:", f"{np.mean(f_circuit):.7f}".rjust(27))


######################################################################
# Quantum supremacy
# -----------------
#
# Why are we calculating this specific fidelity, and what does it actually
# mean if we get a cross-entropy benchmarking fidelity close to 1? This is
# an important question, containing one of the main arguments behind why
# this experiment is able to show quantum supremacy.
#
# The idea behind this fidelity is that it should be impossible to
# effectively get a value close to 1 by any classical sampling methods.
# This is due to the Porter-Thompson probability distribution that the
# random quantum circuits follow, which are hard to simulate classically.
# On the other hand, a quantum device, running a circuit as the one
# constructed above, should be able to sample from such a distribution
# without much overhead. Thus, by showing that a quantum device can produce
# a high enough fidelity value for a large enough circuit, quantum
# supremacy can be claimed. This is exactly what the paper discussed in
# this demonstration has done.
#
# There's still one issue that hasn't been touched on yet, and will be left
# for a future demonstration: the addition of noise in quantum hardware.
# Simply put, this noise will lower the cross-entropy benchmark fidelity
# getting it closer to 0, as would also be the case for any classically
# tractable sampling from probability distributions. The larger the
# circuit, the more noise there will also be, and thus the lower the
# fidelity will be. By calculating the specific single-qubit, two-qubit and
# readout errors and simulating a noisy circuit, comparing it to the
# hardware device output, and then extrapolating the time it would take to
# run a larger circuit, the paper is able to predict the 10,000 years it
# would take to classically reach the same fidelity for the particular
# circuit as they do with the Sycamore chip (see Fig. 4 in `[1]
# <https://www.nature.com/articles/s41586-019-1666-5>`__).
#
# For more reading on this, the original Nature paper [1] is highly
# recommended, along with the suplementary information if you want to dive
# deeper into the maths and physics of the experiment. The blogpost in [5],
# along with the accompanying GitHub repo, also provides a nice introduction
# to the cross-entropy benchmark fidelity, along with some calculations
# including the effects of added noise models.
#

######################################################################
# References
# ----------
#
# [1]: `Quantum supremacy using a programmable superconducting processor
# <https://www.nature.com/articles/s41586-019-1666-5>`__
#
# [2]: `Supplementary information for "Quantum supremacy using a
# programmable superconducting processor" <https://arxiv.org/abs/1910.11333>`__
#
# [3]: `Dataset for "Quantum supremacy using a programmable superconducting
# processor" <https://datadryad.org/stash/dataset/doi:10.5061/dryad.k6t1rj8>`__
#
# [4]: `Supplementary material for "Characterizing Quantum Supremacy in
# Near-Term Devices" <https://www.nature.com/articles/s41567-018-0124-x#Sec7>`__
#
# [5]: `Unpacking the Quantum Supremacy Benchmark with Python
# <https://medium.com/@sohaib.alam/unpacking-the-quantum-supremacy-benchmark-with-python-67a46709d>`__
#