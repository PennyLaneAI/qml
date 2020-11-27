"""

.. _quantum_neural_net:

Quantum supremacy using qsim
============================

.. meta::
    :property="og:description": Discover quantum supremacy using Google's qsim simulator, and recreate their benchmarks and circuits.
    :property="og:image": https://pennylane.ai/qml/_images/sycamore.png

.. related::

    tutorial_quantum_metrology Variationally optimizing measurement protocols
    tutorial_noisy_circuit_optimization Optimizing noisy circuits with Cirq

In the paper `Quantum supremacy using a programmable superconducting
processor <https://www.nature.com/articles/s41586-019-1666-5>`__, the
Google AI Quantum team showed that the Sycamore quantum processor could
complete a task that would take a classical computer potentially thousands
of years. They faced their quantum chip off against JEWEL --- one of the
worlds most powerful supercomputers --- using a classical simulator called
`qsim <https://github.com/quantumlib/qsim>`__. The main idea behind this
showdown was to prove that a quantum device could solve a specific task
that no classical methods would be able to in a reasonable amount of time.

To devise a task that could be run on both the Sycamore chip and simulated
classically, a pseudo-random quantum circuit was constructed by alternating
single-qubit and two-qubit gates in a specific, semi-random pattern. This
provided a way of building a random unitary transformation that was also
compatible with the Sycamore hardware. The circuit output could then be
sampled a number of times, producing a set of bitstrings corresponding to
the measurements of the circuit. The more qubits there are, and the deeper
the circuit is, the more difficult it becomes to simulate and sample this
probability distribution classically. By comparing run-times for the
classical simulations along with the outputs of the physical quantum chip
for smaller circuits, and then extrapolating classical run-times for larger
circuits, this was used to announce that quantum supremacy had been
achieved. [#Arute2019]_

In this demonstration, we will walk you through how their circuits and
benchmarks were constructed and run, and provide an example of what their
simulations looked like. We will be using PennyLane along with the
aformentioned ``qsim`` simulator running via our `PennyLane-Cirq plugin
<https://pennylane-cirq.readthedocs.io/en/latest/>`__. To use the
``qsim`` device you also need to install ``qsimcirq``, which is the python
module interfacing the ``qsim`` simulator with Cirq.
"""


######################################################################
# Preparing for supremacy
# -----------------------
#
# As always, we begin by importing the necessary modules. We will use
# PennyLane, along with some PennyLane-Cirq specific operations, as well as
# Cirq.
#

import pennylane as qml
from pennylane_cirq import ops

import cirq
import numpy as np


######################################################################
# To start, we need to define the qubit grid that we will use for mimicking
# Google's Sycamore chip, although we will only use 12 qubits instead of
# the 54 that the actual chip has. The reason for this is simply
# performance constraints, since we want you to be able to run this demo
# without having access to a super-computer.
#
# We define the 12 qubits in a rectangular grid, setting the coordinates for
# each qubit following the paper's suplementary dataset [#Martinis2020]_. We also create
# a mapping between the wire number and the Cirq qubit to more easily reference
# specific qubits later. Feel free to play around with different grids and
# number of qubits. Just keep in mind that the grid needs to stay
# connected. You could, for example, remove the final row (last four qubits
# in the list) to simulate an 8-qubit system.
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
# Now let's create the ``qsim`` device, available via the Cirq plugin, making
# use of the ``wires`` and ``qubits`` keywords that we defined above. ``qsim``
# is a full Schrödinger state-vector simulator that was used for the cross
# entropy benchmarking in Google's supremacy experiment [#Arute2019]_.
#
# First, we need to define the number of shots per circuit instance to
# be used --- where the number of 'shots' simply corresponds to the number
# of times that the circuit is sampled. This will also be needed later when
# calculating the cross-entropy benchmark fidelity. The more shots, the
# more accurate the results will be. 500,000 shots will be used her; the same
# number of samples that are used in the supremacy paper, but feel free to
# change this to whichever value you wish (depending on your own hardware
# restrictions).
#

shots = 500000
dev = qml.device('cirq.qsim', wires=wires, qubits=qubits, shots=shots)


######################################################################
# The next step would be to prepare the gates that will be used. Several
# gates that are not natively supported in PennyLane are needed. Some of
# them are made available through the Cirq plugin, since they are already
# implemented in Cirq, and thus are supported by ``qsim``. To simplify the
# circuit definition, we define the remaining gates before the circuit is
# created.
#
# For the single-qubit gates we need the :math:`\sqrt{X}` gate, which can
# be written as :math:`RX(\pi/2)`, the :math:`\sqrt{Y}` gate which can be
# written as :math:`RY(\pi/2)`, as well as the :math:`\sqrt{W}`, where
# :math:`W = \frac{X + Y}{2}`, which is easiest to define by its unitary
# matrix
#
# .. math::
#
#    \frac{1}{\sqrt{2}}
#    \begin{bmatrix}
#       1 & \sqrt{i}  \\
#       \sqrt{-i} & 1 \\
#    \end{bmatrix}.
#
# The :math:`\sqrt{X}` gate is already implemented in PennyLane, while the
# two other gates can be implemented as follows:

sqrtYgate = lambda wires: qml.RY(np.pi / 2, wires=wires)

sqrtWgate = lambda wires: qml.QubitUnitary(
    np.array([[1,  -np.sqrt(1j)],
              [np.sqrt(-1j), 1]]) / np.sqrt(2), wires=wires
)

single_qubit_gates = [qml.SX, sqrtYgate, sqrtWgate]


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
#    \end{bmatrix},
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
#    \end{bmatrix}.
#
# These two gates are accessible via the Cirq plugin.
#


######################################################################
# Assembling the circuit
# ----------------------
#
# Here comes one of the tricky parts. To decide which qubits that the
# two-qubit gates should be applied to, we have to look at how they are
# connected to each other. In an alternating pattern, each pair of
# neighbouring qubits gets labeled with a letter A-D, where A and B
# correspond to all horizontally neighbouring qubits (on a row), and C and
# D to the vertically neighbouring qubits (on a column). This is depicted
# in the figure below, where you can also see how the single-qubit gates
# are applied as well as the cycles, each consisting of a layer of
# single-qubit gates and a pair of two-qubit gates. Note that each coloured
# two-qubit gate represented in the image is implemented as the two
# consecutive gates iSWAP and CPhase in this demo.
#
# .. figure:: ../demonstrations/qsim_supremacy/supremacy_circuit.png
#     :align: center
#     :width: 90%
#
#     **Image taken from Arute, F., Arya, K., Babbush, R. et al.** [#Arute2019]_
#
# The logic below iterates through all connections and returns a dictionary
# ``gate_order`` with lists of tuples as keys. Each tuple contains two
# neighbouring qubits with the key as their connection label. We will use
# this dictionary inside the circuit to iterate through the different qubit
# pairs and apply the two two-qubit gates that we just defined above. The
# way we iterate through the dictionary will depend on a gate sequence
# defined in the next section.
#

from itertools import combinations

gate_order = {"A":[], "B":[], "C":[], "D":[]}
for i, j in combinations(qubits, 2):
    wire_1 = qb2wire[i]
    wire_2 = qb2wire[j]
    if i in j.neighbors():
        if i.row == j.row and i.col % 2 == 0:
            gate_order["A"].append((wire_1, wire_2))
        elif i.row == j.row and j.col % 2 == 0:
            gate_order["B"].append((wire_1, wire_2))
        elif i.col == j.col and i.row % 2 == 0:
            gate_order["C"].append((wire_1, wire_2))
        elif i.col == j.col and j.row % 2 == 0:
            gate_order["D"].append((wire_1, wire_2))


######################################################################
# At this point we can define the gate sequence, which is the order the
# two-qubit gates are applied to the different qubit pairs. For example,
# ``["A", "B"]`` would mean that the two-qubit gates are first applied to
# all qubits connected with label A, and then, during the next full cycle,
# the two-qubit gates are applied to all qubits connected with label B.
# This would then correspond to a 2-cycle run (or a circuit with a depth of
# 2).
#
# While we can define any patterns we'd like, the two gate sequences below
# are the ones that are used in the supremacy paper. The shorter one is
# used for their classically verifiable benchmarking. The slightly
# longer sequence, which is much harder to simulate classically, is used
# for estimating the cross-entropy fidelity in what they call the "supremacy
# regime". We will use the shorter gate sequence for the following
# demonstration, although feel free to play around with other combinations.
#

m = 14  # number of cycles

# gate_sequence = np.resize(["A", "B", "C", "D", "C", "D", "A", "B"], m)
gate_sequence = np.resize(["A", "B", "C", "D"], m)


######################################################################
# Finally, we can define the circuit itself and create a QNode that we will
# use for circuit evaluation with the ``qsim`` device.
#
# The single-qubit gates are randomly selected and applied to each qubit in
# the circuit, while avoiding the same gate being applied to the same wire
# twice in a row. We do this by creating the following function. It
# generates a list with indices for the order in which the single qubit
# gates should be applied. We can then simply use this list from within the
# circuit to know which gate to apply when.
#

def generate_single_qubit_gate_list():
    # create the first list by randomly selecting indices
    # from `single_qubit_gates`
    g = [list(np.random.choice(range(len(single_qubit_gates)), size=wires))]

    for cycle in range(len(gate_sequence)):
        g.append([])
        for w in range(wires):
            # check which gate was applied to the wire previously
            one_gate_removed = list(range(len(single_qubit_gates)))
            bool_list = np.array(one_gate_removed) == g[cycle][w]

            # and remove it from the choices of gates to be applied
            pop_idx = np.where(bool_list)[0][0]
            one_gate_removed.pop(pop_idx)
            g[cycle + 1].append(np.random.choice(one_gate_removed))
    return g


######################################################################
# The two-qubit gates are applied to the qubits connected by A, B,
# C, or D as defined above. The circuit finally ends with a half-cycle,
# consisting of only a layer of single-qubit gates.
#
# We need both the probabilities of the circuit as well as being able to
# sample from it, as we will see in the next section. To do this, we add a
# keyword argument to our circuit allowing us to switch between the two
# returns. We sample from the Pauli-Z observable on all wires, which will
# give us the eigenvalues :math:`\pm 1` of the observable, corresponding to
# the states :math:`\left|0\right>` and :math:`\left|1\right>`.
#

@qml.qnode(dev)
def circuit(seed=42, return_probs=False):
    np.random.seed(seed)
    gate_idx = generate_single_qubit_gate_list()

    # m full cycles - single-qubit gates & two-qubit gate
    for i, gs in enumerate(gate_sequence):
        for w in range(wires):
            single_qubit_gates[gate_idx[i][w]](wires=w)

        for qb_1, qb_2 in gate_order[gs]:
            ops.ISWAP(wires=(qb_1, qb_2))
            ops.CPhase(-np.pi/6, wires=(qb_1, qb_2))

    # one half-cycle - single-qubit gates only
    for w in range(wires):
        single_qubit_gates[gate_idx[-1][w]](wires=w)

    if return_probs:
        return qml.probs(wires=range(wires))
    else:
        return [qml.sample(qml.PauliZ(i)) for i in range(wires)]


######################################################################
# The cross-entropy benchmarking fidelity
# ---------------------------------------
#
# The performance metric that is used in the paper, and the one that we
# will use in this demo, is called the linear cross-entropy benchmarking
# fidelity. It's defined as
#
# .. math::
#
#    F_{XEB} = 2^{n}\left<P(x_i)\right> - 1,
#
# where :math:`n` is the number of qubits, :math:`P(x_i)` is the
# probability of bitstring :math:`x_i` computed for the ideal quantum
# circuit, and the average is over the observed bitstrings.
#
# The idea behind using this fidelity is that it will be close to 1 for
# samples obtained from random quantum circuits, such as the one we defined
# above, and close to zero for a normal probability distribution, that can
# be effectively sampled from classically. Let us briefly calculate the
# expected theoretical value for the fidelity based on the number of qubits
# in the circuit. Sampling a bitstring from a random quantum circuit would
# follow the Porter-Thomas distribution [#Boixo2018]_, given by
#
# .. math::
#
#    Pr(p) = (N - 1)(1- p)^{N-2},
#
# where :math:`N = 2^n` is the number of possible bitstrings.
# From this we can then calculate the expectation value
# :math:`\left<P(x_i)\right>` as follows:
#
# .. math::
#
#    \left<P(x_i)\right> = \int_0^1 p^2 N (N-1)(1-p)^{N-2}dp = \frac{2}{N+1},
#
# which leads to the theoretical fidelity
#
# .. math::
#
#    F_{XEB} = 2^{n}\left<P(x_i)\right> - 1 = \frac{2N}{N+1} - 1.
#
# We implement this fidelity as the function below, where ``samples`` is a
# list sampled bitstrings, and ``probs`` is a list with corresponding
# sampling probabilities for the same noise-less circuit.
#

def fidelity_xeb(samples, probs):
    sampled_probs = []
    for bitstring in samples:
        # convert each bitstring into an integer
        bitstring_idx = int(bitstring, 2)

        # retrieve the corresponding probability for the bitstring
        sampled_probs.append(probs[bitstring_idx])

    return 2 ** len(samples[0]) * np.mean(sampled_probs) - 1


######################################################################
# We set a random seed and use it to calculate the probability for all the
# possible bitstrings. We can then sample from the same circuit, transform
# the Pauli-Z eigenvalue samples into their correponding bitstrings, and
# use them to calulate the cross-entropy benchmarking fidelity.
#

seed = np.random.randint(0, 42424242)
probs = circuit(seed=seed, return_probs=True)

samples = circuit(seed=seed).T

# take the eigenvalues and transform -1 to 1 and 1 to 0,
# and then join each sample together to form a bitstring
# (-(array([-1, 1] - 1) / 2 = array([1, 0])))
bitstring_samples = []
for sam in samples:
    new_sam = -(sam - 1) // 2
    bitstring_samples.append("".join(str(bs) for bs in new_sam))

f_circuit = fidelity_xeb(bitstring_samples, probs)

######################################################################
# Similarly, we can sample random bitstrings from a normal distribution by
# generating all basis states, along with their corresponding bitstrings,
# and sample directly from them using NumPy.
#

basis_states = dev.generate_basis_states(wires)
random_integers = np.random.randint(0, len(basis_states), size=shots)
bitstring_samples = []
for i in random_integers:
    bitstring_samples.append("".join(str(bs) for bs in basis_states[i]))

f_normal = fidelity_xeb(bitstring_samples, probs)

######################################################################
# Finally, let us print the two different values. Sampling from the
# circuit's probability distribution should give a fidelity close to 1,
# while sampling from a normal distribution should give a fidelity
# close to 0.
#
# .. note::
#
#     The cross-entropy benchmarking fidelity may output
#     values that are negative or that are larger than 1, for any finite
#     number of samples. This is due to the random nature of the sampling.
#     For an infinite amount of samples, or circuit runs, the observed
#     values will tend towards the theoretical ones, and will then always
#     lie in the 0-to-1 interval.
#
print("Circuit's distribution:", f"{f_circuit:.7f}".rjust(12))
print("Normal distribution:", f"{f_normal:.7f}".rjust(15))

######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Circuit's distribution:    1.0398803
#     Normal distribution:       0.0013487
#

######################################################################
# We can also calculate the theoretical result obtained from the equation
# above and compare it to the value obtained from sampling from the
# circuit. If we've done everything correctly, these two values should be
# very close to each other.
#

N = 2 ** wires
theoretical_value = 2 * N / (N + 1) - 1

print("Theoretical:", f"{theoretical_value:.7f}\n".rjust(24))

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Theoretical:               0.9995118
#

######################################################################
# To show that the fidelity from the circuit sampling actually tends
# towards the theoretical value we can run several different random
# circuits, calculate their respective cross-entropy benchmark fidelities
# and then calculate the mean fidelity of all the runs. The more
# evaluations we do, the closer to the theoretical value we should get.
#
# In the supremacy paper, they typically calculate each of their
# presented fidelities over ten circuit instances, which only differ
# in the choices of single-qubit gates. In this demo, we use even more
# instances to demonstrate a value closer to the theoretically obtained
# one.
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

print("Theoretical:", f"{theoretical_value:.7f}".rjust(24))

f_circuit = []
num_of_evaluations = 100
for i in range(num_of_evaluations):
    seed=np.random.randint(0, 42424242)

    probs = circuit(seed=seed, return_probs=True)
    samples = circuit(seed=seed).T

    bitstring_samples = []
    for sam in samples:
        new_sam = -(sam - 1) // 2
        bitstring_samples.append("".join(str(bs) for bs in new_sam))

    f_circuit.append(fidelity_xeb(bitstring_samples, probs))
    print(f"\r{i + 1:4d} / {num_of_evaluations:4d}{' ':17}{np.mean(f_circuit):.7f}", end="")
print("\rObserved:", f"{np.mean(f_circuit):.7f}".rjust(27))

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Theoretical:                0.9995118
#     Observed:                   0.9999512
#

######################################################################
# Quantum supremacy
# -----------------
#
# Why are we calculating this specific fidelity, and what does it actually
# mean if we get a cross-entropy benchmarking fidelity close to 1? This is
# an important question, containing one of the main arguments behind why
# this experiment is used to demonstrate quantum supremacy.
#
# Much is due to the Porter-Thompson probability distribution that the
# random quantum circuits follow, which are hard to simulate classically.
# On the other hand, a quantum device, running a circuit as the one
# constructed above, should be able to sample from such a distribution
# without much overhead. Thus, by showing that a quantum device can produce
# a high enough fidelity value for a large enough circuit, quantum
# supremacy can be claimed. This is exactly what Google's supremacy paper
# has done.
#
# There's still one issue that hasn't been touched on yet: the addition of
# noise in quantum hardware. Simply put, this noise will lower the
# cross-entropy benchmark fidelity, getting it closer to 0. The larger the
# circuit, the more noise there will be, and thus the lower the fidelity.
# By calculating the specific single-qubit, two-qubit, and readout errors
# of the Sycamore chip, using them to simulate a noisy circuit, the Google
# AI quantum team was able to compare the run-times with the output from
# their actual hardware device. This way, they managed to show that a
# significant speedup could be gained from using a quantum computer, and
# thus proclaimed quantum supremacy (see Fig. 4 in [#Arute2019]_).
#
# .. note::
#
#     For more reading on this, the original paper [#Arute2019]_ is highly
#     recommended (along with the suplementary information [#Arute2019sup]_ if you want
#     to dive deeper into the math and physics of the experiment). The blog
#     post in [#Sohaib2019]_, along with the accompanying GitHub repo, also provides
#     a nice introduction to the cross-entropy benchmark fidelity, and
#     includes calculations highlighting the effects of added noise models.
#

######################################################################
# References
# ----------
#
# .. [#Arute2019]
#
#     Arute, F., Arya, K., Babbush, R. et al. "Quantum supremacy using a programmable
#     superconducting processor"
#     `Nature 574, 505-510 (2019) <https://doi.org/10.1038/s41586-019-1666-5>`__.
#
# .. [#Arute2019sup]
#
#     Arute, F., Arya, K., Babbush, R. et al. Supplementary information for "Quantum
#     supremacy using a programmable superconducting processor"
#     `arXiv:1910.11333 (2019) <https://arxiv.org/abs/1910.11333>`__
#
# .. [#Martinis2020]
#
#     Martinis, John M. et al. (2020), `Quantum supremacy using a programmable
#     superconducting processor, Dryad, Dataset <https://doi.org/10.5061/dryad.k6t1rj8>`__
#
# .. [#Boixo2018]
#
#     Boixo, S., Isakov, S.V., Smelyanskiy, V.N. et al. Characterizing quantum supremacy
#     in near-term devices.
#     `Nature Phys 14, 595-600 (2018) <https://doi.org/10.1038/s41567-018-0124-x>`__
#
# .. [#Sohaib2019]
#
#     Sohaib, Alam M. and Zeng, W., `Unpacking the Quantum Supremacy Benchmark with Python
#     <https://medium.com/@sohaib.alam/unpacking-the-quantum-supremacy-benchmark-with-python-67a46709d>`__
#
