r"""Importing noise models from Qiskit
=======================================

Noise models describe the various ways in which a quantum system would interact with its
environment, leading to an evolution that is different from the ideal scenario and often captured as
a quantum master equation. In general, these models are represented by a set of Kraus operators
acting on the quantum state that encapsulates the probabilistic nature of the quantum errors.
Importantly, different sets of Kraus operators can describe the same quantum noise process,
illustrating the non-unique nature of these representations and motivating how different quantum
computing libraries allow storing and building them to construct noise models. In this how-to guide,
we will comparatively study noise constructing noise models in Qiskit and PennyLane. While doing so,
we will also learn to convert a noise model built in Qiskit into an equivalent PennyLane one.
Finally, before diving straight into it, to get a better understanding of noise channels and noise
models, check out the Noisy Circuits and How to use noise models in PennyLane tutorials.
"""

######################################################################
# Noise models in Qiskit and PennnyLane
# -------------------------------------
# 
# In Qiskit, the noise models are built with the ``noise`` module in the ``Qiskit-Aer`` package. Each
# model is a ``NoiseModel`` object that contains ``QuantumError`` objects to describe the errors
# encountered in gate operations. Optionally, it may also contain a ``ReadoutError`` object that
# describes a post-measurement classical readout error as a confusion matrix with elements
# :math:``P_{ij} = \mathrm{P}(i | j)`` storing probability of measuring state ``i`` given the system
# was in the state ``j``. For example, the following builds a Qiskit noise model that would insert
# depolarization errors for single-qubit gates, bit-flip errors for the target qubit of the two-qubit
# gates, and thermalization errors for each measurement:
# 

import numpy as np
from qiskit_aer.noise import (
    depolarizing_error, pauli_error, thermal_relaxation_error, NoiseModel
)

# Noise model metadata
n_qubits = 3
prob_gate, prob_flip, prob_exc = 0.2, 0.1, 0.2
Tgs = np.random.uniform(1000, 1005, size=n_qubits)
T1s = np.random.normal(50e3, 10e3, n_qubits)
T2s = np.random.normal(70e3, 10e3, n_qubits)
T2s = np.min((T2s, 2*T1s), axis=0) # T2 <= 2*T1

# Building the noise models
noise_model_qk = NoiseModel()

error_gate1 = depolarizing_error(prob_gate, 1)
noise_model_qk.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])

error_gate2 = pauli_error([('X', prob_flip), ('I', 1 - prob_flip)]).tensor(
    pauli_error([('I', 1)])
)
noise_model_qk.add_all_qubit_quantum_error(error_gate2, ["cx"])

for idx, qubit in enumerate(range(3)):
    error_meas = thermal_relaxation_error(T1s[idx], T2s[idx], Tgs[idx], prob_exc)
    noise_model_qk.add_quantum_error(error_meas, "measure", [qubit])

print(noise_model_qk)

######################################################################
# In contrast, the noise models in PennyLane are ``NoiseModel`` objects with ``conditionals`` (boolean
# conditions) that help select the operation to which noise is to be inserted and ``noise functions``
# (callables) that apply the corresponding noise for the selected operation or measurement process
# based on some user-provided metadata such as thermal-relaxation times of the qubits. This allows for
# a functional construction as we can see by recreating the noise model from above:
# 

import pennylane as qml

noise_model_pl = NoiseModel()

gate1_fcond = qml.noise.op_in(["U1", "U2", "U3"]) & qml.noise.wires_in(range(n_qubits))
gate1_noise = qml.noise.partial_wires(qml.DepolarizingChannel, prob_gate)

gate2_fcond = qml.noise.op_eq("CNOT")
def gate2_noise(op, **metadata):
    qml.BitFlip(prob_flip, op.wires[1])

rmeas_fcond = qml.noise.meas_eq(qml.counts)
def rmeas_noise(op, **metadata):
    for wire in op.wires:
        qml.ThermalRelaxationError(prob_exc, T1s[wire], T2s[wire], Tgs[wire], wires=wire)

noise_model_pl = qml.NoiseModel(
    {gate1_fcond: gate1_noise, gate2_fcond: gate2_noise}, {rmeas_fcond: rmeas_noise},
)
print(noise_model_pl)

######################################################################
# As one can see defining the noise model this way gives us more flexibility on its essential
# components and makes its definition far more readable. Now it is important to verify whether both of
# these noise models work in the intended (and equivalent) way. For this purpose, we test them for
# simulating a simple 3-qubit GHZ state preparation circuit using the PennyLane-Qiskit plugin:
# 

# Preparing the devices:
n_shots = int(2e5)
dev_pl_ideal = qml.device("default.mixed", wires=n_qubits, shots=n_shots)
dev_pl_noisy = qml.add_noise(dev_pl_ideal, noise_model_pl)
dev_qk_noisy = qml.device("qiskit.aer", wires=n_qubits, shots=n_shots, noise_model=noise_model_qk)

def GHZcircuit():
    qml.U2(0, np.pi, wires=[0])
    for wire in range(n_qubits-1):
        qml.CNOT([wire, wire + 1])
    return qml.counts(wires=range(n_qubits), all_outcomes=True)

# Preparing the circuits:
pl_ideal_circ = qml.QNode(GHZcircuit, dev_pl_ideal)
pl_noisy_circ = qml.QNode(GHZcircuit, dev_pl_noisy)
qk_noisy_circ = qml.QNode(GHZcircuit, dev_qk_noisy)

print(qml.draw(pl_noisy_circ, level="device", decimals=1, max_length=250)())

######################################################################

import matplotlib.pyplot as plt

# Obtain the results from the simulations
pl_ideal_res = pl_ideal_circ()
pl_noisy_res, qk_noisy_res = pl_noisy_circ(), qk_noisy_circ()

# Create the plot
width = 0.15
pos, labels = np.arange(2**n_qubits), list(pl_noisy_res.keys())
plt.figure(figsize=(10, 4))
plt.bar(pos - width, np.array(list(qk_noisy_res.values()))/n_shots, width, label="Noisy (Qiskit)")
plt.bar(pos, np.array(list(pl_ideal_res.values()))/n_shots, width, label="Ideal results")
plt.bar(pos + width, np.array(list(pl_noisy_res.values()))/n_shots, width, label="Noisy (PennyLane)")

# Plot metadata
plt.xlabel("Bitstring")
plt.ylabel("Probability")
plt.xticks(pos, labels)
plt.legend()
plt.show()

######################################################################
# Importing Qiskit noise models
# -----------------------------
# 
# I hope at least some of you will be wondering if there is a way one could automate converting a
# Qiskit noise model into an equivalent PennyLane noise model. The answer to this question is yes!
# PennyLane provides ``qml.from_qiskit_noise`` method that helps with that. We can understand using
# this functionality for a GenericBackendV2 backend instance that gets instantiated with the error
# data generated and sampled randomly from historical IBM backend data.
# 

from qiskit.providers.fake_provider import GenericBackendV2
 
# Generate a 5-qubit simulated backend
backend = GenericBackendV2(num_qubits=2, seed=42)
qk_noise_model = NoiseModel.from_backend(backend)
print(qk_noise_model)

######################################################################

qml.from_qiskit_noise(qk_noise_model)

######################################################################
# As one may see this conversion is one-to-one and leverages the standard Kraus representation of the
# errors present in ``qk_noise_model``. Internally, this is done in a three-step process. First, all
# the basis gates from the noise model are mapped to the corresponding PennyLane gate operations.
# Then, the operations with noise are mapped to the corresponding error channels defined via
# ``QubitChannel``. Finally, the Boolean conditionals for selecting the gate operations are
# constructed and combined based on the errors. This can be done for any noise model defined in Qiskit
# with a minor catch that the classical readout error is currently not supported yet in PennyLane!
# 

######################################################################
# Conclusion
# ----------
# 

######################################################################
# Qiskit provides noise models and tools that one could use to mirror the behavior of their quantum
# devices. Integrating them into PennyLane is a powerful way for enabling users to perform noisy
# simulations that help them study the effects on noise on quantum circuits and develop noise-aware
# quantum algorithms. The ability to model realistic quantum noise is crucial to make simulations
# robust to noise and can now be done in an easy manner using PennyLane’s NoiseModel class. As shown
# in this how-to guide, users can construct these noise models based on the Qiskit ones either
# manually by building one-to-one mapping for each kind of error or do it automatically using the
# provided conversion functionality.
# 
# Should you have any questions about using noise models in PennyLane, you can consult the noise
# module documentation, the nosie model how-to, or create a post on the PennyLane Discussion Forum.
# You can also follow us on X (formerly Twitter) or LinkedIn to stay up-to-date with the latest and
# greatest from PennyLane!
# 