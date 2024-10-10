r"""Importing noise models from Qiskit
=======================================

Noise models describe the various ways in which a quantum system would interact with its
environment, leading to an evolution that is different from the ideal scenario and often
captured as a quantum master equation. In general, these models are represented by a set of 
`Kraus operators <https://pennylane.ai/qml/demos/tutorial_noisy_circuits/#noisy-operations>`_
acting on the quantum state that encapsulates the probabilistic nature of the quantum errors. ⚡

Importantly, different sets of Kraus operators can describe the same quantum noise process,
illustrating the non-unique nature of these representations and motivating how different quantum
computing libraries allow storing and building them to construct noise models. In this how-to
guide, we will first compare constructing them in `Qiskit <https://docs.quantum.ibm.com/>`_
and `PennyLane <https://docs.pennylane.ai/en/stable/code/qml.html>`_, and then
learn converting a Qiskit noise model into an equivalent PennyLane one.
"""

######################################################################
# Noise models in Qiskit and PennnyLane
# -------------------------------------
#
# In Qiskit, the noise models are built using the `noise module
# <https://qiskit.github.io/qiskit-aer/apidocs/aer_noise.html>`_
# in the ``Qiskit-Aer`` package. Each model is a `NoiseModel
# <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.NoiseModel.html>`_
# object that contains ``QuantumError`` to describe the errors encountered in gate operations.
# Optionally, it may also contain a ``ReadoutError`` that describes post-measurement classical
# readout errors.
#
# For example, the following builds a noise model that would insert depolarization
# errors for single-qubit gates, bit-flip errors for the target qubit of the two-qubit
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
# In contrast, the noise models in PennyLane are :class:`~.pennylane.NoiseModel` with
# ``conditionals`` (boolean conditions) that help select the operation to which noise
# is to be inserted and ``noise functions`` (callables) that apply the corresponding
# noise for the selected operation or measurement process based on some user-provided
# ``metadata``. This allows for a more functional construction as we can see by
# recreating the noise model from above:
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
# Therefore, defining the noise model this way gives us more flexibility on
# its essential components and makes its definition far more readable. 🧠
#
# Now it is important to verify whether both of these noise models work in the intended
# (and equivalent) way. For this purpose, we can use them while simulating a
# `GHZ state <https://en.wikipedia.org/wiki/Greenberger–Horne–Zeilinger_state>`_
# preparation circuit using ``default.mixed`` and ``qiskit.aer`` devices:
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
pl_noisy_circ = qml.QNode(GHZcircuit, dev_pl_noisy)
qk_noisy_circ = qml.QNode(GHZcircuit, dev_qk_noisy)

print(qml.draw(pl_noisy_circ, level="device", decimals=1, max_length=250)())

######################################################################
# Now let us compare the results to see the equivalence between the two noise models:
#

# Obtain the results from the simulations
pl_noisy_res, qk_noisy_res = pl_noisy_circ(), qk_noisy_circ()
pl_probs = np.array(list(pl_noisy_res.values())) / n_shots
qk_probs = np.array(list(qk_noisy_res.values())) / n_shots

print("PennyLane Results: ", np.round(pl_probs, 3))
print("Qiskit Results:    ", np.round(qk_probs, 3))
print("Are results equal? ", np.allclose(pl_probs, qk_probs, atol=0.01))

######################################################################
# Importing Qiskit noise models
# -----------------------------
#
# We hope at least some of you will be wondering if one could automate the
# conversion of a Qiskit noise model into an equivalent PennyLane noise model.
# The answer to this question is YES! 🤩
#
# We can see this in practice for a
# `GenericBackendV2 <https://docs.quantum.ibm.com/api/qiskit/qiskit.providers.fake_provider.GenericBackendV2>`_
# backend that gets instantiated with the error data generated
# and sampled randomly from historical IBM backend data:
#

from qiskit.providers.fake_provider import GenericBackendV2

# Generates a two-qubit simulated backend
backend = GenericBackendV2(num_qubits=2, seed=42)
qk_noise_model = NoiseModel.from_backend(backend)
print(qk_noise_model)

######################################################################
# To import this into PennyLane we can use the :func:`~pennylane.from_qiskit_noise` function:
#

pl_noise_model = qml.from_qiskit_noise(qk_noise_model)
print(pl_noise_model)

######################################################################
# This conversion leverages the standard Kraus representation of the errors
# stored in the ``qk_noise_model``. Internally, this is done in a smart three-step process:
#
# 1. First, all the basis gates from the noise model are mapped to the corresponding PennyLane
#    gate `operations <https://docs.pennylane.ai/en/stable/introduction/operations.html>`_.
# 2. Next, the operations with noise are mapped to the corresponding error channels
#    defined via :class:`~.pennylane.QubitChannel`.
# 3. Finally, the `Boolean conditionals <https://docs.pennylane.ai/en/stable/code/qml_noise.html#boolean-functions>`_
#    are constructed and combined based on their associated errors.
#
# This can be done for any noise model defined in Qiskit with a minor catch that the
# classical readout errors are not supported yet in PennyLane. But worry not,
# these will be supported soon too! 🧑‍💻
#

######################################################################
# Conclusion
# ----------
#

######################################################################
# Qiskit provides noise models and tools that one could use to mirror the behavior of their quantum
# devices. Integrating them into PennyLane is a powerful way for enabling users to perform noisy
# simulations that help them study the effects on noise on quantum circuits and develop noise-robust
# quantum algorithms. As shown in this how-to guide, users can construct these noise
# models based on the Qiskit ones either manually by building one-to-one mapping for
# each kind of error or do it automatically using the provided conversion functionality. 💪
#
# Should you have any questions about using noise models in PennyLane, you can consult the
# `noise module documentation <https://docs.pennylane.ai/en/stable/code/qml_noise.html>`_,
# the :doc:`noise model how-to <tutorial_how_to_use_noise_models>`, the PennyLane Codebook
# module on `Noisy Quantum Theory <https://pennylane.ai/codebook/#06-noisy-quantum-theory>`_,
# or create a post on the `PennyLane Discussion Forum <https://discuss.pennylane.ai>`_.
#

######################################################################
# About the author
# ----------------
