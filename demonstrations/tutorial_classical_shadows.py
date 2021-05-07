r"""
Classical Shadows
=================
.. meta::
    :property="og:description": Learn how to efficiently make predictions about an unkown 
        quantum state using the classical shadow approximation. 
    :property="og:image": https://pennylane.ai/qml/_images/qaoa_layer.png

.. related::

*Authors: Roeland Wiersema & Brian Doolittle (Xanadu Residents).
Posted: 7 May 2021. Last updated: 7 May 2021.*

This tutorial is based on the classical shadow approximation and applications discussed
in `this paper <https://arxiv.org/pdf/2002.08953.pdf>`_.

How do you efficiently characterize an unknown quantum state?
This task is formally known as quantum state tomography and simply requires many repetitions
of preparing and measuring a quantum state.
The number of distinct bases in which measurements are needed grows exponentially with the Hilbert
space of the quantum state.
This scaling presents a challenge in quantum computing because it is intractable to perform the
number of measurements needed to completely characterize an unknown quantum state.

A solution to this problem is to use the classical shadow approximation.
In this procedure
1. A quantum state $\rho$ is prepared.
2. A randomly selected unitary $U$ is applied
3. A computational basis measurement is performed.
4. The process is repeated $N$ times.

The classical shadow is then constructed as a list of measurement outcomes and chosen unitaries.

"""

# Outline for Classical shadow algorithm.
import pennylane as qml
import pennylane.numpy as np


def classical_shadow(circuit_template, params, num_shadows: int) -> np.ndarray:
    # create a dict so we can quickly call qml.Observables from integers.
    unitary_ensemble = {0: lambda n: qml.Identity(n),
                        1: lambda n: qml.PauliX(n),
                        2: lambda n: qml.PauliY(n),
                        3: lambda n: qml.PauliZ(n)}
    # each shadow is one shot, so we set this parameter in the qml.device
    device_shadow = qml.device('default.qubit', wires=nqubits, shots=1)
    # sample random unitaries uniformly, where 0 = I and 1,2,3 = X,Y,Z
    randints = np.random.randint(0, 4, size=(number_of_shadows, nqubits))
    unitaries = []
    for ns in range(num_shadows):
        # for each shadow, add a random Clifford observable at each location
        unitaries.extend(unitary_ensemble[int(randints[ns, i])](i) for i in range(nqubits))
    # use the QNodeCollections through qml.map to calculate multiple observables for the same circuit
    samples = qml.map(circuit_template, observables=unitaries, device=device_shadow, measure='sample')(params)
    # reshape from dim 1 vector to matrix, each row now corresponds to a shadow
    samples = samples.reshape((number_of_shadows, nqubits))
    # combine the computational basis outcomes and the sampled unitaries
    return np.concatenate([samples, randints], axis=1)


def estimate_shadow_obervable(shadows, observable):
    # https://github.com/momohuang/predicting-quantum-properties
    sum_product, cnt_match = 0, 0
    # loop over the shadows:
    for single_measurement in shadows:
        not_match = 0
        product = 1
        # loop over all the paulis that we care about
        for pauli_XYZ, position in observable:
            # if the pauli in our shadow does not match, we break and go to the next shadow
            if pauli_XYZ != single_measurement[position + nqubits]:
                not_match = 1
                break
            product *= single_measurement[position]
        # don't record the shadow
        if not_match == 1: continue

        sum_product += product
        cnt_match += 1

    return sum_product / cnt_match


# Simple example of classical shadow computation for qubits

nqubits = 1
theta = np.random.randn(nqubits, )
device_exact = qml.device('default.qubit', wires=nqubits)
number_of_shadows = 1000

paulis = {0: qml.Identity(0).matrix,
          1: qml.PauliX(0).matrix,
          2: qml.PauliY(0).matrix,
          3: qml.PauliZ(0).matrix}


def circuit(params, wires, **kwargs):
    for i in range(nqubits):
        qml.Hadamard(wires=i)
        qml.RY(params[i], wires=i)


# Estimate Tr{X_0 rho}
exact_observable = qml.map(circuit, observables=[qml.PauliX(0)], device=device_exact, measure='expval')(theta)
print(f"Exact value: {exact_observable[0]}")
shadows = classical_shadow(circuit, theta, number_of_shadows)
# The observable X_0 is passed as [(1,0), ] something like X_0 Y_1 would be  [(1,0),(2,1)]
# TODO: Generalize the mapping from Pauli strings to tuples
shadow_observable = estimate_shadow_obervable(shadows, [(1, 0), ])
print(f"Shadow value: {shadow_observable}")

# Multi-qubit application of classical shadow

nqubits = 2
theta = np.random.randn(nqubits, )
device_exact = qml.device('default.qubit', wires=nqubits)
number_of_shadows = 1000

exact_observable = qml.map(circuit, observables=[qml.PauliX(0) @ qml.PauliX(1)], device=device_exact, measure='expval')(
    theta)
print(f"Exact value: {exact_observable[0]}")
shadows = classical_shadow(circuit, theta, number_of_shadows)
shadow_observable = estimate_shadow_obervable(shadows, [(1, 0), (1, 1)])
print(f"Shadow value: {shadow_observable}")

# Application of classical shadow/ performance comparison
# TODO: what is a good application?
