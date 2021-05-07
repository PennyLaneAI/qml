# Metadata

# Introduction to classical shadows

# Outline for Classical shadow algorithm.
import pennylane as qml
import pennylane.numpy as np


def classical_shadow(circuit_template, params, num_shadows: int) -> np.ndarray:
    unitary_ensemble = {0: lambda n: qml.Identity(n),
                        1: lambda n: qml.PauliX(n),
                        2: lambda n: qml.PauliY(n),
                        3: lambda n: qml.PauliZ(n)}

    device_shadow = qml.device('default.qubit', wires=nqubits, shots=1)
    # sample random unitaries, where 0 = I and 1,2,3 = X,Y,Z
    randints = np.random.randint(1, 4, size=(number_of_shadows, nqubits))
    unitaries = []
    for ns in range(num_shadows):
        unitaries.extend(unitary_ensemble[int(randints[ns, i])](i) for i in range(nqubits))
    samples = qml.map(circuit_template, observables=unitaries, device=device_shadow, measure='sample')(params)
    # reshape from dim 1 vector to matrix, each row now corresponds to a shadow
    samples = samples.reshape((number_of_shadows, nqubits))
    # combine the computational basis outcomes and the sampled unitaries
    return np.concatenate([samples, randints], axis=1)


def estimate_shadow_obervable(shadows, observable):
    sum_product, cnt_match = 0, 0
    for single_measurement in shadows:
        not_match = 0
        product = 1
        for pauli_XYZ, position in observable:
            if pauli_XYZ != single_measurement[position + nqubits]:
                not_match = 1
                break
            product *= single_measurement[position]
        if not_match == 1: continue

        sum_product += product
        cnt_match += 1

    return sum_product / cnt_match


# Simple example of classical shadow computation for qubits

nqubits = 1

device_exact = qml.device('default.qubit', wires=nqubits)

number_of_shadows = 1000

paulis = {0: qml.Identity(0).matrix,
          1: qml.PauliX(0).matrix,
          2: qml.PauliY(0).matrix,
          3: qml.PauliZ(0).matrix}

theta = np.random.randn(nqubits, )


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


# Application of classical shadow/ performance comparison
# TODO: what is a good application?
