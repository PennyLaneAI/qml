import pennylane as qml
import numpy as np
np.set_printoptions(linewidth=np.inf)

def state_preparation(state):
    qml.QubitStateVector(state, wires=["A"])

def entangle_qubits():
    qml.Hadamard(wires="a")
    qml.CNOT(wires=["a", "B"])

def basis_rotation():
    qml.CNOT(wires=["A", "a"])
    qml.Hadamard(wires="A")

def measure_and_update():
    m0 = qml.measure("A")
    m1 = qml.measure("a")
    qml.cond(m1, qml.PauliX)("B")
    qml.cond(m0, qml.PauliZ)("B")

dev = qml.device("default.qubit", wires=["A", "a", "B"], shots=None)

@qml.qnode(dev)
def teleport(state):
    state_preparation(state)
    entangle_qubits()
    basis_rotation()
    measure_and_update()
    return qml.density_matrix(wires=["B"])

def teleport_state(state):
    """
    Wrapper function that ensures the density matrix on Bob's wire
    represents the state Alice is teleporting.
    """
    teleported_dm = teleport(state)
    expected_dm = np.outer(state, np.conj(state))
    if not np.allclose(teleported_dm, expected_dm):
        raise Exception(
            "The teleported density matrix does not represent the state being teleported.\n\n"
            f"State being teleported:\n{state}\n\n"
            f"Density matrix:\n{teleported_dm}"
        )
    print("State successfully teleported!")

state = np.array([1/np.sqrt(5), 2j/np.sqrt(5)])
print(qml.draw(teleport_state)(state))
