r"""How to simulate quantum circuits with tensor networks
====================================================================

Tensor networks are a powerful computational tool for simulating quantum circuits.
They provide a way to represent quantum states and operations in a compact form. 
Unlike the state vector approach, tensor networks are particularly useful for large-scale simulations of quantum circuits.

Here, we demonstrate how to simulate quantum circuits using the ``default.tensor`` device in PennyLane.
This simulator is based on `quimb <https://quimb.readthedocs.io/en/latest/>`__, a Python library for tensor network manipulations. 
The ``default.tensor`` device is convenient for simulations with tens, hundreds, or even thousands of qubits.
Other devices based on the state vector approach may be more suitable for small circuits 
since the overhead of tensor network contractions can be significant.

TODO: Insert figure

"""

######################################################################
# Choosing the method to simulate quantum circuits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The ``default.tensor`` device can simulate quantum circuits using two different computational methods.
# We only need to specify the ``method`` keyword argument when instantiating the device.
#
# The first is the Matrix Product State (MPS) representation, and the second is the Tensor Network (TN) approach.
# The MPS representation can be seen as a particular case of the TN method, where the tensor network has a one-dimensional structure.
# If not specified, the default method is the MPS representation.
#

######################################################################
# Simulating a quantum circuit with the MPS method
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's start by showing how to simulate a quantum circuit using the Matrix Product State (MPS) method.
# If the number of wires is not specified when instantiating the device, it is inferred from the circuit at runtime.
#
# We choose the maximum bond dimension as ``None``, meaning the bond dimension is not restricted.
# The ``cutoff`` parameter is set to the machine epsilon of the ``numpy.complex128`` data type.
# This value is the threshold below which to discard the singular coefficients of the Singular Value Decomposition (SVD) in the MPS method.
#

import pennylane as qml
import numpy as np

# Define the keyword arguments for the MPS method
kwargs_mps = {
    "max_bond_dim": None,
    "cutoff": np.finfo(np.complex128).eps,
    "contract": "auto-mps",
}

# The parameters of the quantum circuit
theta = 0.5
phi = 0.1

# Instantiate the device with the MPS method and the specified kwargs
dev = qml.device("default.tensor", method="mps", **kwargs_mps)


# Define the quantum circuit
@qml.qnode(dev)
def circuit(theta, phi, num_qubits):
    for qubit in range(num_qubits - 4):
        qml.RX(theta, wires=qubit + 1)
        qml.CNOT(wires=[qubit, qubit + 1])
        qml.RY(phi, wires=qubit + 1)
        qml.DoubleExcitation(theta, wires=[qubit, qubit + 1, qubit + 3, qubit + 4])
        qml.Toffoli(wires=[qubit + 1, qubit + 3, qubit + 4])
    return qml.expval(
        qml.X(num_qubits - 1) @ qml.Y(num_qubits - 2) @ qml.Z(num_qubits - 3)
    )


######################################################################
# We can now simulate the quantum circuit for different numbers of qubits.
# The execution time will generally increase as the number of qubits grows.
#

import time

# Simulate the circuit for different numbers of qubits
for num_qubits in range(50, 201, 50):
    print(f"Number of qubits: {num_qubits}")
    start_time = time.time()
    result = circuit(theta, phi, num_qubits)
    end_time = time.time()
    print(f"Result: {result}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")


######################################################################
# We can also visualize the tensor network representation of the quantum circuit with the ``draw`` method.
# This method generates a graphical representation of the tensor network using the ``quimb``'s plotting functionalities.
#
# Since we did not specify the number of qubits when instantiating the device,
# the number of tensors in the tensor network is inferred from the last execution of the quantum circuit.
# Let's visualize the MPS representation of the quantum circuit with 12 qubits.
#

circuit(theta, phi, num_qubits=12)
dev.draw(color="auto", show_inds=True, figsize=(8, 6))

##############################################################################
# .. figure:: ../_static/demonstration_assets/how_to_simulate_quantum_circuits_with_tensor_networks/MPS_circuit.png
#    :align: center
#    :width: 90%

######################################################################
# Simulating a quantum circuit with the TN method
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The Tensor Network (TN) method is a more general approach than the Matrix Product State (MPS) method.
# It can be more efficient for simulating quantum circuits when the MPS method
# may require a large bond dimension to accurately represent the state, leading to increased computational and memory costs.
# For example, this method can be helpful in simulating circuits with a higher degree of entanglement.
#

import pennylane as qml
import numpy as np

# Define the keyword arguments for the TN method
kwargs_tn = {
    "contract": "auto-split-gate",
    "local_simplify": "ADCRS",
    "contraction_optimizer": "auto-hq",
}

# The parameters of the quantum circuit
theta = 0.5
phi = 0.1
depth = 10

# Instantiate the device with the TN method and the specified kwargs
dev = qml.device("default.tensor", method="tn", **kwargs_tn)


@qml.qnode(dev)
def circuit(theta, depth, num_qubits):
    for i in range(num_qubits):
        qml.X(wires=i)
    for _ in range(1, depth - 1):
        for i in range(0, num_qubits, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(num_qubits % 5):
            qml.RZ(theta, wires=i)
        for i in range(1, num_qubits - 1, 2):
            qml.CZ(wires=[i, i + 1])
    for i in range(num_qubits):
        qml.CNOT(wires=[i, (i + 1)])
    return qml.var(qml.X(num_qubits - 1))


# Simulate the circuit for different numbers of qubits
for num_qubits in range(25, 101, 25):
    print(f"Number of qubits: {num_qubits}")
    start_time = time.time()
    result = circuit(theta, depth, num_qubits)
    end_time = time.time()
    print(f"Result: {result}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

######################################################################
# As before, let's visualize the tensor network representation of the circuit.
#

circuit(theta, depth, num_qubits=15)
dev.draw(color="auto", show_inds=True, figsize=(7, 7))

##############################################################################
# .. figure:: ../_static/demonstration_assets/how_to_simulate_quantum_circuits_with_tensor_networks/TN_circuit.png
#    :align: center
#    :width: 90%

######################################################################
# We can see that the Tensor Network (TN) method generates a more complex tensor network than the Matrix Product State (MPS) method.
# In this case, the representation depends on the structure of the quantum circuit and the specified keyword arguments.
#

######################################################################
# About the authors
#
#  .. include:: ../_static/authors/pietropaolo_frisoni.txt
#
