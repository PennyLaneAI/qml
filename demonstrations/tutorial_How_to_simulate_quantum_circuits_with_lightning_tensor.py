r"""How to simulate quantum circuits with tensor networks with lightning.tensor
====================================================================

:doc:`Tensor networks </demos/tutorial_tn_circuits>` are widely used for the large-scale quantum circuits simulation. With the approximated represention of quantum states, tensor networks can be used to simulate quantum circuits that state-vector based simulators cannot handle.
The ``lightning.tensor`` device is newly added to Pennylane ecosystem as of `v0.39 release <https://pennylane.ai/blog/2024/11/pennylane-release-0.39>`__, as an alternative to the :class:`~pennylane.devices.default_tensor.DefaultTensor` device. The ``lightning.tensor`` is a high-performance 
simulator, which can harness the computational power of Nvidia GPUs. This deivce is built on top of the C/C++ APIs offered by the `cutensornet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>`__ library, a component of the `Nvidia cuQuantum SDK <https://docs.nvidia.com/cuda/cuquantum/latest/index.html>`__.
The ``lightning.tensor`` device is designed to simulate large-scale quantum circuits efficiently. It can handle circuits with 30+ qubits with a manageable degree of entanglement within only a single GPU. Note that ``lightning.tensor`` is still under active development, and further improvements, 
new features, and additional tutorials are expected in future releases.

In this tutorial, we demonstrate how to simulate quantum circuits using the `lightning.tensor` device in PennyLane. The :class:`~pennylane_lightning.lightning_tensor.LightningTensor` device, similar to the :class:`~pennylane.devices.default_tensor.DefaultTensor` device, supports both Matrix Product State (MPS) and Exact Tensor 
Network (ExactTN) representations of quantum states. For more information on the MPS and TN methods, we refer to the `Matrix Product States and Tensor Networks </tutorial_How_to_simulate_quantum_circuits_with_tensor_networks>`.

Check the latest functionality in the :class:`documentation <.pennylane_lightning.lightning_tensor.LightningTensor>` or pick among other `PennyLane devices <https://pennylane.ai/plugins/#built-in-devices>`__ for your project.

.. figure:: ../_static/demonstration_assets/how_to_simulate_quantum_circuits_with_tensor_networks/TN_MPS.gif
    :align: center
    :width: 90%

"""

######################################################################
# Simulating a quantum circuit with the MPS method
# ------------------------------------------------
#
# Let's start by showing how to simulate a quantum circuit using the MPS method. Generally, the MPS method can be used to simulate quantum circuits that are too large for state-vector simulations or too deep (higher degree of entanglement) for the exact TN contract method.
# The circuit we show here is a simple example to demonstrate the features of a MPS method mentioned above.
#

import pennylane as qml
import numpy as np

# Define the keyword arguments for the MPS method
kwargs_mps = {
    # Maximum bond dimension of the MPS
    "max_bond_dim": 128,
    # Cutoff parameter for the singular value decomposition
    "cutoff": np.finfo(np.complex128).eps,
}

# Parameters of the quantum circuit
theta = 0.5
phi = 0.1
depth = 10
n = 1011
num_qubits = 100

# Instantiate the device with the MPS method and the specified kwargs
dev = qml.device("lightning.tensor", wires=num_qubits, method="mps", **kwargs_mps)


# Define the quantum circuit
@qml.qnode(dev)
def circuit(theta, phi, n, num_qubits):
    for _ in range(1, depth - 1):
        for qubit in range(num_qubits - 4):
            qml.RX(theta, wires=qubit)
            qml.CNOT(wires=[qubit, qubit + 1])
            qml.RY(phi, wires=qubit)
            qml.DoubleExcitation(theta, wires=[qubit, qubit + 1, qubit + 3, qubit + 4])
            qml.Toffoli(wires=[qubit + 1, qubit + 3, qubit + 4])
            qml.FlipSign(n, wires=range(num_qubits))
    return qml.expval(qml.X(num_qubits - 1) @ qml.Y(num_qubits - 2) @ qml.Z(num_qubits - 3))


######################################################################
# We set the maximum bond dimension to 128 and the ``cutoff`` parameter is set to the machine epsilon of the ``numpy.complex128`` data type.
# For this circuit, retaining a maximum of 128 singular values in the singular value decomposition is more than enough to represent the quantum state accurately.
# For an explanation of these parameters, we refer to the :class:`documentation <.pennylane_lightning.lightning_tensor.LightningTensor>` of the ``lightning.tensor`` device.
# Please note that the accepted keyword arguments for ``lightning.tensor`` are slightly different from the ``default.tensor`` device.
#
# In general, a circuit run on a ``lightning.tensor`` device could be faster than on a CPU-based ``default.tensor`` device, given a sufficient large bond dimension is used in the
# calculations. The exact performance of those devices depends on the gates in the specific circuit. For example, the ``lightning.tensor`` device natively supports multi-controlled 1-wire target gates,
# such as the `qml.FlipSign` operator, which is widely used in the Grover algorithm.
#
#

import time

# Simulate the circuit for different numbers of qubits
print(f"Number of qubits: {num_qubits}")
start_time = time.time()
result = circuit(theta, phi, n, num_qubits)
end_time = time.time()
print(f"Result: {result}")
print(f"Execution time: {end_time - start_time:.4f} seconds")

######################################################################
# Unlike ``default.tensor``, the graph contraction operation is not carried out immediately after each gate application.
# Instead, the tensor network is lazily built up, and the contraction is performed only when the final MPS state calcalution is requested.
#
# To learn more about the MPS method and its theoretical background, we refer to the `Default.Tensor Demo<demos/tutorial_How_to_simulate_quantum_circuits_with_tensor_networks>`.
#

######################################################################
# Simulating a quantum circuit with the TN method
# -----------------------------------------------
#
# The TN method fully captures the entanglement among qubits without approximation and is more accurate than the MPS method. While, it might require more computational and memory resources than the MPS method.
# The memory resource required for the TN method is proportional to the number of entangled qubits. Therefore, the TN method is more suitable for simulating shadow quantum circuits with a less degree of entanglement.
# In the following example, we consider a shadow quantum circuit with a configurable depth and less entangled gates.
#

import pennylane as qml
import numpy as np
import time

# Parameters of the quantum circuit
theta = 0.5
phi = 0.1
depth = 10
n = 1011
num_qubits = 100

# Instantiate the device with the TN method and the specified kwargs
dev = qml.device("lightning.tensor", wires=num_qubits, method="tn")


@qml.qnode(dev)
def circuit(theta, depth, n, num_qubits):
    for i in range(num_qubits):
        qml.X(wires=i)
    for _ in range(1, depth - 1):
        for i in range(0, num_qubits, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(num_qubits % 5):
            qml.RZ(theta, wires=i)
        for i in range(1, num_qubits - 1, 2):
            qml.CZ(wires=[i, i + 1])
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, (i + 1)])
    return qml.expval(qml.X(num_qubits - 1))


# Simulate the circuit for different numbers of qubits
print(f"Number of qubits: {num_qubits}")
start_time = time.time()
result = circuit(theta, depth, n, num_qubits)
end_time = time.time()
print(f"Result: {result}")
print(f"Execution time: {end_time - start_time:.4f} seconds")

######################################################################
# Here, we lazily attach each gate to the tensor network and only perform the contraction when a measurement call is requested.
# Note that the TN method could be more memory-intensive than the MPS method, as it requires storing the full tensor network.

######################################################################
# Conclusion
# ----------
# In this tutorial, we have shown how to simulate quantum circuits using the ``lightning.tensor`` device in PennyLane. We have demonstrated how to simulate quantum circuits using the MPS and TN methods, which are supported by the ``lightning.tensor`` device.
# Note that the ``lightning.tensor`` device is still under active development, and further improvements, new features, and additional tutorials/demos are expected in future releases.

# About the author
# ----------------
# .. include:: ../_static/authors/shuli_shu.txt
