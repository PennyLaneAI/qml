r"""
How to run QJIT compiled PennyLane programs on NVIDIA GPUs
==========================================================


Our NVIDIA exclusive GPU-enabled state-vector simulator, Lightning-GPU, has been
recently integrated to `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__.
This enables just-in-time (QJIT) compiled quantum operations to execute on
`cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`__ compatible GPUs.

Here, we'll show you how to use NVIDIA's floating-point GPU workhorses to unlock
the execution of larger quantum functions and achieve greater preformance
for your QJIT-compiled programs.

.. figure:: ../_static/demonstration_assets/qpe/tutorial_qjit_lgpu.png
     :align: center
     :width: 65%
     :target: javascript:void(0);


Set up your environment
-----------------------

To bring the power of `Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__
to `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__, we need a Linux machine with
`the NVIDIA cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`__
libraries which require a CUDA capable GPU of generation SM 7.0 (Volta) and greater.
Then, you can install PennyLane, Lightning-GPU and Catalyst from PyPI via

``` console
    pip install "pennylane==0.40.0" "pennylane-lightning-gpu==0.40.0" "pennylane-catalyst==0.10.0"
```

Now you can simply create a ``lixghtning.gpu`` device, compile your circuit
with ``qml.qjit`` and run as usual!

"""

import pennylane as qml

@qml.qjit
@qml.qnode(qml.device("lightning.gpu", wires=2))
def circuit(theta):
    qml.Hadamard(wires=0)
    qml.RX(theta, wires=1)
    qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(wires=1))

circuit(0.7)

######################################################################
# How it works?
# -------------
#
# ``lightning.gpu`` is interfaced with Catalyst by implementing the C++ Catalyst Runtime device API.
# The runtime treats devices as a black-box through a handful of function calls.
# This would minimize the Catalyst <> Device memory footprints and avoid ownership borrowing
# of the on-device state-vector data.
# The runtime initializes an instance of the device and all quantum operations will be offloaded
# to the appropriate kernel and functions in the cuQuantum
# `cuStateVec library <https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html>`__.
#
# Dispite the use of ``lightning.gpu`` in PennyLane, Catalyst doesn't interface with the device
# in Python, avoiding NumPy data-buffer copies to the GPU device.
# To ensure the best overall performance, once all operations are aopplied to the state-vector
# residing in a GPU data buffer, we leverage the built-in GPU-aware C++ measurement processes
# of Lightning-GPU to directly calculate the results on the GPU data.
#
# 
# In PennyLane v0.40.0, we enhanced and expanded the C++ API of ``lightning.gpu`` with built-in features
# designed to improve the overall integration experience with Catalyst. This update ensures that
# ``lightning.gpu`` achieves feature parity with both ``lightning.qubit`` and ``lightning.kokkos``,
# providing native support for arbitrary-controlled operations and differentiation methods.\
# Consequently, we use v0.40.0 throughout this tutorial.
#
# What about performance?
# -----------------------
#
#
# We use the Quantum Phase Estimation (QPE) algorithm from this
# `demo <https://pennylane.ai/qml/demos/tutorial_qpe>`__
# to highlight the performance of ``lightning.gpu`` with QJIT.
#
# Start with the state :math:`|\psi \rangle |0\rangle`, the QPE problems estimates
# the phase of the eigenvalue of a given unitary operator :math:`U` and one of its
# eigenstates :math:`|\psi \rangle.`
# 
# The algorithm can be defined as follows:
#
# 1. Apply Hadamard gates to all estimation qubits to implement a uniform superposition.
#
# 2. Apply a :class:`~.ControlledSequence` operation to creates a sequence of controlled gates on the estimation qubits.
#
# 3. Apply the inverse quantum Fourier transform to the estimation qubits.
#
# 4. Measure the estimation qubits to recover the phase.
#
# .. figure:: ../_static/demonstration_assets/qpe/qpe.png
#     :align: center
#     :width: 80%
#
#     The quantum phase estimation circuit.
#
# This algorithm is especially interesting for Lightning simulators as its performance
# is tied to the efficiency of both regular and arbitrary-controlled gates in the simulator.

# This algorithm is particularly interesting for
# `PennyLane-Lightning <https://docs.pennylane.ai/projects/lightning/en/stable/>`__
# simulators because its performance depends on how efficiently both regular
# and arbitrarily controlled gates are handled in these simulators.
#
# Let's implement this! We'll use the same code from
# `the QPE demo <https://pennylane.ai/qml/demos/tutorial_qpe>`__
# to accept an arbitrary number of target wires and benchmark it.
#

import numpy as np

target_wires = range(0, 4)
estimation_wires = range(4, 6)

num_wires = len(target_wires) + len(estimation_wires)

def U(wires):
    return qml.PhaseShift(2 * np.pi / 5, wires=wires)

@qml.qjit
@qml.qnode(qml.device("lightning.gpu", wires=num_wires))
def circuit_qpe():
    # initialize state as |1...1>
    for i in range(len(target_wires)):
        qml.PauliX(wires=i)

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(U(wires=target_wires), control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)

######################################################################
#
# In this program, we use 4 target wires and 2 estimated wires to
# estimate the phase by measuring the last 2 wires.
# This program initializes a state-vector with 6-wires and apply
# 12 natively supported regular, adjoint and multi-controlled gates on ``lightning.gpu``.
# We know for the fact that the precision of the estimate is determined
# by the size of `estimation_wires` in the algorithm.
#
# Let's only increase the range of ``estimation_wires`` in the code above for a greater percision.
# We run our benchmarks on a NVIDIA Grace-Hopper (GH200) server with Lightning-GPU
# v0.40.0 Linux ARM PyPI wheel.
#
#
# .. figure:: ../_static/demonstration_assets/qpe/tutorial_qjit_lgpu_results.png
#     :align: center
#     :width: 65%
#
# In this example, we get up to 70x overall execution speedup of the QPE workflow
# comparing ``lightning.qubit`` and ``lightning.gpu`` when running
# the just-in-time (QJIT) compiled of ``circuit_qpe``. As the entire program is QJIT compiled,
# we can observe a better performance for executing the compiled program than non-compilation
# pathway in PennyLane.
#
#
# Conclusion
# ----------
#
# This tutorial has demonstrated how to execute a quantum just-in-time compiled PennyLane program
# on NVIDIA GPUs backed by `PennyLane-Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__.
# We further explored the performance of
# `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__ backed
# by ``lightning.gpu`` with compiling and running the QPE problem.
#
# To learn more about Catalyst and how to use it to compile and optimize your quantum programs and workflows,
# check out the `Catalyst Quick Start <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html>`__ guide.
#
# About the authors
# -----------------
#
