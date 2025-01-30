r"""
How to run QJIT compiled PennyLane programs on CUDA-backed GPUs
===============================================================


Our CUDA-backed GPU-enabled state-vector simulator, Lightning-GPU, has been
recently integrated to `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__.
This enables quantum just-in-time (QJIT) compiled quantum operations to execute on
`cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`__ compatible GPUs.

Here, we'll show you how to use Lightning-GPU
to run larger quantum circuits more efficiently and boost
the performance of your QJIT-compiled programs.


.. figure:: ../_static/demonstration_assets/qpe/tutorial_qjit_lgpu.png
     :align: center
     :width: 80%
     :target: javascript:void(0);


Set up your environment
-----------------------

To bring the power of `Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__
to `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__, we suggest to follow the
`Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/installation.html>`__
and `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/installation.html>`__
installation guidelines. On an up-to-date Linux X86_64 or ARM64 machines with CUDA-12,
you can install PennyLane, Lightning-GPU and Catalyst from PyPI via

.. code-block:: bash

    pip install pennylane pennylane-lightning-gpu pennylane-catalyst --upgrade

Simply create a PennyLane's ``lightning.gpu`` device, compile your circuit with ``qml.qjit``,
and run it as usual!

.. code-block:: python

    import pennylane as qml


    @qml.qjit
    @qml.qnode(qml.device("lightning.gpu", wires=20))
    def circuit(theta):
        for i in range(20):
            qml.Hadamard(wires=i)
            qml.RX(theta, wires=i)
        return qml.probs()


    circuit(0.7)

How it works?
-------------

``lightning.gpu`` interfaces with Catalyst via the C++ Runtime device API.
The runtime treats devices as a black-box through a handful of function calls.
This helps to minimize the Catalyst-to-device memory footprints and avoid ownership borrowing
of the on-device state-vector data.
The runtime initializes an instance of the device and all quantum operations will be offloaded
to the appropriate kernel and functions in the cuQuantum
`cuStateVec library <https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html>`__.

Dispite the use of ``lightning.gpu`` in PennyLane, Catalyst doesn't interface with the device
in Python, avoiding NumPy data-buffer copies to the GPU device.
To ensure the best overall performance, once all operations are applied to the state-vector
we leverage the built-in GPU-aware C++ measurement processes
of Lightning-GPU to directly calculate the results on the GPU-hosted data.


In PennyLane v0.40.0, we enhanced and expanded the C++ API of ``lightning.gpu`` with built-in features
designed to improve the overall integration experience with Catalyst. This update ensures that
``lightning.gpu`` achieves feature parity with both ``lightning.qubit`` and ``lightning.kokkos``,
providing native support for arbitrary-controlled operations and differentiation methods.

What about performance?
-----------------------

We use the quantum phase estimation (QPE) algorithm from this
`demo <https://pennylane.ai/qml/demos/tutorial_qpe>`__
to highlight the performance of ``lightning.gpu`` with QJIT.

Starting with the state :math:`|\psi \rangle |0\rangle`, the QPE algorithm estimates
the phase of the eigenvalue of a given unitary operator :math:`U` and one of its
eigenstates :math:`|\psi \rangle.`

The algorithm can be defined as follows:

1. Apply Hadamard gates to all estimation qubits to implement a uniform superposition.

2. Apply a :class:`~.ControlledSequence` operation to creates a sequence of controlled gates on the estimation qubits.

3. Apply the inverse quantum Fourier transform to the estimation qubits.

4. Measure the estimation qubits to recover the phase.

.. figure:: ../_static/demonstration_assets/qpe/qpe.png
    :align: center
    :width: 80%

    The quantum phase estimation circuit.

This algorithm is particularly interesting for
`PennyLane-Lightning <https://docs.pennylane.ai/projects/lightning/en/stable/>`__
simulators because its performance depends on how efficiently both regular
and arbitrarily controlled gates are handled in these simulators.

Let's implement this in PennyLane! We'll use the same code from
`the QPE demo <https://pennylane.ai/qml/demos/tutorial_qpe>`__
to accept an arbitrary number of target wires.


.. code-block:: python

    import numpy as np

    target_wires = range(0, 4)
    estimation_wires = range(4, 6)

    num_wires = len(target_wires) + len(estimation_wires)


    def U(wires):
        return qml.PhaseShift(2 * np.pi / 5, wires=wires)


    @qml.qjit(autograph=True)
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

In this program, we only use 4 target wires and 2 estimated wires to
estimate the phase by measuring the last 2 wires.
This program initializes a state-vector with 6-wires and apply
12 natively supported regular, adjoint and multi-controlled gates on ``lightning.gpu``
We know for the fact that the precision of the estimated phase is determined
by the size of ``estimation_wires`` in the algorithm.

To show the power of GPU workhorses, let's increase the range of ``estimation_wires``
in the code above for a better estimated phase percision and greater performance.
We run our benchmarks on a NVIDIA Grace-Hopper (GH200) server with an installation of
Lightning-GPU v0.40.0 Linux ARM64 PyPI wheels.


.. figure:: ../_static/demonstration_assets/qpe/tutorial_qjit_lgpu_results.png
    :align: center
    :width: 80%

In this example, we get up to 70x overall execution speedup of the QPE workflow
comparing ``lightning.qubit`` and ``lightning.gpu`` when running
the quantum just-in-time (QJIT) compiled of ``circuit_qpe``.
Since the entire program, including the for-loops, is QJIT compiled,
we observe improved performance when running the compiled program
compared to the non-compiled regular pathway in PennyLane.

Conclusion
----------

This tutorial has demonstrated how to execute a quantum just-in-time compiled PennyLane hybrid program
on CUDA-backed GPUs by leveraging the performance of `Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__.
We further explored the performance of
``lightning.gpu`` integration with `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__
highlighting the run-time of QPE with and without QJIT compilation.

To learn more about Catalyst and how to use it to compile and optimize your quantum programs and workflows,
check out the `Catalyst Quick Start <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html>`__ guide.

About the authors
-----------------

"""
