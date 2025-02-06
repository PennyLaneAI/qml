r"""
Lightning-GPU with QJIT
=======================

Our CUDA-backed GPU-enabled state-vector simulator,
`Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__,
has been recently integrated to `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__.
This enables quantum just-in-time (QJIT) compiled quantum operations to execute on
`cuQuantum <https://developer.nvidia.com/cuquantum-sdk>`__ compatible GPUs.

Here, we'll show you how to use Lightning-GPU
to run larger quantum circuits more efficiently and boost
the performance of your QJIT-compiled programs.

.. figure:: ../_static/demonstration_assets/qpe/tutorial_qjit_lgpu.png
     :align: center
     :width: 80%
     :target: javascript:void(0);

Set up your environment 🏗️
--------------------------

To bring the power of Lightning-GPU to Catalyst,
you need to install PennyLane, the :func:`~.qjit` compiler framework,
and the ``lightning.gpu`` plugin.
On Linux X86_64 or ARM64 machines with CUDA-12, you can install these
packages from the Python Package Index (PyPI):

.. code-block:: bash

    pip install pennylane pennylane-catalyst pennylane-lightning-gpu

You can also install these pacakges on Google Colab.
The installation instructions can be found `here <https://pennylane.ai/install>`__.

How to use 💡
-------------

Simply create a ``lightning.gpu`` device,
compile your PennyLane program with :func:`~.qjit`,
and run it as usual!

.. code-block:: python

    import pennylane as qml
    import jax.numpy as jnp

    @qml.qjit
    def workflow(x):

        @qml.qnode(qml.device("lightning.gpu", wires=3))
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0,1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliX(wires=1))

        return qml.grad(circuit)(x)

    x = jnp.array([0.1, 0.2, 0.3])

    workflow(x)

.. code-block:: bash

    [-0.01894799  0.9316158  -0.05841749]

Lightning-GPU has feature parity with both ``lightning.qubit`` and ``lightning.kokkos``,
providing native support for many PennyLane's operations and measurement processes with
a fast adjoint-Jacobian differentiation implementation. :func:`~.grad` leverages the device
adjoint-jacobian method by default.

Why to use 🌪️
-------------

Given the small number of wires and gates in the above example,
we shouldn't expect any performance gains from executing
the program on GPUs, as the overhead of device initialization
would likely outweigh any potential benefits.

Then when should we use Lightning-GPU?
The answer to this question depends on different parameters from the
structure of your quantum workflow to the specifications of targeted GPUs.

GPUs are well-known to offer great performance to workloads heavily
dependent to the use of many number of concurrent threads that make
these devices suitable for evaluating linear algebra operations 
in a fraction of the time at scale.

As quantum simulation is most often expressed in the language
of linear algebra to apply gates to the state-vector memory buffer,
extract measurement results, or calculate the circuit differentiation,
we can leverage GPUs to scale the heavy-lifting parts of the simulation.

Let's show this on an example!
We use the quantum phase estimation (QPE) algorithm from this
`demo <https://pennylane.ai/qml/demos/tutorial_qpe>`__
with :func:`~.qjit` on ``lightning.gpu``.

Starting with the state :math:`|\psi \rangle |0\rangle`,
the QPE algorithm estimates the phase of the eigenvalue of
a given unitary operator :math:`U` with one of its
eigenstates :math:`|\psi \rangle.`

This algorithm is particularly interesting for
`PennyLane-Lightning <https://docs.pennylane.ai/projects/lightning/en/stable>`__
simulators because its performance requires an efficient implementation of both regular
and arbitrarily controlled gates.

Here, we adapt the same circuit demonstrated in
`the QPE demo <https://pennylane.ai/qml/demos/tutorial_qpe>`__
to simulate arbitrary number of target wires.

.. code-block:: python

    target_wires = range(0, 4)
    estimation_wires = range(4, 6)
    num_wires = len(target_wires) + len(estimation_wires)

    def U(wires):
        return qml.PhaseShift(2 * jnp.pi / 5, wires=wires)

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

In this example, we only use 4 target wires and
we estimate the phase by measuring 2 estimation wires.
This program initializes a state-vector with 6-wire and applies
a few natively supported regular, adjoint and multi-controlled gates on
``lightning.gpu``.

Updating the range of ``estimation_wires`` in the code above
would yeild to a higher precision of the phase estimation, but
for simulators it means applying many more number of gates on a
exponentially larger state-vector.

We run this script on a NVIDIA Grace-Hopper (GH200) server with
Lightning-GPU's and Catalyst's Linux ARM64 PyPI wheels.

.. figure:: ../_static/demonstration_assets/qpe/tutorial_qjit_lgpu_results.png
    :align: center
    :width: 80%

In this example, we get up to 70x overall execution speedup of the QPE workflow
comparing ``lightning.gpu`` to ``lightning.qubit`` when running
the quantum just-in-time (QJIT) compiled ``circuit_qpe``.
Since the entire program, including the for-loops, is QJIT compiled,
we observe improved performance when running the compiled program
compared to the non-compiled regular pathway in PennyLane.

Why is Catalyst faster?
Dispite the use of ``lightning.gpu`` in PennyLane, Catalyst doesn't
interface with the device in Python, avoiding NumPy data-buffer copies to the GPU device
and Python bindings.
To ensure the best overall performance, once all operations are applied to the state-vector
the built-in GPU-aware C++ measurement processes of Lightning-GPU are used to directly
calculate the results on the GPU-hosted data.

Conclusion
----------

This tutorial demonstrates how to execute a quantum just-in-time compiled PennyLane
hybrid program on CUDA-backed GPUs backed by
`Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__.
We further explored the performance of
``lightning.gpu`` integration with `Catalyst <https://docs.pennylane.ai/projects/catalyst>`__
highlighting the run-time performance boost of the QPE algorithm with and without :func:`~.qjit`.

To learn more about Catalyst and how to use it to compile and optimize your quantum programs
and workflows, check out the
`Catalyst Quick Start <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html>`__ guide.

About the authors
-----------------

"""
