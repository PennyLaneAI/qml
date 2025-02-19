r"""
How to use Catalyst with Lightning-GPU
=======================================

`Catalyst <https://docs.pennylane.ai/projects/catalyst>`__
is a quantum just-in-time (QJIT) compiler framework that optimizes
and compiles hybrid (quantum and classical) PennyLane's programs.
`Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__
is a CUDA-backed state-vector simulator that enables fast quantum circuit simulations on NVIDIA GPUs.

Using Lightning-GPU is particularly beneficial when dealing with large quantum circuits or
when running simulations that require significant computational resources.
By offloading the heavy linear algebra operations to GPUs,
we can achieve substantial performance improvements.

Here, we will demonstrate how to leverage the power of Catalyst
and Lightning-GPU to accelerate quantum simulations.

.. figure:: ../_static/demo_thumbnails/large_demo_thumbnails/thumbnail_large_Lightning_GPU_Catalyst.png
     :align: center
     :width: 80%
     :target: javascript:void(0)
     :alt: How to use Catalyst with Lightning-GPU


Let's get started with setting up the environment and running a simple example
to highlight Catalyst with Lightning-GPU features.

Setting up your environment
----------------------------

Lightning-GPU can be used on systems equipped with NVIDIA GPUs of generation SM 7.0 (Volta) and greater,
specifically, it is optimized for Linux X86-64 or ARM64 machines with CUDA-12 installed.

To bring the power of Lightning-GPU to Catalyst, you need to install PennyLane,
the :func:`~.qjit` compiler framework, and the ``lightning.gpu`` device.
These packages can be installed from the Python Package Index (PyPI) using the following command:

.. code-block:: bash

    pip install pennylane pennylane-catalyst pennylane-lightning-gpu

If you are using a different hardware configuration or operating system,
you may need to install these packages from source.
Detailed instructions for source installation can be found
in the respective documentation of each package.

A simple circuit with Lightning-GPU and QJIT
---------------------------------------------

Here we use a simple example to demonstrate the support.

First, we create a ``lightning.gpu`` device with 20 wires.
We then define a quantum circuit that applies layers of :func:`RZ` and :func:`RY`
rotations to each wire, and returns the expectation value as the measurement result.

To compile the circuit with Catalyst, we use the :func:`~.qjit` decorator.
The `autograph=True <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/autograph.html>`__
compilation option compiles the circuit with for-loops,
this allows Catalyst to capture control-flow operations without unrolling,
reduces the compilation time and generates a more efficient program.

.. code-block:: python

    import pennylane as qml
    import jax.numpy as jnp

    import jax

    # Set number of wires
    num_wires = 20

    # Set a random seed
    key = jax.random.PRNGKey(0)

    dev = qml.device("lightning.gpu", wires=20)

    @qml.qjit(autograph=True)
    @qml.qnode(dev)
    def circuit(params):

        # Apply layers of RZ and RY rotations
        for i in range(num_wires):
            qml.RZ(params[3*i], wires=[i])
            qml.RY(params[3*i+1], wires=[i])
            qml.RZ(params[3*i+2], wires=[i])

        return qml.expval(qml.PauliZ(0) + qml.PauliZ(num_wires-1))

    # Initialize the weights
    weights = jax.random.uniform(key, shape=(3 * num_wires,), dtype=jnp.float32)

    >>> circuit(weights)

.. code-block:: bash

    1.7712995142661776

Lightning-GPU has feature parity with
`other Lightning state-vector simulators <https://docs.pennylane.ai/projects/lightning/en/stable>`__,
providing native support for many PennyLane's operations and measurement processes with
a fast adjoint-Jacobian differentiation implementation.

In the next section, we demonstrate how to compute the gradient of your quantum programs
using the native adjoint-Jacobian method on this device.



Computing gradients
--------------------

We use :func:`~.grad` to compute the gradient of the ``circuit`` with respect to ``weights``.
This method is particularly useful for variational quantum algorithms
where the gradient of the cost function with respect to the circuit parameters is required.
You can check on `this demo <https://pennylane.ai/qml/demos/tutorial_how_to_quantum_just_in_time_compile_vqe_catalyst>`__
for more information on how to compile VQE with Catalyst.


.. code-block:: python

    @qml.qjit(autograph=True)
    def workflow(params):
        g = qml.grad(circuit)
        return g(params)

    >>> workflow(weights)

... code-block:: bash

    [ 8.8817842e-16  ...  -6.2915415e-01  0.0000000e+00]

Note that in the above example, we didn't use ``method="adjoint"`` as the adjoint-jacobian
differentaion method is the default gradiant method in Lightning-GPU.
You can learn more about the different gradient methods by checking the
`Catalyst Quantum Gradients <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html#calculating-quantum-gradients>`__
documentation.

If you haven't used Catalyst, we hope this demonstration has encouraged you to check out the
`Catalyst Quick Start <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html>`__ guide
and use it with our fast GPU-enabled state-vector simulator to compile, optimize and execute your hybrid programs.

About the authors
-----------------

"""
