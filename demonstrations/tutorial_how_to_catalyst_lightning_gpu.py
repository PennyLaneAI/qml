r"""
How to use Catalyst with Lightning-GPU
=======================================

We have been sharing details about how to use
`Catalyst <https://docs.pennylane.ai/projects/catalyst>`__,
our quantum just-in-time (QJIT) compiler framework,
to optimize and compile hybrid quantum-classical programs
with different `PennyLane devices <https://pennylane.ai/plugins>`__.
Here, we will demonstrate how to leverage the power of
`Lightning-GPU <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__
to accelerate your quantum simulations.


Lightning-GPU is our high-performance CUDA-backed state-vector simulator for PennyLane.
This device enables fast quantum circuit simulations on NVIDIA GPUs
and has been recently integrated with Catalyst.


.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_Lightning_GPU_Catalyst.png
     :align: center
     :width: 80%
     :target: javascript:void(0)
     :alt: How to use Catalyst with Lightning-GPU


Let's get started with setting up the environment and running a simple example
to highlight Catalyst with Lightning-GPU features.

Setting up your environment
----------------------------

Using Lightning-GPU is particularly beneficial when dealing with large quantum circuits or
when running simulations that require significant computational resources.
By offloading the heavy linear algebra operations to GPUs,
we can achieve substantial performance improvements.

Lightning-GPU can be used on systems equipped with NVIDIA GPUs of generation SM 7.0 (Volta) and greater,
specifically, it is optimized for Linux X86-64 or ARM64 machines with CUDA-12 installed.

To use Catalyst with Lightning-GPU, you need to install PennyLane,
the :func:`~pennylane.qjit` compiler framework, and the Lightning-GPU plugin (``lightning.gpu``).
These packages can be installed from the Python Package Index (PyPI) using the following command:

.. code-block:: bash

    pip install pennylane pennylane-catalyst pennylane-lightning-gpu

If you are using a different hardware configuration or operating system,
you may need to install these packages from source.
Detailed instructions for source installation can be found
in the respective `documentation <https://pennylane.ai/install>`__ of each package.


A simple circuit
-----------------

Here, we use a simple PennyLane circuit to demonstrate
how to compile and run a quantum program on Lightning-GPU with Catalyst.

First, we create a ``lightning.gpu`` device with 28 qubits using the :func:`~pennylane.device` function.
We then define a quantum circuit that applies layers of :func:`~.RZ` and :func:`~.RY`
rotations to each wire and returns the expectation value as the measurement result.

To compile the circuit with Catalyst, we use the :func:`~pennylane.qjit` decorator.
The ``autograph=True`` compilation option compiles the circuit with for-loops.
This allows Catalyst to efficiently
:doc:`capture control-flow operations <catalyst:dev/autograph>`,
reduce the compilation time, and generate a more efficient program.

.. code-block:: python

    import pennylane as qml
    import jax.numpy as jnp

    import jax

    # Set number of wires
    num_wires = 28

    # Set a random seed
    key = jax.random.PRNGKey(0)

    dev = qml.device("lightning.gpu", wires=num_wires)

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
`the rest of Lightning state-vector simulators <https://docs.pennylane.ai/projects/lightning/en/stable>`__,
providing native support for many PennyLane's operations and measurement processes with
a fast adjoint-Jacobian differentiation implementation.

In the next section, we demonstrate how to compute the gradient of your quantum programs
using the native adjoint-Jacobian method on this device.


Compute gradients
------------------

We use :func:`~.grad` to compute the gradient of the ``circuit`` with respect to
a set of trainable arguments.
This method is particularly useful for variational quantum algorithms
where the gradient of the cost function with respect to the circuit parameters is required.
You can check out `this demo <https://pennylane.ai/qml/demos/tutorial_how_to_quantum_just_in_time_compile_vqe_catalyst>`__
for more information on compiling and optimizing Variational Quantum Eigensolver (VQE) with Catalyst.

.. code-block:: python

    @qml.qjit(autograph=True)
    def workflow(params):
        g = qml.grad(circuit)
        return g(params)

    >>> workflow(weights)

.. code-block:: bash

    [ 8.8817842e-16 -1.0920765e-01  6.6613381e-16  6.6613381e-16
      2.2204460e-16  2.2204460e-16  2.2204460e-16  5.5511151e-16
      1.1102230e-16  6.6613381e-16 -4.4408921e-16 -2.2204460e-16
      2.2204460e-16  0.0000000e+00 -1.1102230e-16  5.5511151e-16
      3.3306691e-16 -2.2204460e-16 -1.1102230e-16 -2.2204460e-16
      1.1102230e-16  0.0000000e+00 -3.3306691e-16  4.4408921e-16
      -3.3306691e-16  1.1102230e-16 -1.1102230e-16  0.0000000e+00
      -1.1102230e-16  2.2204460e-16 -1.1102230e-16 -2.2204460e-16
      5.5511151e-16  0.0000000e+00 -3.3306691e-16  0.0000000e+00
      2.2204460e-16  0.0000000e+00 -1.1102230e-16  0.0000000e+00
      0.0000000e+00 -4.4408921e-16  0.0000000e+00 -1.1102230e-16
      3.3306691e-16 -1.1102230e-16  1.1102230e-16  1.1102230e-16
      2.2204460e-16  3.3306691e-16  2.2204460e-16  1.1102230e-16
      -1.1102230e-16  1.1102230e-16  1.1102230e-16  3.3306691e-16
      0.0000000e+00 -1.1102230e-16 -6.2915415e-01  0.0000000e+00]

Note that in the above example, we didn't use ``method="adjoint"``.
The adjoint-Jacobian is the default gradient method when you compile
and run your program on Lightning-GPU.
You can learn more about the different gradient support by checking the
`Catalyst Quantum Gradients <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html#calculating-quantum-gradients>`__
documentation.

Now that we have seen how to use the device gradient method,
we can move on to further optimizing the circuit parameters using Catalyst.
We follow the same steps as in
`the VQE demo <https://pennylane.ai/qml/demos/tutorial_how_to_quantum_just_in_time_compile_vqe_catalyst>`__,
to QJIT-compile the entire optimization workflow.

.. code-block:: python

    import catalyst
    import optax

    opt = optax.sgd(learning_rate=0.4)

    def update_step(i, params, opt_state):
        """Perform a single gradient update step"""
        energy, grads = catalyst.value_and_grad(circuit)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        catalyst.debug.print("Step = {i}", i=i)
        return (params, opt_state)

    @qml.qjit
    def optimization(params):
        opt_state = opt.init(params)
        (params, opt_state) = qml.for_loop(0, 10, 1)(update_step)(params, opt_state)
        return params

    >>> final_params = optimization(weights)
    >>> print(f"Final angle parameters: {final_params}")

.. code-block:: bash

    Step = 0
    Step = 1
    Step = 2
    Step = 3
    Step = 4
    Step = 5
    Step = 6
    Step = 7
    Step = 8
    Step = 9
    Final angle parameters: [1.24479175e-01 2.22040820e+00 2.36939192e-01 9.45755959e-01
      1.59472704e-01 8.45545650e-01 1.32652521e-02 3.71306539e-01
      3.42241764e-01 6.25356674e-01 7.08195210e-01 1.62913680e-01
      2.94586539e-01 1.94847584e-03 5.49157500e-01 8.47560406e-01
      9.25927997e-01 5.43066740e-01 4.00609255e-01 8.86821628e-01
      9.50046659e-01 5.18365145e-01 6.38097048e-01 6.67127371e-02
      6.42662406e-01 3.17808628e-01 2.13315248e-01 2.62881398e-01
      2.57341623e-01 7.23513365e-01 6.26209974e-02 1.64659262e-01
      2.68713236e-02 1.52902603e-01 2.55607367e-01 3.54435444e-02
      3.17132115e-01 3.33583951e-01 9.48547006e-01 9.31932330e-01
      7.38109469e-01 2.07342863e-01 1.35567427e-01 5.76237440e-01
      5.13184071e-02 8.20117712e-01 5.33855081e-01 8.53034496e-01
      9.65461254e-01 9.17515278e-01 3.34429860e-01 9.56996560e-01
      9.54037666e-01 9.20464039e-01 5.59616089e-01 4.27830935e-01
      9.16242123e-01 4.42039609e-01 3.07716966e+00 4.64060426e-01]

We used `Optax <https://github.com/google-deepmind/optax>`__,
a library designed for optimization using JAX,
as well as the QJIT-compatible :func:`~pennylane.for_loop` function
to compile the entire optimization loop,
leading to further performance improvements.

Précis
-------

In this tutorial, we have shown how to use Catalyst with Lightning-GPU
to accelerate quantum simulations on NVIDIA GPUs.
We have compiled a simple quantum circuit with Catalyst,
computed the gradient of the circuit using the adjoint-Jacobian method,
and optimized the circuit parameters using Catalyst and Optax.

To learn more about Catalyst and all the integrated devices,
check out the :doc:`Catalyst Quick Start <catalyst:dev/quick_start>` guide.

About the authors
-----------------

"""
