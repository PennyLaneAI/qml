
.. role:: html(raw)
   :format: html

Key Concepts
============

.. meta::
   :property="og:description": A growing glossary of key concepts for quantum machine learning.
   :property="og:image": https://pennylane.ai/qml/_static/board.png

.. glossary::

    :doc:`Automatic Differentiation </glossary/automatic_differentiation>`
        Automatically computing derivatives of the steps of computer programs.

    :doc:`Circuit Ansatz </glossary/circuit_ansatz>`
        An ansatz is a basic architecture of a circuit, i.e., a set of gates that act on
        specific subsystems. The architecture defines which algorithms a variational circuit can implement by
        fixing the trainable parameters. A circuit ansatz is analogous to the architecture of a neural network.

    Differentiable quantum programming
        The paradigm of making quantum programs differentiable, and thereby trainable. See also
        :doc:`quantum gradient </glossary/quantum_gradient>`.

    :doc:`(Quantum) Embedding </glossary/quantum_embedding>`
        Representation of classical data as a quantum state.

    (Quantum) Feature Map
        The mathematical map that embeds classical data into a quantum state. Usually executed by a variational
        quantum circuit whose parameters depend on the input data. See also
        :doc:`Quantum Embedding </glossary/quantum_embedding>`.

    :doc:`(Quantum) Gradient </glossary/quantum_gradient>`
        The derivative of a quantum computation with respect to the parameters of a circuit.

    :doc:`Hybrid Computation </glossary/hybrid_computation>`
        A computation that includes classical *and* quantum subroutines, executed on different devices.

    (Quantum) Node
        A quantum computation executed as part of a larger :doc:`hybrid computation </glossary/hybrid_computation>`.

    :doc:`Parameter-shift rules </glossary/parameter_shift>`
        A parameter-shift rule is a recipe for how to estimate gradients of quantum circuits.
        See also :doc:`quantum gradient </glossary/quantum_gradient>`.

    :doc:`Variational circuit </glossary/variational_circuit>`
        Variational circuits are quantum algorithms that depend on tunable parameters, and can therefore
        be optimized.

.. toctree::
    :maxdepth: 2
    :hidden:

    /glossary/automatic_differentiation
    /glossary/circuit_ansatz
    /glossary/quantum_embedding
    /glossary/quantum_gradient
    /glossary/hybrid_computation
    /glossary/parameter_shift
    /glossary/variational_circuit
