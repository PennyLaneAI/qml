
.. role:: html(raw)
   :format: html

Glossary
=========

.. glossary::

    :ref:`Automatic Differentiation <glossary_automatic_differentiation>`
        Automatically computing derivatives of the outputs of computer programs.

    :ref:`Circuit Ansatz <glossary_circuit_ansatz>`
        An ansatz is commonly known as a basic architecture of a circuit, i.e., a set of gates that act on
        specific subsystems. The architecture defines which algorithms a variational circuit can implement by
        fixing the trainable parameters. A circuit ansatz is similar to the architecture of a neural network.

    Differentiable quantum programming
        The paradigm of making quantum programs differentiable, and thereby trainable. See also
        :ref:`quantum gradient <glossary_quantum_gradient>`.

    :ref:`(Quantum) Embedding <glossary_quantum_embedding>`
        Representation of classical data as a quantum state.

    (Quantum) Feature Map
        The mathematical map that embedds classical data into a quantum state. Usually executed by a variational
        quantum circuit whose parameters depend on the input data. See also
        :ref:`Quantum Embedding <glossary_quantum_embedding>`

    :ref:`(Quantum) Gradient <glossary_quantum_gradient>`
        The derivative of a quantum computation with respect to the parameters of a circuit.

    :ref:`Hybrid Computation <glossary_hybrid_computation>`
        A computation that includes classical *and* quantum subroutines, executed on different devices.

    (Quantum) Node
        A quantum computation executed as part of a larger :ref:`hybrid computation <glossary_hybrid_computation>`.

    :ref:`Variational circuit <glossary_variational_circuit>`
        Variational circuits are quantum algorithms that depend on tunable parameters, and can therefore
        be optimized.
