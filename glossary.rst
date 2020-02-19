
.. role:: html(raw)
   :format: html

Glossary
=========

Look up the background details on how :ref:`variational quantum circuits <glossary_variational_circuit>` are trained using
`parameter shift rules <glossary_parameter_shift_rule>.

.. glossary::

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

    :ref:`(Quantum) Node <glossary_quantum_node>`
        A quantum computation executed as part of a larger hybrid computation.

    :ref:`Variational circuit <glossary_variational_circuit>`
        Variational circuits are quantum algorithms that depend on tunable parameters, and can therefore
        be optimized.
