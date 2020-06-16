.. role:: html(raw)
   :format: html

.. _glossary_quantum_neural_network:

Quantum Neural Network
----------------------

A quantum neural network (QNN) is a machine learning model that utilizes concepts from both quantum computing and artifical neural networks. Over the last three decades, the term has been used to describe a variety of models and approaches, ranging from implementations of neural networks with quantum technology, to general "trainable" quantum circuits which have only little mathematical resemblance with neural networks.

.. figure:: ../_static/concepts/vc_general.svg
    :align: center
    :width: 40%
    :target: javascript:void(0);


Quantum implementations of neural networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Already in the 1990s, quantum physicists have tried to come up with "quantum" versions of recurrent and feed-forward neural networks REF. The models were essentially attempts to recover the modular structure of the neural nets, as well as the basic functionality of the perceptron building block that applies a nonlinear function to the signals from incoming neurons. 

The challenge for these models lays in the effective implementation of the irreversible nonlinearity, which is in some sense very "unnatural" for quantum computers. More recently, a series of proposals have suggested solutions based on non-trivial measurement strategies or the exploitation of nonlinear physical processes REF. Another idea uses photonic quantum computers, which can implement the linear and modular part of neural networks generically, and use nonlinear optics or ancilla systems for the nonlinear transformations REF.

Potential quantum advantages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is not fully clear yet how quantum mechanics can actually improve neural networks. Several papers suggest loading data in superposition or using interference effect within the neural network, but a concrete example of a boost in runtime, data volume or generalisation power is yet to be found.


Variational circuits as QNNs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Increasingly, the term quantum neural network is used to refer to variational or parametrized quantum circuits. The analogy refers to the trainability of the circuits, as well as the "modular" nature of quantum gates in a circuit. However, beyond these points, quantum circuits give rise to rather different machine learning models as they do not reproduce the mathematical structure of multi-layer perceptrons. 









.. seealso:: In PennyLane, ....

.. rubric:: Footnotes

.. [#] Bla
