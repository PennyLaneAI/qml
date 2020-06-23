.. role:: html(raw)
   :format: html

.. _glossary_quantum_neural_network:

Quantum Neural Network
----------------------

A quantum neural network (QNN) is a machine learning model or algorithm that combines concepts from **quantum computing** and **artifical neural networks**. 

Over the past decades, the term has been used to describe a variety of ideas, ranging from quantum computers *emulating* the exact computations of neural nets, to general *trainable* quantum circuits that bear only little resemblence with the multi-layer perceptron structure.

Quantum versions of feed-forward neural networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/concepts/qnn1.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

`Already in the 1990s, <https://arxiv.org/abs/1408.7005>`_ quantum physicists have tried to come up with "quantum versions" of recurrent and feed-forward neural networks. The models were attempts to translate the modular structure as well as the nonlinear activation functions of neural networks into the language of quantum algorithms. However, one could argue that chains of linear and nonlinear computations are rather "unnatural" for quantum computers [#]_. 

More recent research has tackled this problem, suggesting special measurement schemes or modifications of the neural nets that make them more amenable to quantum computing, but the advantage of these models for machine learning is still not conclusively established.

Quantum versions of Boltzmann machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[Boltzmann machines](https://en.wikipedia.org/wiki/Boltzmann_machine), which are probabilistic graphical models that can be understood as stochastic recurrent neural networks, play an important role in the quantum machine learning literature. For example, `it was suggested <https://mdenil.com/static/papers/2011-mdenil-quantum_deep_learning-nips_workshop.pdf>`_ to use samples from a quantum computer to train classical Boltzmann machines, or to interpret spins as physical units of a `"quantum" Boltzmann machine model <https://arxiv.org/abs/1601.02036>`_.


Variational circuits 
~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/concepts/qnn2.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

Increasingly, the term "quantum neural network" is used to `refer to variational or parametrized quantum circuits <https://arxiv.org/abs/1802.06002>`_. While mathematically rather different from the inner workings of neural networks, the analogy highlights the "modular" nature of quantum gates in a circuit, as well as the wide use of tricks from training neural networks used in the optimization of quantum algorithms. 

.. rubric:: Footnotes

.. [#] This is not necessarily true for photonic quantum computers, which allow for very natural implementations of neural nets (see for example `Killoran et al. (2018) <https://arxiv.org/abs/1806.06871>`_ and `Steinbrecher et al. (2018) <https://arxiv.org/abs/1808.10047>`_).
