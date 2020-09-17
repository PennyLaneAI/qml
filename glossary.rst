
.. role:: html(raw)
   :format: html

QML glossary
============

.. meta::
   :property="og:description": A growing glossary of key concepts for quantum machine learning.
   :property="og:image": https://pennylane.ai/qml/_static/board.png

.. glossary::

    :doc:`Automatic Differentiation </glossary/automatic_differentiation>`
        Automatically computing derivatives of the steps of computer programs.

    :doc:`Barren Plateaus </glossary/barren_plateaus>`
        Points at which the gradient of a parameterized circuit disappear. The mortal enemy of many a variational algorithm. 

    :doc:`Circuit Ansatz </glossary/circuit_ansatz>`
        An ansatz is a basic architecture of a circuit, i.e., a set of gates that act on
        specific subsystems. The architecture defines which algorithms a variational circuit can implement by
        fixing the trainable parameters. A circuit ansatz is analogous to the architecture of a neural network.

    :doc:`Dequantization </glossary/dequantization>`
        The process of creating a classical algorithm that matches or improves on the complexity of the best-known quantum machine learning algorithm for a given task, thus negating its potential for quantum advantage. Such classical algorithms are often "quantum-inspired".

    :doc:`HHL Algorithm </glossary/hhl_algorithm>`
        A quantum algorithm for solving systems of linear equations. Used as a subroutine in numerous quantum machine learning algorithms.

    :doc:`Hybrid Computation </glossary/hybrid_computation>`
        A computation that includes classical *and* quantum subroutines, executed on different devices.

    :doc:`Parameter-shift Rule </glossary/parameter_shift>`
        The parameter-shift rule is a recipe for how to estimate gradients of quantum circuits.
        See also :doc:`quantum gradient </glossary/quantum_gradient>`.

    :doc:`Quantum Approximate Optimization Algorithm (QAOA) </glossary/qaoa>`
        A hybrid variational algorithm that is used to find approximate solutions for combinatorial optimization problems. Characterized by a circuit ansatz featuring two alternating, parameterized components. 

    :doc:`Quantum Boltzmann Machine </glossary/quantum_boltzmann_machine>`
        Quantum analog of a classical Boltzmann machine, in which nodes are replaced by spins or qubits. An energy-based machine learning model.

    :doc:`Quantum Circuit Learning </glossary/quantum_circuit_learning>`
        A variational framework that can be used to teach quantum neural networks to perform both linear and nonlinear function approximation and classification tasks.

    :doc:`Quantum Convolutional Neural Network </glossary/quanvolutional_neural_network>`
        Quantum analog of a convolutional neural network.

    :doc:`Quantum Deep Learning </glossary/quantum_deep_learning>`
        Refers to the paradigm of using a quantum computer to perform machine learning tasks that, like classical deep learning, may require multiple layers of abstraction to learn.

    Quantum Differentiable Programming
        The paradigm of making quantum algorithms differentiable, and thereby trainable. See also
        :doc:`quantum gradient </glossary/quantum_gradient>`.

    :doc:`Quantum Embedding </glossary/quantum_embedding>`
        Representation of classical data as a quantum state.

    Quantum Feature Map
        The mathematical map that embeds classical data into a quantum state. Usually executed by a variational
        quantum circuit whose parameters depend on the input data. See also
        :doc:`Quantum Embedding </glossary/quantum_embedding>`.

    :doc:`Quantum Gradient </glossary/quantum_gradient>`
        The derivative of a quantum computation with respect to the parameters of a circuit.

    :doc:`Quantum Graph Neural Network </glossary/quantum_graph_neural_network>`
        A type of quantum neural network with an ansatz characterized by repeatedly evolving a chosen sequence of parameterizable Hamiltonians. 

    :doc:`Quantum Hamiltonian-Based Models </glossary/quantum_hamiltonian_based_models>`
        A type of generative, energy-based quantum neural network. 

    :doc:`Quantum Machine Learning <whatisqml>`
        A research area that explores ideas at the intersection of machine learning and quantum computing.

    :doc:`Quantum Neural Network </glossary/quantum_neural_network>`
        A term with many different meanings, usually refering to a generalization of artificial neural 
        networks to quantum information processing. Also increasingly used to refer to :doc:`variational circuits </glossary/variational_circuit>` in the context of quantum machine learning.
        
    Quantum Node
        A quantum computation executed as part of a larger :doc:`hybrid computation </glossary/hybrid_computation>`.

    :doc:`Quantum Perceptron </glossary/quantum_perceptron>`
        Quantum system or operation capable of performing a task analogous to that of a classical perceptron.

    :doc:`Quantum RAM </glossary/quantum_ram>`
        A device or unitary operation capable of loading classical data to a quantum computer in superposition. Required as a subroutine by a number of quantum algorithms that need classical data access.   

    :doc:`Quantum Variational Autoencoder (QVAE) </glossary/qvae>`
        Quantum analog of a variational autoencoder. QVAEs are a generative quantum machine learning model that learn a latent representation of a data set, and may use quantum hardware to subsequently generate new random samples from it.

    :doc:`Unitary Parameterization </glossary/unitary_parameterization>`
        A recipe for expressing unitary matrices in terms of real-valued parameters. 

    :doc:`Variational Circuit </glossary/variational_circuit>`
        Variational circuits are quantum algorithms that depend on tunable parameters, and can therefore be optimized.

    :doc:`Variational Quantum Classifier (VQC) </glossary/variational_quantum_classifier>`
        A supervised learning algorithm in which variational circuits (:abbr:`QNNs (Quantum Neural Networks)`) are trained to perform classification tasks.

    :doc:`Variational Quantum Eigensolver (VQE) </glossary/variational_quantum_eigensolver>`
        A variational algorithm used for finding the ground state energy of a quantum system. The VQE is a hybrid algorithm that involves incorporating measurement results obtained from a quantum computer running a series of variational circuits into a classical optimization routine in order to find a set of optimal variational parameters. 

    :doc:`Variational Quantum Linear Solver (VQLS) </glossary/variational_quantum_linear_solver>`
        An algorithm for solving systems of linear equations on quantum computers. Based on short variational circuits, it is amenable to running on near-term quantum hardware. 

    :doc:`Variational Quantum Thermalizer (VQT) </glossary/variational_quantum_thermalizer>`
        A generalization of the :abbr:`VQE (Variational Quantum Eigensolver)` to systems with non-zero temperatures. Uses :abbr:`QHBMs (Quantum Hamiltonian-Based Models)` to generate thermal states of Hamiltonians at a given temperature.


.. toctree::
    :maxdepth: 2
    :hidden:

    /glossary/automatic_differentiation
    /glossary/barren_plateaus
    /glossary/circuit_ansatz
    /glossary/hhl_algorithm
    /glossary/hybrid_computation
    /glossary/parameter_shift
    /glossary/qaoa
    /glossary/quantum_boltzmann_machine
    /glossary/quantum_circuit_learning
    /glossary/quantum_deep_learning
    /glossary/quantum_embedding
    /glossary/quantum_gan
    /glossary/quantum_gradient
    /glossary/quantum_graph_neural_network
    /glossary/quantum_hamiltonian_based_models
    /glossary/quantum_kernel
    /glossary/quantum_natural_gradient
    /glossary/quantum_neural_network
    /glossary/quantum_perceptron
    /glossary/quantum_ram
    /glossary/quantum_svm
    /glossary/quanvolutional_neural_network
    /glossary/qvae
    /glossary/unitary_parameterization
    /glossary/variational_circuit
    /glossary/variational_quantum_classifier
    /glossary/variational_quantum_eigensolver
    /glossary/variational_quantum_linear_solver
    /glossary/variational_quantum_thermalizer
