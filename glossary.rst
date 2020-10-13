
.. role:: html(raw)
   :format: html

Key concepts
============

.. meta::
   :property="og:description": A growing glossary of key concepts for quantum machine learning.
   :property="og:image": https://pennylane.ai/qml/_static/board.png


.. raw:: html

    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity" :
        [

            {
                "@type": "Question", 
                "name" : "What is quantum machine learning?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A research area that explores ideas at the intersection of machine learning and quantum computing."
                }
            },   

            {
                "@type": "Question", 
                "name" : "What is a variational circuit?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "Variational circuits are quantum algorithms that depend on tunable parameters, 
                    and can therefore be optimized."
                }
            },           

            {
                "@type": "Question", 
                "name" : "What is a quantum neural network?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A term with many different meanings, usually referring to a generalization of 
                    artificial neural networks to quantum information processing. Also increasingly used to refer to 
                    variational circuits in the context of quantum machine learning."
                }
            },      

            {
                "@type": "Question", 
                "name" : "What is the variational quantum eigensolver (VQE)?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A variational algorithm used for finding the ground state energy of a quantum system. The 
                    VQE is a hybrid algorithm that involves incorporating measurement results obtained from a quantum 
                    computer running a series of variational circuits into a classical optimization routine in order to 
                    find a set of optimal variational parameters."
                }
            },         
            
            {
                "@type": "Question", 
                "name" : "What is the Quantum Approximate Optimization Algorithm (QAOA)?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A hybrid variational algorithm that is used to find approximate solutions for combinatorial 
                    optimization problems. It's characterized by a circuit ansatz featuring two alternating, parameterized components."
                }
            },

            {
                "@type": "Question", 
                "name" : "What is automatic differentiation?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "The process of automatically computing derivatives of the steps of computer programs."
                }
            },

            {
                "@type": "Question", 
                "name" : "What are barren plateaus?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "Areas in the cost landscape where the gradient of a parameterized circuit disappears. The mortal enemy of many a variational algorithm, 
                    the variance of the gradient at these points is also close to zero in all directions. "
                }
            },

             {
                "@type": "Question", 
                "name" : "What is a circuit ansatz?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A basic architecture of a circuit, i.e., a set of gates that act on
                        specific subsystems. The architecture defines which algorithms a variational circuit can implement by
                        fixing the trainable parameters. A circuit ansatz is analogous to the architecture of a neural network."
                }
            },

            {
                "@type": "Question", 
                "name" : "What is hybrid computation?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "These are computations that include classical *and* quantum subroutines, 
                    executed on different devices."
                }
            },

            {
                "@type": "Question", 
                "name" : "What is the parameter shift rule?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A recipe for how to estimate gradients of quantum circuits."
                }
            },


            {
                "@type": "Question", 
                "name" : "What is a quantum Boltzmann machine?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "The quantum analog of a classical Boltzmann machine, in which nodes are 
                    replaced by spins or qubits. It is an energy-based quantum machine learning model."
                }
            },

            {
                "@type": "Question", 
                "name" : "What is a quantum convolutional neural network?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "The quantum analog of a convolutional neural network. They are affectionately referred 
                    to as *quanvolutional* neural networks."
                }
            },

             {
                "@type": "Question", 
                "name" : "What is quantum differentiable programming?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "The paradigm of making quantum algorithms differentiable, and thereby trainable."
                }
            },

            {
                "@type": "Question", 
                "name" : "What is a quantum embedding?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A representation of classical data as a quantum state."
                }
            },

            {
                "@type": "Question", 
                "name" : "What is a quantum feature map?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "The mathematical map that embeds classical data into a quantum state. Usually executed by a variational 
                    quantum circuit whose parameters depend on the input data."
                }
            },

            {
                "@type": "Question", 
                "name" : "What are quantum generative adversarial networks?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "The quantum analog of a Generative Adversarial Networks (GANs)."
                }
            },

            {
                "@type": "Question", 
                "name" : "What is a quantum gradient?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "The derivative of a quantum computation with respect to the parameters of a circuit."
                }
            },

            {
                "@type": "Question", 
                "name" : "What is a quantum node?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A quantum computation executed as part of a larger hybrid computation."

                }
            },                                         

            {
                "@type": "Question", 
                "name" : "What is the variational quantum classifier (VQC)?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A supervised learning algorithm in which variational circuits are trained to perform 
                    classification tasks."
                }
            },         

            {
                "@type": "Question", 
                "name" : "What is the variational quantum linear solver?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "An algorithm for solving systems of linear equations on quantum computers. Based on short 
                    variational circuits, it is amenable to running on near-term quantum hardware."
                }
            },         

            {
                "@type": "Question", 
                "name" : "What is the variational quantum thermalizer (VQT)?",
                "acceptedAnswer" :
                {
                    "@type": "Answer",
                    "text": "A generalization of the variational quantum eigensolver to systems with non-zero 
                    temperatures. It uses quantum Hamiltonian-based models to to generate thermal states of Hamiltonians 
                    at a given temperature."
                }
            }  
        ]    
    }
    </script>

.. glossary::

    :doc:`Automatic Differentiation </glossary/automatic_differentiation>`
        Automatically computing derivatives of the steps of computer programs.

    :doc:`Barren Plateaus </demos/tutorial_local_cost_functions>`
        Areas in the cost landscape where the gradient of a parameterized circuit disappears. The mortal enemy of many a variational algorithm, 
        the variance of the gradient at these points is also close to zero in all directions. 

    :doc:`Circuit Ansatz </glossary/circuit_ansatz>`
        An ansatz is a basic architecture of a circuit, i.e., a set of gates that act on
        specific subsystems. The architecture defines which algorithms a variational circuit can implement by
        fixing the trainable parameters. A circuit ansatz is analogous to the architecture of a neural network.

    :doc:`Hybrid Computation </glossary/hybrid_computation>`
        A computation that includes classical *and* quantum subroutines, executed on different devices.

    :doc:`Parameter-shift Rule </glossary/parameter_shift>`
        The parameter-shift rule is a recipe for how to estimate gradients of quantum circuits.
        See also :doc:`quantum gradient </glossary/quantum_gradient>`.

    :doc:`Quantum Approximate Optimization Algorithm (QAOA) </demos/tutorial_qaoa_maxcut>`
        A hybrid variational algorithm that is used to find approximate solutions for combinatorial optimization problems. Characterized by a circuit ansatz featuring two alternating, parameterized components. 

    Quantum Boltzmann Machine
        Quantum analog of a classical `Boltzmann machine <https://en.wikipedia.org/wiki/Boltzmann_machine>`_, in which nodes are 
        replaced by spins or qubits. An energy-based quantum machine learning model.

    :doc:`Quantum Convolutional Neural Network </demos/tutorial_quanvolution>`
        Quantum analog of a `convolutional neural network <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_. Affectionately referred to as 
        *quanvolutional* neural networks.

    Quantum Differentiable Programming
        The paradigm of making quantum algorithms differentiable, and thereby trainable. See also
        :doc:`quantum gradient </glossary/quantum_gradient>`.

    :doc:`Quantum Embedding </glossary/quantum_embedding>`
        Representation of classical data as a quantum state.

    Quantum Feature Map
        The mathematical map that embeds classical data into a quantum state. Usually executed by a variational
        quantum circuit whose parameters depend on the input data. See also
        :doc:`Quantum Embedding </glossary/quantum_embedding>`.

    :doc:`Quantum Generative Adversarial Network </demos/tutorial_QGAN>`
        Quantum analog of `Generative Adversarial Networks <https://en.wikipedia.org/wiki/Generative_adversarial_network>`_ (GANs).

    :doc:`Quantum Gradient </glossary/quantum_gradient>`
        The derivative of a quantum computation with respect to the parameters of a circuit.

    :doc:`Quantum Machine Learning <whatisqml>`
        A research area that explores ideas at the intersection of machine learning and quantum computing.

    :doc:`Quantum Neural Network </glossary/quantum_neural_network>`
        A term with many different meanings, usually referring to a generalization of artificial neural 
        networks to quantum information processing. Also increasingly used to refer to :doc:`variational circuits </glossary/variational_circuit>` in the context of quantum machine learning.
        
    Quantum Node
        A quantum computation executed as part of a larger :doc:`hybrid computation </glossary/hybrid_computation>`.

    :doc:`Variational Circuit </glossary/variational_circuit>`
        Variational circuits are quantum algorithms that depend on tunable parameters, and can therefore be optimized.

    :doc:`Variational Quantum Classifier (VQC) </demos/tutorial_variational_classifier>`
        A supervised learning algorithm in which variational circuits (:abbr:`QNNs (Quantum Neural Networks)`) are trained to perform classification tasks.

    :doc:`Variational Quantum Eigensolver (VQE) </demos/tutorial_vqe>`
        A variational algorithm used for finding the ground-state energy of a quantum system. The VQE is a hybrid algorithm that involves incorporating measurement results obtained from a quantum computer running a series of variational circuits into a classical optimization routine in order to find a set of optimal variational parameters. 

    :doc:`Variational Quantum Linear Solver (VQLS) </demos/tutorial_vqls>`
        An algorithm for solving systems of linear equations on quantum computers. Based on short variational circuits, it is amenable to running on near-term quantum hardware. 

    :doc:`Variational Quantum Thermalizer (VQT) </demos/tutorial_vqt>`
        A generalization of the :abbr:`VQE (Variational Quantum Eigensolver)` to systems with non-zero temperatures. Uses :abbr:`QHBMs (Quantum Hamiltonian-Based Models)` to generate thermal states of Hamiltonians at a given temperature.


.. toctree::
    :maxdepth: 2
    :hidden:

    /glossary/automatic_differentiation
    /glossary/circuit_ansatz
    /glossary/hybrid_computation
    /glossary/parameter_shift
    /glossary/quantum_embedding
    /glossary/quantum_gradient
    /glossary/quantum_neural_network
    /glossary/variational_circuit
