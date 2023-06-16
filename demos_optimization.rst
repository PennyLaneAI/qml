 .. role:: html(raw)
   :format: html

Optimization
============

.. meta::
   :property="og:description": Explore various topics and ideas, such as the shots-frugal Rosalin optimizer, the variational quantum thermalizer, or barren plateaus in quantum neural networks.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

Here you will find demonstrations showcasing quantum optimization. Explore
various topics and ideas, such as the shots-frugal Rosalin
optimizer, the variational quantum thermalizer, or barren plateaus
in quantum neural networks.

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">

:html:`<div class="gallery-grid row">`

.. gallery-item::
    :tooltip: Construct and optimize circuits with SU(N) gates
    :figure: demonstrations/here_comes_the_sun/thumbnail_tutorial_here_comes_the_sun.png
    :description: :doc:`demos/tutorial_here_comes_the_sun`
    :tags: quantumcomputing circuitdesign

.. gallery-item::
    :tooltip: Learn how to implement QAOA workflows with PennyLane
    :figure: demonstrations/qaoa_module/qaoa_layer.png
    :description: :doc:`demos/tutorial_qaoa_intro`
    :tags: autograd beginner

.. gallery-item::
    :tooltip: Faster optimization convergence using quantum natural gradient.
    :figure: demonstrations/quantum_natural_gradient/qng_optimization.png
    :description: :doc:`demos/tutorial_quantum_natural_gradient`
    :tags: autograd

.. gallery-item::
    :tooltip: VQE optimization using quantum natural gradient.
    :figure: demonstrations/vqe_qng/vqe_qng_thumbnail.png
    :description: :doc:`demos/tutorial_vqe_qng`
    :tags: chemistry

.. gallery-item::
    :tooltip: Barren plateaus in quantum neural networks.
    :figure: demonstrations/barren_plateaus/surface.png
    :description: :doc:`demos/tutorial_barren_plateaus`
    :tags: autograd

.. gallery-item::
    :tooltip: Understand the difference between local and global cost functions.
    :figure: demonstrations/local_cost_functions/Local_Thumbnail.png
    :description: :doc:`demos/tutorial_local_cost_functions`
    :tags: autograd

.. gallery-item::
    :tooltip: Reduce the number of device executions by using a stochastic approximation optimization.
    :figure: demonstrations/spsa/spsa_mntn.png
    :description: :doc:`demos/tutorial_spsa`
    :tags: gradients qiskit

.. gallery-item::
    :tooltip: Reconstruct and differentiate univariate quantum functions.
    :figure: demonstrations/general_parshift/thumbnail_genpar.png
    :description: :doc:`demos/tutorial_general_parshift`
    :tags: gradients reconstruction

.. gallery-item::
    :tooltip: Doubly stochastic gradient descent.
    :figure: demonstrations/doubly_stochastic/single_shot.png
    :description: :doc:`Doubly stochastic gradient descent <demos/tutorial_doubly_stochastic>`
    :tags: autograd

.. gallery-item::
    :tooltip: Differentiate any qubit gate with the stochastic parameter-shift rule.
    :figure: demonstrations/stochastic_parameter_shift/stochastic_parameter_shift_thumbnail.png
    :description: :doc:`demos/tutorial_stochastic_parameter_shift`
    :tags: autograd

.. gallery-item::
    :tooltip: Rotoselect algorithm.
    :figure: demonstrations/rotoselect/rotoselect_structure.png
    :description: :doc:`demos/tutorial_rotoselect`
    :tags: autograd

.. gallery-item::
    :tooltip: Frugal shot optimization with the Rosalin optimizer.
    :figure: demonstrations/rosalin/rosalin_thumb.png
    :description: :doc:`demos/tutorial_rosalin`
    :tags: autograd

.. gallery-item::
    :tooltip: Solve combinatorial problems without a classical optimizer.
    :figure: demonstrations/falqon/falqon_thumbnail.png
    :description: :doc:`demos/tutorial_falqon`
    :tags: autograd

.. gallery-item::
    :tooltip: Build trigonometric local models of your cost function.
    :figure: demonstrations/quantum_analytic_descent/xkcd.png
    :description: :doc:`demos/tutorial_quantum_analytic_descent`
    :tags: optimization model vqe

.. gallery-item::
    :tooltip: Optimizing measurement protocols with variational methods.
    :figure: demonstrations/quantum_metrology/illustration.png
    :description: :doc:`demos/tutorial_quantum_metrology`
    :tags: cirq metrology autograd

.. gallery-item::
    :tooltip: Learn about the variational quantum thermalizer algorithm, an extension of VQE.
    :figure: demonstrations/vqt/thumbnail_vqt.png
    :description: :doc:`demos/tutorial_vqt`
    :tags: chemistry

.. gallery-item::
    :tooltip: Variational Quantum Linear Solver.
    :figure: demonstrations/vqls/vqls_zoom.png
    :description: :doc:`demos/tutorial_vqls`
    :tags: autograd

.. gallery-item::
    :tooltip: Coherent implementation of a variational quantum linear solver.
    :figure: demonstrations/coherent_vqls/cvqls_zoom.png
    :description: :doc:`demos/tutorial_coherent_vqls`
    :tags: autograd

.. gallery-item::
    :tooltip: Optimize a Quantum Optical Neural Network using NLopt.
    :figure: demonstrations/qonn/qonn_thumbnail.png
    :description: :doc:`demos/qonn`
    :tags: autograd photonics

.. gallery-item::
    :tooltip: Ising model example with PennyLane PyTorch interface.
    :figure: demonstrations/Ising_model/isingspins.png
    :description: :doc:`demos/tutorial_isingmodel_PyTorch`
    :tags: pytorch autograd

.. gallery-item::
    :tooltip: Perform QAOA for MaxCut.
    :figure: demonstrations/qaoa_maxcut/qaoa_maxcut_partition.png
    :description: :doc:`demos/tutorial_qaoa_maxcut`
    :tags: autograd
    
.. gallery-item::
    :tooltip: Quantum natural SPSA optimizer that reduces the number of quantum measurements in the optimization.
    :figure: demonstrations/qnspsa/qnspsa_cover.png
    :description: :doc:`demos/qnspsa`   
    :tags: braket

.. gallery-item::
    :tooltip: Learn how to use zne error mitigation and maintain differentiability.
    :figure: demonstrations/diffable-mitigation/diffable_mitigation_thumb.png
    :description: :doc:`demos/tutorial_diffable-mitigation`
    :tags: mitigation zero noise extrapolation differentiability autograd pytorch tensorflow jax

.. gallery-item::
    :tooltip: Compute gradients of the solution of a variational algorithm using implicit differentiation.
    :figure: demonstrations/implicit_diff/descartes.png
    :description: :doc:`demos/tutorial_implicit_diff_susceptibility`
    :tags: implicit differentiation jax jaxopt ground state energy susceptibility VQA

.. gallery-item::
    :tooltip: Use perturbative gadgets to avoid cost-function-dependent barren plateaus
    :figure: demonstrations/barren_gadgets/thumbnail_tutorial_barren_gadgets.png
    :description: :doc:`demos/tutorial_barren_gadgets`
    :tags: optimization barren plateaus

:html:`</div></div><div style='clear:both'>`


.. toctree::
    :maxdepth: 2
    :hidden:

    demos/tutorial_qaoa_intro
    demos/tutorial_quantum_natural_gradient
    demos/tutorial_vqe_qng
    demos/tutorial_barren_plateaus
    demos/tutorial_local_cost_functions
    demos/tutorial_spsa
    demos/tutorial_general_parshift
    demos/tutorial_doubly_stochastic
    demos/tutorial_stochastic_parameter_shift
    demos/tutorial_rotoselect
    demos/tutorial_rosalin
    demos/tutorial_falqon
    demos/tutorial_quantum_analytic_descent
    demos/tutorial_quantum_metrology
    demos/tutorial_vqt
    demos/tutorial_vqls
    demos/tutorial_coherent_vqls
    demos/qonn
    demos/tutorial_isingmodel_PyTorch
    demos/tutorial_qaoa_maxcut
    demos/qnspsa
    demos/tutorial_diffable-mitigation
    demos/tutorial_implicit_diff_susceptibility
    demos/tutorial_barren_gadgets
    demos/tutorial_here_comes_the_sun

