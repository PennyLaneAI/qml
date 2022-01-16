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

.. customgalleryitem::
    :tooltip: Learn how to implement QAOA workflows with PennyLane
    :figure: demonstrations/qaoa_module/qaoa_layer.png
    :description: :doc:`demos/tutorial_qaoa_intro`
    :tags: autograd beginner

.. customgalleryitem::
    :tooltip: Faster optimization convergence using quantum natural gradient.
    :figure: demonstrations/quantum_natural_gradient/qng_optimization.png
    :description: :doc:`demos/tutorial_quantum_natural_gradient`
    :tags: autograd

.. customgalleryitem::
    :tooltip: VQE optimization using quantum natural gradient.
    :figure: demonstrations/vqe_qng/vqe_qng_thumbnail.png
    :description: :doc:`demos/tutorial_vqe_qng`
    :tags: chemistry

.. customgalleryitem::
    :tooltip: Barren plateaus in quantum neural networks.
    :figure: demonstrations/barren_plateaus/surface.png
    :description: :doc:`demos/tutorial_barren_plateaus`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Understand the difference between local and global cost functions.
    :figure: demonstrations/local_cost_functions/Local_Thumbnail.png
    :description: :doc:`demos/tutorial_local_cost_functions`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Reduce the number of device executions by using a stochastic approximation optimization.
    :figure: demonstrations/spsa/spsa_mntn.png
    :description: :doc:`demos/spsa`
    :tags: qiskit

.. customgalleryitem::
    :tooltip: Reconstruct and differentiate univariate quantum functions.
    :figure: demonstrations/general_parshift/thumbnail_genpar.png
    :description: :doc:`demos/tutorial_general_parshift`
    :tags: gradients reconstruction

.. customgalleryitem::
    :tooltip: Doubly stochastic gradient descent.
    :figure: demonstrations/doubly_stochastic/single_shot.png
    :description: :doc:`Doubly stochastic gradient descent <demos/tutorial_doubly_stochastic>`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Differentiate any qubit gate with the stochastic parameter-shift rule.
    :figure: demonstrations/stochastic_parameter_shift/stochastic_parameter_shift_thumbnail.png
    :description: :doc:`demos/tutorial_stochastic_parameter_shift`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Rotoselect algorithm.
    :figure: demonstrations/rotoselect/rotoselect_structure.png
    :description: :doc:`demos/tutorial_rotoselect`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Frugal shot optimization with the Rosalin optimizer.
    :figure: demonstrations/rosalin/rosalin_thumb.png
    :description: :doc:`demos/tutorial_rosalin`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Solve combinatorial problems without a classical optimizer.
    :figure: demonstrations/falqon/falqon_thumbnail.png
    :description: :doc:`demos/tutorial_falqon`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Build trigonometric local models of your cost function.
    :figure: demonstrations/quantum_analytic_descent/xkcd.png
    :description: :doc:`demos/tutorial_quantum_analytic_descent`
    :tags: optimization model vqe

.. customgalleryitem::
    :tooltip: Optimizing measurement protocols with variational methods.
    :figure: demonstrations/quantum_metrology/illustration.png
    :description: :doc:`demos/tutorial_quantum_metrology`
    :tags: cirq metrology autograd

.. customgalleryitem::
    :tooltip: Learn about the variational quantum thermalizer algorithm, an extension of VQE.
    :figure: demonstrations/vqt/thumbnail.png
    :description: :doc:`demos/tutorial_vqt`
    :tags: chemistry

.. customgalleryitem::
    :tooltip: Variational Quantum Linear Solver.
    :figure: demonstrations/vqls/vqls_zoom.png
    :description: :doc:`demos/tutorial_vqls`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Coherent implementation of a variational quantum linear solver.
    :figure: demonstrations/coherent_vqls/cvqls_zoom.png
    :description: :doc:`demos/tutorial_coherent_vqls`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Optimize a Quantum Optical Neural Network using NLopt.
    :figure: demonstrations/qonn/qonn_thumbnail.png
    :description: :doc:`demos/qonn`
    :tags: autograd photonics

.. customgalleryitem::
    :tooltip: Ising model example with PennyLane PyTorch interface.
    :figure: demonstrations/Ising_model/isingspins.png
    :description: :doc:`demos/tutorial_isingmodel_PyTorch`
    :tags: pytorch autograd

.. customgalleryitem::
    :tooltip: Perform QAOA for MaxCut.
    :figure: demonstrations/qaoa_maxcut/qaoa_maxcut_partition.png
    :description: :doc:`demos/tutorial_qaoa_maxcut`
    :tags: autograd

.. customgalleryitem::
    :tooltip: QAO-Ansatz and DQVA for MIS.
    :figure: demonstrations/dqva_mis/mixer-unitary.png
    :description: :doc:`demos/tutorial_dqva_mis`
    :tags: autograd

:html:`</div></div><div style='clear:both'>`


.. toctree::
    :maxdepth: 2
    :hidden:

    demos/tutorial_qaoa_intro
    demos/tutorial_quantum_natural_gradient
    demos/tutorial_vqe_qng
    demos/tutorial_barren_plateaus
    demos/tutorial_local_cost_functions
    demos/spsa
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
    demos/tutorial_dqva_mis