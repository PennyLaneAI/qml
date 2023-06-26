r"""
A test demo
===========

.. raw:: html

    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.1.9/p5.js"></script>

.. meta::
    :property="og:description": Circuits as a Fourier Series
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_mitigation_advantage.png


Consider a single-qubit Hermitian operator :math:`G`. It acts on a two-dimensional
space, and therefore has two eigenvalues, which we write :math:`\kappa \pm \gamma`
(where :math:`\kappa` is the average and :math:`2\gamma` the difference between them). If
we exponentiate :math:`G`, with a parameter :math:`\theta`, we obtain a unitary gate:

.. math::

    U(\theta) = e^{i\theta G}.

Up to an overall phase :math:`e^{i\theta\kappa}` which we can ignore (it is
not measurable), the unitary :math:`U` has eigenvalues :math:`e^
{\pm i \theta\gamma}`. In the eigenbasis (i.e. the basis of eigenvectors
of :math:`G`, or equivalently :math:`U`), it is diagonal. We will draw this
diagonal matrix as a magenta box:

.. raw:: html

    <script src="../_static/test_demo/sketch1.js"></script>
    <figure>
      <center>
      <div id="sketch0_1"></div>
      </center>
    </figure>
"""
