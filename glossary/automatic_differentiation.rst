.. role:: html(raw)
   :format: html

.. _glossary_automatic_differentiation:

Automatic differentiation
=========================

Derivatives and gradients are ubiquitous throughout science and engineering.
In recent years, automatic differentiation has become a key feature in many numerical software libraries,
in particular for machine learning (e.g., Theano_, Autograd_, Tensorflow_, or Pytorch_).

Generally speaking, automatic differentiation is the ability for a software library to compute
the derivatives of arbitrary numerical code. If you write an algorithm to compute some
function :math:`h(x)` (which may include mathematical expressions, but also control flow
statements like :code:`if`, :code:`for`, etc.), then automatic differentiation provides an
algorithm for :math:`\nabla h(x)` with the same degree of complexity as the original function.

*Automatic* differentiation should be distinguished from other forms of differentiation:

* *Symbolic differentiation*, where the actual equation of the derivative function is computed and
  evaluated, has a limited scope since it requires "hand-written" support for new functions.
* In *numerical differentiation*, such as the finite-difference
  method familiar from high-school calculus, the derivative of a function is approximated by
  numerically evaluating the function at two infinitesimally separated points. However, the approximation can be
  unstable, and in many settings an exact gradient is preferred.

The ability to compute :doc:`quantum gradients <glossary/quantum_gradient>` means that quantum computations
can become part of automatically differentiable :doc:`hybrid computation <glossary/hybrid_computation>` pipelines.

.. _Theano: https://github.com/Theano/Theano
.. _Autograd: https://github.com/HIPS/autograd
.. _Tensorflow: http://tensorflow.org/
.. _Pytorch: https://pytorch.org/
