.. role:: html(raw)
   :format: html

.. _glossary_automatic_differentiation:

Automatic differentiation
=========================

Derivatives and gradients are ubiquitous throughout science and engineering. I
n recent years, automatic differentiation has become a key feature in many numerical software libraries,
in particular for machine learning (e.g., Theano_, Autograd_, Tensorflow_, or Pytorch_).

Generally speaking, automatic differentiation is the ability for a software library to compute
the derivatives of arbitrary numerical code. If you write an algorithm to compute some
function :math:`h(x)` (which may include mathematical expressions, but also control flow
statements like :code:`if`, :code:`for`, etc.), then automatic differentiation provides an
algorithm for :math:`\nabla h(x)` with the same degree of complexity as the original function.

*Automatic* differentiation should be distinguished from other forms of differentiation.
*Manual differentiation*, where an expression is differentiated by hand — often on paper — is extremely
time-consuming and error-prone. In *numerical differentiation*, such as the finite-difference
method familiar from high-school calculus, the derivative of a function is approximated by
numerically evaluating the function at two infinitesimally separated points. However, this
method can sometimes be imprecise due to the constraints of classical floating-point arithmetic.

The ability to compute :ref:`quantum gradients <glossary_quantum_gradient>` means that quantum computations
can become part of automatically differentiable :ref:`hybrid computation <glossary_hybrid_computation>` pipelines.