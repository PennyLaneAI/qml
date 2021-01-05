.. role:: html(raw)
   :format: html

.. _glossary_differentiable_quantum_programming:

Differentiable Quantum Programming
==================================

`Differentiable programming
<https://en.wikipedia.org/wiki/Differentiable_programming>`_ is a style of
programming that uses `automatic differentiation
<https://en.wikipedia.org/wiki/Automatic_differentiation>`_ to compute the
derivatives of functions with respect to program inputs. In quantum computing,
one can automatically compute the derivatives of :doc:`variational circuits
</glossary/variational_circuit>` with respect to their input
parameters. Differentiable quantum programming is a paradigm that leverages this
to make quantum algorithms differentiable, and thereby trainable.


Automatic differentiation
-------------------------

Derivatives and gradients are ubiquitous throughout science and engineering.
In recent years, automatic differentiation has become a key feature in many numerical software libraries,
in particular for machine learning (e.g., Theano_, Autograd_, Tensorflow_,
Pytorch_, or Jax_).

:html:`<br>`

.. figure:: ../_static/concepts/autodiff_classical.png
    :align: center
    :width: 30%
    :target: javascript:void(0);

:html:`<br>`

Generally speaking, automatic differentiation is the ability for a software library to compute
the derivatives of arbitrary numerical code. If you write an algorithm to compute some
function :math:`h(x)` (which may include mathematical expressions, but also control flow
statements like :code:`if`, :code:`for`, etc.), then automatic differentiation provides an
algorithm for :math:`\nabla h(x)` with the same degree of complexity as the original function.

*Automatic* differentiation should be distinguished from other forms of differentiation:

* *Symbolic differentiation*, where the actual equation of the derivative function is computed and
  evaluated; has a limited scope since it requires "hand-written" support for new functions.
* *Numerical differentiation*, such as the finite-difference
  method familiar from high-school calculus, where the derivative of a function is approximated by
  numerically evaluating the function at two infinitesimally separated points. However, the approximation can be
  unstable, and in many settings an exact gradient is preferred.


Automatic differentiation of hybrid computations
------------------------------------------------

The ability to compute :doc:`quantum gradients </glossary/quantum_gradient>`
means that quantum computations can become part of automatically differentiable
:doc:`hybrid computation </glossary/hybrid_computation>` pipelines.

:html:`<br>`

.. figure:: ../_static/concepts/autodiff_quantum.png
    :align: center
    :width: 30%
    :target: javascript:void(0);

:html:`<br>`

For example, in PennyLane parameterized quantum operations carry information
about their parameters and their domains, and specify a "recipe" that details
how to automatically compute gradients. Many operations make use of
:doc:`parameter-shift rules </glossary/parameter_shift>` for this purpose (see,
for example, the arbitrary unitary rotation :class:`~.pennylane.Rot` which uses
parameter-shift rules to compute the derivative with respect to each of its
three parameters). In this way, the gradient of arbitrary sequences of
parameterized gates can be computed. Once evaluated the gradients can be fed
forward into subsequent parts of a larger hybrid computation.

:html:`<br>`

.. figure:: ../_static/concepts/autodiff_quantum_circuit.svg
    :align: center
    :width: 60%
    :target: javascript:void(0);

:html:`<br>`

.. _Theano: https://github.com/Theano/Theano
.. _Autograd: https://github.com/HIPS/autograd
.. _Tensorflow: http://tensorflow.org/
.. _Pytorch: https://pytorch.org/
.. _Jax: https://github.com/google/jax
