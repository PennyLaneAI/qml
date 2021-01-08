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


Types of differentiation in programming
---------------------------------------

Derivatives and gradients are ubiquitous throughout science and engineering.  In
recent years, automatic differentiation has become a key feature in many
numerical software libraries, in particular for machine learning (e.g., Theano_,
Autograd_, Tensorflow_, Pytorch_, or Jax_).

:html:`<br>`

.. figure:: ../_static/concepts/autodiff_classical.png
    :align: center
    :width: 30%
    :target: javascript:void(0);

:html:`<br>`

Generally speaking, automatic differentiation is the ability for a software
library to compute the derivatives of arbitrary numerical code. To better
understand how it works and what the benefits are, let's first analyze two other
forms of differentiation that are used in software: symbolic differentation, and
numerical differentiation.


Symbolic differentation
~~~~~~~~~~~~~~~~~~~~~~~

This method of differentiation is one that you may be familiar with from
calculus class. Symbolic differentiation manipulates expressions directly to
determine the mathematical form of the gradient. Both the input and output of
the differentiation procedure are mathematical expressions. For example,
consider the function :math:`\sin(x)`. Symbolic differentiation produces

.. math::

   \frac{d(\sin(x))}{dx} = \cos(x)


Computer algebra systems such as Mathematica perform symbolic differentiation -
if you ask it for the derivative of :math:`\sin(x)`, it will return to you
explicitly the function :math:`\cos(x)`, and not a numerical value. Under the
hood, a set of differentiation rules are implemented and followed. This includes
things like how to differentiate constants, polynomials, sums, and chain rules,
as well as derivatives of common functions (e.g., trigonometric functions). This
is a very powerful tool because once the set of rules is implemented, we can
symbolically differentiate arbitrary combinations of things that are encompassed
by them. However the scope of this method can be limited since it requires
"hand-written" support for new functions.


Numerical differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic differentation may not be always be possible when a function falls
outside the set of implementated rules. It may also be very computationally
complex. An alternative in these situations is to compute an approximation to
the derivative numerically --- this is something that can *always* be
done. There exist `a variety of such numerical methods
<https://en.wikipedia.org/wiki/Numerical_differentiation>`_, a common one being
the finite difference method. For this method the derivative is computed by
evaluating the function at two infinitesimally separated points. For example, we
can approximately compute the derivative of :math:`\sin(x)` as follows:

.. math::

   \frac{d(\sin(x))}{dx} \approx \frac{\sin(x + \epsilon) - \sin(x - \epsilon)}{2\epsilon}

The quality of this approximation depends on the size of :math:`\epsilon`. A
smaller :math:`\epsilon` is ideal, however this can quickly cause a calculation
to become unstable, and can introduce floating point errors. So while numerical
differentiation is always possible, it may not produce the best results for the
problem at hand.


Automatic differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

Automatic differentiation is also a numerical approach, but what distinguishes
it from methods like finite differences is that it is an *exact* method of
differentiation. In a similar vein as symbolic differentiation, each component
of the computation provides a rule for its derivative with respect to the
input. However, unlike symbolic differentiation where the input and output were
both mathematical expressions, the output of automatic differentiation is the
numerical value of the derivative.

If you write an algorithm to compute some function :math:`h(x)` (which may
include mathematical expressions, but also control flow statements like
:code:`if`, :code:`for`, etc.), then automatic differentiation provides an
algorithm for :math:`\nabla h(x)` with the same degree of complexity as the
original function.


At the end of the computation, the chain rule is used to combine all
these gradient rules and determine the total gradient.


A brief history of differentiable programming
---------------------------------------------

Differentiable programming is a conceptual shift from earlier treatment of deep
learning algorithms. The entire program is treated as differentiable - including
classical control flow such as loops, and if statements.

This means that entire *programs* are trainable and dynamic, while always
remaining differentiable.


Automatic differentiation of quantum computations
-------------------------------------------------

The ability to compute :doc:`quantum gradients </glossary/quantum_gradient>`
means that quantum computations can become part of automatically differentiable
:doc:`hybrid computation </glossary/hybrid_computation>` pipelines. For example,
in PennyLane parameterized quantum operations carry information about their
parameters and specify a "recipe" that details how to automatically compute
gradients.

:html:`<br>`

.. figure:: ../_static/concepts/autodiff_quantum_circuit.svg
    :align: center
    :width: 60%
    :target: javascript:void(0);

:html:`<br>`

Many quantum operations make use of :doc:`parameter-shift rules
</glossary/parameter_shift>` for this purpose. Parameter-shift rules bear some
resemblance to the finite difference method presented above. They involve
expressing the gradient of a function as some combination of that function at
two different points. However, unlike in the finite difference methods, those
two points are not infinitesimally close together, but rather quite far
apart. For example,

.. math::

   \frac{d(\sin(x))}{dx} = \cos(x) = \frac{\sin(x + s) - \sin(x-s)}{2 \sin(s)}

where :math:`s` is a large value, such as :math:`\pi/2`. The formula here comes
from trignometric identities relating :math:`\cos` and :math:`\sin`. This not only
provides us with an *exact* derivative, but handles the issue of instability in
finite differences that occurs when we must use a small shift.

This can be extended directly to the gradients of quantum operations and entire
quantum circuits (see, for example, the arbitrary unitary rotation
:class:`~.pennylane.Rot` which uses parameter-shift rules to compute the
derivative with respect to each of its three parameters). We simply evaluate the
circuit at two different points in parameter space. In this way, the gradient of
arbitrary sequences of parameterized gates can be computed. Once evaluated the
gradients can be fed forward into subsequent parts of a larger hybrid
computation.

:html:`<br>`

.. figure:: ../_static/concepts/autodiff_quantum.png
    :align: center
    :width: 30%
    :target: javascript:void(0);

:html:`<br>`


.. _Theano: https://github.com/Theano/Theano
.. _Autograd: https://github.com/HIPS/autograd
.. _Tensorflow: http://tensorflow.org/
.. _Pytorch: https://pytorch.org/
.. _Jax: https://github.com/google/jax
