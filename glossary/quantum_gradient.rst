.. role:: html(raw)
   :format: html

.. _glossary_quantum_gradient:

Quantum gradients
=================

The output of a :doc:`variational circuit </glossary/variational_circuit>` is the expectation value of a
measurement observable, which can be formally written as a
parameterized "quantum function" :math:`f(\theta)` in the tunable parameters :math:`\theta = \theta_1, \theta_2, \dots`.
As with any other such function, one can define partial derivatives of :math:`f` with respect to its parameters.

:html:`<br>`

.. figure:: ../_static/concepts/quantum_gradient.png
    :align: center
    :width: 40%
    :target: javascript:void(0);

:html:`<br>`


A *quantum gradient* is the vector of partial derivatives of a quantum function :math:`f(\theta)`:

.. math::

    \nabla_{\theta} f(\theta) = \begin{pmatrix}\partial_{\theta_1}f \\ \partial_{\theta_2} f \\ \vdots \end{pmatrix}

Sometimes, quantum nodes are defined by several expectation values, for example if multiple qubits are measured.
In this case, the output is described by a vector-valued function
:math:`\vec{f}(\theta) = (f_1(\theta), f_1(\theta), ...)^T`, and the quantum gradient becomes a "quantum Jacobian":

.. math::

    J_{\theta} f(\theta) = \begin{pmatrix}
                                \partial_{\theta_1}f_1 & \partial_{\theta_1} f_2 & \dots\\
                                \partial_{\theta_2}f_1 & \partial_{\theta_2} f_2 & \dots\\
                                \vdots &  & \ddots\\
                           \end{pmatrix}

It turns out that the gradient of a quantum function :math:`f(\theta)`
can in many cases be expressed as a linear combination of other quantum functions via
:doc:`parameter-shift rules </glossary/parameter_shift>`. This means that quantum gradients can be
computed by quantum computers, opening up quantum computing to gradient-based optimization such as
`gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_, which is widely used in machine learning.
