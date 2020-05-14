.. role:: html(raw)
   :format: html

.. _glossary_parameter_shift:

Parameter-shift rules
=====================

The output of a :doc:`variational circuit </glossary/variational_circuit>` (i.e., the expectation of an observable)
can be written as a "quantum function" :math:`f(\theta)` parametrized by :math:`\theta = \theta_1, \theta_2, \dots`.
The partial derivative of :math:`f(\theta)` can in many cases be expressed as a linear combination of
other quantum functions. Importantly, these other
quantum functions typically use the same circuit, differing only in a shift of the argument. This means that
partial derivatives of a variational circuit can be computed by using the same variational circuit architecture.

Recipes of how to get partial derivatives by evaluated parameter-shifted instances of a variational circuit
are called *parameter-shift rules*, and have been first introduced to quantum machine learning in
`Mitarai et al. (2018) <https://arxiv.org/abs/1803.00745>`_, and extended in
`Schuld et al. (2018) <https://arxiv.org/abs/1811.11184>`_.

:html:`<br>`

.. figure:: ../_static/concepts/gradients2.png
    :align: center
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`

Making a rough analogy to classically computable functions, this is similar to how the
derivative of the function :math:`f(x)=\sin(x)` is identical to
:math:`\frac{1}{2}\sin(x+\frac{\pi}{2}) - \frac{1}{2}\sin(x-\frac{\pi}{2})`. So the same underlying
algorithm can be reused to compute both :math:`\sin(x)` and its derivative (by evaluating at :math:`x\pm\frac{\pi}{2}`).
This intuition holds for many quantum functions of interest: *the same circuit can be
used to compute both the quantum function and the gradient of the quantum function* [#]_.

A more technical explanation
----------------------------

Quantum circuits are specified by a sequence of gates. The unitary transformation
carried out by the circuit can thus be broken down into a product of unitaries:

.. math:: U(x; \theta) = U_N(\theta_{N}) U_{N-1}(\theta_{N-1}) \cdots U_i(\theta_i) \cdots U_1(\theta_1) U_0(x).

Each of these gates is unitary, and therefore must have the form
:math:`U_{j}(\gamma_j)=\exp{(i\gamma_j H_j)}` where :math:`H_j` is a Hermitian operator
which generates the gate and :math:`\gamma_j` is the gate parameter.
We have omitted which wire each unitary acts on, since it is not necessary for the following discussion.

.. note::

    In this example, we have used the input :math:`x` as the argument for gate :math:`U_0`
    and the parameters :math:`\theta` for the remaining gates. This is not required.
    Inputs and parameters can be arbitrarily assigned to different gates.

A single parameterized gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us single out a single parameter :math:`\theta_i` and its associated gate :math:`U_i(\theta_i)`.
For simplicity, we remove all gates except :math:`U_i(\theta_i)` and :math:`U_0(x)` for the moment.
In this case, we have a simplified quantum circuit function

.. math::
    f(x; \theta_i) = \langle 0 | U_0^\dagger(x)U_i^\dagger(\theta_i)\hat{B}U_i(\theta_i)U_0(x) | 0 \rangle = \langle x | U_i^\dagger(\theta_i)\hat{B}U_i(\theta_i) | x \rangle.

For convenience, we rewrite the unitary conjugation as a linear
transformation :math:`\mathcal{M}_{\theta_i}` acting on the operator :math:`\hat{B}`:

.. math::
    U_i^\dagger(\theta_i)\hat{B}U_i(\theta_i) = \mathcal{M}_{\theta_i}(\hat{B}).

The transformation :math:`\mathcal{M}_{\theta_i}` depends smoothly on
the parameter :math:`\theta_i`, so this quantum function will have a well-defined gradient:

.. math::
    \nabla_{\theta_i}f(x; \theta_i) = \langle x | \nabla_{\theta_i}\mathcal{M}_{\theta_i}(\hat{B}) | x \rangle \in \mathbb{R}.

The key insight is that we can, in many cases of interest, express this
gradient as a linear combination of the same transformation :math:`\mathcal{M}`, but with different parameters. Namely,

.. math::
    \nabla_{\theta_i}\mathcal{M}_{\theta_i}(\hat{B}) = c[\mathcal{M}_{\theta_i + s}(\hat{B}) - \mathcal{M}_{\theta_i - s}(\hat{B})],

where the multiplier :math:`c` and the shift :math:`s` are determined completely by the type of
transformation :math:`\mathcal{M}` and independent of the value of :math:`\theta_i`.


.. note::

    While this construction bears some resemblance to the numerical finite-difference method for
    computing derivatives, here :math:`s` is finite rather than infinitesimal.

Multiple parameterized gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To complete the story, we now go back to the case where there are many gates in the circuit.
We can absorb any gates applied before gate :math:`i` into the initial
state: :math:`|\psi_{i-1}\rangle = U_{i-1}(\theta_{i-1}) \cdots U_{1}(\theta_{1})U_{0}(x)|0\rangle`.
Similarly, any gates applied after gate :math:`i` are combined with the observable :math:`\hat{B}`:
:math:`\hat{B}_{i+1} = U_{N}^\dagger(\theta_{N}) \cdots U_{i+1}^\dagger(\theta_{i+1}) \hat{B} U_{i+1}(\theta_{i+1}) \cdots U_{N}(\theta_{N})`.

With this simplification, the quantum circuit function becomes

.. math:: f(x; \theta) = \langle \psi_{i-1} | U_i^\dagger(\theta_i) \hat{B}_{i+1} U_i(\theta_i) | \psi_{i-1} \rangle = \langle \psi_{i-1} | \mathcal{M}_{\theta_i} (\hat{B}_{i+1}) | \psi_{i-1} \rangle,

and its gradient is

.. math:: \nabla_{\theta_i}f(x; \theta) = \langle \psi_{i-1} | \nabla_{\theta_i}\mathcal{M}_{\theta_i} (\hat{B}_{i+1}) | \psi_{i-1} \rangle.

This gradient has the exact same form as the single-gate case, except we modify the state
:math:`|x\rangle \rightarrow |\psi_{i-1}\rangle` and the measurement operator
:math:`\hat{B}\rightarrow\hat{B}_{i+1}`. In terms of the circuit, this means we can leave
all other gates as they are, and only modify gate :math:`U(\theta_i)` when we want to
differentiate with respect to the parameter :math:`\theta_i`.

.. note::

    Sometimes we may want to use the same classical parameter with multiple gates in the circuit.
    Due to the `product rule <https://en.wikipedia.org/wiki/Product_rule>`_, the total gradient will then
    involve contributions from each gate that uses that parameter.

Pauli gate example
~~~~~~~~~~~~~~~~~~~

Consider a quantum computer with parameterized gates of the form

.. math:: U_i(\theta_i)=\exp\left(-i\tfrac{\theta_i}{2}\hat{P}_i\right),

where :math:`\hat{P}_i=\hat{P}_i^\dagger` is a Pauli operator.

The gradient of this unitary is

.. math:: \nabla_{\theta_i}U_i(\theta_i) = -\tfrac{i}{2}\hat{P}_i U_i(\theta_i) = -\tfrac{i}{2}U_i(\theta_i)\hat{P}_i .

Substituting this into the quantum circuit function :math:`f(x; \theta)`, we get

.. math::
   :nowrap:

   \begin{align}
       \nabla_{\theta_i}f(x; \theta) = &
       \frac{i}{2}\langle \psi_{i-1} | U_i^\dagger(\theta_i) \left( P_i \hat{B}_{i+1} - \hat{B}_{i+1} P_i \right) U_i(\theta_i)| \psi_{i-1} \rangle \\
       = & \frac{i}{2}\langle \psi_{i-1} | U_i^\dagger(\theta_i) \left[P_i, \hat{B}_{i+1}\right]U_i(\theta_i) | \psi_{i-1} \rangle,
   \end{align}

where :math:`[X,Y]=XY-YX` is the commutator.

We now make use of the following mathematical identity for commutators involving Pauli
operators (`Mitarai et al. (2018) <https://arxiv.org/abs/1803.00745>`_):

.. math:: \left[ \hat{P}_i, \hat{B} \right] = -i\left(U_i^\dagger\left(\tfrac{\pi}{2}\right)\hat{B}U_i\left(\tfrac{\pi}{2}\right) - U_i^\dagger\left(-\tfrac{\pi}{2}\right)\hat{B}U_i\left(-\tfrac{\pi}{2}\right) \right).

Substituting this into the previous equation, we obtain the gradient expression

.. math::
   :nowrap:

   \begin{align}
       \nabla_{\theta_i}f(x; \theta) = & \hphantom{-} \tfrac{1}{2} \langle \psi_{i-1} | U_i^\dagger\left(\theta_i + \tfrac{\pi}{2} \right) \hat{B}_{i+1} U_i\left(\theta_i + \tfrac{\pi}{2} \right) | \psi_{i-1} \rangle \\
       & - \tfrac{1}{2} \langle \psi_{i-1} | U_i^\dagger\left(\theta_i - \tfrac{\pi}{2} \right) \hat{B}_{i+1} U_i\left(\theta_i - \tfrac{\pi}{2} \right) | \psi_{i-1} \rangle.
   \end{align}

Finally, we can rewrite this in terms of quantum functions:

.. math:: \nabla_{\theta}f(x; \theta) = \tfrac{1}{2}\left[ f(x; \theta + \tfrac{\pi}{2}) - f(x; \theta - \tfrac{\pi}{2}) \right].

Gaussian gate example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For quantum devices with continuous-valued operators, such as photonic quantum computers, it is
convenient to employ the `Heisenberg picture <https://en.wikipedia.org/wiki/Heisenberg_picture>`_, i.e.,
to track how the gates :math:`U_i(\theta_i)` transform the final measurement operator :math:`\hat{B}`.

As an example, we consider the `Squeezing gate <https://en.wikipedia.org/wiki/Squeeze_operator>`_. In the
Heisenberg picture, the Squeezing gate causes the quadrature operators :math:`\hat{x}` and :math:`\hat{p}`
to become rescaled:

.. math::
   :nowrap:

   \begin{align}
       \mathcal{M}^S_r(\hat{x}) = & S^\dagger(r)\hat{x}S(r) \\
                                   = & e^{-r}\hat{x}
   \end{align}

and

.. math::
   :nowrap:

   \begin{align}
       \mathcal{M}^S_r(\hat{p}) = & S^\dagger(r)\hat{p}S(r) \\
                                   = & e^{r}\hat{p}.
   \end{align}

Expressing this in matrix notation, we have

.. math::
   :nowrap:

   \begin{align}
       \begin{bmatrix}
           \hat{x} \\
           \hat{p}
       \end{bmatrix}
       \rightarrow
       \begin{bmatrix}
          e^{-r} & 0 \\
          0      & e^r
       \end{bmatrix}
       \begin{bmatrix}
           \hat{x} \\
           \hat{p}
       \end{bmatrix}.
   \end{align}

The gradient of this transformation can easily be found:

.. math::
   :nowrap:

   \begin{align}
       \nabla_r
       \begin{bmatrix}
           e^{-r} & 0 \\
           0 & e^r
       \end{bmatrix}
       =
       \begin{bmatrix}
           -e^{-r} & 0 \\
           0 & e^r
       \end{bmatrix}.
   \end{align}

We notice that this can be rewritten this as a linear combination of squeeze operations:

.. math::
   :nowrap:

   \begin{align}
       \begin{bmatrix}
           -e^{-r} & 0 \\
           0 & e^r
       \end{bmatrix}
       =
       \frac{1}{2\sinh(s)}
       \left(
       \begin{bmatrix}
           e^{-(r+s)} & 0 \\
           0 & e^{r+s}
       \end{bmatrix}
       -
       \begin{bmatrix}
           e^{-(r-s)} & 0 \\
           0 & e^{r-s}
       \end{bmatrix}
       \right),
   \end{align}

where :math:`s` is an arbitrary nonzero shift [#]_.

As before, assume that an input :math:`y` has already been embedded into a quantum
state :math:`|y\rangle = U_0(y)|0\rangle` before we apply the squeeze gate. If we measure the :math:`\hat{x}` operator,
we will have the following quantum circuit function:

.. math::
   f(y;r) = \langle y | \mathcal{M}^S_r (\hat{x}) | y \rangle.

Finally, its gradient can be expressed as

.. math::
   :nowrap:

   \begin{align}
       \nabla_r f(y;r) = &  \frac{1}{2\sinh(s)} \left[
                            \langle y | \mathcal{M}^S_{r+s} (\hat{x}) | y \rangle
                           -\langle y | \mathcal{M}^S_{r-s} (\hat{x}) | y \rangle \right] \\
                       = & \frac{1}{2\sinh(s)}\left[f(y; r+s) - f(y; r-s)\right].
   \end{align}

.. note::

    For simplicity of the discussion, we have set the phase angle of the Squeezing gate to be zero.
    In the general case, Squeezing is a two-parameter gate, containing a squeezing magnitude and a squeezing angle.
    However, we can always decompose the two-parameter form into a Squeezing gate like the one above,
    followed by a Rotation gate.

.. rubric:: Footnotes

.. [#] This should be contrasted with software which can perform automatic differentiation on classical
       simulations of quantum circuits, such as `Strawberry Fields <https://strawberryfields.readthedocs.io/en/latest/>`_.

.. [#] In situations where no formula for automatic quantum gradients is known,
       one can fall back to approximate gradient estimation using numerical methods.

.. [#] In physical experiments, it is beneficial to choose :math:`s` so that the
       additional squeezing is small. However, there is a tradeoff, because we also want to make sure
       :math:`\frac{1}{2\sinh(s)}` does not blow up numerically.

