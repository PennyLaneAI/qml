r"""
A test demo
===========

.. raw:: html

    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.1.9/p5.js"></script>

.. meta::
    :property="og:description": Circuits as a Fourier Series
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_mitigation_advantage.png

Single-qubit gates
------------------

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
  
We will work in the eigenbasis of :math:`U` from now on. This means that a column
vector :math:`[1, 0]^T` is the eigenvector associated with :math:`-\gamma`, and
:math:`[0, 1]^T` is asssociated with :math:`+\gamma`:

.. tip:: 

    Instructions: Click to toggle between eigenvectors.

.. raw:: html

    <figure>
      <center>
      <div id="sketch0_2"></div>
      </center>
    </figure>

We've written the matrix as a box to suggest a different way to think of it:
a <i>gate</i> in a quantum circuit. Instead of column vectors, we can use
bra-ket notation, with basis states :math:`\vert0\rangle = [1, 0]^T` and $\vert
1\rangle = [0, 1]^T$. We will also add some horizontal lines through the
gate to suggest that the states are "piped" through and pick up the
corresponding phase. <br>
  
.. tip::

  Instructions: Click to toggle between basis states :math:`\vert 0\rangle` and
  :math:`\vert 1\rangle`.
  
.. raw:: html

    <figure>
      <center>
      <div id="sketch1"></div>
      </center>
    </figure>

Frequency components
--------------------

You may wonder why we have introduced the parameter :math:`\theta`.
The idea is that, if we have :math:`U(\theta)` in our circuit, we can
treat :math:`\theta` as a parameter we can tune to improve the results of
the circuit, e.g. if we are trying to approximate a state of interest.
Viewed as functions of this tunable parameter :math:`\theta`, the purples
phases are <i>exponentials</i> of frequency :math:`\omega = \pm \gamma`.
Below, we plot the real and imaginary parts of these frequencies. 

.. tip::

    Instructions: Click to toggle between basis states $\vert
    0\rangle:math:` and `\vert 1\rangle$. While the mouse is in the magenta
    box, its vertical position controls :math:`\gamma`.

.. raw:: html

    <script src="../_static/test_demo/sketch2.js"></script>
    <figure>
      <center>
      <div id="sketch2"></div>
      </center>
    </figure>

Usually, we start a wire in the :math:`\vert 0\rangle` basis state.
We can "split" this into a combination of :math:`\vert 0\rangle` and :math:`\vert1\rangle` by applying a gate :math:`W`. We picture :math:`W` as a blue gate below. The state $\vert\psi(\theta)\rangle =
U(\theta)W\vert 0\rangle$ will then be a superposition of
:math:`e^{-i\theta\gamma}\vert0\rangle` and
:math:`e^{+i\theta\gamma}\vert1\rangle`. Again, viewed as a function of
:math:`\theta`, it has both frequency components, storing them in the
coefficient of the corresponding eigenstate.

.. tip::

    Instructions: Vertical mouse position in the blue box controls relative weight
    of :math:`\vert0\rangle` and :math:`\vert 1\rangle`. Clicking the magenta
    box toggles between them, with vertical mouse position controlling :math:`\gamma`.

.. raw:: html

    <figure>
      <center>
      <div id="sketch3" style></div>
      </center>
    </figure>

Once we've prepared a state :math:`\vert\psi(\theta)\rangle`, we can
measure it with some Hermitian operator :math:`M`. In the context of variational circuits, the result of
measurement can be used to optimize the parameter :math:`\theta`.
To this end, let's define the expectation value as a function of
:math:`\theta`,

.. math::

        f(\theta) = \langle \psi(\theta)\vert M \vert\psi(\theta)\rangle.

We represent the measurement :math:`M` as a yellow box, sandwiched between a
circuit on the left preparing the ket :math:`\vert\psi(\theta)\rangle` and
an adjoint circuit preparing the bra :math:`\langle \psi(\theta)\vert`.

.. note::

    Note that taking the adjoint swaps :math:`\pm\gamma`, and the order of elements
    in the circuit is inverted compared to the expression for
    :math:`f(\theta)`.

We can expand the bra and ket in terms of frequency components
:math:`e^{\pm i\theta\gamma}`, so :math:`f(\theta)` will be a sum of products of
these terms. We show this below.

.. tip::

    Instructions: Click on magenta boxes to toggle between
    frequency components contributing to :math:`f(\theta)`. Vertical position in
    the yellow box controls the constant terms $\langle 0 \vert M \vert
    0\rangle:math:` and `\langle 1 \vert M \vert 1\rangle$.
 

.. raw:: html

    <figure>
      <center>
      <div id="sketch4" style></div>
      </center>
    </figure>

To recap what we've learned so far: the ket :math:`\vert\psi
(\theta)\rangle` is a linear combination of the terms :math:`e^
{-i\theta\gamma}\vert0\rangle` and :math:`e^
{+i\theta\gamma}\vert1\rangle`. If we measure this state, the expectation
value :math:`f(\theta) = \langle\psi(\theta)\vert M\vert\psi
(\theta)\rangle` is a linear combination of products of the frequency
terms :math:`e^{\pmi\theta\gamma}`. More formally, we can write

.. math::

    f(\theta) = c_{-2} e^{-i2\gamma\theta} + c_0 + c_{+2}
    e^{+i2\gamma\theta}

for some coefficients :math:`c_{-2}, c_{0}, c_{+2}`. General expressions of
this form—sums of exponential terms with evenly spaced frequencies—are
called *Fourier series*. This turns out to be a useful way to look at
parameterized circuits!
"""
