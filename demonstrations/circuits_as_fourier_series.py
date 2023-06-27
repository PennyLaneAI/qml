r"""
Circuits as Fourier series
==========================

.. raw:: html

    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.1.9/p5.js"></script>

.. meta::
    :property="og:description": TODO
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_mitigation_advantage.png

Part I: Single-qubit gates
--------------------------

Consider a single-qubit Hermitian operator :math:`G`. It acts on a two-dimensional
space, and therefore has two eigenvalues, which we write :math:`\kappa \pm \gamma`
(where :math:`\kappa` is the average and :math:`2\gamma` the difference between them). If
we exponentiate :math:`G`, with a parameter :math:`\theta`, we obtain a unitary gate:

.. math::

    U(\theta) = e^{i\theta G}.

Up to an overall phase :math:`e^{i\theta\kappa}` which we can ignore (it is
not measurable), the unitary :math:`U` has eigenvalues :math:`e^
{\pm i \theta\gamma}`. In the eigenbasis (i.e., the basis of eigenvectors
of :math:`G`, or equivalently :math:`U`), it is diagonal. We will draw this
diagonal matrix as a magenta box:

.. raw:: html

    <script src="../_static/demos/circuits_as_fourier_series/sketch1.js"></script>
    <figure>
      <center>
      <div id="sketch0_1"></div>
      </center>
    </figure>
  
We will work in the eigenbasis of :math:`U` from now on. This means that a column
vector :math:`[1, 0]^T` is the eigenvector associated with :math:`-\gamma`, and
:math:`[0, 1]^T` is asssociated with :math:`+\gamma`:

.. tip:: 

    Click to toggle between eigenvectors.

.. raw:: html

    <figure>
      <center>
      <div id="sketch0_2"></div>
      </center>
    </figure>

We've written the matrix as a box to suggest a different way to think of it:
a *gate* in a quantum circuit. Instead of column vectors, we can use
bra-ket notation, with basis states :math:`\vert0\rangle = [1, 0]^T` and
:math:`\vert 1\rangle = [0, 1]^T`. We will also add some horizontal lines through the
gate to suggest that the states are "piped" through and pick up the
corresponding phase. 
  
.. tip::

  Click to toggle between basis states :math:`\vert 0\rangle` and
  :math:`\vert 1\rangle`.
  
.. raw:: html

    <figure>
      <center>
      <div id="sketch1"></div>
      </center>
    </figure>

Part II: Frequency components
-----------------------------

You may wonder why we have introduced the parameter :math:`\theta`.
The idea is that, if we have :math:`U(\theta)` in our circuit, we can
treat :math:`\theta` as a parameter we can tune to improve the results of
the circuit, e.g. if we are trying to approximate a state of interest.
Viewed as functions of this tunable parameter :math:`\theta`, the purples
phases are *exponentials* of frequency :math:`\omega = \pm \gamma`.
Below, we plot the real and imaginary parts of these frequencies. 

.. tip::

    Click to toggle between basis states :math:`\vert 0\rangle`
    and :math:`\vert 1\rangle`. While the mouse is in the magenta
    box, its vertical position controls :math:`\gamma`.

.. raw:: html

    <script src="../_static/demos/circuits_as_fourier_series/sketch2.js"></script>
    <figure>
      <center>
      <div id="sketch2"></div>
      </center>
    </figure>

Usually, we start a wire in the :math:`\vert 0\rangle` basis state. We
can "split" this into a combination of :math:`\vert 0\rangle`
and :math:`\vert1\rangle` by applying a gate :math:`W`. We picture :math:`W`
as a blue gate below. The state
:math:`\vert\psi(\theta)\rangle =U(\theta)W\vert 0\rangle`will then be a superposition of
:math:`e^{-i\theta\gamma}\vert0\rangle` and
:math:`e^{+i\theta\gamma}\vert1\rangle`. Again, viewed as a function of
:math:`\theta`, it has both frequency components, storing them in the
coefficient of the corresponding eigenstate.

.. tip::

    Vertical mouse position in the blue box controls relative weight
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

    Click on magenta boxes to toggle between frequency
    components contributing to :math:`f(\theta)`. Vertical position in the
    yellow box controls the constant terms
    :math:`\langle 0 \vert M \vert0\rangle` and
    :math:`\langle 1 \vert M \vert 1\rangle`.
 

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
terms :math:`e^{\pm i\theta\gamma}`. More formally, we can write

.. math::

    f(\theta) = c_{-2} e^{-i2\gamma\theta} + c_0 + c_{+2}
    e^{+i2\gamma\theta}

for some coefficients :math:`c_{-2}, c_{0}, c_{+2}`. General expressions of
this form—sums of exponential terms with evenly spaced frequencies—are
called *Fourier series*. This turns out to be a useful way to look at
parameterized circuits!

Part III: Larger circuits
-------------------------

We can embed this structure, with a single occurrence of :math:`U
(\theta)`, into a larger circuit. Instead of a linear combination
of :math:`\vert0\rangle` and :math:`\vert 1\rangle`, it will be a linear
combination of the form

.. math::

    \alpha_0 \vert 0\rangle \otimes \vert \psi_0\rangle + \alpha_1 \vert
    1\rangle \otimes\vert \psi_1\rangle,

for some states :math:`\vert \psi_0\rangle` and :math:`\vert \psi_1\rangle` on
the rest of the circuit, up to a reordering of wires. (This follows by
factorizing with respect to the tensor product.)


After applying :math:`U(\theta)`, the overall state will become

.. math::

    e^{-i\theta\gamma}\alpha_0 \vert 0\rangle \otimes\vert \psi_0\rangle + e^{+i\theta\gamma}\alpha_1 \vert 1\rangle \otimes\vert \psi_1\rangle.

Whatever subsequent gates we apply, as long as there is only one occurrence
of :math:`U(\theta)`, this expansion in frequencies :math:`e^
{\pm i\theta\gamma}` is the same. We illustrate how a single copy of :math:`U
(\theta)` acts on the larger circuit below.

.. raw:: html

    <script src="../_static/demos/circuits_as_fourier_series/sketch3.js"></script>
    <figure>
      <center>
      <div id="sketch5" style></div>
      </center>
    </figure>

If there are multiple copies of :math:`U(\theta)` in the circuit, we simply
get more frequencies. Each box will multiply existing coefficients
by :math:`e^{-i\gamma\theta}` or :math:`e^{+i\gamma\theta}`, depending on the
state it encounters. This will iteratively build up a state of the following
form:

.. math::

  \vert \psi(\theta)\rangle = \alpha_{0\cdots 00}e^{-in\gamma\theta}\vert
  \psi_{0\cdots 00}\rangle + \alpha_{0\cdots 01}e^{-i(n-2)\gamma\theta}\vert
  \psi_{0\cdots 01}\rangle + \cdots + \alpha_{1\cdots 11}e^{+in\gamma\theta}\vert
  \psi_{1\cdots 11}\rangle,

where the first term corresponds to choosing :math:`e^{-i\gamma \theta}` in
each box, and the last term :math:`e^{+i\gamma \theta}` in each box. Note
that, for intermediate frequencies, many different choices yield the same
final result! We illustrate for :math:`n = 2` below, where the total state at
the end of the circuit is

.. math::

  \vert \psi(\theta)\rangle = \alpha_{00}e^{-i2\theta\gamma} \vert \psi_{00}\rangle + \alpha_{01}e^{i0\gamma}\vert
  \psi_{01}\rangle + \alpha_{10}e^{i0\gamma}\vert \psi_{10}\rangle + \alpha_{1}e^{+i2\theta\gamma} \vert \psi_{11}\rangle.

Here, there are two ways to obtain a frequency of zero.

.. tip::

    Click on the magenta boxes terms to choose
    :math:`e^{\pm i\gamma\theta}` in each box. The final frequency is shown on
    the right. Although we are placing the boxes in parallel, the result is the
    same for series.

.. raw:: html

    <figure>
      <center>
      <div id="sketch6" style></div>
      </center>
    </figure>

  
As usual, we can form the expectation value of a measurement
:math:`f(\theta) = \langle \psi(\theta)\vert M \vert \psi
(\theta)\rangle`. This will be a linear combination of overlaps of the
states above. For instance, when :math:`n =2`, it will be built from
overlaps of the states (including the associated phases)

.. math::

    e^{-i2\theta\gamma}\vert \psi_{00}\rangle, e^{i0\gamma}\vert \psi_{01}\rangle, e^{i0\gamma}\vert
    \psi_{10}\rangle, e^{+i2\theta\gamma}\vert \psi_{11}\rangle.

Thus, the expectation is a Fourier series, with terms arising from
all the different ways of combining these frequencies. For general :math:`n`, this is

.. math::

    f(\theta) = c_{-2n}e^{-2in\gamma\theta} +
    c_{-2(n-1)}e^{-2i(n-1)\gamma\theta} + \cdots + c_0 + \cdots +
    c_{2(n-1)}e^{+2i(n-1)\gamma\theta} + c_{2n}e^{2in\gamma\theta}.

Below, we illustrate for :math:`n = 2`, where the Fourier series consists
of five terms:

.. math::

    f(\theta) = c_{-4}e^{-4i\gamma\theta} + c_{-2}e^{-2i\gamma\theta} +
    c_{0} + c_{+2}e^{+2i\gamma\theta} +c_{+4}e^{+4i\gamma\theta}.


.. tip::

    Click on the magenta boxes terms to choose
    :math:`e^{\pm i\gamma\theta}` in each box, contributing to the final
    frequency. This frequency is shown below the circuit.

.. raw:: html

    <figure>
      <center>
      <div id="sketch7" style></div>
      </center>
    </figure>

Part IV: Coefficient vectors
----------------------------

So far, we've focused on the :math:`\theta`-dependent "pure frequency" terms :math:`e^{i\omega\theta\gamma}` appearing in the Fourier series.
However, the *coefficients* :math:`c_w` also have an important role
to play.
It turns out that, for a given set of frequencies in the Fourier
series, the coefficients *uniquely* characterize :math:`f(\theta)`.
We'll redo the :math:`n = 2` example but highlight the coefficients instead;
these depend on the structure of the circuit and the choice of
measurement.

.. tip::

    Click on the magenta boxes terms to choose
    :math:`e^{\pm i\gamma\theta}` in each box, contributing to the final
    frequency. The coefficient of this final frequency is shown below the circuit. </p>

.. raw:: html

    <script src="../_static/demos/circuits_as_fourier_series/sketch4.js"></script>
    <figure>
        <center>
        <div id="sketch8" style></div>
        </center>
    </figure>

he set of Fourier sums of fixed degree form a vector space, and the pure
frequencies :math:`e^{i\omega \theta\gamma}` are a *basis*. We can assemble
these coefficients into a column vector,
:math:`\vec{c}_f`, which contains the same information as the
function :math:`f(\theta)`:

.. math::

    f(\theta) = c_{-2n} e^{-i2n\theta\gamma} + \cdots + c_{+2n} e^{+i2n\theta\gamma} \quad
    \Leftrightarrow \quad \vec{c}_f =
    \begin{bmatrix}
    c_{-2n} \\
    \vdots \\
    c_{2n}
    \end{bmatrix}.

If an operation
on the function :math:`f(\theta)` preserves the structure of the function,
i.e., it only modifies the coefficients, then we can think of it as
an operation on vectors instead! Our first example is
differentiation. This simply pulls down a constant term from the
exponent of each pure frequency:

.. math::

    f'(\theta) = (-i2n\gamma)c_{-2n} e^{-i2n\theta\gamma} + \cdots +
    (+i2n\gamma)c_{+2n} e^{+i2n\theta\gamma}.

In terms of coefficient vectors, however, it is just a diagonal
matrix :math:`D`:

.. math::
    \vec{c}_{f'} =
    \begin{bmatrix}
    (-i2n\gamma)c_{-2n} \\
    \vdots \\
    (+i2n\gamma)c_{2n}
    \end{bmatrix} =
    \begin{bmatrix}
    -i2n\gamma & & \\
    & \ddots & \\
    && +i2n\gamma
    \end{bmatrix}
    \begin{bmatrix}
    c_{-2n} \\
    \vdots \\ c_{2n}
    \end{bmatrix} = D\vec{c}.


.. tip::

    The derivative of the expectation
    :math:`f(\theta)` as a coefficient vector. Click on the magenta boxes to
    choose terms contributing to the final frequency.

.. raw:: html

    <figure>
        <center>
        <div id="sketch9" style></div>
        </center>
    </figure>

Another particularly simple operation on :math:`f(\theta)` is to
shift the parameter :math:`\theta` by some constant amount :math:`s`, giving a new
function :math:`f_s(\theta) = f(\theta + s)`, also called a
*parameter shift*. From index laws, this adds an exponential
factor to each coefficient:

.. math::

    f(\theta + s) = c_{-2n} e^{-i2n(\theta + s)\gamma} + \cdots +
    c_{+2n} e^{+i2n(\theta + s)\gamma} = e^{-i2ns\gamma}c_{-2n} e^{-i2n\theta\gamma} + \cdots
    + e^{+i2ns\gamma}c_{+2n} e^{+i2n\theta\gamma}.

Once again, this can be viewed as a diagonal matrix :math:`T_s`
acting on the coefficient vector:

.. math::

    \vec{c}_{f_s} =
    \begin{bmatrix}
    e^{-i2ns\gamma}c_{-2n} \\
    \vdots \\
    e^{+i2ns\gamma}c_{2n}
    \end{bmatrix} =
    \begin{bmatrix}
    e^{-i2ns\gamma} & & \\
    & \ddots & \\
    && e^{+i2ns\gamma}
    \end{bmatrix}
    \begin{bmatrix}
    c_{-2n} \\
    \vdots \\ c_{2n}
    \end{bmatrix} = T_s\vec{c}.

.. tip::

    The parameter shifted expectation
    :math:`f(\theta + s)` as a coefficient vector. Click on the magenta boxes to
    choose terms contributing to the final frequency.

.. raw:: html

    <figure>
        <center>
        <div id="sketch10" style></div>
        </center>
    </figure>

Part V: The two-term parameter-shift rule
-----------------------------------------

Our original motivation for introducing :math:`\theta` was to *optimize* the
measurement result :math:`f(\theta)`. If we can differentiate :math:`f
(\theta)`, we can use tools from classical machine learning such as *gradient
descent*. The problem is that circuits are black boxes; all we can do is set
some parameters, pull a lever, and out pops a measurement outcome. It's a bit
like a toaster. How do you differentiate a toaster?

.. tip::

    Click the measurement button for toasty measurement outcomes.

.. raw:: html

    <script src="../_static/demos/circuits_as_fourier_series/sketch5.js"></script>
    <figure>
        <center>
        <div id="sketch11" style></div>
        </center>
    </figure>

Luckily, the magic of Fourier series and coefficient vectors come to the
rescue. The basic idea is to write the differentiation matrix :math:`D` as a
linear combination of shift matrices :math:`T_s`. Although we can't
differentiate directly, we can change parameters! Let's illustrate with the
simple example of :math:`n = 1`. In this case, the matrices are

.. math::

    D =
    \begin{bmatrix}-2i\gamma && \\ & 0 & \\ & & +2i\gamma \end
    {bmatrix}, \quad T_s = \begin{bmatrix}e^{-2i\gamma s} && \\ &1& \\ && e^
    {+2i\gamma s} \end{bmatrix}.


It's easy to check that for any :math:`s`,

.. math::
    
    D = \frac{2\gamma}{\sin(\gamma s)}(T_s - T_{-s}).

Translating back into statements about functions, we learn that

.. math::

    f'(\theta) = \frac{2\gamma}{\sin(\gamma s)}[f(\theta + s) - f
    (\theta - s)].


This is called *two-term parameter-shift rule*.

.. tip::

    Click anywhere for gratuitous toast.

.. raw:: html

    <figure>
        <center>
        <div id="sketch11_5" style></div>
        </center>
    </figure>

The two-term rule has a simple geometric interpretation. Changing
:math:`\theta` takes us around a circle of fixed radius
:math:`r = \vert c_{\pm2}\vert` at speed :math:`\gamma/2\pi`.

.. note::

    Note that parameter shifts add phases, hence don't change :math:`\vert c_
    {\pm 2}\vert`. The radius is given by either, since reality of
    measurement outcomes implies :math:`c_{+2} = \overline{c_{-2}}`.

The derivative :math:`f'(\theta)` is a tangent vector of
length :math:`r\gamma`. We can choose Cartesian coordinates where this
tangent vector is vertical, with components

.. math::

    f'(\theta) = (0, r\gamma).

The parameter shifts :math:`f(\theta \pm s)`, on the other hand, have
components

.. math::

    f(\theta \pm s) = (r \cos(\gamma s), \pm r\sin(\gamma s)).

It follows immediately that

.. math::

    f'(\theta) = \frac{2\gamma}{\sin(\gamma s)} [f(\theta + s) - f(\theta - s)].

We picture the geometry below. On the right, tangent to the circle,
is :math:`f'` as a coefficient vector in the :math:`c_{\pm2}` plane. We
display :math:`f(\theta + s)` and :math:`-f(\theta - s)` as light magenta
lines. Their vector sum :math:`\Delta f` is a dark magenta line.

.. tip::

    Horizontal mouse position controls :math:`s`. 

.. raw:: html

    <figure>
        <center>
        <div id="sketch12" style></div>
        </center>
    </figure>

Part VI: The general parameter-shift rule
-----------------------------------------

For :math:`n > 1`, parameter shift rules don't have a simple geometric
interpretation. The problem is that each pair of coefficients
:math:`c_{\pm \omega}` is associated with a circle governing two components of
the coefficient vector. To find the derivative, we need to understand each
pair of components separately, but parameter shifts *simultaneously* wind us
around all the circles, at different speeds!   Geometrically speaking,
putting all these circles together gives a
*higher-dimensional donut*, which parameter shifts wind us
around. This sounds complicated!

It also looks complicated, as we illustrate for :math:`n = 3` below. We
set :math:`\gamma = 1` so that, on the circle in the :math:`c_{\pm
\omega}` plane, we execute :math:`\omega` revolutions for a single cycle
of :math:`s`.

.. tip::

    Horizontal mouse position controls :math:`s`. 

.. raw:: html

    <script src="../_static/demos/circuits_as_fourier_series/sketch6.js"></script>
    <figure>
        <center>
        <div id="sketch13" style></div>
        </center>
    </figure>


Perhaps surprisingly, the coefficient vector perspective and a few tricks let
us derive the general parameter-shift rule straighforwardly. We start with
the observation that :math:`f'(\theta)` has a coefficient vector
with :math:`2n` nonzero components, since the constant term always vanishes.
Thus, :math:`2n` linearly independent parameters shifts :math:`f
(\theta + s_k)` should be sufficient to reconstruct the derivative, with

.. math::

    f'(\theta) = \sum_{k=1}^{2n} \beta_k f(\theta + s_k) \tag{1}\label{GPS}

for some coefficients :math:`\beta_k`. The problem is how to find the shifts
and coefficients! We can invoke linear algebra to find coefficients, but only
once we choose shifts, and it's not obvious how to get them to be
independent. We can see the problem for :math:`n = 3`, where we choose six
random shifts. Are they independent or not? It seems hard to tell. Is there a
better approach?

.. tip::

    Horizontal mouse position controls :math:`s`,
    which is now "quantized" with random shifts. 

.. raw:: html

    <figure>
        <center>
        <div id="sketch14" style></div>
        </center>
    </figure>

Thankfully, there is! We can solve two problems at once by introducing
an *inner product*, i.e., a way to find the scalar overlap of two vectors.
This will let us identify orthogonal and hence independent
shifts :math:`s_k`. Since they are orthogonal, we can also easily determine
the coefficients :math:`\beta_k`. The idea is straightforward: since the
matrices of interest are diagonal,

.. math::

    D = \mbox{diag}(-2in\gamma, \ldots, +2in\gamma), \quad T_s = \mbox{diag}
    (e^{-2in\gamma s}, \ldots, e^{+2in\gamma s}),

we can just pluck out the vector of diagonal entries and define a complex
inner product in the usual way. (Technically, this is the `Frobenius inner
product <https://en.wikipedia.org/wiki/Frobenius_inner_product>`__ for
matrices:

.. math::

    \langle A, B\rangle = \mbox{Tr}[A^\dagger B].)

Consider two shifts :math:`s, t \in [0, 2\pi/\gamma)`, and
define :math:`\omega = e^{2\pi i\gamma(t-s)}`. The inner product of diagonal
shift matrices :math:`T_s, T_t` is

.. math::

    \langle T_s, T_t\rangle = \sum_{j=-n}^n \omega^j = \omega^{-n}\sum_{j=0}^
    {2n} \omega^j= \frac{\omega^{-n}(1 - \omega^{2n+1})}{1 - \omega} \tag
    {2}\label{inner}


using the geometric series. Before moving on, let's visualize what these inner
products look like for :math:`n = 3`. The expression(:math:`\ref{inner}`) is
a sum of phases, which we can add top-to-tail on the complex plane. We've
added a big :math:`\mathbb{C}` to distinguish this from other planes we've
been looking at.


.. tip::

    We display the inner product :math:`\langle T_s, T_t\rangle` below.
    Horizontal mouse position controls :math:`s`. The choice of :math:`t`
    is "quantized" to the random shifts from above; click to the left to set
    it. The phases summed are light magenta, and the total is dark magenta.

.. raw:: html

    <figure>
        <center>
        <div id="sketch14_5" style></div>
        </center>
    </figure>

As expected, our random shifts are not orthogonal. But with :math:`(\ref
{inner})` in hand, it's easy to choose orthogonal vectors! We simply select
our shifts :math:`s_j` so that the numerator of :math:`(\ref
{inner})` vanishes:

.. math::

    \omega^{2n + 1} = e^{2\pi i \gamma(2n+1)(s_k - s_j)} = 1.

This occurs if :math:`\gamma(2n + 1)(s_k - s_j)` is always an integer. A
natural choice is

.. math::

    s_k = \frac{k}{\gamma(2n+1)},

for :math:`k = 1, \ldots, 2n`. In this case, we have the orthogonality
relation

.. math::

    \langle T_{s_k}, T_{s_k}\rangle = (2n+1)\delta_{jk}.

Thus, spacing shifts equally around the :math:`s` circle gives us an
orthogonal set of shifts. We picture these equally spaced shifts, and check
visually they are orthogonal, for :math:`n=3` below. Select any of the
equally spaced points, and you can see that its inner product with another of
the equally spaced points vanishes.

.. tip::

    Horizontal mouse position controls :math:`s`. 

.. raw:: html

    <figure>
        <center>
        <div id="sketch15" style></div>
        </center>
    </figure>

Orthogonality makes finding the coefficients :math:`\beta_k` easy: we simply
take the inner product between :math:`T_{s_k}` and the left-hand side
of :math:`(\ref{GPS})`, expressed in terms of the differentiation
matrix :math:`D`. This gives

.. math::

    \begin{align*}
    \beta_k & = \frac{2i\gamma}{2n+1}\sum_{j=-n}^n j \omega_k^j
    \end{align*}

for :math:`\omega_k = e^{-2\pi i k/(2n+1)}`. This looks tricky, but we can
start with a geometric series, *differentiate* with respect
to :math:`\omega`, and multiply by :math:`\omega_k`, to we get what we want:

.. math::

    \omega_k \partial_{\omega_k} \sum_{j=-n}^n \omega_k^j = \sum_
     {j=-n}^n j\omega_k^j.

We already computed the geometric series in
:math:`(\ref{inner})`. Plugging that back in, differentiating, and using the
fact that :math:`\omega_k^{2n+1} = 1`, we finally get

.. math::

    \begin{align*}
    \beta_k & =
    \frac{2i\gamma}{2n+1}\cdot\frac{\omega_k^{-n}(2n+1)(\omega_k -
    1)}{(\omega_k - 1)^2} = \frac{2 i\gamma \omega_k^{-n}}{\omega_k-1}.
    \end{align*}


Putting these together with our shifts, we have our general parameter-shift
rule:

.. math::

    f'(\theta) = \sum_{k=1}^{2n}\frac{2 i\gamma \omega_k^{-n}}
    {\omega_k-1}f\left(\theta + \frac{k}{\gamma(2n+1)}\right). \tag{3}\label
    {GPS+}


The approach outlined here only works when the frequencies in the
problem are evenly spaced. However, there are ways to generalize
further. Even without orthogonality, we can find
independent shifts and solve the linear algebra problem
:math:`(\ref{GPS})` for the coefficients. Alternatively, we can use
randomization to obtain shifts which are orthogonal on average, leading to
the *stochastic parameter-shift rule*.

About the author
----------------

.. include:: ../_static/demos/authors/david_wakeham.txt
"""
