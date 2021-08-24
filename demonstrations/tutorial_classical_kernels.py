r"""

.. _classical_kernels:

Emulating classical kernels
============================

.. meta::
    :property="og:description": Approximating the Gaussian kernel with quantum circuits.
    :property="og:image": https://pennylane.ai/qml/_images/toy_qek.png

.. related::

   tutorial_kernels_module Training and evaluating quantum kernels
   tutorial_kernel_based_training Kernel-based training of quantum models with
   scikit-learn
   tutorial_expressivity_fourier_series Quantum models as Fourier series


*Author: Elies Gil-Fuster (Xanadu resident). Posted: DD MMMM 2021.*

In this demo we will briefly revisit some notions of kernel-based Machine
Learning (ML) and introduce one very specific `Quantum Embedding Kernel (QEK)
<https://pennylane.ai/qml/demos/tutorial_kernels_module.html>`_.
We will use this QEK to demonstrate we can approximate an important class
of classical kernels with it, known as shift-invariant or stationary kernels.
Both the math and the code will stay high-level throughout the explanation,
this is not one of those very technical demos!
So, if you feel like exploring one link between classical and quantum kernels,
this is the place for you!

|

.. figure:: ../demonstrations/classical_kernels/sketch.PNG
    :align: center
    :width: 80%
    :target: javascript:void(0)

    The Quantum Embedding Kernel covered in this demo.



Kernel-based Machine Learning
----------------------------------------------

As we just said, in the interest of keeping the concepts at a high level we
will not be reviewing all the notions of kernels in-depth here.
Instead, we only need to know that there's an entire branch of ML which
revolves around some functions we call kernels.
If you'd still like to know more about where these functions come from, why
they're important, and how we can use them (e.g. with PennyLane), luckily there
are already two very nice demos that cover different aspects extensively:

#. `Training and evaluating quantum kernels <https://pennylane.ai/qml/demos/tutorial_kernels_module.html>`_
#. `Kernel-based training of quantum models with scikit-learn <https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html>`_

Now, for the puyrpose of this demo, a kernel is a real-valued function of two
variables :math:`k(\cdot,\cdot)` from a given data domain :math:`x_1,
x_2\in\mathcal{X}`. Further, we require a kernel to be symmetric to exchanging
the variable positions :math:`k(x_1,x_2) = k(x_2,x_1)`.
Finally, we will also want to enforce the kernels be positive semi-definite,
but let's avoid getting lost in mathematical definitions, you can trust that
all kernels featuring in this demo are positive semi-definite.

The Gaussian kernel
^^^^^^^^^^^^^^^^^^^^

If you take whichever textbook on kernel methods and search for the word
"prominent", chances are you'll find it next to the word "example" in a
sentence that introduces the so-called Gaussian (or Radial Basis Function)
kernel :math:`k_\sigma`.
For the sake of simplicity, we assume we are dealing with real numbers
:math:`\mathcal{X}\subseteq\mathbb{R}`, in which case the Gaussian kernel looks
like

.. math:: k_\sigma(x_1, x_2) = e^{-\frac{\lvert x_1 - x_2\rvert^2}{\sigma}},

where the variance :math:`\sigma` is a positive real tunable parameter.
The generalization to higher-dimensional data is straightforward using the
Euclidean norm :math:`\lVert x_1 - x_2 \rVert_2^2`.)
Now for practical purposes the Gaussian kernel has the advantage of being
simple enough to study as a function, while yielding good performance for
a wide range of real-life tasks.

Shift-invariant kernels
^^^^^^^^^^^^^^^^^^^^^^^^

In particular, one of the properties of the Gaussian kernel is that it is a
shift-invariant (also called *stationary*) function.
That means that adding a constant shift to both arguments does not change the
value of the kernel, that is for :math:`a\in\mathcal{X}`, we have
:math:`k_\sigma(x_1 + a, x_2 + a) = k(x_1, x_2)`.
At the same time, it also means we can express the kernel as a function of only
one variable, the so-called lag (or shift) :math:`\delta = x_1 -
x_2\in\mathbb{R}`, where we have

.. math:: k_\sigma(x_1, x_2) = e^{-\frac{\lvert x_1 - x_2 \rvert^2}{\sigma}} =e^{-\frac{\delta^2}{\sigma}} = k_\sigma(\delta).

Combined with the property :math:`k(x_1, x_2) = k(x_2, x_1)`, this results
in the new property :math:`k(\delta)=k(-\delta)`.

Of course the Gaussian kernel is not the only shift-invariant kernel out there.
As it turns out, there are many others [#Rasmussen]_ [#Scholkopf]_
which are also used in practice and have the shift-invariance property.
Nevertheless, here we will only look at the simple Gaussian kernel with
:math:`\sigma = 1`:

.. math:: k_1(\delta) = e^{-\delta^2},

but all the arguments and code we use are also amenable to other kernels which
fulfill the following mild restrictions:

#. Shift-invariance.
#. Normalization :math:`k(0)=1`.
#. Smoothness (seen as quickly decaying Fourier spectrum).

Implementation example
^^^^^^^^^^^^^^^^^^^^^^^^

Let's warm up by implementing a classical Gaussian kernel!

First, importers gonna import B-).
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import math


np.random.seed(53173)

###############################################################################
# Now we define the kernel function and plot it as a mild sanity check:

def gaussian_kernel(delta):
    return math.exp(-delta ** 2)

def make_data(n_samples, lower=-np.pi, higher=np.pi):
    X = np.linspace(lower, higher, n_samples)
    Y = np.array([gaussian_kernel(x) for x in X])

    return X, Y

X, Y_gaussian = make_data(100)

plt.plot(X, Y_gaussian)
plt.suptitle("The gaussian kernel with $\sigma=1$")
plt.xlabel("$\delta$")
plt.ylabel("$k(\delta)$")
plt.show();

###############################################################################
# That's exactly what I have in mind when I think of a Gauss bell curve.
# So far so good!
#
# Quantum Embedding Kernels
# --------------------------
#
# The quantum algorithms we build today belong to a family of kernels dubbed
# Quantum Embedding Kernels (QEKs) in [#QEK]_ (after others had proposed the
# ideas of quantum kernels), corresponding to the `Training and evaluating
# quantum kernels <https://pennylane.ai/qml/kernels_module.html>`_ demo.
# In a nutshell, a QEK can be thought of as a function estimated with a quantum
# computer in which, by doubling the size of the circuit, we spare ourselves
# the trouble of finding a good measurement observable.
# Indeed, for a given feature map :math:`\rho:\mathcal{X}\to\mathcal{H}_d`,
# where :math:`\mathcal{H}_d` is the Hilbert space of a quantum system of
# :math:`d` dimensions (also called a qu-:math:`d`-it), we have the associated
# QEK:
#
# .. math:: k_Q(x_1, x_2) = \operatorname{tr}[\rho(x_1)\rho^\dagger(x_2)].
#
# .. Note::
#
#       Working with one qu-:math:`d`-it instead of with several qubits is a
#       matter of taste. Indeed, since a qu-:math:`d`-it is "a quantum system
#       with :math:`d` levels", if we would rather think of qubits, we can just
#       set :math:`d` to be some integer power of :math:`2`, :math:`d=2^n`.
#       Then, everything we say or do on the qu-:math:`2^n`-it is equivalent
#       and portable to a system of :math:`n` qubits.
#       We will see all of this when it comes to implementing the algorithm on
#       a quantum computer, where we are forced to work with qubits.
#
# Now, oftentimes the feature map will be nothing other than applying a unitary
# gate depending on the data :math:`U(x)` to the ground state of the
# qu-:math`d`-it:
# 
# .. math:: \rho(x) = U(x)\vert0\rangle\!\langle0\vert U^\dagger(x).
#
# In this very natural case, the QEK reduces to an expression which maybe looks
# more familiar to some
#
# .. math:: k_Q(x_1, x_2) = \lvert\langle0\vert U^\dagger(x_1)U(x_2)\vert0\rangle\rvert^2.
#
# And this is the bare bone of what we call Quantum Embedding Kernels!
#
# By now you may have already realized QEKs are very cool and all but, if our
# goal is to emulate the Gaussian kernel on a quantum computer (which is what
# we're trying to do here), they have
# one shortcoming: In general, :math:`k_Q` won't be a shift-invariant function.
# The key word here is *general*.
# General QEKs are not shift-invariant, so one reasonable question would be:
# which restrictions do we have to impose for these kernels to be
# shift-invariant?
# We are not going to fully answer this question here, but rather take one
# first step towards it.
# Namely, we next introduce one particular class of QEKs which are indeed
# shift-invariant.
# Behold, ladies and gents, *the stationary toy QEK*!
#
# The Stationary toy QEK 
# -----------------------
#
# The stationary toy QEK can be estimated with a qu-:math:`d`-it based algorithm
# with only three gates.
# The first and last gates are adjoint of one another, we name them :math:`W`
# and :math:`W^\dagger` respectively.
# :math:`W` is a trainable :math:`d`-dimensional unitary, independent of the
# input data.
#
# To make things simple, we need a Hamiltonian with consecutive integer numbers
# on its diagonal, and zeroes everywhere else
#
# .. math::`H_d=\operatorname{diag}(0, 1, \ldots, d-1)`;
#
# this is known in continuous-variable quantum computing as a number operator.
# The data is then encoded as the time parameter in the time evolution of this
# Hamiltonian.
# We can write this as unitary gate :math:`S(x)=e^{-ixH_d}`.
# Circuitwise, :math:`S(x)` is a diagonal unitary whose entries are
# :math:`e^{-ijx}` for :math:`j\in\{0, \ldots, d-1\}`
#
# .. math:: S(x) = \operatorname{diag}(0, e^{-ix}, e^{-i2x}, \ldots, e^{-i(d-1)x}).
#
# Despite this construction looking more or less arbitrary, having this
# consecutive integer Hamiltonian simplifies the analysis of approximating
# functions under-the-hood.
#
# .. figure:: ../demonstrations/classical_kernels/toy_qek.png
#       :align: center
#       :width: 60%
#       :target: javascript:void(0)
#
# A priori we can think of :math:`W` as *any* :math:`d`-dimensional unitary.
# Only when it comes to actually implementing the stationary toy QEK in a
# computer will we have to think about how to parametrize it.
#
# Owing to the study presented in [#Fourier]_ (further illustrated in `this
# demo
# <https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html>`_),
# we know that many quantum kernels, ours included, can be expressed as Fourier
# series of only a few terms.
# For a study of the Fourier representation of quantum kernels specifically,
# do check [#qkernels]_ out!
# (In turn, `this demo
# <https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html>`_ is
# based on that reference).
# Using a similar formalism to those references we reach an expression for the
# stationary toy QEK kernel :math:`k_d`:
#
# .. math:: k_d(\delta) = \sum_{s=-(d-1)}^{d-1} a_s e^{i\delta s},
#
# where
#
# .. math:: a_s = \sum_{j=\lvert s\rvert}^{d-1} w_j w_{j-\lvert s\rvert},
#
# and where :math:`w_j = \lvert W_{j,0}\rvert^2` is the absolute value squared
# of the :math:`j^\text{th}` element of the first column of the matrix
# representation of :math:`W`.
#
# 
#
# Implementing the stationary toy QEK on PennyLane
# -------------------------------------------------
#
# Now that we've laid out the formulas, we only need to write down the PL code
# that realizes the quantum circuit, *right?*
#
# *Wrong!!*
#
# *But, kind of right.*
#
# We do only have to realize the quantum circuit, but as it turns out, we can
# only operate on qubits, not qu-:math:`d`-its!
# So, what can we do?
# We can set :math:`d=2^n` for some :math:`n\in\mathbb{N}` and then use
# :math:`n` qubits, which are analogous to one qu-:math:`2^n`-it.
#
# This is the point where we will have to fix an Ansatz for :math:`W`, and
# since we want to work with qubits we shall use a few trainable Pauli
# rotations interleaved with also trainable entangling gates so we can get
# reasonably close to an arbitrary unitary.
#
# But what about :math:`S(x)`?
# Conceptually, the definition we've given with the number operator hamiltonian
# is already quite simple, but how do we implement it on a qubit based system?
#
# The fact is we can replicate the eigenvalue spectrum of :math:`S` with a
# layer of Pauli-Z rotations, one on each qubit
# 
# .. math:: R_{Z_1}(\vartheta_1 x)\otimes R_{Z_2}(\vartheta_2x)\otimes\cdots\otimes R_{Z_m}(\vartheta_n x) = \operatorname{diag}(e^{-i\lambda_1x}, \ldots, e^{-i\lambda_{2^n}x}).
#
# In particular, we want to find :math:`\vartheta_1, \ldots, \vartheta_n` such
# that for all :math:`j` the eigenvalues fulfill :math:`\lambda_{j+1} -
# \lambda_j = 1`.
# This might seem a bit strange, but since multiplication times a global phase
# :math:`e^{i\alpha}` has no physical effect, if we find the combination of
# parameters that yields :math:`\lambda_j = \lambda_1 + (j - 1)`, then we can
# multiply everything with :math:`e^{-i\lambda_1x}` to obtain the desired
# spectrum :math:`(0, 1, \ldots, 2^n-1)`.
#
# We will also be spared of some details here (or, as your quantum mechanics
# Prof would say, the derivation is left as an exercise for the reader), but
# one valid choice for this to happen is :math:`\vartheta_j = -2^{j-1}`.
#
# With this, we can *finally* start getting our hands dirty pushing towards our
# ultimate goal: using the stationary toy QEK to approximate the classical
# Gaussian kernel.
#
# Writing down the circuit
# -------------------------
#
# The stationary toy QEK is defined irrespective of the qu-:math:`d`-it
# dimension.
# We can directly define :math:`S(x)`, where we include a ``thetas`` argument
# in case at a later stage we want to try encoding with different diagonal
# Hamiltonians:

def S(x, thetas, wires):
    for (i, wire) in enumerate(wires):
        qml.RZ(thetas[i] * x, wires = [wire])

###############################################################################
# For :math:`W` we use a few layers of single qubit Pauli rotations and two
# qubit entangling gates, all trainable:

def W(parameters, wires):
    # 1st layer: trainable Pauli X rotations
    for (i, wire) in enumerate(wires):
        qml.RX(parameters[0][i], wires = [wire])

    # 2nd layer: ring of controlled Pauli X rotations
    qml.broadcast(unitary=qml.CRX, pattern="ring", wires=wires,
                  parameters=parameters[1])

    # 3rd layer: trainable Pauli X rotations
    for (i, wire) in enumerate(wires):
        qml.RX(parameters[2][i], wires=[wire])

    # 4th layer: ring of controlled Pauli X rotations
    qml.broadcast(unitary=qml.CRX, pattern="ring", wires=wires,
                  parameters=parameters[3])

###############################################################################
# Next we can define the function that implements the entire circuit
# :math:`WS(x)W^\dagger`, where as already anticipated we feed the rotation
# angles into the algorithm with the ``parameters`` variable:

def ansatz(x1, x2, thetas, parameters, wires):
    W(parameters, wires)
    S(x1-x2, thetas, wires)
    qml.adjoint(W)(parameters, wires)

###############################################################################
# And we also provide a function that generates a random tensor of parameters
# with the correct size.
# For now we'll be happy with uniformly distributed parameters in the interval
# :math:`[0,2\pi)`.

def random_parameters(n_wires):
    return np.random.uniform(0, 2*np.pi, (4, n_wires))

###############################################################################
# And finally we also define a function that gives us the vector of parameters
# we need to feed to :math:`S(x)`:

def make_thetas(n_wires):
    return [-2**i for i in range(n_wires)]

###############################################################################
# Computing the stationary toy QEK on the quantum computer
# ---------------------------------------------------------
#
# That is to say, on the ``default.qubit`` PL quantum simulator ;).
#
# At this point we need to fix the number of qubits.
# For the present example it suffices to set :math:`n=5`, which yields
# :math:`d=2^5=32` dimensions.

n_wires = 5

dev = qml.device("default.qubit", wires=n_wires, shots=None)
wires = dev.wires.tolist()

###############################################################################
# Next we define the circuit function, which outputs the vector of
# probabilities for the computational basis states.
# This is also the point where we need the vector of ``thetas``:

thetas = make_thetas(n_wires)

@qml.qnode(dev)
def toy_qek_circuit(x1, x2, thetas, parameters):
    ansatz(x1, x2, thetas, parameters, wires=wires)
    return qml.probs(wires=wires)

###############################################################################
# The output of ``toy_qek_circuit` is an array of real numbers.
# In particular, each entry of this array contains the probability of obtaining
# each computational basis state at the end of the circuit.
# This is because we didn't specify any observable to measure on, which is what
# we do with a number of quantum algorithms, e.g.
# `VQE <https://pennylane.ai/qml/demos/tutorial_vqe.html>.
# But we only need to keep the probability of the state
# :math:`\vert00000\rangle` being measured, so we take the first entry of the
# array of probabilities and discard everything else.

def toy_qek(x1, x2, thetas, parameters):
    return toy_qek_circuit(x1, x2, thetas, parameters)[0]

###############################################################################
# We can do now a small test with random parameters:

random_pars = random_parameters(n_wires)
print("The stationary toy QEK with parameters:")
print(random_pars)
print("for x1 = 0.1 and x2 = 0.6 is\n\tk(x1, x2) =", toy_qek(.1, .6, thetas, random_pars))

###############################################################################
# Quality of life improvement: let's make a small function that evaluates
# the stationary toy QEK on an array of shifts :math:`\delta`.
# Notice that the argument of this function ought to be thought of as
# :math:`\delta = x_1 - x_2`, that's why the function ``toy_qek`` is called
# with :math:`0` on the second argument, because :math:`S(x_1)S^\dagger(x_2) =
# S(x_1-x_2)S^\dagger(0) = S(x_1-x_2)`.

def toy_qek_on_dataset(deltas, thetas, parameters):
    y = np.array([toy_qek(delta, 0, thetas, parameters) for delta in deltas])
    return y

###############################################################################
# Let's see what this function looks like for the same data interval as before:

Y_toy = toy_qek_on_dataset(X, thetas, random_pars)

plt.plot(X, Y_toy)
plt.suptitle("Stationary toy QEK with random parameters")
plt.xlabel("$\delta$")
plt.ylabel("$k_d(\delta)$")
plt.show();

###############################################################################
# Granted, it does not look like the Gaussian kernel at all, but at least as a
# sanity check we see it fulfills :math:`k_d(0) = 1` and :math:`k_d(\delta) =
# k_d(-\delta)`.
#
# Approximating the Gaussian kernel with the toy stationary QEK
# --------------------------------------------------------------
#
# Next step is to tune the parameters of :math:`W` to make the stationary toy
# QEK become closer to the Gaussian kernel.
# We lay this out as a supervised learning problem, where we define the loss as
# the pointwise distance
#
# .. math:: 
#     
#     L = \frac{1}{2N} \sum_{n=1}^N \lvert k_1(\delta_n0-k_d(\delta_n)\rvert^2,
#
# where :math:`k_1` is the Gaussian kernel with :math:`\sigma=1`, :math:`k_d`
# is the :math:`d`-dimensional stationary toy QEK, and :math:`\{\delta_1,
# \ldots, \delta_n\}` is the dataset ``X`` in the code.

def square_loss(targets, predictions):
    loss = 0
    for t, p in zip(targets, predictions):
        loss += (t - p)**2
    loss = loss / len(targets)
    return .5*loss

def cost(parameters, thetas, X, Y):
    predictions = toy_qek_on_dataset(X, thetas, parameters)
    return square_loss(Y, predictions)

###############################################################################
# We'll use the Adam Optimizer, with the following optimnization
# hyperparameters:

max_steps = 100
opt = qml.AdamOptimizer(0.2)
batch_size = 5

###############################################################################
# And next we only need to hit play!

cst = [cost(random_pars, thetas, X, Y_gaussian)]

for step in range(max_steps):

    batch_index = np.random.randint(0, len(X), (batch_size,))
    x_batch = X[batch_index]
    y_batch = Y_gaussian[batch_index]

    random_pars = opt.step(lambda p: cost(p, thetas, x_batch, y_batch),
                           random_pars)

    c = cost(random_pars, thetas, X, Y_gaussian)
    cst.append(c)
    if (step+1)%10 == 0:
        print("Cost at step {0:3}: {1}".format(step+1, c))

trained_pars = random_pars

###############################################################################
# After the optimization, we plot one last time the values of the stationary
# toy QEK, this time superposed to those of the Gaussian one:

Y_trained = toy_qek_on_dataset(X, thetas, trained_pars)

plt.plot(X, Y_gaussian, label='Gaussian kernel')
plt.plot(X, Y_trained, label='Stationary toy QEK')
plt.suptitle("Comparison between Gaussian and stationary toy QEK")
plt.xlabel("$\delta$")
plt.legend()
plt.show();

###############################################################################
# *et voilà!*
#
# This was how you can approximate the Gaussian kernel using a stationary toy
# QEK!
#
# .. figure:: ../demonstrations/classical_kernels/salesman.PNG
#       :align: center
#       :width: 70%
#       :target: javascript:void(0)
#
# References
# -----------
#
# .. [#Rasmussen]
#
#       Carl Edward Rasmussen, Christopher K. I. Williams.
#       `"Gaussian Processes for Machine Learning" <gaussianprocess.org/qpml/chapters>`__.
#       MIT Press, 2006.
#
# .. [#Scholkopf]
#
#       Bernhard Schölkopf, Alexander J. Smola.
#       `"Learning with Kernels" <mitpress.mit.edu/books/learning-kernels>`__.
#       MIT Press, 2001.
#
# .. [#QEK]
#
#       Thomas Hubregtsen, David Wierichs, Elies Gil-Fuster, Peter-Jan HS
#       Derks, Paul K Faehrmann, Johannes Jakob Meyer.
#       "Training Quantum Embedding Kernels on Near-Term Quantum Computers".
#       `arXiv preprint arXiv:2105.02276 <https://arxiv.org/abs/2105.02276>`__.
#
# .. [#Fourier]
#
#       Maria Schuld, Ryan Sweke, Johannes Jakob Meyer.
#       "The effect of data encoding on the expressive power of variational
#       quantum machine learning models".
#       `Phys. Rev. A 103, 032430 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.032430>`__,
#       `arXiv preprint arXiv:2008.08605 <https://arxiv.org/abs/2008.08605>`__.
#
# .. [#qkernels]
#
#       Maria Schuld.
#       "Supervised quantum machine learning models are kernel methods".
#       `arXiv preprint arXiv:2101.11020 <https://arxiv.org/abs/2101.11020>`__.
