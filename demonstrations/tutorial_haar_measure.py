r""".. role:: html(raw)
   :format: html

Understanding the Haar Measure
==============================

.. meta::
    :property="og:description": Using light to perform tasks beyond the reach of classical computers.

    :property="og:image": https://pennylane.ai/qml/_images/tutorial_haar_thumb.png

.. related::

    quantum_volume Quantum volume
    qsim_beyond_classical Beyond classical computing with qsim


*Author: PennyLane dev team. Posted: 4 March 2021. Last updated: 4 March 2021.*

If you've ever dug into the literature about random quantum circuits,
variational ansatz structure, or anything related to the structure and
properties of unitary operations, you've likely come across a statement like the
following: "Assume that :math:`U` is sampled uniformly at random from the Haar
measure".  In this demo, we're going to unravel this cryptic statement and take
an in-depth look at what it means. You'll gain an understanding of the general
concept of *measure*, the Haar measure and its special properties, and you'll
learn how to sample from it using the tools available in PennyLane and other
scientific computing frameworks. By the end of this demo, you'll be able to
include that important statement in your own work with confidence!

.. note::

   To get the most out of this demo, it is helpful if you are familiar with
   integration of multi-dimensional functions, the Bloch sphere, and the
   conceptual idea behind factorizations and different parametrizations of
   unitary matrices. [TODO: add some links]

Measure
-------

`Measure theory <https://en.wikipedia.org/wiki/Measure_(mathematics)>`__ is a
branch of mathematics that studies things that are measurable --- think length,
area, or volume, but generalized to mathematical spaces and even higher
dimensions. Loosely, the measure tells you about how "stuff" is distributed and
concentrated in a mathematical set or space. An intuitive way to understand
measure is to think about a sphere. An arbitrary point on a sphere can be
parameterized by three numbers --- depending on what you're doing, you may use
Cartesian coordinates :math:`(x, y, z)`, or it may be more convenient to use
spherical co-ordinates :math:`(\rho, \phi, \theta)`.

.. image:: /demonstrations/haar_measure/spherical_coords.png
    :align: center
    :width: 90%

    The size of an elementary volume element with fixed angular difference may
    be smaller or larger depending on its position on the sphere.

[TODO: simple graphic of spherical coordinates (?)]

Suppose you wanted to compute the volume of a sphere with radius :math:`r`. Just
like you can compute the area under the curve of a function by integrating over
its parameters, you can compute the volume of a sphere by integrating over
:math:`\rho, \phi`, and :math:`\theta`. Your first thought when taking an integral
might be to just sandwich a function between a :math:`\int` and a :math:`dx`,
where :math:`x` is some parameter your function depends on. Let's see what
happens if we do that with the spherical coordinates:

.. math::

    V = \int_0^{r} \int_0^{2\pi} \int_0^{\pi} d\rho d\phi d\theta = 2\pi^2 r

Now, we know that the volume of a sphere of radius :math:`r` is
:math:`\frac{4}{3}\pi r^3`, so what we got from this integral is clearly wrong!
Taking the integral naively like this doesn't take into account the structure of
the sphere with respect to its parameters. For example, consider two small,
infinitesimal elements of volume with the same difference in :math:`\theta`, but
at different parts of the sphere:

.. image:: /demonstrations/haar_measure/spherical_int_dtheta.png
    :align: center
    :width: 50%

Even though the :math:`d\theta` themselves are the same, there is way more
"stuff" near the equator of the sphere than there is near the poles. We are
going to need to take that into account when computing the integral. The same
holds true if we consider a :math:`d\rho`.  The contribution to volume of parts of
the sphere with a large :math:`\rho` is far more than for a small :math:`\rho` --- we should
expect the contribution to be proportional to :math:`\rho^2`, given that
the surface area of a sphere of radius :math:`r` is :math:`4\pi r^2`.

.. image:: /demonstrations/haar_measure/spherical_int_dr.png
    :align: center
    :width: 90%

On the other hand, for a fixed :math:`\rho` and :math:`\theta`, the length of the
:math:`d\phi` is the same all around the circle. If we did the math and put all these
things together, we would find that the actual expression for the integral
should look like this:


.. math::

    V = \int_0^r \int_0^{2\pi} \int_0^{\pi} \rho^2 \sin \theta d\rho d\phi d\theta = \frac{4}{3}\pi r^3

These extra terms that we had to add to the integral, :math:`\rho^2 \sin \theta`,
constitute the *measure*. The measure weights portions of the sphere differently
depending on where they are in the space.

The Haar measure
----------------

Like points on a sphere, unitary
matrices can be expressed in terms of a fixed set of parameters. For example,
the most general single-qubit rotation implemented in PennyLane
(:class:`~.pennylane.Rot`) is expressed as

.. math::

    U(\phi, \theta, \omega) = \begin{pmatrix} e^{-i(\phi + \omega)/2}
                        \cos(\theta/2) & -e^{i(\phi - \omega)/2} \sin(\theta/2)
                        \\ e^{-i(\phi - \omega)/2} \sin(\theta/2) & e^{i(\phi +
                        \omega)/2} \cos(\theta/2). \end{pmatrix}

The *Haar measure* tells us about how to sample these parameters in order to
sample unitaries uniformly at random from the unitary group :math:`U(N)`. This
measure, often denoted by :math:`\mu_N`, is what sits inside integrals over the
unitary group and ensures things are properly weighted. For example, suppose
:math:`f` is a function that acts on elements of :math:`U(N)`. We can write the
integral of :math:`f` with respect to the Haar measure like so:

.. math::

    \int_{V \in U(N)} f(V) d\mu_N(V).

The :math:`d\mu_N` itself can be broken down into components depending on each of
the parameters individually.  As such, while the Haar measure can be defined for
every dimension :math:`N`, the mathematical form can get quite hairy for larger
dimensions.  In general, a :math:`N`-dimensional unitary requires at least
:math:`N^2 - 1` parameters, which is a lot of stuff to keep track of! Therefore
we'll start with the case of a single qubit :math:`(N=2)`, then show how things
generalize.

Single-qubit Haar measure
~~~~~~~~~~~~~~~~~~~~~~~~~

The Haar measure allows us to sample single-qubit unitaries uniformly at
random. One useful consequence of this is that it provides a method to sample
quantum *states* uniformly at random from the Hilbert space.  We can simply
generate Haar-random unitaries, and apply them to a fixed basis state such as
:math:`\vert 0\rangle`.

Keeping with the idea of spheres, let's try this out and investigate what
happens to a single-qubit state on the Bloch sphere when we apply Haar-random
unitaries. But first, let's suppose that we sample parameters for our unitaries
from the uniform distribution of each parameter - the angles :math:`\omega,
\phi`, and :math:`\theta` are all sampled from uniformly from the range
:math:`[0, 2\pi)`.

"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# set the random seed
np.random.seed(42)

# Use the mixed state simulator to save some steps in plotting later
dev = qml.device('default.mixed', wires=1)

@qml.qnode(dev)
def not_a_haar_random_unitary():
    # Sample all parameters uniformly 
    phi, theta, omega = 2 * np.pi * np.random.uniform(size=3)
    qml.Rot(phi, theta, omega, wires=0)
    return qml.state()

num_samples = 2021

not_haar_samples = [not_a_haar_random_unitary() for _ in range(num_samples)]

######################################################################
# In order to plot these on the Bloch sphere, we'll need to do one more
# step, and convert the quantum states into Bloch vectors.
#

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Used the mixed state simulator so we could have the density matrix for this part!
def convert_to_bloch_vector(rho):
    """Convert a density matrix to a Bloch vector."""
    ax = np.trace(np.dot(rho, X)).real
    ay = np.trace(np.dot(rho, Y)).real
    az = np.trace(np.dot(rho, Z)).real
    return [ax, ay, az]

not_haar_bloch_vectors = np.array([convert_to_bloch_vector(s) for s in not_haar_samples])

######################################################################
# With this done, let's find out where our random states ended up:

def plot_bloch_sphere(bloch_vectors):
    """ Helper function to plot vectors on a sphere."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    ax.scatter(
        bloch_vectors[:,0], bloch_vectors[:,1], bloch_vectors[:, 2], c='#e29d9e', alpha=0.3
    )
    plt.show()

plot_bloch_sphere(not_haar_bloch_vectors)

######################################################################
# You can see from this plot that even though our parameters were sampled from a
# uniform distribution, there is some noticeable concentration around the poles
# of the sphere. Despite the input parameters being uniform, the output is very
# much *not* uniform. Just like the regular sphere, the measure is larger near
# the equator, and if we just sample uniformly, we won't end up populating that
# area as much. To take that into account we will need to sample from the proper
# Haar measure, and weight the different parameters appropriately.
#
# For a single qubit, the Haar measure looks much like the case of a sphere,
# minus the radial component (intuitively, all qubit state vectors have length
# 1, so it makes sense that this wouldn't play a role here). The parameter that
# we will have to weight differently is :math:`\theta`, and in fact the
# adjustment in measure is identical to that we had to do with the polar axis of
# the sphere, i.e., :math:`\sin \theta`. In order to sample the :math:`\theta`
# uniformly at random in this context, we must sample from the distribution
# :math:`\hbox{Pr}(\theta) = \sin \theta`. We can accomplish this by setting up
# a custom probability distribution with 
# `rv_continuous <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`__
# in ``scipy``.

from scipy.stats import rv_continuous

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)

# Samples of theta should be drawn from between 0 and pi
sin_sampler = sin_prob_dist(a=0, b=np.pi)

@qml.qnode(dev)
def haar_random_unitary():
    phi, omega = 2 * np.pi * np.random.uniform(size=2) # Sample phi and omega as normal
    theta = sin_sampler.rvs(size=1) # Sample theta from our new distribution
    qml.Rot(phi, theta, omega, wires=0)
    return qml.state()

haar_samples = [haar_random_unitary() for _ in range(num_samples)]
haar_bloch_vectors = np.array([convert_to_bloch_vector(s) for s in haar_samples])

plot_bloch_sphere(haar_bloch_vectors)

######################################################################
# We see that when we use the correct measure, our qubit states are now
# much-better distributed over the sphere. Putting this information together,
# we can now write the explicit form for the single-qubit Haar measure:
#
# .. math::
#
#    d\mu_2 = \sin \theta d\theta \cdot d\omega \cdot d\phi.
#
#


######################################################################
# Show me more math!
# ~~~~~~~~~~~~~~~~~~
#
# While we can easily visualize the single-qubit case, this is no longer
# possible when we increase the number of qubits. Regardless, we can still
# obtain a mathematical expression for the Haar measure in arbitrary
# dimensions. In what follows, we will leave qubits behind and explore the many
# parametrizations of the :math:`N`-dimensional `special unitary group
# <https://en.wikipedia.org/wiki/Special_unitary_group>`__. This group,
# written as :math:`SU(N)`, is the continuous group consisting of all :math:`N
# \times N` unitary operations with determinant 1.
#
# Instead of qubits, we are going to shift to the world of *qumodes*. Our
# unitaries will be expressed as interferometers made up of beamsplitters (2- or
# 3-parameter operations), and phase shifts (1-parameter operations). These
# unitaries can still be considered as multi-qubit operations in the cases where
# :math:`N` is a power of 2, but they will have to be translated from the
# continuous-variable operations into qubit operations. In PennyLane, this can
# be done by feeding the unitaries to the :class:`~.pennylane.QubitUnitary`
# operation directly. Alternatively, one can use *quantum compilation* to
# express the operations as a sequence of elementary gates such as Pauli
# rotations and CNOTs.
#
# .. admonition:: Tip
#
#    If you haven't had many opportunities to work in terms of qumodes, check out
#    [insert resource here] for a nice introduction to the topic.
#
# There are multiple ways to parameterize a :math:`N`-dimensional unitary
# operation. We saw already above that for :math:`N=2`, we can write
#
# .. math::
#
#    U(\phi, \theta, \omega) = \begin{pmatrix} e^{-i(\phi + \omega)/2}
#                        \cos(\theta/2) & -e^{i(\phi - \omega)/2} \sin(\theta/2)
#                        \\ e^{-i(\phi - \omega)/2} \sin(\theta/2) & e^{i(\phi +
#                        \omega)/2} \cos(\theta/2) \end{pmatrix}
#
# This operation can be decomposed into beamsplitters and phase shifts
# as follows:
#
# .. math::
#
#    U(\phi, \theta, \omega) =
#        \begin{pmatrix}
#          e^{i\phi/2} & 0 \\ 0 & e^{-i\phi/2}
#        \end{pmatrix}
#        \begin{pmatrix}
#          \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2)
#        \end{pmatrix}
#       \begin{pmatrix}
#          e^{i\omega/2} & 0 \\ 0 & e^{-i\omega/2}
#        \end{pmatrix}
#
# [TODO: Verify signs and angles since the decomposition was pulled from the paper].
#
# The middle operation is a beamsplitter; the other two operations are phase
# shifts.  We saw in the previous section that for :math:`N=2`, :math:`d\mu_2 =
# \sin\theta d\theta d\omega d\phi` -- note how the contribution of the
# parameter in beamsplitter contributes to the measure in a different way than
# that of the two phase shifts. As mentioned above, for larger values of
# :math:`N` there are multiple ways to decompose the unitary. Such
# decompositions rewrite elements in :math:`SU(N)` acting on :math:`N` modes as
# a sequence of operations acting only on 2 modes, :math:`SU(2)`, and
# single-mode phase shifts.  Shown below are three examples [#deGuise2018]_,
# [#Clements2016]_, [#Reck1994]_ (images taken from [#deGuise2018]_):
#
# .. image:: /demonstrations/haar_measure/sun_decomp.svg
#    :align: center
#    :width: 90%
#
# [TODO: hand-draw graphics]
#
# In these graphics, every wire is a different mode. Every box represents on
# operation on one or more modes, and the number in the box indicates the number
# of parameters of that operation.  The boxes containing a ``1`` are simply
# phase shifts on individual modes. The blocks containing a ``3`` are
# :math:`SU(2)` transform with 3 parameters, such as the :math:`U(\phi, \theta,
# \omega)` above. Those containing a ``2`` are :math:`SU(2)` are transforms on
# pairs of modes with 2 parameters, similar to the 3-parameter ones but with
# :math:`\omega = \phi`.
#
# Although the decompositions can all produce the same set of operations, their
# structure and parametrization may have consequences in practice.  The first [#deGuise2018]_
# has a particularly convenient form that leads to a recursive definition
# of the Haar measure. The decomposition is formulated such that an
# :math:`SU(N)` operation can be implemented by sandwiching an :math:`SU(2)`
# transformation between two :math:`SU(N-1)` transformations, like so:
#
# .. figure:: /demonstrations/haar_measure/sun.svg
#    :align: center
#    :width: 70%
#
#    |
#
# The Haar measure can then be constructed recursively as a product of 3 terms. The
# first term depends on the parameters in the first :math:`SU(N-1)` transformation.
# The second term depends on the parameters in the lone :math:`SU(2)` transformation.
# The third term depends on the parameters in the other  :math:`SU(N-1)` transformation. 
#
# :math:`SU(2)` is the "base case" of the recursion - we simply have the Haar measure
# as expressed above.
#
# .. figure:: /demonstrations/haar_measure/su2_haar.svg
#    :align: center
#    :width: 25%
#
#    |
#
# Moving on up, we can write elements of :math:`SU(3)` as a sequence of three
# :math:`SU(2)` transformations. The Haar measure :math:`d\mu_3` then consists
# of two copies of :math:`d\mu_2`, with an extra term in between to take into
# account the middle transformation.
#
# .. figure:: /demonstrations/haar_measure/su3_haar.svg
#    :align: center
#    :width: 80%
#
#    |
#
# For :math:`SU(4)` and upwards, the form changes slightly, but still follows
# the pattern of two copies of :math:`d\mu_{N-1}` with a middle term in between.
# For larger systems, however, the recursive composition allows for some of the
# :math:`SU(2)` transformations on the lower modes to be grouped. We can take advantage
# of this and aggregate some of the parameters, which leads to one copy of :math:`d\mu_{N-1}`
# containing only a portion of the full set of terms (as detailed in [#deGuise2018]_, this is
# called a *coset measure*).
#
# .. image:: /demonstrations/haar_measure/su4_haar.svg
#    :align: center
#    :width: 100%
#
#    |
#
# Putting everything together, we have that
#
# .. math::
#
#    d\mu_N = d\mu_{N-1}^\prime \cdot \sin \theta_{N-1} \sin^{2(N-2)}\left(\frac{1}{2}\theta_{N-1}\right) \cdot  d\mu_{N-1}
#
# where :math:`d\mu^{\prime}` is a reduced copy of the measure depending on only
# a subset of the parameters. This is thus a convenient, systematic way to construct
# the :math:`N`-dimensional Haar measure. [TODO: fix notation to include parameter indices]
#
# As a final note, even though unitaries can be parametrized in different ways, the underlying
# Haar measure is *unique*.
# 
# Haar-random matrices from the :math:`QR` decomposition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Nice-looking math aside, sometimes you just need to generate a large number of
# high-dimensional Haar-random matrices. It would be very cumbersome to sample
# and keep track of the distributions of so many parameters; furthermore, the
# measure above requires you to parametrize your operations in a fixed way.
# There is a much quicker way to perform the sampling by taking a (slightly
# modified) `QR decomposition
# <https://en.wikipedia.org/wiki/QR_decomposition>`__ of complex-valued
# matrices.  This algorithm is detailed in [#Mezzadri2006]_, and consists of the
# following steps:
#
# 1. Generate an :math:`N \times N` matrix with complex numbers :math:`a+bi` where
#    both :math:`a` and :math:`b` are normally distributed with mean 0 and variance 1
#    (this is sampling from the distribution known as the *Ginibre ensemble*)  
# 2. Compute a QR decomposition :math:`Z = QR`.
# 3. Compute the diagonal matrix :math:`\Lambda = \hbox{diag}(R_{ii}/|R_{ii}|)`
# 4. Compute :math:`Q^\prime = Q \Lambda`, which will be Haar-random.
#
#

from numpy.linalg import qr

def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2
    Q, R = qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    return np.dot(Q, Lambda)


######################################################################
# Let's check that this method actually generates Haar-random unitaries
# by trying it out for :math:`N=2` and plotting on the Bloch sphere.
#

@qml.qnode(dev)
def qr_haar_random_unitary():
    qml.QubitUnitary(qr_haar(2), wires=0)
    return qml.state()

qr_haar_samples = [qr_haar_random_unitary() for _ in range(num_samples)]
qr_haar_bloch_vectors = np.array([convert_to_bloch_vector(s) for s in qr_haar_samples])
plot_bloch_sphere(qr_haar_bloch_vectors)

######################################################################
# As expected, we find our qubit states are distributed uniformly over the
# sphere.  This particular method is what's implemented in packages like
# ``scipy``'s `unitary_group
# <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.unitary_group.html>`__
# function. 
#
# Now, it's clear that this method works, but it is also important to
# understand *why* it works.  Step 1 is fairly straightforward - the base of our
# samples is a matrix full of complex values chosen from a typical
# distribution. This isn't enough by itself, since unitary matrices also
# have constraints -- their rows and columns must be orthonormal.
#
# These are constraints are where step 2 comes in - the outcome of a generic
# QR decomposition are an *orthonormal* matrix :math:`Q`, and upper
# triangular matrix :math:`R`. Since our original matrix was complex-valued, we end
# up with a :math:`Q` that is in fact already unitary. But why not stop there? Why 
# do we then perform steps 3 and 4?
#
# Steps 3 and 4 are needed because, while the QR decomposition yields a unitary,
# it is not a unitary that is properly Haar-random. In [#Mezzadri2006]_, it is
# explained that a uniform distribution over unitary matrices should also yield
# a uniform distribution over the *eigenvalues* of those matrices, i.e., every
# eigenvalue should be equally likely. Just using the QR decomposition out of
# the box produces an *uneven* distribution of eigenvalues of the unitaries!
#
# This discrepancy stems from the fact that the QR decomposition is not unique. 
# We can take any unitary diagonal matrix :math:`\Lambda`, and re-express the decomposition
# as :math:`QR = Q\Lambda \Lambda^\dagger R = Q^\prime R^\prime`. Step 3 removes this
# redundancy by fixing a :math:`\Lambda` that depends on :math:`R`, leading to a unique
# value of :math:`Q^\prime = Q \Lambda`, and a uniform distribution of eigenvalues.
# 
# .. admonition:: Try it!
#
#    Use the ``qr_haar`` function above to generate random unitaries and construct
#    a distribution of their eigenvalues. Then, comment out the lines for steps 3 and
#    4 and do the same - you'll find that the distribution is no longer uniform.
#    Check out the reference [#Mezzadri2006]_ for additional details and examples.
#
#
#
#

######################################################################
# Fun (and not-so-fun) facts
# --------------------------
#
# We've now learned what the Haar measure is, and both an analytical and numerical means
# of sampling unitaries uniformly at random. The Haar measure also has many interesting
# properties that play a role in quantum computing.
#
# Invariance of the Haar measure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Earlier, we showed that the Haar measure is used when integrating functions over 
# the unitary group:
#
# .. math::
#
#    \int_{V \in U(N)} f(V) d\mu_N(V).
#
# One interesting feature of the Haar measure is that it is both left and right *invariant* 
# under unitary transformations. That is,
#
# .. math::
#
#    \int_{V \in U(N)} f(\color{red}{W}V) d\mu_N(V) =  \int_{V \in U(N)} f(V\color{red}{W}) d\mu_N(V) =  \int_{V \in U(N)} f(V) d\mu_N(V).
#
# This holds true for *any* other :math:`N\times N` unitary :math:`W`! A
# consequence of such invariance is that if :math:`V` is Haar-random, then so is
# :math:`V^T,` :math:`V^\dagger,`, and any product of another unitary matrix and
# :math:`V` (where the product may be taken on either side).
#
# Another consequence of this invariance has to do with the structure of the entries
# themselves: they must all come from the same distribution. This is because the
# measure remains invariant under permutations, since they are unitary --
# the whole thing still has to be Haar random no matter how the entries are ordered,
# so all distributions must be the same.  The specific distribution is complex
# numbers :math:`a+bi` where both :math:`a` and :math:`b` has mean 0 and variance
# :math:`1/N` [#Meckes2014]_ (so, much like Ginibre ensemble we used in the QR decomposition
# above, but with a different variance and constraints due to orthonormality). 
#
# Concentration of measure
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# An unfortunate (although interesting) property of the Haar measure is that it
# suffers from the phenomenon of `concentration of measure
# <https://en.wikipedia.org/wiki/Concentration_of_measure>`__. Most of the "stuff"
# in the space concentrates around a certain area, and this gets worse as the 
# size of the system increases.
#
# You can see the beginnings of by looking at the sphere. For the 3-dimensional
# sphere, we saw how there is a concentration of "stuff" around the
# equator. This becomes increasingly prominent for `higher-dimensional spheres
# <https://en.wikipedia.org/wiki/N-sphere>`__. Suppose we have the sphere
# :math:`S^{2n-1}`, and some function :math:`f` that maps points on that sphere
# to real numbers. Sample a point :math:`x` on that sphere from the uniform
# measure, and compute the value of :math:`f(x)`. How close do you think the
# result will be to the mean value of the function, :math:`E[f]`, over the entire
# sphere?
#
# A result called *Levy's lemma* bounds the probability of a randomly selected
# :math:`x` result that deviates from :math:`E[f]` by an amount :math:`\epsilon`:
#
# .. math::
# 
#    \hbox{Pr}(|f(x) - E[f]| \ge \epsilon) \leq 2 \exp\left[-\frac{n\epsilon^2}{9\pi^3 \eta^2}\right]
#
# A constraint on the function :math:`f` is that it must be `Lipschitz
# continuous <https://en.wikipedia.org/wiki/Lipschitz_continuity>`__, where
# :math:`\eta` is the *Lipschitz constant* of the function. Clearly the larger
# the deviation :math:`\epsilon`, the less likely you would be to encounter that
# value at random. Furthermore, increasing the dimension :math:`n` also makes
# the deviation exponentially less likely.
#
# Now, this result seems unrelated to quantum states -- it concerns higher-
# dimensional spheres. However, recall that a quantum state vector is a complex
# vector whose squared values sum to 1, just like a spheres. If you "unroll"
# a quantum state vector of dimension :math:`2^n` by stacking its real and
# complex parts, you end with a space isomorphic to that of a sphere of
# dimension X [TODO: verify dimensions in this whole section]. Given that
# measure concentrates on sphere, and quantum states can be trivially made to
# look like spheres, functions on random quantum states will also demonstrate
# concentration! This is bad news because it means that the more qubits you have
# the more the random states will look the same.  [TODO: rephrase]
#
# Haar measure and barren plateaus
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Suppose you are venturing out to solve a new problem using an algorithm such
# as the :doc:`variational quantum eigensolver </demos/tutorial_vqe>`. A
# critical component of such methods is the choice of :doc:`variational ansatz
# </glossary/circuit_ansatz>`. Having now learned a bit about the properties of
# the Haar measure, you may think it would make sense to use this for the
# parametrization. The initial parameter selection will give you a state in the
# Hilbert space uniformly at random. Then, since this ansatz spans the entire
# Hilbert space, you're guaranteed to be able to represent the target state with
# your ansatz, and it should be able to find it with no issue... right?
#
# Unfortunately, while such an ansatz is extremely *expressive* (i.e., it is
# capable of representing any possible state), these ansatze actually suffer the
# most from the barren plateau problem [#McClean2018]_, [#Holmes2021]_.
# :doc:`Barren plateaus </demos/tutorial_barren_plateaus>` are regions in the
# space of a parametrized circuit where both the gradient and its variance
# rapidly approach 0, leading your optimizer to get stuck in a local minimum.
# This was explored recently in the work of [#Holmes2021]_, wherein closeness to
# the Haar measure was actually used as a metric for expressivity. The closer
# things are to the Haar measure, the more expressive they are, but they are
# also more prone to exhibiting barren plateaus.
#
# 
# .. figure:: /demonstrations/haar_measure/holmes-costlandscapes.png
#    :align: center
#    :width: 50%
#
#    Image source: [#Holmes2021]_. A highly expressive ansatz that can access
#    much of the space of possible unitaries (i.e., an ansatz capable of
#    producing unitaries in something close to a Haar-random manner) is very
#    likely to have flat cost landscapes and suffer from the barren plateau
#    problem.
#    
# It turns out that the types of ansatze know as *hardware efficient ansatze*
# also suffer from this problem if they are "random enough" (this notion will be
# formalized in a future demo). It was shown in [#McClean2018]_ that this is a
# consequence of the concentration of measure phenomenon described above. The
# values of gradients and variances can be computed for classes of circuits on
# average by integrating with respect to the Haar measure, and it is shown that
# these values decrease exponentially in the number of qubits, and that huge
# swaths of the cost landscape are simply flat.
#
#

######################################################################
# Conclusion
# ----------
#  
# The Haar measure plays an important role in quantum computing - anywhere
# you might be dealing with sampling random circuits, or averaging over
# all possible unitary operations, you'll want to do so with respect
# to the Haar measure.
#
# There are two important aspects of this that we have yet to touch upon,
# however. The first is whether it is efficient to sample from the Haar measure
# --- given that the number of parameters to keep track of is exponential in the
# number of qubits, certainly not. But a more interesting question is do we
# *need* to always sample from the full Haar measure?  The answer to this is
# "no" in a very interesting way. Depending on the task at hand, you may be able
# to take a shortcut using something called a *unitary design*. In an upcoming
# demo, we will explore the amazing world of unitary designs and their
# applications!
#
# References
# ----------
#
# .. [#deGuise2018]
#
#     H. de Guise, O. Di Matteo, L. L. Sánchez-Soto. "Simple factorization of unitary
#     transformations" `Phys. Rev. A 97 022328 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.022328>`__
#     (2018). (`arXiv <https://arxiv.org/abs/1708.00735>`__)
#
# .. [#Clements2016]
#
#     W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S.Kolthammer, and I. A. Walmsley, “Optimal design for universal multiport interferometers,” \
#     `Optica 3, 1460–1465 <https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`__ (2016).
#     (`arXiv <https://arxiv.org/abs/1603.08788>`__)
#
# .. [#Reck1994]
#
#    M. Reck, A. Zeilinger, H. J. Bernstein, and P. Bertani, “Experimental
#    realization of any discrete unitary operator,” `Phys. Rev. Lett.73, 58–61
#    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58>`__
#    (1994).
#
# .. [#Mezzadri2006]
#
#     F. Mezzadri. "How to generate random matrices from the classical compact groups"
#     (`arXiv <https://arxiv.org/abs/math-ph/0609050>`__)
#
# .. [#Meckes2014]
#
#     E. Meckes. `The Random Matrix Theory of the Classical Compact Groups  
#     <https://case.edu/artsci/math/esmeckes/Haar_book.pdf>`_
#
#
# .. [#McClean2018]
#
#     McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven,
#     H. (2018). Barren plateaus in quantum neural network training
#     landscapes. `Nature Communications, 9(1),
#     <http://dx.doi.org/10.1038/s41467-018-07090-4>`__.
#     `(arXiv) <https://arxiv.org/abs/1803.11173>`__
#
# 
# .. [#Holmes2021]
#
#     Holmes, Z., Sharma, K., Cerezo, M., & Coles, P. J. (2021). Connecting ansatz
#     expressibility to gradient magnitudes and barren plateaus. `(arXiv) 
#     <https://arxiv.org/abs/2101.02138>`__.
#
#
