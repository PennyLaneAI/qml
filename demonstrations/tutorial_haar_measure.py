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

If you've ever dug into the literature about random circuits, variational ansatz
structure, or anything related to the structure and properties of unitary
operations, you've likely come across a statement like the following: "Assume
that :math:`U` is sampled uniformly at random from the Haar measure".  In this
demo, we're going to unravel this cryptic statement and take an in-depth look at
what it means. You'll gain an understanding of the general concept of *measure*,
the Haar measure and its special properties, and you'll learn how to sample from
it using the tools available in PennyLane and other open-source quantum
computing frameworks. By the end of this demo, you'll be able to include that
important statement in your own work with confidence!

Measure 
-------

`Measure theory <"https://en.wikipedia.org/wiki/Measure_(mathematics)">`__ is a
branch of mathematics that studies things that are measurable --- think length, area, or
volume, but the generalized idea across mathematical spaces and multiple
dimensions. Loosely speaking, the measure tells you about how "stuff" is
distributed and concentrated in a mathematical set, or space. An intuitive way
to understand measure is to think about a sphere. An arbitrary point on a sphere
can be parameterized by three numbers --- depending on what you're doing, you
may use Cartesian coordinates :math:`(x, y, z)`, or it may be more convenient to
use spherical co-ordinates :math:`(r, \phi, \theta)`.

.. image:: /demonstrations/haar_measure/spherical_coords.png
    :align: center
    :width: 90%

    The size of an elementary volume element with fixed angular difference may
    be smaller or larger depending on its position on the sphere.
     

Suppose you wanted to compute the volume of this sphere. Just like you can
compute the area under the curve of a function by integrating over its
parameters, you can compute the volume of a sphere by integrating over :math:`r,
\phi`, and :math:`\theta`. Your first thought when taking an integral might be
to just sandwich a function between a :math:`\int` and a :math:`dx`. Let's see
what happens if we do that with the spherical coordinates:

.. math::
 
    V = \int_0^{r} \int_0^{2\pi} \int_0^{\pi} dr d\phi d\theta = 2\pi^2 r

Now, we know that the volume of a sphere of radius :math:`r` is
:math:`\frac{4}{3}\pi r^3`, so what we got from this integral is clearly wrong!
Taking the integral naively doesn't like this doesn't take into account the
structure of the sphere with respect to its parameters. For example, consider
two small, infinitesimal elements of volume with the same difference in
:math:`\theta`, but at different parts of the sphere:

.. image:: /demonstrations/haar_measure/spherical_int_dtheta.png
    :align: center
    :width: 50%

Even though the :math:`d\theta` themselves are the same, there is way more
"stuff" near the equator of the sphere than there is near the poles. We are
going to need to take that into account when computing the integral. The same
holds true if we consider a :math:`dr`.  The contribution to volume of parts of
the sphere with a large :math:`r` is far more than for a small :math:`r` --- we should
expect the contribution to be proportional to :math:`r^2`, given that 
the surface area of a sphere of radius :math:`r` is :math:`4\pi r^2`.

.. image:: /demonstrations/haar_measure/spherical_int_dr.png
    :align: center
    :width: 90%

On the other hand, for a fixed :math:`r` and :math:`\theta`, the length of the
:math:`d\phi` is the same all around the circle. If we did the math and put all these
things together, we would find that the actual expression for the integral
should look like this:


.. math::
 
    V = \int_0^r \int_0^{2\pi} \int_0^{\pi} r^2 \sin \theta dr d\phi d\theta = \frac{4}{3}\pi r^3

These extra terms that we had to add to the integral, :math:`r^2 \sin \theta`,
constitute the *measure*. The measure weights portions of the sphere differently
depending on where they are in the space.

The Haar measure
----------------

Like points on a sphere, unitary matrices can be expressed in terms of a fixed
set of parameters. For example, the most general single-qubit rotation implemented in
PennyLane (:class:`~.pennylane.Rot`) is expressed as

.. math::

    U(\phi, \theta, \omega) = \begin{pmatrix} e^{-i(\phi + \omega)/2}
                        \cos(\theta/2) & -e^{i(\phi - \omega)/2} \sin(\theta/2)
                        \\ e^{-i(\phi - \omega)/2} \sin(\theta/2) & e^{i(\phi +
                        \omega)/2} \cos(\theta/2) \end{pmatrix}

The *Haar measure* tells us about how to sample these parameters in order to
sample unitaries uniformly at random.  The mathematical form of the Haar measure
can get quite hairy, especially as we increase the dimension of the system.

The Haar measure is defined for the unitary group :math:`U(d)` for every
dimension :math:`d`.  In general. a :math:`d`-dimensional unitary requires at
least :math:`d^2 - 1` parameters, which is a lot of stuff to keep track of!
Therefore we'll start with the case of a single qubit :math:`(d=2)`, then show
how things generalize.

Example: single-qubit system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Haar measure allows us to sample uniformly at random. One useful consequence
of this is that it provides a method to sample quantum *states* uniformly at
random from the Hilbert space.  We can simply generate Haar-random unitaries,
and apply them to a fixed basis state such as :math:`\vert 0\rangle`.

Keeping with the idea of spheres, let's try this out and investigate what
happens to a single-qubit state on the Bloch sphere when we apply Haar-random
unitaries. But first, let's suppose that we sample parameters for our unitaries
from the uniform distribution of each parameter - the angles :math:`\omega,
\phi`, and :math:`\theta` are all sampled from uniformly from the range
:math:`[0, 2\pi)`. We will use PennyLane to compute these states, and then use
the `qutip <"http://qutip.org/">`__ library to plot them on the Bloch sphere.

"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# set the random seed
np.random.seed(42)

# Use the mixed state simulator to save some steps in plotting later
dev = qml.device('default.mixed', wires=1)

@qml.qnode(dev)
def not_a_haar_random_unitary():
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

def convert_to_bloch_vector(rho):
    """Convert a density matrix to a Bloch vector."""
    # Used the mixed state simulator so we could have the density matrix for this part!
    ax = np.trace(np.dot(rho, X)).real
    ay = np.trace(np.dot(rho, Y)).real
    az = np.trace(np.dot(rho, Z)).real
    return [ax, ay, az]

not_haar_bloch_vectors = np.array([convert_to_bloch_vector(s) for s in not_haar_samples])

######################################################################
# With this, we can now plot these points on the Bloch sphere:

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
# of the sphere. To fix this, we will need to sample from the proper Haar
# measure, and weight the different parameters appropriately.
#
# For a single qubit, the Haar measure looks much like the case of a sphere,
# minus the radial component (intuitively, all qubit state vectors have length
# 1, so it makes sense that this wouldn't play a role here). The parameter
# that we will have to weight differently is :math:`\theta`, and in fact the
# adjustment in measure is identical to that we had to do with the polar axis
# of the sphere, i.e., :math:`\sin \theta`. 
#
# In order to sample the :math:`\theta` uniformly at random in this context, we
# must sample from the distribution :math:`\hbox{Pr}(\theta) = \sin \theta`. We
# can accomplish this by setting up a custom probability distribution in scipy.
#
#

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
# much-better distributed over the sphere.


######################################################################
# Show me more math!
# ~~~~~~~~~~~~~~~~~~
#
# While we can easily visualize the single-qubit case, this is no longer
# possible when we increase the number of qubits. Regardless, we can still
# obtain a mathematical expression for the Haar measure in arbitrary
# dimensions. In what follows, we will leave qubits behind and explore the many
# parameterizations of the :math:`N`-dimensional special unitary group.  This
# group, written as :math:`SU(N)`, is the continuous group consisting of
# all :math:`N \times N` unitary operations with determinant 1.
#
# Instead of qubits, we are going to shift to the world of *qumodes*. Our
# unitaries will be expressed as interferometers made up of beamsplitters (2- or
# 3-parameter operations), and phase shifts (1-parameter operations). These
# unitaries can still be considered as multi-qubit operations in the cases where
# :math:`N=2^n` for some number :math:`n` qubits, but they will have to be
# translated from the continuous-variable operations into qubit operations. In
# PennyLane, this can be done by feeding the unitaries to the
# :class:`~.pennylane.QubitUnitary` operation directly. Alternatively, one can
# use *quantum compilation* to express the operations as a sequence of
# elementary gates such as Pauli rotations and CNOTs.
#
# .. tip::
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
# This operation can be decomposes into beamsplitters and phase shifts 
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

######################################################################
# Fun facts
# ---------
#
# In this section: 
#
# * Levy's lemma and concentration of measure
# * How to sample Haar-random unitaries using QR decomposition
# * [Maybe] Entries of Haar-random unitaries look like complex numbers :math:`a+bi`
#   where :math:`a, b` are normally distributed with mean 0 and variance related to the
#   dimension of the unitary (something I came across during quantum volume demo)
#
#
# Haar-random matrices from the :math:`QR` decomposition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Suppose that you are working with a system that is large enough that it would
# be very cumbersome to sample and keep track of the distributions of so many
# parameters.  There is a much quicker way to generate Haar-random unitaries by
# taking a (slightly modified) QR decomposition of complex-valued matrices.
# This algorithm is detailed in [#Mezzadri2006]_, and consists of the following
# steps:
#
# 1. Generate an :math:`N \times N` matrix with normally-distributed complex numbers
# 2. Compute the QR decomposition :math:`Z = QR`.
# 3. Compute the diagonal matrix :math:`\Lambda = \hbox{diag}(R_{ii}/|R_{ii})`
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
# As expected, we find our qubit states are distributed uniformly over the sphere.
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
# There are a couple important aspects of this that we have yet to touch upon,
# however.  Is it *efficient* to sample from the Haar measure? Do we *need* to
# always sample from the full Haar measure? The answer to both of these
# questions is "no", but it is "no" in a very interesting way. Depending on the
# task at hand, you may not actually need to sample from the full Haar measure,
# but can take a shortcut using something called a *unitary design*. In an
# upcoming demo, we will explore the amazing world of unitary designs and their
# applications!
#
# References
# ----------
#
# .. [#deGuise2018]
#
#     H. de Guise, O. Di Matteo, L. L. Sánchez-Soto. "Simple factorization of unitary
#     transformations" `Phys. Rev. A 97 022328 <"https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.022328">`__
#     (2018). (`arXiv <"https://arxiv.org/abs/1708.00735">`__)
#
# .. [#Mezzadri2006]
#
#     F. Mezzadri. "How to generate random matrices from the classical compact groups"
#     (`arXiv <"https://arxiv.org/abs/math-ph/0609050">`__). 

