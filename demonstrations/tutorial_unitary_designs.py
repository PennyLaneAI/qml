r""".. role:: html(raw)
   :format: html

Unitary Designs
===============

.. meta::
    :property="og:description": Learn about designs and their uses in quantum computing.

    :property="og:image": https://pennylane.ai/qml/_images/fano.png

.. related::

    tutorial_haar_measure Understanding the Haar measure

*Author: PennyLane dev team. Posted: XX April 2021. Last updated: XX April 2021.*


.. note::

   This demo is intended to be a sequel to the `demo about the Haar measure
   </demos/tutorial_haar_measure>`__. If you are not familiar with the Haar measure,
   we recommend going through that demo first before exploring this one.

Take a close look at the following mathematical object:

.. figure:: /demonstrations/unitary_designs/fano_no_labels.svg
   :align: center
   :width: 30%

|

There are many things we can say about it: it consists of seven points and seven
lines (the circle counts as a line!); each line contains three points, and each
point is contained in exactly three lines. Furthermore, any pair of points is
contained in exactly one line. This object, called the `Fano plane
<https://en.wikipedia.org/wiki/Fano_plane>`__, is an example of a mathematical
structure called a *combinatorial design*. Designs are sets of objects and
associated groups, or blocks, of those objects that satisfy certain
symmetries. They are found throughout the mathematical literature and have been
studied for hundreds of years in a huge variety of contexts, from error
correcting codes to agriculture. So---what about quantum computing?

Designs are actually quite prevalent in quantum computing. It's highly likely
that you've unknowingly come across one before. At the end of the Haar measure
demo, we asked a very important question: "do we always *need* to sample from
the full Haar measure?". The answer to this lies in the study of *unitary
designs*. 

In this demo, you'll learn the definition of :math:`t`-designs, what it means to
generalize them to unitary :math:`t`-designs, and you'll see some canonical
examples of designs in quantum computing. You'll also learn about their
connection with the Haar measure, what it means to *twirl* a quantum channel,
and we'll explore how to use PennyLane to generate an approximate unitary
2-design. All of this will help us work towards answer our key question
above---instead of sampling from the Haar measure, there are some cases where we
can sample from :math:`t`-designs as a shortcut, making the sampling much more
efficient!


From spheres to unitary :math:`t`-designs
-----------------------------------------

Spherical designs
^^^^^^^^^^^^^^^^^

In the `Haar measure demo <demos/tutorial_haar_measure>`__ we began the
discussion by providing some intuition based on the plain old sphere. We're
going to take a similar approach here and discuss spherical :math:`t`-designs
before discussing unitary designs.

Suppose we have a polynomial in :math:`d` variables, and we would like to
compute its average over the surface of a :math:`N`-dimensional sphere. One can
do so by integrating that function over the sphere (using the proper measure!),
but that would be a lot of parameters to keep track of. In fact, it might even
be overkill. If a polynomial has terms with the same degree of at most :math:`t`, you can
compute the average value over the sphere using only a representative set of
points rather than integrating over the entire sphere! That representative set of
points is called a :math:`t`-design.

The formal definition of a :math:`t`-design is as follows:

.. admonition:: Definition     
    :class: defn

    Let :math:`p_t: \mathcal{S}(R^d)\rightarrow R` be a polynomial in :math:`d`
    variables, with all monomial terms homogenous in degree at most :math:`t`. A
    set :math:`X = \{x: x \in \mathcal{S}(R^d)\}` is a spherical :math:`t`-design if

    .. math::

        \frac{1}{|X|} \sum_{x \in X} p_t(x) = \int_{\mathcal{S}(R^d)} p_t (u) d\mu(u)

    holds for all possible :math:`p_t`, where :math:`d\mu` is the uniform,
    normalized spherical measure.


Now, this is a fairly abstract picture. Let's consider a sphere that we can
visualize, i.e., the 3-dimensional sphere. What this definition tells us is that
if we want to take the average of a polynomial where all terms have the same
degree 2 in the coordinates of the sphere, we can do so with a representative
set of points called a 2-design, rather than the whole sphere.  Similarly, if
all terms of the polynomial have degree 3, we would need a 3-design. But what
are these representative sets of points?

For the case of a sphere, there is actually a very interesting progression - the
points that comprise the design look like some familiar solids:

.. figure:: /demonstrations/unitary_designs/shapes.svg
   :align: center
   :width: 60%

|

Clearly, as :math:`t` increases, the shapes of the designs become increasingly 
more sphere-like, which makes intuitive sense.


Complex projective designs
^^^^^^^^^^^^^^^^^^^^^^^^^^

There was nothing quantum in the above description of spherical designs--it's
time now to start bringing this aspect in. There is quite a nice progression we
can follow to build up some intuition about what unitary designs are, and what
they should do. The most pressing matter though is that so far, spherical
designs have dealt only with real numbers, and surely we will need to include
complex values, given that unitary matrices have complex entries.

*Complex projective designs* lie conceptually in-between spherical and unitary
designs. Rather than considering :math:`d`-dimensional vectors on the unit
sphere of :math:`\mathcal{S}(R^d)`, we will shift to complex-valued unit
vectors on the sphere :math:`\mathcal{S}(C^d)`. Complex projective designs are
defined similarly to spherical designs, in that they can be used to compute
average values of polynomials of homogenous degree :math:`t` in *both* the real
and complex variables (so :math:`2d` variables total).

.. admonition:: Definition     
    :class: defn

    Let :math:`p_{t,t}: \mathcal{S}(C^d)\rightarrow C` be a polynomial with
    homogenous degree :math:`t` in :math:`d` variables, and degree :math:`t` in
    the complex conjugates of those variables. A complex projective :math:`t`-design is a
    subset :math:`X = \{x: x \in \mathcal{S}(C^d)\}` is a spherical
    :math:`t`-design if

    .. math::

        \frac{1}{|X|} \sum_{x \in X} p_{t,t}(x) = \int_{\mathcal{S}(C^d)}
        p_{t,t} (u) d\mu(u)

    holds for all possible :math:`p_{t,t}`.

An interesting fact about complex projective designs is that, if you "unroll"
the real and complex parts into real-valued vectors of length :math:`2d`, you have yourself
a regular spherical design! This works in the other direction, too---spherical :math:`t`-designs
can be transformed into :math:`t/2`-designs over :math:`\mathcal{S}(C^{d/2})`.

Unitary designs
^^^^^^^^^^^^^^^

We've seen now how both spherical and complex projective designs are defined as
representative sets of points in the space that you can use as "short cuts" to
evaluate the average of polynomials up to a given degree :math:`t`. A *unitary
design* in the abstract takes this one step further---it considers polynomials that
are functions of entries of the unitary matrices.

.. admonition:: Definition     
    :class: defn

    Let :math:`P_{t,t}(U)` be a polynomial with homogenous degree :math:`t` in
    :math:`d` variables in the entries of a unitary matrix :math:`U`, and degree
    :math:`t` in the complex conjugates of those entries. A unitary
    :math:`t`-design is a set of :math:`K` unitaries :math:`\{U_k\}` such that

    .. math::

        \frac{1}{K} \sum_{k=1}^{K} P_{t,t}(U_k) = \int_{\mathcal{U}(d)}
        P_{t,t} (U) d\mu(U)

    holds for all possible :math:`P_{t,t}`, and where :math:`d\mu` is the
    uniform *Haar measure*.


Unitary :math:`t`-designs and you
---------------------------------

Unitary :math:`t`-designs are not as easy to visualize as the spherical designs
above; and in general they are hard to construct. That said, there are some well
known results for how to construct, unitary 1-, 2-, and even unitary 3-designs
using some objects that will be quite familiar. But first, now that we know what
they are, it's time to see why they are useful.

What can we do with them?
^^^^^^^^^^^^^^^^^^^^^^^^^

A key application of unitary designs is in the benchmarking of quantum
operations. We won't cover this topic in detail here---characterizing quantum
systems is an exciting and active area of research. One important task is to
determine how well an operation performs *on average* over the full set of
quantum states.

Suppose we have a quantum channel :math:`\Lambda` and we wish to gauge its
quality. A simple way of doing this would be to compute, for example, the
fidelity of the operation when performed on a state chosen uniformly at random.
We can select a state uniformly at random using the Haar measure; however, this
only tells about how well it works on that particular quantum state. A better
measure of success would be to calculate an average fidelity over *all* possible
states. But surely this is impossible, as there are an infinite number of
quantum states!

This operation is known as *twirling* a quantum channel. More formally, what we are looking
to do is compute the average fidelity of....

.. math::

   \rho = \int U^\dagger \Lambda(U \rho U^\dagger) U d\mu

where :math:`d\mu` is the Haar measure.

Now, note what would happen if we were to fix a dimension :math:`d` and expand
out the contents of the integral for a single :math:`U`. We see that :math:`U`
is applied twice, and the same for :math:`U^\dagger`. This is just matrix
multiplication, and it will produce a matrix in which the entries are
polynomials with degree 2 in both the entries of :math:`U`, and
:math:`U^\dagger`!  This is incredible - it means we can perform the experiment
using only a limited subset of quantum states, but obtain the correct
result. The question then becomes, what is the representative set of unitaries?

Design design
^^^^^^^^^^^^^

Do they actually work?
^^^^^^^^^^^^^^^^^^^^^^^^

Let's find out! In what follows, we'll construct the single-qubit Clifford group, 
and compute the average fidelity of our operation.

"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Scipy allows us to sample Haar-random unitaries directly!
from scipy.stats import unitary_group

# set the random seed
np.random.seed(42)

# Use the mixed state simulator to save some steps in plotting later
dev = qml.device("default.mixed", wires=1)

######################################################################
# First, let's set up a noisy quantum channel. To keep things simple, we'll assume that
# the channel consists of applying :func:`~.pennylane.operation.SX`, the square-root of :math:`X` gate,
# but followed by a light amount of a few different types of noise.
#

# The strengths of various types of noise
damp_factor = 0.02
depo_factor = 0.02
flip_prob = 0.01

# The sequence of noisy operations
def noisy_operation_sequence(damp_factor, depo_factor, flip_prob):
    qml.AmplitudeDamping(damp_factor, wires=0)
    qml.DepolarizingChannel(depo_factor, wires=0)
    qml.BitFlip(flip_prob, wires=0)


@qml.qnode(dev)
def noisy_experiment(state_prep_unitary):
    # Prepare the state
    qml.QubitUnitary(state_prep_unitary, wires=0)

    # Apply the operation, followed by the noisy channel
    qml.SX(wires=0)
    noisy_operation_sequence(damp_factor, depo_factor, flip_prob)

    # Rotate back to the computational basis
    qml.QubitUnitary(state_prep_unitary.conj().T, wires=0)

    return qml.state()


######################################################################
# Now in order to perform a comparison, we're going to also need to be
# able to compute the true output value of the state, as well as a measure of
# similarity. For this, we'll use the `fidelity <https://en.wikipedia.org/wiki/Fidelity_of_quantum_states>`__.
#


@qml.qnode(dev)
def ideal_experiment(state_prep_unitary):
    # Prepare the state
    qml.QubitUnitary(state_prep_unitary, wires=0)
    qml.SX(wires=0)
    qml.QubitUnitary(state_prep_unitary.conj().T, wires=0)

    return qml.state()


def fidelity(state_1, state_2):
    fid = np.trace(np.dot(state_1, state_2)) + 2 * np.sqrt(
        np.linalg.det(state_1) * np.linalg.det(state_2)
    )
    return fid.real


######################################################################
# It's now time to experiment. We'll choose 10000 Haar-random unitaries, run both
# experiments, then compute the fidelity.

n_samples = 10000

fidelities = []

for _ in range(n_samples):
    # Select a Haar-random unitary
    x = unitary_group.rvs(2)

    # Run the two experiments
    state_1 = ideal_experiment(x)
    state_2 = noisy_experiment(x)
    fidelities.append(fidelity(state_1, state_2))

######################################################################
# Let's take a look at the results---we'll compute the mean and variance of their
# fidelity, and also take a look at a histogram.
#
fid_mean = np.mean(fidelities)
fid_std = np.std(fidelities)

print(f"Mean fidelity      = {fid_mean}")
print(f"Std. dev. fidelity = {fid_std}")

plt.hist(fidelities, bins=20)
plt.xlabel("Fidelity", fontsize=12)
plt.ylabel("Num. occurrences", fontsize=12)
plt.axvline(np.mean(fidelities), color="red")
plt.axvline(np.mean(fidelities) + np.std(fidelities), color="red", linestyle="dashed")
plt.axvline(np.mean(fidelities) - np.std(fidelities), color="red", linestyle="dashed")

######################################################################
# Conclusion
# ----------
#
#
#
# References
# ----------
#
# .. [#NandC2000]
#
#     M. A. Nielsen, and I. L. Chuang (2000) "Quantum Computation and Quantum Information",
#     Cambridge University Press.
#

