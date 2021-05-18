r""".. role:: html(raw)
   :format: html

Unitary Designs
===============

.. meta::
    :property="og:description": Learn about designs and their uses in quantum computing.

    :property="og:image": https://pennylane.ai/qml/_images/fano.png

.. related::

    tutorial_haar_measure Understanding the Haar measure

*Author: PennyLane dev team. Posted: XX May 2021. Last updated: XX May 2021.*


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
and explore how to leverage 2-designs in PennyLane to compute the average
fidelity of a noisy channel. All of this will help us to answer the key question
above---instead of sampling from the Haar measure, there are situations where we
use a :math:`t`-designs as a shortcut, making some protocols much more efficient!


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
points is called a :math:`t`-design, which is defined formally as follows:

.. admonition:: Definition
    :class: defn

    Let :math:`p_t: \mathcal{S}(R^d)\rightarrow R` be a polynomial in :math:`d`
    variables, with all monomial terms homogeneous in degree at most :math:`t`. A
    set :math:`X = \{x: x \in \mathcal{S}(R^d)\}` is a spherical :math:`t`-design if

    .. math::

        \frac{1}{|X|} \sum_{x \in X} p_t(x) = \int_{\mathcal{S}(R^d)} p_t (u) d\mu(u)

    holds for all possible :math:`p_t`, where :math:`d\mu` is the uniform,
    normalized spherical measure.


Now, this is a pretty abstract picture, so let's consider the 3-dimensional
sphere. What this definition tells us is that if we want to take the average of
a polynomial where all terms have the same degree 2 in the coordinates of the
sphere, we can do so with a representative set of points called a 2-design,
rather than the whole sphere.  Similarly, if all terms of the polynomial have
degree 3, we would need a 3-design. But what are these representative sets of
points?

For the case of a sphere, the points that comprise the design look like some
familiar solids:

.. figure:: /demonstrations/unitary_designs/shapes.svg
   :align: center
   :width: 60%

|

Clearly, as :math:`t` increases, the shapes of the designs become increasingly
more sphere-like, which makes intuitive sense.


Complex projective designs
^^^^^^^^^^^^^^^^^^^^^^^^^^

There was nothing quantum in the above description of spherical designs--it's
time now to see how things fit together. There is quite a nice progression we
can follow to build up some intuition about what unitary designs are, and what
they should do. The most pressing matter though is that so far we have dealt
only with real numbers. Surely we will need to include complex values, given
that unitary matrices have complex entries.

*Complex projective designs* lie conceptually between spherical and unitary
designs. Rather than considering :math:`d`-dimensional vectors on the unit
sphere of :math:`\mathcal{S}(R^d)`, we will shift to complex-valued unit vectors
on the sphere :math:`\mathcal{S}(C^d)`. Complex projective designs are defined
similarly to spherical designs, in that they can be used to compute average
values of polynomials of homogeneous degree :math:`t` in *both* the real and
complex variables (so :math:`2d` variables total).

.. admonition:: Definition
    :class: defn

    Let :math:`p_{t,t}: \mathcal{S}(C^d)\rightarrow C` be a polynomial with
    homogeneous degree :math:`t` in :math:`d` variables, and degree :math:`t` in
    the complex conjugates of those variables. A subset :math:`X = \{x: x \in
    \mathcal{S}(C^d)\}` is a complex projective :math:`t`-design if

    .. math::

        \frac{1}{|X|} \sum_{x \in X} p_{t,t}(x) = \int_{\mathcal{S}(C^d)}
        p_{t,t} (u) d\mu(u)

    holds for all possible :math:`p_{t,t}`.

An interesting fact about complex projective designs is that if you "unroll"
the real and complex parts into real-valued vectors of length :math:`2d`, you have yourself
a regular spherical design! This works in the other direction, too---spherical :math:`t`-designs
can be transformed into :math:`t/2`-designs over :math:`\mathcal{S}(C^{d/2})`.

.. admonition:: Fun fact

    If you've ever studied the characterization of quantum systems, you may have come across
    some special sets of measurements called mutually unbiased bases (MUBs), or symmetric,
    informationally complete positive operator valued measurements (SIC-POVMs). Both
    these objects are examples of complex projective 2-designs! [cite]


Unitary designs
^^^^^^^^^^^^^^^

We've seen now how both spherical and complex projective designs are defined as
representative sets of points in the space that you can use as "short cuts" to
evaluate the average of polynomials up to a given degree :math:`t`. A *unitary
design* in the abstract takes this one step further---it considers polynomials that
are functions of entries of the unitary matrices.

.. admonition:: Definition
    :class: defn

    Let :math:`P_{t,t}(U)` be a polynomial with homogeneous degree :math:`t` in
    :math:`d` variables in the entries of a unitary matrix :math:`U`, and degree
    :math:`t` in the complex conjugates of those entries. A unitary
    :math:`t`-design is a set of :math:`K` unitaries :math:`\{U_k\}` such that

    .. math::

        \frac{1}{K} \sum_{k=1}^{K} P_{t,t}(U_k) = \int_{\mathcal{U}(d)}
        P_{t,t} (U) d\mu(U)

    holds for all possible :math:`P_{t,t}`, and where :math:`d\mu` is the
    uniform *Haar measure*.

We can "unroll" unitary designs as well to create complex projective designs and
spherical designs. For each unitary matrix in the set, vectorize it by stacking its
columns on top of each other. You then have a set of vectors which form a complex
projective design in :math:`S(C^{d^2})`!


Unitary :math:`t`-designs in action
-----------------------------------

Unitary :math:`t`-designs are not as easy to visualize as the spherical designs
above; and in general they are hard to construct. That said, there are some well
known results for how to construct, unitary 1-, 2-, and even unitary 3-designs
using some objects that will be quite familiar. But first, now that we know what
they are, it's time to see why they are useful.

Average fidelity
^^^^^^^^^^^^^^^^

A key application of unitary designs is benchmarking quantum operations. Let's
explore a simple example: suppose we have a noisy quantum channel
:math:`\Lambda` that hope will perform something close to the unitary operation
:math:`V`.  What can we say about the performance of this channel?

One metric of interest is the *fidelity* of the channel. Consider the state
:math:`|0\rangle`.  Applying the unitary operation :math:`V` gives
:math:`V|0\rangle`.  Applying the channel :math:`\Lambda` gives us something a
little different, as we have to consider the state as a density matrix. The
action of :math:`\Lambda` on our starting state is :math:`\Lambda(|0\rangle
\langle 0|)`.  If :math:`\Lambda` was perfect, then :math:`\Lambda(|0\rangle
\langle 0|) = V|0\rangle \langle 0|V^\dagger`, and the fidelity is

.. math::

    F(\Lambda, V) = \langle 0 | V^\dagger \cdot \Lambda(|0\rangle \langle 0|) \cdot V|0\rangle = 1.

But that's the ideal case--in reality, :math:`\Lambda` is not going to
implement :math:`V` perfectly. Furthermore, all we've computed so far is the
fidelity when the initial state is :math:`|0\rangle`. What if the initial state
is something different? What is the fidelity *on average*?

To compute an average fidelity, we must do so with respect to the full set
of Haar-random states. We usually generate random states by applying a
Haar-random unitary :math:`U` to :math:`|0\rangle`. Thus to compute the average
fidelity over all such :math:`U`, we must evaluate

.. math::

    \bar{F}(\Lambda, V) = \int_{\mathcal{U}} d\mu(U) \langle 0 | U^\dagger V^\dagger \Lambda(U |0\rangle \langle 0| U^\dagger) V U |0\rangle.

This is known as *twirling* the channel :math:`\Lambda`. As expressed above, computing this
average fidelity would be a nightmare--we'd have to compute the fidelity with
respect to an infinite number of states!

However, consider the expression in the integral above. We have an inner product
involving two instances of :math:`U`, and two instances of
:math:`U^\dagger`. This means that the expression is a polynomial of degree 2 in
both the elements of :math:`U` and its complex conjugates--this is exactly the
same situation as above when we defined unitary 2-designs! This means that if we
can find a set of :math:`K` unitaries that form a 2-design, we can compute the
average fidelity using only a finite set of initial states:

.. math::

    \frac{1}{K} \sum_{j=1}^K \langle 0 | U_j^\dagger V^\dagger \Lambda(U_j |0\rangle \langle 0|
    U_j^\dagger) V^\dagger U_j |0\rangle = \int_{\mathcal{U}} d\mu(U) \langle 0
    | U^\dagger V^\dagger \Lambda(U |0\rangle \langle 0| U^\dagger) V U |0\rangle

This is incredible! But a question remains: what is the representative set of unitaries?

Design design
^^^^^^^^^^^^^

A beautiful result in quantum computing is that some special groups of matrices
you may already be familiar with are unitary 2-designs. More specifically: the
Pauli group forms a unitary 1-design, and the Clifford group forms a unitary
3-design. By the definition of designs, this means that the Clifford group is
also a 1- and 2-design.

The :math:`n`-qubit Pauli group, :math:`\mathcal{P}(n)` is the set of all tensor
products of Pauli operations :math:`X`, :math:`Y`, :math:`Z`, and :math:`I`. The
:math:`n`-qubit Clifford group :math:`\mathcal{C}(n)` is the *normalizer* of the
Pauli group. In simpler terms, the Clifford group is the set of operations that
send Paulis to Paulis when acting under conjugation (up to a phase), i.e.,

.. math::

   C P C^\dagger = \pm P^\prime, \quad \forall P, P^\prime \in \mathcal{P}(n), C \in \mathcal{C}(n)

The Pauli and Clifford groups both have some profoundly interesting
properties. The Clifford group itself is *massive*, as it has elements to
perform a one-to-one mapping from Paulis to Paulis. [TODO: elaborate a bit here]

The single-qubit Clifford group is built from just two operations. One is the
Hadamard. We know that :math:`H` sends

.. math::

   H X H^\dagger = Z, \quad H Y H^\dagger = -Y, \quad H Z H^\dagger = X.

This clearly maps Paulis to Paulis. The other generator is the phase gate
:math:`S`:

.. math::

   S X S^\dagger = Y, \quad S Y S^\dagger = -X, \quad S Z S^\dagger = Z.

If both :math:`H` and :math:`S` map Paulis to Paulis, then products of them will
do so as well (in group theory terms, the single-qubit Clifford group is
generated by :math:`H` and :math:`S`).  For example, consider the action of
:math:`HS`:

.. math::

   (HS) X (HS)^\dagger = -Y, \quad (HS) Y (HS)^\dagger = -Z, \quad (HS) Z (HS)^\dagger = X.

Since :math:`Y = iXZ`, it is enough to specify Clifford operations by how they
act on :math:`X` and :math:`Z`.  For a particular Clifford, there are 6 possible
ways it can transform :math:`X` (:math:`\pm X, \pm Y`, or :math:`\pm Z`).  Once
that is determined, there are four remaining options for the transformation of
:math:`Z`, leading to 24 elements total. They are provided below in string format--we will
make use of them in the code example that follows.

"""

single_qubit_cliffords = [
 '',
 'H', 'S',
 'HS', 'SH', 'SS',
 'HSH', 'HSS', 'SHS', 'SSH', 'SSS',
 'HSHS', 'HSSH', 'HSSS', 'SHSS', 'SSHS',
 'HSHSS', 'HSSHS', 'SHSSH', 'SHSSS', 'SSHSS',
 'HSHSSH', 'HSHSSS', 'HSSHSS'
]

######################################################################
# .. admonition:: Fun fact
#
#    Despite its size, the multi-qubit Clifford can also be specified by only a
#    small set of generators. In fact, it is only one more gate than is needed
#    for the single-qubit case.  Using just :math:`H`, :math:`S`, and CNOT (on
#    every possible qubit or pair of qubits), you can generate the
#    :math:`n`-qubit group. Be careful though--the size of the group increases
#    exponentially. The 2-qubit group alone has 11520 elements! The size can be
#    worked out in a manner analogous to that we used above in the single qubit
#    case: by looking at the combinatorics of the possible ways the gates can
#    map Paulis with only :math:`X` and :math:`Z` to other Paulis.
#
# An experiment
# ^^^^^^^^^^^^^
# The whole idea of unitary designs may sound too good to be true. In this
# section, we'll put them to the test and compute the average fidelity of a
# noisy operation in two ways: once with experiments using a large but finite
# amount of Haar-random unitaries, and then again with only the Clifford group.

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Scipy allows us to sample Haar-random unitaries directly!
from scipy.stats import unitary_group

# set the random seed
np.random.seed(42)

# Use the mixed state simulator
dev = qml.device("default.mixed", wires=1)

######################################################################
# First, let's set up a noisy quantum channel. To keep things simple, we'll
# assume that the channel consists of applying :class:`~.pennylane.SX`, the
# square-root of :math:`X` gate, followed by a small amount of a few different
# types of noise.

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
# Now in order to perform a comparison, we're going to also need to be able to
# compute the true output value of the state, as well as the `fidelity
# <https://en.wikipedia.org/wiki/Fidelity_of_quantum_states>`__ compared
# to the ideal operation.

@qml.qnode(dev)
def ideal_experiment(state_prep_unitary):
    qml.QubitUnitary(state_prep_unitary, wires=0)
    qml.SX(wires=0)
    qml.QubitUnitary(state_prep_unitary.conj().T, wires=0)

    return qml.state()


def fidelity(state_1, state_2):
    # state_1 and state_2 are single-qubit density matrices.
    # This particular expression requires state_1 to be a pure state
    fid = np.trace(np.dot(state_1, state_2)) + 2 * np.sqrt(
        np.linalg.det(state_1) * np.linalg.det(state_2)
    )
    return fid.real

######################################################################
# It's now time to experiment. We'll choose 50000 Haar-random unitaries, run
# both experiments, then compute the fidelity.

n_samples = 50000

fidelities = []

for _ in range(n_samples):
    # Select a Haar-random unitary
    x = unitary_group.rvs(2)

    # Run the two experiments
    state_1 = ideal_experiment(x)
    state_2 = noisy_experiment(x)
    fidelities.append(fidelity(state_1, state_2))

######################################################################
# Let's take a look at the results---we compute the mean and variance of the
# fidelities, and plot a histogram.

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
plt.tight_layout()
plt.show()

######################################################################
# Now let's repeat the same experiment, but using only Clifford group
# elements. To perform these experiments, we will make use of the quantum
# transform capabilities of PennyLane. We'll start with a transform that
# applies a Clifford operation based on its string representation.

@qml.single_tape_transform
def apply_single_clifford(tape, clifford_string, inverse=False):
    for gate in clifford_string:
        if gate == 'H':
            qml.Hadamard(wires=0)
        else:
            if inverse:
                qml.PhaseShift(np.pi/2, wires=0)
            else:
                qml.PhaseShift(-np.pi/2, wires=0)

######################################################################
# Note that here we've coded up the inverse operation manually; this is because
# the ``default.mixed`` device does not currently support inverse operations.
#
# Next, we need a transform that applies a Clifford in the context of the full
# experiment; i.e., apply the Clifford, then the operations, followed by the
# inverse of the Clifford. We use another transform for this:

@qml.single_tape_transform
def conjugate_with_clifford(tape, clifford):
    apply_single_clifford(tape, clifford, inverse=False)

    for op in tape.operations:
        op.queue()

    apply_single_clifford(tape, clifford, inverse=True)

    for op in tape.measurements:
        op.queue()

######################################################################
# Now let's run the experiments. For both the noisy and ideal experiments,
# we create a list of tapes, one for each Clifford. We then execute all
# the tapes on the device, and compute the fidelity.

# Note: This whole section is a good place for a QNode transform since
# it's a list of tapes, and we are doing something with the execution results.
ideal_tapes = []
noisy_tapes = []

for clifford in single_qubit_cliffords:
    # Set up the ideal tape
    with qml.tape.QuantumTape() as tape:
        qml.SX(wires=0)
        qml.state()

    twirled_tape = conjugate_with_clifford(tape, clifford)
    ideal_tapes.append(twirled_tape)

    # Set up the noisy tape
    with qml.tape.QuantumTape() as tape:
        qml.SX(wires=0)
        noisy_operation_sequence(damp_factor, depo_factor, flip_prob)
        qml.state()

    twirled_tape = conjugate_with_clifford(tape, clifford)
    noisy_tapes.append(twirled_tape)

# Execute all the tapes on the device
ideal_results = [tape.execute(dev) for tape in ideal_tapes]
noisy_results = [tape.execute(dev) for tape in noisy_tapes]

# Use the results to compute the fidelities
fidelities = []
for state_1, state_2 in zip(ideal_results, noisy_results):
    fidelities.append(fidelity(state_1[0], state_2[0]))

######################################################################
# Let's see how our results compare to the earlier simulation:

clifford_fid_mean = np.mean(fidelities)

print(f"Haar-random mean fidelity = {fid_mean}")
print(f"Clifford mean fidelity    = {clifford_fid_mean}")

######################################################################
# Incredible! We were able to compute the average fidelity using only 24
# experiments rather than 50000! It is also interesting to note the distribution
# of fidelities is much less spread out than that of the fully Haar-random case.

plt.hist(fidelities, bins=20)
plt.xlabel("Fidelity", fontsize=12)
plt.ylabel("Num. occurrences", fontsize=12)
plt.axvline(np.mean(fidelities), color="red")
plt.axvline(np.mean(fidelities) + np.std(fidelities), color="red", linestyle="dashed")
plt.axvline(np.mean(fidelities) - np.std(fidelities), color="red", linestyle="dashed")
plt.tight_layout()
plt.show()

######################################################################
# Conclusion
# ----------
# In this demo, we've barely scratched the surface of designs and their applications
# in quantum computing. [TODO: finish conclusion]
#
# References
# ----------
#
# [TODO: add citations]
#
# .. [#NandC2000]
#
#     M. A. Nielsen, and I. L. Chuang (2000) "Quantum Computation and Quantum Information",
#     Cambridge University Press.
#

