r"""
.. role:: html(raw)
   :format: html

Quantum advantage with Gaussian Boson Sampling
==============================================

.. meta::
    :property="og:description": Using light to perform tasks beyond the reach of classical computers.

    :property="og:image": https://pennylane.ai/qml/_images/tutorial_gbs_expt2.png

.. related::

    tutorial_gaussian_transformation Gaussian transformation
    qsim_beyond_classical Beyond classical computing with qsim
    qonn Optimizing a quantum optical neural network

On the journey to large-scale fault-tolerant quantum computers, one of the first major
milestones is to demonstrate a quantum device carrying out tasks that are beyond the reach of
any classical algorithm. The Google Quantum team was the first to claim this achievement,
announced in their paper `Quantum supremacy using a programmable superconducting
processor <https://www.nature.com/articles/s41586-019-1666-5>`__ [#Arute2019]_. Now a team led
by Chao-Yang Lu and Jian-Wei Pan has performed a similar feat using quantum photonics. While
Google's experiment performed the task of :doc:`random circuit sampling </demos/qsim_beyond_classical>`
using a superconducting processor, the new experiment, published in the paper
`Quantum computational advantage using photons
<https://science.sciencemag.org/content/early/2020/12/02/science.abe8770?rss=1>`__
[#Zhong2020]_ leverages the quantum properties of light to tackle a task called
`Gaussian Boson Sampling <https://strawberryfields.ai/photonics/concepts/gbs.html>`__ (GBS).

This tutorial will walk you through the basic elements of GBS, motivate why it is
classically challenging, and show you how to explore GBS using PennyLane and the photonic
quantum devices accessible via the
`PennyLane-Strawberry Fields plugin <https://pennylane-sf.readthedocs.io>`__. If you are
interested in possible applications of GBS, or want to access programmable GBS hardware
via the cloud, check out the
`Strawberry Fields website <https://strawberryfields.ai/>`__ for more details.

|

.. image:: /demonstrations/tutorial_gbs_expt2.png
    :align: center
    :width: 80%
    :target: javascript:void(0);

.. figure:: /demonstrations/tutorial_gbs_expt1.png
    :align: center
    :width: 80%
    :target: javascript:void(0);

    *Illustration of the experimental setup used by Zhong et al. in Quantum computational advantage
    using photons* [#Zhong2020]_.

The origins of GBS
------------------

Let's first explain the name. `Boson <https://en.wikipedia.org/wiki/Boson>`__ refers to bosonic
matter, which, along with fermions, makes up one of the two elementary classes of particles.
The most prevalent bosonic system in our everyday lives is light, which is made of particles
called photons. Another famous example, though much harder to find, is the Higgs boson.
The distinguishing characteristic of bosons is that they follow "Bose-Einstein statistics",
which very loosely means that the particles like to bunch together (contrast this to fermionic
matter like electrons, which must follow the Pauli Exclusion Principle and keep apart).

This property can be observed in simple interference experiments such as the
`Hong-Ou Mandel setup <https://en.wikipedia.org/wiki/Hong%E2%80%93Ou%E2%80%93Mandel_effect>`__.
If two single photons are interfered on a balanced beamsplitter, they will both emerge at
the same output port---there is zero probability that they will emerge at separate outputs.
This is a simple but notable quantum property of light; if electrons were brought
together in a similar experiement, they would always appear at separate output ports.

Gaussian Boson Sampling [#hamilton2017]_ is, in fact, a member of a larger family of
"Boson Sampling" algorithms,
stemming back to the initial proposal of Aaronson and Arkhipov [#aaronson2013]_ in 2013.
Boson Sampling is quantum interferometry writ large. Aaronson and Arkhipov's original proposal
was to inject many single photons into distinct input ports of a large interferometer, then
measure which output ports they appear at. The natural interference properties of bosons
means that photons will appear at the output ports in very unique and specific ways. Boson
Sampling was not proposed with any kind of practical real-world use-case in mind. Like
the random circuit sampling, it's just a quantum system being its best self. With sufficient
size and quality, it is strongly believed to be hard for a classical computer to simulate this efficiently.

Finally, the "Gaussian" in GBS refers to the fact that we modify the original Boson Sampling
proposal slightly: instead of injecting single photons---which are hard to jointly create in the
size and quality needed to demonstrate Boson Sampling conclusively---we instead use states of
light that are experimentally less demanding (though still challenging!).
These states of light are called Gaussian states,
because they bear strong connections to the
`Gaussian (or Normal) distribution <https://en.wikipedia.org/wiki/Normal_distribution>`__
from statistics. In practice, we use a particular Gaussian state called a
`squeezed state <https://en.wikipedia.org/wiki/Squeezed_states_of_light>`__ for the inputs,
since these are arguably the most non-classical of Gaussian states.


.. note:: While computationally hard to simulate, Boson Sampling devices, on their own,
          are not capable of universal quantum computing. However, in combination with other
          components, GBS is a key building block for a
          universal device [#Bourassa2020]_.


Coding a GBS algorithm
----------------------

The researchers in [#Zhong2020]_ experimentally demonstrate a GBS device by preparing 50
squeezed states and injecting them into a 100-mode interferometer. In this demo,
in order to keep things classically simulable, we will stick to a much simpler setting
consisting of 4 squeezed states injected into a 4-mode interferometer. At a high level,
an interferometer on :math:`N` modes can be represented using an :math:`N\times N` unitary
matrix :math:`U`. When decomposed into a quantum optical circuit, the interferometer will
be made up of beamsplitters and phase shifters.

.. image:: /demonstrations/tutorial_gbs_circuit2.png
    :align: center
    :width: 90%
    :target: javascript:void(0);

.. raw:: html

    <br>

Simulating this circuit using PennyLane is easy; we can simply read off the gates from left
to right, and convert it into a QNode.
"""

import numpy as np

# set the random seed
np.random.seed(42)

# import PennyLane
import pennylane as qml

######################################################################
# We must define the unitary matrix we would like to embed in the circuit.
# We will use SciPy to generate a Haar-random unitary:

from scipy.stats import unitary_group

# define the linear interferometer
U = unitary_group.rvs(4)
print(U)

######################################################################
# We can now use this to construct the circuit, choosing a compatible
# device. For the simulation, we can use the Strawberry Fields
# Gaussian backend. This backend is perfectly suited for simulation of GBS,
# as the initial states are Gaussian, and all gates transform Gaussian states to other
# Gaussian states.

n_wires = 4
cutoff = 10

dev = qml.device("strawberryfields.gaussian", wires=n_wires, cutoff_dim=cutoff)

@qml.qnode(dev)
def gbs_circuit():
    # prepare the input squeezed states
    for i in range(n_wires):
        qml.Squeezing(1.0, 0.0, wires=i)

    # linear interferometer
    qml.Interferometer(U, wires=range(n_wires))
    return qml.probs(wires=range(n_wires))


######################################################################
# A couple of things to note in this particular example:
#
# 1. To prepare the input single mode squeezed vacuum state :math:`\ket{re^{i\phi}}`,
#    where :math:`r = 1` and :math:`\phi=0`, we
#    apply a squeezing gate (:class:`~pennylane.Squeezing`) to each of the wires (initially in
#    the vacuum state).
#
# 2. Next we apply the linear interferometer to all four wires using
#    :class:`~pennylane.Interferometer` and the unitary matrix ``U``. This operator
#    decomposes the unitary matrix representing the linear interferometer into single-mode
#    rotation gates (:class:`~pennylane.PhaseShift`) and two-mode beamsplitters
#    (:class:`~pennylane.Beamsplitter`). After applying the interferometer, we will denote the
#    output state by :math:`\ket{\psi'}`.
#
# 3. GBS takes place physically in an infinite-dimensional Hilbert space,
#    which is not practical for simulation. We need to set an upper limit on the maximum
#    number of photons we can detect. This is the
#    ``cutoff`` value we defined above; we will only be considering detection events
#    containing 0 to 9 photons per mode.
#
# We can now execute the QNode, and extract the resulting probability distribution:

probs = gbs_circuit().reshape([cutoff] * n_wires)
print(probs.shape)

######################################################################
# For example, element ``[1,2,0,1]`` represents the probability of
# detecting 1 photon on wire
# ``0`` and wire ``3``, and 2 photons at wire ``1``, i.e., the value
#
# .. math:: \text{prob}(1,2,0,1) = \left|\braketD{1,2,0,1}{\psi'}\right|^2.
#
# Let's extract and view the probabilities of measuring various Fock states.

# Fock states to measure at output
measure_states = [(0,0,0,0), (1,1,0,0), (0,1,0,1), (1,1,1,1), (2,0,0,0)]

# extract the probabilities of calculating several
# different Fock states at the output, and print them out
for i in measure_states:
    print(f"|{''.join(str(j) for j in i)}>: {probs[i]}")

######################################################################
# The GBS Distribution
# --------------------
#
# Hamilton et al. [#hamilton2017]_ showed that the probability of
# measuring a final state containing only 0 or 1 photons per mode is given by
#
# .. math::
#
#     \left|\left\langle{n_1,n_2,\dots,n_N}\middle|{\psi'}\right\rangle\right|^2 =
#     \frac{\left|\text{Haf}[(U(\bigoplus_i\tanh(r_i))U^T)]_{st}\right|^2}{\prod_{i=1}^N \cosh(r_i)}
#
# i.e., the sampled single-photon probability distribution is proportional to the **hafnian** of a
# submatrix of :math:`U(\bigoplus_i\tanh(r_i))U^T`.
#
# .. note::
#
#     The hafnian of a matrix is defined by
#
#     .. math:: \text{Haf}(A) = \sum_{\sigma \in \text{PMP}_{2N}}\prod_{i=1}^N A_{\sigma(2i-1)\sigma(2i)},
#
#     where :math:`\text{PMP}_{2N}` is the set of all perfect matching permutations of :math:`2N` elements. In graph theory, the
#     hafnian calculates the number of perfect `matchings
#     <https://en.wikipedia.org/wiki/Matching_(graph_theory)>`_ in a graph with
#     adjacency matrix :math:`A`.
#
#     Compare this to the permanent, which calculates the number of perfect matchings on a *bipartite*
#     graph. Notably, the permanent appears in vanilla Boson Sampling in a similar way
#     that the hafnian appears in GBS.
#     The hafnian turns out to be a generalization of the permanent, with the relationship
#
#     .. math::
#
#         \text{Per(A)} = \text{Haf}\left(\left[\begin{matrix}
#             0&A\\ A^T&0
#         \end{matrix}\right]\right).
#
# As any algorithm that could calculate (or even approximate) the hafnian could also calculate the
# permanent---a `#P-hard problem <https://en.wikipedia.org/wiki/%E2%99%AFP>`__---it follows that
# calculating or approximating the hafnian must also be a classically hard problem. This lies behind
# the classical hardness of GBS.
#
# In this demo, we will use the same squeezing parameter, :math:`z=r`, for
# all input states; this allows us to simplify this equation. To start with, the hafnian expression
# simply becomes :math:`\text{Haf}[(UU^T\tanh(r))]_{st}`, removing the need for the direct sum.
#
# Thus, we have
#
# .. math::
#
#     \left|\left\langle{n_1,n_2,\dots,n_N}\middle|{\psi'}\right\rangle\right|^2 =
#     \frac{\left|\text{Haf}[(UU^T\tanh(r))]_{st}\right|^2}{n_1!n_2!\cdots n_N!\cosh^N(r)}.
#
# Now that we have the theoretical formulas, as well as the probabilities from our simulated GBS
# QNode, we can compare the two and see whether they agree.
#
# In order to calculate the probability of different GBS events classically, we need a
# method for calculating the hafnian.
# For this, we will use `The Walrus
# <https://the-walrus.readthedocs.io>`_ library (which is installed as a dependency of the
# PennyLane-SF plugin):

from thewalrus import hafnian as haf

######################################################################
# Now, for the right-hand side numerator, we first calculate the submatrix
# :math:`A = [(UU^T\tanh(r))]_{st}`:

A = (np.dot(U, U.T) * np.tanh(1))

######################################################################
# In GBS, we determine the submatrix by taking the
# rows and columns corresponding to the measured Fock state. For example, to calculate the submatrix
# in the case of the output measurement :math:`\left|{1,1,0,0}\right\rangle`,
# we have

print(A[:, [0, 1]][[0, 1]])

######################################################################
# i.e., we consider only the rows and columns where a photon was detected, which gives us
# the submatrix corresponding to indices :math:`0` and :math:`1`.

######################################################################
# Comparing to simulation
# -----------------------
#
# Now that we have a method for calculating the hafnian, let's compare the output to that provided by
# the PennyLane QNode.
#
# **Measuring** :math:`\ket{0,0,0,0}` **at the output**
#
# This corresponds to the hafnian of an *empty* matrix, which is simply 1:

print(1 / np.cosh(1) ** 4)
print(probs[0, 0, 0, 0])

######################################################################
# **Measuring** :math:`\ket{1,1,0,0}` **at the output**

A = (np.dot(U, U.T) * np.tanh(1))[:, [0, 1]][[0, 1]]
print(np.abs(haf(A)) ** 2 / np.cosh(1) ** 4)
print(probs[1, 1, 0, 0])

######################################################################
# **Measuring** :math:`\ket{0,1,0,1}` **at the output**

A = (np.dot(U, U.T) * np.tanh(1))[:, [1, 3]][[1, 3]]
print(np.abs(haf(A)) ** 2 / np.cosh(1) ** 4)
print(probs[0, 1, 0, 1])

######################################################################
# **Measuring** :math:`\ket{1,1,1,1}` **at the output**
#
# This corresponds to the hafnian of the full matrix :math:`A=UU^T\tanh(r)`:

A = (np.dot(U, U.T) * np.tanh(1))
print(np.abs(haf(A)) ** 2 / np.cosh(1) ** 4)
print(probs[1, 1, 1, 1])

######################################################################
# **Measuring** :math:`\ket{2,0,0,0}` **at the output**
#
# Since we have two photons in mode ``q[0]``, we take two copies of the
# first row and first column, making sure to divide by :math:`2!`:

A = (np.dot(U, U.T) * np.tanh(1))[:, [0, 0]][[0, 0]]
print(np.abs(haf(A)) ** 2 / (2 * np.cosh(1) ** 4))
print(probs[2, 0, 0, 0])

######################################################################
# The PennyLane simulation results agree (with almost negligible numerical error) to the
# expected result from the Gaussian boson sampling equation!
#
# This demo provides an entry-level walkthrough to the ideas behind GBS,
# providing you with the basic code needed for exploring the ideas behind
# the photonic quantum advantage paper. Try changing the number of modes,
# the number of injected squeezed states, or the cutoff dimension, and
# see how each of these affect the classical computation time. If you're
# interested in learning more about GBS, or about photonic quantum
# computing in general, the
# `Strawberry Fields website <https://strawberryfields.ai/>`__ is a great resource.
#
# References
# ----------
#
# .. [#Arute2019]
#
#     Arute, F., Arya, K., Babbush, R., et al. "Quantum supremacy using a programmable
#     superconducting processor"
#     `Nature 574, 505-510 (2019) <https://doi.org/10.1038/s41586-019-1666-5>`__.
#
# .. [#Zhong2020]
#
#     Zhong, H.-S., Wang, H., Deng, Y.-H., et al. (2020). Quantum computational advantage using photons. Science, 10.1126/science.abe8770.
#
# .. [#hamilton2017]
#
#     Craig S. Hamilton, Regina Kruse, Linda Sansoni, Sonja Barkhofen, Christine Silberhorn,
#     and Igor Jex. Gaussian boson sampling. Physical Review Letters, 119:170501, Oct 2017.
#     arXiv:1612.01199, doi:10.1103/PhysRevLett.119.170501.
#
# .. [#aaronson2013]
#
#     Scott Aaronson and Alex Arkhipov. The computational complexity of linear optics. Theory of
#     Computing, 9(1):143–252, 2013. doi:10.4086/toc.2013.v009a004.
#
# .. [#Bourassa2020]
#
#     Bourassa, J. E., Alexander, R. N., Vasmer, et al. (2020). Blueprint for a scalable
#     photonic fault-tolerant quantum computer. arXiv preprint arXiv:2010.02905.
#
