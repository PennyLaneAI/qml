r"""Demo: generative quantum advantage for classical and quantum problems
=====================================================================

Generative machine learning is all about inferring and sampling from probability distributions, and
sampling output distributions of quantum computers is known to be classically hard. So proving
quantum advantages for generative quantum machine learning is easy, right? Unfortunately, things are
not so simple. As H. Huang and colleagues point out in their recent preprint “Generative quantum
advantage for classical and quantum problems” [#genquantumadv]\_, claiming an advantage for
generative machine learning should not only require the ability to sample from a hard distribution,
but also to be able to learn it efficiently from data, and they investigate a specific scenario in
which this is possible.

In this demo we will unpack one of the main results of the paper to understand its core mechanics.
We will see that the problem is constructed so that learning the hard distribution boils down to
performing single qubit tomography, and we will debate the scope of this technique in relation to
practical AI. In particular, we will focus on the first theorem (Theorem 1) of the paper, since it
aligns closest with the notion of generative machine learning in the classical literature. It is
informally stated as:

::

    *Theorem 1 (Informal: Classically hard, quantumly easy generative models). Under standard
    complexity-theoretic conjectures, there exist distributions p(y|x) mapping classical n-bit strings
    to m-bit strings that a quantum computer can efficiently learn to generate using classical data
    samples, but are hard to generate with classical computers.*

To show the above, we need to do a couple of things

- Identify a classically ‘hard’ conditional distribution p(y|x) that corresponds to a family of
  quantum circuits. For this we can leverage some existing results about the hardness of sampling.
- Show that, with access to a dataset obtained by querying and sampling from p(y|x), we can infer
  the circuits that produced the data, and can therefore generate more data.

The paper gives a couple of circuit structures that can be used. We will focus on the simplest,
which they term instantaneously deep quantum neural networks (IDQNNs).
"""

######################################################################
# Instantaneously deep quantum neural networks
# --------------------------------------------
# Instantaneously deep quantum neural networks, or IDQNNs, are a particular type of shallow
# parameterized quantum circuit. The qubits of the circuit live on a lattice, which we’ll take to be a
# 2D lattice, and index the qubits by their lattice positions :math:`(i,j)`. To sample from the
# circuit one does the following, which we also depict below
#
# Recipe for sampling from an IDQNN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# 1. Prepare each qubit on the lattice in the :math:`\ket{+}` state.
# 2. Entangle the qubits by performing controlled-Z gates between some pairs of nearest neighbour
#    qubits on the lattice. If the qubits are horizontal neighbours on the lattice, a CZ is always
#    applied.
# 3. Perform a single qubit Z rotation :math:`U_{z}(\theta_{ij})=\exp(-\frac{i}{2}\theta_{ij}Z)` with
#    parameter :math:`\theta_{ij}` on each of the qubits
# 4. Measure all qubits in the X basis to produce outcomes :math:`y_{ij}`
# 
# Note that this is not a circuit diagram, but a graphical representation of the circuit description
# above: qubits are denoted by black dots, CZ gates are lines between dots, the angles specify the
# single qubit rotations, and the blue :math:`y_{ij}` are the X-measurement outcomes. We’ll also use
# the vector :math:`\boldsymbol{y}` from now on to denote all the :math:`y_{ij}`.
# 
# Remember that the above corresponds to a circuit acting on a 2D lattice of 12 qubits, and is a
# shallow circuit since the circuit depth does not depend on the size of the lattice. However, we can
# map this onto an equivalent deep 1D circuit with only 3 qubits by viewing the circuit as a
# measurement based quantum computation (MBQC) recipe. When viewed from this perspective, the
# horizontal dimension of the lattice becomes a time axis, so that at time 0, the system consists of
# just three qubits (the first vertical axis) prepared in the :math:`\ket{+}` state. After applying
# all rotation gates and CZ gates that involve these qubits and measuring them in the X basis, the
# state is teleported to the next line of three qubits (the precise state will depend on the
# measurement outcomes :math:`y_{11}`, :math:`y_{12}`, :math:`y_{13}`). Repeating this process until
# we arrive at the end of the lattice therefore defines a type of stochastic quantum circuit acting on
# three qubits. The precise way to make this mapping is well known from MBQC theory, and is a bit
# tricky (see Appendix H2 of the paper), so we won’t bore you with the details here.
# 
# If you apply the mapping to our example IDQNN, you find the following circuit:
# 
# The circuit structure for layers 2 and 3 is the same as for layer 1, where the CZ structure is
# determined by the vertically acting CZ gates that appear in the second and third vertical axis of
# the lattice. The inputs to the Z gates are classical controls, i.e. the gate is applied only if
# :math:`y_{ij}=1`.
# 
# To generate samples from this deep circuit we do the following:
#
# Recipe for sampling from the deep circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 1. Generate all bits :math:`y_{ij}` for :math:`i<4`
# uniformly at random.
# 2. Run the above circuit, controlling the Z gates on these bits. 3. Measure the
# output of the circuit to obtain the final three bits :math:`y_{41}`, :math:`y_{42}`, :math:`y_{43}`
# 
# One can show that the distribution :math:`p(\boldsymbol{y})` obtained with the above recipe is
# identical to the one we described for the IDQNN, and so the two methods are indistinguishable if
# just given samples :math:`\boldsymbol{y}`.
# 
# If these two circuits lead to the same distribution then why did we do this? The reason is that
# qubit counts on quantum hardware are still limited, so that by implementing the deep 1D circuit on a
# few qubits, you can simulate the distribution of a 2D shallow circuit on many more qubits. The
# authors used this trick to simulate a shallow IDQNN circuit on 816 qubits using a deep circuit with
# 68 qubits. To do this they actually work with a deep circuit on a 2D lattice, and map it to a
# shallow circuit on a 3D lattice. This obviously complicates things a bit (and makes drawing pictures
# a lot harder!) so we will stick to the 2D vs 1D example above; in the end, it will contain
# everything we need to understand the result for higher dimensional lattices.
# 
#
# Proving hardness for sampling
# -----------------------------
# 
# It turns out that—if we remove the classically controlled Z gates for now—the circuit structure of
# the deep circuit above is universal. That is, any :math:`n` qubit circuit with two qubit gates can
# be efficiently approximated by sequential layers of Hadamards, Z rotations and controlled Z gates on
# an :math:`n` qubit computational basis input. We can therefore use this fact to define a circuit
# that is hard to sample from classically; simply take your favourite pre-existing hardness results
# for sampling [add refs] and compile the circuit to the H, RZ, CZ gateset. We can then embed this
# into the precise structure we had above by inserting the classically controlled Z gates at every
# layer. If we happen to sample the all zero bitstring for :math:`y_{ij}` values that control these
# gates, then we will sample from this hard distribution. In this sense the distribution
# :math:`p(\boldsymbol{y})` is ‘hard’ since any classical algorithm will fail to reproduce this part
# of the distribution. Moreover, since the distribution of the IDQNN is identical, it follows that the
# corresponding IDQNN is also hard to sample from.
# 
#
# Adding inputs states
# --------------------
# 
# At this point we have a shallow circuit called an IDQNN, a way to map it to a deep circuit
# structure, and an argument that the distributions :math:`p(\boldsymbol{y})` resulting from these
# circuits are hard to sample from classically. However, we don’t yet have everything in order to be
# able to learn. The last ingredient we need comes in the form of an input :math:`x`. This will mean
# that rather than working with the the probability distribution :math:`p(\boldsymbol{y})`, we will
# work with a conditional probability distribution :math:`p(\boldsymbol{y}|x)`.
# 
# For each :math:`x`, the probability distribution :math:`p(\boldsymbol{y}|x)` corresponds to a IDQNN
# where—rather than all qubits being in the \|+> state—each input qubit can be prepared in either the
# \|0> or \|+> state, which is determined by the input :math:`x`. We therefore adapt the first step of
# our recipe for the IDQNN:
# 
# Recipe for sampling from an IDQNN with inputs
# 
# 1. Prepare each qubit in either the :raw-latex:`\ket{+}` or :raw-latex:`\ket{0}` state, depending on
#    :math:`x`.
# 2. Perform steps 2-4 as before
# 
# In order to be able to prove the result, the choice and distribution of possible input states must
# satisfy a particular property called ‘local decoupling’ (see Appendix C2 of the paper). One
# particularly simple choice that will work for our 2D IDQNN is the following choice of three inputs,
# :math:`x=0,1,2` (in the paper a different choice is used, but the result will be the same)
# 
# If :math:`x=0` all input qubits are prepared in the :raw-latex:`\ket{+}` state If :math:`x=1` all
# qubits on the ‘even diagonals’ of the lattice are prepared in :raw-latex:`\ket{+}`, the remaining
# are prepared in :raw-latex:`\ket{0}` If :math:`x=2` all qubits on the ‘odd diagonals’ of the lattice
# are prepared in :raw-latex:`\ket{+}`, the remaining are prepared in :raw-latex:`\ket{0}`
# 
# Pictorially, the choice looks like this.
# 
# If the input is 0, we already know what happens; this is just the IDQNN described in the previous
# section. If the input is 1 or 2, things get very simple. Note that the CZ gate is symmetric, and can
# be written
# 
# $ CZ = :raw-latex:`\ket{0}`:raw-latex:`\bra{0}` :raw-latex:`\otimes `:raw-latex:`\mathbb{I}` +
# :raw-latex:`\ket{1}`:raw-latex:`\bra{1}` :raw-latex:`\otimes `Z = :raw-latex:`\mathbb{I}`
# :raw-latex:`\otimes `:raw-latex:`\ket{0}`:raw-latex:`\bra{0}` + Z
# :raw-latex:`\otimes `:raw-latex:`\ket{1}`:raw-latex:`\ket{1}` $
# 
# Thus, if a CZ gate acts on the state :math:`\ket{0}` on either side, the effect is that it becomes
# the identity. Since every CZ gate hits a qubit in the :math:`\ket{0}` state for these inputs, we can
# actually remove them all. For example, the input :math:`x=1` actually just corresponds to an
# unentangled product state:
# 
#
# By performing mid-circuit measurements and resetting qubits, we can easily reproduce the statistics
# for the inputs :math:`x=1,2`. For example, for :math:`x=1`, the circuit looks as follows (we have
# removed the rotation gates for the qubits prepared in :math:`\ket{0}` since this results in a global
# phase only).
# 
# The authors argue that the conditional distribution :math:`p(\boldsymbol{y}|x)` should also be
# considered hard to sample from classically, since for the input :math:`x=0` we can use the argument
# of the previous section. For the inputs :math:`x=1` and :math:`x=2` however the resulting
# distribution has an efficient classical simulation since it corresponds to measurements made on
# unentangled single qubits.
# 
#
# The learning problem
# --------------------
# 
# We now have a classically hard conditional distribution :math:`p(\boldsymbol{y}|x)`, where each
# input :math:`x` corresponds to a IDQNN with inputs that we know how to simulate with a deeper
# circuit on fewer qubits. At this point we are ready to learn.
# 
# We first need a dataset, which we create by repeating the following :math:`N` times
# 
# - Randomly sample an input :math:`x=0,1,2`
# - Implement the deep circuit that simulates the IDQNN for this input to generate a set of outcomes
#   :math:`\boldsymbol{y}` Add the pair :math:`(x,\boldsymbol{y})`\ to the dataset
# 
# The precise definition of learning is given by definition 13 in the Appendix:
# 
# *Definition 13 (The task of learning to generate classical bitstrings). We are given a dataset of
# input-output bitstring pairs :math:`(x,\boldsymbol{y})`. Each output bitstring
# :math:`\boldsymbol{y}` is sampled according to an unknown conditional distribution
# :math:`p(\boldsymbol{y}|x)`. The goal is to learn a model from the dataset that can generate new
# output bitstrings :math:`\boldsymbol{y}` according to the unknown distribution
# :math:`p(\boldsymbol{y}|x)` for any given new input bitstring 𝑥.*
# 
# Although the above suggests the conditional distribution is unknown, we actually know a lot about
# it. In particular, we need to work under the assumption that we know the exact structure of the
# quantum circuits that produce the data, except for the rotation angles :math:`\theta_{ij}`
# (i.e. this is included in the \`prior knowledge’ of the problem). To learn, we therefore just need
# to infer the parameters :math:`\theta_ij` from the data, which will allow us to generate new data by
# simply implementing the resulting circuits. This is very different from real world problems the
# typical situation in classical generative machine learning, where a precise parametric form of the
# ground truth distribution is not known.
# 
# So how do we infer the parameters :math:`\theta_ij` from data? Consider for example the data for
# input :math:`x=1`, and the outcome :math:`y_12`. From the above circuit we see that in this case the
# outcome is produced by this single qubit circuit
# 
# This is a measurement on a rotated single qubit state, for which the expectation value for
# :math:`y_12` is
# 
# :math:`<y_12> = (1-\cos(\theta_{12}))/2`.
# 
# Rearranging this equation we have
# 
# :math:`\theta_{12} = \frac{1}{2} \arccos(1 - 2<y_12>)`.
# 
# All we have to do to infer :math:`\theta_{12}` is to look at the data for :math:`x=1`, estimate the
# expectation :math:`<y_12>` from the corresponding :math:`y_12` values, and use the above formula; no
# gradients or training required (so clearly no barren plateaus either)! Note that all we are doing
# here is a form of single qubit tomography, which you might encounter in a first course of quantum
# information.
# 
# The remaining parameters can be estimated in a similar way depending on whether they live on the
# even diagonal (which requires :math:`x=1`) or the odd diagonal (which requires :math:`x=2`). From
# the Hoeffding inequality, we can be sure that the estimates are close to the true values with high
# probability, and we thus have learned the parameters to low error. With this knowledge we can now
# sample data for the :math:`x=0` input and this is known to be classically hard, and in this sense we
# have learned the distribution.
# 
#
# Does this bring us closer to useful quantum AI?
# -----------------------------------------------
# 
# Now that the technicalities are out of the way, we can ask the really important question: does this
# bring us closer to genuinely useful QML algorithms? This question is speculated on briefly in the
# conclusion of the paper where it is admitted that \`the precise remaining steps towards useful
# generative quantum advantage remain an open question’. But why is usefulness so enigmatic? As we
# explain below, a large part of the reason is due to the fact that the setup we considered is
# significantly different to that of real-world generative machine learning problems.
# 
# The necessity of prior knowledge
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The first of these differences concerns the ‘prior knowledge’ that needs to be assumed in order to
# prove the result. Notice that, if we were given the data :math:`{(x, \boldsymbol{y})}` but told
# nothing about the underlying circuit that produced it, we would not have known how to infer the
# parameters, nor how to produce new samples once these parameters were known (since these both
# required knowledge of the circuit). That is, precise knowledge of the circuit structures that
# produced the data was necessary in order to learn. In reality, such a precise parametric form of the
# ground truth distribution is not known and models have to be learnt with vastly less prior
# knowledge. To evoke the eternal cat picture analogy, it is like we are provided with a model that
# generates near perfect pictures of cats, and all we need to do is find the right parameters for the
# colors of the fur. In their current form, these techniques therefore only appear useful for tasks
# related to quantum circuit tomography, where such knowledge is part of the problem description.
# 
# Learning from simple statistics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The second difference concerns how the parameters were learned. The parameters of modern classical
# generative models must be learned by an iterative process (often gradient descent) that extracts
# their values from complex, multidimensional correlations that are present in the statistics of the
# training data. For example, the parameters of convolutional filters in image models are adjusted to
# capture highly non-linear correlations, such as how edges, textures, and object parts co-occur
# across different spatial locations. In the quantum setup, although the distribution for the input
# :math:`x=0` is undoubtedly complex, the data that is used for learning comes from the much simpler
# product distributions in which there are no correlations between bits.
# 
# As an illustrative comparison, imagine a large classical generative model (such as a diffusion model
# or a transformer) with parameters :math:`\theta` and corresponding output distribution
# :math:`p(\boldsymbol{y})`. Suppose we want to learn the parameters :math:`\theta` from data. To do
# this we construct a conditional distribution :math:`p(\boldsymbol{y}|x)` which does the following:
# 
# - For :math:`x=0` the model samples the generative model distribution :math:`p(\boldsymbol{y})`
# - For :math:`x=1`, :math:`\boldsymbol{y}` just returns the parameters :math:`\theta`
# 
# Obviously, we can learn the parameters of the model from :math:`p(\boldsymbol{y}|x)`: we just look
# at the data for :math:`x=1` and read them off directly from :math:`\boldsymbol{y}`. Our quantum
# example is not dramatically different from this, since for the inputs :math:`x=1,2` we have a simple
# method to read off the parameters from the statistics (single qubit tomography), and this method is
# known beforehand rather than being learned from the data. In effect, we have set up the problem so
# that inferring parameters is straightforward for some inputs, whilst sampling is hard for others,
# and process of learning is therefore very different from the complex process that occurs in modern
# neural networks. We note that the specific example in the paper is more involved that this, and uses
# a higher dimensional lattice and a different set of inputs, but the strategy is the same: for each
# parameter, there is a reasonable fraction of the inputs that leaves the relevant qubit unentangled
# from the rest, and single qubit statistics reveals the desired value.
# 
# What can lead us to genuine usefulness?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In order to uncover genunine usefulness in quantum we therefore need to move to scenarios that
# mirror the assumptions of realistic learning problems. If the flipside of this is that proving
# complexity theoreic separations becomes seemingly impossible, then perhaps this is not the right
# goal to be persuing? At Xanadu, we are taking a realted but different method of attack: first
# understand what *qualitative* properties of quantum theory can be potential game-changers for
# machine learning [qft], and construct genuniely scalable algorithms that can be applied to real
# world learning problems [iqp]. Although we might have to wave goodbye to the possibility of proving
# formal complexity theoretic separations,
#
#
# References
# ----------
#
# .. [#genquantumadv]
#
#     H. Huang, M. Broughton, N. Eassa, H. Neven, R. Babbush, J. R. McClean
#     "Generative quantum advantage for classical and quantum problems."
#     `arXiv:2509.09033 <https://arxiv.org/abs/2509.09033>`__, 2025.