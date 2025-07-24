r"""
It's all about groups: how quantum computers implement Fourier transforms
=========================================================================

Quantum Fourier Transforms (QFTs) are unitary operations that turn a quantum state :math:`\sum_x f(x) |x \rangle` of amplitudes :math:`f(x)` into
another quantum state whose amplitudes are the Fourier coefficients :math:`\hat{f}(x)` of :math:`f(x)`. They appear
literally everywhere in quantum computing: even if you're not interested in `Shor's algorithm
<https://pennylane.ai/codebook/shors-algorithm/shors-algorithm>`__, `hidden subgroup problems <https://pennylane.ai/qml/demos/tutorial_period_finding>`__,
`quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe>`__
or `quantum arithmetics <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics>`__,
you probably have worked with a circuit that starts by applying a Hadamard gate to each qubit.
Well, this is a Quantum Fourier Transform as well!

.. figure:: ../_static/demonstration_assets/qft_groups/FIG.png
    :align: center
    :width: 75%
    :target: javascript:void(0);

    Figure 1. Applying Hadamards to each qubit is a Quantum Fourier Transform, but with respect to the "bitstring" group :math:`Z_2^n`.

The reason why you might not have appreciated this fact is that Hadamards do not form the
`famous QFT <https://pennylane.ai/qml/demos/tutorial_qft>__` we know from Nielsen and Chuang. They move into a
Fourier basis nevertheless -- only of a different _group_.

Sometimes, knowing about the Fourier-theoretic interpretation of a quantum algorithm helps to understand what is
going on under the hood. But group theory comes with a lot of jargon that can be overwhelming at first. This demo
illuminates the fascinating link between Fourier Transforms, Quantum Fourier Transforms and groups, for those who have
not taken a course in group theory (yet). We will see that a group can be used to _define_ what a Fourier transform is, a fact
that explains a lot of seemingly arbitrary assumptions in the standard (discrete and continuous) Fourier transforms.

But that's not all. Groups are implicitly used to design one of the world's most important scientific subroutines, the _Fast Fourier Transform_ (FFT).
The FFT is an algorithmic implementation of a Fourier transform that is polynomially faster than the naive one. This does not
sound like much, but when transforming, say, :math:`N=10,000` numbers, the difference between of the order of :math:`N^2 = 100` Mio and
:math:`N \log N = 40,000` operation can be game changing.  It turns out that the recipe of a Fast Fourier Transform
can be implemented in ``quantum parallel'', which is the basic idea behind exponentially faster QFTs!

In short, groups are the fundamental structure behind quantum and classical Fourier transforms, and exploiting this
structure is one of the main reasons one might believe that quantum computers could change how humans process information!

But let us start with the basics...
"""

#####################################################################
# The Fourier transform through a group-theoretic lense
# -----------------------------------------------------
# 
# Let's focus on the `discrete Fourier transform <https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`__
# for now. As a reminder, this mathematical operation
# transforms a sequence :math:`f_1,...,f_N` of complex numbers into another sequence of complex numbers.
# This is already suggestive of another notation,
# closer in spirit to the continuous Fourier transform, where the complex values are written as a
# function :math:`f(x_1), ...,f(x_N)` evaluated or "sampled" at equidistant
# x-values :math:`x_1,...,x_N`. The Fourier coefficients are then given as
# 
# .. math:
#           \hat{f}(k) = \sum_{i=1}^N f(x_i) e^{2 \pi i  \frac{k x}{N}}
# 
# The expressions :math:`e^{-2 \pi i  \frac{k x}{N}}` correspond to Fourier basis functions with integer-valued
# frequencies, and the Fourier coefficient :math:`\hat{f}(x_i)`
# can be seen as the projection of :math:`f(x)` onto the :math:`i`'th basis function. The function beyond
# the interval :math:`0,..,N-1` is thought to be "periodically continued", which means that :math:`f(x_i) = f(x_i + N)`.
#
# Let's code this up:
#

def f(x): 
    return None
    
# DFT

#####################################################################
# But why do -- at least if we don't want to incur additional headaches -- the x-values have to be equidistant? Aren't they just a discretisation
# of :math:`\mathbb{R}`? Why this notion of "periodic continuation"? Why are the basis function of this exponential form? And
# what domain, exactly, is :math:`k` chosen from? It turns out that all of these questions have an elegant answer if we
# interpret the x-domain as a group. More precisely, we have to consider the :math:`x_i` as elements from the set of integers :math:`\{0,...,N-1\}`,
# together with a prescription of how to combine two integers to a third from this set. Choosing "addition modulo N" for
# this operation (which means that :math:`(N-1)+1 = 0`), we get the _cyclic group_ :math:`Z_N` as our x domain.
#
# This choice explains the equidistant property: integers are by nature equidistant in :math:`\mathbb{R}`. It also explains the
# periodic continuation, as :math:`x_i = x_i + N` implies :math:`f(x_i) = f(x_i +N)`. The :math:`e^{2 \pi i  \frac{k x}{N}}` are
# central group-theoretic objects called the "characters" of the group; they are functions :math:`\chi_k(x) = e^{2 \pi i  \frac{k x}{N}}`
# that represent the structure of the group (in ways that exceed this tutorial by far). Furthermore, the :math:`k` turn out to be elements
# from the "dual group", which in this case looks exactly like the original one; we can therefore think of the :math:`k`
# also as integers :math:`\{0,...,N-1\}` and treat Fourier space as a mirror image of the original space. Lastly,
# while the details exceed our scope here, the group perspective also singles out the Fourier basis as the basis that decomposes the
# space of functions into "invariant subspaces". Hence, symmetries (such as
# `periodicity <https://pennylane.ai/qml/demos/tutorial_period_finding>`__), are particularly visible
# in the Fourier basis.
#
#####################################################################
# Changing the group
# -------------------
#
# What happens if we exchange the group :math:`Z_N` by another one? First of all, if we change to the infinite group :math:`\mathbb{R}`,
# which are just the real numbers under addition, we get the continuous Fourier transform, whose characters look like the discrete ones.
# Next, we can change to a direct product of groups, such as :math:`Z_N x Z_N x ...` or :math:`\mathbb{R} x \mathbb{R} x ...` and get the
# multi-dimensional Fourier transform whose basis functions are products of characters, one for each dimension.
# The group :math:`Z_2^N` mentioned above is a special case, where we consider N dimensions of the integer set :math:`\{0,1\}`,
# which leads to the boolean logic that is ubiquitous in quantum algorithms. The Fourier transform is also known as the
# "Walsh transform", and reads
#
# .. math:
#           \hat{f}(k) = \sum_{x_0=0}^1 \dots \sum_{x_{N-1}=0}^1 f(x_0 \dots x_{N-1}) e^{i k_0 x_0} \dots e^{i k_{N-1} x_{N-1}},
#
# where :math:`k_i x_i` is the XOR of the two bits, or the product modulo 2. The characters :math:`e^{- i k_i x_i}` evaluate either to
# -1 or 1. The Fourier transform over one such group :math:`Z_2` can be written as a Hadamard matrix, and the full N-dimensional
# version is therefore a product of N Hadamards -- as shown in the circuit above.
#
# Let's code up an example of a Fourier transform over the group :math:`Z_2^N`.
#
#
# But we can also move to any other compact group.
# This even includes non-Abelian groups, which are those where the composition of group elements does not commute, or
# :math:`g_1 g_2 \neq g_2 g_1`. Here Fourier coefficients become matrix-valued objects, as the characters are replaced
# by matrix-valued functions called _irreducible representations_. (If this interests you, Parsi Diaconis' book on
# data analysis with group Fourier transforms is a great resource for Fourier transforms over the group of permutations\
# [#Diaconis]__...)
#

def f(x):
    return None

#####################################################################
# Transforming amplitudes
# ------------------------
#
# As mentioned, a Quantum Fourier Transform is just a Fourier transform of the amplitudes of a quantum state.
# Each computational basis state :math:`|x \rangle` is associated with an x-value from the group, and the corresponding amplitude encodes
# the original function value :math:`f(x)`. The QFT then maps to a new quantum state where the computational basis states
# are interpreted as :math:`k` values from the dual group, and the amplitudes are the Fourier coefficients. For example:
#
# .. math:
#           \sum_{x=1}^N f(x) |x \rangle \rightarrow \sum_{k=1}^N f(k) |k \rangle.
#
# Here is an interesting subtlety: The standard discrete QFT over :math:`Z_2^n`, and the Hadamard Fourier transform
# over :math:`Z_N` interpret the bitstrings in the computational basis differently, and hence map the same state to very different
# Fourier spaces. For example, using :math:`Z_N`, the computational basis state :math:`| 010 \rangle` is interpreted
# as the integer :math:`| 2 \rangle`, while using :math:`Z_2^N` interprets it genuinely as a collection of three bits.
# In other words, when we see Hadamards at the beginning of a quantum algorithm, we implicitly think of qubits as relating
# to a binary data structure, whereas the standard QFT interprets qubits as the digital representations of an integer.
#
# To show this, let's apply the two QFTs to the same state:
#


#####################################################################
# The FFT: divide-and-conquer on group structure
# ----------------------------------------------------
#
# Ok, Fourier transforms are all about groups. But how is this used to come up with the Fast Fourier Transform, the
# workhorse implementation of the standard Fourier transforms that takes "only" time :math:`\mathcal{O}(|G| log |G|)`?
# And how does this give rise to Quantum Fourier Transforms with _poly-logarithmic_ runtimes in the size of the group :math:`|G|`?
#
# As it turns out, the famous Cooley-Tukey implementation of a Fast Fourier transform CITE can be interpreted as
# a decomposition into Fourier transforms on _subgroups_ of the original group. (A subgroup is a subset of the original
# set that, under the group operation, fulfills all group axioms.)
# These subgroups can sometimes be decomposed into even smaller subgroups, leading to a recursive "divide-and-conquer"
# algorithm. This technique is always possible for Abelian groups, but also for some non-Abelian groups. Even more
# important for us, such an FFT generically gives rise to an efficient QFT!
#
# We will illustrate the basic idea using the cyclic group :math:`Z_{6}`.
#

#######################################################################
# The idea of Cooley-Tukey
# +++++++++++++++++++++++++
#
# Consider the Fourier transform of a function on :math:`Z_{6}` -- which for all matters and purposes one can think of
# as a function on the integers from 0 to 5:
#
# .. math::
#           \hat{f}(k) = \sum_{x=0}^5 f(x) e^{\frac{2 \pi i}{6} x k }
#
# The entire trick of the Cooley-Tukey algorithm is a change of variables,
#
# .. math::
#          x \rightarrow 3 x_1 + x_2, \;\; x_1 = 0,1, \; x_2 = 0,1,2
#          k \rightarrow 2k_2 + k_1 , \;\; k_1 = 0,1,2 \; x_2 = 0,1
#
# Implicitly, we are representing the set of integers :math:`\{0,1,2,3,4,5\}` first as :math:`\{3*0+0, 3*0+1, 3*0+2, 3*1+0, 3*1+1, 3*1+2\}`,
# and then as :math:`\{2*0+0, 2*0+1, 2*1+0, 2*1+1, 2*2+0, 2*2+1\}`. While doubling the amount of variables, the
# new variables only run over fewer integers. This
# will allow us to "chop" the full Fourier transform into Fourier transforms over :math:`x_1, k_1` only, and is the
# secret of the runtime reduction.
#
# Rewriting the Fourier transform in these variables reads
#
# .. math::
#           \hat{f}(k_2, k_1) = \sum_{x_2=0}^2 e^{\frac{2 \pi i}{6} (2k_2+k_1) x_2} \sum_{x_1=0}^1 e^{\frac{2 \pi i}{2} x_1 k_1 } f(x_1, x_2)
#
# Besides some reordering, there is one slightly non-trivial identity we used above:
#
# .. math::
#           e^{\frac{2 \pi i}{6} x_1 3(2k_2+k_1) } = e^{\frac{2 \pi i}{2} x_1k_1 } \underbrace{e^{2 \pi i x_1 k_2 }}{1}  = e^{\frac{2 \pi i}{2} x_1 k_1 }
#
# Essentially, the change in variables turns the Fourier basis function over period 6 into a Fourier basis function with period 2,
# which makes the dependency on :math:`k_2` disappear. This effectively turns the Fourier transform into a sum over "smaller" Fourier transforms.
#
#######################################################################
# Quantifying the runtime gains
# +++++++++++++++++++++++++++++++
#
# The FFT algorithm devises to compute :math:`\hat{f}{2k_2+k_1}` in two steps:
# 1. First, compute the terms in the inner sum, :math:`\tilde{f}(k_1, x_2) = \sum_{x_1=0}^1 e^{\frac{2 \pi i}{2} x_1 k_1 } f(3 x_1 + x_2)`.
#    There are 9 such terms (as we have 3 options to pick :math:`k_1` and 3 options to pick :math:`x_2`). Each term requires
#    summing over 2 expressions (as :math:`x_1` runs over 2 values). Overall, there are 18 terms involved.
#
# 2. Second, combine the 9 terms using the outer sum :math:`\hat{f}(k_2, k_1) = \sum_{x_2=0}^2 e^{\frac{2 \pi i}{6} (2k_2+k_1) x_2}  \tilde{f}(k_1, x_2)`.
#    This computes 2*3=6 Fourier coefficients, and each has to sum over 3 terms (as :math:`x_2` runs over 3 values).
#    Overall, there are 12 terms involved.
#
# Together, the FFT algorithm includes 12 + 18 = 30 terms. The naive sum, instead, would compute the 6 Fourier coefficients
# with a sum over 6 terms each, using 6*6=36 terms altogether. Of course, in this mini example the savings are not very
# impressive. The overall scaling however reduces from :math:`O(N^2)` to :math:`O(N \mathrm{log}N)`, which is a quadratic
# speedup, and made the FFT one of the most important algorithms in science and engineering.
#
#######################################################################
# A group-theoretic interpretation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# As we hinted at before, the change of variables that was the core idea behind the Cooley-Tukey implementatino of the Fast Fourier Transform is
# in fact a decomposition of the original group -- here :math:`Z_{6}` -- into a subgroup :math:`Z_{3}` with the
# elements :math:`\{0,1,2\}` and its "coset" :math:`\{3,4,5\}`. The new variable :math:`x_1` then selected between the two.
# The new variable :math:`x_2` picked an element within the subgroup or coset. For example, :math:`x_1=1`
# creates a shift of 3*1 = 3 selecting the coset, and :math:`x_2=2` picks the second element in this coset. While in
# our small example, there was only one coset, a subgroup usually gives rise to many cosets that are "shifted copies"
# of the subgroup. Together they tile the entire group, which is why the new coordinates can always be used to express any
# element :math:`x`.
#
# The change of the :math:`k` variable is related to the concept of "restricting a character" to the subgroup, in our case
# :math:`\{0,1,2\}`. Essentially, it allows us turn the characters, or Fourier basis functions :math:`e^{\frac{2 \pi i}{6} x k }`,
# related to the original group into characters of the subgroup, :math:`e^{\frac{2 \pi i}{2} x_1 k_1 }`. CHECK FACTORS.
# Such a trick generalises to much more complicated groups, including those where there is no "cyclic" notion of
# ordered integers.
#
#######################################################################
# From FFTs to QFTs
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#
#
#
# References
# ------------
#
# .. [#Moore]
#    
#     Scott Aaronson, "How Much Structure Is Needed for Huge Quantum Speedups?", 
#     `arXiv:2209.06930 <https://arxiv.org/pdf/2209.06930>`__, 2022
# 
# .. [#Diaconis]
#    
#     Andrew Childs, Vim van Dam, "Quantum algorithms for algebraic problems", 
#     `Reviews of Modern Physics 82.1: 1-52. <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.82.1>`__, 2010
#
# .. [#Maslen]
#
#    https://library2.msri.org/books/Book46/files/11maslen.pdf
#
# .. [#Rockmore]
#
#    https://www.cs.dartmouth.edu/~rockmore/nato-1.pdf
#
#
# About the author
# ----------------
#
