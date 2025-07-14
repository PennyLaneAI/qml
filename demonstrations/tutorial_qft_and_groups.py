r"""
It's all about groups: how quantum computers implement Fourier transforms
=========================================================================

Quantum Fourier Transforms (QFTs) are unitary operations that turn a quantum state of amplitudes :math:`f(x)` into another quantum state whose amplitudes are the Fourier coefficients :math:`\hat{f}(x)` of :math:`f(x)`. They appear literally everywhere in quantum computing: even if you're not interested in `Shor's algorithm <https://pennylane.ai/codebook/shors-algorithm/shors-algorithm>`__, `hidden subgroup problems <https://pennylane.ai/qml/demos/tutorial_period_finding>`__, `quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe>`__ or `quantum arithmetics <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics>`__, you probably have worked with a circuit that starts by applying a Hadamard gate to each qubit. Well, this is a Quantum Fourier Transform as well! 

.. figure:: ../_static/demonstration_assets/qft_groups/FIG.png
    :align: center
    :width: 75%
    :target: javascript:void(0);

    Figure 1. Applying Hadamards to each qubit is a Quantum Fourier Transform, but with respect to the "bitstring" group :math:`Z_2^n`.

The reason why you might not have appreciated this fact is that Hadamards do not form the `famous QFT <https://pennylane.ai/qml/demos/tutorial_qft>__` we know from Nielsen and Chuang. They move into a Fourier basis nevertheless -- only of a different _group_. 

Sometimes, knowing about the Fourier-theoretic interpretation of a quantum algorithm helps to understand what is going on under the hood. But group theory comes with a lot of jargon that can be overwhelming at first. This demo illuminates the fascinating link between Fourier Transforms, Quantum Fourier Transforms and groups for those who have not taken a course in group theory. We will see that a group can be used to _define_ what a Fourier transform is, a fact that explains a lot of seemingly arbitrary assumptions in the standard (discrete and continuous) Fourier transform. Groups are also implicitly used in one of the world's most important scientific subroutines, the _Fast Fourier Transform_, which is an implementation of a Fourier transform that is polynomially faster than the naive one. Finally, in a fascinating twist laid out in Moore, ... [#Moore]_, the recipe of a Fast Fourier Transform can be implemented in ``quantum parallel'', which is the basic idea behind the exponentially faster _Quantum Fourier Transforms_ applied to amplitudes of quantum states. In short, groups are the fundamental structure behind quantum and classical Fourier transforms, and exploiting this structure is one of the main reasons we believe that quantum computers could change how humans process information!

But let us start with the basics...
"""

#####################################################################
# The Fourier transform through a group-theoretic lense
# -----------------------------------------------------
# 
# Let's focus on the `discrete Fourier transform <https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`__ for now. As a reminder, this mathematical operation 
# transforms a sequence :math:`f_1,...,f_N` of complex numbers into another sequence of complex numbers. This is already suggestive of another notation, 
# closer in spirit to the continuous Fourier transform, where the complex values are written as a function :math:`f(x_1), ...,f(x_N)` evaluated or "sampled" at equidistant 
# x-values :math:`x_1,...,x_N`. The Fourier coefficients are then given as
# 
# .. math:
#           \hat{f}(k) = \sum_{i=1}^N f(x_i) e^{-2 \pi i  \frac{k x}{N}}
# 
# The expressions :math:`e^{-2 \pi i  \frac{k x}{N}}` correspond to Fourier basis functions with integer-valued frequencies, and the Fourier coefficient :math:`\hat{f}(x_i)` 
# can be seen as the projection of :math:`f(x)` onto the :math:`i`'th basis function. The function outside of the N x-values is thought to be "periodically continued",
# which means that we assume that :math:`f(x_i) = f(x_i + N)`.
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
# periodic continuation, as :math:`x_i = x_i + N` implies :math:`f(x_i) = f(x_i +N)`. The :math:`e^{-2 \pi i  \frac{k x}{N}}` are
# central group-theoretic objects called the "characters" of the group; they are functions :math:`\chi_k(x) = e^{-2 \pi i  \frac{k x}{N}}`
# that represent the structure of the group (in ways that exceed this tutorial by far). Finally, the :math:`k` turn out to be elements
# from the "dual group", which in this case looks exactly like the original one; we can therefore think of the :math:`k`
# also as integers :math:`\{0,...,N-1\}` and treat Fourier space as a mirror image of the original space.
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
#           \hat{f}(k) = \sum_{x_0=0}^1 \dots \sum_{x_{N-1}=0}^1 f(x_0 \dots x_{N-1}) e^{- i k_0 x_0} \dots e^{- i k_{N-1} x_{N-1}},
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
# Meet the Fast Fourier Transform
# -------------------------------
#
# Ok, Fourier transforms are all about groups. But how is this used to come up with the Fast Fourier Transform, the
# workhorse implementation of the standard Fourier transforms that takes "only" time :math:`\mathcal{O}(|G| log |G|)`?
# And how does this give rise to Quantum Fourier Transforms with _logarithmic_ runtimes in the size of the group :math:`|G|`?
# 
#
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
# About the author
# ----------------
#
