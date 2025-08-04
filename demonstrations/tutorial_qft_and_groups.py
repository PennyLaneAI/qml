r"""
It's all about groups: From Fast Fourier Transforms to QFTs
=========================================================================

Quantum Fourier Transforms (QFTs) are unitary operations that turn a quantum state :math:`\sum_x f(x) |x \rangle` of amplitudes :math:`f(x)` into
another quantum state whose amplitudes are the Fourier coefficients :math:`\hat{f}(x)` of :math:`f(x)`. They appear
literally everywhere in quantum computing: even if you're not interested in `Shor's algorithm
<https://pennylane.ai/codebook/shors-algorithm/shors-algorithm>`__, `hidden subgroup problems <https://pennylane.ai/qml/demos/tutorial_period_finding>`__,
`quantum phase estimation <https://pennylane.ai/qml/demos/tutorial_qpe>`__
or `quantum arithmetics <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics>`__,
you probably have worked with a circuit that starts by applying a Hadamard gate to each qubit.
Well, this is a Quantum Fourier Transform as well!

.. figure:: ../_static/demonstration_assets/qft_groups/hadamards.png
    :align: center
    :width: 75%
    :target: javascript:void(0);

    Figure 1. Applying Hadamards to each qubit is a Quantum Fourier Transform, but with respect to the "bitstring" group :math:`Z_2^n`.

The reason why you might not have appreciated this fact is that Hadamards do not form the
`famous QFT <https://pennylane.ai/qml/demos/tutorial_qft>`__ we know from Nielsen \& Chuang's standard textbook. They move into a
Fourier basis nevertheless -- only of a different *group*.

Sometimes, knowing about the Fourier-theoretic interpretation of a quantum algorithm helps to understand what is
going on under the hood. But group theory comes with a lot of jargon that can be overwhelming at first. This demo
illuminates the fascinating link between Fourier Transforms, Quantum Fourier Transforms and groups, for those who have
not taken a course in group theory (yet).

We will see that a group can be used to *define* what a Fourier transform is, a fact
that explains a lot of seemingly arbitrary assumptions in the standard (discrete and continuous) Fourier transforms.
But that's not all. Groups are implicitly used to design one of the world's most important scientific subroutines, the *Fast Fourier Transform* (FFT).
The FFT is an algorithmic implementation of a Fourier transform that is polynomially faster than the naive one. This does not
sound like much, but when transforming, say, :math:`N=10,000` numbers, the difference between of the order of :math:`N^2 = 100` Mio and
:math:`N \log N = 40,000` operations can be game changing.  It turns out that the recipe of a Fast Fourier Transform
can be implemented in "quantum parallel", which is the basic idea behind exponentially faster QFTs!

In short, groups are the fundamental structure behind quantum and classical Fourier transforms, and exploiting this
structure is one of the main reasons we believe that quantum computers could change how humans process information!

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
# .. math:: \hat{f}(k) = \frac{1}{\sqrt{N}}\sum_{i=1}^N f(x_i) e^{2 \pi i  \frac{k x}{N}}
#
# The expressions :math:`e^{2 \pi i  \frac{k x}{N}}` (and sometimes, following a different convention, :math:`e^{-2 \pi i  \frac{k x}{N}}`)
# correspond to Fourier basis functions with integer-valued
# frequencies, and the Fourier coefficient :math:`\hat{f}(x_i)`
# can be seen as the projection of :math:`f(x)` onto the :math:`i`'th basis function. The function beyond
# the interval :math:`0,..,N-1` is thought to be "periodically continued", which means that :math:`f(x_i) = f(x_i + N)`.
#
# Let's code this up:
#

import matplotlib.pyplot as plt
import numpy as np

N = 16

def f(x):
    """Some function on the integers 0,...,N-1."""
    x = x % N
    return 0.5*(x-4)**3

def f_hat(k):
    """Fourier coefficients of f."""
    projection = [ f(x) * np.exp(2 * np.pi * 1j * k * x / N)/np.sqrt(N) for x in range(N)]
    return  np.sum(projection)

integers = list(range(N))

def plot(f, f_hat, group_elements):

    f_vec = np.array([f(x) for x in group_elements])
    f_hat_vec = np.array([f_hat(k) for k in group_elements])
    x_ticks = np.array(range(len(group_elements)))
    x_labels = [str(x) for x in group_elements]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.bar(x_ticks, np.real(f_vec), color='dimgray')  # casting to real is needed in case we perform an inverse FT
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_title(f"function f")

    ax2.bar(x_ticks + 0.05, np.imag(f_hat_vec), color='lightpink', label="imaginary part")
    ax2.bar(x_ticks, np.real(f_hat_vec), color='dimgray', label="real part")
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_title("Fourier coefficients")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot(f, f_hat, integers)

######################################################################
# But why do -- at least if we don't want to incur additional headaches -- the x-values have to be equidistant? Aren't they just a
# discretisation of :math:`\mathbb{R}`? Why this notion of "periodic continuation"? Why are the basis functions of exponential form? And
# what domain, exactly, is :math:`k` chosen from? It turns out that all of these questions have an elegant answer if we
# interpret the x-domain as a group!
#
# More precisely, we have to consider the :math:`x_i` as elements from the set of integers :math:`\{0,...,N-1\}`,
# together with a prescription of how to combine two integers to a third from this set. Choosing "addition modulo N" for
# this operation (which means that :math:`(N-1)+1 = 0`), we get the *cyclic group* :math:`Z_N` as our x domain.
#
# This choice explains the equidistant property: integers are by nature equally spaced in :math:`\mathbb{R}`. It also explains the
# periodic continuation, as :math:`x_i = x_i + N` implies :math:`f(x_i) = f(x_i +N)`. The :math:`e^{2 \pi i  \frac{k x}{N}}` are
# central group-theoretic objects called the "characters" of the group; they are functions :math:`\chi_k(x) = e^{2 \pi i  \frac{k x}{N}}`
# that represent the structure of the group. Furthermore, the :math:`k` turn out to be elements
# from the "dual group", which in this case looks exactly like the original one; we can therefore think of the :math:`k`
# also as integers in :math:`\{0,...,N-1\}` and treat Fourier space as a "mirror image" of the original space -- a convenient fact
# that is not always true. Lastly,
# while the details exceed our scope here, the group perspective singles out the Fourier basis as the basis that decomposes the
# space of functions into "invariant subspaces". Hence, symmetries (such as
# `periodicity <https://pennylane.ai/qml/demos/tutorial_period_finding>`__), are particularly visible
# in the Fourier basis.
#
######################################################################
# Changing the group
# ------------------
#
# What happens if we exchange the cyclic group :math:`Z_N` by another one? First of all, if we change to the infinite group :math:`\mathbb{R}`,
# which are just the real numbers under addition, we get the continuous Fourier transform, whose characters or
# basis functions look similar to the discrete ones.
# We could also change to a direct product of groups, such as :math:`Z_N \times Z_N \times ...` or :math:`\mathbb{R} \times \mathbb{R} \times ...` and get the
# multi-dimensional Fourier transform whose basis functions are products of characters, one for each dimension.
# Choosing :math:`N=2` we get the group :math:`Z_2^n` mentioned above, where we consider N "copies" of the binary set :math:`\{0,1\}`.
# This captures boolean logic ubiquitous in quantum algorithms. The characters for :math:`Z_2^n` are :math:`e^{i k_0 x_0} \dots e^{i k_{N-1} x_{N-1}}`,
# where the product is the XOR of the two bits :math:`x_i k_i`, and evaluates either to -1 or 1. The Fourier transform with these characters is also known as the
# "Walsh transform", and reads
#
# .. math::
#           \hat{f}(k) = \sum_{x_0=0}^1 \dots \sum_{x_{N-1}=0}^1 f(x_0 \dots x_{N-1}) e^{i k_0 x_0} \dots e^{i k_{N-1} x_{N-1}},
#
# The Fourier transform over one component group :math:`Z_2` can be written as a Hadamard matrix, and the full N-dimensional
# version is therefore a product of N Hadamards -- as shown in the circuit above.
#
# Let's code up an example of a Fourier transform over the group :math:`Z_N`.
#

n = 4
bitstrings = [[int(c) for c in format(i, f'0{n}b')] for i in range(2**n)]

def g(x):
    """Majority function for length-n bitstrings."""
    count_ones = sum(x[i] for i in range(n))
    return count_ones/n

def chi(x, k):
    """Character for Z_2^n"""
    return np.prod([np.exp(1j * k[i] * x[i]) for i in range(n)])

def g_hat(k):
    """Fourier coefficients of f."""
    projection = [ g(x) * chi(x, k) for x in bitstrings]
    return  np.sum(projection)

plot(g, g_hat, bitstrings)

#####################################################################
# But we can also move to any other compact group.
# This even includes non-Abelian groups, which are those where the composition of group elements does not commute, or
# :math:`g_1 g_2 \neq g_2 g_1`. Here, Fourier coefficients become matrix-valued objects, as the characters are replaced
# by matrix-valued functions called *irreducible representations*. (If this interests you, Parsi Diaconis' book on
# data analysis with group Fourier transforms is a great resource for Fourier transforms over the group of permutations
# [#Diaconis]_ ...)
#

#####################################################################
# Transforming amplitudes
# ------------------------
#
# As mentioned, a Quantum Fourier Transform (explained in `this demo <https://pennylane.ai/qml/demos/tutorial_qft>`_ for the standard group :math:`Z_N`)
# is just a Fourier transform of the amplitudes of a quantum state.
# Each computational basis state :math:`|x \rangle` is associated with an x-value from the group, and the corresponding amplitude encodes
# the original function value :math:`f(x)`. The QFT then maps to a new quantum state where the computational basis states
# are interpreted as :math:`k` values from the dual group, and the amplitudes are the Fourier coefficients. For example:
#
# .. math:
#           \sum_{x=1}^N f(x) |x \rangle \rightarrow \sum_{k=1}^N \hat{f}(k) |k \rangle.
#
# Let us verify this, but now with a function that, written as a vector, produces a normalised quantum state!
#

norm = np.linalg.norm(np.array([f(x_) for x_ in range(N)]))

def h(x):
    """Normalised version of f above, such that \sum_x |h(x)|^2 = 1."""
    x = x % N
    return f(x) / norm

#####################################################################
# This function can be represented as a quantum state:
#

h_state = np.array([h(x) for x in range(N)])
print(np.isclose(np.vdot(h_state, h_state), 1))

#####################################################################
# The QFT produces a state that is equivalent to the "classical" Fourier transform.
#

import pennylane as qml

dev = qml.device("default.qubit", wires=4, shots=None)

@qml.qnode(dev)
def circuit_integers(state):
    """Prepare a state \sum_x f(x) |x> and apply the Fourier transform over Z_N."""
    qml.StatePrep(state, wires=range(4))
    qml.QFT(wires=range(4))
    return qml.state()

# state after the quantum Fourier transform
h_hat_state = circuit_integers(h_state)

def h_hat(k):
    """Fourier coefficients of h."""
    projection = [ h(x) * np.exp(2 * np.pi * 1j * k * x / N)/np.sqrt(N) for x in range(N)]
    return  np.sum(projection)

# vectorised function after the classical Fourier transform
h_hat_vec = np.array([h_hat(x) for x in range(N)])

print(np.allclose(h_hat_state, h_hat_vec))

#####################################################################
# But of course, ``qml.QFT`` only implements the Fourier transform with respect to the group :math:`Z_N`, which
# inteprets computational basis states as integers.
# We can interpret the bitstrings in the computational basis as different group elements, for example of the group
# :math:`Z_2^N` introduced above, which interprets the computational basis elements as "genuine" bitstrings
# with boolean logic. Both map into the Fourier basis, but with respect a different interpretation
# of the underlying function domain structure!
#

@qml.qnode(dev)
def circuit_bitstrings(state):
    """Prepare a state \sum_x f(x) |x> and apply the Fourier transform over Z_2^n."""
    qml.StatePrep(state, wires=range(4))
    for i in range(4):
        qml.Hadamard(wires=i)
    return qml.state()


h_hat_state2 = circuit_bitstrings(h_state)
print(np.allclose(h_hat_state, h_hat_state2))

#####################################################################
# The FFT: divide-and-conquer on group structure
# ----------------------------------------------------
#
# Ok, Fourier transforms are all about groups. But how is this used to come up with the Fast Fourier Transform, the
# workhorse implementation of the standard Fourier transforms that takes "only" time :math:`\mathcal{O}(|G| log |G|)`?
# And how does this give rise to Quantum Fourier Transforms with *poly-logarithmic* runtimes in the size of the group :math:`|G|`?
#
# As it turns out, the famous Cooley-Tukey implementation of a Fast Fourier transform can be interpreted as
# a decomposition into Fourier transforms on _subgroups_ of the original group. (A subgroup is a subset of the original
# set that, under the group operation, fulfills all group axioms.)
# These subgroups can sometimes be decomposed into even smaller subgroups, leading to a recursive "divide-and-conquer"
# algorithm. This technique is always possible for Abelian groups, but also for some non-Abelian groups such as the symmetric
# group of permutations. Even more
# important for us, an FFT generically gives rise to an efficient QFT [#Moore]_ (but more about that later).
#
# We will illustrate the basic idea of the FFT using the cyclic group :math:`Z_{6}`.
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
#          k \rightarrow 2k_2 + k_1 , \;\; k_1 = 0,1,2 \; k_2 = 0,1
#
# Implicitly, we are representing the set of integers :math:`\{0,1,2,3,4,5\}` first as :
# .. math:: \{3*0+0, 3*0+1, 3*0+2, 3*1+0, 3*1+1, 3*1+2\},
#
# and then as
#
# .. math:: \{2*0+0, 2*0+1, 2*1+0, 2*1+1, 2*2+0, 2*2+1\}.
#
# While doubling the amount of variables, the
# new variables only run over fewer integers. This
# will allow us to "chop" the full Fourier transform into Fourier transforms over :math:`x_1, k_1` only, and is the
# secret of the runtime reduction.
#
# Rewriting the Fourier transform in these variables reads
#
# .. math::
#           \hat{f}(k_2, k_1) = \frac{1}{\sqrt{3}}  \sum_{x_2=0}^2 e^{\frac{2 \pi i}{6} (2k_2+k_1) x_2} \frac{1}{\sqrt{2}} \sum_{x_1=0}^1 e^{\frac{2 \pi i}{2} x_1 k_1 } f(x_1, x_2)
#
# Besides some reordering, there is one slightly non-trivial identity we used above:
#
# .. math::
#           e^{\frac{2 \pi i}{6} x_1 3(2k_2+k_1) } = e^{\frac{2 \pi i}{2} x_1k_1 } \underbrace{e^{2 \pi i x_1 k_2 }}{1}  = e^{\frac{2 \pi i}{2} x_1 k_1 }
#
# Essentially, the change in variables turns the Fourier basis function over period 6 into a Fourier basis function with period 2,
# which makes the dependency on :math:`k_2` disappear. This effectively turns the Fourier transform into a sum of "smaller" Fourier transforms
# over :math:`3x_1`, namely  :math:`\{0, 3\}`, :math:`\{1, 4\}`, and :math:`\{2, 5\}`. These are combined with
# an appropriate "twiddle factor" :math:`e^{\frac{2 \pi i}{6} (2k_2+k_1) x_2}` that weighs the building blocks.
#
# We explicitly implement this with the function from above:
#

N = 6

def f_twovars(x1, x2):
    """Function f from before, but with input split into two variables, and running from 0,...5."""
    x = 3*x1 + x2
    x = x % N
    return 0.5*(x-4)**3

def f_subgroup(x2, k1):
    """Fourier transform over a smaller group or coset."""
    projection = [ f_twovars(x1, x2) * np.exp(2 * np.pi * 1j * k1 * x1 / 2)/np.sqrt(2) for x1 in range(2)]
    return  np.sum(projection)

def f_hat_fft(k1, k2):
    """Putting the smaller transforms together."""
    res = 0
    for x2 in range(3):
        twiddle_factor = np.exp(2 * np.pi * 1j * (2*k2+k1) * x2 / 6) / np.sqrt(3)
        res += twiddle_factor * f_subgroup(x2, k1)
    return res

f_hat_vec_fft = np.array([f_hat(2*k2+k1) for k1 in range(2) for k2 in range(3)])
print(f_hat_vec_fft)

#######################################################################
#
# Let's compare the result to the DFT from before.
#

def f(x):
    """Same function f as before, but on the integers 0,...,5."""
    x = x % N
    return 0.5*(x-4)**3

def f_hat(k):
    """Fourier coefficients of f."""
    projection = [ f(x) * np.exp(2 * np.pi * 1j * k * x / N)/np.sqrt(N) for x in range(N)]
    return  np.sum(projection)

f_hat_vec = np.array([f_hat_fft(k1, k2) for k1 in range(2) for k2 in range(3)])
print(np.allclose(f_hat_vec, f_hat_vec_fft))



#######################################################################
# Quantifying the runtime gains
# +++++++++++++++++++++++++++++
#
# The FFT algorithm devises to compute :math:`\hat{f}{2k_2+k_1}` in two steps:
#
# 1. First, compute the terms in the inner sum, :math:`\tilde{f}(x_2, k_1) = \sum_{x_1=0}^1 e^{\frac{2 \pi i}{2} x_1 k_1 } f(3 x_1 + x_2)` or `f_subgroup`.
#    There are 9 such terms (as we have 3 options to pick :math:`k_1` and 3 options to pick :math:`x_2`). Each term requires
#    summing over 2 expressions (as :math:`x_1` runs over 2 values). Overall, there are 18 terms involved.
#
# 2. Second, combine the 9 terms using the outer sum :math:`\hat{f}(k_2, k_1) = \sum_{x_2=0}^2 e^{\frac{2 \pi i}{6} (2k_2+k_1) x_2}  \tilde{f}(k_1, x_2)`.
#    This computes 2*3=6 Fourier coefficients, and each has to sum over 3 terms (as :math:`x_2` runs over 3 values).
#    Overall, there are 12 terms involved.
#
# Together, the FFT algorithm includes 12 + 18 = 30 terms. The naive sum, instead, would compute the 6 Fourier coefficients
# with a sum over 6 terms each, using 6*6=36 terms altogether. Of course, in this "mini" example the savings are not very
# impressive. The overall scaling however reduces from :math:`O(N^2)` to :math:`O(N \mathrm{log}N)`, which is a quadratic
# speedup, and made it possible for the FFT to become one of the most important algorithms in science and engineering.
#

#######################################################################
# A group-theoretic interpretation
# ++++++++++++++++++++++++++++++++
#
# As hinted at before, the change of variables that was the core idea behind the Cooley-Tukey implementation of the Fast Fourier Transform is
# in fact a decomposition of the original group -- here :math:`Z_{6}` -- into a subgroup isomorphic to :math:`Z_{2}` with the
# elements :math:`\{0,3\}`, and its "cosets" or shifted copies :math:`\{2,4\}` and :math:`\{3,5\}` [#Maslen, #Rockmore]_.
# The new variable :math:`x_2` selects between the copy. For example, if :math:`5=3*x_1 + x_2` then :math:`x_2=2` selects the third copy of the subgroup,
# while :math:`x_1=1` selects the second element within the copy. It is a fundamental result in group theory that
# "shifted" copies of a subgroup tile the entire group, which is why the new coordinates can always be used to express any
# element :math:`x \in G`.
#
# The change of the :math:`k` variable is related to the concept of "restricting a character" to the subgroup.
# Essentially, it allows us turn the characters, or Fourier basis functions :math:`e^{\frac{2 \pi i}{6} x k }`
# related to the original group, into characters of the subgroup, :math:`e^{\frac{2 \pi i}{2} x_1 k_1 }`.
# Such a trick generalises to much more complicated groups, including those where there is no "cyclic" notion of
# ordered integers.
#

#######################################################################
# From FFTs to QFTs
# -----------------
#
# Well then, Fourier transforms are all about groups, and sometimes allow the construction of fast algorithms. But we also claimed [#Moore]_ that
# the FFT suggests a QFT which transforms the amplitudes of a quantum state in time that is logarithmic in the group size.
# The generic algorithm is not trivial, but it basically works upwards in the "tower" of subgroups, and combines the
# results. The computational basis state first holds information on the group element expressed as "coordinates" :math:`x_1, x_2` that tells us in
# which subgroup (copy) it is in for each level of the subgroup tower. The coordinates then successively get replaced by frequencies,
# as intermediate results like :math:`\tilde{f}(x_2, k_1)` above get computed. Finally, the computational basis state refers
# to the frequency components :math:`k_1, k_2`, and the amplitudes correspond to the Fourier coefficients. The
# crucial point is that the Fourier transforms over the subgroup copies are implemented in quantum parallel - this is
# the magic of the quantum speedup!
#
# By the way, if you look closely at the circuit for the QFT, you'll find the divide-and-conquer strategy
# of the FFT.
#

@qml.qnode(dev)
def qft():
    qml.QFT(wires=range(4))
    return qml.state

fig, ax = qml.draw_mpl(qft, level=2)()
fig.show()


#######################################################################
# For simplicity, let's look at this schematic circuit diagram.
#
# .. figure:: ../_static/demonstration_assets/qft_fft/qft3.jpeg
#     :align: center
#     :width: 75%
#     :target: javascript:void(0);
#
#     Figure 2. Circuit of the standard QFT.
#
#
# Firstly, note that the algorithm has a recursive structure on subgroups.
# The most significant bit in a binary representations of the cyclic group, say :math:`Z_{2^4}` of elements
# :math:`0 \hat{=} 0000,...,2^n-1 \hat{=} 1111`, "cuts" the integers into a subgroup :math:`Z_{2^3}` with elements :math:`0 \hat{=} 0000,...,7 \hat{=} 0111`
# and its coset :math:`8 \hat{=} 1000,...,15 \hat{=} 1111`. Hence, by applying a gate pattern to qubits 1,...,n, then to
# 2,...,n etc implements an operation to smaller and smaller subgroups.
# Secondly, note that the controlled operations somehow control for the "twiddle factor" above!
# And lastly, the Hadamard gate sums the "smaller" Fourier transforms performed on the :math:`q_1 = 0` and :math:`q_1=1`
# branches together (with a phase that is taken into account by the twiddle factor).
#

#######################################################################
# Conclusion
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
# The take-away from this tutorial are three important insights:
#
# 1. The Fourier transform is defined with respect to functions over groups. The usual discrete Fourier transform, for
#    example, uses the cyclic group of integers.
# 2. The Fast Fourier Transform exploits that for some groups, a Fourier transform can be decomposed into smaller blocks on copies of subgroups.
#    This provides a speedup to nearly-linear runtime in the size of the group.
# 3. The Quantum Fourier Transform implements the smaller blocks of the FFT in "quantum parallel", thereby realising a runtime
#    that is logarithmic in the size of the group.
#
# Group structure and Fourier transforms are important building blocks of quantum algorithms, and a fascinating
# reason for why quantum computers might offer unique ways of information processing. Of course, the caveat is
# that quantum implementations of Fourier transforms cannot compute Fourier coefficients directly, as they are
# amplitudes of a quantum state. All we can do is sample from the final state, or manipulate it in interesting ways --
# creating a wonderful playground for quantum algorithms design.
#

#######################################################################
# References
# ------------
#
# .. [#Moore]
#    Moore, C., Rockmore, D. and Russell, A., 2006.
#    Generic quantum Fourier transforms. ACM Transactions on Algorithms (TALG), 2(4), pp.707-723.
# 
# .. [#Diaconis]
#    Diaconis, P., 1988. Group representations in probability and statistics.
#    Lecture notes-monograph series, 11, pp.i-192.
#
# .. [#Maslen]
#    Maslen, D.K. and Rockmore, D.N., 2001. The Cooley-Tukey FFT and group theory.
#    Notices of the AMS, 48(10), pp.1151-1160. `<https://library2.msri.org/books/Book46/files/11maslen.pdf>`__
#
# .. [#Rockmore]
#    Rockmore, D.N., 2002, November. Recent progress and applications in group FFTs.
#    In Conference Record of the Thirty-Sixth Asilomar Conference on Signals, Systems and Computers, 2002.
#    (Vol. 1, pp. 773-777). IEEE. `<https://www.cs.dartmouth.edu/~rockmore/nato-1.pdf>`__.
#
#
# About the author
# ----------------
#
