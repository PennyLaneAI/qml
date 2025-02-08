r""".. role:: html(raw)
   :format: html

JIT compiling Shor's algorithm with PennyLane and Catalyst
===============================================================

.. meta::
    :property="og:description": JIT compile Shor's algorithm from end-to-end with PennyLane and Catalyst.

    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets//fano.png

.. related::

    tutorial_iterative_quantum_phase_estimation IPE demo (update when available)

*Author: Olivia Di Matteo — Posted: X Y 2024. Last updated: X Y 2024.*
"""

##############################################################################
# The past few years stimulated a lot of discussion about *hybrid
# quantum-classical algorithms*. For a time, this terminology was synonymous
# with *variational algorithms*. However, integration with classical
# co-processors is necessary for every quantum algorithm, even ones considered
# quintessentially quantum.
#
# Shor's famous factoring algorithm [#Shor1997]_ is one such example. Have a look at the
# example code below:

import jax.numpy as jnp


def shors_algorithm(N):
    p, q = 0, 0

    while p * q != N:
        a = jnp.random.choice(jnp.arange(2, N - 1))

        if jnp.gcd(N, a) != 1:
            p = jnp.gcd(N, a)
            return p, N // p

        guess_r = guess_order(N, a)

        if guess_r % 2 == 0:
            guess_square_root = (a ** (guess_r // 2)) % N

            if guess_square_root not in [1, N - 1]:
                p = jnp.gcd(N, guess_square_root - 1)
                q = jnp.gcd(N, guess_square_root + 1)

    return p, q


######################################################################
# If you saw this code out-of-context, would it even occur to you that it was a
# quantum algorithm? There are no quantum circuits in sight, and the only "q" is
# a variable name!
#
# As quantum hardware continues to scale up, the way we think about quantum
# programming is evolving in tandem. Writing circuits gate-by-gate for
# algorithms with hundreds or thousands of qubits is unsustainable. Morever, a
# programmer doesn't actually need to know anything quantum is happening, if the
# software library can generate and compile appropriate quantum code (though,
# they should probably have at least some awareness, since the output of
# ``guess_order`` is probabilistic!).  This raises some questions: what gets
# compiled, where and how does compilation happen, and what do we gain?
#
# Over the past year, PennyLane has become increasingly integrated with
# `Catalyst
# <https://docs.pennylane.ai/projects/catalyst/en/latest/index.html>`_, which
# for just-in-time compilation of classical and quantum code together. In this
# demo, we will leverage this to develop an implementation of Shor's factoring
# algorithm that is just-in-time compiled from end-to-end, classical control
# structure and all. In particular, we will see how to leverage Catalyst's
# mid-circuit measurement capabilities to reduce the size of quantum circuits,
# and how JIT compilation enables faster execution overall.
#
# Crash course on Shor's algorithm
# --------------------------------
#
# Looking back at the code above, we can see that Shor's algorithm is broken
# down into a couple distinct steps. Suppose we wish to decompose an integer
# :math:`N` into its two constituent prime factors, :math:`p` and :math:`q`.
#
#  - First, we randomly select a candidate integer, :math:`a`, between 2 and
#    :math:`N-1` (before proceeding, we double check that we did not get lucky and randomly select one of the true factors)
#  - Using our chosen a, we proceed to the quantum part of the algorithm: order-finding.
#    Quantum circuits are generated, and the circuit is executed on a device. The results
#    are used to make a guess for a non-trivial square root.
#  - If the square root is non-trivial, we test whether we found the factors. Otherwise, we try again
#    with more shots. Eventually, we try with a different value of a.
#
# For a full description of Shor's algorithm, the interested reader is referred
# to the relevant module in the `PennyLane Codebook
# <https://pennylane.ai/codebook/10-shors-algorithm/>`_. What's important here
# for us is to note that for each new value of :math:`a` (and more generally,
# each possible :math:`N`), we must compile and optimize many large quantum
# circuits, each of which consists of many nested subroutines.
#
# In both classical and quantum programming, compilation is the process of
# translating operations expressed in high-level languages down to the language
# of the hardware. As depicted below, it involves multiple passes over the code,
# through one or more intermediate representations, and both machine-independent and
# dependent optimizations.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/compilation-stack.svg
#    :scale: 75%
#    :align: center
#    :alt: The quantum compilation stack.
#
#    High-level overview of the quantum compilation stack and its constituent
#    parts. Each step contains numerous subroutines and passes of its own, and
#    many require solving computationally hard problems (or very good heuristic
#    techniques).
#
# Developing automated compilation tools is a very active and important area of
# research, and is a major requirement for today's software stacks. Even if a
# library contains many functions for pre-written quantum circuits, without a
# proper compiler a user would be left to optimize and map them to hardware by hand.
# This is an extremely laborious (and error-prone!) process, and furthermore,
# is unlikely to be optimal.
#
# However, our implementation of Shor's algorithm surfaces another complication.
# Even if we have a good compiler, every random choice of ``a`` yields a
# different quantum circuit (as we will discuss in the implementation details
# below). Each of these circuits, generated independently at runtime, would need
# to be compiled and optimized, leading to a huge overhead in computation
# time. One could potentially generate, optimize, and store circuits and
# subroutines for reuse. But note that they depend on both ``a`` and ``N``,
# where in a cryptographic context, ``N`` relates to a public key which is
# unique for every entity. Morever, for sizes of cryptographic relevance, ``N``
# will be a 2048-bit integer or larger!
#
# The previous discussion also neglects the fact that the quantum computation
# happens within the context of an algorithm that includes classical code and
# control flow. In Shor's algorithm, this is fairly minimal, but one can imagine
# larger workflows with substantial classical subroutines that themselves must
# be compiled and optimized, perhaps even in tandem with the quantum code. This
# is where Catalyst and quantum just-in-time compilation come into play.
#
#
# JIT compiling classical and quantum code
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In *compiled* languages, like C and C++, compilation is a process that happens
# offline prior to executing your code. An input program is sent to a compiler,
# which outputs a new program in assembly code. An assembler then turns this
# into a machine-executable program which you can run and feed inputs
# to. [#PurpleDragonBook]_.
#
# On the other hand, in *interpreted* languages (like Python), both the source
# program and inputs are fed to the interpreter, which processes them line
# by line, and directly gives us the program output.
#
# Compiled and interpreted languages, and languages within each category, all
# have unique strengths and weakness. Compilation will generally lead to faster
# execution, but can be harder to debug than interpretation, where execution can
# halt partway and provide direct diagnostic information about where something
# went wrong [#PurpleDragonBook]_. *Just-in-time compilation* offers a solution
# that lies, in some sense, at the boundary between the two.
#
# Just-in-time compilation involves compiling code *during* execution, for instance,
# while an interpreter is doing its job.
#

######################################################################
# The quantum part
# ^^^^^^^^^^^^^^^^
#
# In this section we describe the circuits comprising the quantum subroutine
# in Shor's algorithm, i.e., the order-finding algorithm. The presented
# implementation is based on [#Beauregard2003]_. For an integer :math:`N` with
# an :math:`n = \lceil \log_2 N \rceil`-bit representation, the circuit requires
# :math:`2n + 3` qubits, where :math:`n + 1` are for computation and
# :math:`n + 2` are auxiliary.
#
# Order finding is an application of *quantum phase estimation* (QPE) for the
# operator
#
# .. math::
#
#     U_a \vert x \rangle = \vert ax \pmod N \rangle
#
# where :math:`\vert x \rangle` is the binary representation of integer
# :math:`x`, and :math:`a` is the randomly-generated integer discussed
# above. The full QPE circuit for producing a :math:`t`-bit estimate of the
# phase :math:`\theta` is presented below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full.svg
#    :width: 600
#    :align: center
#    :alt: Quantum phase estimation circuit for order finding.
#
#    Initial circuit for order finding with quantum phase estimation (QPE).
#
# This high-level view of the circuit hides its complexity, since the
# implementation details of :math:`U_a` are not shown and auxiliary qubits are
# omitted. In what follows, we'll leverage shortcuts afforded by the hybrid
# nature of computation, and Catalyst. Specifically, with mid-circuit
# measurement and reset we can reduce the number of estimation wires to
# :math:`t=1`. Most of the arithmetic will be performed in the Fourier
# basis. Moreover, since we know :math:`a` in advance, we can vary circuit
# structure on the fly and save resources.
#
# First, we'll use our classical knowledge of :math:`a` to simplify the
# implementation of the controlled :math:`U_a^{2^k}`. Naively, it looks like we
# must apply a controlled :math:`U_a` operation :math:`2^k` times. However, note
#
# .. math::
#
#     U_a^{2^k}\vert x \rangle = \vert (a \cdot a \cdots a) x \pmod N \rangle = \vert a^{2^k}x \pmod N \rangle = U_{a^{2^k}} \vert x \rangle.
#
# But since :math:`a` is known, we can classically evaluate :math:`a^{2^k} \pmod
# N` and implement controlled-:math:`U_{a^{2^k}}` instead.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power.svg
#    :width: 600
#    :align: center
#    :alt: Order finding with controlled operations that take advantage of classical precomputation.
#
#    Leveraging knowledge of :math:`a` allows us to precompute powers for the
#    controlled :math:`U_a`.
#
# There is a tradeoff here, however. Each controlled operation is different, so
# we must compile and optimize more than one circuit. However, additional
# compilation time could be outweighed by the fact that we now run only
# :math:`t` controlled operations, instead of :math:`1 + 2 + 4 + \cdots +
# 2^{t-1} = 2^t - 1`.
#
# Next, let's zoom in on an arbitrary controlled-:math:`U_a`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/c-ua.svg
#    :width: 700
#    :align: center
#    :alt: Quantum phase estimation circuit for order finding.
#
#    Implementing a controlled :math:`U_a` with modular arithmetic
#    [#Beauregard2003]_.
#
# The control qubit, :math:`\vert c\rangle`, is an estimation qubit. The
# register :math:`\vert x \rangle` and the auxiliary register contain :math:`n`
# and :math:`n + 1` qubits respectively, for reasons we elaborate on below.
#
# :math:`M_a` multiplies the contents of one register by :math:`a` and adds it to
# another register, in place and modulo :math:`N`,
#
# .. math::
#
#     M_a \vert x \rangle \vert b \rangle \vert 0 \rangle =  \vert x \rangle \vert (b + ax) \pmod N \rangle \vert 0 \rangle.
#
# Ignoring the control qubit, we can validate this circuit implements
# :math:`U_a`:
#
# .. math::
#
#     \begin{eqnarray}
#       M_a \vert x \rangle \vert 0 \rangle^{\otimes n + 1} \vert 0 \rangle &=&  \vert x \rangle \vert ax \rangle \vert 0 \rangle \\
#      SWAP (\vert x \rangle \vert ax \rangle ) \vert 0 \rangle &=&  \vert ax \rangle \vert x \rangle \vert 0 \rangle \\
#     M_{a^{-1}}^\dagger \vert ax \rangle \vert x \rangle  \vert 0 \rangle &=& \vert ax\rangle \vert x - a^{-1}(ax) \rangle \vert 0 \rangle \\
#      &=& \vert ax \rangle \vert 0 \rangle^{\otimes n + 1} \vert 0 \rangle,
#     \end{eqnarray}
#
# where we've omitted the "mod :math:`N`" for readability, and used the fact
# that the adjoint of addition is subtraction.
#
# A high-level implementation of a controlled :math:`M_a` is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder-with-control.svg
#    :width: 700
#    :align: center
#    :alt: Controlled addition of :math:`ax` using a series of double-controlled Fourier adders.
#
#    Circuit for controlled addition of :math:`ax` using a series of double-controlled Fourier adders.
#    [#Beauregard2003]_.
#
# First, note the controls on the quantum Fourier transforms (QFTs) are not
# needed. If we remove them and :math:`\vert c \rangle = \vert 1 \rangle`, the
# circuit works as expected. If :math:`\vert c \rangle = \vert 0 \rangle`, they
# run, but cancel each other out since none of the interior operations execute
# (this optimization is broadly applicable to controlled operations, and quite
# useful!). This yields the circuit below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder-with-control-not-on-qft.svg
#    :width: 700
#    :align: center
#    :alt: Controlled addition of :math:`ax` using a series of double-controlled Fourier adders, controls on QFT removed.
#
#    The controls on the QFT can be removed without altering the function of the circuit
#    [#Beauregard2003]_.
#
# At first glance it not clear how :math:`a x` is obtained. The qubits in
# register :math:`\vert x \rangle` control operations that depend on :math:`a`
# multiplied by various powers of 2. There is a QFT before and after, whose
# purpose is unclear, and we have yet to define :math:`\Phi_+`.
#
# These special operations perform *addition in the Fourier basis*
# [#Draper2000]_. This is another trick we can leverage given prior knowledge of
# :math:`a`. Rather than performing addition on bits in computational basis
# states, we can apply a QFT, adjust the phases based on the bits of the number
# being added, then inverse QFT to obtain the result. The circuit for *Fourier
# addition*, :math:`\Phi`, is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder.svg
#    :width: 700
#    :align: center
#    :alt: Addition in the Fourier basis.
#
#    Circuit for addition in the Fourier basis [#Draper2000]_.
#
# In the third circuit, we've absorbed the Fourier transforms into the basis
# states, and denoted this by calligraphic letters. The :math:`\mathbf{R}_k` are
# phase shifts based on :math:`a`, and will be described below.
#
# To understand how Fourier addition works, we begin by revisiting the QFT.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-1.svg
#    :width: 800
#    :align: center
#    :alt: The Quantum Fourier Transform.
#
#    Circuit for the quantum Fourier transform. Note the big-endian qubit ordering with
#    no terminal SWAP gates.
#
# In this circuit we define the phase gate
#
# .. math::
#
#     R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{\frac{2\pi i}{2^k}} \end{pmatrix}.
#
# The qubits are ordered using big endian notation. For an :math:`n`-bit integer
# :math:`b`, :math:`\vert b\rangle = \vert b_{n-1} \cdots b_0\rangle` and
# :math:`b = \sum_{k=0}^{n-1} 2^k b_k`.
#
# Suppose we wish to add :math:`a` to :math:`b`. We can add a new register
# prepared in :math:`\vert a \rangle`, and use its qubits to control the
# addition of phases to qubits in :math:`\vert b \rangle` (after a QFT is
# applied) in a very particular way:
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-2.svg
#    :width: 800
#    :align: center
#    :alt: Adding one integer to another with the Quantum Fourier Transform.
#
#    Fourier addition of :math:`a` to :math:`b`. The bit values of :math:`a`
#    determine the amount of phase added to the qubits in :math:`b` [#Draper2000]_.
#
# Each qubit in :math:`\vert b \rangle` picks up a phase that depends on the
# bits in :math:`a`. In particular, the :math:`k`'th bit of :math:`b`
# accumulates information about all the bits in :math:`a` with an equal or lower
# index, :math:`a_0, \ldots, a_{k}`. This adds :math:`a_k` to :math:`b_k`; the
# cumulative effect adds :math:`a` to :math:`b`, up to an inverse QFT!
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-3.svg
#    :width: 800
#    :align: center
#    :alt: Adding one integer to another with the Quantum Fourier Transform.
#
# However, we must be careful. Fourier basis addition is *not* automatically
# modulo :math:`N`. If the sum :math:`b + a` requires :math:`n+ 1` bits, it will
# overflow. To handle that, one extra qubit is added to the :math:`\vert
# b\rangle` register (initialized to :math:`\vert 0 \rangle`). This is the
# source of one of the auxiliary qubits mentioned earlier.
#
# In our case, a second register of qubits is not required. Since we know
# :math:`a` in advance, we can precompute the amount of phase to apply: qubit
# :math:`\vert b_k \rangle` must be rotated by :math:`\sum_{\ell=0}^{k}
# \frac{a_\ell}{2^{\ell+1}}`. We'll express this as a new gate,
#
# .. math::
#
#     \mathbf{R}_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i\sum_{\ell=0}^{k} \frac{a_\ell}{2^{\ell+1}}} \end{pmatrix}.
#
# The final circuit for the Fourier adder is
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-4.svg
#    :width: 500
#    :align: center
#    :alt: Full Fourier adder.
#
# As one may expect, :math:`\Phi^\dagger` performs subtraction. However, we must
# also consider the possibility of underflow.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_adjoint.svg
#    :width: 500
#    :align: center
#    :alt: Subtraction in the Fourier basis.
#
# Returning to :math:`M_a`, we have a doubly-controlled
# :math:`\Phi_+`. :math:`\Phi_+` is similar to :math:`\Phi` but by using an
# auxiliary qubit and some extra operations, it will work modulo :math:`N`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n.svg
#    :width: 800
#    :align: center
#    :alt: Addition in the Fourier basis modulo N.
#
#    Circuit for doubly-controlled Fourier addition modulo :math:`N` [#Beauregard2003]_.
#
# Let's analyze this circuit, keeping in mind :math:`a < N` and :math:`b < N`,
# and the main register has :math:`n + 1` bits.
#
# Suppose both control qubits are :math:`\vert 1 \rangle`. We add :math:`a` to
# :math:`b`, then subtract :math:`N`. If :math:`a + b \geq N`, we get overflow
# on the top qubit (which was included to account for precisely this), but
# subtraction gives us the correct result modulo :math:`N`. After shifting back
# to the computational basis with the :math:`QFT^\dagger`, the topmost qubit is
# in :math:`\vert 0 \rangle` so the CNOT does not trigger. Next, we subtract
# :math:`a` from the register, in state :math:`\vert a + b - N \pmod N \rangle`,
# to obtain :math:`\vert b - N \rangle`. Since :math:`b < N`, there is
# underflow. The top qubit, now in state :math:`\vert 1 \rangle`, does not
# trigger the controlled-on-zero CNOT, and the auxiliary qubit is untouched.
#
# If instead :math:`a + b < N`, we subtracted :math:`N` for no reason, leading
# to underflow. The topmost qubit is :math:`\vert 1 \rangle`, the CNOT will
# trigger, and :math:`N` gets added back. The register then contains
# :math:`\vert b + a \rangle`. Subtracting :math:`a` puts the register in state
# :math:`\vert b \rangle`, which by design will not overflow; the
# controlled-on-zero CNOT triggers, and the auxiliary qubit is returned to
# :math:`\vert 0 \rangle`.
#
# If the control qubits are not both :math:`\vert 1 \rangle`, :math:`N` is
# subtracted and the CNOT triggers (since :math:`b < N`), but :math:`N` is
# always added back. By similar reasoning, the controlled-on-zero CNOT always
# triggers and correctly uncomputes the auxiliary qubit.
#
# Recall that :math:`\Phi_+` is used in :math:`M_a` to add :math:`2^{k}a` to
# :math:`b` (controlled on :math:`x_{k}`) in the Fourier basis.  In total, we
# obtain (modulo :math:`N`)
#
# .. math::
#
#     \begin{equation*}
#     b + x_{0} \cdot 2^0 a + x_{1} \cdot 2^1 a + \cdots x_{n-1} \cdot 2^{n-1} a  = b + a \sum_{k=0}^{n-1} x_{k} 2^k =  b + a x.
#     \end{equation*}
#
# This completes our implementation of the controlled-:math:`U_{a^{2^k}}`. The
# full circuit, on :math:`t + 2n + 2` qubits, is below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_combined_multi_est_wires.svg
#    :width: 500
#    :align: center
#    :alt: Full QPE circuit, all t estimation wires, and decompositions.
#
#    Initial circuit for order finding with QPE, and subroutine decompositions.
#
#
# Taking advantage of classical information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Above, we incorporated information about :math:`a` in the controlled
# :math:`U_a` and the Fourier adder. There are some other places we can use
# classical information to our advantage.
#
# First, consider the controlled :math:`U_{a^{2^0}} = U_a` at the beginning of
# the algorithm. The only basis state this operation applies to is :math:`\vert
# 1 \rangle`, which gets sent to :math:`\vert a \rangle`. This is effectively
# just doing controlled addition of :math:`a - 1` to :math:`1`. Since :math:`a`
# is selected from between :math:`2` and :math:`N-2` inclusive, the addition is
# guaranteed to never overflow. This means we can simply do a controlled Fourier
# addition, and save a significant number of resources!
#
# We can also make some optimizations to the end of the algorithm by keeping
# track of the powers of :math:`a`. If at iteration :math:`k` we have
# :math:`a^{2^k} = 1`, no further multiplication is necessary, and we can
# effectively terminate the algorithm early.
#
# Next, consider the sequence of doubly-controlled adders in the controlled
# :math:`M_a`. Below, we show the initial instance of the controlled
# :math:`M_a`, where the auxiliary register is in state :math:`\vert 0 \rangle`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder-simplified.svg
#    :width: 600
#    :align: center
#    :alt: The controlled multiplier circuit in the context of Shor's algorithm.
#
# The doubly-controlled adders are controlled on both estimation qubits, and the
# bits in the target register. Consider the state of the system after the
# initial controlled-:math:`U_a` (or, rather, controlled addition of :math:`a-1`,
#
# .. math::
#
#     \begin{equation*}
#     \vert + \rangle ^{\otimes (t - 1)} \frac{1}{\sqrt{2}} \left( \vert 0 \rangle \vert 1 \rangle + \vert 1 \rangle \vert a \rangle \right)
#     \end{equation*}
#
# The next controlled operation is controlled-:math:`U_{a^2}`. Since the only
# two basis states present are :math:`\vert 1 \rangle` and :math:`\vert a
# \rangle`, the only doubly-controlled :math:`\Phi_+` that trigger are the first
# one (with the second control on the bottom-most qubit), and those controlled
# on qubits that are :math:`1` in the binary representation of
# :math:`a`. Mathematically, we only need doubly-controlled operations on qubits
# where the logical OR of the bit representations of :math:`1` and `a` are 1! An
# example, for :math:`a = 5`, is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder-simplified-states.svg
#    :width: 800
#    :align: center
#    :alt: Leveraging knowledge of :math:`a` to eliminate unnecessary doubly-controlled additions.
#
#    Leveraging knowledge of :math:`a` to eliminate unnecessary doubly-controlled additions.
#
# Depending on the choice of :math:`a`, this could be major savings, especially
# at the beginning of the algorithm where very few basis states are
# involved. The same trick can be used in the portion of the
# controlled-:math:`U_a` after the controlled SWAPs, as demonstrated below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/c-ua-basis-states-optimization.svg
#    :width: 600
#    :align: center
#    :alt: Removing doubly-controlled additions based on expected terms in the register superposition.
#
#    Doubly-controlled operations can be removed from both controlled multiplications, if
#    one keeps track of the basis states involved.
#
# Eventually we expect diminishing returns, because each controlled
# :math:`U_{a^{2^k}}` contributes more terms to the superposition. Before the
# :math:`k`'th iteration, the control register contains a superposition of
# :math:`\{ \vert a^j\}, j = 0, \ldots, 2^{k - 1}` (inclusive), and after the
# controlled SWAPs, the relevant superposition is :math:`\{ \vert a^j\}, j =
# 2^{k-1}+1, \ldots, 2^{k} - 1`.
#
# A similar simplification can be made within each doubly-controlled
# :math:`\Phi_+`. Recall that the bulk of :math:`\Phi_+` is for detecting and
# correcting for overflow after running :math:`\Phi(a)`. But, if we keep track
# of the powers of :math:`a` modulo :math:`N` in the superposition classically,
# we know in advance when it will be needed. A hypothetical example is shown
# below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n_streamlined.svg
#    :width: 600
#    :align: center
#    :alt: Advanced detection and removal of overflow correction in doubly-controlled operations.
#
#    Advanced detection and removal of overflow correction in doubly-controlled
#    operations can lead to significant resource savings.
#
#  This last optimization will not be included in our implementation. We again
#  expect it to have diminishing returns as more iterations of the algorthm are
#  performed.
#
#
# The "single-qubit" QPE
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The optimizations described in the previous section focus on reducing the
# number of operations, or depth of the circuit. The number of qubits, or width,
# can be reduced using a well-known trick for the QFT. Let's return to the QPE
# circuit and expand the final inverse QFT.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded.
#
# Look carefully at the qubit in :math:`\vert \theta_0\rangle`. After the final
# Hadamard, it is used only for controlled gates. Thus, we can just measure it
# and apply subsequent operations controlled on the classical outcome,
# :math:`\theta_0`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-2.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded and last estimation qubit measured off.
#
# Once again, we can do better by dynamically modifying the circuit based on
# classical information. Instead of applying controlled :math:`R^\dagger_2`, we
# can apply :math:`R^\dagger` where the rotation angle is 0 if :math:`\theta_0 =
# 0`, and :math:`-2\pi i/2^2` if :math:`\theta_0 = 1`, i.e., :math:`R^\dagger_{2 \theta_0}`.
# The same can be done for all other gates controlled on :math:`\theta_0`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-3.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded, last estimation qubit measured, and rotation gates adjusted.
#
# We'll leverage this trick again with the second-last estimation
# qubit. Moreover, we can make a further improvement by noting that once the
# last qubit is measured, we can reset and repurpose it to play the role of the
# second last qubit.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-4.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded and last estimation qubit reused.
#
# Once again, we adjust rotation angles based on measurement values, removing
# the need for classical controls.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-5.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded, last estimation qubit reused, and rotation gates adjusted.
#
# We can do this for all remaining estimation qubits, adding more rotations
# depending on previous measurement outcomes.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-6.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with one estimation qubit, and unmerged rotation gates.
#
# Finally, since these are all :math:`RZ`, we can merge them. Let
#
# .. math::
#
#     \mathbf{M}_{k} = \begin{pmatrix} 1 & 0 \\ 0 & e^{-2\pi i\sum_{\ell=0}^{k}  \frac{\theta_{\ell}}{2^{k + 2 - \ell}}} \end{pmatrix}.
#
# With a bit of index gymnastics, we obtain our final QPE algorithm with a single estimation qubit:
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-7.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with one estimation qubit.
#
#
# The final Shor circuit
# ~~~~~~~~~~~~~~~~~~~~~~
# Replacing the controlled-:math:`U_a` with the subroutines derived above, we
# see Shor's algorithm requires :math:`2n + 3` qubits in total, as summarized in
# the graphic below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_combined.svg
#    :width: 800
#    :align: center
#    :alt: Full implementation of QPE circuit.
#
# From this, we see some additional optimizations are possible. In particular,
# the QFT and inverse QFT are applied before and after each :math:`M_a` block,
# so we can remove the ones that occur between the different controlled
# :math:`U_a` such that the only ones remaining are the very first and last, and
# those before and after the controlled SWAPs.

######################################################################
# Catalyst implementation
# -----------------------
#
# With all our circuits in hand, we can code up the full implementation of
# Shor's algorithm in PennyLane and Catalyst.
#
# First, we require some utility functions for modular arithemetic:
# exponentiation by repeated squaring (to avoid overflow when exponentiating
# large integers), and computation of inverses modulo :math:`N`. Note that the
# first can be accomplished in regular Python using the built-in ``pow``
# method. However, this is not JIT-compatible and there is no equivalent in JAX NumPy.

from jax import numpy as jnp


def repeated_squaring(a, exp, N):
    """QJIT-compatible function to determine (a ** exp) % N.

    Source: https://en.wikipedia.org/wiki/Modular_exponentiation#Left-to-right_binary_method
    """
    bits = jnp.array(jnp.unpackbits(jnp.array([exp]).view("uint8"), bitorder="little"))
    total_bits_one = jnp.sum(bits)

    result = jnp.array(1, dtype=jnp.int32)
    x = jnp.array(a, dtype=jnp.int32)

    idx, num_bits_added = 0, 0

    while num_bits_added < total_bits_one:
        if bits[idx] == 1:
            result = (result * x) % N
            num_bits_added += 1
        x = (x**2) % N
        idx += 1

    return result


def modular_inverse(a, N):
    """QJIT-compatible modular multiplicative inverse routine.

    Source: https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#Modular_integers
    """
    t = jnp.array(0, dtype=jnp.int32)
    newt = jnp.array(1, dtype=jnp.int32)
    r = jnp.array(N, dtype=jnp.int32)
    newr = jnp.array(a, dtype=jnp.int32)

    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr

    if t < 0:
        t = t + N

    return t


######################################################################
# Next, we require a few helper functions for the phase estimation.


def fractional_binary_to_float(sample):
    """Convert an n-bit sample [k1, k2, ..., kn] to a floating point
    value using fractional binary representation,

        k = (k1 / 2) + (k2 / 2 ** 2) + ... + (kn / 2 ** n)
    """
    powers_of_two = 2 ** (jnp.arange(len(sample)) + 1)
    return jnp.sum(sample / powers_of_two)


def as_integer_ratio(f):
    """QJIT compatible conversion of a floating point number to two 64-bit
    integers such that their quotient equals the input to available precision.
    """
    mantissa, exponent = jnp.frexp(f)

    i = 0
    while jnp.logical_and(i < 300, mantissa != jnp.floor(mantissa)):
        mantissa = mantissa * 2.0
        exponent = exponent - 1
        i += 1

    numerator = jnp.asarray(mantissa, dtype=jnp.int64)
    denominator = jnp.asarray(1, dtype=jnp.int64)
    abs_exponent = jnp.abs(exponent)

    if exponent > 0:
        num_to_return, denom_to_return = numerator << abs_exponent, denominator
    else:
        num_to_return, denom_to_return = numerator, denominator << abs_exponent

    return num_to_return, denom_to_return


def phase_to_order(phase, max_denominator):
    """Given some floating-point phase, estimate integers s, r such
    that s / r = phase, where r is no greater than some specified value.

    Uses a JIT-compatible re-implementation of Fraction.limit_denominator.
    """
    numerator, denominator = as_integer_ratio(phase)

    order = 0

    if denominator <= max_denominator:
        order = denominator

    else:
        p0, q0, p1, q1 = 0, 1, 1, 0

        a = numerator // denominator
        q2 = q0 + a * q1

        while q2 < max_denominator:
            p0, q0, p1, q1 = p1, q1, p0 + a * p1, q2
            numerator, denominator = denominator, numerator - a * denominator

            a = numerator // denominator
            q2 = q0 + a * q1

        k = (max_denominator - q0) // q1
        bound1 = p0 + k * p1 / q0 + k * q1
        bound2 = p1 / q1

        loop_res = 0

        if jnp.abs(bound2 - phase) <= jnp.abs(bound1 - phase):
            loop_res = q1
        else:
            loop_res = q0 + k * q1

        order = loop_res

    return order


######################################################################
# Next, we provide the implementations of the circuits derived in the previous section.

import pennylane as qml
import catalyst
from catalyst import measure

catalyst.autograph_strict_conversion = True


def QFT(wires):
    """The standard QFT, redefined because the PennyLane one uses terminal SWAPs."""
    shifts = jnp.array([2 * jnp.pi * 2**-i for i in range(2, len(wires) + 1)])

    for i in range(len(wires)):
        qml.Hadamard(wires[i])

        for j in range(len(shifts) - i):
            qml.ControlledPhaseShift(shifts[j], wires=[wires[(i + 1) + j], wires[i]])


def fourier_adder_phase_shift(a, wires):
    """Sends QFT(|b>) -> QFT(|b + a>)."""
    n = len(wires)
    a_bits = jnp.unpackbits(jnp.array([a]).view("uint8"), bitorder="little")[:n][::-1]
    powers_of_two = jnp.array([1 / (2**k) for k in range(1, n + 1)])
    phases = jnp.array([jnp.dot(a_bits[k:], powers_of_two[: n - k]) for k in range(n)])

    for i in range(len(wires)):
        if phases[i] != 0:
            qml.PhaseShift(2 * jnp.pi * phases[i], wires=wires[i])


def doubly_controlled_adder(N, a, control_wires, wires, aux_wire):
    """Sends |c>|xk>QFT(|b>)|0> -> |c>|xk>QFT(|b + c xk a) mod N>)|0>."""
    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)

    qml.adjoint(fourier_adder_phase_shift)(N, wires)

    qml.adjoint(QFT)(wires)
    qml.CNOT(wires=[wires[0], aux_wire])
    QFT(wires)

    qml.ctrl(fourier_adder_phase_shift, control=aux_wire)(N, wires)

    qml.adjoint(qml.ctrl(fourier_adder_phase_shift, control=control_wires))(a, wires)

    qml.adjoint(QFT)(wires)
    qml.PauliX(wires=wires[0])
    qml.CNOT(wires=[wires[0], aux_wire])
    qml.PauliX(wires=wires[0])
    QFT(wires)

    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)


def controlled_ua(N, a, control_wire, target_wires, aux_wires, mult_a_mask, mult_a_inv_mask):
    """Sends |c>|x>|0> to |c>|ax mod N>|0> if c = 1.

    The mask arguments allow for the removal of unnecessary double-controlled additions.
    """
    n = len(target_wires)

    # Apply double-controlled additions where bits of a can be 1.
    for i in range(n):
        if mult_a_mask[n - i - 1] > 0:
            pow_a = (a * (2**i)) % N
            doubly_controlled_adder(
                N, pow_a, [control_wire, target_wires[n - i - 1]], aux_wires[:-1], aux_wires[-1]
            )

    qml.adjoint(QFT)(wires=aux_wires[:-1])

    # Controlled SWAP the target and aux wires; note that the top-most aux wire
    # is only to catch overflow, so we ignore it here.
    for i in range(n):
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])
        qml.Toffoli(wires=[control_wire, target_wires[i], aux_wires[i + 1]])
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])

    # Adjoint of controlled multiplication with the modular inverse of a
    a_mod_inv = modular_inverse(a, N)

    QFT(wires=aux_wires[:-1])

    for i in range(n):
        if mult_a_inv_mask[i] > 0:
            pow_a_inv = (a_mod_inv * (2 ** (n - i - 1))) % N
            qml.adjoint(doubly_controlled_adder)(
                N,
                pow_a_inv,
                [control_wire, target_wires[i]],
                aux_wires[:-1],
                aux_wires[-1],
            )


######################################################################
# Next, let's put everything together into the order-finding routine
# that is part of Shor's algorithm. We can implement the entire algorithm
# within the ``@qml.qjit`` decorator.


@qml.qjit(autograph=True, static_argnums=(2))
def shors_algorithm(N, a, n_bits):
    """Execute Shor's algorithm: return a guess for the prime factors of N."""
    est_wire = 0
    target_wires = jnp.arange(n_bits) + 1
    aux_wires = jnp.arange(n_bits + 2) + n_bits + 1

    dev = qml.device("lightning.qubit", wires=2 * n_bits + 3, shots=1)

    @qml.qnode(dev)
    def run_qpe(a):
        meas_results = jnp.zeros((n_bits,), dtype=jnp.int32)
        cumulative_phase = jnp.array(0.0)
        phase_divisors = 2.0 ** jnp.arange(n_bits + 1, 1, -1)

        a_mask = jnp.zeros(n_bits, dtype=jnp.int64)
        a_mask = a_mask.at[0].set(1) + jnp.array(
            jnp.unpackbits(jnp.array([a]).view("uint8"), bitorder="little")[:n_bits]
        )
        a_inv_mask = a_mask

        qml.PauliX(wires=target_wires[-1])

        QFT(wires=aux_wires[:-1])

        # First iteration: add a - 1 using the Fourier adder.
        qml.Hadamard(wires=est_wire)

        QFT(wires=target_wires)
        qml.ctrl(fourier_adder_phase_shift, control=est_wire)(a - 1, target_wires)
        qml.adjoint(QFT)(wires=target_wires)

        qml.Hadamard(wires=est_wire)
        meas_results[0] = measure(est_wire, reset=True)
        cumulative_phase = -2 * jnp.pi * jnp.sum(meas_results / jnp.roll(phase_divisors, 1))

        # For subsequent iterations, determine powers of a, and apply controlled
        # U_a when the power is not 1. Unnecessarily double-controlled
        # operations are removed, based on values stored in the two "mask" variables.
        powers_cua = jnp.array([repeated_squaring(a, 2**p, N) for p in range(n_bits)])

        loop_bound = n_bits
        if jnp.min(powers_cua) == 1:
            loop_bound = jnp.argmin(powers_cua)

        for pow_a_idx in range(1, loop_bound):
            pow_cua = powers_cua[pow_a_idx]

            if not jnp.all(a_inv_mask):
                for power in range(2**pow_a_idx, 2 ** (pow_a_idx + 1)):
                    next_pow_a = jnp.array([repeated_squaring(a, power, N)])
                    a_inv_mask = a_inv_mask + jnp.array(
                        jnp.unpackbits(next_pow_a.view("uint8"), bitorder="little")[:n_bits]
                    )

            qml.Hadamard(wires=est_wire)

            controlled_ua(N, pow_cua, est_wire, target_wires, aux_wires, a_mask, a_inv_mask)

            a_mask = a_mask + a_inv_mask
            a_inv_mask = jnp.zeros_like(a_inv_mask)

            qml.PhaseShift(cumulative_phase, wires=est_wire)
            qml.Hadamard(wires=est_wire)
            meas_results[pow_a_idx] = measure(est_wire, reset=True)
            cumulative_phase = (
                -2 * jnp.pi * jnp.sum(meas_results / jnp.roll(phase_divisors, pow_a_idx + 1))
            )

        qml.adjoint(QFT)(wires=aux_wires[:-1])

        return meas_results

    # The "classical part" of Shor's algorithm is JIT-compiled along with the circuit
    p, q = jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)

    sample = run_qpe(a)
    phase = fractional_binary_to_float(sample)
    guess_r = phase_to_order(phase, N)

    if guess_r % 2 == 0:
        guess_square_root = (a ** (guess_r // 2)) % N

        if guess_square_root != 1 and guess_square_root != N - 1:
            p = jnp.gcd(guess_square_root - 1, N).astype(jnp.int32)

            if p != 1:
                q = N // p
            else:
                q = jnp.gcd(guess_square_root + 1, N).astype(jnp.int32)

                if q != 1:
                    p = N // q

    return p, q


######################################################################
# To actually run this, we must choose a value of ``a``. In principle, we could
# incorporate random generation of ``a`` within the function above. However,
# this cannot be qjitted. Passing ``n_bits`` as a static argument is also to
# work around limitations of JIT compilation. These will be explored in the next
# section. For now, let us verify our algorithm can successfully factor.

from jax import random


def factor_with_shor(N, n_shots=100):
    key = random.PRNGKey(123456789)
    key, subkey = random.split(key)

    a_choices = jnp.array(list(range(2, N - 1)))
    a = random.choice(subkey, a_choices)

    while jnp.gcd(a, N) == 1:
        key, subkey = random.split(key)
        a = random.choice(subkey, a_choices)

    # The number of bits of N determines the size of the registers.
    n_bits = int(jnp.ceil(jnp.log2(N)))

    p, q = 0, 0

    # Get the success probabilities
    num_success = 0
    for _ in range(n_shots):
        candidate_p, candidate_q = shors_algorithm(N, a, n_bits)
        if candidate_p * candidate_q == N:
            p, q = candidate_p, candidate_q
            num_success += 1

    return p, q, num_success / n_shots


N = 15
p, q, success_prob = factor_with_shor(N)
print(f"N = {p} x {q}. Success probability is {success_prob}")

######################################################################
# JIT compilation and performance
# -------------------------------
#
# Let us now validate that JIT compilation is happening properly. We will run
# the algorithm for a series of different ``N`` with the same bit width, and
# different values of ``a``. We expect the very first execution, for the very
# first ``N`` and ``a``, to take longer than the rest.

import time
import matplotlib.pyplot as plt

# Some 6-bit numbers
N_values = [33, 39, 51, 55, 57]
n_bits = int(jnp.ceil(jnp.log2(N_values[0])))

num_a = 3

execution_times = []

key = random.PRNGKey(1010101)

for N in N_values:
    a_choices = jnp.array(list(range(2, N - 1)))
    unique_a = []

    while len(unique_a) != num_a:
        key, subkey = random.split(key)
        a = random.choice(subkey, a_choices)
        if jnp.gcd(a, N) == 1:
            continue
        unique_a.append(a)

    for a in unique_a:
        # Initial JIT compilation time
        start = time.time()
        p, q = shors_algorithm(N, a, n_bits)
        end = time.time()
        execution_times.append((N, a, end - start))

        # Get subsequent runtimes
        start = time.time()
        p, q = shors_algorithm(N, a, n_bits)
        end = time.time()
        execution_times.append((N, a, end - start))

labels = [str(ex[:1]) for ex in execution_times]
times = [ex[2] for ex in execution_times]

plt.scatter(range(len(times)), times)
plt.ylabel("Time (s)")

# TODO: discussions about technical details and challenges; autograph and
# control flow, dynamically-sized arrays, etc.
#
# TODO: plots of performance

# TODO: relevant code

######################################################################
# Conclusions
# -----------
#
# TODO!
#
# References
# ----------
#
# .. [#Shor1997]
#
#     Peter W. Shor (1997) *Polynomial-Time Algorithms for Prime Factorization
#     and Discrete Logarithms on a Quantum Computer*. SIAM Journal on Computing,
#     Vol. 26, Iss. 5.
#
# .. [#PurpleDragonBook]
#
#     Alfred V Aho, Monica S Lam, Ravi Sethi, Jeffrey D Ullman (2007)
#     *Compilers Principles, Techniques, And Tools*. Pearson Education, Inc.
#
# .. [#Beauregard2003]
#
#     Stephane Beauregard (2003) *Circuit for Shor's algorithm using 2n+3 qubits.*
#     Quantum Information and Computation, Vol. 3, No. 2 (2003) pp. 175-185.
#
# .. [#Draper2000]
#
#     Thomas G. Draper (2000) *Addition on a Quantum Computer.* arXiv preprint,
#     arXiv:quant-ph/0008033.
#
# About the author
# ----------------
# .. include:: ../_static/authors/olivia_di_matteo.txt
