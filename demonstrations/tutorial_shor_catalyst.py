r""".. role:: html(raw)
   :format: html

JIT compilation of Shor's algorithm with PennyLane and Catalyst
===============================================================

.. meta::
    :property="og:description": JIT compile Shor's algorithm from end-to-end.

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
# In this section we describe the circuits that make up the quantum subroutine
# in Shor's algorithm, i.e., the order-finding routine. The presented
# implementation is based on [#Beauregard2003]_. For an integer :math:`N` with
# an :math:`n = \lceil \log_2 N \rceil`-bit representation, the circuit requires
# :math:`2n + 3` qubits. Of these, :math:`n + 1` are for computation and
# :math:`n + 2` are auxiliary.
#
# Order finding is an application of *quantum phase estimation*. We wish to 
# estimate the phase, :math:`\theta`, of the operator :math:`U_a`,
#
# .. math::
#
#     U_a \vert x \rangle = \vert ax \pmod N \rangle
#
# where the :math:`\vert x \rangle` is the binary representation of integer
# :math:`x`, and :math:`a` is the randomly-generated integer discussed
# above. The full QPE circuit, using :math:`t` estimation wires, is presented
# below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full.svg
#    :width: 600
#    :align: center
#    :alt: Quantum phase estimation circuit for order finding.
#
# This high-level view of the circuit hides its complexity, given that the
# implementation details of :math:`U_a` are not shown and auxiliary qubits are
# omitted. In what follows, we'll leverage shortcuts afforded by the hybrid
# nature of computation, and from Catalyst. Specifically, with mid-circuit
# measurement and reset we can reduce the number of estimation wires to
# :math:`t=1`. Most of the required arithmetic will be performed in the Fourier
# basis. Since we know :math:`a` in advance, we can vary circuit structure on
# the fly and save resources. Finally, additional mid-circuit measurements can
# be used in lieu of uncomputation.
#
# First, we'll use our knowledge of classical parameters to simplify the
# implementation of the controlled :math:`U_a^{2^k}`. Naively, it looks like we
# must apply a controlled :math:`U_a` operation :math:`2^k` times. However, note
#
# .. math::
#
#     U_a^{2^k}\vert x \rangle = \vert (a \cdot a \cdots a) x \pmod N \rangle = \vert a^{2^k}x \pmod N \rangle = U_{a^{2^k}} \vert x \rangle
#
# Since :math:`a` is known in advance, we can classically evaluate
# :math:`a^{2^k}` and implement controlled-:math:`U_{a^{2^k}}` instead.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power.svg
#    :width: 600
#    :align: center
#    :alt: Order finding with controlled operations that take advantage of classical precomputation.
#
# There is a tradeoff here: since each controlled operation is now different, we
# will have to optimize each circuit separately during compilation. However,
# additional compilation time could be outweighed by the fact that we must now
# run only :math:`t` controlled operations, instead of :math:`1 + 2 + 4 + \cdots
# + 2^{t-1} = 2^t - 1`. Later we'll also jit-compile the circuit construction.
#
# Next, let's zoom in on an arbitrary controlled-:math:`U_a`. 
# 
# .. figure:: ../_static/demonstration_assets/shor_catalyst/c-ua.svg
#    :width: 700
#    :align: center
#    :alt: Quantum phase estimation circuit for order finding.
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
# Ignoring the control qubit, let's validate that this circuit implements
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
# A high-level implementation of a controled :math:`M_a` is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder-with-control.svg
#    :width: 700 
#    :align: center
#    :alt: Doubly-controlled adder.
#
# First, note that the controls on the QFTs are not needed. If we remove them
# and :math:`\vert c \rangle = \vert 1 \rangle`, the circuit works as
# expected. If :math:`\vert c \rangle = \vert 0 \rangle`, they would run but
# cancel each other out, since none of the interior operations execute (note
# that this optimization is broadly applicable, and quite useful!). This yields
# the circuit below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder-with-control-not-on-qft.svg
#    :width: 700 
#    :align: center
#    :alt: Doubly-controlled adder.
#
# At first glance, it may not be clear how :math:`a x` is created. The qubits in
# register :math:`\vert x \rangle` are controlling operations that depend on
# :math:`a`, multiplied by various powers of 2. There is also a QFT before and
# after, whose purpose is unclear.
#
# These special operations perform *addition in the Fourier basis*
# [#Draper2000]_. This is another trick we can leverage, given prior knowledge
# of :math:`a`. Rather than performing explicit addition on bits in
# computational basis states, we can apply a Fourier transform, adjust the
# phases based on the bits in the number we wish to add, then inverse Fourier
# transform to obtain the result. We present the circuit for the *Fourier
# adder*, :math:`\Phi`, below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder.svg
#    :width: 800 
#    :align: center
#    :alt: Addition in the Fourier basis.
#
# In the third circuit, we've absorbed the Fourier transforms into the basis
# states, and denoted this using a calligraphic letter.  The
# :math:`\mathbf{R}_k` are phase shifts, to be described below. To see how this
# works, let's first take a closer look at the QFT. The qubit ordering in the
# circuit is such that for an :math:`n`-bit integer :math:`b`, :math:`\vert
# b\rangle = \vert b_{n-1} \cdots b_0\rangle` and :math:`b = \sum_{k=0}^{n-1}
# 2^k b_k`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-1.svg 
#    :width: 800 
#    :align: center
#    :alt: The Quantum Fourier Transform.
#
# where
#
# .. math::
#
#     R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{\frac{2\pi i}{2^k}} \end{pmatrix}.
#
# Let's add a new register, prepared in the basis state :math:`\vert a \rangle`.
# In the next circuit, we control on qubits in :math:`\vert a \rangle` to modify
# the phases in :math:`\vert b \rangle` (after a QFT is applied) in a very
# particular way:
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-2.svg 
#    :width: 800 
#    :align: center
#    :alt: Adding one integer to another with the Quantum Fourier Transform.
#
# We observe each qubit in :math:`\vert b \rangle` picks up a phase that depends
# on the bits in :math:`a`. In particular, bit :math:`b_k` accumulates
# information about all the bits in :math:`a` with an equal or lower index,
# :math:`a_0, \ldots, a_{k}`. The effect is that of adding :math:`a_k` to
# :math:`b_k`; looking across the entire register, we are adding :math:`a` to
# :math:`b`, up to an inverse Fourier transform!
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-3.svg 
#    :width: 800 
#    :align: center
#    :alt: Adding one integer to another with the Quantum Fourier Transform.
#
# However, we must be careful. Fourier basis addition is *not* automatically
# modulo :math:`N`. If the sum :math:`b + a` requires :math:`n+ 1` bits, it will
# overflow. To handle that, one extra qubit is added to the top of the
# :math:`\vert b\rangle` register (initialized to :math:`\vert 0 \rangle`). This
# is the source of one of the auxiliary qubits mentioned earlier.
#
# In our case, we don't actually need a second register of qubits. Since we know :math:`a`
# in advance, we can precompute the amount of phase to apply: on qubit
# :math:`\vert b_k \rangle`, we must rotate by :math:`\sum_{\ell=0}^{k}
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
# Returning to :math:`M_a`, we have :math:`\Phi_+` which is similar to
# :math:`\Phi`, but it (a) uses an auxiliary qubit, and (b) works modulo
# :math:`N`. :math:`\Phi_+` still uses Fourier basis addition and subtraction,
# but also applies corrections if overflow is detected. Let's consider a single
# instance of a controlled :math:`\Phi_+(a)`,
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n.svg
#    :width: 800 
#    :align: center
#    :alt: Addition in the Fourier basis modulo N.
#
# Let's step through the various cases of this circuit, keeping in mind 
# :math:`a < N` and :math:`b < N`, and the register has :math:`n + 1` bits.
# 
# Suppose the control qubits are both :math:`\vert 1 \rangle`. We add
# :math:`a` to :math:`b`, then subtract :math:`N`. If :math:`a + b \geq N`, we
# get overflow on the top qubit (which was included to account for precisely
# this), but subtraction gives us the correct result modulo :math:`N`. After
# shifting back to the computational basis, the topmost qubit is in :math:`\vert
# 0 \rangle` so the CNOT does not trigger. Next, we subtract :math:`a` from the
# register in, now in state :math:`\vert a + b - N \pmod N \rangle`. We obtain :math:`\vert b -
# N \rangle`, which, since `b < N`, leads to underflow. The top bit in
# state :math:`\vert 1 \rangle` does not trigger the controlled-on-zero CNOT,
# and our auxiliary qubit is untouched.
#
# If we didn't have overflow before, we would have subtracted :math:`N` for no
# reason, leading to underflow. The topmost qubit would be :math:`\vert 1
# \rangle`, the CNOT triggers, and :math:`N` would be added back. The register
# then contains :math:`\vert b + a \rangle`. Subtracting :math:`a` puts the register in
# state :math:`\vert b \rangle`, which by design will not overflow; the
# controlled-on-zero CNOT triggers, and the auxiliary qubit is returned to
# :math:`\vert 0 \rangle`.
#
# If the control qubits were not both :math:`\vert 1 \rangle`, :math:`N` is
# always subtracted and the CNOT triggers (since :math:`b < N`), but :math:`N`
# is added back. For the same reasoning, though, the controlled-on-zero CNOT
# always triggers and correctly uncomputes the auxiliary qubit.
#
# Recall that :math:`\Phi_+` is used as part of :math:`M_a` to add
# :math:`2^{k}a` modulo :math:`N` to :math:`b` (conditioned on the value of
# :math:`x_{k}`) in the Fourier basis. Re-expressing this as a sum (all
# modulo :math:`N`), we find
#
# .. math::
#
#     \begin{equation*}
#     b + x_{0} \cdot 2^0 a + x_{1} \cdot 2^1 a + \cdots x_{n-1} \cdot 2^{n-1} a  = b + a \sum_{k=0}^{n-1} x_{k} 2^k =  b + a x.
#     \end{equation*}
#
# This completes our implementation of the controlled-:math:`U_{a^{2^k}}`. The
# full circuit, on all :math:`t + 2n + 2` qubits,  is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_combined_multi_est_wires.svg
#    :width: 500
#    :align: center
#    :alt: Full QPE circuit, all t estimation wires, and decompositions.
#
# Taking advantage of classical information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Above, we incorporated information about :math:`a` in the implementation
# of the Fourier adder. There are a few other places we can use classical information
# to our advantage.
#
# First, consider the initial controlled :math:`U_{a^{2^0}} = U_a`. The only
# basis state this operation applies to is :math:`\vert 1 \rangle`, which gets
# sent to :math:`\vert a \rangle` when the operation triggers. This is
# effectively just doing controlled addition of :math:`a - 1` to
# :math:`1`. Since :math:`a` is selected from between :math:`2` and :math:`N-2`
# inclusive, the addition is guaranteed to never overflow. This means we can
# simply do a controlled Fourier addition, and save a significant number of
# resources!  TODO: count them.
#
# Next, let's look to the doubly-controlled adders in the implementation of the
# controlled :math:`M_a`. Below, we show a simplified instance of the controlled
# :math:`M_a` that reflects the state of the auxiliary registers.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder-simplified.svg
#    :width: 600
#    :align: center
#    :alt: The controlled multiplier circuit in the context of Shor's algorithm.
#
# The doubly-controlled adders are controlled on both estimation qubits, and the
# bits in the target register. Now, consider the state of the system after the
# initial controlled-:math:`U_a`:
#
# .. math::
#
#     \begin{equation*}
#     \vert + \rangle ^{\otimes (t - 1)} \frac{1}{\sqrt{2}} \left( \vert 0 \rangle \vert 1 \rangle + \vert 1 \rangle \vert a \rangle \right)
#     \end{equation*}
#
# In the next controlled operation, controlled-:math:`U_{a^2}`, the
# doubly-controlled :math:`\Phi_+` will only trigger when the bits in the target
# register are :math:`1`. Since the only two basis states present are
# :math:`\vert 1 \rangle` and :math:`\vert a \rangle`, the only operations that
# triger are those with the second control on the bottom-most qubit, and any
# qubits corresponding to locations of :math:`1`s in the binary representation
# of :math:`a`. More formally, we only need doubly-controlled operations on
# qubits where the logical OR of the bit representations of :math:`1` and `a`
# are 1! An example, for :math:`a = 5`, is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder-simplified-states.svg
#    :width: 800
#    :align: center
#    :alt: The controlled multiplier circuit in the context of Shor's algorithm.
# 
# Depending on the choice of :math:`a`, this could be major savings, especially
# at the beginning of the algorithm, since not many basis states are involved
# yet. The same trick can be used in the portion of the controlled-:math:`U_a`
# after the controlled SWAPs, as demonstrated below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/c-ua-basis-states-optimization.svg
#    :width: 600
#    :align: center
#    :alt:
# 
# Eventually, we will see diminishing returns, because each
# controlled-:math:`U_{a^{2^k}}` leads to more terms in the superposition. At
# the :math:`k`'th iteration, the control register contains a superposition of
# :math:`\{ \vert a^j\}, j = 0, \ldots, 2^{k - 1}` (inclusive), and after the
# controlled SWAPs, the relevant superposition is :math:`\{ \vert a^j\}, j =
# 2^{k-1}+1, \ldots, 2^{k} - 1`.
#
# A similar simplification can be made within each doubly-controlled
# :math:`\Phi_+` operation. Recall that the bulk of the :math:`\Phi_+` is
# necessary for detecting and correcting for overflow after running
# :math:`\Phi(a)`. But, if we keep track of the powers of :math:`a` modulo
# :math:`N` in the superposition classically, we know in advance when it will be
# needed. A hypothetical example is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n_streamlined.svg
#    :width: 600
#    :align: center
#    :alt:
#
# The "single-qubit" QPE
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Let's return to the QPE routine and expand the final inverse QFT.
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
# the QFT and inverse QFT are being applied before and after each :math:`M_a`
# block, so we can remove the ones that occur between the different
# controlled-:math:`U_a` such that the only ones remaining are the very first
# and last, and those before and after the controlled SWAPs.

######################################################################
# Catalyst implementation
# -----------------------
#
# With all our circuits in hand, we can code up the full implementation of
# Shor's algorithm. Below, we have the set of subroutines defined in the
# previous section.
#

from jax import numpy as jnp

import pennylane as qml
import catalyst
from catalyst import measure
catalyst.autograph_strict_conversion = True

from utils import modular_inverse


def QFT(wires):
    """The standard QFT, redefined because the PennyLane one uses terminal SWAPs."""
    shifts = jnp.array([2 * jnp.pi * 2**-i for i in range(2, len(wires) + 1)])

    for i in range(len(wires)):
        qml.Hadamard(wires[i])

        for j in range(len(shifts) - i):
            qml.ControlledPhaseShift(shifts[j], wires=[wires[(i + 1) + j], wires[i]])


def fourier_adder_phase_shift(a, wires):
    """Sends QFT(|b>) -> QFT(|b + a>). Acts on n + 1 qubits, n = ceil(log2(b)) + 1."""
    n = len(wires)
    a_bits = jnp.unpackbits(jnp.array([a]).view("uint8"), bitorder="little")[:n][::-1]
    powers_of_two = jnp.array([1 / (2**k) for k in range(1, n + 1)])
    # Computing the phases requires a bit of index gymnastics
    phases = jnp.array([jnp.dot(a_bits[k:], powers_of_two[: n - k]) for k in range(n)])

    for i in range(len(wires)):
        qml.PhaseShift(2 * jnp.pi * phases[i], wires=wires[i])


def doubly_controlled_adder(N, a, control_wires, wires, aux_wire):
    """Sends |c>|xk>QFT(|b>)|0> -> |c>|xk>QFT(|b + c xk a) mod N>)|0>."""

    # Add a + b, then subtract N to account for potential overflow
    qml.ctrl(fourier_adder_phase_shift, control=control_wires)(a, wires)
    qml.adjoint(fourier_adder_phase_shift)(N, wires)

    # Check overflow conditions using CNOT, then measurement. Re-add N if needed.
    qml.adjoint(QFT)(wires)
    qml.CNOT(wires=[wires[0], aux_wire])
    QFT(wires)

    was_greater_than_N = measure(aux_wire, reset=True)

    if was_greater_than_N:
        fourier_adder_phase_shift(N, wires)


def doubly_controlled_subtraction(N, a, control_wires, wires, aux_wire):
    """Sends |c>|xk>QFT(|b>)|0> -> |c>|xk>QFT(|b - c xk a) mod N>)|0>.
    We need this as a dedicated routine because we cannot take the adjoint of
    doubly_controlled_adder due to the mid-circuit measurement."""
    
    # Subtract a, check the overflow wire, and add N back if needed.
    qml.ctrl(qml.adjoint(fourier_adder_phase_shift), control=control_wires)(a, wires)

    qml.adjoint(QFT)(wires)
    qml.CNOT(wires=[wires[0], aux_wire])
    QFT(wires)

    was_less_than_zero = measure(aux_wire, reset=True)

    if was_less_than_zero:
        fourier_adder_phase_shift(N, wires)


def controlled_ua(N, a, control_wire, target_wires, aux_wires):
    """Sends |c>|x>|0> to |c>|ax mod N>|0> if |c> = |1>."""
    n = len(target_wires)

    # Controlled multiplication by a mod N; |c>|x>|b>|0> to |c>|x>|(b + ax) mod N>|0>
    for i in range(n):
        power_of_a = (a * (2**i)) % N
        doubly_controlled_adder(
            N, power_of_a, [control_wire, target_wires[n - i - 1]], aux_wires[:-1], aux_wires[-1]
        )

    qml.adjoint(QFT)(wires=aux_wires[:-1])

    # C-SWAP with two CNOTs and a Toffoli. Note that the target and aux
    # registers have n and n + 1 qubits respectively.
    for i in range(n):
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])
        qml.Toffoli(wires=[control_wire, target_wires[i], aux_wires[i + 1]])
        qml.CNOT(wires=[aux_wires[i + 1], target_wires[i]])

    # Adjoint of controlled multiplication with the modular inverse of a
    a_mod_inv = modular_inverse(a, N)

    QFT(wires=aux_wires[:-1])

    for i in range(n):
        power_of_a_inv = (a_mod_inv * (2 ** (n - i - 1))) % N
        doubly_controlled_subtraction(
            N, power_of_a_inv, [control_wire, target_wires[i]], aux_wires[:-1], aux_wires[-1]
        )

    
######################################################################
# Next, let's put everything together into the order-finding routine
# that is part of Shor's algorithm. We can implement the entire algorithm
# within the ``@qml.qjit`` decorator.

@qml.qjit(autograph=True, static_argnums=(2, 3))
def shors_algorithm(N, a, n_bits, max_shots=100):
    """Execute Shor's algorithm and return the factors of N, if found."""
    estimation_wire = 0
    target_wires = jnp.arange(n_bits) + 1
    aux_wires = jnp.arange(n_bits + 2) + n_bits + 1

    dev = qml.device("lightning.qubit", wires=2 * n_bits + 3, shots=1)

    # Order-finding routine - the "quantum part"
    @qml.qnode(dev)
    def run_qpe(a):
        meas_results = jnp.zeros((n_bits,), dtype=jnp.int32)
        cumulative_phase = jnp.array(0.0)
        phase_divisors = 2.0 ** jnp.arange(n_bits + 1, 1, -1)
        
        qml.PauliX(wires=target_wires[-1])

        QFT(wires=aux_wires[:-1])
        
        for i in range(n_bits):
            power_of_a = (a ** (2 ** (n_bits - 1 - i))) % N

            qml.Hadamard(wires=estimation_wire)
            controlled_ua(N, power_of_a, estimation_wire, target_wires, aux_wires)

            # Measure, then compute corrective phase for next round
            qml.PhaseShift(cumulative_phase, wires=estimation_wire)
            meas_results[i] = measure(estimation_wire, reset=True)
            cumulative_phase = -2 * jnp.pi * jnp.sum(meas_results / jnp.roll(phase_divisors, i + 1))

        qml.adjoint(QFT)(wires=aux_wires[:-1])            

        return meas_results

    # The classical part
    shot_idx = 0
    p, q = jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)

    while p * q != N and shot_idx < max_shots:
        sample = run_qpe(a)
        phase = fractional_binary_to_float(sample)
        guess_r = phase_to_order(phase, N)

        # If the guess order is even, we may have a non-trivial square root.
        # If so, try to compute p and q.
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
        shot_idx += 1

    return p, q, shot_idx


######################################################################
# To actually run this, we need to choose a value of ``a``. In principle, we
# could incorporate random generation of ``a`` within the function
# above. However, this cannot be qjitted. Note too that we passed in ``n_bits``
# as a static argument.


######################################################################
# JIT compilation and performance
# -------------------------------
# 
# TODO: show how everything gets put together and JITted
#
# TODO: discussions about technical details and challenges; autograph and
# control flow, dynamically-sized arrays, etc.
# 
# TODO: plots of performance 

# TODO: relevant code

######################################################################
# Conclusions
# -----------
# 
# TODO
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
