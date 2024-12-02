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
# Shor's famous factoring algorith [CITE] is one such example. Have a look at the
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

            # Ensure the guessed solution is non-trivial
            if guess_square_root not in [1, N - 1]:
                p = jnp.gcd(N, guess_square_root - 1)
                q = jnp.gcd(N, guess_square_root + 1)

    return p, q

######################################################################
# If you saw this code out-of-context, would you even realize this is a quantum
# algorithm? There are no quantum circuits in sight!
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
# program and inputs are fed to the intepreter, which processes them line
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
# In this section we'll describe the circuits used for the quantum part of
# Shor's algorithm, i.e., the order-finding routine. The presented
# implementation is based on that of [#Beauregard2003]_. For an integer
# :math:`N` whose binary representation requires :math:`n = \lceil \log_2 N
# \rceil` bits, our circuit requires :math:`2n + 3` qubits. Of these qubits,
# :math:`n + 1` are used for computation, while the remaining :math:`n + 2` are
# auxiliary.
#
# Order finding is an application of *quantum phase estimation*. The operator
# whose phase is being estimated is :math:`U_a`,
#
# .. math::
#
#     U_a \vert x \rangle = \vert ax \pmod N \rangle
#
# where :math:`\vert x \rangle` is the basis state corresponding to the binary
# representation of the integer :math:`x`, and :math:`a` is the
# randomly-generated integer discussed above. The full circuit for QPE is shown
# below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full.svg
#    :scale: 120%
#    :align: center
#    :alt: Quantum phase estimation circuit for order finding.
#
# This high-level view of the circuit hides its complexity, though. First and
# foremost, the actual implementation details of :math:`U_a` are not yet
# defined. In fact, a significant number of auxiliary qubits are required. There
# is also the issue of precision, which is governed by the number of estimation
# wires, :math:`t`. More estimation wires means finding a solution is more
# likely. However, increasing :math:`t` adds overhead in circuit depth, and adds
# classical simulation due to the increased size of Hilbert space.
#
# In what follows, we will take some shortcuts afforded by the hybrid nature of
# the computation, and from Catalyst. Specifically, with mid-circuit measurement
# and reset we can reduce the number of estimation wires to :math:`t=1`. A great
# deal of the arithmetic operations will be performed in the Fourier basis; and
# since we know :math:`a` in advance, we can vary circuit structure on the fly
# and save resources.  Finally, additional mid-circuit measurements can be used
# in lieu of uncomputation.
#
# First, let us consider the controlled :math:`U_a^{2^k}`. This is a prime
# example of how our classical knowledge can simplify the computation. Naively,
# we may this we must implement a controlled :math:`U_a` operation :math:`2^k`
# times. However, note 
#
# .. math::
#
#     U_a^{2^k}\vert x \rangle = \vert (a \cdot a \cdots a) x \pmod N \rangle = \vert a^{2^k}x \pmod N \rangle = U_{a^{2^k}} \vert x \rangle
#
# Since :math:`a` is known in advance, we can classically evaluate its powers,
# and implement the circuit below instead:
#
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power.svg
#    :scale: 120%
#    :align: center
#    :alt: Order finding with controlled operations that take advantage of classical precomputation.
#
# There is a tradeoff here: if we were performing circuit optimization, we would
# have to separately optimize the circuit for each :math:`a^{2^k}`. However,
# the additional compilation time is outweighed by the fact that we now have
# only :math:`t` controlled operations to implement, rather than :math:`1 + 2 +
# 4 + ... 2^{t -1}`. Furthermore, we will be able to jit-compile the circuit
# construction.
#
# Next, let's zoom in on the controlled :math:`U_a`. 
# 
# .. figure:: ../_static/demonstration_assets/shor_catalyst/c-ua.svg
#    :scale: 120%
#    :align: center
#    :alt: Quantum phase estimation circuit for order finding.
#
# The control qubit, :math:`\vert c\rangle`, is an estimation qubit. The
# register :math:`\vert x \rangle` and the auxiliary register contain :math:`n +
# 1` and :math:`n + 2` qubits respectively, for reasons we elaborate on
# below.
#
# The next operation of interest, :math:`M_a`, adds the (multiplied) contents of one register to another, in place, and modulo :math:`N`,
# 
# .. math::
#
#     M_a \vert x \rangle \vert b \rangle \vert 0 \rangle =  \vert x \rangle \vert (b + ax) \pmod N \rangle \vert 0 \rangle
#
#
# Ignoring the control qubit, we can validate that the circuit above implements
# :math:`U_a`: (for readability, we omit the "mod :math:`N`", which is implicit on all arithmetic):
#
# .. math::
#
#     \begin{eqnarray}
#       M_a \vert x \rangle \vert 0 \rangle^{\otimes n + 1} \vert 0 \rangle &=&  \vert x \rangle \vert ax \rangle \vert 0 \rangle \\
#      SWAP (\vert x \rangle \vert ax \rangle ) \vert 0 \rangle &=&  \vert ax \rangle \vert x \rangle \vert 0 \rangle \\
#     M_{a^{-1}}^\dagger \vert ax \rangle \vert x \rangle  \vert 0 \rangle &=& \vert ax\rangle \vert x - a^{-1}(ax) \rangle \vert 0 \\
#      &=& \vert ax \rangle \vert 0 \rangle^{\otimes n + 1} \vert 0 \rangle
#     \end{eqnarray}
#
# At a high level, the implementation of :math:`M_a` looks like this:
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder.svg
#    :scale: 120%
#    :align: center
#    :alt: In-place addition modulo N with the Fourier adder.
#
# At first glance, is it not clear how exactly :math:`a x` is getting
# created. The qubits of :math:`\vert x \rangle` are being used as control
# qubits for some operation that depends on :math:`a` multiplied by various
# powers of 2. There is also a QFT before and after those operations, whose
# purpose is unclear.
#
# These special operations are actually performing *addition in the Fourier
# basis*. This is another trick that we can leverage because we know :math:`a`
# in advance. Rather than performing explicit addition on the bits in the
# computational basis states registers, we can apply a Fourier transform, adjust
# the phases based on the bit values of the number we wish to add, then inverse
# Fourier transform to obtain the result. We present the circuit for the
# *Fourier adder*, :math:`\Phi`, below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder.svg
#    :scale: 120% 
#    :align: center
#    :alt: Addition in the Fourier basis.
#
# The gates XX are phase shifts; the explicit value depends on. TODO: finish.
#
# An important point about Fourier basis addition is that it is *not* modulo
# :math:`N`. While both :math:`a` and :math:`b` are less than :math:`N`, their
# sum may be greater than :math:`N`. To avoid overflow, we use a register of
# :math:`n + 1` qubits instead (this is the source of some of the auxiliary
# qubits, referred to earlier).
#
# As one may expect, :math:`\Phi^\dagger` performs subtraction. However, one
# must now take into consideration the possibility of underflow.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_adjoint.svg 
#    :scale: 120%
#    :align: center
#    :alt: Subtraction in the Fourier basis.
#
# Returning to our implementation of :math:`M_a`, we see an operation
# :math:`\Phi_+` which is similar to :math:`\Phi`, but it (a) uses an auxiliary
# qubit, and (b) works modulo :math:`N`. The idea behind :math:`\Phi_+` is that
# we still use Fourier basis addition and subtraction, but apply corrections
# if overflow is detected.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n.svg
#    :scale: 100%
#    :align: center
#    :alt: Addition in the Fourier basis modulo N.
#
# In the circuit above, we first add :math:`a` to :math:`b`, then subtract
# :math:`N` in case we had :math:`a + b > N`. However, if that wasn't the case,
# we subtracted :math:`N` for no reason, causing underflow. This would manifest
# as a 1 in the top-most qubit (the auxiliary qubit added to account for
# overflow). That 1 can be detected by applying a CNOT down to the auxiliary
# qubit, which performs controlled addition to add back :math:`N` if
# needed. Note we must exit the Fourier basis to detect it the underflow. The
# remainder of the circuit returns the auxiliary qubit to its original state.
#
# Uncomputing the auxiliary qubit here is basically as much work as performing
# the operation itself. Here is where we can leverage Catalyst to perform a
# major optimization: rather than uncomputing, we can simply measure the
# auxiliary qubit, add back :math:`N` based on the outcome, then reset it to
# :math:`\vert 0 \rangle`!
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n_mcm.svg
#    :scale: 120%
#    :align: center
#    :alt: Addition in the Fourier basis modulo N.
#
# This optimization cuts down the number of gates in the :math:`M_a` circuit by
# essentially half, which is a major savings.
#
#
# TODO: explain how iterative phase estimation is being used here with Catalyst
# and mid-circuit measurements.

# TODO: insert code here


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
# .. [#PurpleDragonBook]
#
#     Alfred V Aho, Monica S Lam, Ravi Sethi, Jeffrey D Ullman. (2007)
#     *Compilers Principles, Techniques, And Tools*. Pearson Education, Inc.
#
# .. [#Beauregard2003]
#
#     Stephane Beauregard. (2003) *Circuit for Shor's algorithm using 2n+3 qubits.*
#     Quantum Information and Computation, Vol. 3, No. 2 (2003) pp. 175-185.
#
# About the author
# ----------------
# .. include:: ../_static/authors/olivia_di_matteo.txt
