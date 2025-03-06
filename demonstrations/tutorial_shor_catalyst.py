r""".. role:: html(raw)
   :format: html

JIT compiling Shor's algorithm with PennyLane and Catalyst
===============================================================

.. meta::
    :property="og:description": JIT compile Shor's algorithm from end-to-end with PennyLane and Catalyst.

    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets//fano.png


*Author: Olivia Di Matteo — Posted: X Y 2025. Last updated: X Y 2025.*

.. related::

    tutorial_jax_transformations
    tutorial_how_to_quantum_just_in_time_compile_vqe_catalyst
    tutorial_qjit_compile_grovers_algorithm_with_catalyst
"""

##############################################################################
# The past few years stimulated a lot of discussion about *hybrid
# quantum-classical algorithms*. For a time, this terminology was synonymous
# with *variational algorithms*. However, integration with classical
# co-processors is necessary for every quantum algorithm, even ones considered
# quintessentially quantum.
#
# Shor's famous factoring algorithm [#Shor1997]_ is one such example. Suppose we
# are given an integer :math:`N`, and promised it is the product of two
# primes, :math:`p` and :math:`q`. Shor's algorithm uses a quantum computer
# to solve this problem with exponentially-better scaling than the best-known
# classical algorithm. Have a look at the example implementation below:

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
# algorithms with hundreds or thousands of qubits is unsustainable. Moreover, a
# programmer doesn't really need to know anything quantum is happening, if the software
# library can generate and compile appropriate quantum code (though, they should
# probably have at least some awareness, since the output of ``guess_order`` is
# probabilistic!).  This raises the important question of what exactly gets
# compiled, as well as where, when, and how that compilation happens.
#
# In PennyLane, classical and quantum code can be compiled together using the
# `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/latest/index.html>`_
# library. This demo leverages their integration to implement a version of
# Shor's factoring algorithm that is just-in-time compiled from end-to-end,
# classical control structure and all. Even better, that compilation happens
# only once per distinct bit-width of the integers being factored.
#
# Compilation
# -----------
#
# Classical compilation
# ^^^^^^^^^^^^^^^^^^^^^
#
# Compilation is the process of translating operations expressed in a high-level
# language to a low-level language.  In languages like C and C++, compilation
# happens offline prior to code execution. A compiler takes a program as input
# and sends it through a sequence of *passes* that perform tasks such as syntax
# analysis, code generation, and optimization. The compiler outputs a new
# program in assembly code, which is passed on to an assembler. The assembler
# translates this code into a machine-executable program that we can feed inputs
# to, then run [#PurpleDragonBook]_.
#
# Compilation is not the only way to execute a program. Python, for example, is
# an *interpreted* language. Both a source program and inputs are fed to the
# interpreter, which processes them line by line and directly yields the program
# output.
#
# Compilation and interpretation each have strengths and weaknesses. Compilation
# generally leads to faster execution, because optimizations can consider
# the overall structure of a program. However, the executable code is not
# human-readable and thus harder to debug. Interpretation is slower, but
# debugging is often easier because execution can be paused to inspect
# the program state or view diagnostic information [#PurpleDragonBook]_.
#
# In between these extremes lies *just-in-time compilation*.  Just-in-time, or
# JIT compilation, happens *during* execution. If a programmer specifies a
# function should be JIT compiled, the first time the interpreter sees it, it
# will spend a little more time to construct and cache a compiled version of
# that function. The next time that function is executed, the compiled version
# can be reused, provided the structure of the inputs hasn't changed.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/compilation_comparison.png
#    :scale: 75%
#    :align: center
#    :alt: From compilation to interpretation.
#
# Quantum compilation
# ^^^^^^^^^^^^^^^^^^^
# Quantum compilation, like its classical counterpart, lowers an algorithm from
# high-level instructions down to low-level ones. The bulk of
# this process involves converting a circuit from a generic, abstract gate
# set to a sequence of gates that satisfy the constraints of a particular
# hardware device. Quantum compilation also involves multiple passes through
# one or more intermediate representations, and both
# machine-independent and dependent optimizations, as depicted below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/compilation-stack.svg
#    :scale: 75%
#    :align: center
#    :alt: The quantum compilation stack.
#
#    High-level overview of the quantum compilation stack. Each step contains
#    numerous subroutines and passes of its own, and many require solving
#    computationally hard problems (or very good heuristic techniques).
#
# Developing automated compilation tools is an active area of research that is
# critical for quantum software stacks. Even if a library contains
# implementations of specific quantum circuits (e.g., the Fourier transform or a
# multi-controlled operation), without a proper compiler a user would be left to
# optimize and map them to hardware by hand. This is a laborious (and
# error-prone!) process, and furthermore, is unlikely to be optimal.
#
# Suppose we want compile and optimize quantum circuits for Shor's algorithm to
# factor an integer :math:`N`. Recalling the pseudocode above, let's break the
# algorithm down into a few distinct steps to identify where quantum
# compilation happens.
#
#  - Randomly select an integer, :math:`a`, between 2 and
#    :math:`N-1` (double check we didn't get lucky and randomly select one of the true factors)
#  - Execute order finding on a quantum computer, and use the measurement results to  guess a candidate non-trivial square root.
#  - If the square root is non-trivial, test whether we found the factors. Otherwise, try taking more shots, or try a different :math:`a`.
#
# For a full description of Shor's algorithm, the interested reader is referred
# to the `PennyLane Codebook
# <https://pennylane.ai/codebook/10-shors-algorithm/>`_.  The key aspect here is
# that even with a good compiler, every random choice of ``a`` yields a
# different quantum circuit. Each circuit is generated at runtime, and must be
# compiled and optimized, leading to a huge overhead in computation time. One
# could potentially store circuits and subroutines for reuse. But note that they
# depend on both ``a`` *and* ``N``. In a cryptographic context, ``N`` relates to
# a public key which that is unique for every entity. Moreover, for sizes of
# cryptographic relevance, ``N`` will be a 2048-bit integer or larger! This
# amount of reuse and recompilation suggests JIT compilation is a worthwhile
# option to explore.
#
# Quantum just-in-time compilation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In standard PennyLane, quantum circuit execution can be JIT-compiled with
# JAX. To learn more, check out the `JAX documentation
# <https://docs.jax.dev/en/latest/jit-compilation.html>`_  and the `PennyLane
# demo <https://pennylane.ai/qml/demos/tutorial_jax_transformations>`_ on the
# subject. But this only compiles a single circuit; what about all the other
# code around it? What if you also wanted to optimize that quantum
# circuit based on contextual information? This is where Catalyst comes in.
#
# Catalyst enables quantum JIT (QJIT) compilation of the *entire* algorithm from
# end-to-end. On the surface, it looks to be as simple as the following:

import pennylane as qml


@qml.qjit
def shors_algorithm(N):
    # Implementation goes here
    return p, q


######################################################################
# Unfortunately it is not quite so simple, and requires some special-purpose
# functions and data manipulation. But in the next section, we will show how to
# QJIT the most important parts, resulting in the following signature:


@qml.qjit(autograph=True, static_argnums=(2))
def shors_algorithm(N, a, n_bits):
    # Implementation goes here
    return p, q


######################################################################
# We will find ways to leverage knowledge of :math:`a` to
# construct more optimal quantum circuits within the QJITted function.
#
# QJIT compiling Shor's algorithm
# -------------------------------
#
# Quantum subroutines
# ^^^^^^^^^^^^^^^^^^^
#
# This section describes the circuits comprising the quantum subroutine
# in Shor's algorithm, i.e., the order-finding algorithm. The presented
# implementation is based on [#Beauregard2003]_. For an integer :math:`N` with
# an :math:`n = \lceil \log_2 N \rceil`-bit representation, the circuit requires
# :math:`2n + 3` qubits, where :math:`n + 1` are for computation and
# :math:`n + 2` are auxiliary.
#
# Order finding is an application of *quantum phase estimation* (:doc:` QPE
# </demos/tutorial_qpe>`) for the operator
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
# Since :math:`a` is known, we can classically evaluate :math:`a^{2^k} \pmod
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
#    Circuit for controlled multiplication of :math:`ax` using a series of double-controlled Fourier adders (modulo :math:`N`).
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
# being added, and then reverse the QFT to obtain the result. The circuit for
# *Fourier addition*, :math:`\Phi`, is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder.svg
#    :width: 700
#    :align: center
#    :alt: Addition in the Fourier basis.
#
#    Circuit for addition in the Fourier basis [#Draper2000]_. The calligraphic
#    letters indicate states already transformed to the Fourier basis.
#
# Fourier addition of two :math:`n`-bit numbers requires :math:`n+1` qubits, to
# account for the possibility of overflow during addition (this constitutes one
# of our auxiliary qubits). The rotation angles, which depend on the binary
# representation of :math:`a`, are defined as
#
# .. math::
#
#     \mathbf{R}_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i\sum_{\ell=0}^{k} \frac{a_\ell}{2^{\ell+1}}} \end{pmatrix}.
#
# A detailed derivation of this circuit is included in the
# :ref:`appendix <appendix_fourier_adder>`.
#
# Returning to :math:`M_a`, we have a doubly-controlled
# :math:`\Phi_+`. :math:`\Phi_+` is similar to :math:`\Phi`, but by using
# another auxiliary qubit and some extra operations, it will work modulo
# :math:`N`.  An explanation of this circuit's structure is also provided in an
# :ref:`appendix <appendix_fourier_adder_modulo_n>`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n.svg
#    :width: 800
#    :align: center
#    :alt: Addition in the Fourier basis modulo N.
#
#    Circuit for doubly-controlled Fourier addition modulo :math:`N`
#    [#Beauregard2003]_. The calligraphic letters indicate states already
#    transformed to the Fourier basis.
#
# This completes our implementation of the controlled-:math:`U_{a^{2^k}}`. The
# full circuit, shown below, uses :math:`t + 2n + 2` qubits.
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
# :math:`a^{2^k} = 1`, no further multiplication is necessary because we would
# be multiplying by 1. In fact, we can terminate the algorithm early because
# we've found the order of :math:`a` is simply :math:`2^k`.
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
# This last optimization will not be included in our implementation. We again
# expect it to have diminishing returns as more iterations of the algorithm are
# performed.
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
# Consider the bottom estimation wire. After the final Hadamard, this qubit only
# applies controlled operations. Rather than preserving its state, we can make a
# measurement and apply rotations classically controlled on the outcome
# (essentially, the reverse of the deferred measurement process). The same can
# be done for the next estimation wire; rotations applied to the remaining
# estimation wires then depend on the previous two measurement outcomes. We can
# repeat this process for all remaining estimation wires. Moreover, we can
# simply reuse the *same* qubit for *all* estimation wires, provided we keep
# track of the measurement outcomes classically, and apply an appropriate
# :math:`RZ` rotation,
#
# .. math::
#
#     \mathbf{M}_{k} = \begin{pmatrix} 1 & 0 \\ 0 & e^{-2\pi i\sum_{\ell=0}^{k}  \frac{\theta_{\ell}}{2^{k + 2 - \ell}}} \end{pmatrix}.
#
# This allows us to reformulate the QPE algorithm as:
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-7.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with one estimation qubit.
#
# A step-by-step example is included in :ref:`appendix <appendix_single_qubit_qpe>`.
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
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Circuits in hand, we now present the full implementation of Shor's algorithm.
#
# First, we require some utility functions for modular arithmetic:
# exponentiation by repeated squaring (to avoid overflow when exponentiating
# large integers), and computation of inverses modulo :math:`N`. Note that the
# first can be done in regular Python with the built-in ``pow`` method. However,
# it is not JIT-compatible and there is no equivalent in JAX NumPy.

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
# We require a few helper functions for phase estimation.


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
# Below, we have the implementations of the circuits derived in the previous section.

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
# Finally, let's put it all together in the core portion of Shor's algorithm,
# under the ``@qml.qjit`` decorator.

from jax import random


# ``n_bits`` is a static argument because ``jnp.arange`` does not currently
# support dynamically-shaped arrays when jitted.
@qml.qjit(autograph=True, static_argnums=(3, 4))
def shors_algorithm(N, key, a, n_bits, n_trials):
    # If no explicit a is passed (denoted by a = 0), randomly choose a
    # non-trivial value of a that does not have a common factor with N.
    if a == 0:
        key, subkey = random.split(key)
        a = random.randint(subkey, (1,), 2, N - 1)[0]

        while jnp.gcd(a, N) != 1:
            key, subkey = random.split(key)
            a = random.randint(subkey, (1,), 2, N - 1)[0]

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

    p, q = jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)
    successful_trials = jnp.array(0, dtype=jnp.int32)

    for _ in range(n_trials):
        sample = run_qpe(a)
        phase = fractional_binary_to_float(sample)
        guess_r = phase_to_order(phase, N)

        # If the guess order is even, we may have a non-trivial square root.
        # If so, try to compute p and q.
        if guess_r % 2 == 0:
            guess_square_root = (a ** (guess_r // 2)) % N

            if guess_square_root != 1 and guess_square_root != N - 1:
                candidate_p = jnp.gcd(guess_square_root - 1, N).astype(jnp.int32)

                if candidate_p != 1:
                    candidate_q = N // candidate_p
                else:
                    candidate_q = jnp.gcd(guess_square_root + 1, N).astype(jnp.int32)

                    if candidate_q != 1:
                        candidate_p = N // candidate_q

                if candidate_p * candidate_q == N:
                    p, q = candidate_p, candidate_q
                    successful_trials += 1

    return p, q, key, a, successful_trials / n_trials


######################################################################
# Let's ensure the algorithm can successfully factor a small case, :math:`N =
# 15`. We will randomly generate :math:`a` within the function, and do 100 shots
# of the phase estimation procedure to get an idea of the success probability.


key = random.PRNGKey(123456789)

N = 15
n_bits = int(jnp.ceil(jnp.log2(N)))

p, q, _, a, success_prob = shors_algorithm(N, key.astype(jnp.uint32), 0, n_bits, 100)

print(f"Found {N} = {p} x {q} (using random a = {a}) with probability {success_prob:.2f}")

######################################################################
# Performance and validation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next, let's verify QJIT compilation is happening properly. We will run the
# algorithm for different ``N`` with the same bit width, and different values of
# ``a``. We expect the first execution, for the very first ``N`` and ``a``, to
# take longer than the rest.

import time
import matplotlib.pyplot as plt

# Some 6-bit numbers
N_values = [33, 39, 51, 55, 57]
n_bits = int(jnp.ceil(jnp.log2(N_values[0])))

num_a = 3

execution_times = []

key = random.PRNGKey(1010101)

for N in N_values:
    unique_a = []

    while len(unique_a) != num_a:
        key, subkey = random.split(key.astype(jnp.uint32))
        a = random.randint(subkey, (1,), 2, N - 1)[0]
        if jnp.gcd(a, N) == 1 and a not in unique_a:
            unique_a.append(a)

    for a in unique_a:
        start = time.time()
        p, q, key, _, _ = shors_algorithm(N, key.astype(jnp.uint32), a, n_bits, 1)
        end = time.time()
        execution_times.append((N, a, end - start))

        start = time.time()
        p, q, key, _, _ = shors_algorithm(N, key.astype(jnp.uint32), a, n_bits, 1)
        end = time.time()
        execution_times.append((N, a, end - start))

labels = [f"{ex[0]}, {int(ex[1])}" for ex in execution_times][::2]
times = [ex[2] for ex in execution_times]

plt.scatter(range(len(times)), times, c=[ex[0] for ex in execution_times])
plt.xticks(range(0, len(times), 2), labels=labels, rotation=80)
plt.xlabel("N, a")
plt.ylabel("Runtime (s)")
plt.tight_layout()
plt.show()

######################################################################
# This plot demonstrates exactly what we suspect: changing :math:`N` and
# :math:`a` does not lead to recompilation of the program! This will be
# particularly valuable for large :math:`N`, where compilation time may become
# long.


######################################################################
# Conclusions
# -----------
#
# The ability to leverage a tool like Catalyst means we can quickly generate,
# compile, and optimize very large circuits, even within the context of a larger
# workflow. There is still much work to be done, however. For one, the generated
# circuits are not optimized at the individual gate level, so the resource
# counts will be large and impractical. One also observes that, even though
# "higher-level" optimizations become possible, there is still a significant
# amount of manual labour involved in implementing them. Compilation tools that
# can automatically determine how subroutines can be simplified based on
# classical information or on the input quantum superposition would be valuable
# to develop, as they would enable co-optimization of the classical and quantum
# parts of workflows.
#
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
#     *Compilers: Principles, Techniques, And Tools*. Pearson Education, Inc.
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
#
#
# Appendix: fun with Fourier transforms
# -------------------------------------
#
# This appendix provides a detailed derivation of the circuits for addition and
# subtraction in the Fourier basis (both the generic and the modulo :math:`N`
# case), as well as the full explanation of the one-qubit QPE trick.
#
# .. _appendix_fourier_adder:
#
# Standard Fourier addition
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We recall the circuit from the main text.
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
#
# As one may expect, :math:`\Phi^\dagger` performs subtraction. However, we must
# also consider the possibility of underflow.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_adjoint.svg
#    :width: 500
#    :align: center
#    :alt: Subtraction in the Fourier basis.
#
# .. _appendix_fourier_adder_modulo_n:
#
# Doubly-controlled Fourier addition modulo :math:`N`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let's analyze below circuit, keeping in mind :math:`a < N` and :math:`b < N`,
# and the main register has :math:`n + 1` bits.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n.svg
#    :width: 800
#    :align: center
#    :alt: Addition in the Fourier basis modulo N.
#
#    Circuit for doubly-controlled Fourier addition modulo :math:`N`
#    [#Beauregard2003]_. The calligraphic letters indicate states already
#    transformed to the Fourier basis.
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
# .. _appendix_single_qubit_qpe:
#
# QPE with one estimation wire
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Here we show how the inverse QFT in QPE can be implemented using a single
# estimation wire when mid-circuit measurement and feedforward are available.
# Our starting point is the full algorithm below.
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
