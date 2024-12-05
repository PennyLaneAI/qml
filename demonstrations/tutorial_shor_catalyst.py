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
# In this section we describe the circuits that comprise that quantum part of
# Shor's algorithm, i.e., the order-finding routine. The presented
# implementation is based on [#Beauregard2003]_. For an integer :math:`N` with
# an :math:`n = \lceil \log_2 N \rceil`-bit representation, the circuit requires
# :math:`2n + 3` qubits: :math:`n + 1` are used for computation, while the
# remaining :math:`n + 2` are auxiliary.
#
# Order finding is an application of *quantum phase estimation*. The operator
# whose phase is being estimated is :math:`U_a`,
#
# .. math::
#
#     U_a \vert x \rangle = \vert ax \pmod N \rangle
#
# where a computational basis state :math:`\vert x \rangle` is the binary
# representation of integer :math:`x`, and :math:`a` is the randomly-generated
# integer discussed above. The full QPE circuit is shown below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full.svg
#    :scale: 120%
#    :align: center
#    :alt: Quantum phase estimation circuit for order finding.
#
# This high-level view hides its complexity, though. First and foremost, the
# actual implementation details of :math:`U_a` are not yet defined. In fact, a
# significant number of auxiliary qubits are required. There is also the issue
# of precision, which is governed by the number of estimation wires,
# :math:`t`. Increasing :math:`t` means finding a solution is more likely, but
# add overhead in circuit depth, and in classical simulation due to the
# increased size of Hilbert space.
#
# In what follows, we'll leverage shortcuts afforded by the hybrid nature of the
# computation, and from Catalyst. Specifically, with mid-circuit measurement and
# reset we can reduce the number of estimation wires to :math:`t=1`. A great
# deal of the arithmetic will be performed in the Fourier basis; and since we
# know :math:`a` in advance, we can vary circuit structure on the fly and save
# resources. Finally, additional mid-circuit measurements can be used in lieu of
# uncomputation.
#
# First, let's consider the controlled :math:`U_a^{2^k}` to see how our
# knowledge of classical parameters can simplify the computation. Naively, we
# it looks like we must implement a controlled :math:`U_a` operation :math:`2^k`
# times. However, note
#
# .. math::
#
#     U_a^{2^k}\vert x \rangle = \vert (a \cdot a \cdots a) x \pmod N \rangle = \vert a^{2^k}x \pmod N \rangle = U_{a^{2^k}} \vert x \rangle
#
# Since :math:`a` is known in advance, we can classically evaluate all the :math:`a^{2^k}`,
# and implement controlled-:math:`U_{a^{2^k}}` instead.
#
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power.svg
#    :scale: 120%
#    :align: center
#    :alt: Order finding with controlled operations that take advantage of classical precomputation.
#
# There is a tradeoff here: if we were performing circuit optimization, we would
# have to separately optimize the circuit for each :math:`a^{2^k}`. However, the
# additional compilation time could be outweighed by the fact that we now have
# only :math:`t` controlled operations to implement, rather than :math:`1 + 2 +
# 4 + \cdots + 2^{t-1} = 2^t - 1`. Furthermore, we'll be able to jit-compile the
# circuit construction.
#
# Next, let's zoom in on an arbitrary controlled-:math:`U_a`. 
# 
# .. figure:: ../_static/demonstration_assets/shor_catalyst/c-ua.svg
#    :scale: 120%
#    :align: center
#    :alt: Quantum phase estimation circuit for order finding.
#
# The control qubit, :math:`\vert c\rangle`, is an estimation qubit. The
# register :math:`\vert x \rangle` and the auxiliary register contain :math:`n`
# and :math:`n + 1` qubits respectively, for reasons we elaborate on below.
#
# The constituent operation :math:`M_a` multiplies the contents of one register
# by :math:`a` and adds it to another register, in place and modulo :math:`N`,
# 
# .. math::
#
#     M_a \vert x \rangle \vert b \rangle \vert 0 \rangle =  \vert x \rangle \vert (b + ax) \pmod N \rangle \vert 0 \rangle
#
#
# Ignoring the control qubit, let's validate that this circuit implements
# :math:`U_a`. For readability, we'll omit the "mod :math:`N`", which is
# implicit on all arithmetic.
#
# .. math::
#
#     \begin{eqnarray}
#       M_a \vert x \rangle \vert 0 \rangle^{\otimes n + 1} \vert 0 \rangle &=&  \vert x \rangle \vert ax \rangle \vert 0 \rangle \\
#      SWAP (\vert x \rangle \vert ax \rangle ) \vert 0 \rangle &=&  \vert ax \rangle \vert x \rangle \vert 0 \rangle \\
#     M_{a^{-1}}^\dagger \vert ax \rangle \vert x \rangle  \vert 0 \rangle &=& \vert ax\rangle \vert x - a^{-1}(ax) \rangle \vert 0 \rangle \\
#      &=& \vert ax \rangle \vert 0 \rangle^{\otimes n + 1} \vert 0 \rangle
#     \end{eqnarray}
#
# Note that the adjoint of :math:`M_a` is a subtraction operation.
#
# At a high level, the implementation of :math:`M_a` looks like this:
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/doubly-controlled-adder.svg
#    :scale: 110%
#    :align: center
#    :alt: In-place addition modulo N with the Fourier adder.
#
# At first glance, is it not clear how exactly :math:`a x` is getting
# created. The qubits in register :math:`\vert x \rangle` are being used as
# control qubits for operations that depend on :math:`a` multiplied by various
# powers of 2. There is also a QFT before and after those operations, whose
# purpose is unclear.
#
# These special operations are actually performing *addition in the Fourier
# basis* [#Draper2000]_. This is another trick we can leverage with prior
# knowledge of :math:`a`. Rather than performing explicit addition on bits in
# the computational basis state registers, we can apply a Fourier transform,
# adjust the phases based on the bit values of the number we wish to add, then
# inverse Fourier transform to obtain the result. We present the circuit for the
# *Fourier adder*, :math:`\Phi`, below.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder.svg
#    :scale: 120% 
#    :align: center
#    :alt: Addition in the Fourier basis.
#
# The :math:`\mathbf{R}_k` gates are phase shifts, to be described below. To see
# how this works, let's take a closer look first at the QFT itself, as seen in
# the circuit below. The qubit ordering is such that for an :math:`n`-bit
# integer :math:`b`, :math:`\vert b\rangle = \vert b_{n-1} \cdots b_0\rangle`
# and :math:`b = \sum_{k=0}^{n-1} 2^k b_k`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-1.svg 
#    :scale: 140%
#    :align: center
#    :alt: The Quantum Fourier Transform.
#
# The :math:`R_k` gates are defined as
#
# .. math::
#
#     R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{\frac{2\pi i}{2^k}} \end{pmatrix}
#
# Let's now add a new register of qubits, :math:`\vert a \rangle` representing
# integer :math:`a`. Below, we will modify the phases in :math:`b` following the
# Fourier transform in a very particular way:
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-2.svg 
#    :scale: 140%
#    :align: center
#    :alt: Adding one integer to another with the Quantum Fourier Transform.
#
# Using these phase gates, we observe each qubit in the :math:`\vert b \rangle`
# register picks up an amount of phase depending on the bits in :math:`a`. In
# particular, bit :math:`b_k` accumulates information about all the bits in
# :math:`a` with an equal or lower index, :math:`a_0, \ldots, a_{k}`. The effect
# is that of adding :math:`a_k` to :math:`b_k`. Across the entire register, we
# are then adding :math:`a` to :math:`b`, up to an inverse Fourier transform!
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-3.svg 
#    :scale: 140%
#    :align: center
#    :alt: Adding one integer to another with the Quantum Fourier Transform.
#
# However, we must be careful. Fourier basis addition is *not* automatically
# modulo :math:`N`. If the sum :math:`b + a` requires :math:`n+ 1` bits, it will
# overflow. So, we add one extra qubit to the top of the :math:`\vert b\rangle`
# register, initialized to :math:`\vert 0 \rangle`, to handle that. This is the
# source of one of the auxiliary qubits mentioned earlier.
#
# Moreover, suppose we know all the :math:`a_k` in advance. We don't
# even need a second register of qubits controlling the phase addition;
# we can simply add as much phase as we need each time! In other words, on qubit
# :math:`\vert b_k \rangle`, we need to apply a phase rotation of
# :math:`\sum_{\ell=0}^{k} \frac{a_\ell}{2^{\ell+1}}`. This can be achieved using the gate 
#
# .. math::
#
#     \mathbf{R}_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i\sum_{\ell=0}^{k} \frac{a_\ell}{2^{\ell+1}}} \end{pmatrix}
#
# The final circuit for the Fourier adder is
# 
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_explanation-4.svg 
#    :scale: 140%
#    :align: center
#    :alt: Full Fourier adder.
#
# As one may expect, :math:`\Phi^\dagger` performs subtraction. However, we must
# also consider the possibility of underflow in the output.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_adjoint.svg 
#    :scale: 120%
#    :align: center
#    :alt: Subtraction in the Fourier basis.
#
# Returning to our implementation of :math:`M_a`, we see an operation
# :math:`\Phi_+` which is similar to :math:`\Phi`, but it (a) uses an auxiliary
# qubit, and (b) works modulo :math:`N`. The idea behind :math:`\Phi_+` is to
# still use Fourier basis addition and subtraction, but apply corrections if
# overflow is detected.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/fourier_adder_modulo_n.svg
#    :scale: 100%
#    :align: center
#    :alt: Addition in the Fourier basis modulo N.
#
# In the circuit above, we first add :math:`a` to :math:`b`, then subtract
# :math:`N` if :math:`a + b > N`. However, if that wasn't the case, we
# subtracted :math:`N` for no reason, causing underflow. This would manifest as
# a 1 in the top-most qubit (recall this is the auxiliary qubit added to the
# original Fourier adder :math:`\Phi` to account for overflow). That 1 can be
# detected by applying a CNOT down to the auxiliary qubit, which performs
# controlled addition of :math:`N` if needed. Note we must exit the Fourier
# basis to detect the underflow. The remainder of the circuit returns the
# auxiliary qubit to its original state.
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
# Recall once again that we are using this to implement a single :math:`M_a`
# operation.  Returning to the implementation of :math:`M_a`, we are adding
# :math:`2^{k}a` modulo :math:`N` to :math:`b`, conditioned on the value of the
# bit :math:`x_{k}`, in the Fourier basis. We can re-express this as a sum (all
# modulo :math:`N`) to find
#
# .. math::
#
#     \begin{equation*}
#     b + x_{0} \cdot 2^0 a + x_{1} \cdot 2^1 a + \cdots x_{n-1} \cdot 2^{n-1} a  = b + a \sum_{k=0}^{n-1} x_{k} 2^k =  b + a x
#     \end{equation*}
#
# This completes our implementation of the controlled-:math:`U_{a^{2^k}}`
# operations. The current qubit count is :math:`t + 2n + 2`. There is one
# major optimization left to make: reducing the number of estimation qubits from
# :math:`t` to 1.
#
# Let's return to our original picture of the QPE routine, and expand the
# inverse QFT at the end.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded.
#
# Look carefully at the last estimation qubit: after the final Hadamard, it is
# used only for controlled gates. As such, we can simply measure it, and apply
# subsequent operations controlled on the classical outcome, :math:`\theta_0`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-2.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded and last estimation qubit measured off.
#
# But, we can once again do better by dynamically modifying the circuit based on
# this classical information. Instead of applying a controlled
# :math:`R^\dagger_2`, we can apply :math:`R^\dagger` where the rotation angle is 0 if :math:`\theta_0 = 0`, and
# :math:`\pi` if :math:`\theta_1`, i.e., :math:`R^\dagger_{2 \theta_0}`.
# The same can be done for all other gates controlled on :math:`\theta_0`.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-3.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded, last estimation qubit measured, and rotation gates adjusted.
#
# Now, let's leverage this trick again with the second last estimation qubit,
# and improve things even more by noting that once the last qubit is measured,
# we can reset and repurpose it to play the role of the second last qubit.
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
#    :scale: 90%
#    :align: center
#    :alt: QPE circuit with inverse QFT expanded, last estimation qubit reused, and rotation gates adjusted.
#
# We can now do this for each remaining estimation qubit, adding more and more
# rotations as we go, each depending on the previous measurement outcomes.
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-6.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with one estimation qubit, and unmerged rotation gates.
#
# Finally, since these are simply :math:`RZ` gates, we can merge each group into
# one. Define
#
# .. math::
#
#     \mathbf{M}_{k} = \begin{pmatrix} 1 & 0 \\ 0 & e^{-2\pi i\sum_{\ell=0}^{k}  \frac{\theta_{\ell}}{2^{k + 2 - \ell}}} \end{pmatrix}
#
# With a bit of index gymnastics, we obtain our final QPE algorithm with a single estimation qubit:
#
# .. figure:: ../_static/demonstration_assets/shor_catalyst/qpe_full_modified_power_with_qft-7.svg
#    :width: 800
#    :align: center
#    :alt: QPE circuit with one estimation qubit.
#
# Replacing the controlled :math:`U` gates with the subroutines derived above, Shor's
# algorithm requires :math:`2n + 3` qubits in total.
#

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
# .. [#Draper2000]
#
#     Thomas G. Draper (2000) *Addition on a Quantum Computer.*
#     arXiv preprint, arXiv:quant-ph/0008033.
#
# About the author
# ----------------
# .. include:: ../_static/authors/olivia_di_matteo.txt
