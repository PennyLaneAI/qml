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
# As quantum computing hardware continues to scale up, the way we write and
# interact with quantum software is evolving. Writing and optimizing quantum
# circuits by hand for algorithms with hundreds or thousands of qubits is
# unsustainable, even for the most seasoned quantum programmers. To develop
# large-scale algorithms, we need frameworks that allow us to sit at a
# comfortable level of abstraction, and tools we can trust to do the heavy
# lifting under the hood. The integration of version 0.34 of PennyLane with
# `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/latest/index.html>`_
# represents a positive step in this direction. This demonstration shows how
# Catalyst enables an implementation of Shor's factoring algorithm that is
# just-in-time compiled from end-to-end, classical control structure and all.
# We also focus on how 
# 
# Compiling classical and quantum code
# ------------------------------------
#
# Hybrid quantum-classical algorithms
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The past few years stimulated a lot of discussion about *hybrid
# quantum-classical algorithms*. For a time, this terminology was synonymous
# with *variational algorithms*. However, integration with classical
# co-processors is necessary for every quantum algorithm, even ones we consider
# quintessentially quantum.
#
# For instance, Shor's famous factoring algorithm leverages an exponential
# speedup afforded by quantum order finding. But, have a look at the expression
# of Shor's algorithm in the code below:

def shors_algorithm(N):
    # Potential factors, p * q = N
    p, q = 0, 0

    while p * q != N:
        # Randomly select a number between 2 and N - 1
        a = random.choice(2, N - 1)

        # Check if it is already a factor of N
        if gcd(N, a) != 1:
             p = gcd(N, a)
             return p, N // p

        # If not, run a quantum subroutine to guess the order r s.t. a ** r = N
        guess_r = guess_order(N, a)

        # Check validity of solution
        if guess_r % 2 == 0:
            guess_square_root = (a ** (guess_r // 2)) % N

            # Ensure the guessed solution is non-trivial
            if guess_square_root not in [1, N - 1]:
                p = gcd(N, guess_square_root - 1)
                q = gcd(N, guess_square_root + 1)

    return p, q

######################################################################
# If you didn't know this was Shor's algorithm, would you even realize it was
# quantum? The classical and quantum parts are closely intertwined, as output
# sampled from a quantum subroutine is post-processed by classical
# number-theoretic routines. Furthermore, the abstraction level is so high that
# there is not a quantum circuit in sight! A programmer doesn't actually need to
# know anything quantum is happening, as long as the software library can
# effectively generate and compile appropriate quantum code (though, they should
# probably have at least some awareness, since the output of ``guess_order`` is
# probabilistic!). This raises the question, then, of what gets compiled, and
# how.
#
# Quantum compilation
# ^^^^^^^^^^^^^^^^^^^
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
#
#
#
#
# Shor's algorithm ----------------

# First, let's do a quick recap of Shor's algorithm.

# The classical part
# ^^^^^^^^^^^^^^^^^^

# TODO: graphic

# TODO: brief explanation of the number theory behind the algo


# TODO: insert code here

######################################################################
# The quantum part
# ^^^^^^^^^^^^^^^^
# 
# TODO: explain how the modular exponentiation circuits work (not sure yet how to
# best include these, because some are quite large)
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
# About the author
# ----------------
# .. include:: ../_static/authors/olivia_di_matteo.txt
