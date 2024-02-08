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
# 
#

# JIT compiling classical and quantum code
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# In a more traditional quantum programming setting, the `run_qpe` subroutine is
# like a quantum black box. It is written separately, optimized separately
# (using your favourite quantum circuit optimization tricks), and simply plugged
# into the classical control flow as needed.

# TODO: Flow chart of Shor's algo with a black box

# More importantly, though, consider that in Shor's algorithm, the structure of
# this black box is not fixed. It depends on a randomly-chosen integer,
# :math:`a`.  All the modular exponentiation circuits using in the QPE algorithm
# under the hood depend directly on this :math:`a`. Even if you automate the
# construction of the circuits a function of :math:`a`, you would still have to
# run them through a quantum compilation procedure prior to execution, which
# would take a substantial amount of time!

# TODO: A graphic depicting circuits constructed with different a, each being
# fed into the compilation stack separately.

# This is where Catalyst comes in. With Catalyst, we can apply
# just-in-compilation to the construction and optimization of these
# circuits. Moreover, we can do so within the larger context of the rest of the
# algorithm, including all the classical control flow around it!

# Shor's algorithm
# ----------------

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
# .. [#Handbook]
#
#     C. J. Colbourn and J. H. Dinitz (2006) *Handbook of Combinatorial Designs,
#     Second Edition*.  Chapman & Hall/CRC.
#
# About the author
# ----------------
# .. include:: ../_static/authors/olivia_di_matteo.txt
