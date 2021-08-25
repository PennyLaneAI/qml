r"""

.. _general_rotosolve:

Rotosolve for general quantum gates
===================================

.. meta::
    :property="og:description": Use the Rotosolve optimizer for general quantum gates.
    :property="og:image": general_rotosolve/general_rotosolve_thumbnail.png

.. related::

   tutorial_rotoselect Leveraging trigonometry with Rotoselect
   tutorial_general_parshift Parameter-shift rules for general quantum gates
   tutorial_expressivity_fourier_series Understand quantum models as Fourier series


*Author: David Wierichs (Xanadu resident). Posted: ?? September 2021.*

.............................

|

.. figure:: general_rotosolve/general_rotosolve_thumbnail.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Rotosolve optimization for higher-order Fourier series cost functions.


In this demo we will show how to use the :class:`~.pennylane.optimize.RotosolveOptimizer`
on advanced quantum circuits.
That is, we will consider circuits in which multiple gates are parametrized by the same
variational parameter and look at cost functions that arise when we measure an observable
in the resulting quantum state.
The PennyLane implementation of Rotosolve was updated to optimize such cost functions---if
we only know the quantum gates we are using well enough. *What does that mean,* well enough, *?*,
you might ask---read on and find out!


The Rotosolve idea
------------------

  #. *Super*brief overview of cost functions as Fourier series -> Reference to 
     *Quantum models as Fourier series* demo.
  #. Reconstruction concept via Fourier trafo -> Reference to *General parameter-shift
     rule* demo.
  #. Rotosolve: Slowly walk through one step containing multiple substeps.
"""

def code():
    return

###############################################################################
# New features
# ------------
# 
#   #. Explain ``Rs`` input parameter.
#   #. Explain ``full_output`` and ``reconstructed_output`` options.
#   #. Use on example circuit from *The Rotosolve idea*.

def code():
    return

###############################################################################
# Application: QAOA
# -----------------
# 
#   #. Code up an example (for MaxCut), reference to *QAOA for MaxCut* demo.
#   #. Run new Rotosolve on the QAOA circuit.
#   #. Introduce bound on R for MaxCut
#   #. Run new Rotosolve on the QAOA circuit with bounded R
#   #. Compare cost between 2. and 4.

def code():
    return

###############################################################################
# Who invented Rotosolve?
# -----------------------
# 
# The idea of the Rotosolve algorithm was discovered at least *four* times independently:
# the first preprint to propose the method was `Calculus on Parameterized Quantum Circuits
# <https://arxiv.org/abs/1812.06323>`_. In this work, the authors show how we can reconstruct
# the cost function as described and implemented in the :doc:`demo on general parameter-shift
# rules </demos/tutorial_general_parshift>`. They then propose to exploit this for optimization
# of any parameters that enter the circuit with equidistant frequencies via the Rotosolve
# algorithm, which they name *Coordinate Descent for Quantum Circuits*.
# 
# Second, `Sequential minimal optimization for quantum-classical hybrid algorithms
# <https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043158>`_ proposed
# Rotosolve for Pauli rotations, including the case of multiple rotations being controlled by 
# the same parameter. This yields integer-valued---and in particular equidistant---frequencies
# in the cost function, so that this work covers essentially the same class of functions as the
# first one, leveraging a different perspective on how the cost function spectra arise from the
# circuit. Simultaneous optimization via more expensive higher-dimensional reconstruction is
# considered as well.
# 
# Third, `A Jacobi Diagonalization and Anderson Acceleration Algorithm For Variational Quantum 
# Algorithm Parameter Optimization <https://arxiv.org/pdf/1904.03206.pdf>`_ proposed the 
# Rotosolve method for Pauli rotations only. The specialty here is to combine the method,
# which this preprint calls *Jacobi-1* with `Anderson acceleration
# <https://en.wikipedia.org/wiki/Anderson_acceleration>`_ and to look at simultaneous optimization
# via multi-dimensional Fourier transforms as well.
# 
# Fourth, `Structure optimization for parameterized quantum circuits 
# <https://arxiv.org/abs/1905.09692>`_ presented *Rotosolve* and gave the method its most-used
# name. It is presented together with *Rotoselect*, an extension of Rotosolve that not only 
# optimizes the circuit parameters, but also its structure.
# 
# The first preprints of these four works appeared on the `arXiv <https://arxiv.org/>`_ on
# Dec, 15th, 2018; March, 28th, 2019; April, 5th, 2019 and May, 23rd, 2019.
# A further generalization to arbitrary frequency spectra, albeit at large cost, was presented
# in `General parameter-shift rules for quantum gradients <https://arxiv.org/abs/2107.12390>`_
# in July, 2021.
# 
# |
# 
# And this is it! Hopefully you will join the club of Coordinate Descent / Jacobi-1 / Rotosolve
# enthusiasts and find appliations to use it on.
#
# References
# ----------
#
# .. [#QAD]
#
#     Balint Koczor, Simon Benjamin. "Quantum Analytic Descent".
#     `arXiv preprint arXiv:2008.13774 <https://arxiv.org/abs/2008.13774>`__.
#
# .. [#Rotosolve]
#
#     Mateusz Ostaszewski, Edward Grant, Marcello Benedetti.
#     "Structure optimization for parameterized quantum circuits".
#     `arXiv preprint arXiv:1905.09692 <https://arxiv.org/abs/1905.09692>`__.
#
# .. [#higher_order_diff]
#
#     Andrea Mari, Thomas R. Bromley, Nathan Killoran.
#     "Estimating the gradient and higher-order derivatives on quantum hardware".
#     `Phys. Rev. A 103, 012405 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.012405>`__, 2021.
#     `arXiv preprint arXiv:2008.06517 <https://arxiv.org/abs/2008.06517>`__.
