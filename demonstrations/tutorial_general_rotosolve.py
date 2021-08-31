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


Rotosolve--Coordinate descent meets Fourier series
--------------------------------------------------
The first ingredient we will need are the cost functions that Rotosolve can tackle.
They will typically take the form

.. math ::

    E(\boldsymbol{x}) = \langle \psi|U_1(x_1)^\dagger\cdots U_n(x_n)^\dagger B U_n(x_n)\cdots U_1(x_1) |\psi\rangle

where $|\psi\rangle$ is some initial state, $B$ is a Hermitian observable, and $U_j(x_j)$ are 
parametrized unitaries that encode the dependence on the variational parameters $\boldsymbol{x}$.
Let's set up a simple example circuit as a toy model; more details on which cost functions can 
be handled by Rotosolve and a more complex example can be found further below.
"""

import pennylane as qml

dev = qml.device('default.qubit', wires=3)
@qml.qnode(dev)
def cost(x, y, z):
    for _x, w in zip(x, dev.wires):
        qml.RX(_x, wires=w, id=f"x{w}")

    for i in range(dev.num_wires):
        qml.CRY(y[0], wires=[i, (i + 1) % dev.num_wires], id="y0")

    qml.RZ(z[0], wires=0, id="z0")
    qml.RZ(z[1], wires=1, id="z1")
    qml.RZ(z[1], wires=2, id="z1")

    return qml.expval(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2))

###############################################################################
# This function takes three arguments, an array ``x`` with three parameters,
# a single parameter ``y``, and a two-parameter array ``z``. We will need some
# initial parameters later on:

from pennylane import numpy as np

x_0 = np.array([-0.2, 1.6, -0.3])
y_0 = np.array([0.711])
z_0 = np.array([-1.24, 0.84])
initial_cost = cost(x_0, y_0, z_0)
print(f"The initial cost is: {initial_cost:.5f}")

###############################################################################
# It turns out that we then know $E$ to be a $n$-input *Fourier series* of the parameters.
# This enables us to understand the full functionality of the circuit and once we know the
# coefficients of this series, we do not even need to fire up a quantum machine anymore.
# However, obtaining all coefficients of the series is expensive---very expensive, actually---as we
# would need to measure the original $E$ at a number of sampling points that grows *exponentially*
# with $n$.
# 
# Instead, Rotosolve makes use of `coordinate descent
# <https://en.wikipedia.org/wiki/Coordinate_descent>`__. This is a basic idea for optimizing
# functions depending on multiple parameters: simply optimize the parameters one at a time!
# If we restrict cost functions like the one above to a single parameter $x_j$, we again get 
# a Fourier series, but this time it only depends on $x_j$:
# 
# .. math ::
# 
#     E_j(x_j) = a_0 + \sum_{\ell=1}^R a_\ell \cos(\Omega_\ell x_j) + b_\ell \sin(\Omega_\ell x_j)
# 
# Here, $\{a_\ell\}$ and $\{b_\ell\}$ are the coefficients and $\{\Omega_\ell\}$ are the frequencies
# of the series, the number $R$ and values of which are dictated by the unitary $U_j(x_j)$.
# 
# For our simple toy cost function above, the :mod:`~.pennylane.fourier` module can tell us
# what the frequency spectra for the different parameters are:

spectra = qml.fourier.spectrum(cost)(x_0, y_0, z_0)
print(*(f"{key}: {val}" for key, val in spectra.items()), sep='\n')
frequencies = [[f for f in freqs if f>0.0] for freqs in spectra.values()]

###############################################################################
# In the last line we here prepared ``frequencies`` for usage with 
# :class:`~.pennylane.optimize.RotosolveOptimizer`  later on, by simply removing 
# non-positive frequencies.
#
# If you are interested in details on why $E_j$ is a Fourier series, how the frequencies come about,
# and how one can make use of this knowledge for quantum machine learning, we recommend the
# :doc:`Fourier series expressiveness tutorial </demos/tutorial_expressivity_fourier_series>` as
# further reading.
# 
# Finding the coefficients of the Fourier series $E_j$ for a single parameter is much more
# reasonable than the exponential cost of doing so for the full cost function $E$.
# And finding the coefficients of a one-dimensional (finite) Fourier series actually is 
# well-known---it's simply a `(discrete) Fourier transform (DFT)
# <https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`__!
# How this can be implemented using PennyLane, which is also how Rotosolve will
# do it, is discussed in the :doc:`General parameter-shift rule tutorial 
# </demos/tutorial_general_parshift>`.
# 
# Once we know the coefficients $\{a_\ell\}$ and $\{b_\ell\}$, we know the entire one-dimensional
# function $E_j$---recall that we know the frequencies $\{\Omega_\ell\}$ because we know which 
# unitary operations we use in the circuit---and can optimize it using a classical computer. 
# This is not such a hard challenge because many global optimization strategies are very 
# efficient in one dimension.
# 
# Now that we have all ingredients, let us recollect the full Rotosolve algorithm:
# 
#   #. Loop over Rotosolve iterations
#   #.    Loop over parameters (index $j$)
#   #.        Reconstruct $E_j$ via DFT $\Rightarrow$ $\hat{E}_j$
#   #.        Minimize $\hat{E}_j$ and update $x_j$ to the minimizer
# 
# As we can see, the structure is quite short and simple. As inputs we only require
# the cost function (to obtain the samples for the DFT), the frequency spectrum 
# $\{\Omega_\ell\}$ per parameter (to choose where to sample $E_j$ and perform the DFT), 
# and some initial parameters.
#
# Let's continue by applying the PennyLane implementation 
# :class:`~.pennylane.optimize.RotosolveOptimizer` of Rotosolve to our toy circuit from above.

num_freqs = [len(freqs) for freqs in frequencies]
num_freqs = [num_freqs[:3], num_freqs[3], num_freqs[4:]]
opt = qml.RotosolveOptimizer()
num_steps = 10
x, y, z = x_0, y_0, z_0
for step in range(num_steps):
    print(f"After {step} steps, the cost is:       {cost(x, y, z):.5f}")
    x, y, z = opt.step(cost, x, y, z, num_freqs=num_freqs)
print(f"The final cost after {num_steps} steps is: {cost(x, y, z):.5f}")
print(x_0)

###############################################################################
# The optimization with Rotosolve worked! We arrived at the minimal eigenvalue of the
# observable $B=X\otimes X\otimes X$ that we are measuring in th circuit.
# 
# New features
# ------------
# 
# Maybe you already noticed the ``num_freqs`` argument we passed to ``opt.step`` above
# during the optimization. It describes the number of frequencies per parameter, i.e.
# it captures the numbers $R$ for each of the inputs. The current implementation of
# Rotosolve assumes the frequencies to be the integers $[1,\cdots R]$ and will use
# this assumption to perform the DFT during its ``step``. See below for details on how
# to cover more general parameter dependencies.
#
# Note that the ``num_freqs`` argument is expected to either be
#  
#   #. an integer that applies to all input arguments of ``cost``,
#   #. an integer per argument of ``cost``,
#   #. a list of integers per argument of ``cost``, or
#   #. a mixture of 2. and 3.
#
# Please refer to the `documentation 
# <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.RotosolveOptimizer.html>`__ 
# for details.
#
# Regarding the one-dimensional minimization, we did not tell the ``RotosolveOptimizer`` which
# method to use. It therefore used its default, which is a `grid search
# <https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search>`__ implemented via
# `SciPy's optimize.brute 
# <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html>`__.
# If we want to use a different strategy, we can pass it as a function that takes a 
# one-dimensional function and some keyword arguments and returns a minimum position ``x_min``
# and value ``y_min``. An example based on `SciPy's optimize.shgo
# <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html>`__
# would be:

from scipy.optimize import shgo

def shgo_optimizer(fun, **kwargs):
    r"""Wrapper for ``scipy.optimize.shgo`` (Simplicial Homology global optimizer)."""
    opt_res = shgo(fun, **kwargs)
    return opt_res.x, opt_res.fun
kwargs = {'bounds': ((-np.pi, np.pi),)}

###############################################################################
# Then we can reset the parameters to the initial values and run the optimization
# a second time with this new one-dimensional optimizer subroutine.

x, y, z = x_0, y_0, z_0
for step in range(num_steps):
    print(f"After {step} steps, the cost is:       {cost(x, y, z):.5f}")
    x, y, z = opt.step(
        cost, x, y, z, num_freqs=num_freqs, optimizer=shgo_optimizer, optimizer_kwargs=kwargs
    )
print(f"The final cost after {num_steps} steps is: {cost(x, y, z):.5f}")

###############################################################################
# Alternatively, the SHGO optimizer can be selected via ``optimizer='shgo'``.
# Note that for parameters with $R=1$, the minimum is known analytically and the
# optimizer is not used for those parameters.
# 
# The updated implementation also comes with a few convenience functionalities. 
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
