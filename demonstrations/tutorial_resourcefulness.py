r"""
Analysing quantum resourcefulness with the generalized Fourier transform
========================================================================

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_resourcefulness.png
    :align: center
    :width: 70%
    :alt: DESCRIPTION.
    :target: javascript:void(0)

Resource theories in quantum information theory ask how "complex" a given quantum state is with respect to a certain
measure of complexity. For example, using the resource of entanglement, we can ask how entangled a quantum state is. Other well-known
resources are Clifford stabilizerness, which measures how close a state is from being prepared by a circuit that only uses
classically simulatable Clifford gates, or Gaussianity, which measures how far away a state is from a so-called "Gaussian state"
that is relatively easy to prepare in quantum optics, and likewise classically simulatable. As the name "resourceful" suggests,
these measures of complexity often relate to how much "effort" states are, for example with respect to classical simulation or
preparation in the lab.

It turns out that the resourcefulness of quantum states can be investigated with tools from generalised Fourier analysis [#Bermejo_Braccia]_.
Fourier analysis here refers to the well-known technique of computing Fourier coefficients of a mathematical object, which in our case
is not a function over :mathbb:`R` or :mathbb:`Z`, but a quantum state. "Generalised" indicates that we don't use the
standard Fourier transform, but its group-theoretic generalisations [LINK TO RELATED DEMOS]. This is important, because
[#Bermejo_Braccia]_ link a resource to a group -- essentially, by defining the set of unitaries that maps resource-free
states to resource-free states as a "representation" of a group. The intuition, however, is exactly the same as in the
standard Fourier transform, where large higher-order Fourier coefficients indicate a less "smooth" function.

In this tutorial we will explain the idea of generalised Fourier analysis for resource theories first using the standard
Fourier decomposition of a function. We will then consider the same concepts, but analysing the entanglement
 resource of quantum states, reproducing Figure 2 in [#Bermejo_Braccia]_.

.. figure:: ../_static/demonstration_assets/resourcefulness/figure2_paper.png
   :align: center
   :width: 70%
   :alt: Fourier coefficients, or projections into "irreducible subspaces", of different states using 2-qubit entanglement as a resource.
         A Bell state, which is maximally entangled, has high contributions in higher-order Fourier coefficients, while
         a tensor product state with little entanglement has contributions in lower-order Fourier coefficients. The interpolation
         between the two extremes, exemplified by a Haar random state, has a Fourier spectrum in between.

Luckily, in this case the bases for the subspaces are associated with Pauli operators, and generalised Fourier analysis
can be done by computing Pauli expectations, saving us from diving too deep into representation theory.

.. note::
    Note that all methods discussed here are classical methods to analyse properties of quantum states,
    and of course, they will scale only as much as the mathematical objects involved can be efficiently described classically.
    It is a fascinating question in which situations the Fourier coefficients of a physical states could be read out on a quantum computer, which can
    sometimes perform the block-diagonalisation efficiently.


Standard Fourier analysis through the lense of resources
--------------------------------------------------------

[TODO: code up usual Fourier transform, and link to regular representation]

"""

import numpy as np

######################################################################
# Fourier analysis of entanglement
# --------------------------------
#
# [TODO: explain the change to other representations, and why the Pauli stuff works. Reproduce Fig2]
#
#


#
# References
# ----------
#
# .. [#Bermejo_Braccia]
#
#     Bermejo, Pablo, Paolo Braccia, Antonio Anna Mele, Nahuel L. Diaz, Andrew E. Deneris, Martin Larocca, and M. Cerezo. "Characterizing quantum resourcefulness via group-Fourier decompositions." arXiv preprint arXiv:2506.19696 (2025).
#


######################################################################
#
# About the author
# ----------------
#
