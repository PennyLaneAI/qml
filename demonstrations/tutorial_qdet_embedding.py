r"""Quantum Defect Embedding Theory (QDET)
=========================================
Efficient simulation of complex quantum systems remains a significant challenge in chemistry and
physics. These simulations often require computationally intractable methods for a complete
solution. However, many interesting problems in quantum chemistry and condensed matter physics
feature a strongly correlated region, which requires accurate quantum treatment, embedded within a
larger environment that could be properly treated with cheaper approximations.  Examples of such
systems include point defects in materials [], active site of catalysts [], surface phenomenon such
as adsorption [] and many more. Embedding theories serve as powerful tools for effectively
addressing such problems.

The core idea behind embedding methods is to partition the system and treat the strongly correlated
subsystem accurately, using high-level quantum mechanical methods, while approximating the effects
of the surrounding environment in a way that retains computational efficiency. In this demo, we show
how to implement the quantum defect embedding theory (QDET). The method has been successfully
applied to calculate [...]. An important advantage of QDET is its compatibility with quantum
algorithms as we explain in the following sections. The method can be implemented for calculating
a variety of ground state, excited state and dynamic properties of materials. These make QDET a
powerful method for affordable quantum simulation of materials.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# title
# ---------------------
# PennyLane
#
# sub
# ^^^^^^^^^^^^^^^^^^^
# The
#
# .. math::
#
#     H =
#
# The terms :math:`c`
#
# Conclusion
# ----------
# The
#
# References
# ----------
#
# .. [#ashcroft]
#
#     N. W. Ashcroft, D. N. Mermin,
#     "Solid State Physics", Chapter 4, New York: Saunders College Publishing, 1976.
#
# .. [#jovanovic]
#
#     D. Jovanovic, R. Gajic, K. Hingerl,
#     "Refraction and band isotropy in 2D square-like Archimedean photonic crystal lattices",
#     Opt. Express 16, 4048, 2008.
#
# About the authors
# -----------------
#
