r"""How to build vibrational Hamiltonians
=========================================
Vibrational motions are crucially important to describe the quantum properties of molecules and
materials. Molecular vibrations can significantly affect the outcome of chemical reactions. There
are also several spectroscopy techniques that rely on the vibrational properties of molecules to
provide valuable insight to understand chemical systems and design new materials. Classical quantum
computations have been routinely implemented to describe vibrational motions of molecules. However,
efficient classical methods typically have fundamental theoretical limitations that prevent their
practical implementation for describing challenging vibrational systems. This makes quantum
algorithms an ideal choice where classical methods are not efficient or accurate.

Quantum algorithms require a precise description of the system Hamiltonian to compute vibrational
properties. In this demo, we learn how to use PennyLane features to construct and manipulate
different representations of vibrational Hamiltonians. We also briefly discuss the implementation of
the Hamiltonian in an interesting quantum algorithm for computing the vibrational dynamics of a
molecule.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# Vibrational Hamiltonian
# -----------------------
# A molecular vibrational Hamiltonian can be defined in terms of the kinetic energy operator of the
# nuclei, :math:`T`, and the potential energy operator, :math:`V`, that describes the interaction  between
# the nuclei as:
#
# .. math::
#
#     H = T + V.
#
#
# Christiansen representation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The
#
# Conclusion
# ----------
# The
#
# References
# ----------
#
# .. [#aaa]
#
#     N. W. ,
#     "".
#
# .. [#bbb]
#
#     D. ,
#     "",
#     2008.
#
# About the authors
# -----------------
#
