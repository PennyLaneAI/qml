r"""Differentiable error mitigation
===========================================

.. meta::
    :property="og:description": Differentiable error mitigation
    :property="og:image": https://pennylane.ai/qml/_images/QEK_thumbnail.png

.. related::

    tutorial_error_mitigation Error mitigation with Mitiq and PennyLane

*Author: KK,.. Posted: 27 July 2022*

This is to try out style stuff without having to re-render the whole tutorial

New emoji lightning ⚡ in math mode :math:`f^{⚡}` or with math mode and in-text :math:`f^\text{⚡}`

in text \Lightning \lightning . inline math mode :math:`\Lightning`, :math:`\lightning`, and major math mode

.. math:: \Lightning \lightning

.. math:: \text{\Lightning} \text{\lightning}

We start by initializing a noisy device under the ``qml.DepolarizingChannel``.
"""



##############################################################################
# Note that the discrepancies between the ideal simulation and exact result are due to the limited expressivity of our Ansatz.


##############################################################################
# Showing off other interfaces and how to get their gradients
# -----------------------------------------------------------
# also discuss jitting here. Comes at a higher cost for compilation but execution is 3 orders of magnitude faster
#
# 
# Discussing differentiation of the mitigation scheme itself
# ----------------------------------------------------------
# Havent figured out how this works, but could be worth it to potentially ignite some sparks.
#
# References
# ----------
#
# .. [#DiffableTransforms]
#
#     Olivia Di Matteo, Josh Izaac, Tom Bromley, Anthony Hayes, Christina Lee, Maria Schuld, Antal Száva, Chase Roberts, Nathan Killoran.
#     "Quantum computing with differentiable quantum transforms."
#     `arXiv:2202.13414 <https://arxiv.org/abs/2202.13414>`__, 2021.
#
# .. [#VAQEM]
#
#     Gokul Subramanian Ravi, Kaitlin N. Smith, Pranav Gokhale, Andrea Mari, Nathan Earnest, Ali Javadi-Abhari, Frederic T. Chong.
#     "VAQEM: A Variational Approach to Quantum Error Mitigation."
#     `arXiv:2112.05821 <https://arxiv.org/abs/2112.05821>`__, 2021.
