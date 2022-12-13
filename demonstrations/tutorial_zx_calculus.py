"""
ZX calculus
===========

.. meta::
    :property="og:description": Investigation of ZX calculus and its applications to quantum computing
    :property="og:image": https://pennylane.ai/qml/_images/zx.png

*Author: Romain. Posted: January 2023.*


The ZX calculus is a graphical language that can represent any linear maps. Therefore it can be used to reason
quantum computations and circuits. Its foundations are based on category theory, which makes it rigorous. It was
introduced in 2008 by Coecke and Duncan [#Coecke]_ .

many uses

In this tutorial, we first give an overview of the building blocks of the ZX-diagram and the main rewriting rules.
Then we will explore how to optimize the number of T-gates of a benchmark circuit with PennyLane and PyZX [PyZX]_.
We also show that reduction do not always end up with diagram-like graph, and that circuit extraction is a main
pain-point of the framework. Finally, we give some leads about ZX-calculus advanced uses.

Introduction
------------

[#JvdW2020]

.. figure:: ../demonstrations/zx_calculus/zx.png
    :align: center
    :width: 70%

    Comment here

"""

import pennylane as qml
import pyzx


######################################################################
# Math
# ----
#
# .. math::
#
#    S_s X_i = \left( Z_i Z_a Z_b Z_c \right) X_i = - X_i S_s.
#
#
######################################################################
# T-count optimization
# ^^^^^^^^^^^^^^^^^^^
#
######################################################################
# Graph simplification and circuit extraction
# -------------------------------------------

######################################################################
# Advanced use
# ^^^^^^^^^^^^
#
# References
# ----------
#
# .. [#Kissinger2021]
#
#    Aleks Kissinger and John van de Wetering. "Reducing T-count with the ZX-calculus."
#    `ArXiv <https://arxiv.org/pdf/1903.10477.pdf__.
#
# .. [#Coecke2011]
#
#    Bob Coecke and Ross Duncan. "Interacting quantum observables: categorical algebra and diagrammatics."
#    `New Journal of Physics <https://iopscience.iop.org/article/10.1088/1367-2630/13/4/043016/pdf>`__.
#
#
# .. [#Coecke]
#
#    Bob Coecke and Ross Duncan. "A graphical calculus for quantum observables."
#    `Oxford <https://www.cs.ox.ac.uk/people/bob.coecke/GreenRed.pdf>`__.
#
# .. [#East2021]
#
#    Richard D. P. East, John van de Wetering, Nicholas Chancellor and Adolfo G. Grushin. "AKLT-states as ZX-diagrams: diagrammatic reasoning for quantum states."
#    `ArXiv <https://arxiv.org/pdf/2012.01219.pdf>`__.
#
# .. [#PyZX]
#
#    John van de Wetering. "PyZX."
#    `PyZX GitHub <https://github.com/Quantomatic/pyzx>`__.
#
# .. [#JvdW2020]
#
#    John van de Wetering. "ZX-calculus for the working quantum computer scientist."
#    `ArXiv <https://arxiv.org/abs/2012.13966>`__.
#
# About the author
# ----------------
# .. include:: ../_static/authors/romain_moyard.txt
