r"""Fixed Depth Hamiltonian Simulation via Cartan Decomposition
===============================================================

abstract

Introduction
------------


"""
import numpy as np
import pennylane as qml
from pennylane import X, Y, Z

##############################################################################
# 

##############################################################################
# 

##############################################################################
# 

##############################################################################
# 

##############################################################################
# 

##############################################################################
# 

##############################################################################
# 
#


##############################################################################
# 
# Conclusion
# ----------
#
# With this introduction, we hope to clarify some terminology, introduce the basic concepts of Lie theory and motivate their relevance in quantum physics by touching on universality and symmetries.
# While Lie theory and symmetries are playing a central role in established fields such as quantum phase transitions (see note above) and `high energy physics <https://en.wikipedia.org/wiki/Standard_Model>`_,
# they have recently also emerged in quantum machine learning with the onset of geometric quantum machine learning [#Meyer]_ [#Nguyen]_
# (see our recent :doc:`introduction to geometric quantum machine learning <tutorial_geometric_qml>`).
# Further, DLAs have recently become instrumental in classifying criteria for barren plateaus [#Fontana]_ [#Ragone]_ and designing simulators based on them [#Goh]_.
#



##############################################################################
# 
# References
# ----------
#
# .. [#Wiersma]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
#
# .. [#Meyer]
#
#     Johannes Jakob Meyer, Marian Mularski, Elies Gil-Fuster, Antonio Anna Mele, Francesco Arzani, Alissa Wilms, Jens Eisert
#     "Exploiting symmetry in variational quantum machine learning"
#     `arXiv:2205.06217 <https://arxiv.org/abs/2205.06217>`__, 2022.
#
# .. [#Nguyen]
#
#     Quynh T. Nguyen, Louis Schatzki, Paolo Braccia, Michael Ragone, Patrick J. Coles, Frederic Sauvage, Martin Larocca, M. Cerezo
#     "Theory for Equivariant Quantum Neural Networks"
#     `arXiv:2210.08566 <https://arxiv.org/abs/2210.08566>`__, 2022.
#
# .. [#Fontana]
#
#     Enrico Fontana, Dylan Herman, Shouvanik Chakrabarti, Niraj Kumar, Romina Yalovetzky, Jamie Heredge, Shree Hari Sureshbabu, Marco Pistoia
#     "The Adjoint Is All You Need: Characterizing Barren Plateaus in Quantum Ansätze"
#     `arXiv:2309.07902 <https://arxiv.org/abs/2309.07902>`__, 2023.
#
# .. [#Ragone]
#
#     Michael Ragone, Bojko N. Bakalov, Frédéric Sauvage, Alexander F. Kemper, Carlos Ortiz Marrero, Martin Larocca, M. Cerezo
#     "A Unified Theory of Barren Plateaus for Deep Parametrized Quantum Circuits"
#     `arXiv:2309.09342 <https://arxiv.org/abs/2309.09342>`__, 2023.
#
# .. [#Goh]
#
#     Matthew L. Goh, Martin Larocca, Lukasz Cincio, M. Cerezo, Frédéric Sauvage
#     "Lie-algebraic classical simulations for variational quantum computing"
#     `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__, 2023.
#
# .. [#Somma]
#
#     Rolando D. Somma
#     "Quantum Computation, Complexity, and Many-Body Physics"
#     `arXiv:quant-ph/0512209 <https://arxiv.org/abs/quant-ph/0512209>`__, 2005.
#
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
