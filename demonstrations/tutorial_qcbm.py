"""
.. qcbm:

QCBM with tensor networks ansatz
===============

.. meta::
    :property="og:description": Implementing the Quantum Circuit Born Machine (QCBM) using a tensor network ansatz
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets//qaoa_maxcut_partition.png

.. related::

*Author: Ahmed Darwish and Emiliano Godinez — Posted: 08 May 2024. Last updated: 08 May 2024.*

In this tutorial we employ the NISQ-friendly generative model known as the Quantum Circuit Born Machine (QCBM) introduced by `Benedetti, Garcia-Pintos, Perdomo, Leyton-Ortega, Nam and Perdomo-Ortiz (2019) in [#Benedetti]_. to obtain the probability distribution of the bars and stripes data set. To this end, we use the tensor-network inspired templates available in Pennylane to construct the model's ansatz.
"""

##############################################################################
# Generative modeling and quantum physics
# -----------------------
# In the classical setup of unsupervised learning, a generative statistical model is tasked with creating new data instances that follow the probability distribution of the input training data. This is in contrast to other statistical models like the discriminative model, which allows us to "tell apart" the different instances present in the the input data, assigning a label to each of them. 
# 
# add image here showing an example of "generative model"
#
# An important trait of the generative model that allows for this generation of new samples is rooted on its ability to capture correlations present in the input data. Among the different models employed in the literature for generative machine learning, the so called "Born machine" model is based on representing the probability distribution :math:`p(x)` in terms of the amplitudes of the quantum wave function :math:`\ket{\psi}`, with the relation given by Born's rule
# 
# .. math::
#   p(x) = \norm*{\bra{x}\ket{\psi}}^2
#
# As done in [#Cheng]_ the efficient representation provided by tensor network ansätze invites us to represent the wave function in terms of tensor networks classes. In particular, the ubiqutious class of Matrix Product States (MPS) and Tree Tensor Networks (TTN) are capable of capturing the local correlations present in the training data, this making them suitable candidates be employed in the generative model.
# 
# add tensor network image with psi as label both for MPS and TTN.
# 
# Tensor networks and Quantum Circuits
# -----------------------
# As part of the generative model pipeline, it is necessary to draw finite samples from the resulting wave function in order to approximate the target probability distribution. While many algorithms have been proposed to sample tensor networks classes efficiently ([#Ferris]_, [#Ballarin]_), as suggested in [#Harrow]_, this task appears as a suitable candidate to attempt and achieve quantum advantage in Near Intermediate Scale Quantum (NISQ) devices. As presented in [#Benedetti]_, this approach employing quantum circuits to model the probabililty distribution is known as the Quantum Circuit Born Machine (QCBM). 
# 
# The problem formulation starts by looking at the training data set :math:`\mathcal{D} = (\mathrb{x}^{(1)}, \mathrb{x}^{(2)}, ldots, \mathrb{x}^{(D)})` made up of :math:`D` binary vectors of length :math:`N`. Each of this vectors have an associated probability of occurring within the data, resulting in the target probability distribution :math:`P_{\mathcal{D}}`. For a quantum circuit with :math:`N` qubits, this formulation gives rise to the one-to-one mapping betweeen the computational states and the input vectors
# 
# .. math::
#   \mathrb{x} := (x_1, x_2, \ldots, x_N) \leftrightarrow \ket{\mathrb{x}} := \ket{x_1, x_2, \ldots, x_N}
# 
# To approximate the target distribution, we can create an ansatz for the quantum circuit parametrized by a vector :math:`\theta`, such that the output state vector is defined as :math:`\ket{\psi(\theta)} = U(\theta) \ket{0}`.
# 
# add drawing of circuit.
# 
# Then, we can look at the probability of finding the output wave function in the computational basis state :math:`\ket{\mathrb{x}}` expressed as
# 
# .. math::
#   P_\theta(\mathrb{x}) = \norm*{ \bra{\mathrb{x}}\ket{\psi(\theta)}}^2
# 
# We can then use this quantity to define a cost function to be minimized by iteratively optimizing the parameter vector :math:`\theta` in order to obtain the wave function that best approximates the target distribution. In other words, we can formulate this problem as the minimization 
# 
# .. math::
#   min_{\theta} C(P_\theta(\mathrb{x})),
# 
# where :math:`C` is the cost function to be optimized, which can take different forms based on the specific scenario, with the common denominator being that all of them should quantify the difference between the target probability and the probability distribution obtained from the quantum circuit. Due to the nature of the finite sampling used to estimate the distribution, analogous to what is done in classsical generative machine learning [#Goodfellow]_, in this tutorial we choose the cost function to be the Kullback-Leibler (KL) divergence.
# 
# :math::
#   C(\theta) = \sum_\mathrb{x} P_D(\mathrb{x}) ln \left ( \frac{P_D(\mathrb{x})}{P_\theta(\mathrb{x})} \right)
# 
# 
# Tensor Network Ansatz
# ---------------------------
# The algorithm presented in [#Benedetti]_, proposes the use of a hardware-efficient ansatz to prepare the probability distriubtion using a  quantum circuit. However, in this work we take inspiration from previous approaches to generative modelling using tensor networks to represent the wave function, as done in [#Han]_ and [#Cheng]_ employing MPS and TTN, respectively. Since quantum circuits are a restricted class of tensor networks, there is a natural relation between them that we can exploit to define a tensor-network inspired ansatz. In particular, we take into consideration the local correlations of the data, and employ the MPS and TTN circuit ansatz implemented in Pennylane. See this tutorial `this tutorial <https://pennylane.ai/qml/demos/tutorial_tn_circuits/>`_ for a deeper study of these ansätze.
# 
#
# Pennylane implementation
# ---------------------------
# 
# References
# ^^^^^^^^^^
#
# .. [#Benedetti]
#    M. Benedetti, D. Garcia-Pintos, O. Perdomo, V. Leyton-Ortega, Y. Nam, and A. Perdomo-Ortiz,
#    *npj Quantum Information* **5**, 1 (2019), ISSN 2056-6387,
#    URL `<http://dx.doi.org/10.1038/s41534-019-0157-8>`__
#
# .. [#Cheng]
#    S. Cheng, L. Wang, T. Xiang, and P. Zhang,
#    *Physical Review B* **99**, 15 (2019), ISSN 2469-9969,
#    URL `<http://dx.doi.org/10.1103/PhysRevB.99.155131>`__

# .. [#Ferris]
#    A. J. Ferris and G. Vidal,
#    *Physical Review B* **85**, 16 (2012), ISSN 1550-235X,
#    URL `<http://dx.doi.org/10.1103/PhysRevB.85.165146>`__

# .. [#Ballarin]
#    M. Ballarin, P. Silvi, S. Montangero, and D. Jaschke,
#    "Optimal sampling of tensor networks targeting wave function's fast decaying tails," (2024),
#    arXiv:2401.10330 [quant-ph],
#    URL `<https://arxiv.org/abs/2401.10330>`__

# .. [#Harrow]
#    A. W. Harrow and A. Montanaro,
#    *Nature* **549**, 7671, 203–209 (2017), ISSN 1476-4687,
#    URL `<http://dx.doi.org/10.1038/nature23458>`__

# .. [#Goodfellow]
#    I. Goodfellow, Y. Bengio, and A. Courville,
#    *Deep Learning* (2016),
#    MIT Press,
#    URL `<http://www.deeplearningbook.org>`__

# .. [#Han]
#    Z.-Y. Han, J. Wang, H. Fan, L. Wang, and P. Zhang,
#    *Physical Review X* **8**, 3 (2018), ISSN 2160-3308,
#    URL `<http://dx.doi.org/10.1103/PhysRevX.8.031012>`__





I can then explain what are generative models generally, how born machine is one model possible, and how this has been attempted by using tensor networks. On the other hand it has also been attempted with quantum circuits, and here we combine both methods. I can say "quantum circuits are a particular instance of tensor networks, and therefore there is a natural connection" (cite the diagrammatic representation)
