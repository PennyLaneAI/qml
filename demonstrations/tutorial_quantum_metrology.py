r"""
Quantum Metrology
==================================

.. meta::
    :property="og:description": In this demonstration, a variational algorithm is 
        used to optimize a quantum sensing protocol.
    :property="og:image": https://pennylane.ai/qml/_images/single_shot.png

In this tutorial we use the variational quantum algorithm from
`Johannes Jakob Meyer et al. (2020) <https://arxiv.org/abs/2006.06303>`__ to optimize a quantum
sensing protocol.

Background
----------

Quantum metrology is a particular application of quantum technologies that exploit non-classical
effects to enhance the sensitivity of measurement processes. A sensing protocol can be modeled in
the following way:

As a first step, a *probe state* :math:`\rho_0(\boldsymbol{\theta})` with variational parameters
:math:`\boldsymbol{\theta}` is prepared. This probe state then undergoes a possibly noisy quantum
evolution that depends on a vector of parameters :math:`\boldsymbol{\phi}`. The resulting state
:math:`\rho(\boldsymbol{\theta}, \boldsymbol{\phi})` is then measured using a parametrized positive
operator-valued measurement :math:`\{ \Pi_l(\boldsymbol{\mu}) \}`, yielding an output probability
distribution

.. math:: p_l(\boldsymbol{\theta}, \boldsymbol{\phi}, \boldsymbol{\mu}) =
    \operatorname{Tr}(\rho(\boldsymbol{\theta}, \boldsymbol{\phi}), \Pi_l(\boldsymbol{\mu}))

We now seek to estimate the vector of parameters :math:`\boldsymbol{\phi}` or a multi-variate function
:math:`\boldsymbol{f}(\boldsymbol{\phi})` thereof from the output probability distribution. The best achievable
estimation precision in doing this can be quantified by the *Cramér-Rao bound*: For any unbiased estimator
:math:`\mathbb{E}(\hat{\boldsymbol{f}}) = \boldsymbol{f}`, we have

.. math:: \operatorname{Cov}(\hat{\boldsymbol{f}}) \geq \frac{1}{n} I^{-1}_{\boldsymbol{f}},

where :math:`n` is the number of samples and :math:`I_{\boldsymbol{f}}` is the *Classical Fisher Information Matrix*
with respect to the entries of :math:`\boldsymbol{f}`.

- The Variational Algorithm

"""


##############################################################################
# References
# ----------
#
# 1. Johannes Jakob Meyer, Johannes Borregaard, Jens Eisert. 
#    "A variational toolbox for quantum multi-parameter estimation." `arXiv:2006.06303
#    <https://arxiv.org/abs/2006.06303>`__, 2020.
