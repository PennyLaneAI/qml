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

As a first step, a *probe state* :math:`\rho_0` is prepared. This probe state then undergoes a possibly noisy quantum
evolution that depends on a vector of parameters :math:`\boldsymbol{\phi}`. The resulting state
:math:`\rho(\boldsymbol{\phi})` is then measured using a parametrized positive
operator-valued measurement :math:`\{ \Pi_l \}`, yielding an output probability
distribution

.. math:: p_l(\boldsymbol{\phi}) =
    \operatorname{Tr}(\rho(\boldsymbol{\phi}) \Pi_l).

We now seek to estimate the vector of parameters :math:`\boldsymbol{\phi}` from this probability distribution.
Intuitively, we will get the best precision in doing so if the probe state is most "susceptible" to the
physical evolution and the corresponding measurement can distinguish the states for different parameter values well. 

Luckily, there exists a mathematical tool to quantify the best achievable esitmation precision,
the *Cramér-Rao bound*: For any unbiased estimator
:math:`\mathbb{E}(\hat{\boldsymbol{\varphi}}) = \boldsymbol{\phi}`, we have

.. math:: \operatorname{Cov}(\hat{\boldsymbol{\varphi}}) \geq \frac{1}{n} I^{-1}_{\boldsymbol{\phi}},

where :math:`n` is the number of samples and :math:`I_{\boldsymbol{\phi}}` is the *Classical Fisher Information Matrix*
with respect to the entries of :math:`\boldsymbol{\phi}`, defined as

.. math:: [I_{\boldsymbol{\phi}}]_{jk} := \sum_l \frac{(\partial_j p_l)(\partial_k p_l)}{p_l}.

The variational algorithm now proceeds by parametrizing both the probe state :math:`\rho_0 = \rho_0(\boldsymbol{\theta})`
and the POVM :math:`\Pi_l = \Pi_l(\boldsymbol{\mu})`. The parameters :math:`\boldsymbol{\theta}` and :math:`\boldsymbol{\mu}`
are then adjusted to reduce a cost function derived from the Cramér-Rao bound. Its right-hand side already gives
the best attainable precision, but is only a scalar if there is only one paramter to be estimated. To also obtain a scalar
quantity in the multi-variate case, we apply a positive-semidefinite weighting matrix :math:`W` to both side of the bound
and perform a trace, yielding the scalar inequality

.. math:: \operatorname{Tr}(W\operatorname{Cov}(\hat{\boldsymbol{\varphi}})) \geq \frac{1}{n} \operatorname{Tr}(W I^{-1}_{\boldsymbol{\phi}}).

As its name suggests, :math:`W` can be used to weight the importance of the different entries of :math:`\boldsymbol{\phi}`.
The right-hand side is now a scalar quantifying the best attainable weighted precision and can be readily used as a cost function

.. math:: C_W(\boldsymbol{\theta}, \boldsymbol{\mu}) = \operatorname{Tr}(W I^{-1}_{\boldsymbol{\phi}}(\boldsymbol{\theta}, \boldsymbol{\mu}))

Ramsay spectroscopy
------------------

As an example, we will study Ramsay spectroscopy, a widely used technique for quantum metrology with atoms and ions. 
The metrological parameters are phase shifts :math:`\boldsymbol{\phi}` arising from the interaction of probe ions 
modeled as two-level systems with an external driving force. We model the noise in the parameter encoding as local
dephasing with dephasing constant :math:`\gamma`. We consider a pure probe state on three qubits and a projective measurement, where
the computational basis is parametrized by local unitaries.

To add another interesting aspect, we will seek an optimal protocols for the estimation of the Fourier amplitudes
of the phases:

.. math:: f_j(\boldsymbol{\pphi}) = |\sum_k \phi_k \mathrm{e}^{-i j k \frac{2\pi}{N}}|^2


Code 
----

"""
import pennylane as qml
from pennylane import numpy as np

##############################################################################
# We will first specify the device to carry out the simulations. As we want to 
# model a noisy system, it needs to be capable of mixed-state simulations.
# We will choose the `cirq.mixedsimulator` device for this tutorial.
dev = qml.device("cirq.mixedsimulator", wires=3)
from pennylane_cirq import ops as cirq_ops

##############################################################################
# Next, we model the parameter encoding. Phase shifts can be recreated using
# the Pauli Z rotation gate.
@qml.template
def encoding(phi, gamma):
    for i in range(3):
        qml.RZ(phi[i], wires=[i])
        cirq_ops.PhaseDamp(gamma, wires=[i])

##############################################################################
# We now choose an ansatz for our circuit and the POVM. We make use of the 
# Arbitrary state preparation templates from pennylane.
@qml.template
def ansatz(weights):
    qml.ArbitraryStatePreparation(weights, wires=[0, 1, 2])

NUM_ANSATZ_PARAMETERS = 14

@qml.template
def measurement(weights):
    for i in range(3):
        qml.ArbitraryStatePreparation(weights[2 * i : 2 * (i + 1)], wires=[i])
        
NUM_MEASUREMENT_PARAMETERS = 6

##############################################################################
# We now have everything at hand to model the quantum part of our experiment
# as a QNode. We will return the output probabilities necessary to compute the
# Classical Fisher Information Matrix.
@qml.qnode(dev)
def experiment(weights, phi, gamma):
    ansatz(weights[ : NUM_ANSATZ_PARAMETERS])
    encoding(phi, gamma)
    measurement(weights[NUM_ANSATZ_PARAMETERS : ])

    return qml.probs(wires=[0, 1, 2])


##############################################################################
# Now, let's turn to the cost function itself. The most important ingredient
# is the Classical Fisher Information Matrix, which we compute using a separate
# function.
def CFIM(weights, phi, gamma):
    p = experiment(weights, phi, gamma)
    dp = []

    for idx in range(3):
        # We use the parameter-shift rule explicitly
        # to compute the derivatives
        shift = np.zeros_like(phi)
        shift[idx] = np.pi / 2

        plus = experiment(weights, phi + shift, gamma)
        minus = experiment(weights, phi - shift, gamma)

        dp.append(0.5 * (plus - minus))

    matrix = [0] * 9
    for i in range(3):
        for j in range(3):
            matrix[3 * i + j] = np.sum(dp[i] * dp[j] / p)

    return np.array(matrix).reshape((3, 3))

##############################################################################
# As the cost function contains an inversion, we add a small regularization
# to it to avoid inverting a singular matrix.
def cost(weights, phi, gamma, W, epsilon=1e-10):
    return np.trace(W @ np.linalg.inv(CFIM(weights, phi, gamma) + np.eye(3) * epsilon))

##############################################################################
# References
# ----------
#
# 1. Johannes Jakob Meyer, Johannes Borregaard, Jens Eisert. 
#    "A variational toolbox for quantum multi-parameter estimation." `arXiv:2006.06303
#    <https://arxiv.org/abs/2006.06303>`__, 2020.
