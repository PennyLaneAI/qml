"""
Error mitigation with Mitiq and PennyLane
=========================================

.. meta::
    :property="og:description": Learn how to mitigate quantum circuits using Mitiq and PennyLane.

    :property="og:image": https://pennylane.ai/qml/_images/laptop.png

.. related::

   tutorial_vqe A brief overview of VQE
   tutorial_noisy_circuits Explore NISQ devices

*Author: Mitiq and PennyLane dev teams. Last updated: 10 November 2021*

Have you ever run a circuit on quantum hardware and not quite got the result you were expecting?
If so, welcome to the world of noisy intermediate scale quantum (NISQ) devices! These devices must
function in noisy environments and are unable to execute quantum circuits perfectly, resulting in
outputs that can have a significant error. The long-term plan of quantum computing is to develop a
subsequent generation of error-corrected hardware. In the meantime, how can we best utilize our
error-prone NISQ devices for practical tasks? One proposed solution is to adopt an approach called
error *mitigation*, which aims to minimize the effects of noise by executing a family of related
circuits and using the results to estimate an error-free value.

.. figure:: ../demonstrations/error_mitigation/laptop.png
    :align: center
    :scale: 55%
    :alt: Mitiq and PennyLane
    :target: javascript:void(0);

This demo shows how error mitigation can be carried out by combining PennyLane with the
`Mitiq <https://github.com/unitaryfund/mitiq>`__ package, a Python-based library providing a range
of error mitigation techniques. Integration with PennyLane is available from the ``0.11`` version
of Mitiq, which can be installed using

.. code-block:: bash

    pip install "mitiq>=0.11"

We'll begin the demo by jumping straight into the deep end and seeing how to mitigate a simple noisy
circuit in PennyLane with Mitiq as a backend. After, we'll take a step back and discuss the theory
behind the error mitigation approach we used, known as zero-noise extrapolation. The final part of
this demo showcases how mitigation can be applied in quantum chemistry, allowing us to more
accurately calculate the potential energy surface of molecular hydrogen.

Mitigating noise in a simple circuit
------------------------------------

We first need a noisy device to execute our circuit on. Let's keep things simple for now by loading
the :mod:`default.mixed <pennylane.devices.default_mixed>` device and artificially adding
:class:`PhaseDamping <pennylane.PhaseDamping>` noise.
"""

import pennylane as qml

noise_gate = qml.PhaseDamping
noise_strength = 0.1
n_wires = 4

dev_ideal = qml.device("default.mixed", wires=n_wires)
dev_noisy = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)

###############################################################################
# In the above, we load a noise-free device ``dev_ideal`` and a noisy device ``dev_noisy`` which
# is constructed from the :func:`qml.transforms.insert <pennylane.transforms.insert>` transform.
# This transform works by intercepting each circuit executed on the device and adding the
# :class:`PhaseDamping <pennylane.PhaseDamping>` noise channel directly after every gate in the
# circuit. To get a better understanding of noise channels like
# :class:`PhaseDamping <pennylane.PhaseDamping>`, check out the :doc:`tutorial_noisy_circuits`
# tutorial.
#
# The next step is to define our circuit. Inspired by the mirror circuits concept introduced by
# Proctor *et al.* [#proctor2020measuring]_ let's fix a circuit of the form :math:`U^{\dagger} U`
# with :math:`U` a unitary given by the :class:`SimplifiedTwoDesign <pennylane.SimplifiedTwoDesign>`
# template. We also fix a measurement of the :class:`PauliZ <pennylane.PauliZ>` observable on our
# first qubit. Importantly, such a circuit performs an identity transformation
# :math:`U^{\dagger} U = \mathbb{I}` and we can show that the expected value of an ideal circuit
# execution is
#
# .. math::
#
#     \langle 0 | U U^{\dagger} Z U^{\dagger} U | 0 \rangle = 1.
#
# Although this circuit seems trivial, it provides an ideal test case for benchmarking noisy
# devices where we expect the output to be less than one due to the detrimental effects of noise.
# Let's check this out in PennyLane code:

from pennylane import numpy as np
from pennylane.beta import QNode
np.random.seed(1967)

n_layers = 1
template = qml.SimplifiedTwoDesign
weights_shape = template.shape(n_layers, n_wires)
w1, w2 = [2 * np.pi * np.random.random(s) for s in weights_shape]


def circuit(w1, w2):
    template(w1, w2, wires=range(n_wires))
    qml.adjoint(template)(w1, w2, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))


ideal_qnode = QNode(circuit, dev_ideal)
noisy_qnode = QNode(circuit, dev_noisy)

##############################################################################
# First, we'll visualize the circuit:

print(qml.draw(ideal_qnode, expansion_strategy="device")(w1, w2))

##############################################################################
# As expected, executing the circuit on an ideal noise-free device gives a result of ``1``.

ideal_qnode(w1, w2).numpy()

##############################################################################
# On the other hand, we obtain a noisy result when running on ``dev_noisy``:

noisy_qnode(w1, w2).numpy()

##############################################################################
# So, we have set ourselves up with a benchmark circuit and seen that executing on a noisy device
# gives imperfect results. Can the results be improved? Time for error mitigation! We'll first
# show how easy it is to add error mitigation in PennyLane with Mitiq as a backend, before
# explaining what is going on behind the scenes.

from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory
from pennylane.transforms import mitigate_with_zne

extrapolate = RichardsonFactory.extrapolate
scale_factors = [1, 2, 3]

mitigated_qnode = mitigate_with_zne(scale_factors, fold_global, extrapolate)(noisy_qnode)
mitigated_qnode(w1, w2)

##############################################################################
# Amazing! Using PennyLane's :func:`mitigate_with_zne <pennylane.transforms.mitigate_with_zne>`
# transform, we can create a new ``mitigated_qnode`` whose result is closer to the ideal noise-free
# value of ``1``. How does this work?
#
# Understanding error mitigation
# ------------------------------
#
# Error mitigation can be realized through a number of techniques and the Mitiq
# `documentation <https://mitiq.readthedocs.io/en/stable/>`__ is a great resource to learn more. In
# this tutorial we focus upon the zero-noise extrapolation (ZNE) method originally introduced by
# Temme et al. [#temme2017error]_ and Li et al. [#li2017efficient]_.

##############################################################################
# .. [#proctor2020measuring] T. Proctor, K. Rudinger, K. Young, E. Nielsen, R. Blume-Kohout
#             `"Measuring the Capabilities of Quantum Computers" <https://arxiv.org/abs/2008.11294>`_,
#             arXiv:2008.11294 (2020).
#
# .. [#temme2017error] K. Temme, S. Bravyi, J. M. Gambetta
#             `"Error Mitigation for Short-Depth Quantum Circuits" <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`_,
#             Phys. Rev. Lett. 119, 180509 (2017).
#
# .. [#li2017efficient] Y. Li, S. C. Benjamin
#             `"Efficient Variational Quantum Simulator Incorporating Active Error Minimization" <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.021050>`_,
#             Phys. Rev. X 7, 021050 (2017).


