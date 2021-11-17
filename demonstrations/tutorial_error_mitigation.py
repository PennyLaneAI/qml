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
# Proctor *et al.* [#proctor2020measuring]_ let's fix a circuit that applies a unitary :math:`U`
# followed by its inverse :math:`U^{\dagger}`, with :math:`U` given by the
# :class:`SimplifiedTwoDesign <pennylane.SimplifiedTwoDesign>`
# template. We also fix a measurement of the :class:`PauliZ <pennylane.PauliZ>` observable on our
# first qubit. Importantly, such a circuit performs an identity transformation
# :math:`U^{\dagger} U |\psi\rangle = |\psi\rangle` to any input state :math:`|\psi\rangle` and we
# can show that the expected value of an ideal circuit execution with an input state
# :math:`|0\rangle` is
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
# `documentation <https://mitiq.readthedocs.io/en/stable/>`__ is a great resource to start learning
# more. In this tutorial we focus upon the zero-noise extrapolation (ZNE) method originally
# introduced by Temme et al. [#temme2017error]_ and Li et al. [#li2017efficient]_.
#
# The ZNE method works by assuming that the amount of noise present when a circuit is run on a
# noisy device is enumerated by a parameter :math:`\gamma`. Suppose we have an input circuit
# that experiences an amount of noise equal to :math:`\gamma = \gamma_{0}` when executed.
# Ideally, we would like to evaluate the result of the circuit in the :math:`\gamma = 0`
# noise-free setting.
#
# To do this, we create a family of equivalent circuits whose ideal noise-free value is the
# same as our input circuit. However, when run on a noisy device, each circuit experiences
# an amount of noise equal to :math:`\gamma = s \gamma_{0}` for some scale factor :math:`s`. By
# evaluating the noisy outputs of each circuit, we can extrapolate to :math:`s=0` to estimate
# the result of running a noise-free circuit.
#
# A key element of ZNE is the ability to run equivalent circuits for a range of scale factors
# :math:`s`. When the noise present in a circuit scales with the number of gates, :math:`s`
# can be varied using unitary folding [#giurgica2020digital]_.
# Unitary folding works by noticing that any unitary :math:`V` is equivalent to
# :math:`V V^{\dagger} V`. This type of transform can be applied to individual gates in the
# circuit or to the whole circuit.
# When no folding occurs, the scale factor is
# :math:`s=1` and we are running our input circuit. On the other hand, when each gate has been
# folded once, we have tripled the amount of noise in the circuit so that :math:`s=3`. For
# :math:`s \geq 3`, each gate in the circuit will be folded more than once.
#
# Let's see how
# folding works in code using Mitiq's
# `fold_global <https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.zne.scaling.folding.fold_global>`__
# function, which folds globally by setting :math:`V` to be the whole circuit.
# We begin by making a copy of our above circuit using a
# :class:`QuantumTape <pennylane.tape.QuantumTape>`, which provides a low-level approach for circuit
# construction in PennyLane.

with qml.tape.QuantumTape() as circuit:
    template(w1, w2, wires=range(n_wires))
    qml.adjoint(template)(w1, w2, wires=range(n_wires))

##############################################################################
# Don't worry, in most situations you will not need to worry about working with a PennyLane
# :class:`QuantumTape <pennylane.tape.QuantumTape>`! We are just dropping down to this
# representation to gain a greater understanding of the Mitiq integration. Let's see how folding
# works for some typical scale factors:

scale_factors = [1, 2, 3]
folded_circuits = [fold_global(circuit, scale_factor=s) for s in scale_factors]

for s, circuit in zip(scale_factors, folded_circuits):
    print(f"Globally-folded circuit with a scale factor of {s}:")
    print(circuit.draw())

##############################################################################
# Although these circuits are a bit deep, if you look carefully you might be able to convince
# yourself that they are all equivalent! In fact, since we have fixed our original circuit to be
# of the form :math:`U U^{\dagger}`, we get:
#
# - When the scale factor is :math:`s=1`, the resulting circuit is
#
#   .. math::
#
#       V = U^{\dagger} U = \mathbb{I}.
#
#   Hence, the :math:`s=1` setting gives us the original unfolded circuit.
#
# - When :math`s=3`, the resulting circuit is
#
#   .. math::
#
#       V V^{\dagger} V = U^{\dagger} U U U^{\dagger} U^{\dagger} U = \mathbb{I}.
#
#   In other words, we fold the whole circuit once when :math:`s=3`. Generally, whenever :math:`s`
#   is an odd integer, we fold :math:`(s - 1) / 2` times.
#
# - The :math:`s=2` setting is a bit more subtle. Now we apply folding only to the second half of
#   the circuit, which is in our case given by :math:`U^{\dagger}`. The resulting partially-folded
#   circuit is
#
#   .. math::
#
#       (U^{\dagger} U U^{\dagger}) U = \mathbb{I}.
#
#   Visit Ref. [#giurgica2020digital]_ to gain a deeper understanding of unitary folding.
#
# If you're still not convinced, we can evaluate the folded circuits on our noise-free device
# ``dev_ideal``. To do this, we must first transform our circuits to add the
# :class:`PauliZ <pennylane.PauliZ>` measurement on the first qubit.

folded_circuits_with_meas = []

for circuit in folded_circuits:
    with qml.tape.QuantumTape() as c:
        for op in circuit.operations:
            qml.apply(op)
        qml.expval(qml.PauliZ(0))
    folded_circuits_with_meas.append(c)

##############################################################################
# We need to do this step as part of the Mitiq integration with the low-level PennyLane
# :class:`QuantumTape <pennylane.tape.QuantumTape>`. You will not have to worry about these details
# when using the main :func:`mitigate_with_zne <pennylane.transforms.mitigate_with_zne>` function we
# encountered earlier.
#
# Now, let's execute these circuits:

qml.execute(folded_circuits_with_meas, dev_ideal, gradient_fn=None)

##############################################################################
# By construction, these circuits are equivalent to the original and have the same output value of
# :math:`1`. On the other hand, each circuit has a different depth. If we expect each gate in a
# circuit to contribute an amount of noise when running on NISQ hardware, we should expect to see
# result of the execute circuit degrade with increased depth. This can be confirmed using the
# ``dev_noisy`` device

qml.execute(folded_circuits_with_meas, dev_noisy, gradient_fn=None)

##############################################################################
# Although this degradation may seem undesirable, it is part of the standard recipe for ZNE error
# mitigation: we have a family of equivalent circuits that experience a varying amount of noise
# when executed on hardware, and we are able to control the amount of noise by varying the folding
# scale factor :math:`s` which determines the circuit depth. The final step is to extrapolate our
# results back to :math:`s=0`, providing us with an estimate of the noise-free result of the
# circuit.
#
# There are many extrapolation methods available and Mitiq provides access.

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
#
# .. [#giurgica2020digital] T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, W. J. Zeng
#             `"Digital zero noise extrapolation for quantum error mitigation" <https://ieeexplore.ieee.org/document/9259940>`_,
#             IEEE International Conference on Quantum Computing and Engineering (2020).


