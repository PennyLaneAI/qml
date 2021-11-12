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

Mitigating a simple circuit
---------------------------

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
# circuit.
