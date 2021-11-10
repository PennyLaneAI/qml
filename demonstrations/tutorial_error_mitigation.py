"""
Error mitigation with Mitiq and PennyLane
=========================================

.. meta::
    :property="og:description": Learn how to mitigate quantum circuits using Mitiq and PennyLane.

    :property="og:image": https://pennylane.ai/qml/_images/laptop.png

*Author: Mitiq and PennyLane dev teams. Last updated: 10 November 2021*

Have you ever run a circuit on quantum hardware and not quite got the result you were expecting?
If so, welcome to the world of noisy intermediate scale quantum (NISQ) devices! These devices must
function in noisy environments and are unable to execute quantum circuits perfectly, resulting in
outputs that can have a significant error. The long-term plan of quantum computing is to develop a
subsequent generation of error-corrected hardware. In the meantime, how can we best utilize our
error-prone NISQ devices for practical tasks? One proposed solution is to adopt an approach called
error *mitigation*, which aims to minimize the effects of noise by executing a family of related
circuits and using the results to estimate an error-free value.

This demo shows how error mitigation can be carried out by combining PennyLane with the
`Mitiq <https://github.com/unitaryfund/mitiq>`__ package, a Python-based library providing a range
of error mitigation techniques. Integration with PennyLane is available from the ``0.11`` version
of Mitiq, which can be installed using

.. code-block:: bash

    pip install "mitiq>=0.11"

.. figure:: ../demonstrations/error_mitigation/laptop.png
    :align: center
    :scale: 55%
    :alt: Mitiq and PennyLane
    :target: javascript:void(0);

"""