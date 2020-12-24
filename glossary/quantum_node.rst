.. role:: html(raw)
   :format: html

.. _glossary_quantum_node:

Quantum Node
============

In :doc:`hybrid computation </glossary/hybrid_computation>`, algorithms
may consist of a mixture of classical and quantum components.

.. figure:: ../_static/concepts/hybrid_graph.png
    :align: center
    :width: 90%
    :target: javascript:void(0);

In PennyLane, these units of quantum computations are represented using an
object called a *quantum node*, or ``QNode``. A quantum node consists of a quantum
function (such as a :doc:`variational circuit </glossary/variational_circuit>`),
and a device on which it executes.

.. figure:: ../_static/concepts/qnode.svg
    :align: center
    :width: 80%
    :target: javascript:void(0);


.. seealso::

    In PennyLane, quantum nodes can be constructed by using either the
    ``qnode`` `decorator
    <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.qnode.html>`_,
    or the ``QNode`` `constructor <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.QNode.html>`_.
