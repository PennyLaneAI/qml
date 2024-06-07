r"""Intro to the BlueQubit (CPU) device 
=============================================================

Here we will show how you can run Pennylane circuits of up to 33 qubits for free on BlueQubit backend.
Running large scale simulations usually requires lots of memory and compute power.
Regular laptops already struggle above 20 qubits and most of the time 30+ qubits are no go.
Using the BlueQubit device, now Pennylane users can run circuit simulations on souped up machines and go up to 33 qubits!

.. note::

    To follow along with this tutorial on your own computer, you will need the
    `BlueQubit SDK <https://app.bluequbit.io/sdk-docs/index.html>`_. It can be installed via pip:

    .. code-block:: bash

        pip install bluequbit


.. figure:: ../_static/demonstration_assets/qft/socialthumbnail_large_QFT_2024-04-04.png
    :align: center
    :width: 60%
    :target: javascript:void(0)



Build your Pennylane circuits as always
---------------------------------------

Here we will build a simple Bell Pair and simulate it on BlueQubit backend.
Later in the tuturial we will show a larger example - a 26 qubit quantum adder.

Here is the example circuit we will be simulating:
"""

import pennylane as qml


def bell_pair():
    qml.Hadamard(0)
    qml.CNOT(wires=(0, 1))
    return qml.state()


print(qml.draw(bell_pair)())


##############################################################################
# Add the BlueQubit device
# ---------------------------------------
# Here are the 3 easy steps to simulate the above circuit on the BlueQubit backend:
#
# 1. Open an account at `app.bluequbit.io <https://app.bluequbit.io/>`__ to get a token. You can also view your submitted jobs here later.
# 2. Initialize the bluequbit device with your token.
# 3. Submit the circuit for similation!
#

import bluequbit

# TODO: using a guest token here. need to figure out a better solution.
bq_dev = qml.device("bluequbit.cpu", wires=2, token="q0RSAhGZGns0KpZ6vDZptg0SrdDqYpty")

bell_qnode = qml.QNode(bell_pair, bq_dev)

result = bell_qnode()
print(result)

#############################################
# And that's it!
#
# Now you can run even larger (up to 33 qubit!) circuit simulations the same way!
#
# Larger Example: 26 qubits
# ---------------------------------------
# Here we will see a much larger example a 26 qubit circuit

bq_dev = qml.device("bluequbit.cpu", wires=26, token="q0RSAhGZGns0KpZ6vDZptg0SrdDqYpty")


@qml.qnode(bq_dev)
def large_circuit():
    qml.X(0)
    # TODO: fill in a larger circuit
    return qml.expval(qml.PauliZ(0))


result = large_circuit()
print(result)

#############################################
# Conclusion
# ---------------------------------------
#
# In this tutorial we saw how Pennylane users can run large circuit simulations on BlueQubit's souped up machines.
#

##############################################################################
# About the author
# ----------------
#
