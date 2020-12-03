"""
Computing gradients in parallel with Braket
===========================================

PennyLane integrates with `Amazon Braket <https://aws.amazon.com/braket/>`__ to enable quantum
machine learning and optimization on high-performance remote simulators and quantum processing
units (QPUs) through a range of `providers <https://aws.amazon.com/braket/hardware-providers/>`__.

In PennyLane, Amazon Braket is accessed through the
`PennyLane-Braket <https://amazon-braket-pennylane-plugin-python.readthedocs.io>`__ plugin. The
plugin can be installed using

.. code-block:: bash

    pip install amazon-braket-pennylane-plugin

A central feature of the Amazon Braket remote simulator is that it can execute multiple circuits
in Parallel. This capability can be harnessed in PennyLane during circuit training,
which requires lots of variations of a circuit to be executed. Hence, the PennyLane-Braket plugin
provides a method for scalable optimization of large circuits with many parameters. This tutorial
will explain the importance of this feature and allow you to benchmark it yourself.

.. figure:: ../_static/remote-multi-job-simulator.png
    :align: center
    :scale: 75%
    :alt: PennyLane can leverage Braket for parallelized gradient calculations

Why is training circuits so expensive?
--------------------------------------

Quantum-classical hybrid optimization of quantum circuits is the workhorse algorithm of near-term
quantum computing. It is not only fundamental for training variational quantum circuits but also
more broadly for applications like quantum chemistry and quantum machine learning. Today’s most
powerful optimization algorithms rely on the efficient computation of gradients—which tell us how
to adapt parameters a little bit at a time to improve the algorithm.

Calculating the gradient involves multiple device executions: for each
trainable parameter we must execute our circuit on the device typically
:doc:`more than once </glossary/parameter_shift>`. Reasonable applications involve many
trainable parameters (just think of a classical neural net with millions of tunable weights). The
result is a huge number of device executions for each optimization step.

.. figure:: ../_static/grad-circuits.png
    :align: center
    :scale: 75%
    :alt: Calculating the gradient requires multiple circuit executions

In the standard ``default.qubit`` device, gradients are calculated in PennyLane through
sequential device executions—in other words, all these circuits have to wait in the same queue
until they can be evaluated. This approach is simpler, but quickly becomes slow as we scale the
number of parameters. Moreover, as the number of qubits, or "width", of the circuit is scaled,
each device execution will slow down and eventually become a noticeable bottleneck. In
short—**the future of training quantum circuits relies on high-performance remote simulators and
hardware devices that are highly parallelized**.

Fortunately, the PennyLane-Braket plugin provides a solution for scalable quantum circuit training
by giving access to the remote Amazon Braket simulator known as
`SV1 <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`__.
The SV1 simulator is a high-performance state vector simulator that is
designed with parallel execution in mind. Together with PennyLane, we can use SV1 to run in
parallel all the circuits needed to compute a gradient!

Accessing remote devices on Amazon Braket
-----------------------------------------

The remote simulator and quantum hardware devices available on Amazon Braket can be found
`here <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`__. Each
device has a unique identifier known as an
`ARN <https://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html>`__. In PennyLane,
all remote Braket devices are accessed through a single PennyLane device named ``braket.aws.qubit``,
along with specification of the corresponding ARN.

.. note::

    To access remote services on Amazon Braket, you must first create an account on AWS and also
    follow the `setup instructions <https://github.com/aws/amazon-braket-sdk-python>`__ for
    accessing Braket from Python.

Let's load the SV1 remote simulator in PennyLane with 25 qubits. We must specify both the ARN and
the address of the `S3 bucket <https://aws.amazon.com/s3/>`__ for results to be stored:
"""

my_bucket = f"amazon-braket-Your-Bucket-Name"  # the name of the bucket
my_prefix = "Your-Folder-Name"  # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)

device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

##############################################################################
# SV1 can now be loaded with the standard PennyLane :func:`~.pennylane.device`:

import pennylane as qml
from pennylane import numpy as np

qml.enable_tape()  # Unlocks the latest features in PennyLane
wires = 25

dev_remote = qml.device(
    "braket.aws.qubit",
    device_arn=device_arn,
    wires=wires,
    s3_destination_folder=s3_folder,
    parallel=True,
)

##############################################################################
# Note the ``parallel=True`` argument. This setting allows us to unlock the power of parallel
# execution on SV1 for gradient calculations. We'll also load ``default.qubit`` for comparison.

dev_local = qml.device("default.qubit", wires=wires)

##############################################################################
# Benchmarking circuit evaluation
# -------------------------------
#
# We will now compare the execution time for the remote Braket device and ``default.qubit``. Our
# first step is to create a simple circuit:


def circuit(params):
    for i in range(wires):
        qml.RX(params[i], wires=i)
    for i in range(wires):
        qml.CNOT(wires=[i, (i + 1) % wires])
    return qml.expval(qml.PauliZ(wires - 1))


##############################################################################
#
# .. figure:: ../_static/circuit.png
#     :align: center
#     :scale: 75%
#     :alt: A simple circuit used for benchmarking
#
# In this circuit, each of the 25 qubits has a controllable rotation. A final block of two-qubit
# CNOT gates is added to entangle the qubits. Overall, this circuit has 25 trainable parameters.
# Although not particularly relevant for practical problems, we can use this circuit as a testbed
# for our comparison.
#
# The next step is to convert the above circuit into a PennyLane :func:`~.pennylane.QNode`.

qnode_remote = qml.QNode(circuit, dev_remote)
qnode_local = qml.QNode(circuit, dev_local)

##############################################################################
# .. note::
#     The above uses :func:`~.pennylane.QNode` to convert the circuit. In other tutorials,
#     you may have seen the :func:`~.pennylane.qnode` decorator being used. These approaches are
#     interchangeable, but we use :func:`~.pennylane.QNode` here because it allows us to pair the
#     same circuit to different devices.
#
# Let's now compare the execution time between the two devices:

import time

params = np.random.random(wires)

t_0_remote = time.time()
qnode_remote(params)
t_1_remote = time.time()

t_0_local = time.time()
qnode_local(params)
t_1_local = time.time()

print("Execution time on remote device (seconds):", t_1_remote - t_0_remote)
print("Execution time on local device (seconds):", t_1_local - t_0_local)

##############################################################################
#
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      Execution time on remote device (seconds): 3.5898206680030853
#      Execution time on local device (seconds): 23.50668462700196
#
# Nice! These timings highlight the advantage of using the Amazon Braket SV1 device for simulations
# with large qubit numbers. In general, simulation times scale exponentially with the number of
# qubits, but SV1 is highly optimized and running on AWS remote servers. This allows SV1 to
# outperform ``default.qubit`` in this 25-qubit example. The time you see in practice for the
# remote device will also depend on factors such as your distance to AWS servers.
#
# .. note::
#     Given these timings, why would anyone want to ``default.qubit``? You should consider
#     using local devices when your circuit has few qubits. In this regime, the latency
#     times of communicating the circuit to a remote server dominate over simulation times,
#     allowing local simulators to be faster.
#
# Benchmarking gradient calculations
# ----------------------------------
#
# Now let us compare the gradient-calculation times between the two devices. Remember that when
# loading the remote device, we set ``parallel=True``. This allows the multiple device executions
# required during gradient calculations to be performed in parallel, so we expect the
# remote device to be much faster.
#
# First, consider the remote device:

d_qnode_remote = qml.grad(qnode_remote)

t_0_remote_grad = time.time()
d_qnode_remote(params)
t_1_remote_grad = time.time()

print("Gradient calculation time on remote device (seconds):", t_1_remote_grad - t_0_remote_grad)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      Gradient calculation time on remote device (seconds): 20.92005863400118
#
# Now, the local device:
#
# .. warning::
#     Evaluating the gradient with ``default.qubit`` will take a long time, consider
#     commenting-out the following lines unless you are happy to wait.

d_qnode_local = qml.grad(qnode_local)

t_0_local_grad = time.time()
d_qnode_local(params)
t_1_local_grad = time.time()

print("Gradient calculation time on local device (seconds):", t_1_local_grad - t_0_local_grad)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      Gradient calculation time on local device (seconds): 941.8518133479993
#
# Wow, the local device needs around 15 minutes or more! Compare this to less than a minute spent
# calculating the gradient on SV1. This provides a powerful lesson in parallelization.
#
# What if we had run on SV1 with ``parallel=False``? It would have taken around 3 minutes—still
# faster than a local device, but much slower than running SV1 in parallel.
