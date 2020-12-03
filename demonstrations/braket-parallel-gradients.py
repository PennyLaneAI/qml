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
will explain the importance of this feature, allow you to benchmark it yourself, and explore its
use for solving a scaled-up graph problem with QAOA.

.. figure:: ../_static/remote-multi-job-simulator.png
    :align: center
    :scale: 75%
    :alt: PennyLane can leverage Braket for parallelized gradient calculations
    :target: javascript:void(0);

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
    :target: javascript:void(0);

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
SV1 is a high-performance state vector simulator that is
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
    follow the `setup instructions
    <https://github.com/aws/amazon-braket-sdk-python#prerequisites>`__ for accessing Braket from
    Python.

Let's load the SV1 remote simulator in PennyLane with 25 qubits. We must specify both the ARN and
the address of the `S3 bucket <https://aws.amazon.com/s3/>`__ where results are to be stored:
"""

my_bucket = "amazon-braket-Your-Bucket-Name"  # the name of the bucket
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
# We will now compare the execution time for the remote Braket SV1 device and ``default.qubit``. Our
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
#     :target: javascript:void(0);
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
# .. warning::
#     Running the contents of this notebook will result in simulation fees charged to your
#     AWS account. We recommend monitoring your usage on the AWS dashboard.
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
#     Given these timings, why would anyone want to use ``default.qubit``? You should consider
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
#
# Scaling up QAOA for larger graphs
# ---------------------------------
#
# The quantum approximate optimization algorithm (QAOA) is a candidate algorithm for near-term
# quantum hardware that can find approximate solutions to combinatorial optimization
# problems such as graph-based problems. We have seen in the main
# :doc:`QAOA tutorial<tutorial_qaoa_intro>` how QAOA successfully solves the minimum vertex
# cover problem on a four-node graph.
#
# Here, let's be ambitious and try to solve the maximum cut problem on a twenty-node graph! In
# maximum cut, the objective is to partition the graph's nodes into two groups so that the
# greatest number of edges are shared between the groups (see the diagram below). This problem is
# NP-hard, so we expect it to be tough as we increase the number of graph nodes.
#
# .. figure:: ../_static/max-cut.png
#     :align: center
#     :scale: 100%
#     :alt: The maximum cut problem
#     :target: javascript:void(0);
#
# Let's first set the graph:

import networkx as nx

nodes = wires = 20
edges = 60
seed = 1967

g = nx.gnm_random_graph(nodes, edges, seed=seed)
positions = nx.spring_layout(g, seed=seed)

nx.draw(g, with_labels=True, pos=positions)

##############################################################################
# .. figure:: ../_static/20_node_graph.png
#     :align: center
#     :scale: 100%
#     :target: javascript:void(0);
#
# We will use the remote SV1 device to help us optimize our QAOA circuit as quickly as possible.
# First, the device is loaded again for 20 qubits

dev = qml.device(
    "braket.aws.qubit",
    device_arn=device_arn,
    wires=wires,
    s3_destination_folder=s3_folder,
    parallel=True,
    max_parallel=40,
    poll_timeout_seconds=30,
)

##############################################################################
# Note the specification of ``max_parallel=40``. This means that up to ``40`` circuits will be
# executed in parallel on SV1 (the default value is ``10``).
#
# .. warning::
#     Increasing the maximum number of parallel executions can result in a greater rate of
#     spending on simulation fees on Amazon Braket. The value must also be set bearing in mind your
#     service
#     `quota <https://docs.aws.amazon.com/braket/latest/developerguide/braket-quotas.html>`__.
#
# The QAOA problem can then be set up following the standard pattern, as discussed in detail in
# the :doc:`QAOA tutorial<tutorial_qaoa_intro>`.

cost_h, mixer_h = qml.qaoa.maxcut(g)
cost_h = cost_h + qml.Hamiltonian([edges / 2], [qml.Identity(0)])
n_layers = 2


def qaoa_layer(gamma, alpha):
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(alpha, mixer_h)


def circuit(params, **kwargs):
    for i in range(wires):  # Prepare an equal superposition over all qubits
        qml.Hadamard(wires=i)

    qml.layer(qaoa_layer, n_layers, params[0], params[1])


cost_function = qml.ExpvalCost(circuit, cost_h, dev, optimize=True)
optimizer = qml.AdagradOptimizer(stepsize=0.1)

##############################################################################
# We're now set up to train the circuit!
#
# .. note::
#     The time to complete each iteration will depend on factors such as your distance to AWS
#     servers.

import time

np.random.seed(1967)
params = 0.01 * np.random.uniform(size=[2, n_layers])
iterations = 30

for i in range(iterations):
    t0 = time.time()

    params, cost_before = optimizer.step_and_cost(cost_function, params)

    t1 = time.time()

    if i == 0:
        print("Initial cost:", cost_before)
    else:
        print(f"Cost at step {i}:", cost_before)

    print("----------------------------------------------")
    print(f"Completed iteration {i + 1}")
    print(f"Time to complete iteration: {t1 - t0} seconds")

print(f"Cost at step {iterations}:", cost_function(params))

np.save("params.npy", params)
print("Parameters saved to params.npy")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Initial cost: 0.014297659040095268
#     ----------------------------------------------
#     Completed iteration 1
#     Time to complete iteration: 63.39044307900076 seconds
#     Cost at step 1: 2.8459282313672807
#     ----------------------------------------------
#     Completed iteration 2
#     Time to complete iteration: 63.58402075000049 seconds
#     Cost at step 2: 0.012737699936959238
#     ----------------------------------------------
#     Completed iteration 3
#     Time to complete iteration: 63.990285107998716 seconds
#     Cost at step 3: 0.000836846399183608
#     ----------------------------------------------
#     Completed iteration 4
#     Time to complete iteration: 67.2918808200011 seconds
#     Cost at step 4: -0.002158646044720085
#     ----------------------------------------------
#     Completed iteration 5
#     Time to complete iteration: 63.04679181599931 seconds
#     Cost at step 5: -0.012058444012211175
#     ----------------------------------------------
#     Completed iteration 6
#     Time to complete iteration: 65.0637985909998 seconds
#     Cost at step 6: -0.0637097126129031
#     ----------------------------------------------
#     Completed iteration 7
#     Time to complete iteration: 68.77382928400039 seconds
#     Cost at step 7: -0.32522304705382693
#     ----------------------------------------------
#     Completed iteration 8
#     Time to complete iteration: 64.11835629199959 seconds
#     Cost at step 8: -1.4110303319771835
#     ----------------------------------------------
#     Completed iteration 9
#     Time to complete iteration: 63.46840504300053 seconds
#     Cost at step 9: -3.871539656168175
#     ----------------------------------------------
#     Completed iteration 10
#     Time to complete iteration: 62.60511550100091 seconds
#     Cost at step 10: -6.054248744387342
#     ----------------------------------------------
#     Completed iteration 11
#     Time to complete iteration: 63.309116153001014 seconds
#     Cost at step 11: -6.999471198173209
#     ----------------------------------------------
#     Completed iteration 12
#     Time to complete iteration: 63.393460999001036 seconds
#     Cost at step 12: -7.411425766975379
#     ----------------------------------------------
#     Completed iteration 13
#     Time to complete iteration: 63.335060223000255 seconds
#     Cost at step 13: -7.67034923221981
#     ----------------------------------------------
#     Completed iteration 14
#     Time to complete iteration: 60.62144135000017 seconds
#     Cost at step 14: -7.893530730751327
#     ----------------------------------------------
#     Completed iteration 15
#     Time to complete iteration: 61.422398073000295 seconds
#     Cost at step 15: -8.11848197411851
#     ----------------------------------------------
#     Completed iteration 16
#     Time to complete iteration: 65.89559789499981 seconds
#     Cost at step 16: -8.35429067698969
#     ----------------------------------------------
#     Completed iteration 17
#     Time to complete iteration: 64.06827740299923 seconds
#     Cost at step 17: -8.596594085415477
#     ----------------------------------------------
#     Completed iteration 18
#     Time to complete iteration: 61.12303119799981 seconds
#     Cost at step 18: -8.832368276651332
#     ----------------------------------------------
#     Completed iteration 19
#     Time to complete iteration: 64.6255340939988 seconds
#     Cost at step 19: -9.043629677360917
#     ----------------------------------------------
#     Completed iteration 20
#     Time to complete iteration: 61.8558455880011 seconds
#     Cost at step 20: -9.213663732533174
#     ----------------------------------------------
#     Completed iteration 21
#     Time to complete iteration: 64.9965052819989 seconds
#     Cost at step 21: -9.334474719884758
#     ----------------------------------------------
#     Completed iteration 22
#     Time to complete iteration: 62.273182810000435 seconds
#     Cost at step 22: -9.409991169684407
#     ----------------------------------------------
#     Completed iteration 23
#     Time to complete iteration: 64.10957817500093 seconds
#     Cost at step 23: -9.452060650251452
#     ----------------------------------------------
#     Completed iteration 24
#     Time to complete iteration: 64.19466979000026 seconds
#     Cost at step 24: -9.473469809098832
#     ----------------------------------------------
#     Completed iteration 25
#     Time to complete iteration: 64.16464572800032 seconds
#     Cost at step 25: -9.483700288478586
#     ----------------------------------------------
#     Completed iteration 26
#     Time to complete iteration: 63.894581133999964 seconds
#     Cost at step 26: -9.488398771679048
#     ----------------------------------------------
#     Completed iteration 27
#     Time to complete iteration: 63.317793481999615 seconds
#     Cost at step 27: -9.490507105884868
#     ----------------------------------------------
#     Completed iteration 28
#     Time to complete iteration: 63.84673080999892 seconds
#     Cost at step 28: -9.491441272863154
#     ----------------------------------------------
#     Completed iteration 29
#     Time to complete iteration: 63.57664666900018 seconds
#     Cost at step 29: -9.491852670484572
#     ----------------------------------------------
#     Completed iteration 30
#     Time to complete iteration: 63.12685334999878 seconds
#     Cost at step 30: -9.492033508244045
#     Parameters saved to params.npy
#
# This example shows us that a 20-qubit QAOA problem can be trained in a reasonable timeframe by
# using the parallelized capabilities of the Amazon Braket SV1 device to speed up gradient
# calculations. If this problem were run on ``default.qubit`` without parallelization,
# we would expect for training to take much longer.
#
# The results of this optimization can be investigated by saving the parameters
# :download:`here </demonstrations/braket/params.npy>` to your working directory. See if you can
# analyze the performance of this optimized circuit following a similar strategy to the
# :doc:`QAOA tutorial<tutorial_qaoa_intro>`. Did we find a large graph cut?
