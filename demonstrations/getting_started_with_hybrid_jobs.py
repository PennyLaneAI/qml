r"""

Getting started with Amazon Braket Hybrid Jobs
=======================================

*Author: Matthew Beach — Posted: 1 October 2023. 

This notebook provides an introduction to running hybrid quantum-classical algorithms using
PennyLane on Amazon Braket. With Amazon Braket, you gain access to both real quantum devices and
both scalable classical compute, enabling you to push the boundaries of your algorithm.

Learning outcomes
-----------------

-  Able to create a hybrid job on AWS or locally
-  Understand the hybrid jobs queue and QPU priority queuing
-  Scale up classical resources for resource-intensive workloads
-  Load and save data in a hybrid job
-  Add additional Python packages
-  Add additional source code

Amazon Braket Hybrid Jobs
=========================

Amazon Braket Hybrid Jobs offers a way for you to run hybrid quantum-classical algorithms that
require both classical resources and quantum processing units (QPUs). Hybrid Jobs is designed to
spin up the requested classical compute, run your algorithm, and release the instances after
completion so you only pay for what you use. This workflow is ideal for long-running iterative
algorithms involving both classical and quantum resources. Simply package up your code into a single
function, create a hybrid job with a single line of code, and Braket will schedule it to run as soon
as possible without interruption.

Hybrid jobs have a separate queue from quantum tasks, so once your algorithm starts running, it will
not be interrupted by variations in the quantum task queue. This helps your long-running algorithms
run efficiently and predictably. Any quantum tasks created from a running hybrid job will be run
before any other quantum tasks in the queue. This is particularly beneficial for iterative hybrid
algorithms where subsequent task depend on the outcomes of prior quantum tasks. Examples of such
algorithms include the Quantum Approximate Optimization Algorithm (QAOA), variational quantum
eigensolver, or quantum machine learning. You can also monitor your algorithm progress in near-real
time, enabling you to keep track of costs, budget, or custom metrics such as training loss or
expectation values.

Importantly, on specific QPUs, running your algorithm a hybrid job benefits from `parametric compilation <https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-parametric-compilation.html>`__. 
This reduces the overhead associated with the computationally expensive compilation step by compiling a circuit only once and not for every iteration in your hybrid algorithm. 
This dramatically reduces the total runtime for many variational algorithms.
For long running hybrid jobs, Braket automatically uses the updated calibration data from the hardware provider when compiling your circuit to ensure the highest quality results.

Getting started with PennyLane
==============================

Let’s setup an algorithm that makes use of both classical and quantum resources. We adapt the
PennyLane qubit rotation tutorial from https://pennylane.ai/qml/demos/tutorial_qubit_rotation.

.. warning::
  This demo is only compatible with Python version 3.10.

First, we import the necessary packages for the algorithm:

"""

import pennylane as qml
from pennylane import numpy as np

from braket.aws import AwsQuantumJob
from braket.jobs import log_metric, hybrid_job

from braket.devices import Devices

from braket.circuits import Circuit, Observable

######################################################################
# Next, we define a quantum simulator to run the algorithm on. In this example, we will use the Braket
# local simulator before moving onto a QPU.
#

device = qml.device("braket.local.qubit", wires=1)

######################################################################
# Now we define a circuit with two rotation gates and measure the expectation value in the
# :math:`Z`-basis.
#


@qml.qnode(device)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


######################################################################
# Finally, we create a classical-quantum loop that uses gradient descent to optimizer the parameters
# in the circuit.
#
# We add the ``log_metric`` function from Braket to record the training progress during the algorithm.
# When running on AWS, ``log_metric`` records the metrics in Amazon CloudWatch, which as accessible
# through the Braket console page or the Braket SDK. When running locally on your laptop,
# ``log_metric`` prints the iteration numbers.
#


def qubit_rotation(num_steps=10, stepsize=0.5):
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    params = np.array([0.5, 0.75])

    for i in range(num_steps):
        # update the circuit parameters
        params = opt.step(circuit, params)
        expval = circuit(params)

        log_metric(metric_name="expval", iteration_number=i, value=expval)

    return params.tolist()


######################################################################
# To run the entire algorithm, we call the qubit rotation function to see that it runs correctly.
#

qubit_rotation(5, stepsize=0.5)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#       Metrics - timestamp=1695043649.822114; expval=0.38894534132396147; iteration_number=0;
#       Metrics - timestamp=1695043649.8757634; expval=0.12290715413453956; iteration_number=1;
#       Metrics - timestamp=1695043649.9254549; expval=-0.09181374013482171; iteration_number=2;
#       Metrics - timestamp=1695043649.973501; expval=-0.2936094099948542; iteration_number=3;
#       Metrics - timestamp=1695043650.020348; expval=-0.5344079938678078; iteration_number=4;
#       [0.6767967215302757, 2.3260934173312653]

######################################################################
# Great! We see the expectation value change with each iteration number and the final parameters were
# returned as a list. Now, instead of running on our laptop, let’s submit this same function to be run
# on the AWS cloud.
#

######################################################################
# Running as a hybrid job
# -----------------------
#
# To run our algorithm for a long time, we can run it asynchronously with Amazon Braket Hybrid Jobs,
# which fully manages the classical infrastructure so you can focus on the algorithm. For example, you
# can train the a larger circuit, or increase the number of iterations. Note that each hybrid job has
# at least a one minute startup time since it creates a containerized environment on Amazon EC2. So
# for very short workloads, such as a single circuit or a batch of circuits, it may suffice for you to
# use quantum tasks.
#
# We now show how you can go from running your local Python function to running it as a hybrid job.
# Note that only Python 3.10 is supported by default. For custom environments, you can opt to use a
# custom container from Amazon Elastic Container Registry (ECR) (see `containers
# documentation <https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs-byoc.html>`__).
#
# The first step to creating a hybrid job is to annotate which function you want to run with the
# ``@hybrid_job`` decorator. Then you create the job by invoking the function as you would for normal
# Python functions. However, the decorated function returns the hybrid job handle rather than the
# result of the function. To retrieve the results after it has completed, use ``job.result()``.
#
# The required device argument in the ``@hybrid_job`` decorator specifies the QPU that the hybrid job
# will have priority access to. In this example, we set ``device=None`` since we don’t require
# priority queueing to a QPU.
#
# In the following code, we annotate the ``qubit_rotation`` function from above.
#


@hybrid_job(device=None)
def qubit_rotation_hybrid_job(num_steps=1, stepsize=0.5):
    return qubit_rotation(num_steps=num_steps, stepsize=stepsize)


######################################################################
# Now we create a hybrid job by calling the function as usual. This returns a ``AwsQuantumJob`` object
# that contains the device ARN, region, and job name.
#

job = qubit_rotation_hybrid_job(num_steps=10, stepsize=0.5)
print(job)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#       AwsQuantumJob('arn':'arn:aws:braket:<aws-region>:<account_id>:job/qubit-rotation-hybrid-job-1695044583')

######################################################################
# The hybrid job automatically captures the function arguments as hyperparameters. 
# Function arguments can be of the four built-in Python types: ``bool, int, float, str``.
# In this case, we set ``num_steps = 10`` and ``stepsize = 0.5`` as the hyperparameters.
#
# We can check the status with:
#

job.state()

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#       'QUEUED'

######################################################################
# One the hybrid job starts, it will change the status to ``RUNNING``. We can also check the hybrid
# job status in the Braket console.
#
# Once the hybrid job is complete, we can get the results with ``job.result()``. For this example, it
# should take approximate 2 minutes.
#

job.result()

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#       {'result': [0.036420360224358496, 3.1008192940506736]}

######################################################################
# Any objects in the return statement are automatically captured by Braket. Note that the objects
# returned by the function must be a tuple with each element being serializable as text. For common
# libraries such as numpy, use the ``.tolist()`` method to create a Python list object that are
# serializable.
#
# Additionally, we can plot the metrics recording during the algorithm. Below we show the expectation
# value decreases with each iteration as expected.
#

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(job.metrics())
df.sort_values(by=["iteration_number"], inplace=True)

plt.plot(df["iteration_number"], df["expval"], "-o", color="orange")
plt.xlabel("iteration_number")
plt.ylabel("expval")
plt.title("Simulator results")

plt.show()

######################################################################
#
# .. figure:: /demonstrations/getting_started_with_hybrid_jobs/simulator.png
#     :align: center
#     :width: 75%
#     :alt: Expectation value per iteration number on QPU.
#     :target: javascript:void(0);


######################################################################
# Running on a QPU
# ----------------
#
# The next step is to run this on a real QPU to see how well the simple qubit rotation works. We
# create a hybrid job with the Rigetti devices as the priority QPU. We also increase the number of
# steps to 10.
#
# Using hybrid jobs for iterative algorithms is very benefit because you retain priority access to the
# target QPU. So once your quantum tasks are created in the hybrid job, they run ahead of other tasks
# waiting in the regular quantum task queue. This is because hybrid jobs has a separate queue from
# standalone tasks so that only a single hybrid job can run on a QPU at a time. This means your
# algorithm will not be interrupted by other quantum tasks, so it will run more efficiently and
# predictably. However, hybrid jobs has a separate queue from standalone tasks so that only a single
# hybrid job can run on a QPU at a time. So for a single quantum circuit, or a batch of circuit, it’s
# recommended to create quantum tasks instead of hybrid jobs.
#
# To get QPU priority, you must ensure that the device ARN used within the function matches that
# specified in the decorator. For convenience, you can use the helper function ``get_device_arn()`` to
# automatically capture the device ARN declared in ``@hybrid_job``.
#
# In the previous example, we declared the local simulator outside the decorated function scope.
# However, for AWS devices such as QPUs or on-demand simulators, the device must be declared within
# the function scope. This is avoid unintentional sharing of AWS credentials.
#
# .. note::
#   AWS devices must be declared within the body of the decorated function.
#

device_arn = Devices.Rigetti.AspenM3.value


@hybrid_job(device=device_arn)
def qpu_qubit_rotation_hybrid_job(num_steps=10, stepsize=0.5):
    # AWS devices must be declared within the decorated function.
    device = qml.device(
        "braket.aws.qubit",
        device_arn=device_arn,  # Make sure the device ARN matches the hybrid job device ARN
        wires=2,
        shots=1_000,
    )

    @qml.qnode(device)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    params = np.array([0.5, 0.75])

    for i in range(num_steps):
        # update the circuit parameters
        params = opt.step(circuit, params)
        expval = circuit(params)

        log_metric(metric_name="expval", iteration_number=i, value=expval)

    return params.tolist()


######################################################################
# To get a sense of how long we will wait before the hybrid job runs, we can check the hybrid job
# queue depth with ``AwsDevice(device_arn).queue_depth().job``. We can also check if the device is
# currently available with ``AwsDevice(device_arn).is_available()``.
#
# When there are no other hybrid jobs in the queue ahead of you, and the device is available, the
# hybrid job will start running.
#
# .. warning:: 
#
#    Running the following cell will only run once the QPU is available. This may take a long
#    time and will result in usage fees charged to your AWS account. Only uncomment the cell if you
#    are comfortable with the potential wait-time and costs. We recommend monitoring the Billing &
#    Cost Management Dashboard on the AWS console. .
#

qpu_job = qpu_qubit_rotation_hybrid_job(num_steps=10, stepsize=0.5)
print(qpu_job)

qpu_job.result()

######################################################################
# Next, we plot the expectation value per iteration number below. We see that on a real QPU, the data
# is a lot noisier than our ideal simulator.
#

df = pd.DataFrame(qpu_job.metrics())
df.sort_values(by=["iteration_number"], inplace=True)

plt.plot(df["iteration_number"], df["expval"], "-o", color="teal")
plt.xlabel("iteration_number")
plt.ylabel("expval")
plt.title("QPU results")
plt.show()

######################################################################
#
# .. figure:: /demonstrations/getting_started_with_hybrid_jobs/qpu.png
#     :align: center
#     :width: 75%
#     :alt: Expectation value per iteration number on QPU.
#     :target: javascript:void(0);


######################################################################
# Summary
# =======
#
# In this notebook, we showed how to migrate from local Python functions to algorithms running on AWS.
# We adapted the simple example of rotating a qubit using gradient descent, running this on both a
# local simulator and a real QPU. It was beneficial to run as a hybrid jobs so that we offload all
# classical compute onto AWS EC2, and retain priority queueing for our iterative algorithm.
#
# In the next tutorial, we will cover uploading additional training data and additional Python code.
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/matthew_beach.txt
