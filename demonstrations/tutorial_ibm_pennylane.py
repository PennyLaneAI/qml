r"""
Using Pennylane with IBM's quantum devices and Qiskit
===================================

.. meta::
    :property="og:description": Learn how to use IBM devices with Pennylane.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_ibm_pennylane.png

.. related::

   quantum_volume Quantum Volume
   tutorial_vqe A brief overview of VQE

*Authors: Kaur Kristjuhan, Clara Ferreira Cores, Mark Nicholas Jones; Molecular Quantum Solutions (MQS) — Posted: 29 March 2023. Last updated: 29 March 2023.*

Bigger and better quantum computers are built every year. Instead of waiting for the perfect quantum computer to be
released, we can already try out the best hardware which exists today. Experimenting on cutting-edge devices helps us
understand the technology and improve the way we develop quantum software. Pennylane is a fantastic tool for prototyping
quantum algorithms of all kinds while IBM provides access to the newest and most powerful superconducting quantum devices
available today. Let's combine the two!

In this tutorial, we'll show you how to use Pennylane to interface with IBM's quantum computing
platform. We will learn how to:

* Discover what kind of devices IBM offers
* Connect to IBM devices through Pennylane's device class
* Use Qiskit Runtime to run hybrid algorithms
* Compare different devices to improve our quantum algorithms
"""

##############################################################################
# Using IBM devices
# -----------------
# IBM offers access to a variety of devices, both classical simulators and real quantum hardware.
# By default, these devices are not included in Pennylane, but after installing the
# pennylane-qiskit package with the command ``pip install pennylane-qiskit``, they can be used just like any other device offered in Pennylane!
# Currently, there are three devices available: Aer, BasicAer and IBMQ, which can be initialized
# as follows:
import pennylane as qml
import qiskit
import os 

os.environ["IBMQX_TOKEN"]

qubits = 4
dev_aer = qml.device("qiskit.aer", wires=qubits)
dev_basicaer = qml.device("qiskit.basicaer", wires=qubits)
try:
    qiskit.IBMQ.load_account()
    dev_ibmq = qml.device("qiskit.ibmq", wires=qubits)
except Exception as e:
    print(e)

##############################################################################
# The last device on can cause an error if we don't provide a valid account
# token through qiskit. The IBMQ device is used to access quantum hardware, so it also requires access to an IBMQ
# account, which can be specified using an identifying token. You can find your token by creating
# or logging into your `IBMQ account <https://quantum-computing.ibm.com>`__. Be careful not to
# publish code which reveals your token to other people! One way to avoid this is by saving your
# token in a `Pennylane configuration file <https://docs.pennylane.ai/en/stable/introduction/configuration.html>`__.
# To specify which machine or computational framework these devices actually connect to, we can
# use the backend argument.

dev_aer = qml.device("qiskit.aer", wires=qubits, backend="aer_simulator_statevector")

##############################################################################
# For the IBMQ device, different quantum computers can be used by changing the backend to the name
# of the specific quantum computer, such as ibmq_manila or imb_nairobi. To see which
# backends exist, we can call the capabilities function:

dev_aer.capabilities()["backend"]

##############################################################################
# You can find even more details about these devices directly from the IBMQ platform. You can find
# information about the size, topology, quantum volume and noise profile of all the devices that they
# have available. Currently, the smallest device has 5 qubits and the largest has 127. On the IBMQ
# platform you can also check which devices are free to use and whether any of them are temporarily
# unavailable. You can even check your active jobs and estimated time in the queue for any programs
# you execute.

##############################################################################
# Qiskit Runtime
# ---------------
# Qiskit Runtime is a quantum computing service provided by IBM intended to make hybrid algorithms
# more efficient to execute. Hybrid algorithms are algorithms, where a classical computer and
# quantum computer work together. Often, this involves the classical algorithm iteratively
# optimizing the quantum circuit, which the quantum computer repeatedly runs.
#
# One such example is the VQE algorithm, which can be used to calculate the ground state energy of
# molecules. It contains an optimization loop, which repeatedly requests the device to run a
# parameterized quantum circuit. Because the optimization algorithm changes the values of the
# parameters, the circuit requested is different each iteration. Also, the change is dependent on
# the results of the previous circuit, which means that there needs to be constant communication
# back and forth between the quantum computer and the classical computer in charge of the
# optimization.
#
# The solution that Qiskit Runtime provides is placing a classical computer in close physical
# proximity of the quantum computer. The user uploads a job to the classical computer, which runs
# the entire hybrid algorithm together with the quantum hardware, with no intermediate user input.
# This automates the iterative process, which otherwise requires time and resources for
# communication between the user and the hardware provider.

##############################################################################
# Using Qiskit Runtime
# --------------------
# The pennylane-qiskit plugin includes some tools to help create a Qiskit Runtime job. Since using
# Qiskit Runtime only makes sense when using real quantum hardware, we must again specify our IBMQ
# account details to run these jobs.
#
# First, we set up our problem as usual, and then retrieve a program ID from IBM, which gives us a
# place to upload our job

from pennylane_qiskit import vqe_runner
from pennylane import qchem
from pennylane import numpy as np

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
basis_set = "sto-3g"
electrons = 2

H, qubits = qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    basis=basis_set,
)

try:
    dev = qml.device("qiskit.ibmq.circuit_runner", wires=4)
except Exception as e:
    print(e)

##############################################################################
# Next, we specify our quantum circuit. Although there are many circuits to choose from, it is
# important to know that before a circuit is executed on hardware, it undergoes a transpilation
# step, which converts your circuit into a different, but equivalent circuit. The purpose of this
# step is to ensure that only operations that are native to the quantum computer are used. With
# parameterized gates however, this may cause some unexpected behavior, such as the emergence of
# more parameters when the transpiler attempts to decompose a complicated gate such as
# :class:`~pennylane.AllSinglesDoubles`. These types of issues will likely be fixed in the future, but when in doubt,
# it is preferable to use simpler gates where possible. We will use a simple four qubit circuit
# with one parameter, designed specifically for the H2 molecule:


def four_qubit_ansatz(theta):
    # initial state 1100:
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)

    # change of basis
    qml.RX(np.pi / 2, wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)

    qml.CNOT(wires=[3, 2])
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[1, 0])

    qml.RZ(theta, wires=0)

    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 1])
    qml.CNOT(wires=[3, 2])

    # invert change of basis
    qml.RX(-np.pi / 2, wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)

##############################################################################
# Finally, we can run our example VQE algorithm, by using the ``vqe_runner`` function. It has many
# options, which you can specify, such as the number of shots, the maximum number of iterations
# and the initial values of the parameters.

try:
    job = vqe_runner(
        backend="ibmq_qasm_simulator",
        hamiltonian=H,
        ansatz=four_qubit_ansatz,
        x0=[0.0],
        shots=8000,
        optimizer="SPSA",
        optimizer_config={"maxiter": 30},
        kwargs={"hub": "ibm-q", "group": "open", "project": "main"},
    )
    print(job)

except Exception as e:
    print(e)

##############################################################################
# The results are saved in the job variable, in SciPy optimization format. You can also check the
# results produced by any IBM device by logging in to your IBMQ account.

##############################################################################
# Benchmarking
# ~~~~~~~~~~~~
# One of the reasons why we even want to have access to these various devices and backends is so
# that we can benchmark the capabilities of the algorithms that we develop. Some simulators are
# particularly good with certain types of circuits, whereas other simulators are more general and
# may provide resources for simulating noise which mimics the kind of errors that real quantum
# hardware produces. Switching between your devices helps you learn more about your algorithm and
# can potentially provide guidance on how to make it better. For example, we can compare the
# performance of the default pennylane simulator to the qiskit aer_simulator by running the same
# VQE algorithm on both. The difference between these two devices is that the aer_simulator uses a
# finite number of shots to estimate the energy in each iteration, rather than performing an exact
# calculation using the information hidden in the vector representation of the quantum state.

dev1 = qml.device("default.qubit", wires=4)
shots = 8000
dev2 = qml.device("qiskit.aer", wires=4, backend="aer_simulator", shots=shots)


@qml.qnode(dev1)
def cost_fn_1(theta):
    four_qubit_ansatz(theta)
    return qml.expval(H)


@qml.qnode(dev2)
def cost_fn_2(theta):
    four_qubit_ansatz(theta)
    return qml.expval(H)

# we can also use the qnode to draw the circuit
import matplotlib.pyplot as plt
qml.draw_mpl(cost_fn_1, decimals=2)(theta=1.)
plt.show()

stepsize = 0.4
max_iterations = 40
opt = qml.GradientDescentOptimizer(stepsize=stepsize)
theta_1 = np.array(0., requires_grad=True)
theta_2 = np.array(0., requires_grad=True)
energies_1 = []
energies_2 = []
for n in range(max_iterations):
    theta_1, prev_energy_1 = opt.step_and_cost(cost_fn_1, theta_1)
    theta_2, prev_energy_2 = opt.step_and_cost(cost_fn_2, theta_2)
    print(prev_energy_1, prev_energy_2)
    energies_1.append(prev_energy_1)
    energies_2.append(prev_energy_2)


##############################################################################
# We can see the difference between the two devices clearly, when we plot the energies over each
# iteration:

plt.plot(energies_1, color="r", label="default.qubit")
plt.plot(energies_2, color="b", label="qiskit.aer")

# min energy = min eigenvalue
min_energy = min(qml.eigvals(H))
z = [min_energy] * max_iterations

plt.plot(z, "--", color="k", label="Exact answer")
plt.xlabel("VQE iterations")
plt.ylabel("Energy (Ha)")
plt.legend()
plt.show()

##############################################################################
# The device with the finite number of shots is unable to converge to the right answer, because it
# is limited by the precision of the result in each iteration. This is an effect that will
# certainly appear in real quantum devices too, and it can be instructive to study this effect
# independently of all the other limitations on real devices, such as decoherence, limited
# topology and readout errors.
#
# This tutorial has demonstrated how and why to use quantum computing hardware provided by IBM using Pennylane. To read
# more about the details and possibilities of the Qiskit plugin for Pennylane, `read the documentation <https://docs.pennylane.ai/projects/qiskit/en/latest/index.html>`__
#
# About the authors
# ----------------
# .. include:: ../_static/authors/kaur_kristjuhan.txt
# .. include:: ../_static/authors/clara_ferreira_cores.txt
# .. include:: ../_static/authors/mark_nicholas_jones.txt
