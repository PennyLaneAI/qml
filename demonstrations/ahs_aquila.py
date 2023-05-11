r"""Access Rydberg atom hardware in PennyLane
==============================================

Neutral atom hardware is a new innovation in quantum technology that has been gaining traction in
recent years thanks to new developments in optical tweezer technology. One such device, QuEra’s Aquila,
is capable of running circuits with up to 256 physical qubits! The Aquila device is now accessible and
programmable via pulse programming in PennyLane and the Braket SDK plugin.
In this demo, we will learn how to define a Hamiltonian for a driven Rydberg atom system in PennyLane,
and use it first simulate a pulse program on Rydberg atoms, and then upload it and measure
the effect of Rydberg blockade on a hardware device!

|

.. figure:: ../demonstrations/ahs_aquila/aquila_demo_image.png
    :align: center
    :scale: 15%
    :alt: Illustration of robotic hand controlling Rubidium atoms with electromagnetic pulses
    :target: javascript:void(0);

|


Pulse programming basics in PennyLane
-------------------------------------

Pulse programming in PennyLane is a paradigm that looks at how control pulses interact with specific
hardware Hamiltonians. Quantum algorithms are written directly on the hardware level, and pulse
programming thus skips the abstraction of decomposing algorithms into fixed native gate sets. While
these abstractions are necessary for error correction to achieve fault tolerance in a universal
quantum computer, in noisy and intermediate-sized quantum computers, they can add unnecessary
overhead (and thereby introduce more noise).

In quantum computing architectures where qubits are realized through physical systems with discrete
energy levels, transitions from one state to another are driven by electromagnetic fields tuned to
be at or near the relevant energy gap. These electromagnetic fields can vary as a function of time.
The full system Hamiltonian is then a combination of the Hamiltonian describing the state of the
hardware when unperturbed, and a time-dependent drive.

Pulse control gives some insight into the low-level implementation of more abstract quantum
computations. In most digital quantum architectures, the native gates of the computer are, at the
implementation level, electromagnetic control pulses that have been finely tuned to perform a
particular logical gate.

This alternative approach requires a different type of control than what you might be used to in
PennyLane, where circuits are generally defined in terms of a series of gates. Specifically, pulse control
is implemented via the functionality provided in the Pennylane :mod:`~pennylane.pulse` module. For
more information on pulse programming in PennyLane, see the
`PennyLane docs <https://docs.pennylane.ai/en/stable/code/qml_pulse.html>`__, or check out the demo
about
`running a ctrl-VQE algorithm with pulse control <https://pennylane.ai/qml/demos/tutorial_pulse_programming101.html>`__.


Analog Hamiltonian Simulation
-----------------------------

Analog Hamiltonian simulation (AHS) is an alternative to the typical gate-based paradigm of quantum
computation. With analog Hamiltonian simulation, rather than implementing gates that represent a logical
abstraction, we aim to compute the behaviour of physical systems by using a programmable, controllable
device that emulates the target system’s behaviour. This allows us to investigate the behaviour of the
system of interest in different regimes or with different physical parameters, because we study the
effects in an engineered system that can be more precisely manipulated than the analogous system of interest.

This approach is in the spirit of Feynman’s original proposal for quantum computation:

   “Nature isn’t classical […] and if you want to make a simulation of Nature, you’d better make it
   quantum mechanical, and by golly it’s a wonderful problem because it doesn't look so easy. […] I
   want to talk about the possibility that there is to be an exact simulation, that the computer will
   do *exactly* the same as nature.” (emphasis in original)

   – Richard P. Feynman, International Journal of Theoretical Physics, Vol 21, Nos. 6/7, 1982

The ability to implement low-level modification of the Hamiltonian through application of a control pulses
makes pulse programming an ideal tool for implementing an AHS program, if an appropriately engineered system
is selected.

Researchers are already using AHS devices to study quantum mechanical phenomena and fundamental
physics models that are difficult to study directly or simulate on classical computers, using
realizations of the technology based on a number of physical platforms, including trapped ions,
superconducting qubits, and Rydberg atom devices (like Aquila!).

Rydberg devices are useful for a variety of tasks in simulation of complex physical systems that may
be difficult to measure directly, and have been proposed or demonstrated to have applications in
fields ranging from `condensed matter physics <https://arxiv.org/abs/1708.01044>`__,
`high-energy physics <https://arxiv.org/abs/2007.07258>`__,
and `quantum dynamics <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021070>`__, to
`quantum gravity <https://arxiv.org/abs/1911.06314>`__.

For example, recent results demonstrated using a Rydberg system to run an AHS program implementing
controlled experimental exploration of topological quantum matter by simulating the behaviour of
`quantum spin liquids <https://arxiv.org/abs/2104.04119>`__!



The QuEra Aquila device
-----------------------

The Aquila QPU works with programmable arrays of up to 256 Rb-87 atoms, trapped in vacuum by tightly
focused laser beams. These atoms can be arranged in (almost) arbitrary user-specified 1D and 2D
geometries to determine inter-qubit interactions. Different energy levels of these atoms are used to
encode qubits.

The hardware is accessible via the Braket SDK, and requires an account to access (see below). It is
available online in particular time windows, which can be found
`here <https://us-east-1.console.aws.amazon.com/braket/home?region=us-east-1#/devices/arn:aws:braket:us-east-1::device/qpu/quera/Aquila>`__
(requires AWS Braket account), though you can upload tasks to the queue at any time. Note that depending
on queue lengths, there can be some wait-time to receive results even during the availability window of
the device.

A simulated version on the Aquila hardware is also available, and is an excellent resource for
testing out programs before committing to a particular hardware task. It is important to be aware
that some things that succeed in simulation will not be able to be sent to hardware due to physical
constraints of the measurement and control setup. It is important to be aware of the hardware
specifications and capabilities when planning your pulse program. These capabilities are accessible
at any time from the hardware device; we will demonstrate in more detail where to find these
specifications and where they are relevant as we go through this demo.

.. note::

    Some cells of this notebook will only run when hardware is online. If you want to run it at
    other times to experiment with the concepts, the hardware device can be switched out with the Braket
    simulator. When interpreting the section of the demo regarding discretization for hardware, bear in
    mind that the simulator does not discretize the functions before upload, and so will not accurately
    demonstrate the discretization behaviour.

But before we get into any details, lets take a moment to get familiar with the physical system we
will be interacting with, and specifically the Hamiltonian describing it. In this treatment of the
system Hamiltonian for Rydberg atoms, we will assume that we are operating such that we only allow
access to two states; the low and high energy states are referred to as the ground and Rydberg states
respectively.

Constructing a pulse program to run on the Aquila hardware is done in two steps, each of which allows us to modify
different parts of the system Hamiltonian:

1. Define atom positions, which determines qubit connectivity
2. Specify the quantum evolution via the drive parameters

Let's start with the atom positions and the resulting Hamiltonian term describing inter-qubit interactions.


Interaction term and atom arrangement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Qubit interaction in a Rydberg atom system is mediated by a mechanism called Rydberg blockade, which arises
due to van der Waals forces between the atoms. In the Aquila system, to modify these interactions, we define
a **register** (the layout of the atoms) by providing coordinates.

If the atoms are placed at a large distance from one another, they operate as essentially independent
systems, and we can easily drive all of them to the Rydberg state simultaneously. However, as we decrease the
distance between atoms, it becomes harder and harder to excite neighboring atom pairs.

Below is a conceptual diagram demonstrating this interaction for a pair of atoms. At a distance, a similar
energetic cost is paid to from 0 to 1 excitation and from 1 to 2 excitations. However, as we move the
atoms into closer proximity, we see a rapidly increasing energy cost to drive to the doubly excited state.

|

.. figure:: ../demonstrations/ahs_aquila/rydberg_blockade_diagram.png
    :align: left
    :scale: 30%
    :alt: A diagram of the energy levels for the ground, single excitation, and double excitation states
    :target: javascript:void(0);

|

Mathematically, this interaction is described by the following Hamiltonian:

.. math:: \hat{H}_{j, k} = \sum_{j=1}^{N-1}\sum_{k=j+1}^{N} V_{jk}\hat{n}_j\hat{n}_k = \sum_{j=1}^{N-1}\sum_{k=j+1}^{N} \frac{C_6}{R^6_{jk}}\hat{n}_j\hat{n}_k

where :math:`n_j` is the number operator acting on atom *j*, :math:`R_{jk} = \lvert x_j - x_k \lvert` is the
distance between atoms *j* and *k*, and :math:`C_6` is a fixed value determined by the nature of the ground
and Rydberg states (for Aquila, :math:`5.24 \times 10^{-24} \text{rad m}^6 / \text{s}`, referring to the
:math:`\ket{70S_{1/2}} state of the :math:`^{87}Rb atom).

Here we can see the behaviour described above: the energy contribution of the interaction between each pair of
atoms is only non-zero when both atoms are in the Rydberg state, such that :math:`n_k n_j`\ket{\psi}`=1`, and is
inversely proportional to the distance. Thus, as we move two atoms closer together, it becomes increasingly
energetically expensive for both to be in the Rydberg state.

The radius within which two neighboring atoms are prevented from both being excited is referred to as the
*blockade radius* :math:`R_b`. The blockade radius is proportional to the :math:`C_6^{1/6}` value for the transition
(determining the scale of the coefficient :math:`V_{jk}`). However, it is not determined by the interaction term alone -
the blockade radius is also inversely proportional to how hard we drive the atoms.

This brings us to our discussion of the second part of the Hamiltonian: the drive term.


The driven Rydberg Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The atoms in a Rydberg system can be driven by application of a laser pulse, which can be described by 3 parameters:
amplitude (also called Rabi frequency) :math:`\Omega`, detuning :math:`\Delta`, and phase :math:`\phi`. While in
theory, a drive pulse can be applied to individual atoms, the current control setup for the Aquila hardware only
allows application of a global drive pulse.

Let’s look at how this plays out in the Hamiltonian describing a global drive targeting the ground
to Rydberg state transition. The driven Hamiltonian of the system is:

.. math::  \hat{H}_{drive} = \sum_{k=1}^N \frac{\Omega(t)}{2} (e^{i \phi(t)}\ket{g_k}\bra{r_k} - e^{-i \phi(t)} \ket{r_k}\bra{g_k}) - \Delta(t) \hat{n}_k

where :math:`\ket{r}` and :math:`\ket{g}` are the Rydberg and ground states respectively.

Now that we know a bit about the system we will be manipulating, let us look at how to connect to a
real device.



Getting started with Amazon Braket
----------------------------------

For this demo, we will integrate PennyLane with Amazon Braket to perform analog Hamiltonian
simulation on Rydberg atom based hardware provided by QuEra.

In PennyLane, Amazon Braket is accessed through the PennyLane-Braket plugin. The installation
instructions for the plugin can be found `here <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/installation.html>`__.

The remote hardware devices available on Amazon Braket can be found
`here <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`__, along with
information about about each system, including which paradigm (gate-based, continuous variable or
analog Hamiltonian simulation) it operates under. Each device has a unique identifier known as an
ARN. In PennyLane, AHS-based Braket devices are accessed through a PennyLane device named
``braket.aws.ahs``, along with specification of the corresponding ARN.

.. note::

    To access remote services on Amazon Braket, you must first
    `create an account on AWS <https://aws.amazon.com/braket/getting-started/>`__ and also follow the
    `setup instructions <https://github.com/aws/amazon-braket-sdk-python>`__ for accessing Braket from Python.

Let us access both the remote hardware device, and a local Rydberg atom simulator from AWS.

"""

import pennylane as qml

s3 = ("my-bucket", "my-prefix")
aquila = qml.device("braket.aws.ahs", 
                    device_arn="arn:aws:braket:us-east-1::device/qpu/quera/Aquila", 
                    s3_destination_folder=s3,
                    wires=3)

rydberg_simulator = qml.device("braket.local.ahs", 
                               wires=3)

######################################################################
# Creating a Rydberg Hamiltonian
# ------------------------------
#
# First we will create a :class:`~pennylane.pulse.ParametrizedHamiltonian` that describes a Rydberg system and drive we want
# to implement. Once created, this can be used with the ``default.qubit`` device to simulate the system's
# behaviour directly in PennyLane, as well as with the AWS simulator and hardware services.
# 
#
#
# Atom layout and interaction term
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Recall that placing atoms in close proximity creates a blockade effect, where it is energetically
# favourable for only one atom in each pair to be in the Rydberg state.
#
# Here we define a lattice of 3 atoms, all close enough together that we would expect only one of them
# to be excited at a time. We can see the hardware specifications for the atom lattice via:
# 

# units from the hardware backend are specified in SI units, in this case metres
aquila.hardware_capabilities['lattice'].dict()

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      {'area': {'width': Decimal('0.000075'), 'height': Decimal('0.000076')},
#       'geometry': {'spacingRadialMin': Decimal('0.000004'),
#        'spacingVerticalMin': Decimal('0.000004'),
#        'positionResolution': Decimal('1E-7'),
#        'numberSitesMax': 256}}
#
# We can see that the atom field has a width of :math:`75 \, \mu m` and a height of :math:`76 \, \mu m`.
# Additionally, we can see that the minimum radial spacing and minimal vertical spacing between two
# atoms are both :math:`4 \, \mu m`, and the resolution for atom placement is :math:`0.1 \, \mu m`. For more
# details accessing and interpreting these specifications, see Amazon Braket’s starter `Aquila example notebook
# <https://github.com/aws/amazon-braket-examples/blob/main/examples/analog_hamiltonian_simulation/01_Introduction_to_Aquila.ipynb>`__.
#
# In PennyLane, we will specify these distances in micrometres. Let's set the coordinates to be three
# points on an equilateral triangle with a side length of :math:`5 \, \mu m`, which should be well within
# the blockade radius:
# 

import numpy as np
import matplotlib.pyplot as plt

a = 5

coordinates = [(0, 0), (a, 0), (a/2, np.sqrt(a**2 - (a/2)**2))]

print(f"coordinates: {coordinates}")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      coordinates: [(0, 0), (5, 0), (2.5, 4.330127018922194)]
#

plt.scatter([x for x, y in coordinates], [y for x, y in coordinates])
plt.xlabel("μm")
plt.ylabel("μm")

##############################################################################
# .. figure:: ../demonstrations/ahs_aquila/rydberg_blockade_coordinates.png
#     :align: center
#     :scale: 50%
#     :alt: The layout of the 3 atoms defined by `coordinates`
#     :target: javascript:void(0);
#
#
# If we want to create a Hamiltonian that we can use in PennyLane to accurately simulate a system, we
# need the correct physical constants; in this case, we need an accurate value of :math:`C_6` to
# calculate the interaction term (different atoms and different sets of energy levels will have different
# physical constants). We can access these via
# 

settings = aquila.settings
settings
##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      {'interaction_coeff': 862619.7915580727}
#
#
# PennyLane provides a helper function that creates the relevant Hamiltonian,
# :func:`~pennylane.pulse.rydberg_interaction`. We pass this function the atom coordinates, along with the
# ``settings`` we retrieved above, to create the interaction term for the Hamiltonian:
# 

H_interaction = qml.pulse.rydberg_interaction(coordinates, **settings)

######################################################################
# Driving field
# ~~~~~~~~~~~~~
# 
#
# The global drive is in relation to the transition between the ground and rydberg states. It is
# defined by 3 components: the amplitude (Rabi frequency), the phase, and the detuning. Let us consider the hardware
# limitations on each of these. We can access the dictionary for hardware specifications for driving the
# Rydberg transition just as we did for the lattice specifications above:
#

aquila.hardware_capabilities['rydberg'].dict()

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      {'c6Coefficient': Decimal('5.42E-24'),
#       'rydbergGlobal': {'rabiFrequencyRange': (Decimal('0.0'),
#         Decimal('15800000.0')),
#        'rabiFrequencyResolution': Decimal('400.0'),
#        'rabiFrequencySlewRateMax': Decimal('250000000000000.0'),
#        'detuningRange': (Decimal('-125000000.0'), Decimal('125000000.0')),
#        'detuningResolution': Decimal('0.2'),
#        'detuningSlewRateMax': Decimal('2500000000000000.0'),
#        'phaseRange': (Decimal('-99.0'), Decimal('99.0')),
#        'phaseResolution': Decimal('5E-7'),
#        'timeResolution': Decimal('1E-9'),
#        'timeDeltaMin': Decimal('5E-8'),
#        'timeMin': Decimal('0.0'),
#        'timeMax': Decimal('0.000004')}}
#
# It is important to note that these quantities are in radians rather than Hz where relevant, and
# are all in SI units. This means that for amplitude and detuning, we will need to convert from angular
# frequency in rad/s to standard frequency in MHz (the expected input unit in PennyLane) to understand
# the limits on PennyLane inputs. For example, for the largest possible detuning value specified in
# PennyLane should be 19.89 MHz:

def angular_SI_to_MHz(angular_SI):
    """Converts a value in rad/s or (rad/s)/s into MHz or MHz/s"""
    return angular_SI/(2*np.pi)*1e-6

angular_SI_to_MHz(125000000.00)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      19.89436788648692
#
#
# For further details on how to access the specifications and their descriptions for the device,
# including accessing units and more detailed descriptions, see
# `AWS Aquila Notebook 01 <https://github.com/aws/amazon-braket-examples/blob/main/examples/analog_hamiltonian_simulation/01_Introduction_to_Aquila.ipynb>`__.
#
# A summary of the general hardware restrictions for Rydberg drive in PennyLane's expected units
# can be seen in the table below. Note that these can be subject to change, and for thoroughness
# it is best practice to confirm these numbers by accessing the device's ``hardware_capabilities``
# as shown above.
#
# All values in units of frequency (amplitude and detuning) are provided here in the input units
# expected by PennyLane (MHz).
#
# Note that when uploaded to hardware, the amplitude and detuning will be piecewise linear functions,
# while phase is piecewise constant. For amplitude and detuning, there is a maximum rate of change for
# the hardware output. For simulations, these numbers will be converted to angular frequency
# (multiplied by :math:`2 \pi`) internally as needed.
#
#
# .. rst-class:: docstable
#
#     +---------------+------------------+------------------+---------------+---------------+-------------------------+
#     | .. centered:: | .. centered::    | .. centered::    | .. centered:: | .. centered:: | .. centered::           |
#     |  Parameter    |  Minimum value   |   Maximum value  |   Resolution  |  PWC or PWL   |  Maximum rate of change |
#     +===============+==================+==================+===============+===============+=========================+
#     | Amplitude     | 0 MHz            |  2.51465 MHz     | 64 MHz        |  PWL          | 39788735 MHz/s          |
#     +---------------+------------------+------------------+---------------+---------------+-------------------------+
#     | Phase         | -99 rad          |  +99 rad         | 5e-7 rad      |  PWC          | N/A                     |
#     +---------------+------------------+------------------+---------------+---------------+-------------------------+
#     | Detuning      | -19.89436788 MHz | +19.89436788 MHz | 3.2e-8 MHz    |  PWL          |  39788735 MHz/s         |
#     +---------------+------------------+------------------+---------------+---------------+-------------------------+
#
# For amplitude, there is an additional restriction that the first and last set-point in the pulse must
# be 0 MHz. Phase has a similar restriction for the first set-point, though the last set-point can take
# any value in the allowed range. There are no special restriction on start and end points for detuning.
#
# A few additional limitations to be aware of are:
# 
# -  for hardware upload, the full pulse program must not exceed :math:`4 \, \mu s`
# -  the conversion from PennyLane to hardware upload will place set-points every 50ns - consider this
#    time resolution when defining pulses.
#
# Each of the 3 parameters can either be constant for the duration of the pulse, or they can be defined by a
# callable, where the callable should respect the above hardware output capabilities at all time-points.
# For an initial drive term, let's start by defining a simple pulse with a time-dependent amplitude.
# Phase and detuning will both be set to 0.
#
# For the pulse shape, we'll create a gaussian envelope. Because we also want to run the
# simulation in PennyLane, we need to define the pulse function using ``jax.numpy``.
# The time expected to be specified in microseconds for the callable.
# 

import jax.numpy as jnp

def gaussian_fn(p, t):
    return p[0] * jnp.exp(-(t-p[1])**2/(2*p[2]**2))

#Visualize pulse, time in μs

max_amplitude = 0.6
displacement = 0.9
sigma = 0.3

amplitude_params = [max_amplitude, displacement, sigma]

time = np.linspace(0, 1.75, 176)
y = [gaussian_fn(amplitude_params, t) for t in time]

plt.plot(time, y)

##############################################################################
#
# .. figure:: ../demonstrations/ahs_aquila/gaussian_fn.png
#     :align: center
#     :scale: 50%
#     :alt: Plot of the gaussian_fn as a function of time for the above parameters
#     :target: javascript:void(0);
#
#
#
# We can then define our drive using via :func:`~pennylane.pulse.rydberg_drive`:
# 

global_drive = qml.pulse.rydberg_drive(amplitude=gaussian_fn,
                                       phase=0,
                                       detuning=0,
                                       wires=[0, 1, 2])

# With only amplitude as non-zero, the overall driven Hamiltonian in this case simplifies to:
#
# .. math::  \sum_{k=1}^N \frac{\Omega(t)}{2} (\ket{g_k}\bra{r_k} - \ket{r_k}\bra{g_k}) + \sum_{j=1}^{N-1}\sum_{k=j+1}^{N} \frac{C_6}{R^6_{jk}}n_jn_k

# Now lets use our ``ParametrizedHamiltonian`` terms to run a pulse program!

######################################################################
# Simulating in PennyLane to find a pi-pulse
# ------------------------------------------
#
# A pi-pulse is any pulse calibrated to perform a 180 degree (:math:`\pi` radian) rotation on the
# Bloch Sphere that takes us from the ground state of the un-driven system to the excited state when
# applied. Here we will create one, and observe the effect of applying it with the interaction term
# “turned off”. Ignoring the inter-qubit interactions for now allows us to calibrate a pi-pulse without
# worrying about the effect of Rydberg blockade.
#
# We will implement the pi-pulse using the drive term defined above, and tune the parameters of
# the gaussian envelope to implement the desired pulse.
# 
# In the absence of the interaction term, each atom acts as a completely independent system, so
# we don't see any Rydberg blockade. Below, we’ve experimented with the parameters of
# the gaussian pulse envelope via trial-and-error to find settings that result in a pi-pulse:
# 

import jax

max_amplitude = 2.
displacement = 1.
sigma = 0.3

amplitude_params = [max_amplitude, displacement, sigma]
    
params = [amplitude_params]
ts = [0.0, 1.75]

default_qubit = qml.device("default.qubit.jax", wires=3, shots=1000)

@qml.qnode(default_qubit, interface="jax")
def circuit(parameters):
    qml.evolve(global_drive)(parameters, ts)
    return qml.counts()

circuit(params)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      {'110': 1, '111': 999}
#
#
# Simulating Rydberg blockade in PennyLane
# ----------------------------------------
#
# To simulate the effect of Rydberg blockade, we create a new circuit that includes both
# the drive and the interaction term. We can run this simulation either in PennyLane, or
# using the local simulator provided by AWS:
#


def circuit(params):
    qml.evolve(H_interaction + global_drive)(params, ts)
    return qml.counts()


circuit_qml = qml.QNode(circuit, default_qubit, interface="jax")
circuit_ahs = qml.QNode(circuit, rydberg_simulator)

print(f"PennyLane simulation: {circuit_qml(params)}")
print(f"AWS local simulation: {circuit_ahs(params)}")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      PennyLane simulation: {'000': 76, '001': 286, '010': 300, '100': 338}
#      AWS local simulation: {'000': 63, '001': 354, '010': 273, '100': 310}
#
#
# When we apply the pi-pulse, but now to the full Hamiltonian including the interaction term, we
# observe that only one of the three qubits is in the excited state. This is indeed the expected
# effect of Rydberg blockade arising from the interaction term.
#
# .. figure:: ../demonstrations/ahs_aquila/rydberg_blockade.png
#     :align: center
#     :scale: 20%
#     :alt: Illustration: three atoms trapped in optical tweezers in their Rydberg state 'clash' with one another
#     :target: javascript:void(0);
#
#
#
# Rydberg blockade on the QuEra hardware
# --------------------------------------
#
# Let’s look at how we would move this simple pulse program from local simulations to hardware.
#
# Before uploading to hardware, it’s best to consider whether there are any constraints we need to be
# aware of. Only our amplitude parameter is non-zero, so let’s review the limitations we need to
# respect for defining an amplitude on hardware:
# 
# -  All values must be within 0 to 2.51465 MHz
# -  The output will be linear between set-points, and the rate of change must never exceed 39788735 MHz/s
# -  The amplitude sequence must start and end at 0 MHz
#

times = np.linspace(0, 1.75, 1000)
amplitude = [gaussian_fn(amplitude_params, t) for t in times]

start_val = amplitude[0]
stop_val = amplitude[-1]
max_val = np.max(amplitude)
max_rate = np.max([(amplitude[i+1] - amplitude[i])/50e-9 for i in range(999)])

print(f"start value: {start_val:.3} MHz")
print(f"stop value: {stop_val:.3} MHz")
print(f"maximum value: {max_val:.3} MHz")
print(f"maximum rate of change: {max_rate:.3} MHz/s")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      start value: 0.00773 MHz
#      stop value: 0.0879 MHz
#      maximum value: 2.0 MHz
#      maximum rate of change: 1.42e+05 MHz/s
#
# Our maximum amplitude value and maximum rate of change are well below hardware limits, so the only
# constraint we need to enforce for our pulse program is ensuring the values at timestamps 0 and 1.75
# :math:` \, \mu s` are 0. For this, we can use a convenience function provided in the pulse module,
# :func:`~pennylane.pulse.rect`. We can wrap an existing function with it in order to apply a rectangular window
# within which the pulse has non-zero values.
# 
# Note that the function is non-zero outside the window, and the window is defined as including the
# end-points. This means to ensure that 0 and 1.75 return 0, they need to be outside the interval
# defining the window; we’ll use ``windows=[0.01, 1.749]``. Our modified global drive is then:
# 

amp_fn = qml.pulse.rect(gaussian_fn, windows=[0.01, 1.749])
global_drive = qml.pulse.rydberg_drive(amplitude=amp_fn,
                                       phase=0,
                                       detuning=0,
                                       wires=[0, 1, 2])

######################################################################
# We’re almost ready to run on hardware. Before we do, let’s take a look at how our the parameters
# we’ve used to define our pulse program will be converted into hardware upload data. To do this, we
# create the operator we will be using in our circuit, and pass it to a method on the hardware device
# that creates an AHS program for upload:
#
op = qml.evolve(H_interaction + global_drive)(params, ts)

ahs_program = aquila.create_ahs_program(op)

######################################################################
# On a hardware device, the ``create_ahs_program`` method will modify both the register and the pulses
# before upload. Float variables are rounded to specific, valid set-points, producing a discretized
# version of the input (for example, atom locations the register lock into grid points). For this
# pulse, we’re interested in the amplitude and the register.
# 
# For the register, recall that we defined our coordinates in micrometres as
# ``[(0, 0), (5, 0), (2.5, 4.330127018922194)]``, and that we expect the hardware upload program to be
# in SI units, i.e. micrometres have been converted to metres. We can access the
# ``ahs_program.register.coordinate_list`` to see the x and y coordinates that will be passed to
# hardware:
# 

ahs_x_coordinates = ahs_program.register.coordinate_list(0)
ahs_x_coordinates

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      [Decimal('0E-7'), Decimal('0.0000050'), Decimal('0.0000025')]

ahs_y_coordinates = ahs_program.register.coordinate_list(1)
ahs_y_coordinates

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      [Decimal('0E-7'), Decimal('0E-7'), Decimal('0.0000043')]

op_register = op.H.settings.register

op_x_coordinates = [x*1e-6 for x,_ in op_register]
op_y_coordinates = [y*1e-6 for _,y in op_register]

plt.scatter(ahs_x_coordinates, ahs_y_coordinates, label = 'AHS program')
plt.scatter(op_x_coordinates, op_y_coordinates, marker='x', label = 'Input register')
plt.xlabel("μm")
plt.ylabel("μm")
plt.legend()

##############################################################################
#
# .. figure:: ../demonstrations/ahs_aquila/rydberg_blockade_coordinates_discretized.png
#     :align: center
#     :scale: 50%
#     :alt: The input coordinates for the atom arrangement, and the shifted uploaded coordinates after discretization
#     :target: javascript:void(0);
#
#
# We can see that the final y-coordinate has been set to :math:`4.3 \, \mu m`. We’re happy with this very
# minor change due to discretization, but it's important to check - for more intricate atom layouts, small
# adjustments in coordinates could have a meaningful impact in executing the program.
#
# Let’s also look at the amplitude data. We can access the set-points for hardware upload from the program as
# ``ahs_program.hamiltonian.amplitude.time_series``, which contains both the ``times()`` and
# ``values()`` for set-points. The ``amplitude`` can be switched for ``phase`` or ``detuning`` to
# access other relevant quantities.
# 

# hardware set-points after conversion and discretization
amp_setpoints = ahs_program.hamiltonian.amplitude.time_series

# values for plotting the function defined in PennyLane for amplitude
input_times = np.linspace(*ts, 1000)
input_amplitudes = [qml.pulse.rect(gaussian_fn, windows=[0.01, 1.749])(amplitude_params, _t) for _t in np.linspace(*ts, 1000)]

# plot PL input and hardware setpoints for comparison
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(input_times, input_amplitudes)
ax1.set_xlabel('Time [$\mu s$]')
ax1.set_ylabel('MHz')
ax1.set_title('gaussian_fn')

ax2.plot(amp_setpoints.times(), amp_setpoints.values())
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('rad/s')
ax2.set_title('upload data')

plt.tight_layout()
plt.show()

##############################################################################
#
# .. figure:: ../demonstrations/ahs_aquila/gaussian_fn_vs_upload.png
#     :align: center
#     :scale: 50%
#     :alt: A plot showing the amplitude function, and the piecewise-linear approximation of it uploaded to hardware
#     :target: javascript:void(0);
#
#
# Since we are happy with this, we can send this task to hardware now. If there are any issues we’ve
# missed regarding ensuring the upload data is hardware compatible, we will be informed immediately.
# Otherwise, the task will be sent to the remote hardware; it will be run when the hardware is online, and we
# reach the front of the queue.
#
# To run this without connecting the hardware, switch the aquila device out with the ``rydberg_simulator`` below.
# Note that running on hardware is a paid service and will incur a fee.

#@qml.qnode(rydberg_simulator)
qml.qnode(aquila)
def circuit(params):
    qml.evolve(H_interaction + global_drive)(params, ts)
    return qml.counts()

circuit(params)


######################################################################
# Include result + comment on result (find AWS run and generate image!)
#
#

#
#
# Conclusion
# ----------
# 
# -  AHS is an interesting and active area of research in a different kind of quantum computation
# -  “Already today, researchers are using such devices to study quantum phenomena that otherwise
#    would be hard to simulate on classical computers.”
# -  lots of applications
# -  we made a phase transition - yay! More complex phase trasitions have been simulated... spin liquid...
#
#
# References
# ----------
#
# .. [#Cubitt]
#
#     Toby S. Cubitt, Ashley Montanaro, Stephen Piddock
#     "Universal quantum Hamiltonians"
#     `arxiv.1701.05182 <https://arxiv.org/abs/1701.05182>`__, 2018
#
# .. [#Semeghini]
#
#     G. Semeghini, H. Levine, A. Keesling, S. Ebadi, T.T. Wang, D. Bluvstein, R. Verresen, H. Pichler,
#     M. Kalinowski, R. Samajdar, A. Omran, S. Sachdev, A. Vishwanath, M. Greiner, V. Vuletic, M.D. Lukin
#     "Probing topological spin liquids on a programmable quantum simulator"
#     `arxiv.2104.04119 <https://arxiv.org/abs/2104.04119>`__, 2021.
#
# .. [#BraketDevGuide]
#
#     Amazon Web Services: Amazon Braket
#     "Hello AHS: Run your first Analog Hamiltonian Simulation"
#     `AWS Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html>`__
#
# .. [#Asthana2022]
#
#     Alexander Keesling, Eric Kessler, and Peter Komar
#     "AWS Quantum Technologies Blog: Realizing quantum spin liquid phase on an analog Hamiltonian Rydberg simulator"
#     `arXiv:2203.06818 <https://aws.amazon.com/blogs/quantum-computing/realizing-quantum-spin-liquid-phase-on-an-analog-hamiltonian-rydberg-simulator/>`__, 2021.
