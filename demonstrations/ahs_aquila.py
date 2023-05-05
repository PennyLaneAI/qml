r"""Analog Hamiltonian simulation with Rydberg atom hardware in PennyLane
=====================================================================
"""

######################################################################
# Neutral atom hardware is a new innovation in quantum technology that has been gaining traction in
# the private sector in recent years thanks to new developments in optical tweezer technology. One
# such device, QuEra’s Aquila, is capable of running circuits with up to 256 physical qubits! The
# Aquila device is now accessible and programmable via pulse programming in PennyLane and the Braket
# SDK plugin. In this demo, we will simulate a simple quantum phase transition on the Aquila hardware
# using analog Hamiltonian simulation, an alternative to gate-based quantum computing.
# 

######################################################################
# Analog Hamiltonian Simulation
# -----------------------------
# 
# Analog Hamiltonian simulation (AHS) is an alternative to the typical gate-based paradigm of quantum
# computation. With analog Hamiltonian simulation, we aim to compute the behaviour of physical systems
# by using a programmable, controllable device that emulates the target system’s behaviour. This
# allows us to investigate the behaviour of the system of interest in different regimes or with
# different physical parameters, because we study any effects in an engineered system that can be more
# precisely manipulated than the analogous system of interest.
# 
# This approach is in the spirit of Feynman’s original proposal for quantum computation:
# 
# ::
# 
#    “Nature isn’t classical […] and if you want to make a simulation of Nature, you’d better make it quantum mechanical, and by golly it’s a wonderful problem because it doesn’t look so easy. […] I want to talk about the possibility that there is to be an exact simulation, that the computer will do _exactly_ the same as nature.” (emphasis in original)
#    – Richard P. Feynman, International Journal of Theoretical Physics, Vol 21, Nos. 6/7, 1982
# 
# Researchers are already using AHS devices to study quantum mechanical phenomena and fundamental
# physics models that are difficult to study directly or simulate on classical computers, using
# realizations of the technology based on a number of physical platforms, including trapped ions,
# superconducting qubits, and Rydberg atom devices (like Aquila!).
# 
# Rydberg devices are useful for a variety of tasks in simulation of complex physical systems that may
# be difficult to measure directly, and have been proposed or demonstrated to have applications in
# fields ranging from condensed matter physics [https://arxiv.org/abs/1708.01044], high-energy physics
# [https://arxiv.org/abs/2007.07258], and quantum dynamics
# [https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021070], to quantum gravity
# [https://arxiv.org/abs/1911.06314].
# 
# For example, recent results demonstrated using a Rydberg system to run an AHS program implementing
# controlled experimental exploration of topological quantum matter by simulating the behaviour of
# quantum spin liquids [https://arxiv.org/pdf/2104.04119.pdf]!
# 

######################################################################
# Pulse programming basics in PennyLane
# -------------------------------------
# 
# Pulse programming in PennyLane is a paradigm that looks at how control pulses interact with specific
# hardware hamiltonians. Quantum algorithms are written directly on the hardware level, and pulse
# programming thus skips the abstraction of decomposing algorithms into fixed native gate sets. While
# these abstractions are necesary for error correction to achieve falut tolerance in a universal
# quantum comptuer, in noisy and intermediate-sized quantum computers, they can add unneccessary
# overhead (and thereby introduce more noise) without providing computational advantages.
# 
# In quantum computing architectures where qubits are realized through physical systems with discrete
# energy levels, transitions from one state to another are driven by electromagnetic fields tuned to
# be at or near the relevant energy gap. These electromagnetic fields can vary as a function of time.
# The full system Hamiltonian is then a combination of the Hamiltonian describing the state of the
# hardware when unperturbed, and the Hamiltonian describing a time-dependent drive.
# 
# This low-level modification of the Hamiltonian through application of a control pulse makes pulse
# programming an ideal tool for implementing an AHS program.
# 
# Pulse control also gives some insight into the low-level implementation of more abstract quantum
# computations. In most digital quantum architectures, the native gates of the computer are, at the
# implementation level, electromagnetic control pulses that have been finely tuned to perform a
# particular logical gate.
# 
# This alternative approach requires a different type of control than what you might be used to in
# PennyLane, where cirucits are generally defined in terms of a series of gates; specifically, control
# of AHS devices is implemented via the functionality provided in the Pennylane ``pulse`` module. For
# more information on pulse programming in PennyLane, see the
# ``PennyLane docs [https://docs.pennylane.ai/en/stable/code/qml_pulse.html]``, or check out the demo
# about
# ``running a ctrl-VQE algorithm with pulse control [https://pennylane.ai/qml/demos/tutorial_pulse_programming101.html]``.
# 

######################################################################
# The QuEra Aquila device
# -----------------------
# 
# The Aquila QPU works with programmable arrays of up to 256 Rb-87 atoms, trapped in vacuum by tightly
# focused laser beams. These atoms can be arranged in (almost) arbitrary user-specified 1D and 2D
# geometries to determine inter-qubit interactions. Different energy levels of these atoms are used to
# encode qubits.
# 
# The hardware is accessible via the Braket SDK, and requires an account to access (see below). It is
# available online in particular windows, which can be found HERE LINK GOES HERE!!!, though you can
# upload tasks to the queue at any time. Note that depending on queue lengths, there can be some
# wait-time to recieve results even during the availability window of the device.
# 
# A simulated version on the Aquila hardware is also available, and is an excellent resource for
# testing out programs before committing to a particular hardware task. It is important to be aware
# that some things that succeed in simulation will not be able to be sent to hardware due to physical
# constraints of the measurement and control setup. Its important to be aware of the hardware
# specifications and capabilities when planning your pulse program. These capabilities are accessible
# at any time from the hardware device; we will demonstrate in more detail where to find these
# specifications and where they are relevant as we go through this demo.
# 
# But before we get into any details, lets take a moment to get familiar with the physical system we
# will be interacting with:
# 
# ..note:: Some cells of this notebook will only run when hardware is online. If you want to run it at
# other times to experiment with the concepts, the hardware device can be switched out with the Braket
# simulator. When interpreting the section of the demo regarding discretization for hardware, bear in
# mind that the simulator does not discretize the functions before upload, and so will not accurately
# demonstrate the discretization behaviour.
# 

######################################################################
# The Rydberg atom Hamiltonian
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Basics (ground, rydberg, optical tweezers, modifiable atom arrangement)
# 
# Encoding a problem in the Aquila hardware is done in two steps:
# 
# 1. Define atom positions, which determines qubit connectivity
# 2. Specify the quantum evolution via the drive parameters
# 
# Currently accessible features:
# 
# -  A **register** (the layout of the atoms) can be specified by providing coordinates. The sets the
#    :math:`H_{vd}` term of the Hamiltonian.
# -  A global **drive field** can be implemented by passing a set of points for **amplitude**,
#    **phase** and **detuning**
# 
# Let’s look at how this plays out in the Hamiltonian describing a global drive targeting the ground
# to Rydberg state transition: Hamiltonian (targeting a single transition with a global drive)
# 
# Does braket notation work in the LaTeX rendering on the website? Does not work in Jupyter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# .. math::  \sum_{k=1}^N \frac{\Omega(t)}{2} (e^{i \phi(t)}\ket{g_k}\bra{r_k} - e^{-i \phi(t)} \ket{r_k}\bra{g_k}) - \Delta(t) \sigma_q^z
# 
# .. math:: H_{vdW, j, k} = \sum_{j=1}^{N-1}\sum_{k=j+1}^{N} V_{jk}n_jn_k = \sum_{j=1}^{N-1}\sum_{k=j+1}^{N} \frac{C_6}{R^6_{jk}}n_jn_k; \quad R_{jk} = \lvert x_j - x_k \lvert
# 
# (:math:`C_6` is a fixed value determined by the nature of the ground and Rydberg states, for QuEra
# device 5.24e-24)
# 
# -  Effect of detuning
# -  Effect of amplitude
# -  Effect of phase
# -  Effect of proximity (interaction term/connectivity, Rydberg blockade): the interactions between
#    each pair of atoms, based on their mutual van der Waals interaction, is only non-zero when both
#    atoms are in the Rydberg state, such that $n_k n_j:raw-latex:`\ket{\psi}`=1, and is inversely
#    proportional to the distance. Thus, as we move two atoms closer together, it becomes increasingly
#    energetically expensive for both to be in the Rydberg state.
# 
# Now that we know a bit about the system we will be manipulating, lets look at how to connect to a
# real device.
# 

######################################################################
# Getting started with Amazon Braket
# ----------------------------------
#
# For this demo, we will integrate PennyLane with Amazon Braket to perform Analogue Hamiltonian
# Simulation on Rydberg atom based hardware provided by QuEra.
# 
# In PennyLane, Amazon Braket is accessed through the PennyLane-Braket plugin. The plugin can be
# installed using
# 
# ``pip install amazon-braket-pennylane-plugin``
# 

######################################################################
# The remote hardware devices available on Amazon Braket can be found here
# (https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html), along with
# information about about each system, including which paradigm (gate-based, continuous variable or
# analog Hamiltonian simulation) it operates under. Each device has a unique identifier known as an
# ARN. In PennyLane, AHS-based Braket devices are accessed through a PennyLane device named
# ``braket.aws.ahs``, along with specification of the corresponding ARN.
# 
# ..note:: To access remote services on Amazon Braket, you must first create an account on AWS and
# also follow the setup instructions for accessing Braket from Python.
# 

######################################################################
# Let’s access both the remote hardware device, and a local Rydberg atom simulator from AWS.
# 

import pennylane as qml

s3 = ("my-bucket", "my-prefix")
aquila = qml.device("braket.aws.ahs", 
                    device_arn="arn:aws:braket:us-east-1::device/qpu/quera/Aquila", 
                    s3_destination_folder=s3,
                    wires=3)

rydberg_simulator = qml.device("braket.local.ahs", 
                               wires=3)

# add aquila.lattice_specs
# add aquila.global_drive_specs
# fix aquila.hardware_capabilities so it sa dict all the way down - do aquila._device.properties.paradigm.dict() instead of dict(capabilities)

dict(dict(aquila.hardware_capabilities['rydberg'])['rydbergGlobal'])

######################################################################
# Creating a Rydberg Hamiltonian
# ------------------------------
# 

######################################################################
# does this go here, or above? or both places, said two different ways
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 
# Encoding a problem in the Aquila hardware is done in two steps:
# 
# 1. Define atom positions, which determines qubit connectivity
# 2. Specify the quantum evolution via the drive parameters
# 

######################################################################
# Here we will create a ``ParametrizedHamiltonian`` that describes a rydberg system. This can be used
# with the ``default.qubit`` device to simulate behavour in PennyLane, as well as with the AWS
# simulator and hardware services.
# 

######################################################################
# Atom layout and interaction term
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

######################################################################
# Recall that placing atoms in close proximity creates a blockade effect, where it is energetically
# favourable for only one atom in each pair to be in the Rydberg state.
# 
# PennyLane provides a helper function that creates the relevant hamiltonian,
# ``qml.pulse.rydberg_interaction``. We will need to provide this function with atom coordinates.
# Optionally we can also specify ``wires`` to provide custom wire labels (the coordinates will
# otherwise be assigned wires sequentially), and an ``interaction_coeff``, :math:`C6`, to mimic a
# particular physical system (different atoms and different sets of energy levels will have different
# physical constants).
# 

######################################################################
# :math:`H_{vdW, j, k} = V_{jk}n_jn_k = \frac{C_6}{R^6_{jk}}n_jn_k; \quad R_{jk} = \lvert x_j - x_k \lvert`
# 

######################################################################
# Here we create a lattice of 3 atoms, all close enough together that we would expect only one of them
# to be excited at a time. We can see the hardware specifications for the atom lattice via:
# 

# units from the hardware backend are specified in SI units, in this case meters
dict(aquila.hardware_capabilities['lattice'])

######################################################################
# We can see that the atom field has a width of :math:`75\um` and a height of :math:`76 \um`.
# Additionally, we can see that the minimum radial spacing and minimal vertical spacing between two
# atoms are bot :math:`4 \um`, and the resolution for atom placement is :math:`0.1 \um`. For more
# details accessing and interpreting these specifications, see Amazon Braket’s Aquila example notebook
# (https://github.com/aws/amazon-braket-examples/blob/main/examples/analog_hamiltonian_simulation/01_Introduction_to_Aquila.ipynb).
# 

######################################################################
# In PennyLane, we will specify these distances in micrometers. Let's set the coordinates to be three
# points on an equilateral triangle with a side length of :math:`5 \um`, which should be well within
# the blockade radius:
# 

import numpy as np
import matplotlib.pyplot as plt

a = 5

coordinates = [(0, 0), (a, 0), (a/2, np.sqrt(a**2 - (a/2)**2))]

plt.scatter([x for x, y in coordinates], [y for x, y in coordinates])
plt.xlabel("μm")
plt.ylabel("μm")

print(f"coordinates: {coordinates}")

######################################################################
# If we want to create a Hamiltonian that we can use in PennyLane to accurately simulate a system, we
# need the correct physical constants; in this case, we need an accurate value of :math:`C6` to
# calculate the interaction term. We can access these via
# 

settings = aquila.settings
settings

######################################################################
# This can be passed into the PennyLane helper function for creating the interaction term:
# 

H_interaction = qml.pulse.rydberg_interaction(coordinates, **settings)

######################################################################
# Driving field
# ~~~~~~~~~~~~~
# 

######################################################################
# The global drive is in relation to the transition between the ground and rydberg states. It is
# defined by 3 components: the amplitude, the phase, and the detuning. Let’s consider the hardware
# limitations on each of these.
# 

def angular_SI_to_MHz(angular_SI):
    """Converts a value in rad/s or (rad/s)/s into MHz or MHz/s"""
    return angular_SI/(2*np.pi)*1e-6

aquila._device.properties.paradigm.dict()['rydberg']['rydbergGlobal']

angular_SI_to_MHz(-125000000.00)

angular_SI_to_MHz(0.2)

######################################################################
# Make this a table! Add min vals, max vals, ramps, requirements for start/end points.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Is there a way for values in a table to be pulled from ``aquila._device.properties.paradigm`` instead of “set in stone” here?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# All values in units of frequency (amplitude and detuning) are provided here in the input units
# expected by PennyLane (MHz). For simulations, these numbers will be converted to angular frequency
# (multiplied by :math:`2 \pi`) as needed. Note that these values are subject to change with hardware
# updates; current values can be accessed as shown above (in angular frequency (rad/s)) to confirm
# they remain the same.
# 
# Note that amplitude and detuning are ramped smoothly from one set-point to the next - that is to
# say, they are described by piecewise linear functions. Phase jumps at each setpoint; i.e. is
# described by a piecewise constant function. For ramped parameters, there is a maximum ramp rate for
# the hardware control, and its not possible to jump abruptly between values.
# 
# Make this a table:
# 
# Amplitude
# '''''''''
# 
# -  minimum value: 0 MHz
# -  maximum value: 2.51465 MHz
# -  resolution: 64 Hz
# -  ramped or pwc : ramped
# -  maximum ramp rate: 39788735 MHz/s
# -  start/end: must start and end at 0 MHz
# 
# Phase
# '''''
# 
# -  minimum value: -99 rad
# -  maximum value: 99 rad
# -  resolution: 5e-7 rad
# -  ramp or pwc : pwc
# -  maximum ramp rate: -
# -  start/end: must start at 0 (no corresponding limitation on final point)
# 
# Detuning
# ''''''''
# 
# -  minimum value: -19.89436788 MHz
# -  maximum value: 19.89436788 MHz
# -  ramped or pwc : ramped
# -  maximum ramp rate: 39788735 MHz/s
# -  start/end: no special restrictions
# 
# For further details on how to access the specifications and their descriptions for the device, see
# AWS Notebook 01
# https://github.com/aws/amazon-braket-examples/blob/main/examples/analog_hamiltonian_simulation/01_Introduction_to_Aquila.ipynb.
# 

######################################################################
# A few additional limitations to be aware of are:
# 
# -  for hardware upload, the full pulse program must not exceed :math:`4\mu s`
# -  the conversion from PennyLane to hardware upload will place setpoints every 50ns - consider this
#    time resolution when defining pulses.
# 

######################################################################
# The first thing we will create here is a simple pi-pulse implemented via the amplitude term of the
# hamiltonian. We will leave phase and detuning at 0. The overall driven Hamiltonian will then
# simplify to:
# 

######################################################################
# .. math::  \sum_{k=1}^N \frac{\Omega(t)}{2} (\ket{g_k}\bra{r_k} - \ket{r_k}\bra{g_k}) + \sum_{j=1}^{N-1}\sum_{k=j+1}^{N} \frac{C_6}{R^6_{jk}}n_jn_k
# 

######################################################################
# For the pulse shape, we create a gaussian envelope, and use ``qml.pulse.rect`` with a window of 1.75
# :math:`\mu`\ s to ensure that the first and last point are 0. Because we also want to run the
# simulation in PennyLane, we need to define the pulse function using ``jax.numpy``.
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

######################################################################
# And our drive will be
# 

global_drive = qml.pulse.rydberg_drive(amplitude=gaussian_fn, phase=0, detuning=0, wires=[0, 1, 2])

######################################################################
# Simulating in PennyLane to find a pi-pulse
# ------------------------------------------
# 

######################################################################
# A pi-pulse is any pulse calibrated to perform a 180 degree (:math:`\pi` radian) rotation on the
# Bloch Sphere that takes us from the ground state of the undriven system to the excited state when
# applied. Here we will create one, and observe the effect of applying it with and without Rydberg
# blockade.
# 
# In contrast to the AHS simulator from AWS, which interacts with a Hamiltonian that provides an
# accurate portrayal of a Rydberg atom system, PennyLane pulse programming works as an open-ended tool
# that will simulate whatever Hamiltonian provided. This has the advantage of allowing us to simulate
# informative but non-physical scenarios, and more freely modify our hamiltonian. (However, if your
# goal is to create a pulse program compatible with existing hardware, this can be a double edged
# sword, as it leave the onus of responsibility for ensuring your Hamiltonian is an accurate
# reflection of the relevant system entirely on you.)
# 
# Here, we will use the flexibility provided by the PennyLane simulation to observe the system with
# the interaction term “turned on” versus “turned off”. The “turned off” scenario is not an accurate
# physical representation, because it lacks inter-qubit interactions. But simulating it allows us to
# calibrate an approximate pi-pulse without worrying about the effect of Rydberg blockade, and to
# directly observe the influence of the interaction term specifically on the state of the system.
# 
# In the absence of the interaction term, each atom acts as a completely independent system, so we
# only need to measure the response of a single atom. Below, we’ve experimented with the parameters of
# the gaussian pulse envelope via trial-and-error to find settings that result in a pi-pulse.
# 

import jax

max_amplitude = 2.
displacement = 1.
sigma = 0.3

amplitude_params = [max_amplitude, displacement, sigma]
    
params = [amplitude_params]
ts = [0.0, 1.75]

dev = qml.device("default.qubit.jax", wires=3, shots=1000)

@qml.qnode(dev, interface="jax")
def circuit(parameters):
    qml.evolve(global_drive)(parameters, ts)
    return qml.counts(wires=[0])

circuit(params)

######################################################################
# Simulating Rydberg blockade in PennyLane
# ----------------------------------------
# 

######################################################################
# If we apply the same pulse, but now to the full Hamiltonian including the interaction term, we
# observe that only one of the three qubits is in the excited state. This is indeed the expected
# effect of Rydberg blockade.
# 

@qml.qnode(dev, interface="jax")
def circuit(params):
    qml.evolve(H_interaction + global_drive)(params, ts)
    return qml.counts()

circuit(params)

######################################################################
# Rydberg blockade on the QuEra hardware
# --------------------------------------
# 

######################################################################
# Let’s look at how we would move this simple pulse program from PennyLane simulation to hardware.
# First, we can quickly swap out the PennyLane ``default.qubit`` device and make sure the program will
# run on the AWS simulator. Note that while everything that will run on hardware will work on the
# simulator, some things will succeed in simulation but not be able to upload to hardware, so while
# this is a quick and useful check, it is not a guarantee that our program is hardware-ready.
# 

@qml.qnode(rydberg_simulator)  # we switch the device to Rydberg simulator
def circuit(params):
    qml.evolve(H_interaction + global_drive)(params, ts)
    return qml.counts()

circuit(params)

######################################################################
# Before uploading to hardware, it’s best to consider whether there are any constraints we need to be
# aware of. Only our amplitude parameter is non-zero, so let’s review the limitations we need to
# respect for defining an amplitude on hardware:
# 
# Amplitude
# '''''''''
# 
# -  minimum value: 0 MHz
# -  maximum value: 2.51465 MHz
# -  resolution: 64 Hz
# -  ramped or pwc : ramped
# -  maximum ramp rate: 39788735 MHz/s
# -  start/end: must start and end at 0 MHz
# 

start_val = gaussian_fn(amplitude_params, 0)
stop_val = gaussian_fn(amplitude_params, 1.75)

times = np.linspace(0, 1.75, 1000)
amplitude = [gaussian_fn(amplitude_params, t) for t in times]

start_val = amplitude[0]
stop_val = amplitude[-1]
max_val = np.max(amplitude)
max_ramp = np.max([(amplitude[i+1] - amplitude[i])/50e-9 for i in range(999)])

print(f"start value: {start_val:.3} MHz")
print(f"stop value: {stop_val:.3} MHz")
print(f"maximum value: {max_val:.3} MHz")
print(f"maximum ramp: {max_ramp:.3} MHz/s")

######################################################################
# Our maximum amplitude value and maximum ramp rate are well below hardware limits, so the only
# constraint we need to enforce for our pulse program is ensuring the values at timestamps 0 and 1.75
# :math:`\mu s` are 0. For this, we can use a convenience function provided in the pulse module,
# ``qml.pulse.rect``. We can wrap an existing function with it in order to apply a rectangular window
# within which the pulse has non-zero values.
# 
# Note that the function is non-zero outside the window, and the window is defined as including the
# end-points. This means to ensure that 0 and 1.75 return 0, they need to be outside the interval
# defining the window; we’ll use ``windows=[0.01, 1.749]``. Our modified global drive is then:
# 

global_drive = qml.pulse.rydberg_drive(amplitude=qml.pulse.rect(gaussian_fn, windows=[0.01, 1.749]), phase=0, detuning=0, wires=[0, 1, 2])

# omega_max = 2.5 / (2 * np.pi)
# displacement = 1
# sigma = 0.3
    
# params = [jnp.array([omega_max, displacement, sigma])]
# ts = jnp.array([0.0, 1.75])

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
# For the register, recall that we defined our coordinates in micrometers as
# ``[(0, 0), (5, 0), (2.5, 4.330127018922194)]``, and that we expect the hardware upload program to be
# in SI units, i.e. micrometers have been converted to meters. We can access the
# ``ahs_program.register.coordinate_list`` to see the x and y coordinates that will be passed to
# hardware:
# 

ahs_x_coordinates = ahs_program.register.coordinate_list(0)
ahs_x_coordinates

ahs_y_coordinates = ahs_program.register.coordinate_list(1)
ahs_y_coordinates

# this plot is overkill here, but use for larger register below

op_register = op.H.settings.register

op_x_coordinates = [x*1e-6 for x,_ in op_register]
op_y_coordinates = [y*1e-6 for _,y in op_register]

plt.scatter(ahs_x_coordinates, ahs_y_coordinates, label = 'AHS program')
plt.scatter(op_x_coordinates, op_y_coordinates, marker='x', label = 'Input register')
plt.legend()

######################################################################
# We can see that the final y-coordinate has been set to :math:`4.3\mu m`. We’re happy with this very
# minor change due to discretization. Let’s also look at the amplitude data. We can access the
# set-points for hardware upload from the program as
# ``ahs_program.hamiltonian.amplitude.time_series``, which contains both the ``times()`` and
# ``values()`` for setpoints. The ``amplitude`` can be switched for ``phase`` or ``detuning`` to
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

######################################################################
# Since we are happy with this, we can send this task to hardware now. If there are any issues we’ve
# missed regarding ensuring the upload data is hardware compatible, we will be informed immediately.
# Otherwise, the task will be sent to the remote hardware, and run when the hardware is online and we
# reach the front of the queue.
# 

#qml.qnode(aquila)
@qml.qnode(rydberg_simulator)
def circuit(params):
    qml.evolve(H_interaction + global_drive)(params, ts)
    return qml.counts()

circuit(params)

# run this task on hardware, demonstrate how to get a hardware task back and continue with it?
#i.e. set device _task (if so, add public function for this), and then do dev.expval(qml.PauliZ(0)) or similar
# note that it costs money!

######################################################################
# Transitional sentence(s) here
# 

######################################################################
# Z2 phase thing
# --------------
# 

######################################################################
# Now let’s use they physical properties of the device to perform a simple Analogue Hamiltonian
# Simulation task. Here we will be simulating a phase transition in condensed quantum matter -
# specifically, the transition from ferromagnetic to anti-ferromagnetic order in a 1D Ising chain. An
# Ising chain has the Hamiltonian:
# 
# .. math:: - \sum_{ij} J_{ij} \sigma_i \sigma_j - \mu \sum_j h_j \sigma_j
# 
# -  use an atom to represent a single spin in the chain, with spin-up and spin-down encoded as the
#    Rydberg and ground states respectively
# -  the van der Waals interaction term between the atoms, :math:`\sum_j \sum_k V_{jk} n_j n_k`,
#    corresponds to the spin interaction term of the Ising chain Hamiltonian,
#    :math:`-\sum_{ij} J_{ij} \sigma_i \sigma_j`, where the sign for :math:`J_{ij}` can only be
#    negative or 0 (corresponding to antiferromagnetic or noninteracting systems), and the magnitude
#    of :math:`J_{ij}` is adjusted by modifying the distance between atoms and thus their interaction
#    strength
# -  The amplitude term of the driving field couples the down and up states (but does not
#    energetically favour one over the other)
# -  The detuning term controls the energy reward (or penalty) for being in the excited state - in
#    this analogy, this corresponds to applying an external magnetic moment that encourages the spins
#    to align
# 
# Adiabatic phase change etc.
# 

######################################################################
# thoughts to resolve / experiment with - just brainstorming, not demo text
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# in the absence of a magnetic field, the spin system is in a disordered state due to spin degeneracy.
# When detuning is 0, the shift term doesn’t encourage ground vs Rydberg. But there isn’t a
# degeneracy, so surely the ground state is favored? I guess no? Otherwise where is that reflected in
# the Hamiltonian? In the absence of detuning and phase, the drive favors a time-dependent oscillation
# between ground and excited states. This is sychronous for all the atoms, but not all of them are
# driven to the Rydberg state due to blockade. How do we see a difference between this and that? What
# happens if we don’t do this adiabatically, i.e. what if we rapidly change detuning, or even leave it
# constant throughout the pulse program? What if I go way more negative for detuning? Can I compete
# with the Rydberg blockade? Could also just separate the atoms more to simulate decreasing J, with a
# constant detuning output, and observe the effect of that.
# 

######################################################################
# In simulation
# -------------
# 

######################################################################
# To begin, let’s re-initialize the devices, now with more wires, and create a 1D chain of 9 spins.
# While the radius for Rydberg blockade is dependent on drive strength, for a typical drive amplitude
# (on the order of 1 MHz), placing them at a distance of :math:`6 \mu m` should ensure that the energy
# contribution from van der Waals interactions is much larger than the scale of the drive, and we
# expect to observe blockade.
# 

import pennylane as qml

rydberg_simulator = qml.device("braket.local.ahs",  
                           wires=9)

coordinates = [(i*6, 0) for i in range(9)]
settings = rydberg_simulator.settings

H_interaction = qml.pulse.rydberg_interaction(coordinates, wires=rydberg_simulator.wires, **settings)

######################################################################
# In this case, we aim to apply a constant amplitude, while slowly varying detuning. A constraint of
# the hardware is that amplitude must be 0 and the beginning and end of the pulse, and must respect
# the maximum ramp rate of the control hardware. Once we’ve simulated the function and settled on an
# amplitude, we’ll calcualte how much time we need to provide to ramp up, and modify our amplitude
# function accordingly. For now, we will ignore the restriction and set the amplitude to a fixed value
# for the entire duration of the pulse program.
# 
# First let’s look at the behaviour of the system in the absence of phase or detuning, for a
# :math:`4 \mu s` pulse program.
# 

global_drive = qml.pulse.rydberg_drive(amplitude=qml.pulse.constant, phase=0, detuning=0, wires=rydberg_simulator.wires)

amplitude = 2.4
    
params = [amplitude]

ts = jnp.array([0, 4])

@qml.qnode(rydberg_simulator)
def circuit(params):
    qml.evolve(H_interaction + global_drive)(params, ts)
    return qml.sample()

results = circuit(params)

average_density = np.mean(results, axis=0)
plt.bar(range(len(average_density)), average_density)
plt.xlabel("Indices of atoms")
plt.ylabel("Average Rydberg density")
plt.ylim(0, 1)

######################################################################
# The effect of the Rydberg blockade is discernable in the system; even-indexed atoms are more likely
# to be in the excited state than odd-indexed atoms. However, overall, the system is in a fairly
# disordered state, which each atom having 60-80% chance of being in the ground state.
# 

######################################################################
# Now let’s add a slowly varying detuning, starting in the regime where detuning is negative (ground
# state strongly favoured for all atoms), and shifting to a positive detuning (where the excited state
# is strongly favored for each individual atom, but discouraged for adjacent atoms by the interaction
# term).
# 

# the final timestamp in our pulse will be 4us, the maximum duration of a pulse program
max_time=4

def detuning_fn(params, t):
    if t < 0.25:
        return params[0]
    
    if t > max_time-0.25:
        return params[1]
    
    else:
        slope = (params[1] - params[0]) / (max_time-0.5)
        offset = params[0] - slope * 0.25
        return slope * t + offset

detuning_range = [-8, 8]

times = np.linspace(0, max_time, 500)
detuning = [detuning_fn(detuning_range, t) for t in times]

plt.plot(times, detuning)
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Detuning ($2\pi$ MHz)")
plt.show()

global_drive = qml.pulse.rydberg_drive(amplitude=qml.pulse.rect(qml.pulse.constant, windows=[0.001, 3.999]), phase=0, detuning=detuning_fn, wires=rydberg_simulator.wires)

params = [amplitude, detuning_range]

ts = jnp.array([0, 4])

@qml.qnode(rydberg_simulator)
def circuit(params, max_time):
    qml.evolve(H_interaction + global_drive)(params, ts)
    return qml.sample()

results = circuit(params, 4) 

average_density = np.mean(results, axis=0)
plt.bar(range(len(average_density)), average_density)
plt.xlabel("Indices of atoms")
plt.ylabel("Average Rydberg density")
plt.ylim(0, 1)

times = np.linspace(0.5, 4, 8)
results = []

for time in times:
    res = circuit(params, time)
    average_density = np.mean(res, axis=0)
    results.append(average_density)

time_vs_results = np.array(results).transpose()

fig, ax1 = plt.subplots()

data = ax1.imshow(time_vs_results, cmap='hot', vmin=0, vmax=1)
ax1.set_xticks(np.arange(len(times)), labels = times)
plt.colorbar(data)
plt.xlabel('Pulse duration [$\mu s$]')
plt.ylabel('Indices of atoms')

######################################################################
# At strongly negative detuning (:math:`0.5 \mu s`), the drive is not sufficient to excite the atoms
# because of the large energy penalty for being in the excited state in the detuning term of the
# Hamiltonian.
# 
# Around :math:`2.5 \mu s` into the program, as we approach :math:`\Delta = 0`, we see something
# similar to the results above (taken with detuning=0), where the drive has some effect, and the
# probability of being in the excited state is roughly 20-40%. At this point we begin to see the
# antiferromagnetic behaviour, but the system is still fairly disordered.
# 
# Shifting further to positive detuning, we reach a regime where the Rydberg state is strongly favored
# energectically - but the scale of the Rydberg blockade is still high enough to ensure we reach
# antiferromagnetic, rather than ferromagnetic order. Here we see a
# 

break

######################################################################
# On Aquila hardware
# ------------------
# 

######################################################################
# We’ll upload and run the full duration (:math:`4 \mu s`) program to the Aquila hardware. As
# discussed above, it run on hardware we will need to slightly modify our amplitude function, to
# ensure it is 0 at the beginning and end of the pulse program, and that we respect the maximum ramp
# rate for the control hardware, 39788735 MHz/s. We are going to a maximum amplitude of 2.4 MHz, so
# the ramping duration needed to respect the maximum ramp rate is just over 60 nanoseconds:
# 

max_amp = 2.4  # MHz
max_ramp_rate = 39788735 # MHz/s

max_amp/max_ramp_rate  # s

######################################################################
# Let’s define a piecewise constant function that sets the values of an array based on the maximum
# value and ramp rate, assuming a 4 microsecond pulse program. Since we can’t ramp up faster than
# 60ns, and we have a bin size of 50ns, let’s ramp up over two bins:
# 

max_time = 4  # in microseconds
bin_size = 50e-3  # in microseconds
num_bins = int(max_time//bin_size + 1)

def get_amp_array(max_amp):
    ramp_up = [0, max_amp/2] # ramp up over two bins
    constant_output = [max_amp for _ in range(num_bins-4)]
    ramp_down = [max_amp/2, 0] # ramp down over two bins
    return jnp.array(ramp_up + constant_output + ramp_down)

def amp_fn(max_amp, t):
    output_array = get_amp_array(max_amp)
    return qml.pulse.pwc(max_time)(output_array, t)

times = np.linspace(0, 4, 8001)
amp = [amp_fn(2.4, t) for t in times]

plt.plot(times, amp)
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Amplitude ($2\pi$ MHz)")
plt.show()

######################################################################
# Our drive then becomes:
# 

global_drive = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0, detuning=detuning_fn, wires=remote_device.wires)

max_amp = 2.4  # MHz
detuning_range = [-8, 8]  # MHz

params = [max_amp, detuning_range]

######################################################################
# Let’s look at our upload data:
# 

op = qml.evolve(H_interaction + global_drive)(params, [0.0, 4])
ahs_program = aquila.create_ahs_program(op)

ahs_x_coordinates = ahs_program.register.coordinate_list(0)
ahs_y_coordinates = ahs_program.register.coordinate_list(1)

op_register = op.H.settings.register

op_x_coordinates = [x*1e-6 for x,_ in op_register]
op_y_coordinates = [y*1e-6 for _,y in op_register]

plt.scatter(ahs_x_coordinates, ahs_y_coordinates, label = 'AHS program')
plt.scatter(op_x_coordinates, op_y_coordinates, marker='x', label = 'Input register')
plt.legend()

# hardware set-points after conversion and discretization
amplitude_setpoints = ahs_program.hamiltonian.amplitude.time_series

# values for plotting the function defined in PennyLane for amplitude
input_times = np.linspace(*ts, 1000)
input_amplitude = [amp_fn(params[0], _t) for _t in np.linspace(*ts, 1000)]

# plot PL input and hardware setpoints for comparison
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(input_times, input_amplitude)
ax1.set_xlabel('Time [$\mu s$]')
ax1.set_ylabel('MHz')
ax1.set_title('detuning_fn')

ax2.plot(amplitude_setpoints.times(), amplitude_setpoints.values())
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('rad/s')
ax2.set_title('upload data')

plt.tight_layout()
plt.show()


# consider plotting maximum ramp rate on here to compare

# hardware set-points after conversion and discretization
detuning_setpoints = ahs_program.hamiltonian.detuning.time_series

# values for plotting the function defined in PennyLane for amplitude
input_times = np.linspace(*ts, 1000)
input_detuning = [detuning_fn(params[1], _t) for _t in np.linspace(*ts, 1000)]

# plot PL input and hardware setpoints for comparison
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(input_times, input_detuning)
ax1.set_xlabel('Time [$\mu s$]')
ax1.set_ylabel('MHz')
ax1.set_title('detuning_fn')

ax2.plot(detuning_setpoints.times(), detuning_setpoints.values())
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('rad/s')
ax2.set_title('upload data')

plt.tight_layout()
plt.show()

######################################################################
# Everything looks as expected, so let’s do a final sanity check on simulator and then upload our
# program:
# 

# sanity check on simulator
@qml.qnode(remote_device)
def circuit_simulator(params):
    qml.evolve(H_interaction + global_drive)(params, [0.0, 4])
    return qml.sample()

results = circuit_simulator(params)

average_density = np.mean(results, axis=0)
plt.bar(range(len(average_density)), average_density)
plt.xlabel("Indices of atoms")
plt.ylabel("Average Rydberg density")
plt.ylim(0, 1)

# ToDo: hardware can run this with more atoms: maybe also do it with a longer chain (maybe a long, circular chain?) so that it was actually relevant to run on hardware 
# simulator is good for up to 15 qubits, maybe run something on 30? At long range the order is less good (see center)

# reinitialize aquila with the correct number of wires
# s3 = ("my-bucket", "my-prefix")
aquila = qml.device("braket.aws.ahs", 
                    device_arn="arn:aws:braket:us-east-1::device/qpu/quera/Aquila", 
                    # s3_destination_folder=s3, 
                    wires=9)

@qml.qnode(aquila)
def circuit_hardware(params):
    qml.evolve(H_interaction + global_drive)(params, [0.0, 4])
    return qml.sample()

results = circuit_hardware(params)

# we need to use np.nanmean here instead of np.mean, because occasionally a shot returns nan instead of 0 or 1, and np.nanmean handles as having one fewer input instead of returning NaN
average_density = np.nanmean(results, axis=0)
plt.bar(range(len(average_density)), average_density)
plt.xlabel("Indices of atoms")
plt.ylabel("Average Rydberg density")
plt.ylim(0, 1)

# Do we want to include anything like this? Maybe not, long-range order doesn't seem to work that well. Possibly need a longer, slower ramp? But that can't be implemented. But it *kind of* works.

# reinitialize aquila with the correct number of wires
# s3 = ("my-bucket", "my-prefix")
aquila = qml.device("braket.aws.ahs", 
                    device_arn="arn:aws:braket:us-east-1::device/qpu/quera/Aquila", 
                    # s3_destination_folder=s3, 
                    wires=50)

a = 4

coordinates = [(i*a, 0) for i in range(10)]

for i in range(10):
    for j in [1, 2, 3, 4]:
        coordinates.append((i*a, j*a))

settings = remote_device.settings

H_interaction = qml.pulse.rydberg_interaction(coordinates, wires=aquila.wires, **settings)
global_drive = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0, detuning=detuning_fn, wires=aquila.wires)

# @qml.qnode(aquila)
# def circuit_hardware(params):
#     qml.evolve(H_interaction + global_drive)(params, [0.0, 4])
#     return qml.sample()

# results = circuit_hardware(params)

# we need to use np.nanmean here instead of np.mean, because occasionally a shot returns nan instead of 0 or 1, and np.nanmean handles as having one fewer input instead of returning NaN
#average_density_2d = np.nanmean(results, axis=0)
density_data_2d = average_density_2d.reshape(10, 5)

fig, ax1 = plt.subplots()

data = ax1.imshow(density_data_2d, cmap='hot', vmin=0, vmax=1)
#ax1.set_xticks(np.arange(len(times)), labels = times)
plt.colorbar(data)
#plt.xlabel('Pulse duration [$\mu s$]')
#plt.ylabel('Indices of atoms')

######################################################################
# Conclusion
# ----------
# 
# -  AHS is an intesting and active area of research in a different kind of quantum computation
# -  “Already today, researchers are using such devices to study quantum phenomena that otherwise
#    would be hard to simulate on classical computers.”
# -  lots of applications
# -  we made a phase transition - yay!
# 

######################################################################
# Citations
# =========
# 
# [1] https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html [2]
# https://doi.org/10.1073/pnas.1804949115 (Universal quantum Hamiltonians) [3]
# https://doi.org/10.1126/science.abi8794 (Probing topological spin liquids on a programmable quantum
# simulator) [4]
# https://aws.amazon.com/blogs/quantum-computing/realizing-quantum-spin-liquid-phase-on-an-analog-hamiltonian-rydberg-simulator/
# 