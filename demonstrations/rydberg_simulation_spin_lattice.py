# ToDo: flesh out Z2 phase background text
# ToDo: review references and incorporate any relevant information better
# ToDo: flesh out conclusion

r"""Analog Hamiltonian simulation of anti-ferromagnetism
========================================================
Neutral atom hardware is a new innovation in quantum technology that has been gaining traction in
recent years thanks to new developments in optical tweezer technology. One such device, QuEra’s Aquila,
is capable of running circuits with up to 256 physical qubits! The Aquila device is now accessible and
programmable via pulse programming in PennyLane and the Braket SDK plugin.
In this demo, we will simulate a simple quantum phase transition on the Aquila hardware
using analog Hamiltonian simulation, an alternative to gate-based quantum computing. This is possible
in PennyLane using the :mod:`~pennylane.pulse` module for pulse-level control in combination with the `pennylane-braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`_.









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







Now let us use the physical properties of the device to perform a simple analog Hamiltonian
simulation task. Here we will be simulating a phase transition in condensed quantum matter -
specifically, the transition from ferromagnetic to anti-ferromagnetic order in a 1D Ising chain. An
Ising chain has the Hamiltonian:

.. math:: - \sum_{ij} J_{ij} \sigma_i \sigma_j - \mu \sum_j h_j \sigma_j

-  use an atom to represent a single spin in the chain, with spin-up and spin-down encoded as the
   Rydberg and ground states respectively
-  the van der Waals interaction term between the atoms, :math:`\sum_j \sum_k V_{jk} n_j n_k`,
   corresponds to the spin interaction term of the Ising chain Hamiltonian,
   :math:`-\sum_{ij} J_{ij} \sigma_i \sigma_j`, where the sign for :math:`J_{ij}` can only be
   negative or 0 (corresponding to antiferromagnetic or non-interacting systems), and the magnitude
   of :math:`J_{ij}` is adjusted by modifying the distance between atoms and thus their interaction
   strength
-  The amplitude term of the driving field couples the down and up states (but does not
   energetically favour one over the other)
-  The detuning term controls the energy reward (or penalty) for being in the excited state - in
   this analogy, this corresponds to applying an external magnetic moment that encourages the spins
   to align

Adiabatic phase change etc.

|

.. figure:: ../demonstrations/rydberg_simulation_spin_lattice/rydberg_atom_chain.png
    :align: center
    :scale: 20%
    :alt: Four Rubidium atoms trapped in optical tweezers, in alternating ground and excited states
    :target: javascript:void(0);

|

A 1D chain on a local simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To begin, let’s initialize the devices, now with more wires, and create a 1D chain of 9 spins.
While the radius for Rydberg blockade is dependent on drive strength, for a typical drive amplitude
(on the order of 1 MHz), placing them at a distance of :math:`6 \mu m` should ensure that the energy
contribution from van der Waals interactions is much larger than the scale of the drive, and we
expect to observe blockade.

"""
import pennylane as qml

rydberg_simulator = qml.device("braket.local.ahs",
                               wires=9)

coordinates = [(i * 6, 0) for i in range(9)]

H_interaction = qml.pulse.rydberg_interaction(coordinates, wires=rydberg_simulator.wires, **settings)

######################################################################
# In this case, we aim to apply a constant amplitude, while slowly varying detuning. A constraint of
# the hardware is that amplitude must be 0 and the beginning and end of the pulse, and must respect
# the maximum rate of change for the control hardware. Once we’ve simulated the function and settled on an
# amplitude, we’ll calculate how much time we need to provide to ramp up, and modify our amplitude
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
def circuit(params, t):
    qml.evolve(H_interaction + global_drive)(params, t)
    return qml.sample()


results = circuit(params, ts)

average_density = np.mean(results, axis=0)
plt.bar(range(len(average_density)), average_density)
plt.xlabel("Indices of atoms")
plt.ylabel("Average Rydberg density")
plt.ylim(0, 1)

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/atom_chain_no_detuning_simulator.png
#     :align: center
#     :scale: 50%
#     :alt: The Rydberg state density at each atom index for chain of 9 atoms without detuning (0.2-0.4 at each index)
#     :target: javascript:void(0);
#
#
# The effect of the Rydberg blockade is discernible in the system; even-indexed atoms are more likely
# to be in the excited state than odd-indexed atoms. However, overall, the system is in a fairly
# disordered state, which each atom having 60-80% chance of being in the ground state.
#
#
# Now let’s add a slowly varying detuning, starting in the regime where detuning is negative (ground
# state strongly favoured for all atoms), and shifting to a positive detuning (where the excited state
# is strongly favored for each individual atom, but discouraged for adjacent atoms by the interaction
# term).
#

# the final timestamp in our pulse will be 4us, the maximum duration of a pulse program
max_time = 4


def detuning_fn(params, t):
    if t < 0.25:
        return params[0]

    if t > max_time - 0.25:
        return params[1]

    else:
        slope = (params[1] - params[0]) / (max_time - 0.5)
        offset = params[0] - slope * 0.25
        return slope * t + offset


detuning_range = [-8, 8]

times = np.linspace(0, max_time, 500)
detuning = [detuning_fn(detuning_range, t) for t in times]

plt.plot(times, detuning)
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Detuning ($2\pi$ MHz)")
plt.show()

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/detuning_fn.png
#     :align: center
#     :scale: 50%
#     :alt: The function describing detuning for the pulse sequence, starting at -8 and increasing linearly to +8
#     :target: javascript:void(0);
#

amp_fn = qml.pulse.rect(qml.pulse.constant, windows=[0.001, 3.999])
global_drive = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0, detuning=detuning_fn, wires=rydberg_simulator.wires)

params = [amplitude, detuning_range]

ts = jnp.array([0, 4])


@qml.qnode(rydberg_simulator)
def circuit(params, t):
    qml.evolve(H_interaction + global_drive)(params, t)
    return qml.sample()


results = circuit(params, ts)

average_density = np.mean(results, axis=0)
plt.bar(range(len(average_density)), average_density)
plt.xlabel("Indices of atoms")
plt.ylabel("Average Rydberg density")
plt.ylim(0, 1)

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/atom_chain_with_detuning_simulator.png
#     :align: center
#     :scale: 50%
#     :alt: The Rydberg state density at each atom index with the detuning (alternates roughly between 0.1 and 0.9)
#     :target: javascript:void(0);
#

times = np.linspace(0.5, 4, 8)
results = []

for time in times:
    res = circuit(params, time)
    average_density = np.mean(res, axis=0)
    results.append(average_density)

time_vs_results = np.array(results).transpose()

fig, ax1 = plt.subplots()

data = ax1.imshow(time_vs_results, cmap='Blues', vmin=0, vmax=1)
ax1.set_xticks(np.arange(len(times)), labels=times)
plt.colorbar(data)
plt.xlabel('Pulse duration [$\mu s$]')
plt.ylabel('Indices of atoms')

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/atom_chain_heatmap.png
#     :align: center
#     :scale: 50%
#     :alt: The Rydberg density for each atom, taken at different points in the pulse sequence, showing the emergence of antiferromagnetic order
#     :target: javascript:void(0);
#
#
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
# energetically - but the scale of the Rydberg blockade is still high enough to ensure we reach
# antiferromagnetic, rather than ferromagnetic order. Here we see a
#
#
#
# A 1D chain on the Aquila hardware
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We’ll upload and run the full duration (:math:`4 \mu s`) program to the Aquila hardware. As
# discussed above, to run on hardware we will need to slightly modify our amplitude function, to
# ensure it is 0 at the beginning and end of the pulse program, and that we respect the maximum
# rate of change for the control hardware, 39788735 MHz/s. We are going to a maximum amplitude of
# 2.4 MHz, so the duration needed to respect the maximum rate of change is just over 60 nanoseconds:
#

max_amp = 2.4  # MHz
max_rate = 39788735  # MHz/s

max_amp / max_rate  # s

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      6.031858012073014e-08
#
#
# Let’s define a piecewise constant function that sets the values of an array based on the maximum
# value and maximum rate of change, assuming a 4 microsecond pulse program. This will be sampled and converted
# to a piecewise linear function that approximates it for hardware upload, but in this case a
# piecewise constant function is an easy way to define our pulse. Since we can’t go to maximum
# amplitude faster than 60ns, and we have a bin size of 50ns (so it will match the function sampling
# rate when converting from PennyLane to hardware instructions), let’s increase amplitude over two bins:
#

max_time = 4  # in microseconds
bin_size = 50e-3  # in microseconds
num_bins = int(max_time // bin_size + 1)


# define set-points for the function that have the desired shape
def get_amp_array(max_amp):
    ramp_up = [0, max_amp / 2]  # ramp up over two bins
    constant_output = [max_amp for _ in range(num_bins - 4)]
    ramp_down = [max_amp / 2, 0]  # ramp down over two bins
    return jnp.array(ramp_up + constant_output + ramp_down)


# create a pwc function based on the setpoints
def amp_fn(max_amp, t):
    output_array = get_amp_array(max_amp)
    return qml.pulse.pwc(max_time)(output_array, t)


times = np.linspace(0, 4, 8001)
amp = [amp_fn(2.4, t) for t in times]

plt.plot(times, amp)
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Amplitude ($2\pi$ MHz)")
plt.show()

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/square_amplitude_pulse.png
#     :align: center
#     :scale: 50%
#     :alt: The square pulse function used to define the amplitude envelope
#     :target: javascript:void(0);
#
#
# Our drive then becomes:
#

global_drive = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0, detuning=detuning_fn, wires=rydberg_simulator.wires)

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

op_x_coordinates = [x * 1e-6 for x, _ in op_register]
op_y_coordinates = [y * 1e-6 for _, y in op_register]

plt.scatter(ahs_x_coordinates, ahs_y_coordinates, label='AHS program')
plt.scatter(op_x_coordinates, op_y_coordinates, marker='x', label='Input register')
plt.xlabel("μm")
plt.ylabel("μm")
plt.legend()

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/atom_chain_discretization.png
#     :align: center
#     :scale: 50%
#     :alt: The initial and uploaded atom positions still match after discretization
#     :target: javascript:void(0);
#

# hardware set-points after conversion and discretization
amplitude_setpoints = ahs_program.hamiltonian.amplitude.time_series

# values for plotting the function defined in PennyLane for amplitude
input_times = np.linspace(*ts, 1000)
input_amplitude = [amp_fn(params[0], _t) for _t in np.linspace(*ts, 1000)]

# plot PL input and hardware set-points for comparison
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

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/square_amplitude_fn_vs_upload.png
#     :align: center
#     :scale: 50%
#     :alt: The square envelope defining the amplitude matches the corresponding amplitude output created for hardware
#     :target: javascript:void(0);
#

# consider plotting maximum ramp rate on here to compare

# hardware set-points after conversion and discretization
detuning_setpoints = ahs_program.hamiltonian.detuning.time_series

# values for plotting the function defined in PennyLane for amplitude
input_times = np.linspace(*ts, 1000)
input_detuning = [detuning_fn(params[1], _t) for _t in np.linspace(*ts, 1000)]

# plot PL input and hardware set-points for comparison
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


##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/detuning_vs_upload.png
#     :align: center
#     :scale: 50%
#     :alt: The function defining the detuning matches the corresponding detuning output created for hardware
#     :target: javascript:void(0);
#
#
# Everything looks as expected, so let’s do a final sanity check on simulator and then upload our
# program:
#

# sanity check on simulator
@qml.qnode(rydberg_simulator)
def circuit_simulator(params):
    qml.evolve(H_interaction + global_drive)(params, [0.0, 4])
    return qml.sample()


results = circuit_simulator(params)

average_density = np.mean(results, axis=0)
plt.bar(range(len(average_density)), average_density)
plt.xlabel("Indices of atoms")
plt.ylabel("Average Rydberg density")
plt.ylim(0, 1)

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/sanity_check_on_simulator.png
#     :align: center
#     :scale: 50%
#     :alt: The alternating excited-ground-excited pattern is demonstrated on simulator as expected
#     :target: javascript:void(0);
#

# reinitialize aquila with the correct number of wires
s3 = ("my-bucket", "my-prefix")
aquila = qml.device("braket.aws.ahs",
                    device_arn="arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
                    s3_destination_folder=s3,
                    wires=9)


@qml.qnode(aquila)
def circuit_hardware(params):
    qml.evolve(H_interaction + global_drive)(params, [0.0, 4])
    return qml.sample()


results = circuit_hardware(params)

# we need to use np.nanmean here instead of np.mean, because
# occasionally a shot returns nan instead of 0 or 1, and np.nanmean
# handles as having one fewer input instead of returning NaN
average_density = np.nanmean(results, axis=0)
plt.bar(range(len(average_density)), average_density, color='C3')
plt.xlabel("Indices of atoms")
plt.ylabel("Average Rydberg density")
plt.ylim(0, 1)

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/simple_chain_hardware_results.png
#     :align: center
#     :scale: 50%
#     :alt: The alternating excited-ground-excited state for the atoms measured on hardware
#     :target: javascript:void(0);
#
#
# Anti-ferromagnetic order in a 2D grid
# -------------------------------------
#
# We can create the same kind of order in a 2D lattice as well. Here we will create two separate Hamiltonians,
# with different numbers of atoms. The local simulator can handle up to 15 atoms, but is faster below 10; here
# we've elected to use a 3x3 grid for local simulation. Execution time for hardware doesn't scale with the number
# of qubits, so we will create a 5x10 grid for the hardware device.

# reinitialize aquila with the correct number of wires
s3 = ("my-bucket", "my-prefix")
aquila = qml.device("braket.aws.ahs",
                    device_arn="arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
                    s3_destination_folder=s3,
                    wires=50)  # 5x10 = 50 atoms for hardware

rydberg_simulator = qml.device("braket.local.ahs", wires=9)  # 3x3 = 9 atoms for local simulation

##############################################################################
#
# Similarly, we'll create two different sets of coordinates. We'll use a spacing of $6 \mu m$ so that
# we observe Rydberg blockade between nearest neighbors. We also need to create two drive terms, as they
# differ with regard to the wires the drive is applied to.

a = 6

coordinates_aquila = []
coordinates_local_simulator = []

# 10x5 sets of coordinates for hardware
for i in range(10):
    for j in range(5):
        coordinates_aquila.append((i * a, j * a))

# 3x3 sets of coordinates for local simulation
for i in range(3):
    for j in range(3):
        coordinates_local_simulator.append((i * a, j * a))

H_int_aquila = qml.pulse.rydberg_interaction(coordinates_aquila,
                                             wires=aquila.wires,
                                             **aquila.settings)
global_drive_aquila = qml.pulse.rydberg_drive(amplitude=amp_fn,
                                              phase=0,
                                              detuning=detuning_fn,
                                              wires=aquila.wires)

H_int_simulator = qml.pulse.rydberg_interaction(coordinates_local_simulator,
                                                wires=rydberg_simulator.wires,
                                                **rydberg_simulator.settings)
global_drive_simulator = qml.pulse.rydberg_drive(amplitude=amp_fn,
                                                 phase=0,
                                                 detuning=detuning_fn,
                                                 wires=rydberg_simulator.wires)


##############################################################################
#
# text text
#
#

@qml.qnode(rydberg_simulator)
def circuit_simulator(params):
    qml.evolve(H_int_simulator + global_drive_simulator)(params, [0.0, 4])
    return qml.sample()


@qml.qnode(aquila)
def circuit_hardware(params):
    qml.evolve(H_int_aquila + global_drive_aquila)(params, [0.0, 4])
    return qml.sample()


##############################################################################
#
# text text
#
#

results = circuit_simulator(params)

average_density_2d = np.mean(results, axis=0)
density_data_2d = average_density_2d.reshape(3, 3)

fig, ax1 = plt.subplots()

data = ax1.imshow(density_data_2d, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(data, label='Rydberg density')
plt.xlabel('Atom horizontal index')
plt.ylabel('Atom vertical index')

plt.tight_layout

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/3_by_3_grid_simulator_result.png
#     :align: center
#     :scale: 50%
#     :alt: The alternating excited-ground-excited state for the atoms simulated locally
#     :target: javascript:void(0);
#
# comment on results - now we run on hardware

results = circuit_hardware(params)

# we need to use np.nanmean here instead of np.mean, because occasionally a shot returns
# nan instead of 0 or 1, and np.nanmean handles as having one fewer input instead of
# returning NaN
average_density_2d = np.nanmean(results, axis=0)
density_data_2d = average_density_2d.reshape(10, 5)

fig, ax1 = plt.subplots()

data = ax1.imshow(density_data_2d, cmap='Reds', vmin=0, vmax=1)
plt.colorbar(data, label='Rydberg density')
plt.xlabel('Atom horizontal index')
plt.ylabel('Atom vertical index')

##############################################################################
#
# .. figure:: ../demonstrations/rydberg_simulation_spin_lattice/5_by_10_grid_hardware_result.png
#     :align: center
#     :scale: 50%
#     :alt: The Rydberg density for a 5x10 grid of atoms, showing antiferromagnetic order due to the pulse sequence
#     :target: javascript:void(0);
#
# comment on results
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