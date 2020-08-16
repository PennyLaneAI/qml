r"""
Quantum computation with neutral atoms
======================================

.. meta::
    :property="og:description": Neutral atom quantum devices allow you to place 
        qubits within interesting three-dimensional configurations.
    :property="og:image": https://pennylane.ai/qml/_images/pasqal_thumbnail.png

Quantum computing architectures come in many flavours: superconducting qubits, ion traps, 
photonics, silicon, and more. One really interesting physical substrate is *neutral atoms*. These 
quantum devices have some basic similarities to ion-traps. Ion-trap devices make use of atoms 
that have an imbalance between protons (positively charged) and electrons (negatively charged). 
Neutral atoms, on the other hand, have an equal number of protons and electrons. 

In neutral-atom systems, you use lasers to arrange atoms in various two- or 
three-dimensional configurations. This opens up some tantalizing possibilities for 
exotic quantum-computing circuit topologies:

.. figure:: https://raw.githubusercontent.com/lhenriet/cirq-pasqal/fc4f9c7792a8737fde76d4c05828aa538be8452e/pasqal-tutorials/files/eiffel_tower.png
    :align: center
    :width: 90%
    
    ..
    
    Image originally from [#barredo2017]_.
    
The startup company `Pasqal <https://pasqal.io/>`_ is one of the companies working to bring 
neutral-atom quantum computing devices to the world. Recently, Pasqal merged some new 
features into the quantum software library Cirq to support this new class of neutral atom 
devices.
 
In this demo, we will use PennyLane, Cirq, and TensorFlow to show off the unique abilities of 
neutral atom devices, leveraging them to make a quantum machine learning circuit which has a
very unique topology: the Eiffel tower. 

Let's get to it!
"""

# First we load some necessary libraries and functions
from itertools import combinations
import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

##############################################################################
# Our first step will be to load and visualize the data for the Eiffel tower 
# configuration, generously provided to us by the team at Pasqal.

coords = np.loadtxt("pasqal/Eiffel_tower_data.dat")
xs = coords[:,0]
ys = coords[:,1]
zs = coords[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(40, 15)
plt.axis('off')
ax.scatter(xs, ys, zs, c='g',alpha=0.3)
plt.show();

##############################################################################
# This dataset contains 126 points. Each point represents a distinct 
# neutral-atom qubit. This is currently outside the reach of any quantum
# device (hardware or simulator), so for the purposes of this demo, 
# we will cut this down to just 13 points, evenly spaced around the tower.
# These are highlighted in red below:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(40, 15)
plt.axis('off')
ax.scatter(xs, ys, zs, c='g',alpha=0.3)

mask = [3,7,11,15,48,51,60,63,96,97,98,99,125]
subset_coords = coords[mask]

xs = subset_coords[:,0]
ys = subset_coords[:,1]
zs = subset_coords[:,2]
ax.scatter(xs, ys, zs, c='r',alpha=1.0);

##############################################################################
# Our next step will be to convert these coordinates into objects that 
# Cirq understands as qubits. For neutral-atom devices in Cirq, these are the
# ``ThreeDQubit`` objects, which carry information about the three-dimensional
# arrangement of qubits.
#
# Now, neutral atom devices come with some physical restrictions. 
# Specifically, qubits in a particular configuration can't arbitrarily be 
# interacted with one another. Instead, there is the notion of a 
# "control radius"; any atoms which are within the system's control radius 
# can interact with one another. 
# 
# In order to allow our Eiffel tower qubits to interact with 
# one another more easily, we will artificially scale some dimensions
# when placing the atoms. 

from cirq.pasqal import ThreeDQubit
xy_scale = 1.5
z_scale = 0.75
base_qubits = [ThreeDQubit(x,y,z) for x,y,z in subset_coords[:8]]
tower_qubits = [ThreeDQubit(xy_scale*x,xy_scale*y,z_scale*z) 
                for x,y,z in subset_coords[8:]]
qubits = base_qubits + tower_qubits
qubits

##############################################################################
# To simulate a neutral-atom quantum computation, we can use the 
# ``"cirq.pasqal"`` device, available via the Cirq plugin to PennyLane.
# We will need to provide this device with the ``ThreeDQubits`` we created 
# above. We will also need to instantiate it with a fixed control radius.

num_wires = len(qubits)
control_radius = 32.4
dev = qml.device("cirq.pasqal", control_radius=control_radius, 
                 qubits=qubits, wires=num_wires)

##############################################################################
# Our quantum-computing circuit will be based on the Eiffel tower
# configuration. Each of the 13 qubits specified above can be thought of
# as a single wire in a quantum circuit. Our circuit will consist of several
# stages:
#
# i. Data is fed in to the first "ground" vertical level of qubits using
#    parametrized single-qubit Pauli-Y rotations (a simple data-embedding 
#    strategy).
#
# ii. Two-qubit interactions (in this case, CNOTs) are carried out
#     between each ground-level qubit and the corresponding nearest qubit
#     nearest qubits on the second vertical level.
#
# iii. At the second level, two-qubit CNOT gates are applied between all
#      pairs of qubits.
# 
# iv. This pattern is repeated for the third vertical level of qubits.
# 
# v. All qubits from the third level interact with a single "peak" qubit
#    using CNOTs.
#
# vi. At each vertical level, parametrized single-qubit rotations are applied
#     (in this case, Pauli-Y rotations).
#
# The output of our circuit is determined via a Pauli-Z measurement on
# the "peak" qubit.
#
# Now, that's a lot to keep track of, so let's show the circuit via a 
# three-dimensional image:

first_lvl_qubits = range(0,4)
second_lvl_qubits = range(4,8)
third_lvl_qubits = range(8,12)
peak_qubit = 12

first_lvl_coords = subset_coords[first_lvl_qubits]
second_lvl_coords = subset_coords[second_lvl_qubits]
third_lvl_coords = subset_coords[third_lvl_qubits]
peak_coords = subset_coords[peak_qubit]

first_x, first_y, first_z = [first_lvl_coords[:,idx] 
                             for idx in range(3)]
second_x, second_y, second_z = [second_lvl_coords[:,idx] 
                                for idx in range(3)]
third_x, third_y, third_z = [third_lvl_coords[:,idx] 
                             for idx in range(3)]
peak_x, peak_y, peak_z = peak_coords

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(40, 15)
plt.axis('off')

ax.scatter(xs, ys, zs, c='r',alpha=1.0);

# Two-qubit gates between first and second level
for corner in range(4):
    ax.plot(xs=[first_x[corner], second_x[corner]],
            ys=[first_y[corner], second_y[corner]],
            zs=[first_z[corner], second_z[corner]],
            c='k');

# Two-qubit gates between second and third levels
for corner in range(4):
    ax.plot(xs=[second_x[corner], third_x[corner]],
            ys=[second_y[corner], third_y[corner]],
            zs=[second_z[corner], third_z[corner]],
            c='k');
    
# Two-qubit gates between third level and peak
for corner in range(4):
    ax.plot(xs=[third_x[corner], peak_x],
            ys=[third_y[corner], peak_y],
            zs=[third_z[corner], peak_z],
            c='k');

index_pairs = combinations(range(4), 2)

# Two-qubit gates between all pairs at second level
for idx, jdx in index_pairs:
    ax.plot(xs=[second_x[idx], second_x[jdx]],
            ys=[second_y[idx], second_y[jdx]],
            zs=[second_z[idx], second_z[jdx]],
            c='k');
# Two-qubit gates between all pairs at third level
    ax.plot(xs=[third_x[idx], third_x[jdx]],
            ys=[third_y[idx], third_y[jdx]],
            zs=[third_z[idx], third_z[jdx]],
            c='k');
plt.show();

##############################################################################
# In this figure, the red dots represent our qubits, arranged in a 
# three-dimensional configuration roughly resembling the Eiffel tower.
# The black lines indicate CNOT gates between certain qubits.
# Data is loaded in at the bottom qubits (the "tower legs") and the final
# measurement result is read out from the top "peak" qubit.
#
# Our next step is to actually create this configuration in a quantum circuit:

@qml.qnode(dev, interface="tf")
def circuit(weights, data):
    
    # First level
    for idx in first_lvl_qubits:
        qml.RY(data[idx], wires=idx)  # data loading
        qml.RY(weights[idx], wires=idx)  # parameterized rotations on each qubit

    # Interact qubits from first and second levels
    for idx, jdx in zip(first_lvl_qubits, second_lvl_qubits):
        qml.CNOT(wires=[idx,jdx])
    
    # Second level
    for idx in second_lvl_qubits:
        qml.RY(weights[idx], wires=idx)  # parameterized rotations on each qubit 
    for idx, jdx in combinations(second_lvl_qubits, 2):
        qml.CNOT(wires=[idx,jdx])  # interact each qubit on this level    
    
    # Interact qubits from second and third levels
    for idx, jdx in zip(second_lvl_qubits, third_lvl_qubits):
        qml.CNOT(wires=[idx,jdx])
        
    # Third level
    for idx in third_lvl_qubits:
        qml.RY(weights[idx], wires=idx)  # parameterized rotations on each qubit 
    for idx, jdx in combinations(third_lvl_qubits, 2):
        qml.CNOT(wires=[idx,jdx])  # interact each qubit on this level
        
    # Interact qubits from third level with peak
    for idx in third_lvl_qubits:
        qml.CNOT(wires=[idx, peak_qubit])
        
    return qml.expval(qml.PauliZ(wires=peak_qubit))

##############################################################################
# In order to train the circuit, we will need a cost function to analyze

data = tf.constant([-1.,1.,1.,-1], dtype=tf.float64)
init_weights = np.random.rand(16)
weights = tf.Variable(init_weights, dtype=tf.float64)
cost = lambda: tf.abs(circuit(weights, data) - tf.reduce_prod(data))
ds = [tf.constant([-1.,1.,1.,-1], dtype=tf.float64),
      tf.constant([1.,1.,1.,1], dtype=tf.float64)]


##############################################################################
# References
# ----------
#
# .. [#barredo2017]
#
#    Daniel Barredo, Vincent Lienhard, Sylvain de Leseleuc, Thierry Lahaye, and Antoine Browaeys.
#    "Synthetic three-dimensional atomic structures assembled atom by atom."
#    `arXiv:1712.02727
#    <https://arxiv.org/abs/1712.02727>`__, 2017. 
# 

