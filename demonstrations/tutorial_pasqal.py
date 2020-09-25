r"""
Quantum computation with neutral atoms
======================================

.. meta::
    :property="og:description": Neutral atom quantum devices allow you to place 
        qubits within interesting three-dimensional configurations.
    :property="og:image": https://pennylane.ai/qml/_images/pasqal_thumbnail.png

Quantum computing architectures come in many flavours: superconducting qubits, ion traps, 
photonics, silicon, and more. One very interesting physical substrate is *neutral atoms*. These 
quantum devices have some basic similarities to ion traps. Ion-trap devices make use of atoms 
that have an imbalance between protons (positively charged) and electrons (negatively charged). 
Neutral atoms, on the other hand, have an equal number of protons and electrons. 

Uniquely, in neutral-atom systems, lasers can be used to arrange atoms in various two- or 
three-dimensional configurations. This opens up some tantalizing possibilities for 
exotic quantum-computing circuit topologies.

.. figure:: https://raw.githubusercontent.com/lhenriet/cirq-pasqal/fc4f9c7792a8737fde76d4c05828aa538be8452e/pasqal-tutorials/files/eiffel_tower.png
    :align: center
    :width: 50%
    
    ..
    
    Neutral atoms (green dots) arranged in various configurations. Image originally from [#barredo2017]_.
    
The startup `Pasqal <https://pasqal.io/>`_ is one of the companies working to bring 
neutral-atom quantum computing devices to the world. To support this new class of devices, 
Pasqal has contributed some new features to the quantum software library Cirq.
 
In this demo, we will use PennyLane, Cirq, and TensorFlow to show off the unique abilities of 
neutral atom devices, leveraging them to make a quantum machine learning circuit which has a
very unique topology: *the Eiffel tower*. 

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
# Building the Eiffel tower
# -------------------------
#
# Our first step will be to load and visualize the data for the Eiffel tower 
# configuration, generously provided by the team at Pasqal.

coords = np.loadtxt("pasqal/Eiffel_tower_data.dat")
xs = coords[:,0]
ys = coords[:,1]
zs = coords[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(40, 15)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(-40, 10)
plt.axis('off')
ax.scatter(xs, ys, zs, c='g',alpha=0.3)
#plt.show();

##############################################################################
# This dataset contains 126 points. Each point represents a distinct 
# neutral-atom qubit. This is outside the reach of any quantum
# simulator, so for the purposes of this demo, 
# we will pare down to just 13 points, evenly spaced around the tower.
# These are highlighted in red below:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(40, 15)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(-40, 10)
plt.axis('off')
ax.scatter(xs, ys, zs, c='g',alpha=0.3)

mask = [3,7,11,15,48,51,60,63,96,97,98,99,125]
subset_coords = coords[mask]

xs = subset_coords[:,0]
ys = subset_coords[:,1]
zs = subset_coords[:,2]
ax.scatter(xs, ys, zs, c='r',alpha=1.0)
#plt.show();

##############################################################################
# Converting to Cirq qubits
# -------------------------
#
# Our next step will be to convert these datapoints into objects that 
# Cirq understands as qubits. For neutral-atom devices in Cirq, we can use the
# ``ThreeDQubit`` class, which carries information about the three-dimensional
# arrangement of the qubits.
#
# Now, neutral atom devices come with some physical restrictions. 
# Specifically, qubits in a particular configuration can't be arbitrarily  
# interacted with one another. Instead, there is the notion of a 
# *control radius*; any atoms which are within the system's control radius 
# can interact with one another. Qubits separated by a distance larger than
# the control radius cannot interact.
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
qubits = base_qubits[4:] + tower_qubits

print("ThreeDQubits:")
print("\n".join(str(q) for q in qubits))
print("\n".join(str(q) for q in sorted(qubits)))


##############################################################################
# To simulate a neutral-atom quantum computation, we can use the 
# ``"cirq.pasqal"`` device, available via the 
# `PennyLane-Cirq plugin <https://pennylane-cirq.readthedocs.io>`_.
# We will need to provide this device with the ``ThreeDQubit``s we created 
# above. We also need to instantiate the device with a fixed control radius.

num_wires = len(qubits)
control_radius = 32.4
dev = qml.device("cirq.pasqal", control_radius=control_radius, 
                 qubits=qubits, wires=num_wires)

##############################################################################
# Creating a quantum circuit
# --------------------------
# 
# We can now make a quantum computing circuit out of the Eiffel tower configuration
# from above. Each of the 13 qubits we are using can be thought of
# as a single wire in a quantum circuit. Our circuit will consist of several
# stages:
#
# i. Data is fed in to the first vertical level of qubits ("the ground") using
#    parametrized single-qubit Pauli-Y rotations (a simple data-embedding 
#    strategy).
#
# ii. Two-qubit interactions (in this case, CNOTs) are carried out
#     between each ground-level qubit and the corresponding 
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
# That's a lot to keep track of, so let's show the circuit via a 
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
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(-40, 10)
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

# Two-qubit gates between all pairs at second level
for idx in range(4):
    ax.plot(xs=[second_x[idx], second_x[(idx + 1) % 4]],
            ys=[second_y[idx], second_y[(idx + 1) % 4]],
            zs=[second_z[idx], second_z[(idx + 1) % 4]],
            c='k');
# Two-qubit gates between all pairs at third level
    ax.plot(xs=[third_x[idx], third_x[(idx + 1) % 4]],
            ys=[third_y[idx], third_y[(idx + 1) % 4]],
            zs=[third_z[idx], third_z[(idx + 1) % 4]],
            c='k');
#plt.show();

##############################################################################
# In this figure, the red dots represent our qubits, arranged in a 
# three-dimensional configuration roughly resembling the Eiffel tower.
# The black lines indicate CNOT gates between certain qubits.
# Data is loaded in at the bottom qubits (the "tower legs") and the final
# measurement result is read out from the top "peak" qubit.
# The order of gate execution proceeds vertically from bottom to top, and
# clockwise at each level.
# 
# The code below creates this particular quantum circuit configuration in 
# PennyLane:

second_lvl_qubits = range(0,4)
third_lvl_qubits = range(4,8)
#third_lvl_qubits = range(8,12)
peak_qubit = 8

def controlled_rotation(phi, wires):
    qml.RY(phi, wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(-phi, wires=wires[1])
    qml.CNOT(wires=wires)

@qml.qnode(dev, interface="tf")
def circuit(weights, data):
    
    # First level
    for idx in range(4):
        #qml.BasisState(data, wires=test)  # data loading
        if data[idx]:
            qml.PauliX(wires=idx)  # data loading
        #qml.RY(weights[idx], wires=idx)  # parameterized rotations on each qubit

    # Interact qubits from first and second levels
    #for idx, jdx in zip(first_lvl_qubits, second_lvl_qubits):
    #    qml.CNOT(wires=[idx,jdx])
    
    # Second level
    #for idx in second_lvl_qubits:
    #    qml.RY(weights[idx], wires=idx)  # parameterized rotations on each qubit 
    #for idx in range(4):
    #    #qml.SWAP(wires=[idx, (idx + 1) % 4])  # interact each qubit on this level
    #    controlled_rotation(weights[idx], wires=[idx, (idx + 1) % 4])    
    
    # Interact qubits from second and third levels
    for idx in range(4):
        qml.CNOT(wires=[idx, idx + 4])
        
    # Third level
    #for idx in third_lvl_qubits:
    #    qml.RY(weights[idx], wires=idx)  # parameterized rotations on each qubit 
    #for idx in range(4):
    #    #qml.SWAP(wires=[4 + idx, 4 + (idx + 1) % 4])  # interact each qubit on this level
    #    jdx = idx + 4
    #    if jdx == 7:
    #        controlled_rotation(weights[jdx], wires=[jdx, 4])  
    #    else: 
    #        controlled_rotation(weights[jdx], wires=[jdx, jdx + 1])
        
    # Interact qubits from third level with peak
    #for idx in range(4):
    #    #qml.SWAP(wires=[idx, peak_qubit])
    #    jdx = idx + 4
    #    kdx = idx + 8
    #    controlled_rotation(weights[kdx], wires=[jdx, peak_qubit])   
    
    controlled_rotation(weights[0], wires=[4, peak_qubit])
    controlled_rotation(weights[1], wires=[5, peak_qubit])
    controlled_rotation(weights[2], wires=[6, peak_qubit])
    controlled_rotation(weights[3], wires=[7, peak_qubit])
        
    return qml.expval(qml.PauliZ(wires=peak_qubit))
    
    
##############################################################################
# Training the circuit
# --------------------
# 
# In order to train our circuit, we will need a cost function to analyze. For
# the purposes of this demo, we will consider a very simple classifier: If 
# the first input qubit is in state :math:`\vert 0\rangle`, the model should make the 
# prediction "0", and if that qubit is in state :math:`\vert 1 \rangle`,
# the model should predict "1" (independent of what the other qubit 
# states are. In other words, the idealized trained model should learn an 
# identity transformation between the first qubit and the final one, while
# ignoring the states of the other qubits.

np.random.seed(143)
#init_weights = np.zeros(4) #np.random.rand(12)
#init_weights[0] = 0.5 * np.pi / 2
init_weights = np.pi * np.random.rand(4)

weights = tf.Variable(init_weights, dtype=tf.float64)

data = np.random.randint(0, 2, size=4)
circuit(init_weights, data)
print(circuit.draw())

def cost():
    data = np.random.randint(0, 2, size=4)
    label = data[0]
    output = (-circuit(weights, data) + 1) / 2
    print(label, data, output)
    return tf.abs(output - label) ** 2

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

cost_vals = []
for step in range(30):
    cost_vals.append(cost().numpy())
 
    opt.minimize(cost, weights)
    data = np.random.randint(0, 2, size=4)
    print("Step {}: cost={}".format(step, cost()))
    print("        weights={}".format(weights))
        
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(range(step+1), cost_vals)
plt.show()


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

