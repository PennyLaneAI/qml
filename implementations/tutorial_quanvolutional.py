r"""
.. _quanvolutional:

Quanvolutional Neural Networks
==============================
*Author: Andrea Mari*

In this tutorial we implement the *Quanvolutional Neural Network*, a quantum
machine learning model originally introduced in
`Henderson et al. (2019) <https://arxiv.org/abs/1904.04767>`_.



Introduction
------------

General setup
------------------------
This Python code requires *PennyLane* with the *TensorFlow* interface and the plotting library *matplotlib*.
"""

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Plotting
import matplotlib.pyplot as plt




##############################################################################
# Setting of the main hyper-parameters of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_qubits = 3    # Number of system qubits.
num_epochs = 1  # Number of optimization epochs
eta = 0.01      # Learning rate
rng_seed = 0    # Seed for random number generator
tf.keras.backend.set_floatx('float64')

##############################################################################
# We import the MNIST dataset from *Keras*.

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Normalize pixel values from 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add a dimension for convolution channels
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]


##############################################################################
# Quantum circuit


dev = qml.device('default.qubit', wires=4)
@qml.qnode(dev)
def circuit(phi):
    qml.RY(phi[0], wires=0)
    qml.RY(phi[1], wires=1)
    qml.RY(phi[2], wires=2)
    qml.RY(phi[3], wires=3)
    return (
    qml.expval(qml.PauliZ(0)), 
    qml.expval(qml.PauliZ(1)), 
    qml.expval(qml.PauliZ(2)), 
    qml.expval(qml.PauliZ(3))
    )


q = tf.Variable([0.3, 0.3, 0.3, 0.3], dtype=tf.float64)

def quanv(image):
    'Convolves the input image with many applications of the same quantum circuit.'
    out = np.zeros((28, 28, 4))
    # Loop over image coordinates
    for j in range(28):
        for k in range(28):
            # Process a 2X2 region of the image with a quantum circuit
            q_results = circuit([image[j, k, 0], image[j, k + 1, 0], image[j, k, 0], image[j, k, 0]])
            # Assign each expectation value to a different channel of the pixel (j, k)
            for c in range(4)
                out[j, k, c] = q_results[c]
    return out

img_in = train_images[7]
img_out = quanv(img_in)

print('in_shape', img_in.shape)
print('out_shape', img_out.shape)


plt.imshow(img_in[:,:,0],  cmap='gray')
plt.show()
plt.imshow(img_out[:,:,0],  cmap='gray')
plt.show()

"""
##############################################################################
# Custom hybrid model

inputs = keras.Input(shape=(28, 28, 1))   # Returns an input placeholder
x = quanv(inputs)
x = keras.layers.Flatten()(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(train_images, train_labels, epochs=1)

"""
##############################################################################
# References
# ----------
#
# 1. Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook. 
#    "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits."
#    `arXiv:1904.04767 <https://arxiv.org/abs/1904.04767>`__, 2019.
# 
