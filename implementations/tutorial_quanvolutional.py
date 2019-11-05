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
# Add a channels dimension
train_images_2 = train_images[..., tf.newaxis]
test_images_2 = test_images[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((train_images_2, train_labels)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((test_images_2, test_labels)).batch(32)


##############################################################################
# Quantum circuit


dev = qml.device('default.qubit', wires=2)
@qml.qnode(dev, interface='tf')
def circuit(phi):
    qml.RX(phi, wires=0)
    qml.RY(phi, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

q = tf.Variable(0.3, dtype=tf.float64)

##############################################################################
# Custom hybrid model

inputs = keras.Input(shape=(28, 28))   # Returns an input placeholder
x = keras.layers.Flatten()(inputs)
x = x * circuit(q)
predictions = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(train_images, train_labels, epochs=1)


##############################################################################
# References
# ----------
#
# 1. Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook. 
#    "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits."
#    `arXiv:1904.04767 <https://arxiv.org/abs/1904.04767>`__, 2019.
# 
