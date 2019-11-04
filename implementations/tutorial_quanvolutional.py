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
rng_seed = 0    # Seed for random number generator

##############################################################################
# We import the MNIST dataset from *Keras*.

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Normalize pixel values from 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=num_epochs)

predictions = model.predict(test_images)

print('Label 0:', test_labels[0])
print('Prediction 0:', np.argmax(predictions[0]))


##############################################################################
# References
# ----------
#
# 1. Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook. 
#    "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits."
#    `arXiv:1904.04767 <https://arxiv.org/abs/1904.04767>`__, 2019.
# 
