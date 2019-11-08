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

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import RandomLayer

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import time 
init_time = time.time()

##############################################################################
# Setting of the main hyper-parameters of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_epochs = 40   # Number of optimization epochs
eta = 0.03      # Learning rate
n_gates = 8     # Number of random gates
n_train = 50    # Size of train and test datasets
n_test = 30     # Size of train and test datasets


SAVE_PATH = 'quanvolution/'
tf.keras.backend.set_floatx('float64')
np.random.seed(0)      # Seed for NumPy random number generator
tf.random.set_seed(0)  # Seed for TensorFlow random number generator

##############################################################################
# We import the MNIST dataset from *Keras*.

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Reduce size of dataset
train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

# Normalize pixel values from 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add extra dimension for convolution channels
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]


##############################################################################
# Quantum circuit


dev = qml.device('default.qubit', wires=4)
@qml.qnode(dev)
def circuit(phi=None):
    # Encoding of 4 classical input values
    qml.RY(np.pi * phi[0], wires=0)
    qml.RY(np.pi * phi[1], wires=1)
    qml.RY(np.pi * phi[2], wires=2)
    qml.RY(np.pi * phi[3], wires=3)
    # Random quantum circuit
    RandomLayer(list(range(n_gates)), wires=list(range(4)), seed=42)
    # Measurement producing 4 classical output values
    return (
        qml.expval(qml.PauliZ(0)), 
        qml.expval(qml.PauliZ(1)), 
        qml.expval(qml.PauliZ(2)), 
        qml.expval(qml.PauliZ(3))
    )


def quanv(image):
    'Convolves the input image with many applications of the same quantum circuit.'
    out = np.zeros((14, 14, 4))
    # Loop over input image coordinates
    for j in range(0, 28, 2):
        for k in range(2, 28, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(phi=[image[j, k, 0], image[j, (k + 1) % 28, 0], image[(j + 1) % 28, k, 0], image[(j+ 1) % 28, (k + 1) % 28, 0]])
            # Assign quantum expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out


q_train_images = []
print('Quantum preprocessing of train images:')
for idx, img in enumerate(train_images):
    print('{}/{}        '.format(idx +1,n_train), end='\r')
    q_train_images.append(quanv(img))
q_train_images = np.asarray(q_train_images)

q_test_images = []
print('\nQuantum preprocessing of test images:')
for idx, img in enumerate(test_images):
    print('{}/{}        '.format(idx +1,n_test), end='\r')
    q_test_images.append(quanv(img))
q_test_images = np.asarray(q_test_images)

# Save pre-processed images
np.save(SAVE_PATH + 'q_train_images.npy', q_train_images) 
np.save(SAVE_PATH + 'q_test_images.npy', q_test_images) 


# Load pre-processed images
q_train_images = np.load(SAVE_PATH + 'q_train_images.npy') 
q_test_images = np.load(SAVE_PATH + 'q_test_images.npy') 

##############################################################################
# Hybrid Model

q_model = keras.models.Sequential([
  #tf.keras.layers.MaxPool2D(2),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

q_model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

c_model = keras.models.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

c_model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
q_history = q_model.fit(q_train_images, train_labels, validation_data=(q_test_images, test_labels), batch_size=4, epochs=n_epochs)
c_history = c_model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size=4, epochs=n_epochs)

print('Training completed in {} seconds.'.format(time.time() - init_time))

plt.style.use("seaborn")
plt.plot(q_history.history['val_accuracy'], "b", label="Hybrid")
plt.plot(c_history.history['val_accuracy'], "g", label="Classical")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.style.use("seaborn")
plt.plot(q_history.history['val_loss'], "b", label="Hybrid")
plt.plot(c_history.history['val_loss'], "g", label="Classical")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()




##############################################################################
# References
# ----------
#
# 1. Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook. 
#    "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits."
#    `arXiv:1904.04767 <https://arxiv.org/abs/1904.04767>`__, 2019.
# 
