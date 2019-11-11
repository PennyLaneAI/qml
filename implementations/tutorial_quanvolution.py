r"""
.. _quanvolution:

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


##############################################################################
# Setting of the main hyper-parameters of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

n_epochs = 30   # Number of optimization epochs
eta = 0.05      # Learning rate
n_gates = 4     # Number of random gates
n_train = 50    # Size of train dataset
n_test = 30     # Size of test dataset

SAVE_PATH = 'quanvolution/' # Data saving folder
PREPROCESS = True           # If False, load data from SAVE_PATH
np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator

##############################################################################
# We import the MNIST dataset from *Keras*.

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Reduce size of dataset
train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

# Normalize pixel values from 0 to 1
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
    RandomLayer(list(range(n_gates)), wires=list(range(4)), seed=0)
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

if PREPROCESS == True:
    q_train_images = []
    print('Quantum preprocessing of train images:')
    for idx, img in enumerate(train_images):
        print('{}/{}        '.format(idx + 1, n_train), end='\r')
        q_train_images.append(quanv(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print('\nQuantum preprocessing of test images:')
    for idx, img in enumerate(test_images):
        print('{}/{}        '.format(idx + 1, n_test), end='\r')
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
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation='softmax')
])

q_model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

c_model = keras.models.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation='softmax')
])

c_model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

##############################################################################
# Training
# --------

q_history = q_model.fit(q_train_images, train_labels, validation_data=(q_test_images, test_labels), batch_size=4, epochs=n_epochs, verbose=2)
c_history = c_model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size=4, epochs=n_epochs, verbose=2)


##############################################################################
# Results
# -------

import matplotlib.pyplot as plt

plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

ax1.plot(q_history.history['val_accuracy'], "-ob", label="With quantum layer")
ax1.plot(c_history.history['val_accuracy'], "-og", label="Without quantum layer")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(q_history.history['val_loss'],  "-ob", label="With quantum layer")
ax2.plot(c_history.history['val_loss'],  "-og", label="Without quantum layer")
ax2.set_ylabel("Loss")
ax2.set_ylim(top=2.5)
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.show()


##############################################################################
# References
# ----------
#
# 1. Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook. 
#    "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits."
#    `arXiv:1904.04767 <https://arxiv.org/abs/1904.04767>`__, 2019.
# 
