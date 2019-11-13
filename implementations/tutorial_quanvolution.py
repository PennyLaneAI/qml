r"""
.. _quanvolution:

Quanvolutional Neural Networks
==============================
*Author: Andrea Mari*

In this tutorial we implement the *Quanvolutional Neural Network*, a quantum
machine learning model originally introduced in
`Henderson et al. (2019) <https://arxiv.org/abs/1904.04767>`_.

.. figure:: ../implementations/quanvolution/circuit.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

Introduction
------------

Classical convolution
^^^^^^^^^^^^^^^^^^^^^
The *convolutional neural network* (CNN) is a standard model in classical machine learning which is particularely
suitable for processing images.
The model is based on the idea of a *convolution layer* in which, instead of processing the full input data with a global function, 
a local convolution is applied. 

For example, if the input is an image, only small regions are sequentially processed with the same linear kernel (which can be followed by
standard activation functions). The results obtained for each region are usually associated to different channels
of a single output pixel. The union of all the output pixels results in a new image-like object, which can be further processed by
additional layers.


Quantum convolution
^^^^^^^^^^^^^^^^^^^
One can extend the same idea also to the context of quantum variational circuits. 
Given an input image, a small region can be embedded into a quantum circuit
producing :math:`n_c` classical results which will represent :math:`n_c` different channels the output pixel.
Iterating the same procedure over many regions, one can scan the full input image, 
producing a new image-like object. 

The main difference with respect to a classical convolution is that a quantum circuit can 
generate highly complex kernels whose computation could be, at least in principle, classically intractable.

Quantum convolutions can be easily combined with classical layers, obtaining a *hybrid network*. 
In this tutorial we follow the approach of Ref. [1] in which a fixed non-trainable quantum
circuit is used as a "quanvolutional" layer, while subsequent classical layers 
are trained for a specific task.
On the other had, by leveraging the PennyLane capability of evaluating gradients of 
quantum circuits, it should be relatively easy to implement quantum convolution layers 
which can be variationally trained.


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
        for k in range(0, 28, 2):
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
# Let us visualize the effect of the quantum convolution layer on a few samples

n_samples = 4
n_channels = 4
fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
for k in range(n_samples):
    axes[0, 0].set_ylabel('Input')
    if k != 0:
        axes[0, k].yaxis.set_visible(False)
    axes[0, k].imshow(train_images[k, :, :, 0], cmap="gray")

    # Plot all output channels
    for c in range(n_channels):
        axes[c + 1, 0].set_ylabel('Output [ch. {}]'.format(c))
        if k != 0:
            axes[c, k].yaxis.set_visible(False)
        axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")
        
plt.tight_layout()
plt.show()

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
