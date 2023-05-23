"""
Turning quantum nodes into Keras Layers
=======================================

.. meta::
    :property="og:description": Learn how to create hybrid ML models in PennyLane using Keras
    :property="og:image": https://pennylane.ai/qml/_images/Keras_logo.png

.. related::

   tutorial_qnn_module_torch Turning quantum nodes into Torch Layers

*Author: Tom Bromley — Posted: 02 November 2020. Last updated: 28 January 2021.*

Creating neural networks in `Keras <https://keras.io/>`__ is easy. Models are constructed from
elementary *layers* and can be trained using a high-level API. For example, the following code
defines a two-layer network that could be used for binary classification:
"""

import tensorflow as tf

tf.keras.backend.set_floatx('float64')

layer_1 = tf.keras.layers.Dense(2)
layer_2 = tf.keras.layers.Dense(2, activation="softmax")

model = tf.keras.Sequential([layer_1, layer_2])
model.compile(loss="mae")

###############################################################################
# The model can then be trained using `model.fit()
# <https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`__.
#
# **What if we want to add a quantum layer to our model?** This is possible in PennyLane:
# :doc:`QNodes <../glossary/hybrid_computation>` can be converted into Keras layers and combined
# with the wide range of built-in classical
# `layers <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__ to create truly hybrid
# models. This tutorial will guide you through a simple example to show you how it's done!
#
# .. note::
#
#     A similar demo explaining how to
#     :doc:`turn quantum nodes into Torch layers <tutorial_qnn_module_torch>`
#     is also available.
#
# Fixing the dataset and problem
# ------------------------------
#
# Let us begin by choosing a simple dataset and problem to allow us to focus on how the hybrid
# model is constructed. Our objective is to classify points generated from scikit-learn's
# binary-class
# `make_moons() <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html>`__ dataset:

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

X, y = make_moons(n_samples=200, noise=0.1)
y_hot = tf.keras.utils.to_categorical(y, num_classes=2)  # one-hot encoded labels

c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in y]  # colours for each class
plt.axis("off")
plt.scatter(X[:, 0], X[:, 1], c=c)
plt.show()

###############################################################################
# Defining a QNode
# ----------------
#
# Our next step is to define the QNode that we want to interface with Keras. Any combination of
# device, operations and measurements that is valid in PennyLane can be used to compose the
# QNode. However, the QNode arguments must satisfy additional :doc:`conditions
# <code/api/pennylane.qnn.KerasLayer>` including having an argument called ``inputs``. All other
# arguments must be arrays or tensors and are treated as trainable weights in the model. We fix a
# two-qubit QNode using the
# :doc:`default.qubit <code/api/pennylane.devices.default_qubit.DefaultQubit>` simulator and
# operations from the :doc:`templates <introduction/templates>` module.

import pennylane as qml

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

###############################################################################
# Interfacing with Keras
# ----------------------
#
# With the QNode defined, we are ready to interface with Keras. This is achieved using the
# :class:`~pennylane.qnn.KerasLayer` class of the :mod:`~pennylane.qnn` module, which converts the
# QNode to the elementary building block of Keras: a *layer*. We shall see in the following how the
# resultant layer can be combined with other well-known neural network layers to form a hybrid
# model.
#
# We must first define the ``weight_shapes`` dictionary. Recall that all of
# the arguments of the QNode (except the one named ``inputs``) are treated as trainable
# weights. For the QNode to be successfully converted to a layer in Keras, we need to provide the
# details of the shape of each trainable weight for them to be initialized. The ``weight_shapes``
# dictionary maps from the argument names of the QNode to corresponding shapes:

n_layers = 6
weight_shapes = {"weights": (n_layers, n_qubits)}

###############################################################################
# In our example, the ``weights`` argument of the QNode is trainable and has shape given by
# ``(n_layers, n_qubits)``, which is passed to
# :func:`~pennylane.templates.layers.BasicEntanglerLayers`.
#
# Now that ``weight_shapes`` is defined, it is easy to then convert the QNode:

qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

###############################################################################
# With this done, the QNode can now be treated just like any other Keras layer and we can proceed
# using the familiar Keras workflow.
#
# Creating a hybrid model
# -----------------------
#
# Let's create a basic three-layered hybrid model consisting of:
#
# 1. a 2-neuron fully connected classical layer
# 2. our 2-qubit QNode converted into a layer
# 3. another 2-neuron fully connected classical layer
# 4. a softmax activation to convert to a probability vector
#
# A diagram of the model can be seen in the figure below.
#
# .. figure:: /demonstrations/qnn_module/qnn_keras.png
#    :width: 100%
#    :align: center
#
# We can construct the model using the
# `Sequential <https://www.tensorflow.org/api_docs/python/tf/keras/Sequential>`__ API:

clayer_1 = tf.keras.layers.Dense(2)
clayer_2 = tf.keras.layers.Dense(2, activation="softmax")
model = tf.keras.models.Sequential([clayer_1, qlayer, clayer_2])

###############################################################################
# Training the model
# ------------------
#
# We can now train our hybrid model on the classification dataset using the usual Keras
# approach. We'll use the
# standard `SGD <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>`__ optimizer
# and the mean absolute error loss function:

opt = tf.keras.optimizers.SGD(learning_rate=0.2)
model.compile(opt, loss="mae", metrics=["accuracy"])

###############################################################################
# Note that there are more advanced combinations of optimizer and loss function, but here we are
# focusing on the basics.
#
# The model is now ready to be trained!

fitting = model.fit(X, y_hot, epochs=6, batch_size=5, validation_split=0.25, verbose=2)

###############################################################################
# How did we do? The model looks to have successfully trained and the accuracy on both the
# training and validation datasets is reasonably high. In practice, we would aim to push the
# accuracy higher by thinking carefully about the model design and the choice of hyperparameters
# such as the learning rate.
#
# Creating non-sequential models
# ------------------------------
#
# The model we created above was composed of a sequence of classical and quantum layers. This
# type of model is very common and is suitable in a lot of situations. However, in some cases we
# may want a greater degree of control over how the model is constructed, for example when we
# have multiple inputs and outputs or when we want to distribute the output of one layer into
# multiple subsequent layers.
#
# Suppose we want to make a hybrid model consisting of:
#
# 1. a 4-neuron fully connected classical layer
# 2. a 2-qubit quantum layer connected to the first two neurons of the previous classical layer
# 3. a 2-qubit quantum layer connected to the second two neurons of the previous classical layer
# 4. a 2-neuron fully connected classical layer which takes a 4-dimensional input from the
#    combination of the previous quantum layers
# 5. a softmax activation to convert to a probability vector
#
# A diagram of the model can be seen in the figure below.
#
# .. figure:: /demonstrations/qnn_module/qnn2_keras.png
#    :width: 100%
#    :align: center
#
# This model can also be constructed using the `Functional API
# <https://keras.io/guides/functional_api/>`__:

# re-define the layers
clayer_1 = tf.keras.layers.Dense(4)
qlayer_1 = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
qlayer_2 = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
clayer_2 = tf.keras.layers.Dense(2, activation="softmax")

# construct the model
inputs = tf.keras.Input(shape=(2,))
x = clayer_1(inputs)
x_1, x_2 = tf.split(x, 2, axis=1)
x_1 = qlayer_1(x_1)
x_2 = qlayer_2(x_2)
x = tf.concat([x_1, x_2], axis=1)
outputs = clayer_2(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

###############################################################################
# As a final step, let's train the model to check if it's working:

opt = tf.keras.optimizers.SGD(learning_rate=0.2)
model.compile(opt, loss="mae", metrics=["accuracy"])

fitting = model.fit(X, y_hot, epochs=6, batch_size=5, validation_split=0.25, verbose=2)

###############################################################################
# Great! We've mastered the basics of constructing hybrid classical-quantum models using
# PennyLane and Keras. Can you think of any interesting hybrid models to construct? How do they
# perform on realistic datasets?

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/thomas_bromley.txt