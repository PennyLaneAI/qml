"""
Generalization Performance of Quantum Metric Learning Classifiers (Ants & Bees Image Dataset without PCA)
======================================

.. meta::
    :property="og:description": This demonstration illustrates the idea of training
        a quantum embedding for metric learning. This technique is used to train
        a hybrid quantum-classical data embedding to classify images of ants and bees.
    :property="og:image": https://github.com/Rlag1998/QML_Generalization/blob/main/embedding_metric_learning/figures/All_Figures/1.1.2_train.png?raw=true

*Adapted from work authored by Maria Schuld and Aroosa Ijaz*

*Authors: Jonathan Kim and Stefan Bekiranov*

This tutorial uses the idea of quantum embeddings for metric learning presented in 
`Lloyd, Schuld, Ijaz, Izaac, Killoran (2020) <https://arxiv.org/abs/2001.03622>`_ 
by training a hybrid classical-quantum data embedding to classify images of ants and bees. 
Their example was inspired by `Mari et al. (2019) <https://arxiv.org/abs/1912.08278>`_ 
(see also this `tutorial <https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html>`_). 
Data preparation is handled by the ``antbees_original.py`` file in the 
`embedding_metric_learning folder <https://github.com/PennyLaneAI/qml/tree/master/demonstrations/embedding_metric_learning>`_.

More details on this topic can be found in the research paper, "Generalization Performance of Quantum Metric Learning Classifiers", 
currently in review with Biomolecules.

Illustrated below is the general circuit used.

|

.. figure:: ../demonstrations/embedding_metric_learning/classification.png
   :align: center
   :width: 90%
   
|

After all necessary data pre-processing steps, ``n`` input features are reduced via matrix multiplication 
to ``x_1``, ``x_2`` intermediate values, which are then fed into a quantum feature map consisting of ZZ 
entanglers, as well as RX and RY rotational gates. This results in ``2n + 12`` total parameters 
(``2n`` from the classical part, ``12`` from the quantum feature map) which are trained and updated over 
a set number of iterations, resulting in a trained embedding. The trained embedding is able to embed 
input datapoints in Hilbert space such that the Hilbert-Schmidt distance between datapoints of different 
classes is maximized. A linear decision boundary can then be drawn across the datapoints in Hilbert space, 
which corresponds to a complex decision boundary in classical space. This form of embedding training is 
known as Quantum Metric Learning.

When using the ImageNet Ants & Bees image dataset in this demo, we observe poor generalization. 
See the breast cancer dataset demo with just 30 features per sample for an example of good generalization.

Let us begin!
"""


######################################################################
# Setup
# ----
#
# The tutorial requires the following imports:

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pennylane as qml
from pennylane import numpy as np
from pennylane import RX, RY, RZ, CNOT

######################################################################
# The following random seed is used:

np.random.seed(seed=123)


######################################################################
# Embedding
# ----
#
# Quantum metric learning is used to train a quantum embedding, which is 
# used for classifying data. Quantum embeddings are learned by maximizing 
# Hilbert-Schmidt distance of datapoints from two classes. After training, 
# the datapoints of different clases become maximally separated in Hilbert 
# space. This results in a simple linear decision boundary in Hilbert space 
# which represents a complex decision boundary in the original feature space.
#
# A cost function is used to track the progress of the training; the lower 
# the cost function, the greater the class separation in Hilbert space.
#
# The model is ultimately optimized with the ``RMSPropOptimizer`` and data are 
# classified according to a KNN-style classifier.
#
# Below is the code that makes up the quantum feature map:

def feature_encoding_hamiltonian(features, wires):

    for idx, w in enumerate(wires):
        RX(features[idx], wires=w)


def ising_hamiltonian(weights, wires, l):

    # ZZ coupling
    CNOT(wires=[wires[1], wires[0]])
    RZ(weights[l, 0], wires=wires[0])
    CNOT(wires=[wires[1], wires[0]])
    # local fields
    for idx, w in enumerate(wires):
        RY(weights[l, idx + 1], wires=w)


def QAOAEmbedding(features, weights, wires):

    repeat = len(weights)
    for l in range(repeat):
        # apply alternating Hamiltonians
        feature_encoding_hamiltonian(features, wires)
        ising_hamiltonian(weights, wires, l)
    # repeat the feature encoding once more at the end
    feature_encoding_hamiltonian(features, wires)


######################################################################
# By default, the model has 1024 + 12 trainable parameters - 1024 for the 
# classical part of the model and 12 for the quantum part.
#
# The following datafiles were created by standardizing, normalizing and 
# passing images of ants and bees through a ResNet18 network (without its 
# final layer).
# The data preparation code used to create these files can be found in the 
# `embedding_metric_learning folder <https://github.com/PennyLaneAI/qml/tree/master/demonstrations/embedding_metric_learning>`_.

X = np.loadtxt("embedding_metric_learning/X_antbees.txt", ndmin=2)  # pre-prepared training inputs
Y = np.loadtxt("embedding_metric_learning/Y_antbees.txt")  # training labels
X_val = np.loadtxt(
    "embedding_metric_learning/X_antbees_test.txt", ndmin=2
)  # pre-prepared validation inputs
Y_val = np.loadtxt("embedding_metric_learning/Y_antbees_test.txt")  # validation labels

# split data into two classes
A = X[Y == -1]  # bees
B = X[Y == 1]  # ants
A_val = X_val[Y_val == -1]
B_val = X_val[Y_val == 1]

print(A.shape)
print(B.shape)
print(A_val.shape)
print(B_val.shape)


######################################################################
# Quantum node initialization:

n_features = 2
n_qubits = 2 * n_features + 1

dev = qml.device("default.qubit", wires=n_qubits)


######################################################################
# SWAP test for overlap measurement:

x1list = []
x2list = []


@qml.qnode(dev)
def swap_test(q_weights, x1, x2):

    # load the two inputs into two different registers
    QAOAEmbedding(features=x1, weights=q_weights, wires=[1, 2])
    QAOAEmbedding(features=x2, weights=q_weights, wires=[3, 4])

    # perform the SWAP test
    qml.Hadamard(wires=0)
    for k in range(n_features):
        qml.CSWAP(wires=[0, k + 1, 2 + k + 1])
    qml.Hadamard(wires=0)

    return qml.expval(qml.PauliZ(0))


def overlaps(weights, X1=None, X2=None):

    linear_layer = weights[0]
    q_weights = weights[1]

    overlap = 0
    for x1 in X1:
        for x2 in X2:
            # multiply the inputs with the linear layer weight matrix
            w_x1 = linear_layer @ x1
            w_x2 = linear_layer @ x2
            # overlap of embedded intermediate features
            overlap += swap_test(q_weights, w_x1, w_x2)

    mean_overlap = overlap / (len(X1) * len(X2))

    return mean_overlap


######################################################################
# Below is the cost function, which takes both inter-cluster overlaps and intra-
# cluster overlaps into consideration:

def cost(weights, A=None, B=None):

    aa = overlaps(weights, X1=A, X2=A)
    bb = overlaps(weights, X1=B, X2=B)
    ab = overlaps(weights, X1=A, X2=B)

    d_hs = -2 * ab + (aa + bb)

    return 1 - 0.5 * d_hs


######################################################################
# Optimization
# ------------
#
# The intial classical and quantum parameters are generated at random.
#
# The lattermost integer belonging to the ``size`` attribute of the 
# ``init_pars_classical`` variable is set to 512, which corresponds to 
# the 512 output features of the ResNet (as generated by the corresponding 
# data preparation file in the `embedding_metric_learning folder <https://github.com/PennyLaneAI/qml/tree/master/demonstrations/embedding_metric_learning>`_).

# generate initial parameters for the quantum component, such that
# the resulting number of trainable quantum parameters is equal to
# the product of the elements that make up the 'size' attribute
# (4 * 3 = 12).
init_pars_quantum = np.random.normal(loc=0, scale=0.1, size=(4, 3))

# generate initial parameters for the classical component, such that
# the resulting number of trainable classical parameters is equal to
# the product of the elements that make up the 'size' attribute
# (2 * 512 = 1024).
init_pars_classical = np.random.normal(loc=0, scale=0.1, size=(2, 512))

init_pars = [init_pars_classical, init_pars_quantum]


######################################################################
# The ``RMSPropOptimizer`` is used with a step size of 0.01 and batch size 
# of 5 to optimize the model over 400 iterations. The ``pars`` variable 
# is updated after every iteration.
#
# .. note:: Despite the code steps shown below, all figure results in 
#           this demo were generated with a batch size of 10 over 1500 
#           iterations.

optimizer = qml.RMSPropOptimizer(stepsize=0.01)
batch_size = 5
pars = init_pars

cost_list = []
for i in range(400):

    # Sample a batch of training inputs from each class
    selectA = np.random.choice(range(len(A)), size=(batch_size,), replace=True)
    selectB = np.random.choice(range(len(B)), size=(batch_size,), replace=True)
    A_batch = [A[s] for s in selectA]
    B_batch = [B[s] for s in selectB]

    # Walk one optimization step
    pars = optimizer.step(lambda w: cost(w, A=A_batch, B=B_batch), pars)
    # print(pars)
    # print("Step", i+1 , "done.")

    # Print the validation cost every 10 steps
    # if i % 50 == 0 and i != 0:
    #    cst = cost(pars, A=A_val, B=B_val)
    #    print("Cost on validation set {:2f}".format(cst))
    #    cost_list.append(cst)


######################################################################
# The quantum and classical parameters are saved into txt files so 
# they may be used at a future time without having to re-train the 
# initial parameters.

print("quantum pars: ", pars[1])
with open(r"thetas.txt", "w") as file1:
    for item in pars[1]:
        file1.write("%s\n" % item)

print("classical pars: ", pars[0])
with open(r"x1x2.txt", "w") as file2:
    for item in pars[0]:
        file2.write("%s\n" % item)


######################################################################
# Analysis
# --------
#
# Hilbert space mutual data overlap gram matrices can be used to assess 
# the separation in embedded test set datapoints. Scatter plots 
# depicting the pre-training and post-training positions of the 
# ``x_1``, ``x_2`` intermediate points can also be plotted.
#
# For generating mutual data overlap gram matrices, a smaller subset of 
# the test set data is used, as determined by the ``select`` variable.

select = 10


######################################################################
# Final cost values can be printed out here:

# cost_train = cost(pars, A=A[:select], B=B[:select])
# cost_val = cost(pars, A=A_val[:select], B=B_val[:select])

# cost_train = cost(pars, A=A, B=B)
# cost_val = cost(pars, A=A_val, B=B_val)
# print("Cost for pretrained parameters on training set:", cost_train)
# print("Cost for pretrained parameters on validation set:", cost_val)


######################################################################
# Continuation of gram matrices preparation:

# A_B = np.r_[A[:select], B[:select]]
A_B = np.r_[A_val[:select], B_val[:select]]


######################################################################
# Before training, class separation is not observed within the gram matrices:

gram_before = [[overlaps(init_pars, X1=[x1], X2=[x2]) for x1 in A_B] for x2 in A_B]

ax = plt.subplot(111)
im = ax.matshow(gram_before, vmin=0, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
# plt.show()


######################################################################
#
# |
#
# .. figure:: ../demonstrations/embedding_metric_learning/figures/All_Figures/1.1.1.png
#   :align: center
#   :width: 90%
#
# |
#
# After training, the goal is for there to be a clear separation between 
# the two classes, such that there are four clearly defined squares of 
# mutual overlap (two yellow, two purple). However, it is clear that in this 
# case there is still a lack of test set separation after training.

gram_after = [[overlaps(pars, X1=[x1], X2=[x2]) for x1 in A_B] for x2 in A_B]

ax = plt.subplot(111)
im = ax.matshow(gram_after, vmin=0, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
# plt.show()


######################################################################
#
# |
#
# .. figure:: ../demonstrations/embedding_metric_learning/figures/All_Figures/1.1.2.png
#   :align: center
#   :width: 90%
#
# |
#
# The two-dimensional intermediate (``x_1``, ``x_2``) points can be graphed in the 
# form of scatter plots to help visualize the separation progress from 
# a different perspective.
#
# The code below results in the pre-training scatter plot:

red_patch = mpatches.Patch(color="red", label="Training: Bee")
blue_patch = mpatches.Patch(color="blue", label="Training: Ant")
lightcoral_patch = mpatches.Patch(color="lightcoral", label="Test: Bee")
cornflowerblue_patch = mpatches.Patch(color="cornflowerblue", label="Test: Ant")
plt.rcParams["figure.figsize"] = (8, 8)  ###
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)

for b in B:
    intermediate_b = init_pars[0] @ b
    plt.scatter(intermediate_b[:][0], intermediate_b[:][1], c="blue")

for a in A:
    intermediate_a = init_pars[0] @ a
    plt.scatter(intermediate_a[:][0], intermediate_a[:][1], c="red")

for b in B_val:
    intermediate_b = init_pars[0] @ b
    plt.scatter(intermediate_b[:][0], intermediate_b[:][1], c="cornflowerblue")

for a in A_val:
    intermediate_a = init_pars[0] @ a
    plt.scatter(intermediate_a[:][0], intermediate_a[:][1], c="lightcoral")

plt.xlabel(r"$x_1$", fontsize=20)
plt.ylabel(r"$x_2$", fontsize=20)
plt.legend(handles=[blue_patch, cornflowerblue_patch, red_patch, lightcoral_patch], fontsize=12)
# plt.show()

######################################################################
#
# |
#
# .. figure:: ../demonstrations/embedding_metric_learning/figures/All_Figures/1.1.3.png
#   :align: center
#   :width: 90%
#
# |
#
# The below code results in the post-training scatter plot. 
# It is clear only the training set intermediate values are 
# separated well. There are too many features in image 
# data, resulting in poor generalization due to overfitting.

for b in B:
    intermediate_b = pars[0] @ b
    plt.scatter(intermediate_b[:][0], intermediate_b[:][1], c="blue")

for a in A:
    intermediate_a = pars[0] @ a
    plt.scatter(intermediate_a[:][0], intermediate_a[:][1], c="red")

for b in B_val:
    intermediate_b = pars[0] @ b
    plt.scatter(intermediate_b[:][0], intermediate_b[:][1], c="cornflowerblue")

for a in A_val:
    intermediate_a = pars[0] @ a
    plt.scatter(intermediate_a[:][0], intermediate_a[:][1], c="lightcoral")

plt.xlabel(r"$x_1$", fontsize=20)
plt.ylabel(r"$x_2$", fontsize=20)
plt.legend(handles=[blue_patch, cornflowerblue_patch, red_patch, lightcoral_patch], fontsize=12)
# plt.show()

######################################################################
#
# |
#
# .. figure:: ../demonstrations/embedding_metric_learning/figures/All_Figures/1.1.4.png
#   :align: center
#   :width: 90%
#
# |
#

######################################################################
# Classification
# --------------
#
# A KNN-style classifier can be used to determine the class for each new 
# datapoint based on the datapoint's degree of overlap with each of the two 
# separated classes of the training set data.
#
# Below, test set classification is evaluated by means of a ``predict`` 
# function to yield subsequent F1, precision, recall, accuracy and specificity 
# scores. A confusion matrix of the form (TP, FN, FP, TN) is also returned.

def predict(n_samples, pred_low, pred_high, choice):

    truepos = 0
    falseneg = 0
    falsepos = 0
    trueneg = 0

    for i in range(pred_low, pred_high):
        pred = ""
        if choice == 0:
            x_new = A_val[i]  # Bee
        else:
            x_new = B_val[i]  # Ant

        prediction = 0
        for s in range(n_samples):

            # select a random sample from the training set
            sample_index = np.random.choice(len(X))
            x = X[sample_index]
            y = Y[sample_index]

            # compute the overlap between training sample and new input
            overlap = overlaps(pars, X1=[x], X2=[x_new])

            # add the label weighed by the overlap to the prediction
            prediction += y * overlap

        # normalize prediction
        prediction = prediction / n_samples

        # This component acts as the sign function of this KNN-style method.
        # 'Negative' predictions correspond to bees, while 'positive' predictions
        # correspond to ants. The confusion matrix is also constructed here.
        if prediction < 0:
            pred = "Bee"
            if choice == 0:
                truepos += 1
            else:
                falsepos += 1

        else:
            pred = "Ant"
            if choice == 0:
                falseneg += 1
            else:
                trueneg += 1
        # print("prediction: "+str(pred)+", value is "+str(prediction))

    # print(truepos, falseneg, falsepos, trueneg)
    return truepos, falseneg, falsepos, trueneg


totals = [x + y for x, y in zip(predict(20, 0, len(A_val), 0), predict(20, 0, len(B_val), 1))]
print(totals)
precision = totals[0] / (totals[0] + totals[2])
recall = totals[0] / (totals[0] + totals[1])
accuracy = (totals[0] + totals[3]) / (totals[0] + totals[1] + totals[2] + totals[3])
specificity = totals[3] / (totals[3] + totals[2])

f1 = (2 * precision * recall) / (precision + recall)
print("Precision: ", precision)
print("Recall: ", recall)
print("Accuracy: ", accuracy)
print("Specificity: ", specificity)
print("F1 Score: ", f1)


######################################################################
# References
# ----------
#
# Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, Nathan Killoran: "Quantum embeddings for machine learning" 
# arXiv preprint arXiv:2001.03622.
#
# Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, Nathan Killoran: "Transfer learning 
# in hybrid classical-quantum neural networks" arXiv preprint arXiv:1912.08278
#
# Jonathan Kim and Stefan Bekiranov: "Generalization of Quantum Metric Learning Classifiers" 
# submitted to Biomolecules, currently in review stage.
