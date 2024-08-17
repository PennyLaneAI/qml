r"""Post-Variational Quantum Neural Networks
========================================
"""

######################################################################
# You're sitting in front of your quantum computer, excitement buzzing through your veins as your 
# carefully crafted Ansatz for a variational algorithm is finally ready. But oh buttersticks -— 
# after a few hundred iterations, your heart sinks as you realize you have encountered the dreaded barren plateau problem, where 
# gradients vanish and optimisation grinds to a halt. What now? Panic sets in, but then you remember the new technique 
# you read about. You reach into your toolbox and pull out the "post-variational strategy". This approach shifts 
# parameters from the quantum computer to classical computers, ensuring the convergence to a local minimum. By combining 
# fixed quantum circuits with a classical neural network, you can enhance trainability and keep your 
# research on track.
# 
# This tutorial introduces post-variational quantum neural networks with example code from PennyLane and Jax.
# We build variational and post-variational networks through a step-by-step process, and compare their 
# performance on the digits dataset. 
#
# 

######################################################################
# Background
# ---------------------
# Variational algorithms are proposed to solve optimization problems in chemistry, combinatorial
# optimization and machine learning, with potential quantum advantage. [#cerezo2021variational]_ Such algorithms often operate
# by first encoding data :math:`x` into a :math:`n`-qubit quantum state. The quantum state is then
# transformed by an Ansatz :math:`U(\theta)`. The parameters :math:`\theta` are optimized by
# evaluating gradients of the quantum circuit [#schuld2019evaluating]_ and calculating updates of the parameter on a classical
# computer. `Variational algorithms <https://pennylane.ai/qml/glossary/variational_circuit/>`__ are a pre-requisite to this article.
# 
# However, many Ansätze in the variational strategy face the barren plateau problem [#mcclean2018barren]_ , which leads to difficulty in convergence
# using gradient-based optimization techniques. Due general difficulty and lack of training gurantees
# of variational algorithms, we develop an alternative training strategy that does not involve tuning
# the quantum circuit parameters. However, we continue to use the variational method as the
# theoretical basis for optimisation.
# 
# We discuss “post-variational strategies” proposed in [#huang2024postvariational]_ . We take the classical combination of
# multiple fixed quantum circuits and find the optimal combination by feeding them through a classical linear model or feed the outputs to a
# multilayer perceptron. We shift tunable parameters from the quantum computer to the classical
# computer, opting for ensemble strategies when optimizing quantum models. This sacrifices
# expressibility [#du2020expressive]_  of the circuit for better trainability of the entire model. Below, we discuss various
# strategies and design principles for constructing individual quantum circuits, where the resulting
# ensembles can be optimized with classical optimisation methods.
# 

######################################################################
# |image1|
# 
# .. |image1| image:: ../_static/demonstration_assets/PVQNN/PVdrawing.jpeg
#    :width: 90.0%
# 

######################################################################
# We compare the post-variational strategies to the conventional variational quantum neural network in the
# table below.
# 

######################################################################
# |image2|
# 
# .. |image2| image:: ../_static/demonstration_assets/PVQNN/table.png
#    :width: 90.0%
# 

######################################################################
# This example demonstrates how to employ the post-variational quantum neural network on the classical
# machine learning task of image classification. Here, we solve the problem of identifying handwritten
# digits of twos and sixes and obtain training performance better than that of variational
# algorithms. This dataset is chosen such that the differences between variational and post variational 
# are shown, but we note that the performances may vary for different datasets. 
# 

######################################################################
# The Learning Problem
# ---------------------
# 

######################################################################
# We train our models on the digits dataset, which we import using `sklearn`. The dataset has greyscale
# images of size :math:`8\times 8` pixels. We partition :math:`10\%` of the dataset for
# testing.
# 

import pennylane as qml
from pennylane import numpy as np
import jax
from jax import numpy as jnp
import optax
from itertools import combinations
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42) 

# Load the digits dataset with features (X_digits) and labels (y_digits)
X_digits, y_digits = load_digits(return_X_y=True)

# Create a boolean mask to filter out only the samples where the label is 2 or 6
filter_mask = np.isin(y_digits, [2, 6])

# Apply the filter mask to the features and labels to keep only the selected digits
X_digits = X_digits[filter_mask]
y_digits = y_digits[filter_mask]

# Split the filtered dataset into training and testing sets with 10% of data reserved for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.1, random_state=42
)

# Normalize the pixel values in the training and testing data
# Convert each image from a 1D array to an 8x8 2D array, normalize pixel values, and scale them
X_train = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_train])
X_test = np.array([thing.reshape([8, 8]) / 16 * 2 * np.pi for thing in X_test])

# Adjust the labels to be centered around 0 and scaled to be in the range -1 to 1
# The original labels (2 and 6) are mapped to -1 and 1 respectively
y_train = (y_train - 4) / 2
y_test = (y_test - 4) / 2


######################################################################
# A visualization of a few data points are shown below.
# 

plt.figure()
for i in range(3,8):
    plt.subplot(1, 5, i-2)
    plt.matshow(X_train[i], fignum=False)
    plt.axis('off')
plt.show()

######################################################################
# Setting up the Model
# ---------------------
# 
# Here, we will create a simple QML model for optimization. In particular:
# 
# -  We will embed our data through a series of rotation gates, this is called the feature map.
# -  We will then have an Ansatz of rotation gates with parameters weights
# 

######################################################################
# For the feature map, each column of the image is encoded into a single qubit, and each row is
# encoded consecutively via alternating rotation-Z and rotation-X gates. The circuit for our feature
# map is shown below.
# 

######################################################################
# |image3|
# 
# .. |image3| image:: ../_static/demonstration_assets/PVQNN/featuremap.png
#    :width: 90.0%
# 

######################################################################
# We use the following circuit as our Ansatz. This Ansatz is also used as backbone for all our
# post-variational strategies. Note that when we set all initial parameters to 0, the Ansatz evaluates to
# identity. 
# 

######################################################################
# |image4|
# 
# .. |image4| image:: ../_static/demonstration_assets/PVQNN/ansatz.png
#    :width: 90.0%
# 

######################################################################
# We write code for the above Ansatz and feature map as follows.
# 


def feature_map(features):
    # Apply Hadamard gates to all qubits to create an equal superposition state
    for i in range(len(features[0])):
        qml.Hadamard(i)
    
    # Apply angle embeddings based on the feature values
    for i in range(len(features)):
        # For odd-indexed features, use Z-rotation in the angle embedding
        if i % 2:
            qml.AngleEmbedding(features=features[i], wires=range(8), rotation="Z")
        # For even-indexed features, use X-rotation in the angle embedding
        else:
            qml.AngleEmbedding(features=features[i], wires=range(8), rotation="X")

# Define the ansatz (quantum circuit ansatz) for parameterized quantum operations
def ansatz(params):
    # Apply RY rotations with the first set of parameters
    for i in range(8):
        qml.RY(params[i], wires=i)
    
    # Apply CNOT gates with adjacent qubits (cyclically connected) to create entanglement
    for i in range(8):
        qml.CNOT(wires=[(i - 1) % 8, (i) % 8])
    
    # Apply RY rotations with the second set of parameters
    for i in range(8):
        qml.RY(params[i + 8], wires=i)
    
    # Apply CNOT gates with qubits in reverse order (cyclically connected) to create additional entanglement
    for i in range(8):
        qml.CNOT(wires=[(8 - 2 - i) % 8, (8 - i - 1) % 8])
######################################################################
# Variational Algorithm
# ---------------------
# 

######################################################################
# As a baseline comparison, we first test the performance of a shallow variational algorithm on the
# digits dataset shown above. We will build the quantum node by combining the above feature map and
# Ansatz.
# 

dev = qml.device("default.qubit", wires=8)


@qml.qnode(dev)
def circuit(params, features):
    feature_map(features)
    ansatz(params)
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


def square_loss(labels, predictions):
    return np.mean((labels - qml.math.stack(predictions)) ** 2)


def accuracy(labels, predictions):
    acc = sum([np.sign(l) == np.sign(p) for l, p in zip(labels, predictions)])
    acc = acc / len(labels)
    return acc


def cost(params, X, Y):
    predictions = [variational_classifier(params["weights"], params["bias"], x) for x in X]
    return square_loss(Y, predictions)


def acc(params, X, Y):
    predictions = [variational_classifier(params["weights"], params["bias"], x) for x in X]
    return accuracy(Y, predictions)


np.random.seed(0)
weights = 0.01 * np.random.randn(16)
bias = jnp.array(0.0)
params = {"weights": weights, "bias": bias}
opt = optax.adam(0.05)
batch_size = 7
num_batch = X_train.shape[0] // batch_size
opt_state = opt.init(params)
X_batched = X_train.reshape([-1, batch_size, 8, 8])
y_batched = y_train.reshape([-1, batch_size])


@jax.jit
def update_step_jit(i, args):
    params, opt_state, data, targets, batch_no = args
    _data = data[batch_no % num_batch]
    _targets = targets[batch_no % num_batch]
    _, grads = jax.value_and_grad(cost)(params, _data, _targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return (params, opt_state, data, targets, batch_no + 1)


@jax.jit
def optimization_jit(params, data, targets):
    opt_state = opt.init(params)
    args = (params, opt_state, data, targets, 0)
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, 200, update_step_jit, args)
    return params


params = optimization_jit(params, X_batched, y_batched)
var_train_acc = acc(params, X_train, y_train)
var_test_acc = acc(params, X_test, y_test)

print("Training accuracy: ", var_train_acc)
print("Testing accuracy: ", var_test_acc)

######################################################################
# In this example, the variational algorithm is having trouble finding a global minimum (and this
# problem persists even if we do hyperparameter tuning). In the following code, we can see how this
# performance compares to our other proposed strategies.
# 

######################################################################
# The Observable Construction Post-Variational Technique
# ---------------------

######################################################################
# We measure the data embedded state on different combinations of Pauli observables in this
# post-variational strategy. We first define a series of :math:`k`-local trial observables
# :math:`O_1, O_2, \ldots , O_m`. After computing the quantum circuits, the measurement results are
# then combined classically, where the optimal weights of each measurement is computed via feeding our
# measurements through a classical multilayer perceptron.
# 

######################################################################
# We generate the series of :math:`k`-local observables with the following code.
# 

def local_pauli_group(qubits: int, locality: int):
    assert locality <= qubits, f"Locality must not exceed the number of qubits."
    return list(generate_paulis(0, 0, "", qubits, locality))

# This is a recursive generator function that constructs Pauli strings.
def generate_paulis(identities: int, paulis: int, output: str, qubits: int, locality: int):
    # Base case: if the output string's length matches the number of qubits, yield it.
    if len(output) == qubits:
        yield output
    else:
        # Recursive case: add an "I" (identity) to the output string.
        yield from generate_paulis(identities + 1, paulis, output + "I", qubits, locality)
        
        # If the number of Pauli operators used is less than the locality, add "X", "Y", or "Z"
        # systematically builds all possible Pauli strings that conform to the specified locality.
        if paulis < locality:
            yield from generate_paulis(identities, paulis + 1, output + "X", qubits, locality)
            yield from generate_paulis(identities, paulis + 1, output + "Y", qubits, locality)
            yield from generate_paulis(identities, paulis + 1, output + "Z", qubits, locality)


######################################################################
# For each image sample, we measure the output of the quantum circuit using the :math:`k`-local observables
# sequence, and perform logistic regression on these outputs. We do this for 1-local, 2-local and
# 3-local in the `for`-loop below.
# 

# Initialize lists to store training and testing accuracies for different localities.
train_accuracies_O = []
test_accuracies_O = []

for locality in range(1, 4):
    print(str(locality) + "-local: ")
    
    # Define a quantum device with 8 qubits using the default simulator.
    dev = qml.device("default.qubit", wires=8)

    # Define a quantum node (qnode) with the quantum circuit that will be executed on the device.
    @qml.qnode(dev)
    def circuit(features):
        # Generate all possible Pauli strings for the given locality.
        measurements = local_pauli_group(8, locality)
        
        # Apply the feature map to encode classical data into quantum states.
        feature_map(features)
        
        # Measure the expectation values of the generated Pauli operators.
        return [qml.expval(qml.pauli.string_to_pauli_word(measurement)) for measurement in measurements]

    # Vectorize the quantum circuit function to apply it to multiple data points in parallel.
    vcircuit = jax.vmap(circuit)
    
    # Transform the training and testing datasets by applying the quantum circuit.
    new_X_train = np.asarray(vcircuit(jnp.array(X_train))).T
    new_X_test = np.asarray(vcircuit(jnp.array(X_test))).T
    
    # Train a Multilayer Perceptron (MLP) classifier on the transformed training data.
    clf = MLPClassifier(early_stopping=True).fit(new_X_train, y_train)
    
    # Print the log loss for the training data.
    print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))
    
    # Print the log loss for the testing data.
    print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))
    
    # Calculate and store the training accuracy.
    acc = clf.score(new_X_train, y_train)
    train_accuracies_O.append(acc)
    print("Training accuracy: ", acc)
    
    # Calculate and store the testing accuracy.
    acc = clf.score(new_X_test, y_test)
    test_accuracies_O.append(acc)
    print("Testing accuracy: ", acc)
    print()

locality = ("1-local", "2-local", "3-local")
train_accuracies_O = [round(value, 2) for value in train_accuracies_O]
test_accuracies_O = [round(value, 2) for value in test_accuracies_O]
x = np.arange(3)
width = 0.25

# Create a bar plot to visualize the training and testing accuracies.
fig, ax = plt.subplots(layout="constrained")
rects = ax.bar(x, train_accuracies_O, width, label="Training", color="#FF87EB")  # Training accuracy bars.
rects = ax.bar(x + width, test_accuracies_O, width, label="Testing", color="#70CEFF")  # Testing accuracy bars.
ax.bar_label(rects, padding=3)
ax.set_xlabel("Locality")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy of different localities")
ax.set_xticks(x + width / 2, locality)
ax.legend(loc="upper left")
plt.show()


######################################################################
# We can see that the highest accuracy is achieved with the 3-local observables, which gives the
# classical model the most information about the outputs of the circuit. However, this is much
# more computationally resource heavy than its lower-locality counterparts.
# 

######################################################################
# The Ansatz Expansion Post-Variational Technique
# ---------------------
# 

######################################################################
# The Ansatz expansion approach does model approximation by directly expanding the parameterised
# Ansatz into an ensemble of fixed Ansätze. Starting from a variational Ansatz, multiple
# non-parameterized quantum circuits are constructed by Taylor expansion of the Ansatz around a
# suitably chosen initial setting of the parameters :math:`\theta_0`, which we set here as 0. Gradients and higher-order
# derivatives of circuits then can be obtained by parameter-shift rule. The output of the different circuits are then fed 
# into a classical neural network.
# 

######################################################################
# The following code is used to generate a series of fixed parameters that would be encoded into the
# Ansatz, using the above method.
# 

def deriv_params(thetas: int, order: int):
    # This function generates parameter shift values for calculating derivatives of a quantum circuit.
    # 'thetas' is the number of parameters in the circuit.
    # 'order' determines the order of the derivative to calculate (1st order, 2nd order, etc.).

    def generate_shifts(thetas: int, order: int):
        # Generate all possible combinations of parameters to shift for the given order.
        shift_pos = list(combinations(np.arange(thetas), order))
        
        # Initialize a 3D array to hold the shift values.
        # Shape: (number of combinations, 2^order, thetas)
        params = np.zeros((len(shift_pos), 2 ** order, thetas))
        
        # Iterate over each combination of parameter shifts.
        for i in range(len(shift_pos)):
            # Iterate over each possible binary shift pattern for the given order.
            for j in range(2 ** order):
                # Convert the index j to a binary string of length 'order'.
                for k, l in enumerate(f"{j:0{order}b}"):
                    # For each bit in the binary string:
                    if int(l) > 0:
                        # If the bit is 1, increment the corresponding parameter.
                        params[i][j][shift_pos[i][k]] += 1
                    else:
                        # If the bit is 0, decrement the corresponding parameter.
                        params[i][j][shift_pos[i][k]] -= 1
        
        # Reshape the parameters array to collapse the first two dimensions.
        params = np.reshape(params, (-1, thetas))
        return params

    # Start with a list containing a zero-shift array for all parameters.
    param_list = [np.zeros((1, thetas))]
    
    # Append the generated shift values for each order from 1 to the given order.
    for i in range(1, order + 1):
        param_list.append(generate_shifts(thetas, i))
    
    # Concatenate all the shift arrays along the first axis to create the final parameter array.
    params = np.concatenate(param_list, axis=0)
    
    # Scale the shift values by π/2.
    params *= np.pi / 2
    
    return params


######################################################################
# We construct the circuit and measure the top qubit with Pauli-Z.
# 

n_wires = 8
dev = qml.device("default.qubit", wires=n_wires)

@jax.jit
@qml.qnode(dev, interface="jax")
def circuit(features, params, n_wires=8):
    feature_map(features)
    ansatz(params)
    return qml.expval(qml.PauliZ(0))

######################################################################
# For each image sample, we measure the outputs of each parameterised circuit for each feature, and
# feed the outputs into a multilayer perceptron.
# 

# Initialize lists to store training and testing accuracies for different derivative orders.
train_accuracies_AE = []
test_accuracies_AE = []

# Loop through different derivative orders (1st order, 2nd order, 3rd order).
for order in range(1, 4):
    print("Order number: " + str(order))
    
    # Generate the parameter shifts required for the given derivative order.
    to_measure = deriv_params(16, order)

    # Transform the training dataset by applying the quantum circuit with the generated parameter shifts.
    new_X_train = []
    for thing in X_train:
        result = circuit(thing, to_measure.T)
        new_X_train.append(result)
    
    # Transform the testing dataset similarly.
    new_X_test = []
    for thing in X_test:
        result = circuit(thing, to_measure.T)
        new_X_test.append(result)
    
    # Train a Multilayer Perceptron (MLP) classifier on the transformed training data.
    clf = MLPClassifier(early_stopping=True).fit(new_X_train, y_train)
    
    # Print the log loss for the training data.
    print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))
    
    # Print the log loss for the testing data.
    print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))
    
    # Calculate and store the training accuracy.
    acc = clf.score(new_X_train, y_train)
    train_accuracies_AE.append(acc)
    print("Training accuracy: ", acc)
    
    # Calculate and store the testing accuracy.
    acc = clf.score(new_X_test, y_test)
    test_accuracies_AE.append(acc)
    print("Testing accuracy: ", acc)
    print()

locality = ("1-order", "2-order", "3-order")
train_accuracies_AE = [round(value, 2) for value in train_accuracies_AE]
test_accuracies_AE = [round(value, 2) for value in test_accuracies_AE]
x = np.arange(3)
width = 0.25
fig, ax = plt.subplots(layout="constrained")
rects = ax.bar(x, train_accuracies_AE, width, label="Training", color="#FF87EB")
ax.bar_label(rects, padding=3)
rects = ax.bar(x + width, test_accuracies_AE, width, label="Testing", color="#70CEFF")
ax.bar_label(rects, padding=3)
ax.set_xlabel("Order")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy of different derivative orders")
ax.set_xticks(x + width / 2, locality)
ax.legend(loc="upper left")
plt.show()

######################################################################
# Note that similar to the obsewrvable construction method, higher orders give higher testing accuracy. 
# However, it is similarly more computationally expensive to execute.
# 

######################################################################
# Hybrid Strategy
# ---------------------
# 

######################################################################
# When taking the strategy of observable construction, one additionally may want to use Ansatz
# quantum circuits to increase the complexity of the model. Hence, we discuss a simple hybrid
# strategy that combines both the usage of Ansatz expansion and observable construction. For each
# feature, we may first expand the Ansatz with each of our parameters, then use each :math:`k`-local
# observable to conduct measurements.
# 
# Due to the high number of circuits needed to be computed in this strategy, one may choose to
# further prune the circuits used in training, but this is not conducted in this demo.
# 
# Note that in our example, we have only tested 3 hybrid samples to reduce the running time of this
# script, but one may choose to try other combinations of the 2 strategies to potentially obtain
# better results.
# 

# Initialize matrices to store training and testing accuracies for different combinations of locality and order.
train_accuracies = np.zeros([4, 4])
test_accuracies = np.zeros([4, 4])

# Loop through different derivative orders (1st to 3rd) and localities (1-local to 3-local).
for order in range(1, 4):
    for locality in range(1, 4):
        # Skip invalid combinations where locality + order exceeds 3 or equals 0.
        if locality + order > 3 or locality + order == 0:
            continue
        print("Locality: " + str(locality) + " Order: " + str(order))

        # Define a quantum device with 8 qubits using the default simulator.
        dev = qml.device("default.qubit", wires=8)

        # Generate the parameter shifts required for the given derivative order and transpose them.
        params = deriv_params(16, order).T

        # Define a quantum node (qnode) with the quantum circuit that will be executed on the device.
        @qml.qnode(dev)
        def circuit(features, params):
            # Generate the Pauli group for the given locality.
            measurements = local_pauli_group(8, locality)
            feature_map(features)
            ansatz(params)
            # Measure the expectation values of the generated Pauli operators.
            return [qml.expval(qml.pauli.string_to_pauli_word(measurement)) for measurement in measurements]

        # Vectorize the quantum circuit function to apply it to multiple data points in parallel.
        vcircuit = jax.vmap(circuit)

        # Transform the training dataset by applying the quantum circuit with the generated parameter shifts.
        new_X_train = np.asarray(
            vcircuit(jnp.array(X_train), jnp.array([params for i in range(len(X_train))]))
        )
        # Reorder the axes and reshape the transformed data for input into the classifier.
        new_X_train = np.moveaxis(new_X_train, 0, -1).reshape(
            -1, len(local_pauli_group(8, locality)) * len(deriv_params(16, order))
        )

        # Transform the testing dataset similarly.
        new_X_test = np.asarray(
            vcircuit(jnp.array(X_test), jnp.array([params for i in range(len(X_test))]))
        )
        # Reorder the axes and reshape the transformed data for input into the classifier.
        new_X_test = np.moveaxis(new_X_test, 0, -1).reshape(
            -1, len(local_pauli_group(8, locality)) * len(deriv_params(16, order))
        )

        # Train a Multilayer Perceptron (MLP) classifier on the transformed training data.
        clf = MLPClassifier(early_stopping=True).fit(new_X_train, y_train)

        # Calculate and store the training and testing accuracies.
        train_accuracies[order][locality] = clf.score(new_X_train, y_train)
        test_accuracies[order][locality] = clf.score(new_X_test, y_test)

        print("Training loss: ", log_loss(y_train, clf.predict_proba(new_X_train)))
        print("Testing loss: ", log_loss(y_test, clf.predict_proba(new_X_test)))
        acc = clf.score(new_X_train, y_train)
        train_accuracies[locality][order] = acc
        print("Training accuracy: ", acc)
        acc = clf.score(new_X_test, y_test)
        test_accuracies[locality][order] = acc
        print("Testing accuracy: ", acc)
        print()

######################################################################
# Upon obtaining our hybrid results, we may now combine these results with that of the observable
# construction and Ansatz expansion menthods, and plot all the post-variational strategies together on
# a heatmap.
# 

for locality in range(1, 4):
    train_accuracies[locality][0] = train_accuracies_O[locality - 1]
    test_accuracies[locality][0] = test_accuracies_O[locality - 1]
for order in range(1, 4):
    train_accuracies[0][order] = train_accuracies_AE[order - 1]
    test_accuracies[0][order] = test_accuracies_AE[order - 1]

train_accuracies[3][3] = var_train_acc
test_accuracies[3][3] = var_test_acc

cvals = [0, 0.5, 0.85, 0.95, 1]
colors = ["black", "#C756B2", "#FF87EB", "#ACE3FF", "#D5F0FD"]
norm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(norm, cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)


locality = ["top qubit\n Pauli-Z", "1-local", "2-local", "3-local"]
order = ["0th Order", "1st Order", "2nd Order", "3rd Order"]

fig, ax = plt.subplots()
im = ax.imshow(train_accuracies, cmap=cmap, origin="lower")

ax.set_yticks(np.arange(len(locality)), labels=locality)
ax.set_xticks(np.arange(len(order)), labels=order)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(locality)):
    for j in range(len(order)):
        text = ax.text(
            j, i, np.round(train_accuracies[i, j], 2), ha="center", va="center", color="black"
        )
ax.text(3, 3, '\n\n(VQA)', ha="center", va="center", color="black")

ax.set_title("Training Accuracies")
fig.tight_layout()
plt.show()

locality = ["top qubit\n Pauli-Z", "1-local", "2-local", "3-local"]
order = ["0th Order", "1st Order", "2nd Order", "3rd Order"]

fig, ax = plt.subplots()
im = ax.imshow(test_accuracies, cmap=cmap, origin="lower")

ax.set_yticks(np.arange(len(locality)), labels=locality)
ax.set_xticks(np.arange(len(order)), labels=order)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(locality)):
    for j in range(len(order)):
        text = ax.text(
            j, i, np.round(test_accuracies[i, j], 2), ha="center", va="center", color="black"
        )
ax.text(3, 3, '\n\n(VQA)', ha="center", va="center", color="black")

ax.set_title("Test Accuracies")
fig.tight_layout()
plt.show()

######################################################################
# Experimental Results
# ---------------------
#  

######################################################################
# This demonstration shows that all hybrid methods exceed the variational algorithm while using the same
# Ansatz for the Ansatz expansion and hybrid strategies. We do not expect all post-variational methods to outperform variational algorithm. 
# For example, the Ansatz expansion up to the first order is likely to be worse than variational as it is merely a one step gradient update. 
# 
# From these performance results, we can obtain a glimpse of the effectiveness of each strategy. 
# The inclusion of 1-local and 2-local observables provide a boost in accuracy when used
# in conjunction with first order derivatives in the hybrid strategy. This implies that the addition
# of the observable expansion strategy can serve as an heuristic to expand the expressibility to
# Ansatz expansion method, which in itself may not be sufficient as a good training strategy.
# 

######################################################################
# Conclusion
# ---------------------
# 

######################################################################
# This tutorial demonstrates post-variational quantum neural networks,
# an alternative implementation of quantum neural networks in the NISQ setting.
# In this tutorial, we have implemented the post variational strategies to classify handwritten digits
# of twos and sixes.
# 
# Given a well-selected set of good fixed Ansätze, the post-variational method involves training classical
# neural networks, to which we can employ techniques to ensure good trainability. While this property of
# post-variational methods provides well optimised result based on the set of Ansätze given, 
# the barren plateau problems or related exponential concentration is not directly resolved. The hardness of the problem is
# instead delegated to the selection of the set of fixed Ansätze from an exponential amount of
# possible quantum circuits, to which one can find using the three heuristical strategies introduced in this tutorial.
#
# 

######################################################################
#
# References
# ---------------------
#
# .. [#cerezo2021variational]
#
#     M. Cerezo, A. Arrasmith, R. Babbush, S. C. Benjamin, S. Endo, K. Fujii, 
#     J. R. McClean, K. Mitarai, X. Yuan, L. Cincio, and P. J. Coles, 
#     Variational quantum algorithms, 
#     `Nat. Rev. Phys. 3, 625, (2021) <https://doi.org/10.1038/s42254-021-00348-9>`__.
#
#
# .. [#schuld2019evaluating]
#
#     M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, and N. Killoran, 
#     Evaluating analytic gradients on quantum hardware,
#     `Phys. Rev. A. 99, 032331, (2019) <https://doi.org/10.1103/PhysRevA.99.032331>`__. 
#     
#
# .. [#mcclean2018barren]
#
#     J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven,
#     Barren plateaus in quantum neural network training landscapes, 
#     `Nat. Commun. 9, 4812, (2018) <https://doi.org/10.1038/s41467-018-07090-4>`__.
#
#
# .. [#huang2024postvariational]
#
#     P.-W. Huang and P. Rebentrost, 
#     Post-variational quantum neural networks (2024), 
#     `arXiv:2307.10560 [quant-ph] <https://arxiv.org/abs/2307.10560>`__.
#
#
# .. [#du2020expressive]
#
#     Y. Du, M.-H. Hsieh, T. Liu, and D. Tao,  
#     Expressive power of parametrized quantum circuits,  
#     `Phys. Rev. Res. 2, 033125 (2020) <https://doi.org/10.1103/PhysRevResearch.2.033125>`__.
#
#

##############################################################################
# About the authors
# ---------------------
#