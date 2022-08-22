r"""

Introduction to Equivariant Learning
======================================================

.. meta::
    :property="og:description": Using the natural symmetries in a learning problem can improve learning.


*Author: Richard East. Posted: August 2022*





Introduction
~~~~~~~~~~~~~


Symmetries are at the heart of physics, indeed in condensed matter and
particle physics we often define a thing simply by the symmetries it
adheres to. What does symmetry mean for those in machine learning? In
this context the ambition is straight forward - it is a means to
reducting the parameter space and improving generalisation.

Suppose we have a learning task and the data we are learning from has an
underlying symmetry. For example, consider a game of Noughts and
Crosses: if we win a game, we would have won it if the board was rotated
or flipped along any of the lines of symmetry. Now if we want to train
an algorithm to spot the outcome of these games, we can either ignore
the existence of this symmetry or we can somehow include it. The
advantage in paying attention to the symmetry is it identifies multiple
configurations of the board as 'the same thing' as far as the symmetry
is concenred, this means we can reduce our parameter space and so the
amount of data our algorithm must sift through is immediately reduced.
Along the way the fact that our learning model must encode a symmetry
that actually exists in the system we are trying to represent natually
ecourages our results to be more genralisable.

In classical machine learning this often referred to as geometric deep
learning (GDL) due to the traditional association of symmetry to the
world of geometry and the fact that these considerations usually focus on
deep neural networks. We will refer to the quantum computing verison of
this as quantum geometric machine learning (QGML).


Representation theory in circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first thing to discuss is how do we work with symmetries in the
first place? The answer lies in the world of representation
theory.

Fundamentally representation theory is based on the prosaic observation
that linear algebra is easy and group theory is weird: So what if we can
study groups as linear maps?

To understand this let's look at the following definition of a
representation: Let :math:`\varphi` be a map sending :math:`g` in group
:math:`G` to a linear map :math:`\varphi(g): V \rightarrow V`, for some
vector space :math:`V`, which satisfies

.. math:: \varphi\left(g_{1} g_{2}\right)=\varphi\left(g_{1}\right) \circ \varphi\left(g_{2}\right) \quad \text { for all } g_{1}, g_{2} \in G.

Then we call :math:`\varphi` a representation of a group :math:`G` on a
vector space :math:`V` which we can see is a group homomorphism
:math:`\varphi: G \rightarrow G L(V, F)` for some field :math:`F` (like
:math:`\mathbb{R}` or :math:`\mathbb{C}` which are the spaces that the
elements of our matrices will belong to).

Now due to the importance of unitarity in quantum mechnics we are
particularly interested in the unitary representations: Representations
where the linear maps are unitary matrices. If we can
identify these then we will have a way to naturally encode groups in
quantum circuits (which are mostly made up of unitary gates). 


Now how does all this relate to symmetries? Well a large class of
symmetries can be charecterised as a group. Let's consider an example:
The symmetries of a sphere. Now when we think of this symmetry we
probably think something along the lines of "it's the same no matter how
I rotate it or flip it left to right etc". There is this idea of being
invarient under some operation, we also have the idea of being able to
undo these actions, if we rotate one way, we can rotate it back. If we
flip the ball right-to-left we can flip it left-to-right to get back to
where we started (notice too all these inverses are unique). Trivially
we can also do nothing. What exactly are we describing here? We have
elements that corespond to an action on a sphere that can be inverted and
for which there exists an identity. It is also trivially the case here
that if I consider three operations a,b,c from the set of rotations and
reflections of the sphere then if I combine two of them together then
:math:`a\circ (b \circ c) = (a\circ b) \circ c`. The operations are
associative. These features turn out to literally define a group!
 

**Definition**: A group is a set :math:`G` together with a binary operation
on :math:`G`, here denoted :math:`\circ`, that combines any two elements
:math:`a` and :math:`b` to form an element of :math:`G`, denoted
:math:`a \circ b`, such that the following three requirements, known as
group axioms, are satisfied, these are as follows:

**Associativity**:

For all :math:`a, b, c` in :math:`G`, one has
:math:`(a \circ b) \circ c=a \circ (b \circ c)`.

**Identity element**:

There exists an element :math:`e` in :math:`G` such that, for every
:math:`a` in :math:`G`, one has :math:`e \circ a=a` and
:math:`a \circ e=a`. Such an element is unique (see below). It is called
the identity element of the group.


**Inverse element**:

For each :math:`a` in :math:`G`, there exists an element :math:`b` in
:math:`G` such that :math:`a \circ b=e` and :math:`b \circ a=e`, where
:math:`e` is the identity element. For each :math:`a`, the element
:math:`b` is unique (see below); it is called the inverse of :math:`a`
and is commonly denoted :math:`a^{-1}`.

Now the group in itself is a very abstract creature this is why we look to
its representations. The group explains what symmetries we care about,
the unitary representations show us how those symmetries look on a particular
space of unitary matrices. Given that quantum circuits are largely
constructed from unitaries this gives us a direct connection between the
characterisation of symmetries and quantumc circutis. If we want to
encode the structure of the symmeteries in a quantum circuit we must
restrict our gates to being unitary representations of the group.

"""


##############################################################################
#
# Noughts and Crosses
# ----------------
# Let's look again at the game of noughts and crosses. Two players take
# turns to place a 0 or an X, depending on which player you are, in a 9X9
# grid. The aim is to get a 3 of your symbols in a row, column, or
# diagonal. As this is not always possible depending
# on the choices of the players a draw is possible. Our learning task
# is to take a set of completed games labelled with their outcomes and
# teach the algorithm to identify these correctly.
#


######################################################################
# This board of nine elements has the symmetry of the square, also known
# as the 'dihedral group'. This means it is symmetric under
# :math:`\frac{\pi}{2}` rotations and flips about the lines of symmetry of
# a square (vertical, horizontal, and diagonal).

##############################################################################
# .. figure:: ../demonstrations/equivariant_learning/NandC_sym.png
#     :align: center
#     :width: 70%

##############################################################################
# **The question is, how do we encode this in our QML problem?**
#
# First let us encode this problem classically, we will consider a 9
# element vector :math:`V`, each element of which indentifies a square of
# the board. The entries themselves can be
# :math:`+1`,\ :math:`0`,\ :math:`-1` representing a cross, no symbol, or
# a nought. The label is one hot encoded in a vector
# :math:`y=(y_o,y_- , y_x)` with :math:`+1` in the correct label and
# :math:`-1` in the others.


######################################################################
# To create the quantum model let us initialise all the qubits in \|0>,
# which we note is invaraint under the problems symmetries (flip and
# rotate all you want, it's still going to be zeroes whatever your
# mapping). We will then look to apply single qubit :math:`R_x(\theta)`
# rotations on individual qubits, encoding each of the
# possibilites in the board squares at an angle of
# :math:`\frac{2\pi}{3}` from each other. For our parameterised gates we
# will have a single qubit :math:`R_x(\theta)` and :math:`R_y(\theta)`
# rotation at each point, we will then use :math:`CR_Y(\phi)` for 2 qubit
# entangling gates. This implies that, for each encoding, crudely, we'll
# need 18 single qubit rotation parameters and :math:`\binom{9}{2}=36` two
# qubit gate rotations. Let's see how by using symmetries we can reduce
# this.


######################################################################
# The secret will be to encode the symmetries into the gate set so the
# observables we are interested in inherently respect the symmetries. How do
# we do this? We need to select the collections of gates that commute with
# the symmetries. In general we can use the twirling formula for this:
#


######################################################################
# Let S be the group that encodes our symmetries and :math:`U_{s}` be a
# unitary representation of :math:`\mathcal{S}`. Then,
#
# .. math::
#
#
#    \mathcal{T}_{U}[X]=\frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} U(s) X U(s)^{\dagger}
#
# defines a projector onto the set of operators commuting with all
# elements of the representation, i.e.,
# :math:`\left[\mathcal{T}_{U}[X], U(s)\right]=` 0 for all :math:`X` and
# :math:`s \in \mathcal{S}`. To see why this works for yourself apply the
# map to an arbitrary unitary representation and see if you can see how you
# can move it to the other side (remember the representation commutes with
# the group action), you might change the element of the group you're now
# working with, but since this is a sum over all of them that doesn't necessarily
# matter!
#


######################################################################
# So let's look again at our choice of gates, single qubit
# :math:`R_X(\theta)` rotations, and entangling 2 qubit :math:`CR_Y(\phi)`
# gates. What will we get by twirling these?
#


######################################################################
# In this particular instance we can see the action of the twirling
# operation geometrically as the symmtries involved are all
# permutations. Let's consider the R_x rotation acting on one qubit. Now
# if it is in the centre then you can flip around any symmetry axis you
# like, this operation is invarient, so we've identified one equivariant
# gate immediately. If it's on the corners then the flipping will send
# this qubit rotation to each of the other corners. Similairly if it's on the central
# edge then it will be sent round the other edges. So we can see that the
# twirl operation is a sum over all the possible outcomes of performing
# the symmetry action (the sum over the symmetry group actions). Having done this
# We can see that for a single qubit rotation the inavarient maps are rotations
# on the central qubit, at all the corners, and at all the central
# edges.
#


######################################################################
# For entangling gates the situation is similar. There are three invarient
# classes, the centre entagled with all corners, with all edges, and the
# edges paired in a ring.
#


######################################################################
# The prediction of a label is obtained via a one-hot-encoding by measuring
# the expectation values of three invariant observables:
#


######################################################################
# :math:`O_{-}=Z_{\text {middle }}=Z_{8}`
#
# :math:`O_{\circ}=\frac{1}{4} \sum_{i \in \text { corners }} Z_{i}=\frac{1}{4}\left[Z_{1}+Z_{4}+Z_{6}+Z_{7}\right]`
#
# :math:`O_{\times}=\frac{1}{4} \sum_{i \in \text { edges }} Z_{i}=\frac{1}{4}\left[Z_{0}+Z_{3}+Z_{7}+Z_{9}\right]`
#
# :math:`\hat{\boldsymbol{y}}=\left(\left\langle O_{\circ}\right\rangle,\left\langle O_{-}\right\rangle,\left\langle O_{\times}\right\rangle\right)`
#


######################################################################
# This is the quantum encoding of the symmetries into a learning problem.
# A prediction for a given data point will be obtained by selecting the
# class for which the observed expectation value is the largest.


######################################################################
# Now that we have a specific encoding and have decided on our observables
# we need to choose a suitable cost function to optimise.
#


######################################################################
# We will use an :math:`l_2` loss function acting on pairs of games and
# labels :math:`D={(g,y)}`
#


######################################################################
# :math:`\mathcal{L}(\mathcal{D})=\frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{g}, \boldsymbol{y}) \in \mathcal{D}}\|\hat{\boldsymbol{y}}(\boldsymbol{g})-\boldsymbol{y}\|_{2}^{2}`
#


######################################################################
# Let's now impliment this!
# First lets generate some games
#
# Here we are creating a small program that will play noughts and crosses against itself.
# On completion it spits out the winner and the winning board, with noughts as +1, draw as 0, and crosses as -1.

import torch
import random
from toolz import unique

torch.manual_seed(0)

#  create an empty board
def create_board():
    return torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


# Check for empty places on board
def possibilities(board):
    l = []
    for i in range(len(board)):
        for j in range(3):
            if board[i, j] == 0:
                l.append((i, j))
    return l


# Select a random place for the player
def random_place(board, player):
    selection = possibilities(board)
    current_loc = random.choice(selection)
    board[current_loc] = player
    return board


# Check if there is a winner by having 3 in a row
def row_win(board, player):
    for x in range(3):
        lista = []
        win = True

        for y in range(3):
            lista.append(board[x, y])

            if board[x, y] != player:
                win = False

        if win:
            # print("row win")
            break

    return win


# Check if there is a winner by having 3 in a column
def col_win(board, player):
    for x in range(3):
        win = True

        for y in range(3):
            if board[y, x] != player:
                win = False

        if win:
            # print("col win")
            break

    return win


# Check if there is a winner by having 3 along a diagonal
def diag_win(board, player):
    win1 = True
    win2 = True
    for x, y in [(0, 0), (1, 1), (2, 2)]:
        if board[x, y] != player:
            win1 = False

    for x, y in [(0, 2), (1, 1), (2, 0)]:
        if board[x, y] != player:
            win2 = False

    return win1 or win2


# Check if the win conditions have been met or if a draw has occured
def evaluate_game(board):
    winner = 99
    for player in [1, -1]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player

    if torch.all(board != 0) and winner == 99:
        winner = 0

    return winner


# Main function to start the game
def play_game():
    board, winner, counter = create_board(), 99, 1
    # print(board)
    while winner == 99:
        for player in [1, -1]:
            board = random_place(board, player)
            # print("Board after " + str(counter) + " move")
            # print(board)
            counter += 1
            winner = evaluate_game(board)
            if winner != 99:
                break

    return [board.flatten(), winner]


def create_dataset(size_for_each_winner):
    game_d = {-1: [], 0: [], 1: []}

    while min([len(v) for k, v in game_d.items()]) < size_for_each_winner:
        board, winner = play_game()
        if len(game_d[winner]) < size_for_each_winner:
            game_d[winner].append(board)

    res = []
    for winner, boards in game_d.items():
        res += [(board, winner) for board in boards]

    return res


NUM_TRAINING = 450
NUM_VALIDATION = 600

# Create datasets but with even numbers of each outcome
with torch.no_grad():
    dataset = create_dataset(NUM_TRAINING // 3)
    dataset_val = create_dataset(NUM_VALIDATION // 3)


######################################################################
# Now let's create the relevant expectation value circuits that respect
# the symmetry classes we defined over the single site and two site measurements


# %matplotlib inline
import pennylane as qml
import matplotlib.pyplot as plt

# Set up a 9 qubit system
dev = qml.device("default.qubit.torch", wires=9)

ob_center = qml.PauliZ(4)
ob_corner = (qml.PauliZ(0) + qml.PauliZ(2) + qml.PauliZ(6) + qml.PauliZ(8)) * (1 / 4)
ob_edge = (qml.PauliZ(1) + qml.PauliZ(3) + qml.PauliZ(5) + qml.PauliZ(7)) * (1 / 4)

# A reuploadable model for the edge qubit observable
@qml.qnode(dev, interface="torch")
def circuit(x, p):

    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RX(x[2], wires=2)
    qml.RX(x[3], wires=3)
    qml.RX(x[4], wires=4)
    qml.RX(x[5], wires=5)
    qml.RX(x[6], wires=6)
    qml.RX(x[7], wires=7)
    qml.RX(x[8], wires=8)

    # Centre single qubit rotation
    qml.RX(p[0], wires=4)
    qml.RY(p[1], wires=4)

    # Corner single qubit rotation
    qml.RX(p[2], wires=0)
    qml.RX(p[2], wires=2)
    qml.RX(p[2], wires=6)
    qml.RX(p[2], wires=8)

    qml.RY(p[3], wires=0)
    qml.RY(p[3], wires=2)
    qml.RY(p[3], wires=6)
    qml.RY(p[3], wires=8)

    # Edge single qubte rotation
    qml.RX(p[4], wires=1)
    qml.RX(p[4], wires=3)
    qml.RX(p[4], wires=5)
    qml.RX(p[4], wires=7)

    qml.RY(p[5], wires=1)
    qml.RY(p[5], wires=3)
    qml.RY(p[5], wires=5)
    qml.RY(p[5], wires=7)

    qml.IsingYY(p[6], wires=[0, 1])
    qml.IsingYY(p[6], wires=[2, 1])
    qml.IsingYY(p[6], wires=[2, 5])
    qml.IsingYY(p[6], wires=[8, 5])
    qml.IsingYY(p[6], wires=[8, 7])
    qml.IsingYY(p[6], wires=[6, 7])
    qml.IsingYY(p[6], wires=[6, 3])
    qml.IsingYY(p[6], wires=[0, 3])

    qml.IsingYY(p[7], wires=[4, 0])
    qml.IsingYY(p[7], wires=[4, 2])
    qml.IsingYY(p[7], wires=[4, 6])
    qml.IsingYY(p[7], wires=[4, 8])

    qml.IsingYY(p[8], wires=[1, 4])
    qml.IsingYY(p[8], wires=[3, 4])
    qml.IsingYY(p[8], wires=[5, 4])
    qml.IsingYY(p[8], wires=[7, 4])

    return [qml.expval(ob_center), qml.expval(ob_corner), qml.expval(ob_edge)]


fig, ax = qml.draw_mpl(circuit)([0] * 9, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

ob_center = qml.PauliZ(4)
ob_corner = (qml.PauliZ(0) + qml.PauliZ(2) + qml.PauliZ(6) + qml.PauliZ(8)) * (1 / 4)
ob_edge = (qml.PauliZ(1) + qml.PauliZ(3) + qml.PauliZ(5) + qml.PauliZ(7)) * (1 / 4)

# A reuploadable model for the edge qubit observable without symmetry
@qml.qnode(dev, interface="torch")
def circuit_no_sym(x, p):

    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RX(x[2], wires=2)
    qml.RX(x[3], wires=3)
    qml.RX(x[4], wires=4)
    qml.RX(x[5], wires=5)
    qml.RX(x[6], wires=6)
    qml.RX(x[7], wires=7)
    qml.RX(x[8], wires=8)

    # Centre single qubit rotation
    qml.RX(p[0], wires=4)
    qml.RY(p[1], wires=4)

    # Corner single qubit rotation
    qml.RX(p[2], wires=0)
    qml.RX(p[3], wires=2)
    qml.RX(p[4], wires=6)
    qml.RX(p[5], wires=8)

    qml.RY(p[6], wires=0)
    qml.RY(p[7], wires=2)
    qml.RY(p[8], wires=6)
    qml.RY(p[9], wires=8)

    # Edge single qubte rotation
    qml.RX(p[10], wires=1)
    qml.RX(p[11], wires=3)
    qml.RX(p[12], wires=5)
    qml.RX(p[13], wires=7)

    qml.RY(p[14], wires=1)
    qml.RY(p[15], wires=3)
    qml.RY(p[16], wires=5)
    qml.RY(p[17], wires=7)

    qml.CRY(p[18], wires=[0, 1])
    qml.CRY(p[19], wires=[2, 1])
    qml.CRY(p[20], wires=[2, 5])
    qml.CRY(p[21], wires=[8, 5])
    qml.CRY(p[22], wires=[8, 7])
    qml.CRY(p[23], wires=[6, 7])
    qml.CRY(p[24], wires=[6, 3])
    qml.CRY(p[25], wires=[0, 3])

    qml.CRY(p[26], wires=[4, 0])
    qml.CRY(p[27], wires=[4, 2])
    qml.CRY(p[28], wires=[4, 6])
    qml.CRY(p[29], wires=[4, 8])

    qml.CRY(p[30], wires=[1, 4])
    qml.CRY(p[31], wires=[3, 4])
    qml.CRY(p[32], wires=[5, 4])
    qml.CRY(p[33], wires=[7, 4])

    return [qml.expval(ob_center), qml.expval(ob_corner), qml.expval(ob_edge)]


fig, ax = qml.draw_mpl(circuit_no_sym)([0] * 9, [0] * 34)


######################################################################
# We need to feed the vector :math:`\boldsymbol{y}` made up of the expectation value of these
# three operators into the loss function and use this to update our
# parameters


import math


def encode_game(game):
    board, res = game
    x = board * (2 * math.pi) / 3
    if res == 1:
        y = [-1, -1, 1]
    elif res == -1:
        y = [1, -1, -1]
    else:
        y = [-1, 1, -1]
    return x, y


######################################################################
# Remember this is the loss function that we're after
# :math:`\mathcal{L}(\mathcal{D})=\frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{g}, \boldsymbol{y}) \in \mathcal{D}}\|\hat{\boldsymbol{y}}(\boldsymbol{g})-\boldsymbol{y}\|_{2}^{2}`
# we need to define this and then we can begin our optimisation.

# calculate cross entropy for classification problem
from math import log


def cost_function(params, input, target):
    output = torch.stack([circuit(x, params) for x in input])
    vec = output - target
    sum_sqr = torch.sum(vec * vec, dim=1)
    return torch.mean(sum_sqr)


from torch import optim

params = 0.01 * torch.randn(9)
params.requires_grad = True
opt = optim.Adam([params], lr=1e-2)

import numpy as np

max_epoch = 10
max_step = 30
batch_size = 15

encoded_dataset = list(zip(*[encode_game(game) for game in dataset]))
encoded_dataset_val = list(zip(*[encode_game(game) for game in dataset_val]))


def accuracy(p, x_val, y_val):
    with torch.no_grad():
        y_val = torch.tensor(y_val)
        y_out = torch.stack([circuit(x, p) for x in x_val])
        acc = torch.sum(torch.argmax(y_out, axis=1) == torch.argmax(y_val, axis=1)) / len(x_val)
        return acc


print(f"accuracy without training = {accuracy(params, *encoded_dataset_val)}")

x_dataset = torch.stack(encoded_dataset[0])
y_dataset = torch.tensor(encoded_dataset[1], requires_grad=False)

saved_costs_sym = []
saved_accs_sym = []
for epoch in range(max_epoch):
    rand_idx = torch.randperm(len(x_dataset))
    # Shuffled dataset
    x_dataset = x_dataset[rand_idx]
    y_dataset = y_dataset[rand_idx]

    costs = []

    for step in range(max_step):
        x_batch = x_dataset[step * batch_size : (step + 1) * batch_size]
        y_batch = y_dataset[step * batch_size : (step + 1) * batch_size]

        # Following https://pennylane.readthedocs.io/en/stable/introduction/interfaces/torch.html
        def opt_func():
            opt.zero_grad()
            loss = cost_function(params, x_batch, y_batch)
            costs.append(loss.item())
            loss.backward()
            return loss

        opt.step(opt_func)

    cost = np.mean(costs)
    saved_costs_sym.append(cost)

    if (epoch + 1) % 1 == 0:
        # Compute validation accuracy
        acc_val = accuracy(params, *encoded_dataset_val)
        saved_accs_sym.append(acc_val)

        res = [epoch + 1, cost, acc_val]
        print("Epoch: {:2d} | Loss: {:3f} | Validation accuracy: {:3f}".format(*res))


params = 0.01 * torch.randn(34)
params.requires_grad = True
opt = optim.Adam([params], lr=1e-2)

# calculate cross entropy for classification problem


def cost_function_no_sym(params, input, target):
    output = torch.stack([circuit_no_sym(x, params) for x in input])
    vec = output - target
    sum_sqr = torch.sum(vec * vec, dim=1)
    return torch.mean(sum_sqr)


import numpy as np

max_epoch = 10
max_step = 30
batch_size = 15

encoded_dataset = list(zip(*[encode_game(game) for game in dataset]))
encoded_dataset_val = list(zip(*[encode_game(game) for game in dataset_val]))


def accuracy_no_sym(p, x_val, y_val):
    with torch.no_grad():
        y_val = torch.tensor(y_val)
        y_out = torch.stack([circuit_no_sym(x, p) for x in x_val])
        acc = torch.sum(torch.argmax(y_out, axis=1) == torch.argmax(y_val, axis=1)) / len(x_val)
        return acc


print(f"accuracy without training = {accuracy_no_sym(params, *encoded_dataset_val)}")

x_dataset = torch.stack(encoded_dataset[0])
y_dataset = torch.tensor(encoded_dataset[1], requires_grad=False)

saved_costs = []
saved_accs = []
for epoch in range(max_epoch):
    rand_idx = torch.randperm(len(x_dataset))
    # Shuffled dataset
    x_dataset = x_dataset[rand_idx]
    y_dataset = y_dataset[rand_idx]

    costs = []

    for step in range(max_step):
        x_batch = x_dataset[step * batch_size : (step + 1) * batch_size]
        y_batch = y_dataset[step * batch_size : (step + 1) * batch_size]

        # Following https://pennylane.readthedocs.io/en/stable/introduction/interfaces/torch.html
        def opt_func():
            opt.zero_grad()
            loss = cost_function_no_sym(params, x_batch, y_batch)
            costs.append(loss.item())
            loss.backward()
            return loss

        opt.step(opt_func)

    cost = np.mean(costs)
    saved_costs.append(costs)

    if (epoch + 1) % 1 == 0:
        # Compute validation accuracy
        acc_val = accuracy_no_sym(params, *encoded_dataset_val)
        saved_accs.append(acc_val)

        res = [epoch + 1, cost, acc_val]
        print("Epoch: {:2d} | Loss: {:3f} | Validation accuracy: {:3f}".format(*res))


from matplotlib import pyplot as plt

plt.title("Validation accuracies")
plt.style.use("seaborn")
plt.plot(saved_accs_sym, "b", label="Symmetric")
plt.plot(saved_accs, "g", label="Standard")

plt.ylabel("Validation accuracy")
plt.xlabel("Optimization steps")
plt.legend()
plt.show()


######################################################################
# This example is inspired by the model used in `Meyer et al. (2022) <https://arxiv.org/abs/2205.06217>`.
# The author would also like to acknowledge the helpful input of C.-Y. Park.
