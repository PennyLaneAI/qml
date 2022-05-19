r"""

.. function_fitting_qsp:

Function Fitting using Quantum Signal Processing
======================================================

.. meta::
    :property="og:description": Learn how to create polynomial approximations to functions
        using Quantum Signal Processing (QSP).
    :property="og:image": https://pennylane.ai/qml/demonstrations/function_fitting_qsp/cover.png


This demo is inspired by the paper `‘A Grand Unification of Quantum
Algorithms’ <https://arxiv.org/abs/2105.02859>`__. This
paper is centered around the Quantum Singular Value Transform (QSVT)
protocol and how it provides a single framework to generalize some of
the most famous quantum algorithms like Shor’s factoring algorithm, Grover search,
and more.

The QSVT is a method to apply polynomial transformations to the singular
values of *any matrix*. This is powerful
because from polynomial transformations we can generate arbitrary function
transformations using Taylor approximations. The QSVT protocol is an
extension of the more constrained Quantum Signal Processing (QSP)
protocol which presents a method for polynomial transformation of matrix
entries in a single-qubit unitary operator. The QSVT protocol is sophisticated,
but the idea at its core
is quite simple. By studying QSP, we get a relatively simpler path to explore
this idea at the foundation of QSVT.

In this demo, we explore the QSP protocol and how it can be used
for curve fitting. We show how you can fit polynomials, as illustrated in
the animation below.

.. figure:: ../demonstrations/function_fitting_qsp/trained_poly.gif
    :align: center
    :width: 50%

This is a powerful tool that will ultimately allow us
to approximate any function on the interval :math:`[-1, 1]` that
satisfies certain constraints. Before we can dive into function fitting,
let’s develop some intuition. Consider the following single-qubit operator
parameterized by :math:`a \in [-1, 1]`:

.. math:: \hat{W}(a) = \begin{bmatrix} a & i\sqrt{1 - a^{2}} \\ i\sqrt{1 - a^{2}} & a \end{bmatrix}.

:math:`\hat{W}(a)` is called the *signal rotation operator* (SRO). Using
this operator, we can construct another operator called the
*signal processing pperator* (SPO),

.. math::  \hat{U}_{sp} = \hat{R}_{z}(\phi_{0}) \prod_{k=1}^{d} \hat{W}(a) \hat{R}_{z}(\phi_{k}).

The SPO is parameterized by a vector
:math:`\vec{\phi} \in \mathbb{R}^{d+1}`, where :math:`d` is a free
parameter which represents the number of repeated applications of
:math:`\hat{W}(a)`.

The SPO :math:`\hat{U}_{sp}` alternates between applying the SRO :math:`\hat{W}(a)`
and parameterized rotations around the z-axis. Let’s see what happens when we try to compute the
expectation value :math:`\bra{0}\hat{U}_{sp}\ket{0}` for the particular
case where :math:`d = 2` and :math:`\vec{\phi} = (0, 0, 0)` :

.. math::


   \begin{align*}
   \bra{0}\hat{U}_{sp}\ket{0} &= \bra{0} \ \hat{R}_{z}(0) \prod_{k=1}^{2} \hat{W}(a) \hat{R}_{z}(0) \ \ket{0} \\
   \bra{0}\hat{U}_{sp}\ket{0} &= \bra{0} \hat{W}(a)^{2} \ket{0} \\
   \end{align*}

.. math::


   \bra{0}\hat{U}_{sp}\ket{0} = \bra{0} \begin{bmatrix} a & i\sqrt{1 - a^{2}} \\ i\sqrt{1 - a^{2}} & a \end{bmatrix} \ \circ \ \begin{bmatrix} a & i\sqrt{1 - a^{2}} \\ i\sqrt{1 - a^{2}} & a \end{bmatrix} \ket{0}

.. math::


   \bra{0}\hat{U}_{sp}\ket{0} = \bra{0} \begin{bmatrix} 2a^{2} - 1 & 2ai\sqrt{1 - a^{2}} \\ 2ai\sqrt{1 - a^{2}} & 2a^{2} - 1 \end{bmatrix} \ket{0}

.. math::  \bra{0}\hat{U}_{sp}\ket{0} = 2a^{2} - 1

Notice that this quantity is a polynomial in :math:`a`. Equivalently,
suppose we wanted to create a map :math:`S: a \to 2a^2 - 1`.
This expectation value would give us the means to perform such a mapping.
This may seem oddly specific at first, but it turns out that
this process can be generalized for generating a mapping
:math:`S: a \to \text{poly}(a)`. The following theorem shows us how:

Theorem: Quantum Signal Processing
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Given a vector :math:`\vec{\phi} \in \mathbb{R}^{d+1}`, there exist
complex polynomials :math:`P(a)` and :math:`Q(a)` such that the SPO,
:math:`\hat{U}_{sp}`, can be expressed in matrix form as:

.. math::  \hat{U}_{sp} = \hat{R}_{z}(\phi_{0}) \prod_{k=1}^{d} \hat{W}(a) \hat{R}_{z}(\phi_{k}),

.. math::

   \hat{U}_{sp} = \begin{bmatrix} P(a) & iQ(a)\sqrt{1 - a^{2}} \\ iQ^{*}(a)\sqrt{1 - a^{2}} & P^{*}(a) \end{bmatrix},

where :math:`a \in [-1, 1]` and the polynomials :math:`P(a)`,
:math:`Q(a)` satisfy the following constraints:

-  :math:`deg(P) \leq d \ ` and :math:`deg(Q) \leq d - 1`,
-  :math:`P` has parity :math:`d` mod 2 and :math:`Q` has parity,
   :math:`d - 1` mod 2
-  :math:`|P|^{2} + (1 - a^{2})|Q|^{2} = 1`.


The third condition is actually quite restrictive because if we substitute :math:`a = \pm 1`,
we get the result :math:`|P^{2}(\pm 1)| = 1`. Thus it restricts the polynomial to be
pinned to :math:`\pm 1` at the end points of the domain, :math:`a = \pm 1`. This condition
can be relaxed to :math:`|P^{2}(a)| \leq 1` by expressing the signal processing operator
in the Hadamard basis, i.e., :math:`\bra{+}\hat{U}_{sp}(\vec{\phi};a)\ket{+}`). This is equivalent to
redefining :math:`P(a)` such that:

.. math:: P^{'}(a) = \text{Re}(P(a)) + i\text{Re}(Q(a))\sqrt{1 - a^{2}}

*This is the convention we follow in this demo.*

"""

######################################################################
# Let's Plot some Polynomials
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now we put this theorem to the test! In this section we construct
# the SRO :math:`\hat{W}(a)`, and then use PennyLane to define the SPO.
# To test the theorem we will randomly generate parameters
# :math:`\vec{\phi}` and plot the expectation value
# :math:`\bra{+}\hat{U}_{sp}(\vec{\phi};a)\ket{+}` for
# :math:`a \in [-1, 1]`.
#
# We begin by importing the required packages:
#

import math
import torch
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


######################################################################
# Next, we introduce a function called ``compute_signal_rotation_mat(a)``,
# which will construct the SRO matrix. We can also make a helper function
# (``generate_many_sro(a_vals)``) which, given an array of possible values
# for ‘:math:`a`’, will generate an array of :math:`\hat{W}(a)` associated
# with each element. We us Pytorch to construct this array as it will later
# be used as input when training our function fitting model.
#


def compute_signal_rotation_mat(a):
    """Given a fixed value 'a', compute the signal rotation matrix W(a).
    (requires -1 <= 'a' <= 1)
    """
    diag = a
    off_diag = (1 - a**2) ** (1 / 2) * 1j
    W = [[diag, off_diag], [off_diag, diag]]

    return W


def generate_many_sro(a_vals):
    """Given a tensor of possible 'a' vals, return a tensor of W(a)"""
    w_array = []
    for a in a_vals:
        w = compute_signal_rotation_mat(a)
        w_array.append(w)

    return torch.tensor(w_array, dtype=torch.complex64, requires_grad=False)


######################################################################
# Now having access to the matrix elements of the SRO, we can leverage
# PennyLane to define a quantum function that will compute the SPO.
# Recall we are measuring in the Hadamard basis to
# relax the third condition of the theorem. We
# accomplish this by sandwiching the SPO between two Hadamard gates to
# account for this change of basis.
#


def QSP_circ(phi, W):
    """This circuit applies the SPO. The components in the matrix
    representation of the final unitary are polynomials!
    """
    qml.Hadamard(wires=0)  # set initial state |+>
    for i in range(len(phi) - 1):  # iterate through rotations in reverse order
        qml.RZ(phi[i], wires=0)
        qml.QubitUnitary(W, wires=0)

    qml.RZ(phi[-1], wires=0)  # final rotation
    qml.Hadamard(wires=0)  # change of basis |+> , |->
    return


######################################################################
# Finally, we randomly generate the vector :math:`\vec{\phi}` and plot the
# expectation value :math:`\bra{+}\hat{U}_{sp}\ket{+}` as a function of
# :math:`a`. In this case we choose :math:`d = 5`.
# We expect to observe the following:
#
# -  Since :math:`d` is odd, we expect all of the
#    polynomials we plot to have odd symmetry
# -  Since :math:`d = 5`, we expect none of the polynomials will have
#    terms ~ :math:`O(a^6)` or higher
# -  All of the polynomials are bounded by :math:`\pm1`
#

d = 5
a_vals = torch.linspace(-1, 1, 50)
w_mats = generate_many_sro(a_vals)

gen = torch.Generator()
gen.manual_seed(444422)  # set random seed for reproducibility

for i in range(5):
    phi = torch.rand(d + 1, generator=gen) * 2 * torch.tensor([math.pi], requires_grad=False)
    matrix_func = qml.matrix(QSP_circ)
    y_vals = [matrix_func(phi, w)[0, 0].real for w in w_mats]

    plt.plot(a_vals, y_vals, label=f"poly #{i}")

plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -1.0, 1.0, color="black")
plt.legend(loc=1)
plt.show()


######################################################################
# .. figure:: ../demonstrations/function_fitting_qsp/random_poly.png
#     :align: center
#     :width: 50%
#


######################################################################
# Exactly as predicted, all of these conditions are met!
#
# -  All curves have odd symmetry in :math:`a`
# -  None of the curves are of (:math:`O(a^6)` (most of the curves look cubic)
# -  Each plot does not exceed :math:`\pm1` !
#
# Function Fitting with Quantum Signal Processing:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Another observation we can make about this theorem is the fact that it
# holds true in both directions: If we have two polynomials :math:`P(a)`
# and :math:`Q(a)` that satisfy the conditions of the theorem, then there
# exists a :math:`\vec{\phi}` for which we can construct a signal
# processing operator which maps :math:`a \to P(a)`.
#
# In this section we try to answer the question:
#
# **Can we learn the parameter values of** :math:`\vec{\phi}` **to transform
# our signal processing operator polynomial to fit a given function?**.
#
# In order to answer this question, we begin by building a machine
# learning model using Pytorch. The ``__init__()`` method handles the
# random initialization of our parameter vector :math:`\vec{\phi}`. The
# ``forward()`` method takes an array of signal rotation matrices
# :math:`\hat{W}(a)` for varying :math:`a`, and produces the
# predicted :math:`y` values.
#
# Next we leverage the PennyLane function ``qml.matrix()``, which
# accepts our quantum function (it can also accept quantum tapes and
# QNodes) and returns its unitary matrix representation. We are interested
# in the real value of the top left entry, this corresponds to
# :math:`P(a)`.
#

torch_pi = torch.Tensor([math.pi])

class QSP_Func_Fit(torch.nn.Module):
    def __init__(self, degree, num_vals, random_seed=None):
        """Given the degree and number of samples, this method randomly
        initializes the parameter vector (randomness can be set by random_seed)
        """
        super().__init__()
        if random_seed is None:
            self.phi = torch_pi * torch.rand(degree + 1, requires_grad=True)

        else:
            gen = torch.Generator()
            gen.manual_seed(random_seed)
            self.phi = torch_pi * torch.rand(degree + 1, requires_grad=True, generator=gen)

        self.phi = torch.nn.Parameter(self.phi)
        self.num_phi = degree + 1
        self.num_vals = num_vals

    def forward(self, omega_mats):
        """PennyLane forward implementation"""
        y_pred = []
        generate_qsp_mat = qml.matrix(QSP_circ)

        for w in omega_mats:
            u_qsp = generate_qsp_mat(self.phi, w)
            P_a = u_qsp[0, 0]  # Taking the (0,0) entry of the matrix corresponds to <0|U|0>
            y_pred.append(P_a.real)

        return torch.stack(y_pred, 0)


######################################################################
# Next we create a ``Model_Runner`` class to handle running the
# optimization, storing the results, and providing plotting functionality:
#

class Model_Runner:
    def __init__(self, model, degree, num_samples, x_vals, process_x_vals, y_true):
        """Given a model and a series of model specific arguments, store everything in
        internal attributes.
        """
        self.model = model
        self.degree = degree
        self.num_samples = num_samples

        self.x_vals = x_vals
        self.inp = process_x_vals(x_vals)
        self.y_true = y_true

    def execute(
        self, random_seed=13_02_1967, max_shots=25000, verbose=True
    ):  # easter egg: oddly specific seed?
        """Run the optimization protocol on the model using Mean Square Error as a loss
        function and using stochastic gradient descent as the optimizer.
        """
        model = self.model(degree=self.degree,
                           num_vals=self.num_samples,
                           random_seed=random_seed)

        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

        t = 0
        j = 0
        loss_val = 1.0
        prev_loss = 2.0

        while (t <= max_shots) and (loss_val > 0.5):

            self.y_pred = model(self.inp)

            if t == 1:
                self.init_y_pred = self.y_pred

            # Compute and print loss
            loss = criterion(self.y_pred, self.y_true)
            loss_val = loss.item()

            if (t % 100 == 0) and verbose:
                print(f"---- iter: {t}, loss: {round(loss_val, 4)} -----")

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prev_loss = loss_val
            t += 1

        self.model_params = model.phi

    def plot_result(self, show=True):
        """Plot the results"""
        plt.plot(self.x_vals, self.y_true.tolist(), "--b", label="target func")
        plt.plot(self.x_vals, self.y_pred.tolist(), ".g", label="optim params")
        plt.plot(self.x_vals, self.init_y_pred.tolist(), ".r", label="init params")
        plt.legend(loc=1)

        if show:
            plt.show()


######################################################################
# Now that we have a model, lets first attempt to fit a polynomial. We
# expect this to perform well when the target polynomial also obeys the
# symmetry and degree constraints that our quantum signal processing
# polynomial does. To do this, we defined a function ``custom_poly(x)``
# which implements the target polynomial. In this case, we (arbitrarily)
# choose the target polynomial:
#
# .. math::  y = 4x^{5} - 5x^{3} + x
#
# Lets see how well we can fit this polynomial!
#
#
#   .. note::
#       Depending on the initial parameters, training can take
#       anywhere from 10 - 30 mins
#

d = 9  # dim(phi) = d + 1,
num_samples = 50

def custom_poly(x):
    """A custom polynomial of degree <= d and parity d % 2"""
    return torch.tensor(4 * x**5 - 5 * x**3 + x, requires_grad=False, dtype=torch.float)

a_vals = np.linspace(-1, 1, num_samples)
y_true = custom_poly(a_vals)

qsp_model_runner = Model_Runner(
    QSP_Func_Fit, d, num_samples, a_vals, generate_many_sro, y_true
)

qsp_model_runner.execute()
qsp_model_runner.plot_result()


##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     ---- iter: 100, loss: 13.4202 -----
#     ---- iter: 200, loss: 13.2461 -----
#     ---- iter: 300, loss: 13.0731 -----
#     ---- iter: 400, loss: 12.901 -----
#     ---- iter: 500, loss: 12.7297 -----
#     ---- iter: 600, loss: 12.5591 -----
#     ---- iter: 700, loss: 12.3891 -----
#     ---- iter: 800, loss: 12.2197 -----
#     ---- iter: 900, loss: 12.0509 -----
#     ---- iter: 1000, loss: 11.8826 -----
#     ---- iter: 1100, loss: 11.7147 -----
#     ---- iter: 1200, loss: 11.5474 -----
#     ---- iter: 1300, loss: 11.3806 -----
#     ---- iter: 1400, loss: 11.2144 -----
#     ---- iter: 1500, loss: 11.0487 -----
#     ---- iter: 1600, loss: 10.8837 -----
#     ---- iter: 1700, loss: 10.7192 -----
#     ---- iter: 1800, loss: 10.5556 -----
#     ---- iter: 1900, loss: 10.3927 -----
#     ---- iter: 2000, loss: 10.2306 -----
#     ---- iter: 2100, loss: 10.0695 -----
#     ---- iter: 2200, loss: 9.9093 -----
#     ---- iter: 2300, loss: 9.7502 -----
#     ---- iter: 2400, loss: 9.5922 -----
#     ---- iter: 2500, loss: 9.4353 -----
#     ---- iter: 2600, loss: 9.2797 -----
#     ---- iter: 2700, loss: 9.1254 -----
#     ---- iter: 2800, loss: 8.9724 -----
#     ---- iter: 2900, loss: 8.8209 -----
#     ---- iter: 3000, loss: 8.6708 -----
#     ---- iter: 3100, loss: 8.5221 -----
#     ---- iter: 3200, loss: 8.3751 -----
#     ---- iter: 3300, loss: 8.2296 -----
#     ---- iter: 3400, loss: 8.0857 -----
#     ---- iter: 3500, loss: 7.9434 -----
#     ---- iter: 3600, loss: 7.8028 -----
#     ---- iter: 3700, loss: 7.6638 -----
#     ---- iter: 3800, loss: 7.5266 -----
#     ---- iter: 3900, loss: 7.391 -----
#     ---- iter: 4000, loss: 7.257 -----
#     ---- iter: 4100, loss: 7.1248 -----
#     ---- iter: 4200, loss: 6.9943 -----
#     ---- iter: 4300, loss: 6.8654 -----
#     ---- iter: 4400, loss: 6.7382 -----
#     ---- iter: 4500, loss: 6.6127 -----
#     ---- iter: 4600, loss: 6.4888 -----
#     ---- iter: 4700, loss: 6.3665 -----
#     ---- iter: 4800, loss: 6.2459 -----
#     ---- iter: 4900, loss: 6.1269 -----
#     ---- iter: 5000, loss: 6.0095 -----
#     ---- iter: 5100, loss: 5.8937 -----
#     ---- iter: 5200, loss: 5.7794 -----
#     ---- iter: 5300, loss: 5.6668 -----
#     ---- iter: 5400, loss: 5.5556 -----
#     ---- iter: 5500, loss: 5.446 -----
#     ---- iter: 5600, loss: 5.3379 -----
#     ---- iter: 5700, loss: 5.2314 -----
#     ---- iter: 5800, loss: 5.1263 -----
#     ---- iter: 5900, loss: 5.0227 -----
#     ---- iter: 6000, loss: 4.9207 -----
#     ---- iter: 6100, loss: 4.8201 -----
#     ---- iter: 6200, loss: 4.721 -----
#     ---- iter: 6300, loss: 4.6234 -----
#     ---- iter: 6400, loss: 4.5272 -----
#     ---- iter: 6500, loss: 4.4325 -----
#     ---- iter: 6600, loss: 4.3393 -----
#     ---- iter: 6700, loss: 4.2475 -----
#     ---- iter: 6800, loss: 4.1572 -----
#     ---- iter: 6900, loss: 4.0684 -----
#     ---- iter: 7000, loss: 3.981 -----
#     ---- iter: 7100, loss: 3.895 -----
#     ---- iter: 7200, loss: 3.8105 -----
#     ---- iter: 7300, loss: 3.7275 -----
#     ---- iter: 7400, loss: 3.6459 -----
#     ---- iter: 7500, loss: 3.5658 -----
#     ---- iter: 7600, loss: 3.4871 -----
#     ---- iter: 7700, loss: 3.4098 -----
#     ---- iter: 7800, loss: 3.3339 -----
#     ---- iter: 7900, loss: 3.2595 -----
#     ---- iter: 8000, loss: 3.1864 -----
#     ---- iter: 8100, loss: 3.1148 -----
#     ---- iter: 8200, loss: 3.0446 -----
#     ---- iter: 8300, loss: 2.9758 -----
#     ---- iter: 8400, loss: 2.9083 -----
#     ---- iter: 8500, loss: 2.8422 -----
#     ---- iter: 8600, loss: 2.7775 -----
#     ---- iter: 8700, loss: 2.7141 -----
#     ---- iter: 8800, loss: 2.652 -----
#     ---- iter: 8900, loss: 2.5913 -----
#     ---- iter: 9000, loss: 2.5318 -----
#     ---- iter: 9100, loss: 2.4737 -----
#     ---- iter: 9200, loss: 2.4168 -----
#     ---- iter: 9300, loss: 2.3611 -----
#     ---- iter: 9400, loss: 2.3067 -----
#     ---- iter: 9500, loss: 2.2535 -----
#     ---- iter: 9600, loss: 2.2014 -----
#     ---- iter: 9700, loss: 2.1506 -----
#     ---- iter: 9800, loss: 2.1009 -----
#     ---- iter: 9900, loss: 2.0524 -----
#     ---- iter: 10000, loss: 2.005 -----
#     ---- iter: 10100, loss: 1.9586 -----
#     ---- iter: 10200, loss: 1.9134 -----
#     ---- iter: 10300, loss: 1.8692 -----
#     ---- iter: 10400, loss: 1.826 -----
#     ---- iter: 10500, loss: 1.7839 -----
#     ---- iter: 10600, loss: 1.7427 -----
#     ---- iter: 10700, loss: 1.7025 -----
#     ---- iter: 10800, loss: 1.6633 -----
#     ---- iter: 10900, loss: 1.625 -----
#     ---- iter: 11000, loss: 1.5877 -----
#     ---- iter: 11100, loss: 1.5512 -----
#     ---- iter: 11200, loss: 1.5156 -----
#     ---- iter: 11300, loss: 1.4808 -----
#     ---- iter: 11400, loss: 1.4469 -----
#     ---- iter: 11500, loss: 1.4138 -----
#     ---- iter: 11600, loss: 1.3814 -----
#     ---- iter: 11700, loss: 1.3499 -----
#     ---- iter: 11800, loss: 1.3191 -----
#     ---- iter: 11900, loss: 1.2891 -----
#     ---- iter: 12000, loss: 1.2597 -----
#     ---- iter: 12100, loss: 1.2311 -----
#     ---- iter: 12200, loss: 1.2032 -----
#     ---- iter: 12300, loss: 1.1759 -----
#     ---- iter: 12400, loss: 1.1493 -----
#     ---- iter: 12500, loss: 1.1233 -----
#     ---- iter: 12600, loss: 1.0979 -----
#     ---- iter: 12700, loss: 1.0732 -----
#     ---- iter: 12800, loss: 1.049 -----
#     ---- iter: 12900, loss: 1.0254 -----
#     ---- iter: 13000, loss: 1.0024 -----
#     ---- iter: 13100, loss: 0.9799 -----
#     ---- iter: 13200, loss: 0.9579 -----
#     ---- iter: 13300, loss: 0.9365 -----
#     ---- iter: 13400, loss: 0.9155 -----
#     ---- iter: 13500, loss: 0.8951 -----
#     ---- iter: 13600, loss: 0.8751 -----
#     ---- iter: 13700, loss: 0.8557 -----
#     ---- iter: 13800, loss: 0.8366 -----
#     ---- iter: 13900, loss: 0.818 -----
#     ---- iter: 14000, loss: 0.7999 -----
#     ---- iter: 14100, loss: 0.7821 -----
#     ---- iter: 14200, loss: 0.7648 -----
#     ---- iter: 14300, loss: 0.7479 -----
#     ---- iter: 14400, loss: 0.7313 -----
#     ---- iter: 14500, loss: 0.7152 -----
#     ---- iter: 14600, loss: 0.6994 -----
#     ---- iter: 14700, loss: 0.684 -----
#     ---- iter: 14800, loss: 0.6689 -----
#     ---- iter: 14900, loss: 0.6542 -----
#     ---- iter: 15000, loss: 0.6399 -----
#     ---- iter: 15100, loss: 0.6258 -----
#     ---- iter: 15200, loss: 0.6121 -----
#     ---- iter: 15300, loss: 0.5986 -----
#     ---- iter: 15400, loss: 0.5855 -----
#     ---- iter: 15500, loss: 0.5727 -----
#     ---- iter: 15600, loss: 0.5602 -----
#     ---- iter: 15700, loss: 0.548 -----
#     ---- iter: 15800, loss: 0.536 -----
#     ---- iter: 15900, loss: 0.5243 -----
#     ---- iter: 16000, loss: 0.5128 -----
#     ---- iter: 16100, loss: 0.5017 -----
#
#  .. figure:: ../demonstrations/function_fitting_qsp/trained_poly.png
#     :align: center
#     :width: 50%
#


######################################################################
# We were able to fit that polynomial quite well!
# Lets try something more challenging: fitting a
# non-polynomial function. One thing to keep in mind is the symmetry and
# bounds constraints on our polynomials. If our target function does not
# satisfy them as well, then we cannot hope to generate a good polynomial
# fit, regardless of how long we train for.
#
# A good non-polynomial candidate to fit to, that obeys our constraints,
# is the step function. Let’s try it!
#

d = 9  # dim(phi) = d + 1,
num_samples = 50

def step_func(x):
    """A step function (odd parity) which maps all values <= 0 to -1
    and all values > 0 to +1.
    """
    res = [-1.0 if x_i <= 0 else 1.0 for x_i in x]
    return torch.tensor(res, requires_grad=False, dtype=torch.float)

a_vals = np.linspace(-1, 1, num_samples)
y_true = step_func(a_vals)

qsp_model_runner = Model_Runner(
    QSP_Func_Fit, d, num_samples, a_vals, generate_many_sro, y_true
)

qsp_model_runner.execute()
qsp_model_runner.plot_result()


######################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     ---- iter: 100, loss: 32.0297 -----
#     ---- iter: 200, loss: 30.2728 -----
#     ---- iter: 300, loss: 28.589 -----
#     ---- iter: 400, loss: 26.9837 -----
#     ---- iter: 500, loss: 25.4604 -----
#     ---- iter: 600, loss: 24.0214 -----
#     ---- iter: 700, loss: 22.6672 -----
#     ---- iter: 800, loss: 21.3975 -----
#     ---- iter: 900, loss: 20.2107 -----
#     ---- iter: 1000, loss: 19.1044 -----
#     ---- iter: 1100, loss: 18.0754 -----
#     ---- iter: 1200, loss: 17.1204 -----
#     ---- iter: 1300, loss: 16.2354 -----
#     ---- iter: 1400, loss: 15.4164 -----
#     ---- iter: 1500, loss: 14.6593 -----
#     ---- iter: 1600, loss: 13.9598 -----
#     ---- iter: 1700, loss: 13.3141 -----
#     ---- iter: 1800, loss: 12.7181 -----
#     ---- iter: 1900, loss: 12.1681 -----
#     ---- iter: 2000, loss: 11.6606 -----
#     ---- iter: 2100, loss: 11.1921 -----
#     ---- iter: 2200, loss: 10.7596 -----
#     ---- iter: 2300, loss: 10.3602 -----
#     ---- iter: 2400, loss: 9.9912 -----
#     ---- iter: 2500, loss: 9.65 -----
#     ---- iter: 2600, loss: 9.3343 -----
#     ---- iter: 2700, loss: 9.0421 -----
#     ---- iter: 2800, loss: 8.7714 -----
#     ---- iter: 2900, loss: 8.5204 -----
#     ---- iter: 3000, loss: 8.2875 -----
#     ---- iter: 3100, loss: 8.0712 -----
#     ---- iter: 3200, loss: 7.8702 -----
#     ---- iter: 3300, loss: 7.6833 -----
#     ---- iter: 3400, loss: 7.5092 -----
#     ---- iter: 3500, loss: 7.347 -----
#     ---- iter: 3600, loss: 7.1956 -----
#     ---- iter: 3700, loss: 7.0544 -----
#     ---- iter: 3800, loss: 6.9224 -----
#     ---- iter: 3900, loss: 6.799 -----
#     ---- iter: 4000, loss: 6.6835 -----
#     ---- iter: 4100, loss: 6.5753 -----
#     ---- iter: 4200, loss: 6.4738 -----
#     ---- iter: 4300, loss: 6.3786 -----
#     ---- iter: 4400, loss: 6.2892 -----
#     ---- iter: 4500, loss: 6.2051 -----
#     ---- iter: 4600, loss: 6.1261 -----
#     ---- iter: 4700, loss: 6.0516 -----
#     ---- iter: 4800, loss: 5.9815 -----
#     ---- iter: 4900, loss: 5.9153 -----
#     ---- iter: 5000, loss: 5.8529 -----
#     ---- iter: 5100, loss: 5.7939 -----
#     ---- iter: 5200, loss: 5.7382 -----
#     ---- iter: 5300, loss: 5.6855 -----
#     ---- iter: 5400, loss: 5.6356 -----
#     ---- iter: 5500, loss: 5.5883 -----
#     ---- iter: 5600, loss: 5.5435 -----
#     ---- iter: 5700, loss: 5.501 -----
#     ---- iter: 5800, loss: 5.4606 -----
#     ---- iter: 5900, loss: 5.4223 -----
#     ---- iter: 6000, loss: 5.3859 -----
#     ---- iter: 6100, loss: 5.3512 -----
#     ---- iter: 6200, loss: 5.3182 -----
#     ---- iter: 6300, loss: 5.2868 -----
#     ---- iter: 6400, loss: 5.2569 -----
#     ---- iter: 6500, loss: 5.2283 -----
#     ---- iter: 6600, loss: 5.2011 -----
#     ---- iter: 6700, loss: 5.1751 -----
#     ---- iter: 6800, loss: 5.1503 -----
#     ---- iter: 6900, loss: 5.1265 -----
#     ---- iter: 7000, loss: 5.1038 -----
#     ---- iter: 7100, loss: 5.0821 -----
#     ---- iter: 7200, loss: 5.0613 -----
#     ---- iter: 7300, loss: 5.0414 -----
#     ---- iter: 7400, loss: 5.0223 -----
#     ---- iter: 7500, loss: 5.004 -----
#     ---- iter: 7600, loss: 4.9864 -----
#     ---- iter: 7700, loss: 4.9695 -----
#     ---- iter: 7800, loss: 4.9534 -----
#     ---- iter: 7900, loss: 4.9378 -----
#     ---- iter: 8000, loss: 4.9228 -----
#     ---- iter: 8100, loss: 4.9084 -----
#     ---- iter: 8200, loss: 4.8946 -----
#     ---- iter: 8300, loss: 4.8813 -----
#     ---- iter: 8400, loss: 4.8684 -----
#     ---- iter: 8500, loss: 4.856 -----
#     ---- iter: 8600, loss: 4.8441 -----
#     ---- iter: 8700, loss: 4.8326 -----
#     ---- iter: 8800, loss: 4.8215 -----
#     ---- iter: 8900, loss: 4.8108 -----
#     ---- iter: 9000, loss: 4.8005 -----
#     ---- iter: 9100, loss: 4.7905 -----
#     ---- iter: 9200, loss: 4.7808 -----
#     ---- iter: 9300, loss: 4.7715 -----
#     ---- iter: 9400, loss: 4.7624 -----
#     ---- iter: 9500, loss: 4.7537 -----
#     ---- iter: 9600, loss: 4.7452 -----
#     ---- iter: 9700, loss: 4.737 -----
#     ---- iter: 9800, loss: 4.7291 -----
#     ---- iter: 9900, loss: 4.7214 -----
#     ---- iter: 10000, loss: 4.7139 -----
#     ---- iter: 10100, loss: 4.7067 -----
#     ---- iter: 10200, loss: 4.6997 -----
#     ---- iter: 10300, loss: 4.6928 -----
#     ---- iter: 10400, loss: 4.6862 -----
#     ---- iter: 10500, loss: 4.6798 -----
#     ---- iter: 10600, loss: 4.6736 -----
#     ---- iter: 10700, loss: 4.6675 -----
#     ---- iter: 10800, loss: 4.6616 -----
#     ---- iter: 10900, loss: 4.6559 -----
#     ---- iter: 11000, loss: 4.6503 -----
#     ---- iter: 11100, loss: 4.6448 -----
#     ---- iter: 11200, loss: 4.6396 -----
#     ---- iter: 11300, loss: 4.6344 -----
#     ---- iter: 11400, loss: 4.6294 -----
#     ---- iter: 11500, loss: 4.6245 -----
#     ---- iter: 11600, loss: 4.6197 -----
#     ---- iter: 11700, loss: 4.6151 -----
#     ---- iter: 11800, loss: 4.6106 -----
#     ---- iter: 11900, loss: 4.6061 -----
#     ---- iter: 12000, loss: 4.6018 -----
#     ---- iter: 12100, loss: 4.5976 -----
#     ---- iter: 12200, loss: 4.5935 -----
#     ---- iter: 12300, loss: 4.5895 -----
#     ---- iter: 12400, loss: 4.5856 -----
#     ---- iter: 12500, loss: 4.5818 -----
#     ---- iter: 12600, loss: 4.578 -----
#     ---- iter: 12700, loss: 4.5744 -----
#     ---- iter: 12800, loss: 4.5708 -----
#     ---- iter: 12900, loss: 4.5673 -----
#     ---- iter: 13000, loss: 4.5639 -----
#     ---- iter: 13100, loss: 4.5605 -----
#     ---- iter: 13200, loss: 4.5573 -----
#     ---- iter: 13300, loss: 4.554 -----
#     ---- iter: 13400, loss: 4.5509 -----
#     ---- iter: 13500, loss: 4.5478 -----
#     ---- iter: 13600, loss: 4.5448 -----
#     ---- iter: 13700, loss: 4.5419 -----
#     ---- iter: 13800, loss: 4.539 -----
#     ---- iter: 13900, loss: 4.5361 -----
#     ---- iter: 14000, loss: 4.5333 -----
#     ---- iter: 14100, loss: 4.5306 -----
#     ---- iter: 14200, loss: 4.5279 -----
#     ---- iter: 14300, loss: 4.5253 -----
#     ---- iter: 14400, loss: 4.5227 -----
#     ---- iter: 14500, loss: 4.5202 -----
#     ---- iter: 14600, loss: 4.5177 -----
#     ---- iter: 14700, loss: 4.5153 -----
#     ---- iter: 14800, loss: 4.5129 -----
#     ---- iter: 14900, loss: 4.5105 -----
#     ---- iter: 15000, loss: 4.5082 -----
#     ---- iter: 15100, loss: 4.506 -----
#     ---- iter: 15200, loss: 4.5037 -----
#     ---- iter: 15300, loss: 4.5016 -----
#     ---- iter: 15400, loss: 4.4994 -----
#     ---- iter: 15500, loss: 4.4973 -----
#     ---- iter: 15600, loss: 4.4952 -----
#     ---- iter: 15700, loss: 4.4932 -----
#     ---- iter: 15800, loss: 4.4911 -----
#     ---- iter: 15900, loss: 4.4891 -----
#     ---- iter: 16000, loss: 4.4872 -----
#     ---- iter: 16100, loss: 4.4853 -----
#     ---- iter: 16200, loss: 4.4834 -----
#     ---- iter: 16300, loss: 4.4815 -----
#     ---- iter: 16400, loss: 4.4797 -----
#     ---- iter: 16500, loss: 4.4779 -----
#     ---- iter: 16600, loss: 4.4761 -----
#     ---- iter: 16700, loss: 4.4743 -----
#     ---- iter: 16800, loss: 4.4726 -----
#     ---- iter: 16900, loss: 4.4709 -----
#     ---- iter: 17000, loss: 4.4693 -----
#     ---- iter: 17100, loss: 4.4676 -----
#     ---- iter: 17200, loss: 4.466 -----
#     ---- iter: 17300, loss: 4.4644 -----
#     ---- iter: 17400, loss: 4.4628 -----
#     ---- iter: 17500, loss: 4.4613 -----
#     ---- iter: 17600, loss: 4.4597 -----
#     ---- iter: 17700, loss: 4.4582 -----
#     ---- iter: 17800, loss: 4.4567 -----
#     ---- iter: 17900, loss: 4.4552 -----
#     ---- iter: 18000, loss: 4.4538 -----
#     ---- iter: 18100, loss: 4.4523 -----
#     ---- iter: 18200, loss: 4.4509 -----
#     ---- iter: 18300, loss: 4.4495 -----
#     ---- iter: 18400, loss: 4.4481 -----
#     ---- iter: 18500, loss: 4.4467 -----
#     ---- iter: 18600, loss: 4.4454 -----
#     ---- iter: 18700, loss: 4.4441 -----
#     ---- iter: 18800, loss: 4.4427 -----
#     ---- iter: 18900, loss: 4.4414 -----
#     ---- iter: 19000, loss: 4.4402 -----
#     ---- iter: 19100, loss: 4.4389 -----
#     ---- iter: 19200, loss: 4.4376 -----
#     ---- iter: 19300, loss: 4.4364 -----
#     ---- iter: 19400, loss: 4.4352 -----
#     ---- iter: 19500, loss: 4.434 -----
#     ---- iter: 19600, loss: 4.4328 -----
#     ---- iter: 19700, loss: 4.4316 -----
#     ---- iter: 19800, loss: 4.4304 -----
#     ---- iter: 19900, loss: 4.4293 -----
#     ---- iter: 20000, loss: 4.4281 -----
#     ---- iter: 20100, loss: 4.427 -----
#     ---- iter: 20200, loss: 4.4259 -----
#     ---- iter: 20300, loss: 4.4248 -----
#     ---- iter: 20400, loss: 4.4237 -----
#     ---- iter: 20500, loss: 4.4226 -----
#     ---- iter: 20600, loss: 4.4216 -----
#     ---- iter: 20700, loss: 4.4205 -----
#     ---- iter: 20800, loss: 4.4195 -----
#     ---- iter: 20900, loss: 4.4184 -----
#     ---- iter: 21000, loss: 4.4174 -----
#     ---- iter: 21100, loss: 4.4164 -----
#     ---- iter: 21200, loss: 4.4154 -----
#     ---- iter: 21300, loss: 4.4144 -----
#     ---- iter: 21400, loss: 4.4134 -----
#     ---- iter: 21500, loss: 4.4124 -----
#     ---- iter: 21600, loss: 4.4115 -----
#     ---- iter: 21700, loss: 4.4106 -----
#     ---- iter: 21800, loss: 4.4096 -----
#     ---- iter: 21900, loss: 4.4087 -----
#     ---- iter: 22000, loss: 4.4078 -----
#     ---- iter: 22100, loss: 4.4069 -----
#     ---- iter: 22200, loss: 4.406 -----
#     ---- iter: 22300, loss: 4.4051 -----
#     ---- iter: 22400, loss: 4.4042 -----
#     ---- iter: 22500, loss: 4.4034 -----
#     ---- iter: 22600, loss: 4.4025 -----
#     ---- iter: 22700, loss: 4.4016 -----
#     ---- iter: 22800, loss: 4.4008 -----
#     ---- iter: 22900, loss: 4.4 -----
#     ---- iter: 23000, loss: 4.3991 -----
#     ---- iter: 23100, loss: 4.3983 -----
#     ---- iter: 23200, loss: 4.3975 -----
#     ---- iter: 23300, loss: 4.3967 -----
#     ---- iter: 23400, loss: 4.3959 -----
#     ---- iter: 23500, loss: 4.3951 -----
#     ---- iter: 23600, loss: 4.3943 -----
#     ---- iter: 23700, loss: 4.3935 -----
#     ---- iter: 23800, loss: 4.3927 -----
#     ---- iter: 23900, loss: 4.392 -----
#     ---- iter: 24000, loss: 4.3912 -----
#     ---- iter: 24100, loss: 4.3905 -----
#     ---- iter: 24200, loss: 4.3897 -----
#     ---- iter: 24300, loss: 4.389 -----
#     ---- iter: 24400, loss: 4.3882 -----
#     ---- iter: 24500, loss: 4.3875 -----
#     ---- iter: 24600, loss: 4.3868 -----
#     ---- iter: 24700, loss: 4.3861 -----
#     ---- iter: 24800, loss: 4.3853 -----
#     ---- iter: 24900, loss: 4.3846 -----
#     ---- iter: 25000, loss: 4.3839 -----
#
#  .. figure:: ../demonstrations/function_fitting_qsp/trained_step.png
#     :align: center
#     :width: 50%
#


######################################################################
# Conclusion:
# ~~~~~~~~~~~~~~
#
# In this demo, we explored the Quantum Signal Processing theorem. We
# showed that one could use a simple gradient descent model to train a
# parameter vector :math:`\vec{\phi}` to generate reasonably good
# polynomial approximations of arbitrary functions (provided the function
# satisfied the same constraints).
#


######################################################################
# .. figure:: ../demonstrations/function_fitting_qsp/trained_step.gif
#     :align: center
#     :width: 50%
#

######################################################################
# References:
# ~~~~~~~~~~~~~~
#
# [1]: *John M. Martyn, Zane M. Rossi, Andrew K. Tan, Isaac L. Chuang. “A
# Grand Unification of Quantum Algorithms”*\ `PRX Quantum 2,
# 040203 <https://arxiv.org/abs/2105.02859>`__\ *, 2021.*
#


##############################################################################
# .. bio:: Jay Soni
#    :photo: ../_static/authors/jay_soni.png
#
#    Jay completed his BSc. in Mathematical Physics from the University of Waterloo and currently works as a Quantum Software Developer at Xanadu. Fun fact, you will often find him sipping on a Tim Horton's IceCapp while he is coding.
#
