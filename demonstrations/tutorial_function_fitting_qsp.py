"""
.. math:: \newcommand{\ket}[1]{\left|{#1}\right\rangle}

.. math:: \newcommand{\bra}[1]{\left\langle{#1}\right|}

"""


######################################################################
# Function Fitting using Quantum Signal Processing (QSP)
# ======================================================
# 
# 1. Introduction:
# ~~~~~~~~~~~~~~~~
# 
# In this demo we explore the `‘Quantum Signal Processing’ <link>`__ (QSP)
# protocol and how it can be used to fit polynomials. Before we can dive
# into function fitting, lets develope some intuition. Consider the
# following operator parameterized by :math:`a \in [-1, 1]`:
# 
# .. math:: \hat{W}(a) = \begin{bmatrix} a & i\sqrt{1 - a^{2}} \\ i\sqrt{1 - a^{2}} & a \end{bmatrix}
# 
# This operator is called the ‘Signal Rotation Operator’, in this
# particular case we are using a rotation around the x-axis, but in
# general this operator can take other forms. Using this operator we can
# construct a new operator called the ‘Signal Processing Operator’ which
# is parameterized by a vector :math:`\vec{\phi} \in \mathbb{R}^{d+1}`
# 
# .. math::  \hat{U}_{sp} = \hat{R}_{z}(\phi_{0}) \prod_{k=1}^{d} \hat{W}(a) \hat{R}_{z}(\phi_{k})
# 
# This operator alternates between applying the signal rotation operator
# and parameterized rotation around the z-axis. Let’s see what happens
# when we try to compute the expectation value of this operator
# :math:`\bra{0}\hat{U}_{sp}\ket{0}` for :math:`\vec{\phi} = (0, 0, 0)` :
# 
# .. math::
# 
# 
#    \begin{align*}
#    \bra{0}\hat{U}_{sp}\ket{0} &= \bra{0} \ \hat{R}_{z}(0) \prod_{k=1}^{2} \hat{W}(a) \hat{R}_{z}(0) \ \ket{0} \\
#    \bra{0}\hat{U}_{sp}\ket{0} &= \bra{0} \hat{W}(a)^{2} \ket{0} \\
#    \end{align*}
# 
# .. math::
# 
# 
#    \bra{0}\hat{U}_{sp}\ket{0} = \bra{0} \begin{bmatrix} a & i\sqrt{1 - a^{2}} \\ i\sqrt{1 - a^{2}} & a \end{bmatrix} \ \circ \ \begin{bmatrix} a & i\sqrt{1 - a^{2}} \\ i\sqrt{1 - a^{2}} & a \end{bmatrix} \ket{0}
# 
# .. math::
# 
# 
#    \bra{0}\hat{U}_{sp}\ket{0} = \bra{0} \begin{bmatrix} 2a^{2} - 1 & 2ai\sqrt{1 - a^{2}} \\ 2ai\sqrt{1 - a^{2}} & 2a^{2} - 1 \end{bmatrix} \ket{0}
# 
# .. math::  \bra{0}\hat{U}_{sp}\ket{0} = 2a^{2} - 1 
# 
# Notice that this quantity is a polynomial in :math:`a`. So suppose we
# were given a signal $ S: x :raw-latex:`\to `a $, then this expectation
# value would give us a means to map $ x :raw-latex:`\to `2a^{2} - 1$.
# This may seem oddly specific at first, but it turns out that in general:
# 
# Theorem: Quantum Signal Processing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Given a :math:`\vec{\phi} \in \mathbb{R}^{d+1}`, there exist complex
# ploynomials :math:`P(a)` and :math:`Q(a)` such that the signal
# processing operator :math:`\hat{U}_{sp}` can be expressed in matrix form
# as (for :math:`a \in [-1, 1]`):
# 
# .. math::  \hat{U}_{sp} = \hat{R}_{z}(\phi_{0}) \prod_{k=1}^{d} \hat{W}(a) \hat{R}_{z}(\phi_{k}) 
# 
# .. math::
# 
#     
#    \hat{U}_{sp} = \begin{bmatrix} P(a) & iQ(a)\sqrt{1 - a^{2}} \\ iQ^{*}(a)\sqrt{1 - a^{2}} & P^{*}(a) \end{bmatrix} 
# 
# Where the polynomials satisfy the following constraints:
# 
# -  :math:`deg(P) \leq d \ ` and $  deg(Q) :raw-latex:`\leq `d - 1$
# -  :math:`P` has parity d mod 2 and :math:`Q` has parity d - 1 mod 2
# -  :math:`|P|^{2} + (1 - a^{2})|Q|^{2} = 1`
# 
# **Note:** *Condition 3 is actually quite restrictive because we have
# :math:`|P^{2}(\pm 1)| = 1`. Thus it restricts our polynomials to those
# which pass through :math:`\pm 1` for :math:`a = \pm 1`. This condition
# can be relaxed to :math:`|P^{2}(a)| \leq 1` by expressing our signal
# processing operator in the Hadamard basis. This is equivalent to
# redefining :math:`P(a)` such that:*
# 
# .. math:: P^{'}(a) = Re(P(a)) + iRe(Q(a))\sqrt{1 - a^{2}}
# 
# *This is the convention we follow in this demo.*
# 
# 2. Lets Plot some Polynomials!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let us put this theorem to the test! In this section we will construct
# the signal rotation operator, and then use PennyLane to define the
# quantum signal processing sequence. To test the theorem we will randomly
# generate parameters :math:`\vec{\phi}` and plot the expectation value
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
# Here we implement the ``compute_signal_rotation_mat`` function, which
# will construct the signal rotation matrix. We also make a helper
# function which, given an array of :math:`a` values, will generate an
# array of :math:`\hat{W}(a)`
# 

def compute_signal_rotation_mat(a):
    """Given a fixed value 'a', compute the signal rotation matrix W(a).
         (requires -1 <= 'a' <= 1)
    """
    diag = a
    off_diag = (1 - a**2)**(1/2) * 1j

    W = [[diag    , off_diag],
         [off_diag,     diag]]
    
    return W


def generate_inp(a_vals):
    """Given a tensor of 'a' vals, return a tensor of W(a)"""
    w_array = [] 
    
    for a in a_vals: 
        w = compute_signal_rotation_mat(a)
        w_array.append(w)
    
    return torch.tensor(w_array, dtype=torch.complex64, requires_grad=False)


######################################################################
# Here we use PennyLane to define a quantum function which will compute
# our signal processing operator :math:`\hat{U}_{sp}`. Since we are
# measuring in the Hadamard basis (*see note in introduction*) we
# sandwhich the operator between two Hadamarad gates to account for this
# change of basis.
# 


def QSP_circ(phi, W):
    """This circuit applies the QSP operator, the components in the matrix 
    representation of the final Unitary are polynomials! 
    """
    qml.Hadamard(wires=0) # initial state |+> 
    
    for i in range(len(phi) - 1): # iterate through rotations in reverse order 
            qml.RZ(phi[i], wires=0)
            qml.QubitUnitary(W, wires=0)
    
    qml.RZ(phi[-1], wires=0) # final rotation
    qml.Hadamard(wires=0)   # change of basis |+> , |->
    
    return


######################################################################
# Finally, we randomly generate our :math:`\vec{\phi}` and plot the
# expectation value as a function of :math:`a`. In this case we choose
# :math:`d = 5`. As a result of this (due to the conditions of the
# theorem) we expect to observe the following:
# 
# -  Since :math:`d \mod 2 = 1` is odd, we expect all of the resultant
#    plots to have odd symmetry
# -  Since :math:`d = 5`, we expect that non of the polynomials will be of
#    degree :math:`\geq 6`
# -  All of the polynomials are bounded by :math:`\pm1`
# 

d = 5 
a_vals = torch.linspace(-1, 1, 50)
w_mats = generate_inp(a_vals)

gen = torch.Generator()
gen.manual_seed(499567)

for i in range(5):
    phi = torch.rand(d+1, generator=gen) * 2 * torch.tensor([math.pi], requires_grad=False)
    matrix_func = qml.matrix(QSP_circ)
    y_vals = [matrix_func(phi, w)[0, 0].real for w in w_mats]
    
    plt.plot(a_vals, y_vals, label=f"poly #{i}")

plt.vlines(0.0, -1.0, 1.0, color='black')
plt.hlines(0.0, -1.0, 1.0, color='black')
plt.legend(loc=1)
plt.show()


######################################################################
# All of these conditions are met! We see that indeed all of the plots are
# odd in symmetry. We observe a good spread between the degree of each
# plot (eg. poly #3 looks linear, poly #1 and #2 look cubic, poly #0 has
# degree 5), all of which are :math:`\leq 5`. We also see that each plot
# does not exceed :math:`\pm1` !
# 
# 3. Function Fitting with Quantum Signal Processing!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Another observation we can make about this theorem, is the fact that it
# holds true in both directions. This means that if we have two
# polynomials :math:`P(a), Q(a)` that satisfy the conditions of the
# theorem, then there exists a :math:`\vec{\phi}` for which we can
# construct a signal processing operator which maps :math:`a \to P(a)`.
# 
# In this section we try to answer the question:
# 
# **“*Can we learn the parameter values of :math:`\vec{\phi}` to transform
# our signal processing operator polynomial to fit a given function?*”**.
# 
# In order to answer this question, we begin by building a machine
# learning model (using torch). The ``__init__()`` method handles the
# random initialization of our parameter vector :math:`\vec{\phi}`. The
# ``forward`` method takes an array of signal rotation matricies
# (:math:`\hat{W}(a)` where each entry corresponds to a different
# :math:`a`), and produces the predicted y value.
# 
# Here we make use of the PennyLane function ``qml.matrix()`` which
# accepts our quantum function (it can also accept quantum tapes and
# QNodes) and returns its unitary matrix representation. We are interested
# in the real value of the top left entry (this corresponds to
# :math:`P(a)`).
# 

torch_pi = torch.Tensor([math.pi])


class QSPFuncFit(torch.nn.Module):
    
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
        """PennyLane forward implementation (~ 10 - 20 mins to converge)"""
        y_pred = []
        generate_qsp_mat = qml.matrix(QSP_circ)
        
        for w in omega_mats:
            u_qsp = generate_qsp_mat(self.phi, w)
            P_a = u_qsp[0, 0]
            y_pred.append(P_a.real)
        
        return torch.stack(y_pred, 0)

#     def forward(self, omega_mats):  #NOTE: Should we delete this? or keep it in and add a note for users? 
#         """Fast forward implementation using torch directly (~ 1 - 3 mins to converge)"""
        
#         hadamard = torch.sqrt(torch.tensor([1/2], dtype=torch.complex64)) * torch.tensor([[1., 1.], [1., -1.]], dtype=torch.complex64)
#         hadamard = (torch.unsqueeze(hadamard, 0)).expand(self.num_vals, -1, -1)
        
#         pauli_z = torch.Tensor([[1, 0], [0, -1]])
#         pauli_z = (torch.unsqueeze(pauli_z, 0)).expand(self.num_vals, -1, -1) # added dimension for omegas
#         pauli_z = (torch.unsqueeze(pauli_z, 0)).expand(self.num_phi, -1, -1, -1) # added dimension for phis
        
#         cmplx_phi = torch.complex(torch.zeros(self.num_phi), self.phi)
#         rz = (pauli_z.transpose(0 , 3) * cmplx_phi).transpose(0, 3)
#         rz = torch.reshape(rz, (self.num_phi * self.num_vals, 2, 2))
#         rz = torch.linalg.matrix_exp(rz)
#         rz = torch.reshape(rz, (self.num_phi, self.num_vals, 2, 2))
        
#         ident = torch.tensor([[1.0, 0.], [0., 1.0]], dtype=torch.complex64)
#         ident = (torch.unsqueeze(ident, 0)).expand(self.num_vals, -1, -1)
        
#         u_qsp = torch.matmul(hadamard, ident)
        
#         for i in range(self.num_phi - 1):
#             u_qsp = torch.matmul(rz[i], u_qsp)
#             u_qsp = torch.matmul(omega_mats, u_qsp)
        
#         u_qsp = torch.matmul(rz[-1], u_qsp)
#         u_qsp = torch.matmul(hadamard, u_qsp)
        
#         return u_qsp.real[:, 0, 0].to(torch.float)


######################################################################
# Here we create a ``ModelRunner`` class to handle running the
# optimization, storing the results and providing plotting functionality:
# 

class ModelRunner:
    
    def __init__(self, model, degree, num_samples, x_vals, process_x_vals, y_true):
        """Given a model and a series of model specific arguments, store everythign in 
        internal attributes.
        """
        self.model = model
        self.degree = degree
        self.num_samples = num_samples
        
        self.x_vals = x_vals 
        self.inp = process_x_vals(x_vals)
        self.y_true = y_true

    def execute(self, random_seed=2147483647, max_shots=25000, verbose=True):
        """Run the optimization protocol on the model using Mean Square Error as a loss
        function and using stochastic gradient descent as the optimizer.
        """
        model = self.model(degree=self.degree, num_vals=self.num_samples, random_seed=random_seed)
        
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        
        t = 1 
        successive_loss = 1.0
        loss_val = 1.0
        prev_loss = 2.0
        
        while (t <= max_shots) and (loss_val > 1e-2):
            
            y_pred = model(self.inp)

            if t == 1:
                self.init_y_pred = y_pred

            # Compute and print loss
            loss = criterion(y_pred, self.y_true)
            loss_val = loss.item()

            if t > 1: 
                successive_loss = loss_val - prev_loss


            if (t % 100 == 0) and verbose:
                print(f"---- iter: {t}, loss: {loss_val}, successive_loss: {successive_loss} -----")

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prev_loss = loss_val
            t+=1
        
        self.model_params = model.phi
        self.y_pred = y_pred
        
    def plot_result(self):
        """Plot the results"""
        plt.plot(self.x_vals, self.y_true.tolist(), '--b', label="target func")
        plt.plot(self.x_vals, self.y_pred.tolist(), '.g', label="optim params")
        plt.plot(self.x_vals, self.init_y_pred.tolist(), '.r', label="init params")
        plt.legend()
        plt.show()


######################################################################
# Now that we have a model, lets first attempt to fit a polynomial. We
# expect this to perform well when the target polynomial also obeys the
# symmetry and degree constraints that our quantum signal processing
# polynomial does. To do this, we defined a function ``custom_poly(x)``
# which implments the target polynomial. In this case, we (arbitrarily)
# choose the target polynomial:
# 
# .. math::  y = 4x^{5} - 5x^{3} + x 
# 
# Lets see how well we can fit this polynomial:
#

d = 9  # dim(phi) = d + 1,
num_samples = 50


def custom_poly(x):
    """A custom polynomial of degree <= d and parity d % 2"""
    return torch.tensor(4*x**5 - 5*x**3 + x, requires_grad=False, dtype=torch.float)


a_vals = np.linspace(-1, 1, num_samples)
y_true = custom_poly(a_vals)

qsp_model_runner = ModelRunner(QSPFuncFit, d, num_samples, a_vals, generate_inp, y_true)
qsp_model_runner.execute()
qsp_model_runner.plot_result()


######################################################################
# We were able to fit that polynomial almost perfectly (within the given
# iteration limit)! Lets try something more challenging. Lets try and fit
# our polynomials to a non-polynomial function. One thing to keep in mind
# is the symmetry and bounds constriants on our polynomials. If our target
# function does not satisfy them as well, then we cannot hope to generate
# a good polynomial fit, regardless of how long we train for.
# 
# Here we try to generate a polynomial approximation for the sign/step
# function:
# 

d = 9  # dim(phi) = d + 1,
num_samples = 50


def step_func(x):
    """A step function (odd parity) which maps all values <= 0 to -1 
    and all values > 0 to +1.
    """
    res = [-1.0 if x_i <= 0 else 1.0 for x_i in x]
    return torch.tensor(res, requires_grad=False, dtype=torch.float)


y_true = step_func(a_vals)

qsp_model_runner = ModelRunner(QSPFuncFit, d, num_samples, a_vals, generate_inp, y_true)
qsp_model_runner.execute()
qsp_model_runner.plot_result()


######################################################################
# 4. Conclusion:
# ~~~~~~~~~~~~~~
# 
# In this demo we explored the Quantum Signal Processing theorem. We
# showed that one could use a simple gradient descent model to train the
# parameter vector :math:`\vec{\phi}` to generate a reasonably good
# polynomial approximation to fit an arbitrary function (provided the
# function satisfied the same symmetry and bounds constrainints).
# 
# 5. References:
# ~~~~~~~~~~~~~~
# 
# [1]: *John M. Martyn, Zane M. Rossi, Andrew K. Tan, Isaac L. Chuang. “A
# Grand Unification of Quantum Algorithms”*\ `PRX Quantum 2,
# 040203 <https://arxiv.org/abs/2105.02859>`__\ *, 2021.*
# 
