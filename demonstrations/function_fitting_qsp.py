r"""

.. function_fitting_qsp:

Function Fitting using Quantum Signal Processing (QSP)
======================================================

.. meta::
    :property="og:description": Learn how to create polynomial approximations to functions
        using Quantum Signal Processing (QSP).
    :property="og:image": https://pennylane.ai/qml/demonstrations/function_fitting_qsp/cover.png

1. Introduction:
~~~~~~~~~~~~~~~~

This demo was inspired by the paper `‘A Grand Unification of Quantum
Algorithms’ <https://arxiv.org/abs/2105.02859>`__ which may sound like a
very ambitious title, yet the authors fully deliver. This paper centers
around the ‘Quantum Singular Value Transform’ (QSVT) protocol and how it
provides a single framework to generalize some of the most famous
quantum algorithms (from Shor’s factoring and Grover search, to
hamiltonian simulation and more).

The QSVT is a method to apply polynomial transformations to the singular
values of *any* matrix (does not have to be square). From polynomial
transformations we can generate arbitrary function transformations
(using taylor approximations). The QSVT protocol is an extension of the
more constrained ‘Quantum Signal Processing’ (QSP) protocol which
presents a method for polynomial transformation of matrix entries in a
single-qubit unitary operator. The QSVT protocol is quite sophisticated
and involved in its derivation, but the idea at its core is quite
simple. By studying QSP, we get a relatively simpler path to explore
this idea at the foundation of QSVT.

In this demo, we will explore the QSP protocol and how it can be used
for curve fitting. This is a powerful tool that will ultimately allow us
to approximate *any function* on the interval :math:`[-1, 1]` that
satisfies certain constraints. Before we can dive into function fitting,
let’s develop some intuition. Consider the following operator
parameterized by :math:`a \in [-1, 1]`:

.. math:: \hat{W}(a) = \begin{bmatrix} a & i\sqrt{1 - a^{2}} \\ i\sqrt{1 - a^{2}} & a \end{bmatrix}

:math:`\hat{W}(a)` is called the *Signal Rotation Operator* (SRO). In
this particular case we are rotating around the x-axis but it can, in
general, take other forms. Using this operator, we can construct another
operator called the *Signal Processing Operator* (SPO),

.. math::  \hat{U}_{sp} = \hat{R}_{z}(\phi_{0}) \prod_{k=1}^{d} \hat{W}(a) \hat{R}_{z}(\phi_{k})

The SPO is parameterized by a vector
:math:`\vec{\phi} \in \mathbb{R}^{d+1}`, where :math:`d` is a free
parameter which represents the number of repeated applications
:math:`\hat{W}(a)`.

The SPO alternates between applying the SRO and parameterized rotations
around the z-axis. Let’s see what happens when we try to compute the
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
suppose we wanted to create a map $ S: a :raw-latex:`\to `2a^2 - 1$.
This expectation value would give us the means to perform such a mapping
:math:`S`. This may seem oddly specific at first, but it turns out that
this process can be generalized for generating a mapping
:math:`S: a \to \text{poly}(a)`. The following theorem shows us how:

Theorem: Quantum Signal Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a vector :math:`\vec{\phi} \in \mathbb{R}^{d+1}`, there exist
complex polynomials :math:`P(a)` and :math:`Q(a)` such that the SPO,
:math:`\hat{U}_{sp}`, can be expressed in matrix form as:

.. math::  \hat{U}_{sp} = \hat{R}_{z}(\phi_{0}) \prod_{k=1}^{d} \hat{W}(a) \hat{R}_{z}(\phi_{k})

.. math::


   \hat{U}_{sp} = \begin{bmatrix} P(a) & iQ(a)\sqrt{1 - a^{2}} \\ iQ^{*}(a)\sqrt{1 - a^{2}} & P^{*}(a) \end{bmatrix}

Where :math:`a \in [-1, 1]` and the polynomials :math:`P(a)`,
:math:`Q(a)` satisfy the following constraints:

-  :math:`deg(P) \leq d \ ` and :math:`deg(Q) \leq d - 1`
-  :math:`P` has parity :math:`d` mod 2 and :math:`Q` has parity
   :math:`d - 1` mod 2
-  :math:`|P|^{2} + (1 - a^{2})|Q|^{2} = 1`

**Note:** *Condition 3 is actually quite restrictive because if we
substitute :math:`a = \pm 1`, we get the result
:math:`|P^{2}(\pm 1)| = 1`. Thus it restricts our polynomial to be
pinned to :math:`\pm 1` at the end points of our domain
:math:`a = \pm 1`. This condition can be relaxed to
:math:`|P^{2}(a)| \leq 1` by expressing our signal processing operator
in the Hadamard basis (ie.
:math:`\bra{+}\hat{U}_{sp}(\vec{\phi};a)\ket{+}`). This is equivalent to
redefining :math:`P(a)` such that:*

.. math:: P^{'}(a) = \text{Re}(P(a)) + i\text{Re}(Q(a))\sqrt{1 - a^{2}}

*This is the convention we follow in this demo.*

"""

# """
# .. math:: \newcommand{\ket}[1]{\left|{#1}\right\rangle}
#
# .. math:: \newcommand{\bra}[1]{\left\langle{#1}\right|}
#
# """

######################################################################
# 2. Lets Plot some Polynomials:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let us put this theorem to the test! In this section we will construct
# the SRO (:math:`\hat{W}(a)`), and then use PennyLane to define the SPO.
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
# with each element.
# 

def compute_signal_rotation_mat(a):
    """Given a fixed value 'a', compute the signal rotation matrix W(a).
         (requires -1 <= 'a' <= 1)
    """
    diag = a
    off_diag = (1 - a**2)**(1/2) * 1j

    W = [[diag,     off_diag],
         [off_diag,     diag]]
    
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
# PennyLane to define a quantum function which will compute the SPO
# (:math:`\hat{U}_{sp}`). Recall we are measuring in the Hadamard basis to
# relax the 3rd condition of the theorem (*see note in introduction*). We
# accomplish this by sandwiching the SPO between two Hadamard gates to
# account for this change of basis.
# 

def QSP_circ(phi, W):
    """This circuit applies the SPO, the components in the matrix 
    representation of the final Unitary are polynomials! 
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
# :math:`a`. In this case we choose :math:`d = 5`. Due to the conditions
# of the theorem, we expect to observe the following:
# 
# -  Since :math:`d \ \text{mod}(2) = 1` is odd, we expect all of the
#    polynomials we plot to have odd symmetry
# -  Since :math:`d = 5`, we expect none of the polynomials will have
#    terms ~ :math:`O(a^6)`
# -  All of the polynomials are bounded by :math:`\pm1`
# 

d = 5 
a_vals = torch.linspace(-1, 1, 50)
w_mats = generate_many_sro(a_vals)

gen = torch.Generator()
gen.manual_seed(444422)

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
# Exactly as predicted, all of these conditions are met!
# 
# -  all curves are odd in :math:`a`
# -  we observe a good spread between the degree of each polynomial
#    plotted (eg. polynomial #3 looks linear, polynomial #1 and #2 look
#    cubic, polynomial #0 has degree 5), all of which have degree
#    :math:`\leq 5`
# -  each plot does not exceed :math:`\pm1` !
# 
# 3. Function Fitting with Quantum Signal Processing:
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
# **“*Can we learn the parameter values of :math:`\vec{\phi}` to transform
# our signal processing operator polynomial to fit a given function?*”**.
# 
# In order to answer this question, we begin by building a machine
# learning model using Pytorch. The ``__init__()`` method handles the
# random initialization of our parameter vector :math:`\vec{\phi}`. The
# ``forward`` method takes an array of signal rotation matrices
# (:math:`\hat{W}(a)` where each entry corresponds to a different
# :math:`a`), and produces the predicted y values.
# 
# Next we make use of the PennyLane function ``qml.matrix()``, which
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
        """PennyLane forward implementation (~ 10 - 20 mins to converge)"""
        y_pred = []
        generate_qsp_mat = qml.matrix(QSP_circ)
        
        for w in omega_mats:
            u_qsp = generate_qsp_mat(self.phi, w)
            P_a = u_qsp[0, 0]       # Taking the (0,0) entry of the matrix corresponds to <0|U|0> 
            y_pred.append(P_a.real)
        
        return torch.stack(y_pred, 0)


######################################################################
# Next we create a ``Model_Runner`` class to handle running the
# optimization, storing the results, and providing plotting functionality:
# 

class Model_Runner():
    
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

    def execute(self, random_seed=13021967, max_shots=25000, verbose=True):  # easter egg: oddly specific seed? 
        """Run the optimization protocol on the model using Mean Square Error as a loss
        function and using stochastic gradient descent as the optimizer.
        """
        model = self.model(degree=self.degree, num_vals=self.num_samples, random_seed=random_seed)
        
        criterion = torch.nn.MSELoss(reduction='sum')
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
        plt.plot(self.x_vals, self.y_true.tolist(), '--b', label="target func")
        plt.plot(self.x_vals, self.y_pred.tolist(), '.g', label="optim params")
        plt.plot(self.x_vals, self.init_y_pred.tolist(), '.r', label="init params")
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


d = 9  # dim(phi) = d + 1,
num_samples = 50


def custom_poly(x):
    """A custom polynomial of degree <= d and parity d % 2"""
    return torch.tensor(4*x**5 - 5*x**3 + x, requires_grad=False, dtype=torch.float)
#     return torch.tensor(-4*x*(x - 0.5)*(x+0.5), requires_grad=False, dtype=torch.float)


a_vals = np.linspace(-1, 1, num_samples)
y_true = custom_poly(a_vals)

qsp_model_runner = Model_Runner(QSP_Func_Fit, d, num_samples, a_vals, generate_many_sro, y_true)
qsp_model_runner.execute()

qsp_model_runner.plot_result()


######################################################################
# We were able to fit that polynomial almost perfectly (within the given
# iteration limit)! Lets try something more challenging: fitting a
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

qsp_model_runner = Model_Runner(QSP_Func_Fit, d, num_samples, a_vals, generate_many_sro, y_true)
qsp_model_runner.execute()

qsp_model_runner.plot_result()


######################################################################
# 4. Conclusion:
# ~~~~~~~~~~~~~~
# 
# In this demo, we explored the Quantum Signal Processing theorem. We
# showed that one could use a simple gradient descent model to train a
# parameter vector :math:`\vec{\phi}` to generate a reasonably good
# polynomial approximation of an arbitrary function (provided the function
# satisfied the same constraints).
# 
# 5. References:
# ~~~~~~~~~~~~~~
# 
# [1]: *John M. Martyn, Zane M. Rossi, Andrew K. Tan, Isaac L. Chuang. “A
# Grand Unification of Quantum Algorithms”*\ `PRX Quantum 2,
# 040203 <https://arxiv.org/abs/2105.02859>`__\ *, 2021.*
# 
