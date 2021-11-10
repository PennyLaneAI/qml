r"""

.. _adjoint_differentiation_benchmarking:

Adjoint Differentiation
=======================

.. meta::
    :property="og:description": Benchmarking file for adjoint diff demonstration.
    :property="og:image": https://pennylane.ai/qml/_static/thumbs/code.png

"""

##############################################################################
# This page is supplementary material to the
# `Adjoint Differentiation <https://pennylane.ai/qml/demos/tutorial_adjoint_diff.py`>__
# demonstration.  The below script produces the benchmarking images used.

import timeit
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

plt.style.use("bmh")

rng = np.random.default_rng(seed=42)

# Scaling over wires section

n_wires_ls = [3, 6,9,12,15, 18]

n_layers = 5

t_exec_wires = []
t_grad_wires = []
ratio_wires = []

for i_wires in n_wires_ls:
    
    dev = qml.device("lightning.qubit", wires=i_wires)
    
    @qml.qnode(dev, diff_method="adjoint")
    def circuit(params):
        qml.templates.StronglyEntanglingLayers(params, wires=range(i_wires))
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))
    
    # set up the parameters
    param_shape = qml.templates.StronglyEntanglingLayers.shape(n_wires=i_wires, n_layers=n_layers)
    params = rng.standard_normal(param_shape, requires_grad=True)
    
    ti_exec = timeit.timeit("circuit(params)", globals=globals(), number=100)
    t_exec_wires.append(ti_exec)
    
    ti_grad = timeit.timeit("qml.grad(circuit)(params)", globals=globals(), number=100)
    t_grad_wires.append(ti_grad)
    
    ratio_wires.append(ti_grad/ti_exec)

# Generating the graphic with wires scaling

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(n_wires_ls, t_exec_wires, '.-', label="circuit")
ax1.plot(n_wires_ls, t_grad_wires, '.-', label="gradient")

ax1.legend()

ax1.set_xlabel("Number of wires")
ax1.set_ylabel("Log Time")
ax1.set_yscale("log")
#ax1.set_title("Circuit and gradient scaling")
              
ax2.plot(n_wires_ls, ratio_wires)
ax2.set_xlabel("Number of wires")
ax2.set_ylabel("Gradient/ circuit time")
#ax2.set_title("Ratio between gradient and circuit time")

fig.suptitle("Scaling over wires", fontsize="xx-large")

plt.savefig("adjoint_diff/wires_scaling.png")

# Scaling over layers section

n_wires = 9
dev = qml.device("lightning.qubit", wires=n_wires)
@qml.qnode(dev, diff_method="adjoint")
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

n_layers_ls = [1,2,3,4,5,6,7,8]

t_exec_layers = []
t_grad_layers = []
ratio_layers = []

for i_layers in n_layers_ls:
    
    # set up the parameters
    param_shape = qml.templates.StronglyEntanglingLayers.shape(n_wires=n_wires, n_layers=i_layers)
    params = rng.standard_normal(param_shape, requires_grad=True)
    
    ti_exec = timeit.timeit("circuit(params)", globals=globals(), number=100)
    t_exec_layers.append(ti_exec)
    
    ti_grad = timeit.timeit("qml.grad(circuit)(params)", globals=globals(), number=100)
    t_grad_layers.append(ti_grad)
    
    ratio_layers.append(ti_grad/ti_exec)

# Graphic generation for layer scaling

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(n_layers_ls, t_exec_layers, '.-', label="circuit")
ax1.plot(n_layers_ls, t_grad_layers, '.-', label="gradient")

ax1.legend()

ax1.set_xlabel("Number of layers")
ax1.set_ylabel("Time")
#ax1.set_title("Circuit and gradient scaling")

ax2.plot(n_layers_ls, ratio_layers)
ax2.set_xlabel("Number of layers")
ax2.set_ylabel("Gradient / circuit time ratio")
#ax2.set_title("Ratio between gradient and circuit time")

fig.suptitle("Scaling with added layers", fontsize='xx-large')

plt.savefig("adjoint_diff/layers_scaling.png")

