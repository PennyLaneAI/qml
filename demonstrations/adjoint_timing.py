import timeit
import matplotlib.pyplot as plt
import pennylane as qml

plt.style.use("bmh")

n_wires = 4

dev = qml.device("lightning.qubit", wires=n_wires)

@qml.qnode(dev, diff_method="adjoint")
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


reps = 2
num = 3

n_layers = range(1, 21)

t_exec = []
t_grad = []
ratio = []
n_params = []

rng = np.random.default_rng(seed=42)

for i_layers in n_layers:
    
    # set up the parameters
    param_shape = qml.templates.StronglyEntanglingLayers.shape(n_wires=n_wires, n_layers=i_layers)
    params = rng.standard_normal(param_shape)
    params.requires_grad = True
    n_params.append(params.size)
    
    
    ti_exec_set = timeit.repeat("circuit(params)", globals=globals(), number=num, repeat=reps)
    ti_exec = min(ti_exec_set)/num
    t_exec.append(ti_exec)
    
    ti_grad_set = timeit.repeat("qml.grad(circuit)(params)", globals=globals(), number=num, repeat=reps)
    ti_grad = min(ti_grad_set)/num
    t_grad.append(ti_grad)
    
    ratio.append(ti_grad/ti_exec)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(n_params, t_exec, '.-', label="execution")
ax.plot(n_params, t_grad, '.-', label="gradient")

ax.legend()

ax.set_xlabel("Number of parameters")
ax.set_ylabel("Time")

plt.show()

n_params = np.array(n_params)

m, b = np.polyfit(n_params, ratio, deg=1)
ratio_fit = lambda x: m*x+b

print(f"ratio fit: {m}*x + {b}")

fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

ax2.plot(n_params, ratio, '.-', label="ratio")
ax2.plot(n_params, ratio_fit(n_params), label=f"{m:.3f}*x + {b:.2f}")

fig2.suptitle("Gradient time per execution time")
ax2.set_xlabel("number of parameters")
ax2.set_ylabel("Normalized Time")
ax2.legend()

plt.show()
