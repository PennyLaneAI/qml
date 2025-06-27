
# %%
from pyscf import gto, scf, mcscf
import numpy as np

# Create a mol object
r = 0.71
geom = [['H', (0, 0, 0)],
        ['H', (0, 0, r)]]
basis = '631g'
mol = gto.Mole(atom=geom, basis=basis, symmetry=None)
mol.build()

# get MOs
hf = scf.RHF(mol)
hf.run()

charges = hf.mol.atom_charges()
coords = hf.mol.atom_coords()
nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
print("nuc charge centre", nuc_charge_center)
hf.mol.set_common_orig_(nuc_charge_center)

# %% [markdown]
# Next we will use the multiconfigurational self-consistent field methods (`pyscf.mcscf`) to solve for the expansion of the intitial wavefunction as a linear combination of Slater determinants. Running the configuration interaction (CI) method returns the wavefunction as as vector. We will filter out small values in the wavefunction. **Explain the shape of this vector.**

# %%
ncas = hf.mol.nao
nelecas = hf.mol.nelectron

mycasci = mcscf.CASCI(hf, ncas=ncas, nelecas=nelecas)
mycasci.run(verbose=0)

ncas_a = mycasci.ncas
ncas_b = ncas_a
nelecas_a, nelecas_b = mycasci.nelecas

cascivec = mycasci.ci

# filter out small values based on preset tolerance to save more memory
cascivec[abs(cascivec) < 1e-6] = 0
print("cascivec", cascivec)

# %%
dip_ints = hf.mol.intor('int1e_r_cart', comp=3)

# %%
orbcas = hf.mo_coeff
dip_ints = np.einsum('ik,xkl,lj->xij', orbcas.T, dip_ints, orbcas)

# %%
## INSERT CODE ##
dipole_rho = {(2, 1): -0.6902564137617815, (1, 2): -0.6902564137617815, 
                (8, 1): -0.1327113674508237, (1, 8): -0.1327113674508237, 
                (2, 4): -0.07024799874988287, (8, 4): 0.031606880290424764, 
                (4, 2): -0.07024799874988287, (4, 8): 0.03160688029042472}

dipole_norm = 1.3058

from pennylane.qchem.convert import _wfdict_to_statevector

wf_dip = _wfdict_to_statevector(dipole_rho, ncas)

import pennylane as qml

# initialization circuit for m_rho|I>
# propagating device does not use shots -- it does not work with qml.state()
device_type = "lightning.qubit"
dev_prop = qml.device(device_type, wires=int(2*ncas) + 1, shots=None)
@qml.qnode(dev_prop)
def initial_circuit(wf):
    # dipole wavefunction preparation
    qml.StatePrep(wf, wires=dev_prop.wires.tolist()[1:])
    qml.Hadamard(wires=0)
    return qml.state()

# %%
from pyscf import ao2mo

# create h1 -- one-body terms
h_core = hf.get_hcore(mol)
orbs = hf.mo_coeff
core_const = mol.energy_nuc()
one = np.einsum("qr,rs,st->qt", orbs.T, h_core, orbs)

# create h2 -- two-body terms
two = ao2mo.full(hf._eri, orbs, compact=False).reshape([mol.nao]*4)
two = np.swapaxes(two, 1, 3)

# to chemist notation
eri = np.einsum('prsq->pqrs', two)
h1e = one - np.einsum('pqrr->pq', two)/2.

# %%
# factorize hamiltonian, producing matrices
_, Z, U = qml.qchem.factorize(eri, compressed=True)

print("Shape of the factors: ")
print("eri", eri.shape)
print("U", U.shape)
print("Z", Z.shape)

approx_eri = qml.math.einsum("tpk,tqk,tkl,trl,tsl->pqrs", U, U, Z, U, U)
assert qml.math.allclose(eri, approx_eri, atol=1.5e-3)

# %%
# add one-body correction
Z_prime = np.stack([np.diag(np.sum(Z[i], axis = -1)) for i in range(Z.shape[0])], axis = 0)
obc = np.einsum('tpk,tkk,tqk->pq', U, Z_prime, U)

# Diagonalize the one-electron integral matrix
eigenvals, U0 = np.linalg.eigh(h1e + obc)
Z0 = np.diag(eigenvals)

# %%
def U_rotations(U, control_wires):
    """Circuit implementing the basis rotations of the CDF decomposition."""
    norb = U.shape[-1]
    qml.BasisRotation(unitary_matrix=U, wires = [int(2*i+control_wires) for i in range(norb)])
    qml.BasisRotation(unitary_matrix=U, wires = [int(2*i+1+control_wires) for i in range(norb)])

# %%
from itertools import product

def Z_rotations(Z, step, is_one_body_term, control_wires):
    """Circuit implementing the Z rotations of the CDF decomposition. 
    Note that t will range from t = 1 to t = ts, so we use t-1 in the code."""
    norb = Z.shape[-1]

    if is_one_body_term:
        for sigma in range(2):
            for i in range(norb):
                if abs(Z[i, i]) > 1e-15:
                    qml.ctrl(qml.X(wires=int(2*i+sigma+control_wires)),
                                        control = range(control_wires), control_values=0)
                    qml.RZ(-Z[i, i]*step/2, wires=int(2*i+sigma+control_wires))
                    qml.ctrl(qml.X(wires=int(2*i+sigma+control_wires)),
                                        control = range(control_wires), control_values=0)
        globalphase = np.sum(Z)*step

    else:  # a two body term
        for sigma, tau in product(range(2), repeat=2):
            for i, k in product(range(norb), repeat=2):
                if (i != k or sigma != tau) and abs(Z[i, k]) > 1e-15:  # Two body term
                    qml.ctrl(qml.X(wires=int(2*i+sigma+control_wires)), 
                            control = range(control_wires), control_values=0)
                    qml.MultiRZ(Z[i, k]/8.*step,
                            wires=[int(2*i+sigma+control_wires), int(2*k+tau+control_wires)])
                    qml.ctrl(qml.X(wires=int(2*i+sigma+control_wires)),
                            control = range(control_wires), control_values=0)
        globalphase = np.trace(Z)/4.*step - np.sum(Z)*step + np.sum(Z)*step/2.

    qml.PhaseShift(-globalphase, wires = 0)

# %%
def LieTrotter(step, prior_U, final_rotation, reverse=False):
    """Implements a first-order Trotterized circuit for the CDF."""
    _U0 = np.expand_dims(U0, axis = 0)
    _Z0 = np.expand_dims(Z0, axis = 0)
    _U = np.concatenate((_U0, U), axis = 0)
    _Z = np.concatenate((_Z0, Z), axis = 0)

    ts = U.shape[0]
    is_one_body = np.array([True] + [False]*ts)
    order = list(range(len(_Z)))

    if reverse: order = order[::-1]

    for t in order:
        U_rotations(prior_U @ _U[t], 1)
        Z_rotations(_Z[t], step, is_one_body[t], 1)
        prior_U = _U[t].T

    if final_rotation: U_rotations(prior_U, 1)

    qml.PhaseShift(-core_const*step, wires=0)

    return prior_U

# %%
def trotter_circuit(dev, state, step):
    """Implements a second-order Trotterized circuit for the CDF."""
    qubits = dev.wires.tolist()

    def circuit():
        # State preparation -- previous iteration
        qml.StatePrep(state, wires=qubits)

        # Main body of the circuit
        prior_U = np.eye(ncas)  # no inital prior U, so identity
        prior_U = LieTrotter(step/2., prior_U=prior_U, 
                        final_rotation=False, reverse=False)
        prior_U = LieTrotter(step/2., prior_U=prior_U, 
                    final_rotation=True, reverse=True)

        return qml.state()

    return qml.QNode(circuit, dev)

# %%
eta = 0.05
jmax = 40
shots = 1000
norm = 1.5
wgrid = np.linspace(-2, +5, 10000)
w_min, w_step = wgrid[0], wgrid[1] - wgrid[0]

tau = np.pi / (2 * norm)
jrange = np.arange(1, 2*int(jmax)+1, 1)
time_interval = tau * jrange

print(f"tau: {tau:.4}")
print(f"time int: {len(time_interval)}")
print(f"w_step: {w_step:.2} Ha")

# measurement circuit
dev_est = qml.device(device_type, wires=int(2*ncas) + 1, shots=shots)

@qml.qnode(dev_est)
def meas_circuit(state):
    qml.StatePrep(state, wires=dev_est.wires.tolist())
    # measure in PauliX or PauliY to get the real/imag parts
    return [qml.expval(op) for op in \
            [qml.PauliX(wires=0), qml.PauliY(wires=0)]]

# grab an initial state (including the auxiliary qubit)
state = initial_circuit(wf_dip)

results = np.zeros((2, len(time_interval)))  # results list initialization

# perform time steps
for ii in range(0, len(time_interval), 1):

    circuit = trotter_circuit(dev=dev_prop, state=state, step=tau)

    # update state and then measure
    state = circuit()
    measurement = meas_circuit(state=state)
    
    results[:, ii] += dipole_norm**2 * \
                    np.array(measurement).real

L_j = np.exp(-eta * time_interval)
fsignal_func = lambda w: (1./np.pi) *np.sum(L_j * (results[0,:] * np.cos(time_interval*w) -\
                results[1,:] * np.sin(time_interval*w))) 
fsignal = np.array([fsignal_func(w) for w in wgrid])

spectrum_func = lambda w: tau * ( (1/(2.*np.pi))*dipole_norm**2
                                + np.real(fsignal[int((w-w_min)//w_step)]) )
spectrum = np.array([spectrum_func(w) for w in wgrid]) 

# %%
import matplotlib.pyplot as plt
plt.style.use("pennylane.drawer.plot")

fig = plt.figure(figsize=(6.4, 2.4))
ax = fig.add_axes((0.15, 0.3, 0.8, 0.65))
ax.plot(range(len(results[0, :])), results[0, :], label="Real")
ax.plot(range(len(results[1, :])), results[1, :], label="Imaginary", linestyle="--")
ax.set(xlabel=r"$\mathrm{Time step}, j$")
fig.text(0.5, 0.05,
    "Figure 4. Time-domain output of algorithm.",
    horizontalalignment="center",
    size="small",
    weight="normal",
)
ax.legend()
plt.show()

from datetime import datetime

# Get the current datetime object
now = datetime.now()

# Format the datetime object as a string
datetime_string = now.strftime("%Y-%m-%d %H:%M:%S")

# %%
fig = plt.figure(figsize=(6.4, 4))
ax = fig.add_axes((0.15, 0.20, 0.80, 0.72))  # make room for caption

ax.plot(wgrid, spectrum.real)
ax.set_xlabel(r"$\mathrm{Energy}, \omega\ (\mathrm{Ha})$")
ax.set_ylabel(r"$\mathrm{Absorption\ (arb.)}$")

fig.text(0.5, 0.05,
    r"Figure 5. $H_2$ XAS spectrum calculation.",
    horizontalalignment="center",
    size="small",
    weight="normal",
)
plt.show()
# fig.savefig(datetime_string+"_spectrum.png")
