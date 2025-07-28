## compute SU(2)xSU(2)x...xSU(2) Adjoint purities 

import numpy as np
import math
import itertools
import functools
import matplotlib.pyplot as plt

# Single‑qubit Pauli matrices
_pauli_map = {
    'I': np.array([[1, 0],
                   [0, 1]], dtype=complex),
    'X': np.array([[0, 1],
                   [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j],
                   [1j,  0]], dtype=complex),
    'Z': np.array([[1,  0],
                   [0, -1]], dtype=complex),
}

# utils to compute projections

def generate_pauli_strings(n: int):
    """
    Generate all length‑n strings over the Pauli alphabet ['I','X','Y','Z'].
    Returns a list of 4**n strings, e.g. ['IIX', 'IXZ', …].
    """
    return [''.join(p) for p in itertools.product('IXYZ', repeat=n)]

def pauli_string_to_matrix(pauli_str: str):
    """
    Convert a Pauli string (e.g. 'XIY') to its full 2^n × 2^n matrix.
    """
    mats = [_pauli_map[s] for s in pauli_str]
    return functools.reduce(lambda A, B: np.kron(A, B), mats)


# function to project into the modules
def compute_me_purities(op):
    """
    Computes GFD purities of op (assumed to be np.matrix)
    by explicitly computing overlaps with Paulis
    """

    if op.ndim == 1:
        # state vector
        is_vector = True
    elif op.ndim == 2 and op.shape[0] == op.shape[1]:
        # density/operator
        is_vector = False
    else:
        raise ValueError("`op` must be either a 1D state vector or a square matrix")

    d   = op.shape[0]
    n   = int(np.log2(d))

    basis = generate_pauli_strings(n)
    purities = np.zeros(n+1)
    for belem in basis:
        k = n - belem.count('I')
        P = pauli_string_to_matrix(belem)
        
        if is_vector:
            ovp = op.conj() @ (P @ op)
        else:  
            ovp = np.trace(op @ P)
            
        assert ovp.imag < 1e-14
        purities[k] += (ovp.real) ** 2

    return purities / (2**n)



#####################################################################
# functions to generate relevant quantum states

def ghz_state(n: int):
    """Return the |GHZ_n⟩ state vector for *n* qubits."""
    psi = np.zeros(2**n)
    psi[0] = 1/math.sqrt(2)
    psi[-1] = 1/math.sqrt(2)
    return psi

def w_state(n: int):
    """Return the |W_n⟩ state vector for *n* qubits."""
    psi = np.zeros(2**n)
    for q in range(n):
        psi[2**q] = 1/math.sqrt(n)
    return psi

def haar_state(n: int):
    """Return a Haar random state vector for *n* qubits."""
    N = 2**n
    # i.i.d. complex Gaussians
    X = (np.random.randn(N, 1) + 1j*np.random.randn(N, 1)) / np.sqrt(2)
    # QR on the N×1 “matrix” is just Gram–Schmidt → returns Q (N×1) and R (1×1)
    Q, R = np.linalg.qr(X, mode='reduced')
    # fix the overall phase so it’s uniformly distributed
    phase = np.exp(-1j * np.angle(R[0,0]))
    return (Q[:,0] * phase)

def haar_product_state(n: int):
    """Return a Haar random tensor product state of *n* qubits"""
    states = [haar_state(1) for _ in range(n)]
    return functools.reduce(lambda A, B: np.kron(A, B), states)


#######################################################################
def plot_purities(states, labels):
    # generate plotting data
    ns = [int(np.log2(op.shape[0])) for op in states]
    assert np.all([n == ns[0]]), "Provided States must all have the same dimension!"
    
    data_series = [compute_me_purities(op) for op in states]
    
    # Grab default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create two vertically aligned subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    for i, data in enumerate(data_series):
        color = colors[i % len(colors)]
        ax1.plot(data, label=f'{labels[i]}', color=color)
        ax2.plot(np.cumsum(data), label=f'{labels[i]}', color=color)
    
    ax1.set_ylabel('Purity')
    ax2.set_ylabel('Cumulative Purity')
    ax2.set_xlabel('Module weight')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()




n = 2
states = [
    haar_product_state(n),
    ghz_state(n),
    w_state(n),
    haar_state(n)
]

labels = [
    "Product",
    "GHZ",
    "W",
    "Haar"
]

plot_purities(states, labels)
    










    