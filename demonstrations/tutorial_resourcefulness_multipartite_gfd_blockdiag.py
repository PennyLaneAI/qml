## Shows how the adjoint superoperator of an element of SU(2)xSU(2)x...xSU(2)

import numpy as np
import itertools
import functools

# single‑qubit Paulis
_pauli_map = {
    'I': np.array([[1,0],[0,1]],   dtype=complex),
    'X': np.array([[0,1],[1,0]],   dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]],dtype=complex),
    'Z': np.array([[1,0],[0,-1]],  dtype=complex),
}

def adjoint_superoperator(U):
    """
    Adjoint superop representing (U X U†)
    """
    return np.kron(U.conj(), U)

def pauli_basis(n):
    """
    generates the basis of Pauli operators, and orders it by appearence in the isotypical decomp of Times_i SU(2)
    """
    all_strs    = [''.join(s) for s in itertools.product('IXYZ', repeat=n)]
    sorted_strs = sorted(all_strs, key=lambda s: (n-s.count('I'), s))
    norm        = np.sqrt(2**n)
    mats        = []
    for s in sorted_strs:
        factors = [_pauli_map[ch] for ch in s]
        M       = functools.reduce(lambda A,B: np.kron(A,B), factors)
        mats.append(M.reshape(-1)/norm)
    B = np.column_stack(mats)
    return sorted_strs, B

def rotate_superoperator(U):
    """
    Rotates the superop of the adj of a unitary U to the irrep-sorted Pauli basis
    """
    S        = adjoint_superoperator(U)
    n        = int(np.log2(U.shape[0]))
    basis,B  = pauli_basis(n)
    S_rot    = B.conj().T @ S @ B
    return basis, S_rot


# Haar‐random unitary helper
def haar_unitary(N):
    """
    Generates a Haar random NxN unitary matrix
    """
    X = (np.random.randn(N, N) + 1j*np.random.randn(N, N)) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    phases = np.exp(-1j * np.angle(np.diag(R)))
    return Q @ np.diag(phases)


# we showcase the block-diagonalization in the case of two qubits
n = 2
Us = [haar_unitary(2) for _ in range(n)]
# U is a tensor product of single-qubit unitaries
U = functools.reduce(lambda A, B: np.kron(A, B), Us)
basis, S_rot = rotate_superoperator(U)
np.set_printoptions(
    formatter={'float': lambda x: f"{x:5.2g}"},
    linewidth=200,    # default is 75;
    threshold=10000   # so it doesn’t summarize large arrays
)
print("Adjoint Superoperator of U in the computational basis")
Uadj = adjoint_superoperator(U)
Uadj_real = Uadj.real
Ur_round = np.round(Uadj_real, 2)
print("Rounded real part:")
print(Ur_round)
Uadj_imag = Uadj.imag
Ur_round = np.round(Uadj_imag, 2)
print("Rounded imag part:")
print(Ur_round)


# now round and print
## NOTICE THAT IN PAULI BASIS THE UNITARY ADJOINT ACTION IS ORTHOGONAL
print("Adjoint Superoperator of U in the Irrep basis")
S_real = S_rot.real
Sr_round = np.round(S_real, 2)
print("Rounded real part (the operator is real):")
print(Sr_round)
