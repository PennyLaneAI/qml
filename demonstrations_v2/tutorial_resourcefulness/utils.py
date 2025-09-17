import numpy as np
from collections import OrderedDict

def group_rows_cols_by_sparsity(B, tol=0):
    """
    Given matrix B, this function groups identical rows and columns, orders these groups
    by sparsity (most zeros first) and returns the row & column permutation
    matrices P_row, P_col such that B2 = P_row @ B @ P_col is block-diagonal.
    """
    # compute boolean mask where |B| >= tol
    mask = np.abs(B) >= 1e-8
    # convert boolean mask to integer (False→0, True→1)
    C = mask.astype(int)

    # order by sparsity
    n, m = C.shape

    # helper to get a key tuple and zero count for a vector
    def key_and_zeros(vec):
        if tol > 0:
            bin_vec = (np.abs(vec) < tol).astype(int)
            key = tuple(bin_vec)
            zero_count = int(np.sum(bin_vec))
        else:
            key = tuple(vec.tolist())
            zero_count = int(np.sum(np.array(vec) == 0))
        return key, zero_count

    # group rows by key
    row_groups = OrderedDict()
    row_zero_counts = {}
    for i in range(n):
        key, zc = key_and_zeros(C[i, :])
        row_groups.setdefault(key, []).append(i)
        row_zero_counts[key] = zc

    # sort row groups by zero_count descending
    sorted_row_keys = sorted(row_groups.keys(),
                             key=lambda k: row_zero_counts[k],
                             reverse=True)
    # flatten row permutation
    row_perm = [i for key in sorted_row_keys for i in row_groups[key]]

    # group columns by key
    col_groups = OrderedDict()
    col_zero_counts = {}
    for j in range(m):
        key, zc = key_and_zeros(C[:, j])
        col_groups.setdefault(key, []).append(j)
        col_zero_counts[key] = zc

    # sort column groups by zero_count descending
    sorted_col_keys = sorted(col_groups.keys(),
                             key=lambda k: col_zero_counts[k],
                             reverse=True)
    col_perm = [j for key in sorted_col_keys for j in col_groups[key]]

    # build permutation matrices
    P_row = np.eye(n)[row_perm, :]
    P_col = np.eye(m)[:, col_perm]

    return P_row, P_col