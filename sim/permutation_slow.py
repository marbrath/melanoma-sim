import numpy as np


def gen_mat(m):
    if m == 1:
        return np.array([[1, 0]], dtype=np.uint8)

    Ap = gen_mat(m - 1)

    A = np.empty((m, 2**m), dtype=np.uint8)

    for i in range(m - 1):
        A[i, ::2] = Ap[i, :]
        A[i, 1::2] = Ap[i, :]

    A[-1, :2**(m - 1)] = Ap[-1, :]
    A[-1, 2**(m - 1):] = Ap[-1, :]

    return A


def get_symmetric_parent_matrices(fam_size):
    num_children = fam_size - 2
    n = 2**num_children

    P_f = np.concatenate((
        np.ones((1, n), dtype=np.uint8),
        np.zeros((1, n), dtype=np.uint8),
        gen_mat(num_children)
    ))
    P_m = np.concatenate((
        np.zeros((1, n), dtype=np.uint8),
        np.ones((1, n), dtype=np.uint8),
        gen_mat(num_children)
    ))

    return P_f, P_m
