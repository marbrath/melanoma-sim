import numpy as np
from math import log2

def double(perm):
    return sum(([p, p] for p in perm), [])

def double_perms(perms):
    return list(map(double, perms))

def permute_middle(perm):
    new_perm = perm[:]

    for i in range(len(perm)//4):
        new_perm[4*i + 1] = perm[4*i + 2]
        new_perm[4*i + 2] = perm[4*i + 1]

    return new_perm

def permute_right(perm):
    new_perm = perm[:]

    for i in range(len(perm)//4):
        new_perm[4*i + 2] = perm[4*i + 3]
        new_perm[4*i + 3] = perm[4*i + 2]

    return new_perm

def permute(perms):
    m = len(perms[0])
    new_perms = double_perms(perms)
    middle_perms = [permute_middle(p) for p in new_perms[(m//2 - 1):]]
    right_perms = [permute_right(p) for p in middle_perms]

    return new_perms + middle_perms + right_perms

def valid_permutation(m, lhs, rhs):
    return (lhs.dot(rhs) == m//4).all()

def check_valid(perms):
    m = len(perms[0])
    valid_perms = np.array([perms[0]])

    for perm in perms[1:]:
        if not valid_permutation(m, valid_perms, perm):
            return False

        valid_perms = np.vstack((valid_perms, perm))

    return True


def get_permutations(m):
    '''
        returns n \times 2*2**np.ceil(np.log2(m)) with the maximal number of rows n which satisfies the covariance condition   
    '''
   
    perms = [[1, 0]]

    while len(perms) < m - 1:
        perms = permute(perms)

    return perms

def get_parent_matrices(family_size):
    perms = np.array(get_permutations(family_size))

    one_row = np.array([[1]*perms.shape[1]])
    zero_row = np.array([[0]*perms.shape[1]])

    P_f = np.concatenate((one_row, zero_row, perms), axis=0)
    P_m = np.concatenate((zero_row, one_row, perms), axis=0)

    return P_f, P_m


def print_mat(M):
    print('[')
    print(',\n'.join('  [%s]' % ', '.join(map(str, M[i, :])) for i in range(M.shape[0])))
    print(']')

if __name__ == '__main__':
    P_f, P_m = get_parent_matrices(9)

    print(P_m.shape)

    print_mat(P_f)
    print_mat(P_m)
