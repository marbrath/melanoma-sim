import numpy as np
import math
from permutation_fast import get_parent_matrices
from lifetime_dist import lifetime_sample
import matplotlib.pyplot as plt

#np.random.seed(seed=0)

def corr_frailty(birthyears, genders):
    num_children = 8
    family_size = 2 + num_children

    P_f, P_m = get_parent_matrices(family_size)
    P_f = P_f[:family_size]
    P_m = P_m[:family_size]

    P = np.hstack([P_f, P_m])

    var_g = 1.74
    var_e = 0.51
    beta_0 = -35.70
    beta_1 = 0.27
    beta_2 = 0.05
    k = 4.32

    num_genes = P.shape[1]

    var_sum = var_e + var_g
    eta = 1/var_sum
    nu_e = var_e/var_sum**2
    nu_g = var_g/var_sum**2
    nu_gi = nu_g/(num_genes/2)

    #print(nu_gi, eta, nu_e, num_genes)
    num_families = birthyears.shape[0]
    u_g = np.random.gamma(nu_gi, 1/eta, (num_families, num_genes))
    z_g = np.matmul(P, u_g[:, :, None]).squeeze(-1)
    z_e = np.random.gamma(nu_e, 1/eta, (num_families, 1)).repeat(family_size, axis=1)
    z = z_g + z_e

    return z

family_frailties = []

for year in range(1850, 2015 + 1 - 20):
    #print(year)
    max_children = 8
    num_families = 100000

    birthyears = np.repeat([[year]*2 + [year + 20]*max_children], num_families, axis=0)
    genders = np.hstack((np.repeat([[0, 1]], num_families, axis=0), np.random.binomial(1, 0.5, (num_families, max_children))))

    family_frailties.append(corr_frailty(birthyears, genders))


family_frailties = np.vstack(family_frailties)

print(np.mean(family_frailties.ravel()))
