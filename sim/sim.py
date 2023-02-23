import numpy as np
import math
from permutation_fast import get_parent_matrices
from lifetime_dist import lifetime_sample
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

np.random.seed(seed=0)

def corr_frailty(birthyears, genders):
    num_children = 8
    family_size = 2 + num_children
    max_num_sick = 5

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

    family_size = P.shape[0]
    num_genes = P.shape[1]

    var_sum = var_e + var_g
    eta = 1/var_sum
    nu_e = var_e/var_sum**2
    nu_g = var_g/var_sum**2
    nu_gi = nu_g/(num_genes/2)

    num_families = birthyears.shape[0]
    u_g = np.random.gamma(nu_gi, 1/eta, (num_families, num_genes))
    z_g = np.matmul(P, u_g[:, :, None]).squeeze(-1)
    z_e = np.random.gamma(nu_e, 1/eta, (num_families, 1)).repeat(family_size, axis=1)
    z = z_g + z_e

    min_elem = 1853
    
    birthyears = (birthyears - min_elem)/10.
    unif = np.random.uniform(0, 1, (num_families, family_size))
    ts = (-np.log(unif)/(z*np.exp(beta_0 + beta_1*birthyears + beta_2*genders)))**(1/k)

    return ts/12.

fam_genders = []
fam_birthyears = []
fam_lifetimes = []
fam_events = []
fam_num_events = []
fam_num_children = []

for year in range(1850, 2015 + 1 - 20):
    max_children = 8
    num_families = 10000

    birthyears = np.repeat([[year]*2 + [year + 20]*max_children], num_families, axis=0)
    genders = np.hstack((np.repeat([[0, 1]], num_families, axis=0), np.random.binomial(1, 0.5, (num_families, max_children))))

    fam_genders.append(genders)
    fam_birthyears.append(birthyears)

    time_to_death = lifetime_sample(year, (num_families, 2 + max_children))
    time_to_melanoma = corr_frailty(birthyears, genders)
    lifetimes = np.minimum(np.minimum(time_to_death, time_to_melanoma), (2016 - birthyears))
    events = (lifetimes == time_to_melanoma)


    num_children = np.random.randint(0, max_children + 1, num_families) # todo: Use proper distribution

    children_to_remove = np.arange(lifetimes.shape[1])[None] > (num_children[:, None] + 1)
    birthyears[children_to_remove] = 0
    lifetimes[children_to_remove] = 0
    genders[children_to_remove] = 0
    events[children_to_remove] = 0

    fam_num_children.append(num_children)
    fam_birthyears.append(birthyears)
    fam_lifetimes.append(lifetimes)
    fam_events.append(events)
    fam_num_events.append(events.sum(axis=1))


fam_genders = np.vstack(fam_genders)
fam_birthyears = np.vstack(fam_birthyears)
fam_lifetimes = np.vstack(fam_lifetimes)
fam_events = np.vstack(fam_events)
fam_num_events = np.vstack(fam_num_events)
fam_num_children = np.vstack(fam_num_children)

fam_genders = fam_genders.ravel().astype('int64')
fam_birthyears = fam_birthyears.ravel().astype('int64')
fam_lifetimes = fam_lifetimes.ravel().astype('int64')

event_bits = np.packbits(fam_events, bitorder='little', axis=1).astype('int64')
assert(event_bits.shape[1] == 2) # we here assume that 9 <= family_size <= 16, which it is not
fam_sick_ids = (event_bits[:, 0] << 8) + event_bits[:, 1]

fam_num_events = fam_num_events.astype('int64')
fam_num_children = fam_num_children.astype('int64')

fam_truncation_times = (fam_lifetimes.ravel()*0).astype('int64')

#fam_lifetimes = np.empty(fam_size*num_families, np.dtype('int64'))
#fam_truncation_times = np.empty(fam_size*num_families, np.dtype('int64'))
#fam_birthyears = np.empty(fam_size*num_families, np.dtype('int64'))
#fam_genders = np.empty(fam_size*num_families, np.dtype('int64'))

#np.save('npy_files/lifetimes', fam_lifetimes)
#np.save('npy_files/truncations', fam_truncation_times)
#np.save('npy_files/birthyears', fam_birthyears)
#np.save('npy_files/genders', fam_genders)

