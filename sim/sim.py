import numpy as np
import math
from permutation_fast import get_parent_matrices
from lifetime_dist import lifetime_sample
from fam_size_sampler import fam_size_sample
import matplotlib.pyplot as plt
#from lifelines import KaplanMeierFitter
import sys
import os

def corr_frailty(birthyears, genders, num_children):
    family_size = 2 + num_children

    P_f, P_m = get_parent_matrices(family_size)
    P_f = P_f[:family_size]
    P_m = P_m[:family_size]

    P = np.hstack([P_f, P_m])

    var_g =  1.74
    var_e = 0.51
    beta_0 = -35.70 #-25
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

    num_fam_per_year = birthyears.shape[0]
    u_g = np.random.gamma(nu_gi, 1/eta, (num_fam_per_year, num_genes))

    z_g = np.matmul(P, u_g[:, :, None]).squeeze(-1)
    z_e = np.random.gamma(nu_e, 1/eta, (num_fam_per_year, 1)).repeat(family_size, axis=1)
    z = z_g + z_e

    #z = np.random.gamma(1/2, 2, (num_fam_per_year, 1)).repeat(family_size, axis=1)

    min_elem = 1850

    birthyears = (birthyears - min_elem)/10.
    unif = np.random.uniform(0, 1, (num_fam_per_year, family_size))
    ts = (-np.log(unif)/(z*np.exp(beta_0 + beta_1*birthyears + beta_2*genders)))**(1/k)

    return ts
  
def sim(seed, num_fam_per_year, max_children):
    fam_genders = []
    fam_birthyears = []
    fam_lifetimes = []
    fam_events = []
    fam_num_events = []
    fam_num_children = []
    fam_frailties = []

    for year in range(1850, 2015 + 1 - 20):
        #print('YEAR')
        #print(year)

        birthyears = np.random.uniform(year + 20, year + 50, (num_fam_per_year, 2 + max_children))
        birthyears[:, :2] = year
        birthyears[:, 2:].sort(axis=1)

        genders = np.hstack((np.repeat([[0, 1]], num_fam_per_year, axis=0), np.random.binomial(1, 0.5, (num_fam_per_year, max_children))))

        time_to_death = lifetime_sample(year, (num_fam_per_year, 2 + max_children))
        time_to_melanoma = corr_frailty(birthyears, genders, max_children)

        lifetimes = np.minimum(np.minimum(time_to_death, time_to_melanoma), (2016 - birthyears)*12)

        events = (lifetimes == time_to_melanoma)*1

        num_children = fam_size_sample([num_fam_per_year]) - 2

        children_to_remove = (np.arange(time_to_melanoma.shape[1])[None] > (num_children[:, None] + 1)) | (birthyears > 2016)
        num_children = max_children - children_to_remove.sum(axis=1)

        birthyears[children_to_remove] = 0
        lifetimes[children_to_remove] = 0
        genders[children_to_remove] = 0
        events[children_to_remove] = 0

        fam_num_children.append(num_children)
        fam_genders.append(genders)
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

    num_families, family_size = fam_lifetimes.shape

    fam_genders = fam_genders.ravel().astype('int64')
    fam_birthyears = fam_birthyears.ravel().astype('int64')
    fam_lifetimes = fam_lifetimes.ravel().astype('int64')
    fam_ids = (np.ones((family_size, num_families))*np.arange(num_families)[None]).T.ravel().astype('int64')

    event_bits = np.packbits(fam_events, bitorder='little', axis=1).astype('int64')
    assert(event_bits.shape[1] <= 3) # we here assume that family_size <= 24, which it is not

    fam_sick_ids = event_bits[:, 0]

    if event_bits.shape[1] >= 2:
        fam_sick_ids += (event_bits[:, 1] << 8)

    if event_bits.shape[1] >= 3:
        fam_sick_ids += (event_bits[:, 2] << 16)

    fam_num_events = fam_num_events.ravel().astype('int64')
    fam_num_children = fam_num_children.ravel().astype('int64')
    fam_truncation_times = (fam_lifetimes.ravel()*0).astype('int64')

    all_fam_events = np.vstack(fam_events).ravel().astype('int64')


    root_path = 'sim-output/npy_files_%04d_%02d' % (seed, max_children)

    if not os.path.exists(root_path):
        os.mkdir(root_path)


    np.save(root_path + '/fam_num_children', fam_num_children)
    np.save(root_path + '/fam_events', fam_events)
    np.save(root_path + '/all_fam_events', all_fam_events)
    np.save(root_path + '/genders', fam_genders)
    np.save(root_path + '/birthyears', fam_birthyears)
    np.save(root_path + '/lifetimes', fam_lifetimes)
    np.save(root_path + '/sick_ids', fam_sick_ids)
    np.save(root_path + '/all_num_events', fam_num_events)
    np.save(root_path + '/truncations', fam_truncation_times) 
    np.save(root_path + '/fam_ids', fam_ids) 


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: sim.py <seed> <num_fam_per_year> <max_children>')
        sys.exit(1)

    seed = int(sys.argv[1])
    num_fam_per_year = int(sys.argv[2])
    max_children = int(sys.argv[3])
    sim(seed, num_fam_per_year, max_children)


    '''
    fam_events = fam_events.ravel().astype('int64')

    kmf = KaplanMeierFitter()
    kmf.fit(fam_lifetimes, fam_events)
    kmf.survival_function_.plot()
    plt.title('Survival function')
    plt.show()
    '''

