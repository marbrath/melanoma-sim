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

    print(nu_gi, eta, nu_e)
    u_g = np.random.gamma(nu_gi, 1/eta, num_genes)
    z_g = np.dot(P, u_g)
    z_e = np.random.gamma(nu_e, 1/eta, 1)*np.ones(family_size)
    z = z_g + z_e

    x_1 = np.append([0,1], np.random.binomial(1, 0.5, 8))
    min_elem = 1853
    
    birthyears = (birthyears - min_elem)/10.
    unif = np.random.uniform(0, 1, family_size)
    ts = (-np.log(unif)/(z*np.exp(beta_0 + beta_1*x_1 + beta_2*birthyears)))**(1/k)

    return z
    #return ts/12.

family_genders = []
family_birthyears = []
family_lifetimes = []
family_events = []
num_children = []
family_frailties = []

for year in range(1850, 2015 + 1 - 20):
    #print(year)
    max_children = 8
    num_families = 10000

    birthyears = np.repeat([[year]*2 + [year + 20]*max_children], num_families, axis=0)
    genders = np.hstack((np.repeat([[0, 1]], num_families, axis=0), np.random.binomial(1, 0.5, (num_families, max_children))))

    family_genders.append(genders)
    family_birthyears.append(birthyears)

    time_to_death = lifetime_sample(year, (num_families, 2 + max_children))
    #print(time_to_death)

    ##check mean of frailty variables
    #time_to_melanoma = corr_frailty(birthyears, genders)
    #print(time_to_melanoma)
    lifetimes = np.minimum(np.minimum(time_to_death, time_to_melanoma), (2016 - birthyears))
    #family_lifetimes.append(lifetimes)
    family_frailties.append(corr_frailty(birthyears, genders))
    #family_events.append(lifetimes == time_to_melanoma)
    #print(time_to_melanoma.min())

    num_children.append(np.random.randint(0, max_children + 1, num_families)) # todo: Use proper distribution

family_genders = np.vstack(family_genders)
family_birthyears = np.vstack(family_birthyears)
#family_lifetimes = np.vstack(family_lifetimes)
#family_events = np.vstack(family_events)
num_children = np.hstack(num_children)

#ts = family_lifetimes.ravel()
#events = family_events.ravel()

#kmf = KaplanMeierFitter()
#kmf.fit(ts, events)
#kmf.survival_function_.plot()
#plt.title('Survival function')
#plt.show()

## survival function
#x = np.sort(ts)
#y = np.arange(len(ts))/float(len(ts))
#plt.plot(x, 1 - y)
#plt.show()

#plt.hist(ts, bins=30, density=True)
#plt.plot(years[1:], cdf[1:] - cdf[:-1])
#plt.show()