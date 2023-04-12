import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_optim = np.load('all_optim_old.npy')

var_E = np.exp(all_optim[:,0])
var_G = np.exp(all_optim[:,1])
sum_ = var_E + var_G

k = all_optim[:,2]
beta_0 = all_optim[:,3]
beta_1 = all_optim[:,4]
beta_2 = all_optim[:,5]


var_e_ = 0.51
var_g_ = 1.74
#var_e_ = 1.74
#var_g_ = 0.51

sum__ = var_g_ + var_e_
beta_0_ = -35.70 #-25
beta_1_ = 0.27
beta_2_ = 0.05
k_ = 4.32

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

ax1.hist(sum_, bins=10, density=True)
ax1.axvline(x=sum__, color="C1", linewidth=2)
ax1.set_title('sum')

ax2.hist(var_G, bins=10, density=True)
ax2.axvline(x=var_g_, color="C1", linewidth=2)
ax2.set_title('var_g')

ax3.hist(k, bins=10, density=True)
ax3.axvline(x=k_, color="C1", linewidth=2)
ax3.set_title('k')

ax4.hist(beta_0, bins=10, density=True)
ax4.axvline(x=beta_0_, color="C1", linewidth=2)
ax4.set_title('beta 0')

ax5.hist(beta_1, bins=10, density=True)
ax5.axvline(x=beta_1_, color="C1", linewidth=2)
ax5.set_title('beta 1')

ax6.hist(beta_2, bins=10, density=True)
ax6.axvline(x=beta_2_, color="C1", linewidth=2)
ax6.set_title('beta 2')


plt.show()

