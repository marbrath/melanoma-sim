import pandas as pd
import numpy as np
import scipy.interpolate    

np.random.seed(seed=0)

mean_age = pd.read_csv('data/mean_lifetimes.csv', sep=';', decimal=',')
mortality_by_year = pd.read_csv('data/distribution_both_1966.csv', sep=';', decimal=',')
years = np.zeros(len(mortality_by_year) + 1)
qx = np.zeros(len(mortality_by_year) + 1)


years[1:] = mortality_by_year['Year'].values
qx[1:] = mortality_by_year['Both'].values/1e3

#plt.plot(mortality_by_year['Year'], qx)
#plt.show()

Sx = np.cumprod(1-qx)
cdf = 1-Sx

#plt.plot(years, cdf)

# see https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
inv_cdf = scipy.interpolate.interp1d(cdf, years)

#t = np.linspace(0, 1)
#plt.plot(inv_cdf(t), t)
#plt.show()

def lifetime_sample(year, shape):
	ts = inv_cdf(np.random.rand(*shape))
	idx = mean_age.index[mean_age['Year'] == year].tolist()[0]
	cohort_mean = float(mean_age.loc[idx].at["Total"])

	return (cohort_mean/76.)*ts
	return ts

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	ts = lifetime_sample(1846, (10000,5)).ravel()
	#print(ts)
	plt.hist(ts, bins=30, density=True)
	plt.plot(years[1:], cdf[1:] - cdf[:-1])
	plt.show()
	#print(lifetime_sample(1846, (5, 5)))

#print(np.mean(ts.ravel()))
#plt.hist(ts, bins=30, density=True)
#plt.plot(years[1:], cdf[1:] - cdf[:-1])
#plt.show()

