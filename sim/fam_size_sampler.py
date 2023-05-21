import numpy as np
import scipy.interpolate

sizes = list(range(2,21))
sizes_counts = np.array([0, 880372, 898973, 428540, 125740, 37006, 12459, 4530, 1891, 827, 395, 176, 117, 46, 32, 12, 5, 2, 2])

pdf = sizes_counts/sum(sizes_counts)  
cdf = np.cumsum(pdf)

# see https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
inv_cdf = scipy.interpolate.interp1d(cdf, sizes, kind='next')

def fam_size_sample(shape):
	fam_size = inv_cdf(np.random.rand(*shape)).astype(int)
	
	return fam_size