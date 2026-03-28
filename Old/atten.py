import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

file = 'attenuation_coeff.txt'
E, mu_rho = np.loadtxt(file, usecols=(0,1), unpack=True)

I_interp = scipy.interpolate.make_interp_spline(E, mu_rho, k=3)

print(I_interp(1))

plt.plot(np.log(E), np.log(mu_rho))


plt.show()