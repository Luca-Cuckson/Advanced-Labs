import numpy as np

energies = np.array([924, 1146, 1079, 957, 1018, 1079, 519, 985, 1123])

resolutions = energies * -3.38578187e-03 + 1.37315550e+01

errors = resolutions * energies / (100 * 2.355)

print(errors)