import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats


def load_maestro_spe(path):
    with open(path, "r") as f:
        lines = f.readlines()

    # Find the $DATA: section
    for i, line in enumerate(lines):
        if line.strip().startswith("$DATA"):
            data_start = i
            break

    # Next line: "first_channel last_channel"
    first, last = map(int, lines[data_start + 1].split())
    n = last - first + 1

    # Next n lines: counts
    data = np.array([int(x) for x in lines[data_start + 2 : data_start + 2 + n]])
    return data


def atten(counts, mu, x): # thickness x in cm
    return counts * np.exp(-mu * x)


def logging(counts):  # VERY IMPORTANT: FOR VISUALISING PLOTS ONLY, NOT SOUNDS FOR DATA ANALYSIS
    logged = np.empty(len(counts))
    for i in range(len(counts)):
        if -1 <= counts[i] <= 1:
            logged[i] = 0.0
        if 1 < counts[i]:
            logged[i] = np.log10(counts[i])
        if counts[i] < -1:
            logged[i] = -np.log10(-counts[i])
    return logged


def binsum(array, width):
    floor = len(array) // width
    multiple = floor * width
    binned = array[:multiple].reshape(-1, width).sum(axis=1)
    if multiple < len(array):
        binned = np.append(binned, np.sum(array[multiple:]))
    return binned


def binmean(array, width):
    floor = len(array) // width
    multiple = floor * width
    binned = array[:multiple].reshape(-1, width).mean(axis=1)
    if multiple < len(array):
        binned = np.append(binned, np.mean(array[multiple:]))
    return binned


# The interpolation function that finds the continued curve under the peak (assuming a straight line in log-scale)
def no_peaking(x, y, minE, low, high, maxE):  
    shwoopx = np.append(x[minE:low], x[high:maxE])
    shwoopy = np.append(y[minE:low], y[high:maxE])

    coeffs = np.polyfit(shwoopx, np.log10(shwoopy), deg=1)
    no_peak = 10 ** np.polyval(coeffs, x[low:high])
    return no_peak