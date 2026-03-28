import scipy.stats
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = x ** 2

binwidth = 7

#for i in range(len(x)):
#    ...

#result = x.reshape(-1, binwidth).sum(axis=1)
#result2 = x.reshape(-1, binwidth).mean(axis=1)

#print(result)
#print(result2)

# trying to create a lovely binning function where I cann just change the bin-width variable input to get an automatic thingy out

print(15//4)

def binsum(array, width):
    floor = len(array) // width
    multiple = floor * width
    binned = array[:multiple].reshape(-1, width).sum(axis=1)
    if multiple < len(array):
        binned = np.append(binned, np.sum(array[multiple:]))
    return binned

print(binsum(x, binwidth))

def binmean(array, width):
    floor = len(array) // width
    multiple = floor * width
    binned = array[:multiple].reshape(-1, width).mean(axis=1)
    if multiple < len(array):
        binned = np.append(binned, np.mean(array[multiple:]))
    return binned

print(binmean(x, binwidth))