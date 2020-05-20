import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import random

font = {"family": "KaiTi"}
matplotlib.rc("font", **font)

data = np.loadtxt("./USvideos.csv", skiprows=1, delimiter=",", dtype=float)
print(data)

arr = data[:, 4].astype(int)
print(arr, max(arr), min(arr))

plt.hist(arr,bins=100)
plt.show()
