import numpy as np
import matplotlib.pyplot as plt
arr = np.loadtxt('loceta.txt')
plt.plot(arr)
plt.savefig('gaussian_exp.jpg')