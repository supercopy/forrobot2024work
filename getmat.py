import scipy.io
import numpy as np
mat = scipy.io.loadmat('pic.mat')
y = np.array([mat])
print(y)