import numpy as np
import matplotlib.pyplot as plt
import h5py

h5f = h5py.File("../../data/Testreihe_LAP/t000stack.h5", 'r')

data = h5f['data'][:, 600:1500, 900:1700]
print(data.shape)
data = np.swapaxes(data, 0, -1).astype(np.float64)
print(data.shape)

mean = np.mean(data, -1)
var = np.var(data, -1)

mean = mean[var < 35000]
var = var[var < 35000]
var = var[mean < 8000]
mean = mean[mean < 8000]

H, X, Y = np.histogram2d(mean.ravel(), var.ravel(), (100, 50))

for x0, x1 in zip(X[:-1], X[1:]):
    v = var[np.logical_and(mean >= x0, mean < x1)]

    x = (x0 + x0) / 2
    m = np.mean(v)
    s = np.std(v)
    print(x, m, s, m / x)

# plt.scatter(x, m)
plt.hist2d(mean.ravel(), var.ravel(), (100, 50))
plt.show()
