import numpy as np
import matplotlib.pyplot as plt
import h5py
import tqdm

from sklearn import linear_model

# with h5py.File("../../data/LAP/test500.h5", "a") as h5f:
#     del h5f['mean']
#     del h5f['var']

# LAP
h5f = h5py.File("../../data/LAP/test500.h5", 'r')
# # dset = h5py.Dataset(h5f['data'].id)
# # dset._local.astype = np.float32
# # mean = []
# # var = []
# # for i in tqdm.tqdm(range(dset.shape[0] // 208)):
# #     data = dset[i * 208:(i + 1) * 208, :, :]
# #     mean.extend(np.mean(data, -1).ravel())
# #     var.extend(np.var(data, -1).ravel())
# # print(h5f['data'].shape)
# data = h5f['data'][600:1200, 550:1800, :]
# # mean = np.array(mean)
# # var = np.array(var)
# mean = np.mean(data, -1)
# var = np.var(data, -1)
# h5f['mean'] = mean.ravel()
# h5f['var'] = var.ravel()

mean = h5f['mean'][...]
var = h5f['var'][...]
# h5f.close()
mean = mean[var < 50000]
var = var[var < 50000]
var = var[mean > 400]
mean = mean[mean > 400]
'''
'''

# # # PM
# # h5f = h5py.File("../../data/PM/Test500.h5", 'a')
# # dset = h5py.Dataset(h5f['data'].id)
# # dset._local.astype = np.float32
# # mean = []
# # var = []
# # for i in tqdm.tqdm(range(dset.shape[0] // 128)):
# #     data = dset[i * 128:(i + 1) * 128, :, :]
# #     mean.extend(np.mean(data, -1).ravel())
# #     var.extend(np.var(data, -1).ravel())
# # mean = np.array(mean)
# # var = np.array(var)
# # h5f['mean'] = mean
# # h5f['var'] = var

# # mean = h5f['mean'][...]
# # var = h5f['var'][...]

# # mean = mean[var < 300]
# # var = var[var < 300]
# # var = var[mean > 25]
# # mean = mean[mean > 25]
# # '''
# # '''

reg = linear_model.LinearRegression()
reg.fit(mean.reshape((-1, 1)), var)

print(reg.coef_)
print(reg.intercept_)

# H, X, Y = np.histogram2d(mean, var, (100, 50))

# for x0, x1 in zip(X[:-1], X[1:]):
#     v = var[np.logical_and(mean >= x0, mean < x1)]

#     x = (x0 + x0) / 2
#     m = np.nanmean(v)
#     s = np.nanstd(v)
#     # print(x, m, s, m / x)

#     # plt.hist2d(v, 100)

# z = np.polyfit(mean, var, 1)
# print(z)
# p = np.poly1d(z)
x = np.linspace(np.min(mean), np.max(mean), 50)

# plt.scatter(x, m)
from matplotlib.colors import LogNorm
plt.hist2d(mean.ravel(), var.ravel(), (100, 50), norm=LogNorm())
plt.plot(x, x * reg.coef_[0] + reg.intercept_, '-')
plt.title(f"x * {reg.coef_[0]:.4f} + {reg.intercept_:.4f}")
plt.show()
