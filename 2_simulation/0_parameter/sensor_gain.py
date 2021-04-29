#!/usr/bin/env python3

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn import linear_model

PATH = os.path.dirname(os.path.realpath(__file__))
THESIS = os.path.join(os.path.realpath(__file__).split('/thesis/')[0], 'thesis')

# with h5py.File("../../data/LAP/test500.h5", "a") as h5f:
#     del h5f['mean']
#     del h5f['var']

# # LAP
# h5f = h5py.File("../../data/LAP/repo/test500.h5", 'r')
# mean = h5f['mean'][...]
# var = h5f['var'][...]
# mean = mean[var < 50000]
# var = var[var < 50000]
# var = var[mean > 400]
# mean = mean[mean > 400]
'''
'''

# PM
h5f = h5py.File(os.path.join(THESIS, 'data/PM/repo/Test500.h5'), 'r')

if 'mean' not in h5f:
    dset = h5py.Dataset(h5f['data'].id)
    dset._local.astype = np.float32
    mean = []
    var = []
    for i in tqdm.tqdm(range(dset.shape[0] // 128)):
        data = dset[i * 128:(i + 1) * 128, :, :]
        mean.extend(np.mean(data, -1).ravel())
        var.extend(np.var(data, -1).ravel())
    mean = np.array(mean)
    var = np.array(var)
    h5f['mean'] = mean
    h5f['var'] = var

mean = h5f['mean'][:]
var = h5f['var'][:]

mean = mean[var < 300]
var = var[var < 300]
var = var[mean > 25]
mean = mean[mean > 25]
'''
'''

from scipy import stats

res = stats.linregress(mean, var)
print(f'slope: {res.slope}')
print(f'intercept: {res.intercept}')
print(f'r_value: {res.rvalue}')
print(f'p_value: {res.pvalue}')
print(f'std_err: {res.stderr}')
print(f'intercept_stderr: {res.intercept_stderr}')

# reg = linear_model.LinearRegression()
# reg.fit(mean.reshape((-1, 1)), var)

# print(reg.coef_)
# print(reg.intercept_)
"""  """

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
from matplotlib import cm

plt.hist2d(mean.ravel(),
           var.ravel(), (100, 50),
           norm=LogNorm(),
           cmap=plt.get_cmap('cividis'))
plt.plot(x, x * res.slope + res.intercept, '-')
plt.title(
    f"x * ({res.slope:.6f} +- {res.stderr:.6f}) + ({res.intercept:.3f} +- {res.intercept_stderr:.3f}), R:{res.rvalue:.4f}"
)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(PATH, 'output', 'PM_noise.png'))
# plt.show()

###########################
""" save single image """
###########################
fig = plt.figure()
data = h5py.File(os.path.join(THESIS, 'data/PM/repo/Test500.h5'),
                 'r')['data'][:, :, 0].astype(np.float32).T

plt.imshow(data, cmap=plt.get_cmap('gray'))
# plt.imshow(data, cmap=plt.get_cmap('cividis'))
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(PATH, 'output', 'PM_000.png'))

import skimage.transform
import matplotlib

data_ = skimage.transform.resize(data, (512, 512), anti_aliasing=False)
print(f'vmin: 0, vmax: {data_.max()}')
matplotlib.image.imsave(os.path.join(
    PATH, 'output', f'PM_000_vmin_0_vmax_{int(data_.max())}.png'),
                        data_,
                        cmap=plt.get_cmap('gray'),
                        vmin=0,
                        vmax=int(data_.max()))
# data_ = skimage.transform.resize(data, (256, 256), anti_aliasing=False)
# np.savetxt(os.path.join(PATH, 'output', 'PM_000.img'), data_, fmt='%.4e')

data = np.mean(data, 0)
np.savetxt(os.path.join(PATH, 'output', 'PM_000.dat'),
           np.vstack((np.array(range(data.size)), data))[:, ::5].T)
