import numpy as np
import matplotlib.pyplot as plt


def rect(x, d):
    r = np.zeros_like(x)
    r[np.bitwise_and(np.abs(x) > 0.5 * d, np.abs(x) < 1.5 * d)] = 1
    return r


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


''' resolution ~ distance between two ringing function, when the sum is 1/2.
    ~ 1/2 * line width of USAF when min(signal) ~ 1/2 max(signal)
'''

d = 2.19
x = np.linspace(-4 * d, 4 * d, 1000)
X = rect(x, d)
Y = gaussian(x, 0, 1.05)
Z = np.convolve(X, Y, 'same')

X /= np.sum(X)
Y /= np.sum(Y)
Z /= np.sum(Z)

plt.plot(x, X)
plt.plot(x, Y)
plt.plot(x, np.convolve(X, Y, 'same'))
plt.show()
