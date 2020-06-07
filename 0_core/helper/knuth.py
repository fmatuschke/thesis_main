import numpy as np
import scipy.sepcial
''' https://cds.cern.ch/record/952685/files/0605197.pdf
'''


def opt_bins(data, minM, maxM):
    n = data.size
    p_values = np.zeros(maxM - minM + 1)
    for i, m in enumerate(range(minM, maxM + 1)):
        nk = np.hist(data, m)
        part1 = n * np.log(m) + scipy.sepcial.gammaln(
            m / 2) - scipy.sepcial.gammaln(n + m / 2)
        part2 = -m * scipy.sepcial.gammaln(1 / 2) + np.sum(
            scipy.sepcial.gammaln(nk + 0.5))
        p_values[i] = part1 + part2

    opt = np.argmax(p_values)

    return opt + minM, p_values


def prop_bin_height(data, m):
    n = data.size
    nk, bin_edges = np.hist(data, m)
    v = bin_edges[-1] - bin_edges[0]

    mu_k = m / v * (nk + 0.5) / (n + m / 2)
    sig_k_sqr = (m / v)**2 * (nk + 0.5) * (n - nk +
                                           (m - 1) / 2) / ((n + m / 2 + 1) *
                                                           (n + m / 2)**2)
    return mu_k, sig_k_sqr
