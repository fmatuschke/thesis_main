import numpy as np
from . import sknni

from fastpli.analysis.orientation import remap_spherical

# def remap_sph_angles(phi, theta):
#     phi = np.array(phi, copy=False)
#     theta = np.array(theta, copy=False)

#     phi %= 2 * np.pi
#     theta %= 2 * np.pi

#     phi[phi < 0] += 2 * np.pi
#     theta[theta < 0] += 2 * np.pi

#     phi[theta > np.pi] += np.pi
#     theta[theta > np.pi] = np.pi - theta[theta > np.pi]

#     phi[theta > np.pi / 2] += np.pi
#     theta[theta > np.pi / 2] = np.pi - theta[theta > np.pi / 2]

#     phi %= 2 * np.pi

#     return phi, theta


def radii_to_lati_long(phi, theta):
    phi, theta = remap_spherical(phi, theta)
    lati = 90 - np.rad2deg(theta)
    long = np.rad2deg(phi)
    long[long > 180] -= 360
    return lati, long


def on_data(phi, theta, data, phi_i, theta_i):
    ''' https://github.com/PhTrempe/sknni
    '''

    lati, long = radii_to_lati_long(phi, theta)
    lati_i, long_i = radii_to_lati_long(phi_i, theta_i)

    observations = np.vstack((np.vstack(
        (np.atleast_2d(lati), np.atleast_2d(long))), np.atleast_2d(data))).T

    interpolator = sknni.SkNNI(observations)

    interp_coords = np.append(np.atleast_2d(lati_i.ravel()),
                              np.atleast_2d(long_i.ravel()),
                              axis=0).T

    return interpolator(interp_coords)[:, 2]


def on_mesh(phi, theta, data, n_p, n_t):
    '''
    np.random.seed(43)
    phi = np.random.uniform(0, 2 * np.pi, 10)
    theta = np.random.uniform(0, np.pi, phi.size)
    data = np.random.uniform(0, 1, phi.size)
    x, y, z, data_i = on_mesh(phi, theta, data, 50, 25)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=plt.cm.jet(data_i / data_i.max()))
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')

    x = np.multiply(np.cos(phi), np.sin(theta)) * 1.1
    y = np.multiply(np.sin(phi), np.sin(theta)) * 1.1
    z = np.cos(theta) * 1.1
    ax.scatter(x,
            y,
            z,
            marker='o',
            s=50,
            c=plt.cm.jet(data / data.max()),
            alpha=1,
            cmap="jet")

    plt.show()
    '''

    theta_i, phi_i = np.meshgrid(np.linspace(np.pi, 0, n_t),
                                 np.linspace(0, 2 * np.pi, n_p))
    data_i = on_data(phi, theta, data, phi_i.ravel(), theta_i.ravel())

    u = np.linspace(0, 2 * np.pi, n_p)
    v = np.linspace(np.pi, 0, n_t)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    data_i.shape = x.shape

    return x, y, z, data_i
