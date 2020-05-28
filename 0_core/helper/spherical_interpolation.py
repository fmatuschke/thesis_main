import numpy as np
import sknni


def on_data(phi, theta, data, phi_i, theta_i):
    ''' https://github.com/PhTrempe/sknni
    '''

    # phi, theta to latitude, longitude
    lati = 90 - np.rad2deg(theta)
    long = np.rad2deg(phi)
    long[long > 180] -= 360

    lati_i = 90 - np.rad2deg(theta_i)
    long_i = np.rad2deg(phi_i)
    long_i[long_i > 180] -= 360

    observations = np.vstack((np.vstack(
        (np.atleast_2d(lati), np.atleast_2d(long))), np.atleast_2d(data))).T

    interpolator = sknni.SkNNI(observations)

    interp_coords = np.append(np.atleast_2d(lati_i.ravel()),
                              np.atleast_2d(long_i.ravel()),
                              axis=0).T

    return interpolator(interp_coords)[:, 2]


def on_mesh(phi, theta, data, n_p, n_t):
    '''
    phi = [0, 0]
    theta = [0, np.pi / 2]
    data = [20, 0]
    x, y, z, data_i = spherical_interpolation.on_mesh(phi, theta, data, 10, 10)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, facecolors=plt.cm.jet(data_i / data_i.max()))
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
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
