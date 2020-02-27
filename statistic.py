import numpy as np
import matplotlib.pyplot as plt


def _remap_direction(phi):
    phi = phi % np.pi
    phi[phi < 0] += np.pi
    return phi


def _remap_direction_sym(phi):
    phi = phi % np.pi
    phi[phi < 0] += np.pi
    phi[phi > np.pi / 2] -= np.pi
    return phi


# def _remap_direction(phi):
#     phi = np.unwrap(phi)
#     phi[phi < 0] += np.pi
#     return phi

# def _remap_direction_sym(phi):
#     phi = _remap_direction(phi)
#     phi[phi > np.pi / 2] -= np.pi
#     return phi


def _remap_orientation(phi, theta):
    phi = phi % (2 * np.pi)
    theta = theta % np.pi

    phi[phi < 0] += 2 * np.pi

    phi[theta < 0] += np.pi
    theta = np.abs(theta)

    phi[theta > .5 * np.pi] += np.pi
    theta[theta > .5 * np.pi] = np.pi - theta[theta >= .5 * np.pi]

    phi = phi % (2 * np.pi)

    if np.any(phi < 0) or np.any(phi >= 2 * np.pi) or np.any(
            theta < 0) or np.any(theta > 0.5 * np.pi):
        raise ValueError

    return phi, theta


def direction_histogram(data, N=2 * 36, ax=None):
    data = np.array(data, copy=False)
    data = _remap_direction(data.flatten())
    data = np.vstack([data, data + np.pi]).flatten()
    hist, bins = np.histogram(data,
                              np.linspace(0, 2 * np.pi, N + 1, True),
                              density=True)

    # data = np.rad2deg(data)
    # hist, bins = np.histogram(data,
    #                           np.linspace(0, 360, N + 1, True),
    #                           density=True)

    if ax:
        colors = plt.cm.viridis(hist / np.amax(hist))
        ax.bar(np.deg2rad(bins[:-1]),
               hist,
               width=2 * np.pi / N,
               bottom=0,
               color=colors)
        # ax.plot(np.deg2rad(bins[:-1]), hist)

    return (bins[:-1] + bins[1:]) / 2, hist


def orientation_histogram(phi, theta, Nphi=4 * 36, Ntheta=2 * 18, ax=None):
    phi = np.array(phi, copy=False)
    theta = np.array(theta, copy=False)
    phi, theta = _remap_orientation(phi.flatten(), theta.flatten())
    phi = np.rad2deg(phi)
    theta = np.rad2deg(theta)

    #calculate histogram
    H, xedges, yedges = np.histogram2d(phi,
                                       theta, [
                                           np.linspace(0, 360, Nphi + 1, True),
                                           np.linspace(0, 90, Ntheta + 1, True)
                                       ],
                                       density=True)

    H = H.T

    if ax:
        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(np.deg2rad(X), np.deg2rad(Y), H, cmap="viridis")

    X, Y = np.meshgrid(xedges + (xedges[1] - xedges[0]) / 2,
                       yedges + (yedges[1] - yedges[0]) / 2)

    H = np.concatenate([H, np.atleast_2d(H[0, :])], axis=0)
    H = np.concatenate([H, np.atleast_2d(H[:, 0]).T], axis=1)

    # X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
    #                    (yedges[:-1] + yedges[1:]) / 2)

    return H, X, Y


if __name__ == '__main__':
    fig = plt.figure()
    ax = plt.subplot(211, polar=True)
    direction = np.random.normal(0, 0.2, 1000)
    direction_histogram(direction, 2 * 36, ax=ax)
    ax.set_yticklabels([])

    ax = plt.subplot(212, polar=True)
    inclination = np.random.normal(0.2, 0.2, 1000)
    h, x, y = orientation_histogram(direction,
                                    np.pi / 2 - inclination,
                                    18,
                                    6,
                                    ax=ax)
    ax.set_yticklabels([])

    with open('ohist.dat', 'w') as f:
        for i, j, k in zip(x, y, h):
            for a, b, c in zip(i, j, k):
                f.write(f"{a} {b} {c}\n")
            f.write("\n")

    plt.show()
