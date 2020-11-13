import numpy as np


def htm(level=0):
    '''
    Hierarchical Triangular Mesh
    https://arxiv.org/pdf/cs/0701164.pdf
    '''

    points = np.array([(0, 0, 1), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
                       (0, 0, -1)])

    triangle_indices = [(1, 5, 2), (2, 5, 3), (3, 5, 4), (4, 5, 1), (1, 0, 4),
                        (4, 0, 3), (3, 0, 2), (2, 0, 1)]

    triangles = []
    for i, j, k in triangle_indices:
        triangles.append((points[i], points[j], points[k]))

    points = np.unique(np.reshape(triangles, (-1, 3)), axis=0)

    for i in range(level):
        new_triangles = []
        w = [None] * 3
        for t in triangles:
            w[0] = t[1] + t[2]
            w[0] = w[0] / np.linalg.norm(w[0])

            w[1] = t[0] + t[2]
            w[1] = w[1] / np.linalg.norm(w[1])

            w[2] = t[0] + t[1]
            w[2] = w[2] / np.linalg.norm(w[2])

            new_triangles.append(np.array((t[0], w[2], w[1])))
            new_triangles.append(np.array((t[1], w[0], w[2])))
            new_triangles.append(np.array((t[2], w[1], w[0])))
            new_triangles.append(np.array((w[0], w[1], w[2])))

        triangles = new_triangles

        # keep points order
        points_ = np.concatenate((points, np.reshape(triangles, (-1, 3))),
                                 axis=0)
        _, idx = np.unique(points_, return_index=True, axis=0)
        points = points_[np.sort(idx), :]

    return points


def htm_sc(level=0):
    points = htm(level)

    phi = np.arctan2(points[:, 1], points[:, 0])
    theta = np.arccos(points[:, 2])  # length is 1

    return phi, theta


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # print(htm(1))
    # print(htm_sc(1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    phi, theta = htm_sc(2)

    theta = theta[np.logical_and(phi >= 0, phi <= 0.5 * np.pi)]
    phi = phi[np.logical_and(phi >= 0, phi <= 0.5 * np.pi)]
    phi = phi[np.logical_and(theta >= 0, theta <= 0.5 * np.pi)]
    theta = theta[np.logical_and(theta >= 0, theta <= 0.5 * np.pi)]

    # ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(data_i))

    print(phi.shape)

    r = 1.0
    x = np.multiply(np.cos(phi), np.sin(theta)) * r
    y = np.multiply(np.sin(phi), np.sin(theta)) * r
    z = np.cos(theta) * r

    sc = ax.scatter(x,
                    y,
                    z,
                    marker='o',
                    s=50,
                    c=range(len(phi)),
                    alpha=1,
                    vmin=0,
                    vmax=len(phi) - 1,
                    cmap="viridis")
    plt.colorbar(sc)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.view_init(30, 30)
    plt.show()
