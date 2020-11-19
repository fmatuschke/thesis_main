import numpy as np


def htm(level=0, sort=True):
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

        if sort:
            # keep points order
            points_ = np.concatenate((points, np.reshape(triangles, (-1, 3))),
                                     axis=0)
            _, idx = np.unique(points_, return_index=True, axis=0)
            points = points_[np.sort(idx), :]

    if not sort:
        points = np.unique(np.reshape(triangles, (-1, 3)), axis=0)

    return points, triangles


def htm_sc(level=0, sort=True):
    points, _ = htm(level, sort)

    phi = np.arctan2(points[:, 1], points[:, 0])
    theta = np.arccos(points[:, 2])  # length is 1

    return phi, theta


def to_stl(file_name, data):
    import struct
    # s = struct.pack('f'*len(floats), *floats)
    # f = open('file','wb')
    # f.write(s)

    with open(file_name, 'wb') as file:

        # header = np.empty((80), np.chararray)
        # np.save(file, header)

        floats = [0] * 80
        s = struct.pack('b' * 80, *floats)
        file.write(s)

        print(np.array(data).shape)

        s = struct.pack('I', np.array(data).shape[0])
        file.write(s)

        for d in data:
            # print(d)
            cross = np.cross(d[1] - d[0], d[2] - d[0])
            cross = cross / np.linalg.norm(cross)
            d = list(cross) + list(d.ravel())
            # print(list(d.ravel()))
            s = struct.pack('f' * 12, *d)
            file.write(s)

            s = struct.pack('b' * 2, 0, 0)
            file.write(s)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    phi, theta = htm_sc(1, True)

    theta = theta[np.logical_and(phi >= 0, phi <= 0.5 * np.pi)]
    phi = phi[np.logical_and(phi >= 0, phi <= 0.5 * np.pi)]
    phi = phi[np.logical_and(theta >= 0, theta <= 0.5 * np.pi)]
    theta = theta[np.logical_and(theta >= 0, theta <= 0.5 * np.pi)]

    pseudo_theta_sort_htm_0_quadrant = [2, 1, 0]
    pseudo_theta_sort_htm_1_quadrant = [2, 3, 1, 4, 5, 0]
    pseudo_theta_sort_htm_2_quadrant = [
        2, 6, 3, 7, 1, 8, 14, 12, 11, 4, 13, 5, 9, 10, 0
    ]
    pseudo_theta_sort_htm_3_quadrant = [
        2, 15, 6, 16, 3, 18, 7, 17, 1, 19, 37, 39, 38, 30, 31, 27, 25, 8, 41,
        14, 43, 12, 29, 11, 20, 40, 42, 44, 28, 26, 4, 33, 13, 35, 5, 22, 34,
        36, 24, 9, 32, 10, 21, 23, 0
    ]
    # pseudo_sort = range(theta.size)
    pseudo_sort = pseudo_theta_sort_htm_1_quadrant

    theta = theta[pseudo_sort]
    phi = phi[pseudo_sort]

    for t, p in zip(np.rad2deg(theta), np.rad2deg(phi)):
        print(f"{t:.2f}/{p:.2f}")

    r = 1.0
    x = np.multiply(np.cos(phi), np.sin(theta)) * r
    y = np.multiply(np.sin(phi), np.sin(theta)) * r
    z = np.cos(theta) * r
    for i, j, k in zip(x, y, z):
        omega = np.arccos(np.dot(np.array((1, 0, 0)), np.array((i, j, k))))
        print(f"{np.rad2deg(omega):.2f}")

    for t, p, i, j, k in zip(np.rad2deg(theta), np.rad2deg(phi), x, y, z):
        omega = np.arccos(np.dot(np.array((1, 0, 0)), np.array((i, j, k))))
        print(f"{np.rad2deg(omega):.2f}/{t:.2f}/{p:.2f}")

    # ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(data_i))

    # theta = theta[phi.argsort(kind='mergesort')]
    # phi = phi[phi.argsort(kind='mergesort')]

    # theta = np.deg2rad(np.round(np.rad2deg(theta), 0))
    # # print(np.rad2deg(theta))
    # # print(np.rad2deg(phi))
    # # print((-theta).argsort())
    # phi = np.flip(phi[theta.argsort(kind='mergesort')])
    # theta = np.flip(theta[theta.argsort(kind='mergesort')])
    # print(np.rad2deg(theta))
    # print(np.rad2deg(phi))

    r = 1.0
    x = np.multiply(np.cos(phi), np.sin(theta)) * r
    y = np.multiply(np.sin(phi), np.sin(theta)) * r
    z = np.cos(theta) * r

    color = list(range(len(phi)))
    # color.reverse()

    sc = ax.scatter(x,
                    y,
                    z,
                    marker='o',
                    s=50,
                    c=color,
                    alpha=1,
                    vmin=0,
                    vmax=len(phi) - 1,
                    cmap="viridis")

    for n, (i, j, k) in enumerate(zip(x, y, z)):
        ax.text(i, j, k, f"{n}")

    plt.colorbar(sc)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.view_init(30, 30)
    plt.show()

    # for j in range(5):
    #     p, t = htm(j)
    #     t = np.array(t)

    #     data = []
    #     for i in t:
    #         tmp = []
    #         for p in i:
    #             tmp.append(p)
    #         # tmp.append(np.cross(i[1] - i[0], i[2] - i[0]))
    #         data.append(np.array(tmp))

    #     to_stl(f"htm_{j}.stl", data)

    # with open("test.tex", "w") as file:
    #     p, t = htm(0)
    #     t = np.array(t)

    #     for i in range(t.shape[0]):
    #         if np.sum(t[i, :, 2] < 0) > 1:
    #             continue

    #         if np.sum(t[i, :, 2] <= 0) == 3:
    #             continue

    #         file.write(
    #             f"\draw[thick, black] ({t[i,0,0]:.3f}, {t[i,0,1]:.3f}, {t[i,0,2]:.3f}) -- ({t[i,1,0]:.3f}, {t[i,1,1]:.3f}, {t[i,1,2]:.3f}) -- ({t[i,2,0]:.3f}, {t[i,2,1]:.3f}, {t[i,2,2]:.3f}) -- cycle;\n"
    #         )

    #     for i in range(p.shape[0]):
    #         if p[i, 2] < 0:
    #             continue
    #         file.write(
    #             f"\\fill ({p[i,0]:.3f}, {p[i,1]:.3f}, {p[i,2]:.3f}) circle (0.01);\n"
    #         )

    #     p, t = htm(1)
    #     t = np.array(t)

    #     for i in range(t.shape[0]):
    #         if np.sum(t[i, :, 2] < 0) > 1:
    #             continue

    #         if np.sum(t[i, :, 2] <= 0) == 3:
    #             continue

    #         file.write(
    #             f"\draw[dashed, black] ({t[i,0,0]:.3f}, {t[i,0,1]:.3f}, {t[i,0,2]:.3f}) -- ({t[i,1,0]:.3f}, {t[i,1,1]:.3f}, {t[i,1,2]:.3f}) -- ({t[i,2,0]:.3f}, {t[i,2,1]:.3f}, {t[i,2,2]:.3f}) -- cycle;\n"
    #         )

    #     for i in range(p.shape[0]):
    #         if p[i, 2] < 0:
    #             continue
    #         file.write(
    #             f"\\fill ({p[i,0]:.3f}, {p[i,1]:.3f}, {p[i,2]:.3f}) circle (0.01);\n"
    #         )
