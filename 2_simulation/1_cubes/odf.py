import numpy as np
import matplotlib.pyplot as plt
import helper.spherical_harmonics
import fastpli.analysis


def table(directions,
          inclinations,
          max_band=6,
          n_theta=21,
          n_phi=42,
          file_name=None,
          mode="odf"):
    data = helper.spherical_harmonics.real_spherical_harmonics(
        directions, np.pi / 2 - inclinations, max_band)

    odf = np.empty((n_phi, n_theta, 6), np.float32)
    for phi_i, phi in enumerate(np.linspace(0, 2 * np.pi, n_phi, True)):
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        for theta_i, theta in enumerate(np.linspace(0, np.pi, n_theta, True)):
            costheta = np.cos(theta)
            sintheta = np.sin(theta)

            r = 0
            c = 0
            for band in range(0, max_band + 1, 2):
                for order in range(-band, band + 1):
                    r += helper.spherical_harmonics._spherical_harmonics(
                        band, order, costheta, sintheta, phi) * data[c]
                    c += 1

            x = r * sintheta * cosphi
            y = r * sintheta * sinphi
            z = r * costheta
            color = fastpli.analysis.images._orientation_to_hsv(
                phi, np.pi / 2 - theta) / 255

            odf[phi_i, theta_i, :] = [x, y, z, color[0], color[1], color[2]]

    if mode == "odf":
        if file_name is not None:
            with open(file_name, "w") as file:
                file.write(f"x y z rgb\n")
                for i in range(odf.shape[0]):
                    for j in range(odf.shape[1]):
                        file.write(
                            f"{odf[i,j,0]:.3f} {odf[i,j,1]:.3f} {odf[i,j,2]:.3f} {odf[i,j,3]:.3f},{odf[i,j,4]:.3f},{odf[i,j,5]:.3f}\n"
                        )
                    file.write("\n")

        return odf

    elif mode == "hist":
        r = np.sqrt(odf[:, :, 0]**2 + odf[:, :, 1]**2 + odf[:, :, 2]**2)
        phi = np.arctan2(odf[:, :, 1], odf[:, :, 0])
        theta = np.arccos(odf[:, :, 2] / r)
        hist = np.concatenate((phi, theta, r))

        if file_name is not None:
            with open(file_name, "w") as file:
                file.write(f"x y z rgb\n")
                for i in range(odf.shape[0]):
                    for j in range(odf.shape[1]):
                        # r = np.sqrt(odf[i, j, 0]**2 + odf[i, j, 1]**2 +
                        #             odf[i, j, 2]**2)
                        # phi = np.arctan2(odf[i, j, 1], odf[i, j, 0])
                        # theta = np.arcsin(odf[i, j, 2] / r)
                        file.write(
                            f"{hist[i,j,0]} {hist[i,j,1]} {hist[i,j,2]}\n")
                    file.write("\n")

        return hist


def plot(odf_table):
    # plot
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure()  # figsize=(10, 10)
    ax = fig.add_subplot(111, projection='3d')

    X = odf_table[:, :, 0]
    Y = odf_table[:, :, 1]
    Z = odf_table[:, :, 2]

    ax.plot_surface(X, Y, Z, facecolors=odf_table[:, :, 3:])
    set_axes_equal(ax)


if __name__ == "__main__":

    n_theta = 21
    n_phi = 42
    max_band = 4

    # dummy sphere
    data = np.zeros(helper.spherical_harmonics._num_values(max_band))
    data[0] = 1

    # dummy
    phi = np.array([0, 0, 0, 0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
    theta = np.array([
        np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2,
        np.pi / 2, np.pi / 2
    ])

    odf = table(phi, np.pi / 2 - theta, mode="odf")

    # plot
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure()  # figsize=(10, 10)
    ax = fig.add_subplot(111, projection='3d')

    X = odf[:, :, 0]
    Y = odf[:, :, 1]
    Z = odf[:, :, 2]

    ax.plot_surface(X, Y, Z, facecolors=odf[:, :, 3:])
    set_axes_equal(ax)
    plt.show()
