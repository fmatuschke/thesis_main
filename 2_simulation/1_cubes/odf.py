import numpy as np
import matplotlib.pyplot as plt
import helper.spherical_harmonics

n_theta = 42
n_phi = 42
max_band = 4

# dummy sphere
data = np.zeros(helper.spherical_harmonics._num_values(max_band))
data[0] = 1

if data.size > helper.spherical_harmonics._num_values(max_band):
    raise ValueError("data band < vis band")

odf = np.empty((n_phi, n_theta, 6), np.float32)
for phi_i, phi in enumerate(np.linspace(0, 2 * np.pi, n_phi, False)):
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    for theta_i, theta in enumerate(np.linspace(0, np.pi, n_theta, False)):
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
        color = np.random.uniform(0, 1, (3))
        # color = colorsHSV(x, y, z)

        odf[phi_i, theta_i, :] = [x, y, z, color[0], color[1], color[2]]

# plot

fig = plt.figure()  # figsize=(10, 10)
ax = fig.add_subplot(111, projection='3d')

X = odf[:, :, 0]
Y = odf[:, :, 1]
Z = odf[:, :, 2]

ax.plot_surface(X, Y, Z)
