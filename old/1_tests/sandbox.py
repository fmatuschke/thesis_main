import fastpli.model.sandbox as sandbox
import fastpli.io

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


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


def plot_fiber_bundle(fb, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    for fiber in fb:
        ax.plot(fiber[:, 0], fiber[:, 1], fiber[:, 2])
    set_axes_equal(ax)


# define a fiber bundle trajectory
t = np.linspace(-0.25 * np.pi, 1.25 * np.pi, 84, True)
traj = np.array((np.cos(t) * 42, np.sin(t) * 42, t * 0)).T

seeds = sandbox.seeds.triangular_grid(a=42, b=42, spacing=8, center=True)
circ_seeds = sandbox.seeds.crop_circle(radius=21, seeds=seeds)

fiber_bundle = sandbox.build.bundle(traj, circ_seeds, 4.2)

fastpli.io.fiber_bundles.save("sandbox.dat", [fiber_bundle], mode="w")

# df = pd.DataFrame()

# for i, f in enumerate(fiber_bundle):
#     df[f"x{i}"] = f[:, 0]
#     df[f"y{i}"] = f[:, 1]
#     df[f"z{i}"] = f[:, 2]

# print(i)
# df.to_csv("sandbox.csv", index=False, float_format='%.3f')

# # plot_fiber_bundle(fiber_bundle)
# # plt.show()
