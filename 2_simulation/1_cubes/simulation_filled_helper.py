#%%
import numpy as np
import matplotlib.pyplot as plt
import fastpli.model.sandbox


#%%
def rot(phi):
    return np.array(((np.cos(phi), -np.sin(phi)), (np.sin(phi), np.cos(phi))),
                    float)


def plot_seeds(seeds, r):
    fix, ax = plt.subplots(1, 1)
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    for i, j, k in zip(seeds[:, 0], seeds[:, 1], r):
        print(i, j)
        c = plt.Circle((i, j), radius=k, color='b')
        ax.add_patch(c)
    ax.set_aspect('equal', 'box')
    plt.show()


#%%


def seeds_3():
    # 3 seeds for radius == 1
    r = 1 / (1 + 2 / np.sqrt(3))
    phi = np.linspace(0, 2 * np.pi, 3, False)
    x = np.cos(phi) * r / np.cos(np.deg2rad(30))
    y = np.sin(phi) * r / np.cos(np.deg2rad(30))
    seeds = np.array([x, y])
    return seeds.T.copy(), r


def seeds_12():
    # 12 seeds for radius == 1
    R = 1
    r = 1 / 4.0296019301161834974827410413
    # density = 0.739021297514213772343815403033
    phi_inner = np.linspace(0, 2 * np.pi, 3, False)
    x = np.cos(phi_inner) * r / np.cos(np.deg2rad(30))
    y = np.sin(phi_inner) * r / np.cos(np.deg2rad(30))
    seeds = np.array([x, y])

    # outer segment
    xc = -(1 - r)
    yc = 0

    phi_out = 2 * np.arcsin((r / (R - r)))

    for i in range(3):
        v_rot = np.dot(rot(np.deg2rad(120) * i), (xc, yc))
        v0 = np.dot(rot(phi_out), v_rot)
        v1 = np.dot(rot(-phi_out), v_rot)
        x = [v_rot[0], v0[0], v1[0]]
        y = [v_rot[1], v0[1], v1[1]]
        seeds = np.hstack([seeds, np.array([x, y])])
    return seeds.T.copy(), r


# %%
def fill_fb(fbs, r_mean, r_target):

    if r_mean / r_target < 1.0:
        return None
    if r_mean / r_target == 1.0:
        return fbs
    fbs_ = []

    if r_mean / r_target <= 2.0:
        # 3 fibers
        for _, fb in enumerate(fbs):
            fbs_.append([])
            for _, f in enumerate(fb):
                seeds, r = seeds_3()
                r_rel = f[:, -1] / r_mean

                new_fibers = fastpli.model.sandbox.build.bundle(
                    f[:, :-1], seeds, 0, f[:, -1])
                for i, _ in enumerate(new_fibers):
                    new_fibers[i][:, -1] = r_rel * r_mean * r
                fbs_[-1].extend(new_fibers)

    # r_mean / r_target <= 3.0 mit seeds_7 existiert, aber wird hier nicht benötigt

    elif r_mean / r_target <= 4.0:
        # 12 fibers
        for _, fb in enumerate(fbs):
            fbs_.append([])
            for _, f in enumerate(fb):
                seeds, r = seeds_12()
                r_rel = f[:, -1] / r_mean
                new_fibers = fastpli.model.sandbox.build.bundle(
                    f[:, :-1], seeds, 0, f[:, -1])
                for i, _ in enumerate(new_fibers):
                    new_fibers[i][:, -1] = r_rel * r_mean * r
                fbs_[-1].extend(new_fibers)
    else:
        for _, fb in enumerate(fbs):
            fbs_.append([])
            for _, f in enumerate(fb):
                r_mean = np.mean(f[:, -1])
                r_rel = f[:, -1] / r_mean
                seeds = fastpli.model.sandbox.seeds.triangular_circle(
                    r_mean, 2 * r_target, radii=r_target)
                new_fibers = fastpli.model.sandbox.build.bundle(
                    f[:, :-1], seeds, 0, f[:, -1] / r_mean)
                for i, _ in enumerate(new_fibers):
                    new_fibers[i][:, -1] = r_target * r_rel
                fbs_[-1].extend(new_fibers)

    return fbs_


# %%

# import fastpli.io

# fbs = fastpli.io.fiber_bundles.load(
#     '/data/PLI-Group/felix/data/thesis/1_model/1_cubes/output/cube_2pop_120/cube_2pop_psi_0.50_omega_30.00_r_5.00_v0_120_.solved.h5'
# )
# asd = fill_fb(fbs, 5, 5 / 5)

if __name__ == "__main__":
    R = 5
    fbs_0 = [[
        np.array([[0, 0, 0, R * 1.5], [0, 0, 100, R * .75], [0, 0, 200, R * 1]])
    ] * 2,
             [
                 np.array([[0, 0, 0, R * 0.25], [0, 0, 100, R * .75],
                           [0, 0, 200, R * 2]])
             ]]
    fbs_1 = fill_fb(fbs_0, R, R / 2)
    fbs_2 = fill_fb(fbs_1, R / 2, R / 10)

    for j in range(len(fbs_0)):
        for i in range(fbs_0[j][-1].shape[0]):

            fix, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.set(xlim=(-R * 2, R * 2), ylim=(-R * 2, R * 2))
            c = plt.Circle((0, 0), radius=fbs_0[j][-1][i, -1], color='r')
            ax.add_patch(c)

            for f in fbs_1[j]:
                x, y, r = f[i, 0], f[i, 1], f[i, -1]
                c = plt.Circle((x, y), radius=r, color='b')
                ax.add_patch(c)

            for f in fbs_2[j]:
                x, y, r = f[i, 0], f[i, 1], f[i, -1]
                c = plt.Circle((x, y), radius=r, color='g')
                ax.add_patch(c)

            ax.set_aspect('equal', 'box')
            plt.show()

# %%

# %%
