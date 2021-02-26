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


def seeds_12(r):
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

    v = (xc, yc)
    for i in range(3):
        v_rot = np.dot(rot(np.deg2rad(120) * i), (xc, yc))
        v0 = np.dot(rot(phi_out), v_rot)
        v1 = np.dot(rot(-phi_out), v_rot)
        x = [v_rot[0], v0[0], v1[0]]
        y = [v_rot[1], v0[1], v1[1]]
        seeds = np.hstack([seeds, np.array([x, y])])
    return seeds.T


# %%
def fill_fb(fbs, r_mean):

    if r_mean == 0.5:
        return fbs

    fbs_ = []

    if r_mean == 1.0:
        for i, fb in enumerate(fbs):
            fbs_.append([])
            for j, f in enumerate(fb):
                r = 1 / (1 + 2 / np.sqrt(3))
                phi = np.linspace(0, 2 * np.pi, 3, False)
                x = np.cos(phi) * r / np.cos(np.deg2rad(30))
                y = np.sin(phi) * r / np.cos(np.deg2rad(30))
                seeds = np.array([x, y]).T

                fbs_.append(
                    fastpli.model.sandbox.build.bundle(f[:, :-1], seeds, r,
                                                       f[:, -1]))
    elif r_mean == 2.0:
        r = 2 / 4.0296019301161834974827410413
        for i, fb in enumerate(fbs):
            fbs_.append([])
            for j, f in enumerate(fb):
                seeds = seeds_12(1)
                fbs_.append(
                    fastpli.model.sandbox.build.bundle(f[:, :-1], seeds, r,
                                                       f[:, -1]))
    else:
        for i, fb in enumerate(fbs):
            fbs_.append([])
            for j, f in enumerate(fb):
                r_mean = np.mean(f[:, -1])
                r_rel = f[:, -1] / r_mean
                seeds = fastpli.model.sandbox.seeds.triangular_circle(r_mean,
                                                                      1,
                                                                      radii=0.5)
                fbs_.append(
                    fastpli.model.sandbox.build.bundle(f[:, :-1], seeds, 0.5,
                                                       r_rel))

    return fbs_


# %%

if __name__ == "__main__":
    R = 5
    fbs = fill_fb([[np.array([[0, 0, 0, R], [0, 0, 100, R]])]], R)

    fix, ax = plt.subplots(1, 1)
    ax.set(xlim=(-R * 2, R * 2), ylim=(-R * 2, R * 2))
    c = plt.Circle((0, 0), radius=R, color='r')
    ax.add_patch(c)

    for fb in fbs:
        for f in fb:
            for x, y, r in zip(f[:, 0], f[:, 1], f[:, -1]):
                c = plt.Circle((x, y), radius=r, color='b')
                ax.add_patch(c)

    ax.set_aspect('equal', 'box')
    plt.show()

# %%
