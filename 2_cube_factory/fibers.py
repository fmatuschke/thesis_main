import numpy as np
import h5py

import fastpli.analysis
import fastpli.tools
import fastpli.io


def inclinations(n=10):
    return np.linspace(0, np.pi / 2, n, True)


def omega_rotations(omega, dphi=np.deg2rad(10)):

    rot = []

    n_rot = int(np.round(np.sqrt((1 - np.cos(2 * omega)) / (1 - np.cos(dphi)))))
    if n_rot == 0:
        rot.append(0)
    else:
        n_rot += (n_rot + 1) % 2
        n_rot = max(n_rot, 3)
        for f_rot in np.linspace(-180, 180, n_rot, True):
            f_rot = np.round(f_rot, 2)
            rot.append(f_rot)

    return rot


def rotate(fbs, f0_inc, f1_rot):
    if f0_inc != 0 or f1_rot != 0:
        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1_rot))
        rot = np.dot(rot_inc, rot_phi)
        fbs = fastpli.objects.fiber_bundles.Rotate(fbs, rot)
    return fbs


def ori_from_fbs(fbs, omega, f0_inc=0, f1_rot=0, cut=None):
    fbs = rotate(fbs, f0_inc, f1_rot)
    if cut:
        if isinstance(cut, (int, float)):
            cut = [[-cut / 2] * 3, [cut / 2] * 3]
        fbs = fastpli.objects.fiber_bundles.Cut(fbs, cut)
    return fastpli.analysis.orientation.fiber_bundles(fbs)


def ori_from_file(file, f0_inc, f1_rot, cut=None):
    fbs = fastpli.io.fiber_bundles.load(file)
    with h5py.File(file, "r") as h5f:
        omega = h5f['/'].attrs['omega']
    return ori_from_fbs(fbs, omega, f0_inc, f1_rot, cut)