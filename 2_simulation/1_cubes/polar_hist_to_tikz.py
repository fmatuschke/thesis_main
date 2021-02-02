import re
import shutil
import tempfile
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
import glob
import fastpli.tools
import helper.spherical_interpolation
import subprocess


def sed(pattern, replace, source, dest=None):

    fin = open(source, 'r')

    if dest:
        fout = open(dest, 'w')
    else:
        fd, name = tempfile.mkstemp()
        fout = open(name, 'w')

    for line in fin:
        out = re.sub(pattern, replace, line)
        fout.write(out)

    fin.close()
    fout.close()

    if dest is None:
        shutil.copy(name, source)


def omega_rot_to_spherical(omega, f1_rot):
    phi = []
    theta = []
    for f1, om in zip(f1_rot, omega):
        v = np.array([np.cos(np.deg2rad(om)), np.sin(np.deg2rad(om)), 0])
        rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
        rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1))
        rot = np.dot(rot_inc, rot_phi)
        v = np.dot(rot, v)
        theta.extend([np.arccos(v[2])])
        phi.extend([np.arctan2(v[1], v[0])])

    phi = np.array(phi)
    theta = np.array(theta)


def generate(df,
             value,
             file_name,
             crange=None,
             size=None,
             delta=None,
             d_theta=19,
             d_phi=37,
             f0_list=None,
             psi_list=None):

    if f0_list is None:
        f0_list = sorted(df.f0_inc.unique())
    if psi_list is None:
        psi_list = np.array(sorted(df.psi.unique()))
        psi_list = psi_list[psi_list > 0]  # redundant

    f0_list = np.array(f0_list, float)
    psi_list = np.array(psi_list, float)

    f0_list_str = ["%.2f" % x for x in f0_list]
    f0_list_str = ",".join(f0_list_str)
    psi_list_str = ["%.2f" % x for x in psi_list]
    psi_list_str = ",".join(psi_list_str)

    if crange is None:
        crange = (np.min(df[value]), np.max(df[value]))

    if size is None:
        size = 0.9 * 13.8 / len(psi_list)

    if delta is None:
        delta = 4.25  # for 4

    sed("@file_name", file_name, f"polar_hist_to_tikz.tex",
        f"output/tmp/{file_name}.tex")
    sed("@delta_img", str(delta), f"output/tmp/{file_name}.tex")
    sed("@fnull_list", f0_list_str, f"output/tmp/{file_name}.tex")
    sed("@psi_list", psi_list_str, f"output/tmp/{file_name}.tex")
    sed("@cmin", str(crange[0]), f"output/tmp/{file_name}.tex")
    sed("@cmax", str(crange[1]), f"output/tmp/{file_name}.tex")
    sed("@size", str(size) + "cm", f"output/tmp/{file_name}.tex")

    for psi in psi_list:
        for f0_inc in f0_list:
            df_ = df[(df["psi"] == psi) & (df["f0_inc"] == f0_inc)]
            data = df_[value].to_numpy(float)

            if len(df_[df_.omega == 0]) != 1:
                print("FOOOOO")
                exit(1)

            # get points on sphere
            phi = []
            theta = []
            for f1, om in zip(df_.f1_rot, df_.omega):
                v = np.array(
                    [np.cos(np.deg2rad(om)),
                     np.sin(np.deg2rad(om)), 0])
                rot_inc = fastpli.tools.rotation.y(-np.deg2rad(f0_inc))
                rot_phi = fastpli.tools.rotation.x(np.deg2rad(f1))
                rot = np.dot(rot_inc, rot_phi)
                v = np.dot(rot, v)
                theta.extend([np.arccos(v[2])])
                phi.extend([np.arctan2(v[1], v[0])])

            phi = np.array(phi)
            theta = np.array(theta)

            ## apply symmetries
            phi_ = phi.copy()
            theta_ = theta.copy()
            data_ = data.copy()
            # measurement symmetry
            phi_ = np.concatenate((phi_, -phi_), axis=0)
            theta_ = np.concatenate((theta_, theta_), axis=0)
            data_ = np.concatenate((data_, data_), axis=0)
            # orientation symmetry
            phi_ = np.concatenate((phi_, phi_), axis=0)
            theta_ = np.concatenate((theta_, np.pi + theta_), axis=0)
            data_ = np.concatenate((data_, data_), axis=0)

            # rm multiple
            phi_, theta_ = helper.spherical_interpolation.remap_spherical(
                phi_, theta_)

            x_ = np.multiply(np.cos(phi_), np.sin(theta_))
            y_ = np.multiply(np.sin(phi_), np.sin(theta_))
            z_ = np.cos(theta_)

            # remove multiple by checking distance
            data__ = []
            for i in range(phi_.size):
                flag = True
                for j in range(i + 1, phi_.size):
                    dist = np.linalg.norm(
                        np.array((x_[i] - x_[j], y_[i] - y_[j], z_[i] - z_[j])))

                    if dist < 1e-6:
                        flag = False
                        break
                if flag:
                    data__.append((phi_[i], theta_[i], data_[i]))

            data__ = np.array(data__)
            phi_ = data__[:, 0]
            theta_ = data__[:, 1]
            data_ = data__[:, 2]

            # interplate mesh on sphere
            x_i, y_i, z_i, data_i = helper.spherical_interpolation.on_mesh(
                phi_, theta_, data_, d_phi, d_theta)

            with open(
                    f"output/tmp/{file_name}_psi_{psi:.2f}_f0_{f0_inc:.2f}_hist.dat",
                    "w") as f:
                for h_array, x_array, y_array, z_array in zip(
                        data_i, x_i, y_i, z_i):
                    for h, x, y, z in zip(h_array, x_array, y_array, z_array):
                        p = np.rad2deg(np.arctan2(y, x))
                        t = np.rad2deg(np.arccos(z))

                        if t <= 90:
                            f.write(f'{p:.2f} {t:.2f} {h:.6f}\n')
                    f.write('\n')

            with open(
                    f"output/tmp/{file_name}_psi_{psi:.2f}_f0_{f0_inc:.2f}_init.dat",
                    "w") as f:
                f.write(f'{0} {90-f0_inc}\n')
                f.write('\n')

            with open(
                    f"output/tmp/{file_name}_psi_{psi:.2f}_f0_{f0_inc:.2f}_data.dat",
                    "w") as f:
                for d, p, t in zip(data_, phi_, theta_):
                    if t <= np.pi / 2:
                        f.write(
                            f'{np.rad2deg(p):.2f} {np.rad2deg(t):.2f} {d:.6f}\n'
                        )
                f.write('\n')

    with open(f"output/tmp/{file_name}.out",
              "wb") as out, open(f"output/tmp/{file_name}.err", "wb") as err:
        subprocess.run(
            [
                "lualatex", "-interaction=nonstopmode", "-halt-on-error",
                f"{file_name}.tex"
            ],
            # shell=True,
            check=True,
            stdout=out,
            stderr=err,
            cwd="output/tmp")

    shutil.copyfile(f"output/tmp/{file_name}.pdf",
                    f"output/tikz/{file_name}.pdf")

    fileList = glob.glob(f'output/tmp/{file_name}*')
    for file in fileList:
        os.remove(file)

    return crange
