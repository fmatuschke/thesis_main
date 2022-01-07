import numpy as np
import os
import argparse
import multiprocessing as mp
import pandas as pd
import tqdm

import fastpli.tools
import helper.spherical_interpolation

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i",
                    "--input",
                    type=str,
                    required=True,
                    help="input path.")
parser.add_argument("-p",
                    "--num_proc",
                    default=1,
                    type=int,
                    help="Number of processes.")
args = parser.parse_args()


def _nx4_dat(x, y, z, data, file, info=None):

    with open(file, 'w') as f:

        if info:
            if not isinstance(info, list):
                info = [info]

            for i in info:
                f.write(f"%{i}\n")

        f.write(f"x,y,z,c\n")

        for ii in range(x.shape[0]):
            for i, j, k, l in zip(x[ii, :], y[ii, :], z[ii, :], data[ii, :]):
                f.write(f"{i:.3f},{j:.3f},{k:.3f},{l:.3f}\n")
            if ii != x.shape[0] - 1:
                f.write("\n")


def tikz_sphere(x,
                y,
                z,
                data,
                file,
                x2=None,
                y2=None,
                z2=None,
                data2=None,
                f0_inc=0,
                path_to_data=None,
                standalone=False,
                only_dat=False,
                info=None):

    # if x.size > 1:
    #     data = np.vstack((x, y, z, data))
    #     for i in range(3):
    #         data = data[:, data[i, :].argsort()]
    #     x, y, z, data = data[0, :], data[1, :], data[2, :], data[3, :]

    if x.size != y.size or x.size != z.size or x.size != data.size or x2.size != y2.size or x2.size != z2.size or x2.size != data2.size:
        raise TypeError("FOOO type")

    file_path = os.path.dirname(file)
    file_base = os.path.basename(file)
    file_name, _ = os.path.splitext(file_base)
    file_pre = os.path.join(file_path, file_name)

    _nx4_dat(x, y, z, data, f"{file_pre}_0.dat", info)

    if x2 is not None:
        _nx4_dat(np.atleast_2d(x2), np.atleast_2d(y2), np.atleast_2d(z2),
                 np.atleast_2d(data2), f"{file_pre}_1.dat", info)

    if only_dat:
        return

    if path_to_data:
        file_name = path_to_data + "/" + file_name

    with open(file, 'w') as f:
        if standalone:
            f.write("" \
                "\\documentclass[]{standalone}\n"\
                "\\usepackage{pgfplots}\n" \
                "\\pgfplotsset{compat=1.17}\n" \
                "\\usepackage{siunitx}\n" \
                "\\begin{document}\n" \
                "\\tikzset{>=latex}\n" \
                "%\n")
        f.write("" \
            "\\begin{tikzpicture}[trim axis left, baseline]\n" \
            "\\begin{axis}[%\n" \
            "    axis equal,\n" \
            # "    axis lines = center,\n" \
            "    axis line style={opacity=0.0},\n" \
            "    ticks=none,\n" \
            "    view/h=145,\n" \
            "    scale uniformly strategy=units only,\n" \
            "    clip=false, % hide axis,\n" \
            "    width=10cm,\n" \
            "    height=10cm,\n" \
            "    xmin=-1,xmax=1,\n" \
            "    ymin=-1,ymax=1,\n" \
            "    zmin=-1,zmax=1,\n" \
            "    point meta min=0, point meta max=1,\n" \
            "    colormap/viridis,\n" \
            "]\n"
        )
        f.write("" \
            "\\draw[black, thick] (axis cs:-1.5, 0, 0) -- (axis cs:1, 0, 0) {};\n"
            "\\draw[black, thick] (axis cs:0, -1.5, 0) -- (axis cs:0, 1, 0) {};\n"
            "\\draw[black, thick] (axis cs:0, 0, -1.5) -- (axis cs:0, 0, 1) {};\n"
            )
        f.write("" \
            "\\addplot3[\n" \
            "    surf,\n" \
            "    opacity = 0.75,\n" \
            "    z buffer=sort]\n" \
            "    table[x=x,y=y,z=z,point meta=\\thisrow{c}, col sep=comma]\n" \
            f"        {{{file_name}_0.dat}};\n"
            )
        f.write("" \
            "\\addplot3[\n" \
            "    dashed, color=black,\n" \
            "    opacity = 0.75,\n" \
            "    domain=-45:90, y domain=0:0, samples=42]\n" \
            "    ({cos(x)},0,{sin(x)});\n"
            )
        if x2 is not None:
            f.write("" \
                "\\addplot3[\n" \
                "    scatter, only marks,\n" \
                "    z buffer=sort]\n" \
                "    table[x=x,y=y,z=z,point meta=\\thisrow{c}, col sep=comma]\n" \
                f"        {{{file_name}_1.dat}};\n"
            )
            x, y, z = np.cos(np.deg2rad(f0_inc)), 0, np.sin(np.deg2rad(f0_inc))
            f.write("" \
                "\\addplot3[\n" \
                "    only marks, thick,\n" \
                "    mark=o, mark color=black]\n" \
                f"        coordinates {{ ({x:.2f},{y:.2f},{z:.2f}) }};\n"
                )
        f.write("" \
            "\\draw[->, black, thick] (axis cs:1, 0, 0) -- (axis cs:1.5, 0, 0) node[pos=0.5, below]{$x$};\n"
            "\\draw[->, black, thick] (axis cs:0, 1, 0) -- (axis cs:0, 1.5, 0) node[pos=0.5, below]{$y$};\n"
            "\\draw[->, black, thick] (axis cs:0, 0, 1) -- (axis cs:0, 0, 1.5) node[pos=0.5, right]{$z$};\n"
            )
        f.write("" \
            "\end{axis}\n" \
            "\\end{tikzpicture}\n" \
            )

        if standalone:
            f.write("%\n")
            f.write("\\end{document}\n")


sim_path = args.input
os.makedirs(os.path.join(sim_path, "images"), exist_ok=True)
df = pd.read_pickle(
    os.path.join(sim_path, "analysis", f"cube_2pop_simulation.pkl"))
df_acc = pd.read_pickle(
    os.path.join(sim_path, "analysis", f"cube_2pop_simulation_schilling.pkl"))


def run(p):
    radius = p[1].radius
    microscope = p[1].microscope
    species = p[1].species
    model = p[1].model
    psi = p[1].psi
    f0_inc = p[1].f0_inc

    file_name = (f"sphere_r_{radius}_{microscope}_species_{species}"
                 f"_model_{model}_psi_{psi:.2f}_f0_inc_{f0_inc:.2f}")

    sub = (df_acc.radius == radius) & (df_acc.microscope == microscope) & (
        df_acc.species == species) & (df_acc.model == model) & (
            df_acc.psi == psi) & (df_acc.f0_inc == f0_inc)

    sub_ref = (df_acc.radius == radius) & (df_acc.microscope == microscope) & (
        df_acc.species == species) & (df_acc.model == model) & (
            df_acc.psi == 0.0) & (df_acc.f0_inc == 0.0)

    if len(df_acc[sub_ref]) != 1:
        print(df_acc[sub_ref])
        raise ValueError(f"FOOO:1: {len(df_acc[sub_ref])}")
    if psi != 0.0:
        if len(df_acc[sub]) != 22:
            print(df_acc[sub])
            raise ValueError(f"FOOO:22: {len(df_acc[sub])}")

    f1_rot = df_acc[sub].f1_rot.to_numpy(float)
    omega = df_acc[sub].omega.to_numpy(float)

    for name in ["acc", "R", "R2"]:

        data = df_acc[sub][name].to_numpy(float)

        # norm R and R2 to ref
        if name != "acc":
            data /= np.mean(df_acc[sub_ref][name].to_numpy(float))

        # get points on sphere
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

        phi_ = phi.copy()
        theta_ = theta.copy()
        data_ = data.copy()

        # # apply symmetry
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
        phi_, theta_ = helper.spherical_interpolation.remap_sphere(phi_, theta_)

        x_ = np.multiply(np.cos(phi_), np.sin(theta_))
        y_ = np.multiply(np.sin(phi_), np.sin(theta_))
        z_ = np.cos(theta_)

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
            phi_, theta_, data_, 20, 20)

        r = 1
        x = np.multiply(np.cos(phi), np.sin(theta)) * r
        y = np.multiply(np.sin(phi), np.sin(theta)) * r
        z = np.cos(theta) * r

        tikz_sphere(x_i,
                    y_i,
                    z_i,
                    data_i,
                    f"{os.path.join(sim_path,'images',file_name)}_{name}.tikz",
                    x,
                    y,
                    z,
                    data,
                    f0_inc,
                    path_to_data="\\currfiledir",
                    standalone=False)


if __name__ == "__main__":

    df_p = df[[
        "radius",
        "microscope",
        "species",
        "model",
        "psi",
        "f0_inc",
    ]].drop_duplicates()

    with mp.Pool(processes=args.num_proc) as pool:
        df = [
            d for d in tqdm.tqdm(pool.imap_unordered(run, df_p.iterrows()),
                                 total=len(df_p),
                                 smoothing=0.1)
        ]

    # for _, p in tqdm.tqdm(df_p.iterrows(), total=len(df_p)):
