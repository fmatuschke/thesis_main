import numpy as np
import os


def hist2d_2_pgf_dat(h, x, y, file, info=""):

    x, y = np.meshgrid(x + (x[1] - x[0]) / 2, y + (y[1] - y[0]) / 2)
    h = h.T
    h = np.concatenate([h, np.atleast_2d(h[0, :])], axis=0)
    h = np.concatenate([h, np.atleast_2d(h[:, 0]).T], axis=1)

    with open(file, 'w') as f:

        if not isinstance(info, list):
            info = [info]

        for i in info:
            f.write(f"%{i}\n")

        f.write("\n")
        for i, j, k in zip(x, y, h):
            for a, b, c in zip(i, j, k):
                f.write(f"{a:.2f} {b:.2f} {c}\n")
            f.write("\n")


def hist2d_2_tikz(h,
                  x,
                  y,
                  file,
                  y_label=None,
                  path_to_data=None,
                  standalone=False,
                  info=""):

    file_path = os.path.dirname(file)
    file_base = os.path.basename(file)
    file_name, _ = os.path.splitext(file_base)
    file_pre = os.path.join(file_path, file_name)

    hist2d_2_pgf_dat(h, x, y, f"{file_pre}.dat", info)

    if path_to_data:
        file_name = path_to_data + "/" + file_name

    with open(file, 'w') as f:
        if standalone:
            f.write("\\documentclass[]{standalone}\n")
            f.write("\\usepackage{pgfplots}\n")
            f.write("\\usepgfplotslibrary{polar}\n")
            f.write("\\pgfplotsset{compat=1.17}\n")
            f.write("\\usepackage{siunitx}\n")
            f.write("\\begin{document}\n")
            f.write("%\n")
        f.write("\\begin{tikzpicture}[trim axis left, baseline]\n")
        f.write("\\begin{polaraxis}[\n")
        f.write("    xtick={0,45,...,315},\n")
        f.write(
            "    xticklabels={$\\ang{0}$, $\\ang{45}$, $\\ang{90}$, " \
                "$\\ang{135}$, $\\ang{180}$,\n")
        f.write("                 $\\ang{225}$, $\\ang{270}$, $\\ang{315}$},\n")
        f.write("    ytick={20,40,...,80},\n")
        f.write("    yticklabels={$\\ang{80}$, $\\ang{60}$, $\\ang{40}$, " \
                    "$\\ang{20}$},\n")
        f.write("    yticklabel style=white,\n")
        # f.write("    ytick=\\empty,\n")
        # f.write("    y tick label style={anchor=south east},")
        f.write("    colormap/viridis,\n")
        f.write("    tickwidth=0,\n")
        f.write("    xtick distance = 45,\n")
        # f.write("    % separate axis lines,\n")
        f.write("    y axis line style= { draw opacity=0 },\n")
        f.write("    ymin=0, ymax=90,\n")
        f.write("    axis on top=true,\n")
        f.write("    colorbar,\n")
        f.write("    colorbar style={\n")
        f.write("        tickwidth=0,\n")
        if y_label:
            f.write(f"        ylabel={{{y_label}}},\n")
        f.write("    },\n")
        f.write("]\n")
        f.write(f"\\addplot3 [surf] file {{{file_name}.dat}};\n")
        f.write("\\end{polaraxis}\n")
        f.write("\\pgfresetboundingbox \\path ")
        f.write("($(current axis.below south west) - (1,0)$) rectangle ")
        f.write("($(current axis.above north east) + (2,0)$);\n")
        f.write("\\end{tikzpicture}\n")
        if standalone:
            f.write("%\n")
            f.write("\\end{document}\n")
