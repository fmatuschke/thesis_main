#!/bin/bash
set -e

# python3 voxel_size_simulation.py -i ../data/models/1_rnd_seed/cube_2pop_psi_1.00_omega_0.00_r_1.00_v0_210_.solved.h5 -o output/vs_0 -p 1
python3 voxel_size_simulation_analysis.py -i output/vs_0/voxel_size_simulation.pkl
mkdir -p output/tmp
pdflatex -halt-on-error -synctex=1 -interaction=nonstopmode --shell-escape -output-directory output/tmp voxel_size_simulation_analysis.tex 
cp output/tmp/voxel_size_simulation_analysis.pdf output/images/voxel_size_simulation_analysis.pdf
xdg-open output/images/voxel_size_simulation_analysis.pdf
