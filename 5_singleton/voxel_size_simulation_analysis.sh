# python3 voxel_size_simulation.py -i ../data/models/1_rnd_seed/cube_2pop_psi_1.00_omega_0.00_r_1.00_v0_210_.solved.h5 -o output/vs__2 -p 1
python3 voxel_size_simulation_analysis.py -i output/vs__2/voxel_size_simulation.pkl
mkdir -p output/tex
pdflatex -halt-on-error -synctex=1 -interaction=nonstopmode --shell-escape -output-directory output/tex voxel_size_simulation_analysis.tex 
open output/tex/voxel_size_simulation_analysis.pdf
