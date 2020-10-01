#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz
lualatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tmp simulation_analysis_spheres.tex
# (
#    cd output/tmp
#    make -j4 -f simulation_analysis_spheres.makefile
# )
# lualatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tmp simulation_analysis_spheres.tex
mv output/tmp/simulation_analysis_spheres.pdf output/tikz/simulation_analysis_spheres.pdf
xdg-open output/tikz/simulation_analysis_spheres.pdf &>/dev/null 2>&1
