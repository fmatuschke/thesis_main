#!/bin/bash
set -e

# rm -rf output/tmp
mkdir -p output/tmp
mkdir -p output/tikz

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape voxel_size_plots_all.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp -shell-escape voxel_size_plots_all.tex
mv output/tmp/voxel_size_plots_all.pdf output/tikz/voxel_size_plots_all.pdf
xdg-open output/tikz/voxel_size_plots_all.pdf &>/dev/null 2>&1

# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp voxel_size_plots.tex
# mv output/tmp/voxel_size_plots.pdf output/tikz/voxel_size_plots.pdf
# xdg-open output/tikz/voxel_size_plots.pdf &>/dev/null 2>&1
