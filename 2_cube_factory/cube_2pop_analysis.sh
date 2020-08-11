#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz

rm -f output/tikz/cube_2pop_analysis.pdf
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape cube_2pop_analysis.tex
mv output/tmp/cube_2pop_analysis.pdf output/tikz/cube_2pop_analysis.pdf

# rm -f output/tmp/cube_2pop_images.pdf
# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp cube_2pop_images.tex
# mv output/tmp/cube_2pop_images.pdf output/tikz/cube_2pop_images.pdf
