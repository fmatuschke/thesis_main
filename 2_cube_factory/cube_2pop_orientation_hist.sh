#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape cube_2pop_orientation_hist.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape cube_2pop_orientation_hist.tex
mv output/tmp/cube_2pop_orientation_hist.pdf output/tikz/cube_2pop_orientation_hist.pdf
xdg-open output/tikz/cube_2pop_orientation_hist.pdf &> /dev/null 2>&1
