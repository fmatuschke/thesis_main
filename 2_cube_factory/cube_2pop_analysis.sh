#!/bin/bash
set -e

mkdir -p output/tikz/tmp
mkdir -p output/tikz/output/tikz/tmp
rm -f output/tikz/cube_2pop_analysis.pdf

echo "TODO psi=0->omegas"
echo "TODO revers incl"
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tikz cube_2pop_analysis.tex
# lualatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tikz cube_2pop_analysis.tex
