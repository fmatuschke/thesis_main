#!/bin/bash
set -e

mkdir -p output/tikz/
mkdir -p output/tmp/
mkdir -p output/tmp/output/tmp/
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape model_solver_cubes.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape model_solver_cubes.tex
mv output/tmp/model_solver_cubes.pdf output/tikz
xdg-open output/tikz/model_solver_cubes.pdf &> /dev/null 2>&1
