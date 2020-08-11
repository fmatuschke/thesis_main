#!/bin/bash
set -e

mkdir -p output/tikz/
mkdir -p output/tmp/
mkdir -p output/tmp/output/tmp/
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp cube_2pop_statistic_analysis.tex
mv output/tmp/cube_2pop_statistic_analysis.pdf output/tikz
