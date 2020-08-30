#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp parameter_statistic_box_plot.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp parameter_statistic_box_plot.tex
mv output/tmp/parameter_statistic_box_plot.pdf output/tikz/parameter_statistic_box_plot.pdf
open output/tikz/parameter_statistic_box_plot.pdf &>/dev/null 2>&1
