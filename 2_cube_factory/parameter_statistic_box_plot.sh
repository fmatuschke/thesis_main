#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz

# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp parameter_statistic_box_plot.tex
# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp parameter_statistic_box_plot.tex
# mv output/tmp/parameter_statistic_box_plot.pdf output/tikz/parameter_statistic_box_plot.pdf
# xdg-open output/tikz/parameter_statistic_box_plot.pdf &>/dev/null 2>&1

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp parameter_statistic_box_plot_volume.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp parameter_statistic_box_plot_volume.tex
mv output/tmp/parameter_statistic_box_plot_volume.pdf output/tikz/parameter_statistic_box_plot_volume.pdf
xdg-open output/tikz/parameter_statistic_box_plot_volume.pdf &>/dev/null 2>&1
