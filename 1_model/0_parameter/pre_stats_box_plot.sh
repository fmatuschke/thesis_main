#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tikz

cd output/tmp

cp ../../pre_stats_box_plot.tex .
sed -i 's/__PATH__/'"$1"'/g' pre_stats_box_plot.tex
# lualatex -interaction=nonstopmode -halt-on-error pre_stats_box_plot.tex
# lualatex -interaction=nonstopmode -halt-on-error pre_stats_box_plot.tex
# mv pre_stats_box_plot.pdf ../tikz/pre_stats_box_plot.pdf
# xdg-open ../tikz/pre_stats_box_plot.pdf &>/dev/null 2>&1

cp ../../pre_stats_box_plot_volume.tex .
sed -i 's/__PATH__/'"$1"'/g' pre_stats_box_plot_volume.tex
lualatex -interaction=nonstopmode -halt-on-error pre_stats_box_plot_volume.tex
lualatex -interaction=nonstopmode -halt-on-error pre_stats_box_plot_volume.tex
mv pre_stats_box_plot_volume.pdf ../tikz/parameter_statistic_box_plot_volume.pdf
xdg-open ../tikz/pre_stats_box_plot_volume.pdf &>/dev/null 2>&1
