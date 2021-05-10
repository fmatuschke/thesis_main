#!/bin/bash
set -e

append="$(echo $1 | sed 's?/?_?g')"
mkdir -p output/tmp
mkdir -p output/tikz

cd output/tmp

cp ../../pre_stats_box_plot.tex .
sed -i 's?__PATH__?'"$1"'?g' pre_stats_box_plot.tex
# lualatex -interaction=nonstopmode -halt-on-error pre_stats_box_plot.tex
# lualatex -interaction=nonstopmode -halt-on-error pre_stats_box_plot.tex
# cp pre_stats_box_plot.pdf ../tikz/pre_stats_box_plot_$append.pdf
# xdg-open ../tikz/pre_stats_box_plot_$append.pdf &>/dev/null 2>&1

cp ../../pre_stats_box_plot_volume.tex .
sed -i 's?__PATH__?'"$1"'?g' pre_stats_box_plot_volume.tex
lualatex -interaction=nonstopmode -halt-on-error pre_stats_box_plot_volume.tex
lualatex -interaction=nonstopmode -halt-on-error pre_stats_box_plot_volume.tex
cp pre_stats_box_plot_volume.pdf ../tikz/pre_stats_box_plot_volume_$append.pdf
xdg-open ../tikz/pre_stats_box_plot_volume_$append.pdf &>/dev/null 2>&1
