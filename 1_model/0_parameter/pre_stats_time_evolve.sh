#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tikz

cd output/tmp

cp ../../pre_stats_time_evolve.tex .
sed -i 's/__PATH__/'"$1"'/g' pre_stats_time_evolve.tex
lualatex -interaction=nonstopmode -halt-on-error --shell-escape pre_stats_time_evolve.tex
lualatex -interaction=nonstopmode -halt-on-error --shell-escape pre_stats_time_evolve.tex
mv pre_stats_time_evolve.pdf ../tikz/pre_stats_time_evolve.pdf
xdg-open ../tikz/pre_stats_time_evolve.pdf &>/dev/null 2>&1
