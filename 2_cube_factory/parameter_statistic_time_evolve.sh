#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp

rm -f output/tikz/parameter_statistic_time_evolve.pdf
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp parameter_statistic_time_evolve.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp parameter_statistic_time_evolve.tex
mv output/tmp/parameter_statistic_time_evolve.pdf output/tikz/parameter_statistic_time_evolve.pdf
xdg-open output/tikz/parameter_statistic_time_evolve.pdf &>/dev/null 2>&1
