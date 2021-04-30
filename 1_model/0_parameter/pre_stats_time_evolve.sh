#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape pre_stats_time_evolve.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape pre_stats_time_evolve.tex
mv output/tmp/pre_stats_time_evolve.pdf output/tikz/pre_stats_time_evolve.pdf
xdg-open output/tikz/pre_stats_time_evolve.pdf &>/dev/null 2>&1
