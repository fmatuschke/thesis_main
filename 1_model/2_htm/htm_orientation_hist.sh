#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape htm_orientation_hist.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape htm_orientation_hist.tex
mv output/tmp/htm_orientation_hist.pdf output/tikz/htm_orientation_hist.pdf
xdg-open output/tikz/htm_orientation_hist.pdf &> /dev/null 2>&1
