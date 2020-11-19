#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz

lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape htm_orientation_hist2d.tex
lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp --shell-escape htm_orientation_hist2d.tex
mv output/tmp/htm_orientation_hist2d.pdf output/tikz/htm_orientation_hist2d.pdf
xdg-open output/tikz/htm_orientation_hist2d.pdf &> /dev/null 2>&1
