#!/bin/bash
set -e

mkdir -p output/tmp/tikz
mkdir -p output/tmp/tikz/tmp/tikz
pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tmp cube_2pop_statistic_analysis.tex
# pdftk cube_2pop_statistic_analysis.pdf burst output cube_2pop_statistic_analysis_%01d.pdf
