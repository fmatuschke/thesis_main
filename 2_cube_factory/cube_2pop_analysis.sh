#!/bin/bash
set -e

mkdir -p output/tikz/tmp
mkdir -p output/tikz/output/tikz/tmp

echo "TODO psi=0->omegas"
echo "TODO revers incl"
pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tikz model_analysis.tex > /dev/null
