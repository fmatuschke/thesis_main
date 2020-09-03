#!/bin/bash
set -e

mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/tikz

python3 sandbox.py
# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp sandbox.tex
# lualatex -interaction=nonstopmode -halt-on-error -output-directory=output/tmp sandbox.tex
# mv output/tmp/sandbox.pdf output/tikz/sandbox.pdf
# xdg-open output/tikz/sandbox.pdf &>/dev/null 2>&1
