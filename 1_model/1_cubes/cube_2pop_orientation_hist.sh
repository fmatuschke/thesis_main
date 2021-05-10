#!/bin/bash
set -ex

mkdir -p output/tmp
mkdir -p output/tikz

cd output/tmp

cp ../../cube_2pop_orientation_hist.tex .
sed -i 's?__PATH__?'"$1"'?g' cube_2pop_orientation_hist.tex
lualatex -interaction=nonstopmode -halt-on-error cube_2pop_orientation_hist.tex
lualatex -interaction=nonstopmode -halt-on-error cube_2pop_orientation_hist.tex
mv output/tmp/cube_2pop_orientation_hist.pdf output/tikz/cube_2pop_orientation_hist.pdf
xdg-open output/tikz/cube_2pop_orientation_hist.pdf &>/dev/null 2>&1
