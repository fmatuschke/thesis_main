#!/bin/bash
set -e

rm -rf output/tmp
mkdir -p output/tmp
mkdir -p output/tmp/output/tmp
mkdir -p output/images
pdflatex -halt-on-error -synctex=1 -interaction=nonstopmode --shell-escape -output-directory output/tmp vector_field.tex 
cp output/tmp/vector_field.pdf  output/images/vector_field.pdf 
xdg-open output/images/vector_field.pdf
