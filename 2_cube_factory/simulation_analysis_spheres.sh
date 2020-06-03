#!/bin/bash

N=3
cd output/images/spheres/
(
for file in *.tikz; do
   ((i=i%N)); ((i++==0)) && wait
   pdflatex -interaction=nonstopmode $file > /dev/null &
done
)
rm *.aux *.log
