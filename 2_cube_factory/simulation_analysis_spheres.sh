#!/bin/bash

mkdir -p output/tikz/tmp
mkdir -p output/tikz/output/tikz/tmp
pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tikz simulation_analysis_spheres_LAP_r.tex > /dev/null &
pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tikz simulation_analysis_spheres_LAP_p.tex > /dev/null &
pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tikz simulation_analysis_spheres_PM_r.tex > /dev/null &
pdflatex -interaction=nonstopmode -halt-on-error --shell-escape -output-directory=output/tikz simulation_analysis_spheres_PM_p.tex > /dev/null



# N=3
# cd output/images/spheres/
# (
# for file in *.tikz; do
#    ((i=i%N)); ((i++==0)) && wait
#    pdflatex -interaction=nonstopmode $file > /dev/null &
# done
# )
# rm *.aux *.log
